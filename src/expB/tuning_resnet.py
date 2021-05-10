#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import time
import copy
import json
import random
from typing import Dict, List, Optional, Union

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms

from tqdm import tqdm

import shutil
import tempfile

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning import Callback, Trainer, LightningModule

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback, TuneCallback



# adding relevant files to PATH
sys.path.append("../src/expB/")

from ptcifar.models import ResNet18
# import spheres
from spheres import RandomSphere
from myNNs import *


SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class _SameFileTuneCheckpointCallback(TuneCallback):


    def __init__(self,
                 filename: str = "checkpoint",
                 on: Union[str, List[str]] = "validation_end"):
        super(_SameFileTuneCheckpointCallback, self).__init__(on)
        self._filename = filename

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.running_sanity_check:
            return
        # step = f"epoch={trainer.current_epoch}-step={trainer.global_step}"
        step = f"model" # to avoid multiple files
        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            trainer.save_checkpoint(
                os.path.join(checkpoint_dir, self._filename))
        
        if trainer.current_epoch == 100:
            step = f"model_100" # to avoid multiple files
            with tune.checkpoint_dir(step=step) as checkpoint_dir:
                trainer.save_checkpoint(
                    os.path.join(checkpoint_dir, self._filename))



class CustomTuneReportCheckpointCallback(TuneCallback):


    def __init__(self,
                 metrics: Union[None, str, List[str], Dict[str, str]] = None,
                 filename: str = "checkpoint",
                 on: Union[str, List[str]] = "validation_end"):
        super(CustomTuneReportCheckpointCallback, self).__init__(on)
        self._checkpoint = _SameFileTuneCheckpointCallback(filename, on)
        self._report = TuneReportCallback(metrics, on)

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        self._checkpoint._handle(trainer, pl_module)
        self._report._handle(trainer, pl_module)


class MLPDistRegressor(pl.LightningModule):
    """
    This has been adapted from
    https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
    """

    
    def __init__(self, config, data_dir=None):
        
        
        super(MLPDistRegressor, self).__init__()

        self.data_dir = data_dir or os.getcwd()
        
        self.lr = config["lr"]
        self.momentum = config["momentum"]
        
        self.batch_size = config["batch_size"]
        self.optimizer_type = config["optimizer_type"]
        self.scheduler_params = config["scheduler_params"]
        
        # self.N = config["data_params"][0]
        # self.num_neg = config["data_params"][1]

        self.epoch_logits_train = list()
        self.epoch_logits_val = list()

#         self.train_set = config["train_set"]
#         self.val_set = config["val_set"]

        
        self.train_epoch_losses = list()
        
        self.num_epochs = config["num_epochs"]
        
        # defining the model
        # self.model = ResNet18(num_classes=1)
        # self.model.load_state_dict(torch.load(init_wts_fn))
        self.model = MLP(input_size=2, output_size=1, hidden_sizes=[512, 256, 128, 64])
        
        
    def forward(self, x):
        
        x = self.model(x)
        
        return x



    def mean_squared_error(self, logits, labels):
        return F.mse_loss(logits, labels)


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.mean_squared_error(logits, y)

#         self.log("ptl/train_loss", loss)
        self.train_epoch_losses.append(loss.item())
        self.epoch_logits_train.extend(logits.detach().cpu().tolist())
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        self.epoch_logits_val.extend(logits.detach().cpu().tolist())
        loss = self.mean_squared_error(logits, y)
        return {"val_loss": loss}

    def custom_histogram_adder(self):
       
        # iterating through all parameters
        for name,params in self.model.named_parameters():
          
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def training_epoch_end(self, outputs):

        # avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logging histograms
        self.custom_histogram_adder()
        self.logger.experiment.add_histogram("logits/train", np.array(self.epoch_logits_train).reshape(-1), self.current_epoch)
        self.epoch_logits_train = list()
        
        # epoch_dictionary={
        #     # required
        #     'loss': avg_loss}

        # return epoch_dictionary


    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_train_loss = np.mean(self.train_epoch_losses)
        self.log("ptl/val_loss", avg_val_loss)
        self.log("ptl/train_loss", avg_train_loss)
        self.logger.experiment.add_histogram("logits/val", np.array(self.epoch_logits_val).reshape(-1), self.current_epoch)
        self.epoch_logits_val = list()
        self.train_epoch_losses = list()

        # with open("train_losses.txt", "a+") as f:
        #     f.write(str(avg_train_loss) + "\n")

        # with open("val_losses.txt", "a+") as f:
        #     f.write(str(avg_val_loss.detach().cpu().item()) + "\n")

        

    @staticmethod
    def download_data(data_dir, generate=True, N=50000, num_neg=25000):
        
        
        n = 2
        
        train_set = None
        val_set = None
        
        if generate:
            # train_params = {
            #     "N": 50000,
            #     "num_neg": None,
            #     "n": n,
            #     "k": 2,
            #     "r": 100.0,
            #     "D": 25.0,
            #     "max_norm": 500.0,
            #     "mu": 1000,
            #     "sigma": 5000,
            #     "seed": 23
            # }

            with open("/data/adv_geom/src/expB/data_configs/sphere_config.json", "r") as f:
                config = json.load(f)

            # train set
            train_config = config["train"]
            train_config["N"] = N
            train_config["num_neg"] = num_neg
            train_set = RandomSphere(**train_config)
            
            # validation set
            val_config = config["val"]
            val_config["x_ck"] = train_set.x_ck
            val_config["translation"] = train_set.translation
            val_config["rotation"] = train_set.rotation
            val_config["N"] = N
            val_config["num_neg"] = num_neg 
            val_set = RandomSphere(**val_config)

            

            # test set
            test_config = config["test"]
            test_config["x_ck"] = train_set.x_ck
            test_config["translation"] = train_set.translation
            test_config["rotation"] = train_set.rotation
            test_config["N"] = N
            test_config["num_neg"] = num_neg
            test_set = RandomSphere(**test_config)
        
        if generate:
            torch.save(train_set, os.path.join(data_dir, "train_set.pt"))
            torch.save(val_set, os.path.join(data_dir, "val_set.pt"))
            torch.save(test_set, os.path.join(data_dir, "test_set.pt"))
        else:
            train_set = torch.load(os.path.join(data_dir, "train_set.pt"))
            val_set = torch.load(os.path.join(data_dir, "val_set.pt"))

        SEED = 42
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED) 

        # train_perm = torch.randperm(train_set.N)
        # val_perm = torch.randperm(val_set.N)

        # torch.save(train_perm, "/azuredrive/dumps/train_perm.pt")
        # torch.save(val_perm, "/azuredrive/dumps/val_perm.pt")

        # train_set.points_n = train_set.points_n[train_perm]
        # train_set.distances = train_set.distances[train_perm]

        # val_set.points_n = val_set.points_n[val_perm]
        # val_set.distances = val_set.distances[val_perm]

        # train_set.points_n = train_set.points_n
        # train_set.distances = train_set.distances

        # val_set.points_n = val_set.points_n
        # val_set.distances = val_set.distances

        return train_set, val_set

    def prepare_data(self):
        self.train_set, self.val_set = self.download_data(self.data_dir, generate=True)

        self.logger.experiment.add_histogram("distances/train",self.train_set.distances)
        self.logger.experiment.add_histogram("distances/val",self.val_set.distances)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=int(self.batch_size), num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=int(self.batch_size), num_workers=8, shuffle=True)

    def configure_optimizers(self):
        optimizer = None
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        
        lr_sched_factor = lambda epoch: epoch / (self.scheduler_params["warmup"]) if epoch <= self.scheduler_params["warmup"] else (1 if epoch > self.scheduler_params["warmup"] and epoch < self.scheduler_params["cooldown"] else max(0, 1 + (1 / (self.scheduler_params["cooldown"] - self.num_epochs)) * (epoch - self.scheduler_params["cooldown"])))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched_factor)
        
        return [optimizer], [scheduler]


class ResNet18MfldDistRegressor(pl.LightningModule):
    """
    This has been adapted from
    https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
    """

    
    def __init__(self, config, data_dir=None, init_wts_fn="/azuredrive/dumps/expB_learning_distance_from_mfld/init_model_weights.pt"):
        
        
        super(ResNet18MfldDistRegressor, self).__init__()

        self.data_dir = data_dir or os.getcwd()
        
        self.lr = config["lr"]
        self.momentum = config["momentum"]
        
        self.batch_size = config["batch_size"]
        self.optimizer_type = config["optimizer_type"]
        self.scheduler_params = config["scheduler_params"]
        
        

        self.train_set = config["train_set"]
        self.val_set = config["val_set"]

        
        self.train_epoch_losses = list()
        
        self.num_epochs = config["num_epochs"]
        
        # defining the model
        self.model = ResNet18(num_classes=1)
        self.model.load_state_dict(torch.load(init_wts_fn))
        
        
    def forward(self, x):
        
        x = self.model(x)
        
        return x



    def mean_squared_error(self, logits, labels):
        return F.mse_loss(logits, labels)


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.mean_squared_error(logits, y)

        # self.log("ptl/train_loss", loss)
        self.train_epoch_losses.append(loss.item())
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.mean_squared_error(logits, y)
        return {"val_loss": loss}

    def custom_histogram_adder(self):
       
        # iterating through all parameters
        for name,params in self.model.named_parameters():
          
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def training_epoch_end(self, outputs):

        # avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logging histograms
        self.custom_histogram_adder()
        
        # epoch_dictionary={
        #     # required
        #     'loss': avg_loss}

        # return epoch_dictionary


    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_train_loss = np.mean(self.train_epoch_losses)
        self.log("ptl/val_loss", avg_val_loss)
        self.log("ptl/train_loss", avg_train_loss)
        self.train_epoch_losses = list()

        with open("train_losses.txt", "a+") as f:
            f.write(str(avg_train_loss) + "\n")

        with open("val_losses.txt", "a+") as f:
            f.write(str(avg_val_loss.detach().cpu().item()) + "\n")

        

    @staticmethod
    def download_data(data_dir, generate=True):
        
        
        
        n = 32 * 32 * 3
        
        train_set = None
        
        if generate:
            train_params = {
                "N": 50000,
                "num_neg": None,
                "n": n,
                "k": 2,
                "r": 100.0,
                "D": 25.0,
                "max_norm": 500.0,
                "mu": 1000,
                "sigma": 5000,
                "seed": 23
            }
            train_set = spheres.RandomSphere(**train_params)

        
        val_set = None
        if generate:
            val_params = {
                "N": 10000,
                "num_neg": None,
                "n": n,
                "k": 2,
                "r": 100.0,
                "D": 25.0,
                "max_norm": 500.0,
                "mu": 1000,
                "sigma": 5000,
                "seed": 101,
                "x_ck": train_set.x_ck,
                "translation": train_set.translation,
                "rotation": train_set.rotation
            }
            val_set = spheres.RandomSphere(**val_params)
        
        if generate:
            torch.save(train_set, os.path.join(data_dir, "train_cifar_dim.pt"))
            torch.save(val_set, os.path.join(data_dir, "val_cifar_dim.pt"))
        else:
            train_set = torch.load(os.path.join(data_dir, "train_set.pt"))
            val_set = torch.load(os.path.join(data_dir, "val_set.pt"))

        SEED = 42
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED) 

        train_perm = torch.randperm(train_set.N)
        val_perm = torch.randperm(val_set.N)

        torch.save(train_perm, "/azuredrive/dumps/train_perm.pt")
        torch.save(val_perm, "/azuredrive/dumps/val_perm.pt")

        train_set.points_n = train_set.points_n[train_perm]
        train_set.distances = train_set.distances[train_perm]

        val_set.points_n = val_set.points_n[val_perm]
        val_set.distances = val_set.distances[val_perm]

        # train_set.points_n = train_set.points_n
        # train_set.distances = train_set.distances

        # val_set.points_n = val_set.points_n
        # val_set.distances = val_set.distances

        return train_set, val_set

    def prepare_data(self):
        self.train_set, self.val_set = self.download_data(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=int(self.batch_size), num_workers=8, worker_init_fn=seed_worker, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=int(self.batch_size), num_workers=8, worker_init_fn=seed_worker, shuffle=False)

    def configure_optimizers(self):
        optimizer = None
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        
        lr_sched_factor = lambda epoch: epoch / (self.scheduler_params["warmup"]) if epoch <= self.scheduler_params["warmup"] else (1 if epoch > self.scheduler_params["warmup"] and epoch < self.scheduler_params["cooldown"] else max(0, 1 + (1 / (self.scheduler_params["cooldown"] - self.num_epochs)) * (epoch - self.scheduler_params["cooldown"])))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched_factor)
        
        return [optimizer], [scheduler]


# def train_mnist(config, data_dir, num_epochs):
#     model = ResNet18MfldDistRegressor(config, data_dir)
#     trainer = pl.Trainer(max_epochs=num_epochs, gpus=1)

#     trainer.fit(model)


# In[ ]:


def train_model_tune_checkpoint(config,
                                checkpoint_dir=None,
                                data_dir=None,
                                num_epochs=10,
                                num_gpus=1):
    
    
    
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        callbacks=[
            CustomTuneReportCheckpointCallback(
                metrics={
                    "val_loss": "ptl/val_loss"
                },
                filename="checkpoint",
                on="validation_end"),
            LearningRateMonitor(logging_interval='epoch', log_momentum=True)
        ])
    if checkpoint_dir:
        # Currently, this leads to errors:
        # model = LightningMNISTClassifier.load_from_checkpoint(
        #     os.path.join(checkpoint, "checkpoint"))
        # Workaround:
        ckpt = pl_load(
            os.path.join(checkpoint_dir, "checkpoint"),
            map_location=lambda storage, loc: storage)
        model = MLPDistRegressor._load_model_state(
            ckpt, config=config, data_dir=data_dir)
        trainer.current_epoch = ckpt["epoch"]
    else:
        model = MLPDistRegressor(config=config, data_dir=data_dir)

    trainer.fit(model)


# In[ ]:


def tune_model(data_dir, save_dir, num_samples=1, num_epochs=5, gpus_per_trial=1):
    
    
#     ResNet18MfldDistRegressor.download_data(data_dir)

    config = {
        "lr": tune.grid_search([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
        "batch_size": tune.grid_search([512, 128]),
        "optimizer_type": tune.grid_search(["sgd", "adam"]),
        "momentum": tune.grid_search([0.9]),
        "scheduler_params": {"warmup": 10, "cooldown": 150},
        "num_epochs": num_epochs
    }

    
    # config = {
    #     "lr": tune.grid_search([0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]),
    #     "batch_size": tune.grid_search([128]),
    #     "optimizer_type": tune.grid_search(["adam"]),
    #     "momentum": tune.grid_search([0.9]),
    #     "scheduler_params": {"warmup": 10, "cooldown": 150},
    #     "num_epochs": num_epochs
    # }

    # for debugging
    # config = {
    #     "lr": tune.grid_search([1e-6]),
    #     "batch_size": tune.grid_search([128]),
    #     "optimizer_type": tune.grid_search(["sgd"]),
    #     "momentum": tune.grid_search([0.9]),
    #     "scheduler_params": {"warmup": 10, "cooldown": 125},
    #     "num_epochs": num_epochs

    # }


    reporter = CLIReporter(
        parameter_columns=["lr", "momentum", "batch_size", "optimizer_type", "num_epochs", "scheduler_params"],
        metric_columns=["val_loss", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_model_tune_checkpoint,
            data_dir=data_dir,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": 3,
            "gpu": gpus_per_trial
        },
        metric="val_loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter,
        name="tune_dist_learn_mlp_tune_{max_norm=2,D=1,n=2,N=50000,num_neg=25000}",
        local_dir=save_dir)

    print("Best hyperparameters found were: ", analysis.best_config)



# In[ ]:


NUM_EPOCHS = 250
DATA_DIR = "/azuredrive/datasets/expB/one_sphere/max_norm=2,D=1,n=2/tuning/hparams_wo_neg_size/"
SAVE_DIR = "/azuredrive/ray_results/"

# config = {
    
#     "lr": 1e-3,
#     "momentum": 0.9,
#     "batch_size": 512,
#     "optimizer_type": "sgd",
#     "scheduler_params": {"warmup": 10, "cooldown": 300},
#     "num_epochs": NUM_EPOCHS
# }

# train_mnist(config, DATA_DIR, NUM_EPOCHS)


# In[ ]:


tune_model(data_dir=DATA_DIR, save_dir=SAVE_DIR, num_epochs=NUM_EPOCHS)


# In[ ]:




