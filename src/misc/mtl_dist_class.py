import os
import sys
import time
import copy

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets

import shutil
import tempfile
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint




import PIL

import matplotlib
from mpl_toolkits import mplot3d
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

from livelossplot import PlotLosses

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import IncrementalPCA

from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, classification_report

from tqdm import tqdm

import spheres
from myNNs import MTLModelForDistanceAndClass
from ptcifar.models import ResNet18

OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD 
}

seed_everything(42, workers=True)

class MTLForDistanceAndClass(pl.LightningModule):

    """
        An MTL model for learning distance of a point 
        from the data manifold and learning class labels
    """

    def __init__(self, config, data_dir=None):
        
        
        super(ResNet18MfldDistRegressor, self).__init__()

        self.data_dir = data_dir or os.getcwd()
        
        self.optim_type = config["optim_type"]
        self.optim_config = config["optim_config"]
        self.scheduler_params = config["scheduler_params"]
        
        self.batch_size = config["batch_size"]
        self.scheduler_params = config["scheduler_params"]

        self.input_size = config["input_size"]
        self.num_classes = config["num_classes"]

        self.num_epochs = config["num_epochs"]
        
        self.dist_loss_wt = config["dist_loss_wt"]
        self.class_loss_wt = config["class_loss_wt"]

        self.dist_criterion = nn.MSELoss()
        self.class_criterion = nn.CrossEntropyLoss()

        # defining the model
        self.model = MTLModelForDistanceAndClass(input_size=3 * 32 * 32, output_size=self.num_classes)
        

    def forward(self, X):

        dist_logits, class_logits = self.model(X)

        return {"dist_logits": dist_logits, "class_logits": class_logits}


    def mean_squared_error(self, logits, labels):
        return self.dist_criterion(logits, labels)

    def cross_entropy_error(self, logits, labels):
        return self.class_criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        
        # For now; Fix this in the data later
        y_class = y.clone().detach()
        y_class[y == 0] = 0
        y_class[y > 0] = 1
        y_class = y_class.long()

        logits = self.forward(x)
        dist_loss = self.mean_squared_error(logits["dist_logits"], y)
        class_loss = self.cross_entropy_error(logits["class_logits"], y_class)

        total_loss = (self.dist_loss_wt * dist_loss) + (self.class_loss_wt * class_loss_wt)


        return {"train/dist_loss": dist_loss, "train/class_loss", "loss": total_loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        # For now; Fix this in the data later
        y_class = y.clone().detach()
        y_class[y == 0] = 0
        y_class[y > 0] = 1
        y_class = y_class.long()

        logits = self.forward(x)
        dist_loss = self.mean_squared_error(logits["dist_logits"], y)
        class_loss = self.cross_entropy_error(logits["class_logits"], y_class)

        total_loss = (self.dist_loss_wt * dist_loss) + (self.class_loss_wt * class_loss_wt)


        return {"val/dist_loss": dist_loss, "val/class_loss", "val_loss": total_loss}

    def training_epoch_end(self, outputs):
        avg_train_dist_loss = torch.stack([x["train/dist_loss"] for x in outputs]).mean()
        avg_train_class_loss = torch.stack([x["train/class_loss"] for x in outputs]).mean()
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log("ptl/train/dist_loss", avg_train_dist_loss)
        self.log("ptl/train/class_loss", avg_train_class_loss)
        self.log("ptl/train/loss", avg_train_loss)

        # with open("train_losses.txt", "a+") as f:
        #     f.write(str(avg_train_loss.detach().cpu().item()) + "\n")

    def validation_epoch_end(self, outputs):
        avg_val_dist_loss = torch.stack([x["val/dist_loss"] for x in outputs]).mean()
        avg_val_class_loss = torch.stack([x["val/class_loss"] for x in outputs]).mean()
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        self.log("ptl/val/dist_loss", avg_train_dist_loss)
        self.log("ptl/val/class_loss", avg_train_class_loss)
        self.log("ptl/val/loss", avg_train_loss)

        # with open("val_losses.txt", "a+") as f:
        #     f.write(str(avg_val_loss.detach().cpu().item()) + "\n")

    @staticmethod
    def download_data(data_dir, generate=False):
        
        
        
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
            train_set = torch.load(os.path.join(data_dir, "train_cifar_dim.pt"))
            val_set = torch.load(os.path.join(data_dir, "val_cifar_dim.pt"))
            
        return train_set, val_set

    def prepare_data(self):
        self.train_set, self.val_set = self.download_data(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=int(self.batch_size), num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=int(self.batch_size), num_workers=8)

    def configure_optimizers(self):
        
        optimizer = OPTIMIZERS[self.optim_type](**self.optim_config)

        # if self.optimizer_type == "adam":
        #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # elif self.optimizer_type == "sgd":
        #     optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        
        lr_sched_factor = lambda epoch: epoch / (self.scheduler_params["warmup"]) if epoch <= self.scheduler_params["warmup"] else (1 if epoch > self.scheduler_params["warmup"] and epoch < self.scheduler_params["cooldown"] else max(0, 1 + (1 / (self.scheduler_params["cooldown"] - self.num_epochs)) * (epoch - self.scheduler_params["cooldown"])))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched_factor)
        
        return [optimizer], [scheduler]





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="enable train mode")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--cuda", action="store_true", help="use GPUs")    
    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=500)
    parser.add_argument("--savedir", type=str, help="save directory path")
    parser.add_argument("--name", type=str, help="name of experiment")
    parser.add_argument("--warmup", type=int, help="number of warmup steps", default=10)
    parser.add_argument("--cooldown", type=int, help="epoch after which to cooldown", default=300)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--init_wts", type=str, help="path to initial weights", default="/azuredrive/dumps/expB_learning_distance_from_mfld/init_model_weights.pt")

    parser.add_argument("--train_fn", type=str, help="path to train data") 
    parser.add_argument("--val_fn", type=str, help="path to val data") 
    parser.add_argument("--test_fn", type=str, help="path to test data") 


