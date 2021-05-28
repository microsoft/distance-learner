import os
import sys
import json
import time
import copy
import random
import argparse

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets

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

from spheres_v2 import RandomSphere, TwoRandomSpheres
from ptcifar.models import ResNet18
from myNNs import *
from workspace import *




SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def weighted_mse_loss(inp, target, weight):
        return (weight.reshape(-1) * (inp - target) ** 2).mean()


def custom_histogram_adder(writer, model, current_epoch):
       
        # iterating through all parameters
        for name,params in model.named_parameters():
          
            writer.add_histogram(name,params,current_epoch)


def train(model, optimizer, loss_func, dataloaders, device, save_dir, scheduler,\
          feature_name="normed_points", target_name="normed_distances",\
          num_epochs=500, task="regression", name="MLP_512x4_in1000",\
          scheduler_params={"warmup": 10, "cooldown": 300}, specs_dict=None):
    """
        Function to train the model. Also dumps the best model.
        
        Returns the best model and optimizers

    """
    
    # storing the start_lr
    start_lr = optimizer.param_groups[0]['lr']
    
    TIME_STAMP = time.strftime("%d%m%Y-%H%M%S")

    train_loss_matrix = torch.zeros((num_epochs, len(dataloaders["train"])))
    val_loss_matrix = torch.zeros((num_epochs, len(dataloaders["val"])))
    

    save_dir = os.path.join(save_dir, name)
    model_dir = os.path.join(save_dir, TIME_STAMP, "models")
    plot_dir = os.path.join(save_dir, TIME_STAMP, "tensorboard")
    specs_dump_fn = os.path.join(save_dir, TIME_STAMP, "specs.json")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    with open(specs_dump_fn, "w+") as f:
        json.dump(specs_dict, f)
    
    plot_fn = os.path.join(plot_dir, name + "_" + TIME_STAMP)

    writer = SummaryWriter(plot_fn)
    # liveloss = PlotLosses(fig_path=plot_fn)
    
    def plotDataIn3D(points_n):
        
        if points_n.shape[1] == 2:
            plt.figure(figsize = (10,7))
            fig_plot = plt.scatter(points_n[:, 0], points_n[:, 1], s=0.1).get_figure()
            plt.close(fig_plot)
        else:
            fig = plt.figure(figsize = (10, 7))
            ax1 = plt.axes(projection ="3d")
            fig_plot = ax1.scatter3D(points_n[:, 0], points_n[:, 1], points_n[:, 2], s=0.1).get_figure()
            plt.close(fig_plot)
        return fig_plot

    if hasattr(dataloaders["train"].dataset, "distances"):
        writer.add_histogram("train/distances", dataloaders["train"].dataset.distances)
        writer.add_histogram("val/distances", dataloaders["val"].dataset.distances)

        # plt.figure(figsize = (10,7))
        # fig_train = plt.scatter(dataloaders["train"].dataset.points_n[:, 0], dataloaders["train"].dataset.points_n[:, 1], s=0.01).get_figure()
        # plt.close(fig_train)
        fig_train = plotDataIn3D(dataloaders["train"].dataset.points_n)
        writer.add_figure("train/points_n", fig_train)
        writer.add_text("train/data/params", str(vars(dataloaders["train"].dataset)))

        # plt.figure(figsize = (10,7))
        # fig_val = plt.scatter(dataloaders["val"].dataset.points_n[:, 0], dataloaders["val"].dataset.points_n[:, 1], s=0.01).get_figure()
        # plt.close(fig_val)
        fig_val = plotDataIn3D(dataloaders["val"].dataset.points_n)
        writer.add_figure("val/points_n", fig_val)
        writer.add_text("train/val/params", str(vars(dataloaders["val"].dataset)))

        writer.add_graph(model, dataloaders["train"].dataset.points_n[:dataloaders["train"].batch_size])

    elif hasattr(dataloaders["train"].dataset, "all_distances"):
        
        for phase in ["train", "val"]:
            writer.add_histogram(phase + "/S1/distances", dataloaders[phase].dataset.all_distances[:, 0])
            writer.add_histogram(phase + "/S2/distances", dataloaders[phase].dataset.all_distances[:, 1])

            writer.add_histogram(phase + "/S1/normed_distances", dataloaders[phase].dataset.normed_all_distances[:, 0])
            writer.add_histogram(phase + "/S2/normed_distances", dataloaders[phase].dataset.normed_all_distances[:, 1])

        
        capture_attrs =  ["S1_config", "S2_config", "n", "seed"]

        for phase in ["train", "val"]:
            fig = plotDataIn3D(dataloaders[phase].dataset.all_points)
            writer.add_figure(phase + "/all_points", fig)
            data_vars = vars(dataloaders[phase].dataset)
            writer.add_text(phase + "/data/params", str({i: data_vars[i] for i in data_vars if i in capture_attrs}))

        

        writer.add_graph(model, dataloaders["train"].dataset.normed_all_points[:dataloaders["train"].batch_size])


    else:
        writer.add_histogram("train/distances", dataloaders["train"].dataset.tensors[1])
        writer.add_histogram("val/distances", dataloaders["val"].dataset.tensors[1])

        # plt.figure(figsize = (10,7))
        # fig_train = plt.scatter(dataloaders["train"].dataset.tensors[0][:, 0], dataloaders["train"].dataset.tensors[0][:, 1], s=0.01).get_figure()
        # plt.close(fig_train)
        fig_train = plotDataIn3D(dataloaders["train"].dataset.tensors[0])
        writer.add_figure("train/points_n", fig_train)

        # plt.figure(figsize = (10,7))
        # fig_val = plt.scatter(dataloaders["val"].dataset.tensors[0][:, 0], dataloaders["val"].dataset.tensors[0][:, 1], s=0.01).get_figure()
        # plt.close(fig_val)
        fig_val = plotDataIn3D(dataloaders["val"].dataset.tensors[0])
        writer.add_figure("val/points_n", fig_val)

        writer.add_graph(model, dataloaders["train"].dataset.tensors[0][:dataloaders["train"].batch_size])    

    phase = "train"
    
    last_best_epoch_loss = None
    
    losses = {"train_losses": [], "val_losses": []}
    
    for epoch in tqdm(range(num_epochs)):
        
         

        logs = {
            "loss": 0,
            "val_loss": 0,
            "lr": start_lr
        }
        
        for phase in ["train", "val"]:
            
            all_logits = None

            dl = dataloaders[phase]
            
            pred_classes = None
            target_classes = None
            
            if task == "clf":
                pred_classes = torch.zeros(dl.dataset.N)
                target_classes = torch.zeros(dl.dataset.N)
            
            if phase == "train":
                torch.set_grad_enabled(True)
                model.train()
            else:
                torch.set_grad_enabled(False)
                model.eval()
            
            num_batches = 0
            
            prefix = ""
            if phase == "val":
                prefix = "val_"
            
            for (i, batch) in enumerate(dataloaders[phase]):
                
                def get_data(batch_dict, feature_name=feature_name, target_name=target_name):
                    return batch_dict[feature_name], batch_dict[target_name]
                
                points, distances = get_data(batch)

                points = points.to(device)
                distances = distances.to(device)
                # classes = batch[2].to(device)
                
                

                model = model.to(device)
                
                logits = None
                if phase == "val":
                    with torch.no_grad():
                        logits = model(points)
                elif phase == "train":
                    logits = model(points)

                # print(points.shape, distances.shape, logits.shape)

                # weights in case of weighted loss
                # weights = distances.clone().detach()
                # weights[weights == 0] = 1 / torch.max(weights[weights != 0])
                # weights[weights != 0] = 1 / weights[weights != 0]

                # loss = loss_func(logits, distances, weights)

                loss = loss_func(logits, distances)
                
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                
                logs[prefix + "loss"] += loss.detach().cpu().item()

                if phase == "train":
                    train_loss_matrix[epoch, i] = loss.detach().cpu().item()
                elif phase == "val":
                    val_loss_matrix[epoch, i] = loss.detach().cpu().item()
                
                if all_logits is None:
                    all_logits = torch.zeros(len(dataloaders[phase].dataset), logits.shape[1])
                all_logits[i*points.shape[0]:(i+1)*points.shape[0]] = logits.detach().cpu()

                num_batches += 1
                
                points = points.cpu()
                distances = distances.cpu()
                model = model.cpu()
                logits = logits.cpu()
                
                if task == "clf":
                    pred_classes[i*points.shape[0]:(i+1)*points.shape[0]] = torch.max(logits, axis=1)[1]
                    target_classes[i*points.shape[0]:(i+1)*points.shape[0]] = distances
            
            if task == "clf":
                f1 = f1_score(target_classes, pred_classes)
                acc = accuracy_score(target_classes, pred_classes)

                logs[prefix + "f1"] = f1
                logs[prefix + "acc"] = acc
                    
            
            # dividing by the number of batches
            logs[prefix + "loss"] /= num_batches
            writer.add_scalar(phase + "/loss", logs[prefix + "loss"], epoch)
            if phase == "train":
                custom_histogram_adder(writer, model, epoch)
            
            if all_logits.shape[1] == 1:
                writer.add_histogram(phase + "/logits", all_logits.reshape(-1), epoch)
            else:
                for idx in range(all_logits.shape[1]):
                    writer.add_histogram(phase + "/S" + str(idx+1) + "/logits", all_logits[:, idx].reshape(-1), epoch)
            
        
        check = last_best_epoch_loss is None or logs["val_loss"] < last_best_epoch_loss or epoch == 100
        stat = "val_loss"
        if task == "clf":
            check = last_best_epoch_loss is None or logs["val_f1"] > last_best_epoch_loss
            stat = "val_f1"
            
        if check:
            last_best_epoch_loss = logs[stat]
            dump = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': logs['val_loss'],
                'loss': logs["loss"],
                'scheduler_params': scheduler_params,
                'scheduler_state_dict': scheduler.state_dict()
            }
            
            if task == "clf":
                dump["acc"] = logs["acc"]
                dump["val_acc"] = logs["val_acc"]
                dump["f1"] = logs["f1"]
                dump["val_f1"] = logs["val_f1"]
                
            torch.save(dump, os.path.join(model_dir, name + "_"+ TIME_STAMP + "_val_loss_" + str(logs["val_loss"]) + "_epoch_" + str(epoch) + ".pth"))

        
        logs["lr"] = optimizer.param_groups[0]["lr"]

        writer.add_scalar("lr", logs["lr"], epoch)



        # liveloss.update(logs)
        # liveloss.draw()
        
        scheduler.step()
        writer.flush()

    writer.close()
    return model, optimizer, train_loss_matrix, val_loss_matrix

                    
            

def test(model, dataloader, device, task="regression",\
     feature_name="normed_points", target_name="normed_distances"):
    
    model.eval()
    
    all_logits = None
    all_targets = None
    # all_true_distances = None
    # all_classes = None
    
    with torch.no_grad():
        
        for batch in dataloader:
            # print(len(batch))
            def get_data(batch_dict, feature_name=feature_name, target_name=target_name):
                return batch_dict[feature_name], batch_dict[target_name]
        
            points, targets = get_data(batch, feature_name=feature_name, target_name=target_name)

            points = points.to(device)
            targets = targets.to(device)

            # points = batch[0].to(device)
            # distances = batch[1].to(device)
            # classes = batch[2].to(device)
            
            model.zero_grad()
            model = model.to(device)

            logits = model(points).detach().cpu()

            points = points.cpu()
            distances = distances.cpu()
            model = model.cpu()
            
            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits))
            
            if all_targets is None:
                all_targets = targets
            else:
                all_targets = torch.cat((all_targets, targets))

            # if task == "classification":
            #     if all_classes is None:
            #         all_classes = classes
            #     else:
            #         all_classes = torch.cat((all_classes, classes))
            
            
    if task == "regression":
        mse = mean_squared_error(all_targets, all_logits)
        mse_on_mfld = mean_squared_error(all_targets[np.round(all_targets) == 0], all_logits[np.round(all_targets) == 0])



        print("MSE for the learned distances:", mse)
        print("MSE for the learned distances (on-manifold):", mse_on_mfld)
        return mse, mse_on_mfld, all_targets, all_logits
    
    elif task == "clf":
        y_pred = torch.max(all_logits, axis=1)[1]
        print(classification_report(all_targets.reshape(-1), y_pred))
        acc = accuracy_score(all_targets.reshape(-1), y_pred)
        f1 = f1_score(all_targets.reshape(-1), y_pred)
        
        return acc, f1, all_targets, all_logits
    
    #TODO: fix this for concentric dataset





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="enable train mode", required=True)

    parser.add_argument("--specs", type=str, help="specifications file for the experiment, removes need for all other options", default=None)

    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--cuda", help="use GPUs", default=None)    
    parser.add_argument("--task", type=str, help="'classification' or 'regression'", default="regression")
    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=500)
    parser.add_argument("--save_dir", type=str, help="save directory path")
    parser.add_argument("--name", type=str, help="name of experiment")
    parser.add_argument("--warmup", type=int, help="number of warmup steps", default=10)
    parser.add_argument("--cooldown", type=int, help="epoch after which to cooldown", default=300)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    # parser.add_argument("--init_wts", type=str, help="path to initial weights", default="/azuredrive/dumps/expB_learning_distance_from_mfld/init_model_weights.pt")
    parser.add_argument("--init_wts", type=str, help="path to initial weights", default=None)
    parser.add_argument("--num_classes", type=int, help="number of manifolds", default=1)
    parser.add_argument("--input_size", type=int, help="input size", default=2)

    parser.add_argument("--train_fn", type=str, help="path to train data") 
    parser.add_argument("--val_fn", type=str, help="path to val data") 
    parser.add_argument("--test_fn", type=str, help="path to test data")
    parser.add_argument("--ftname", type=str, help="named attribute for features when fetching dataset", default="normed_points") 
    parser.add_argument("--tgtname", type=str, help="named attribute for labels when fetching dataset", default="normed_distances") 
    
    args = parser.parse_args()


    TRAIN_FLAG = args.train
    BATCH_SIZE = args.batch_size
    CUDA = args.cuda
    TASK = args.task
    NUM_EPOCHS = args.num_epochs
    SAVE_DIR = args.save_dir
    NAME = args.name

    WARMUP = args.warmup
    COOLDOWN = args.cooldown
    LR = args.lr

    INIT_WTS = args.init_wts
    NUM_CLASSES = args.num_classes
    INPUT_SIZE = args.input_size

    TRAIN_FN = args.train_fn
    VAL_FN = args.val_fn
    TEST_FN = args.test_fn

    FTNAME = args.ftname
    TGTNAME = args.tgtname


    specs_dict = {
        "batch_size": args.batch_size,
        "cuda": args.cuda,
        "task": args.task,
        "num_epochs": args.num_epochs,
        "save_dir": args.save_dir,
        "name": args.name,
        "warmup": args.warmup,
        "cooldown": args.cooldown,
        "lr": args.lr,
        "init_wts": args.init_wts,
        "num_classes": args.num_classes,
        "input_size": args.input_size,
        "train_fn": args.train_fn,
        "val_fn": args.val_fn,
        "test_fn": args.test_fn,
        "ftname": args.ftname,
        "tgtname": args.tgtname
    }

    # if specs file is provided, override all past values
    if args.specs is not None:

        specs_dict = json.load(open(args.specs))
        BATCH_SIZE, CUDA, TASK, NUM_EPOCHS, SAVE_DIR, NAME,\
            WARMUP, COOLDOWN, LR, INIT_WTS, NUM_CLASSES, INPUT_SIZE,\
            TRAIN_FN,VAL_FN, TEST_FN, FTNAME, TGTNAME = load_specs(specs_dict)


    if TRAIN_FLAG:
        
        train_set = torch.load(TRAIN_FN)
        val_set = torch.load(VAL_FN)
        
        
        # train_perm = torch.randperm(train_set.N)
        # val_perm = torch.randperm(val_set.N)

        # print("train_perm:", train_perm[:20])
        # print("val_perm:", val_perm[:20])

        # train_set.points_n = train_set.points_n[train_perm]
        # train_set.distances = train_set.distances[train_perm]

        # val_set.points_n = val_set.points_n[val_perm]
        # val_set.distances = val_set.distances[val_perm]

        # train_set.points_n = train_set.points_n
        # train_set.distances = train_set.distances

        # val_set.points_n = val_set.points_n
        # val_set.distances = val_set.distances

        NUM_WORKERS = 8

        dataloaders = {
            "train": DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, worker_init_fn=seed_worker),
            "val": DataLoader(dataset=val_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
        }  

        device = torch.device("cuda:{}".format(CUDA) if torch.cuda.is_available() and CUDA else "cpu")

        model = MLPwithNormalisation(input_size=INPUT_SIZE, output_size=NUM_CLASSES, hidden_sizes=[512] * 4, weight_norm=False, use_tanh=False, use_relu=False)
        # model = MLP(input_size=INPUT_SIZE, output_size=NUM_CLASSES, hidden_sizes=[512] * 4, use_tanh=False)
        # model = ResNet18(num_classes=args.num_classes)
        if args.init_wts is not None:
            model.load_state_dict(torch.load(INIT_WTS))
        
        
        loss_func = nn.MSELoss()
        # loss_func = weighted_mse_loss
        if args.task == "classification":
            loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

        NUM_EPOCHS = NUM_EPOCHS 
        scheduler_params = {"warmup": WARMUP, "cooldown": COOLDOWN}
        lr_sched_factor = lambda epoch: epoch / (scheduler_params["warmup"]) if epoch <= scheduler_params["warmup"] else (1 if epoch > scheduler_params["warmup"] and epoch < scheduler_params["cooldown"] else max(0, 1 + (1 / (scheduler_params["cooldown"] - NUM_EPOCHS)) * (epoch - scheduler_params["cooldown"])))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_sched_factor)




        model, optimizer, train_loss_matrix, val_loss_matrix = train(model=model, optimizer=optimizer, loss_func=loss_func, dataloaders=dataloaders,\
                         device=device, task=TASK, num_epochs=NUM_EPOCHS, save_dir=SAVE_DIR, feature_name=FTNAME, target_name=TGTNAME,\
                         name=NAME, scheduler=scheduler, scheduler_params=scheduler_params, specs_dict=specs_dict)