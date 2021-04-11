import os
import sys
import time
import copy
import argparse

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

from spheres import RandomSphere
from ptcifar.models import ResNet18
from myNNs import *



def train(model, optimizer, loss_func, dataloaders, device,\
          num_epochs, save_dir, task="regression", name="MLP_512x4_in1000",\
          scheduler={"start_iter": 150, "end_iter": 500}):
    """
        Function to train the model. Also dumps the best model.
        
        Returns the best model and optimizers

    """
    
    # storing the start_lr
    start_lr = None
    for param_group in optimizer.param_groups:
        start_lr = param_group['lr']
    
    TIME_STAMP = time.strftime("%d%m%Y-%H%M%S")

    
    model_dir = os.path.join(save_dir, TIME_STAMP, "models")
    plot_dir = os.path.join(save_dir, TIME_STAMP, "plots")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_fn = os.path.join(plot_dir, name + "_" + TIME_STAMP)
    
    liveloss = PlotLosses(fig_path=plot_fn)
    
    phase = "train"
    
    last_best_epoch_loss = None
    
    
    
    for epoch in tqdm(range(num_epochs)):
        
        
        
        logs = {
            "loss": 0,
            "val_loss": 0,
            "lr": start_lr
        }
        
        
        
        for phase in ["train", "val"]:
            
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
                
                points = batch[0].to(device)
                distances = batch[1].to(device)
                
                model.zero_grad()
                model = model.to(device)
                
                logits = model(points)
                loss = loss_func(logits, distances)
                
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    
                logs[prefix + "loss"] += loss.detach().cpu().item()
                
                num_batches = i + 1
                
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
        
        check = last_best_epoch_loss is None or logs["val_loss"] < last_best_epoch_loss
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
                'scheduler': scheduler
            }
            
            if task == "clf":
                dump["acc"] = logs["acc"]
                dump["val_acc"] = logs["val_acc"]
                dump["f1"] = logs["f1"]
                dump["val_f1"] = logs["val_f1"]
                
            torch.save(dump, os.path.join(model_dir, NAME + "_"+ TIME_STAMP + "val_loss_" + str(logs["val_loss"]) + ".pth"))
        
        if scheduler is not None:
            
            if epoch > scheduler["start_iter"]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(0, start_lr + (((start_lr - 0) / (scheduler["start_iter"] - scheduler["end_iter"])) * (epoch - scheduler["start_iter"])))
                    logs["lr"] = param_group["lr"]
        liveloss.update(logs)
        liveloss.draw()

        
    return model, optimizer
            
            

def test(model, dataloader, device, task="regression"):
    
    model.eval()
    
    all_logits = None
    all_distances = None
    
    with torch.no_grad():
        
        for batch in tqdm(dataloader):
            
            points = batch[0].to(device)
            distances = batch[1]
            
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
            
            if all_distances is None:
                all_distances = distances
            else:
                all_distances = torch.cat((all_distances, distances))
            
    if task == "regression":
        mse = mean_squared_error(all_distances, all_logits)
        mse_on_mfld = mean_squared_error(all_distances[np.round(all_distances) == 0], all_logits[np.round(all_distances) == 0])



        print("MSE for the learned distances:", mse)
        print("MSE for the learned distances (on-manifold):", mse_on_mfld)
        return mse, mse_on_mfld, all_distances, all_logits
    
    elif task == "clf":
        y_pred = torch.max(all_logits, axis=1)[1]
        print(classification_report(all_distances.reshape(-1), y_pred))
        acc = accuracy_score(all_distances.reshape(-1), y_pred)
        f1 = f1_score(all_distances.reshape(-1), y_pred)
        
        return acc, f1, all_distances, y_pred
    
    #TODO: fix this for concentric dataset



def main():

    


