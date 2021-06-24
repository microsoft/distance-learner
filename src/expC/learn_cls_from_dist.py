"""
Explores various ways to classify points on spheres (or general manifold)
using the distance learned in Experiment B (see: `learn_mfld_distance.py`)
"""

import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import json
import copy
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

from sacred import Experiment

from datagen.synthetic.single import manifold, sphere, swissroll
from datagen.synthetic.multiple import intertwinedswissrolls
from expB import learn_mfld_distance as lmd

from model_ingredients import initialise_model, model_cfg, model_ingredient
from data_ingredients import initialise_data, data_cfg, data_ingredient


# class DistanceBasedClf(object):
#     """
#         Classifier that uses distances computed by 
#         Distance Regressor for manifolds to predict
#         class values
#     """

#     def __init__(self):
#         pass



def argmin_dist_clf(inputs, targets, class_labels, off_mfld_lbl=2):
    """
        argmin distance based clf for on-manifold points

        (currently coded only for at most two spheres)

        
        :param inputs: logits obtained from a distance regressor
        :type inputs: torch.Tensor 
        :param targets: actual values of distances
        :type targets: torch.Tensor
        :param class_labels: class labels from the dataset
        :type class_labels: torch.Tensor
        :param off_mfld_lbl: label for off manifold samples
        :type off_mfld_lbl: int
        :return: minimum and argmin of predicted and target distances
        :rtype: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
    """
    
    OFF_MFLD_LBL = off_mfld_lbl
    # selecting on-manifold points for further computations
    on_mfld_targets = targets[class_labels != OFF_MFLD_LBL]
    on_mfld_inputs = inputs[class_labels != OFF_MFLD_LBL]
    
    min_pred_dist, pred_argmin = torch.min(on_mfld_inputs, axis=1)
    min_true_dist, true_argmin = torch.min(on_mfld_targets, axis=1)

    print(classification_report(true_argmin, pred_argmin, target_names=["S1", "S2"]))

    return min_pred_dist, pred_argmin, min_true_dist, true_argmin

ex = Experiment("stdclf_vs_distlearn", ingredients=[model_ingredient, data_ingredient])

@ex.config
def config(data, model):

    train = True # train flag
    test = False # test flag
    cuda = 0 # GPU device id for training

    num_workers = 8
    OFF_MFLD_LABEL = 2

    batch_size = 512
    task = "regression" # "regression" or "clf"
    train_on_onmfld = False # flag for only training on on-mfld. samples (only useful for stdclf.)
    num_epochs = 500
    warmup = 10
    cooldown = 300
    lr = 1e-5

    num_classes = 2 # basicallly number of manifolds
    input_size = data["data_params"]["train"]["n"] # dimension in which manifold is embedded

    loss_func = "std_mse" # ["std_mse", "masked_mse", "weighted_mse", "cross_entropy"]

    ftname = "normed_points_n" # feature from the dataset to train on
    tgtname = "normed_actual_distances" # target values to train against

    name = "stdclf_vs_distlearn"

    save_dir = "./test_dump"

@ex.capture
def data_setup(task, train, train_on_onmfld, OFF_MFLD_LABEL, batch_size, num_workers):

    train_set, val_set, test_set = initialise_data()

    if task == "clf" and train_on_onmfld:
        if isinstance(train_set, manifold.Manifold):
            train_set.genattrs.all_points = train_set.genattrs.all_points[train_set.genattrs.class_labels != OFF_MFLD_LABEL]
            train_set.genattrs.all_distances = train_set.genattrs.all_distances[train_set.genattrs.class_labels != OFF_MFLD_LABEL]
            train_set.genattrs.normed_all_points = train_set.genattrs.normed_all_points[train_set.genattrs.class_labels != OFF_MFLD_LABEL]
            train_set.genattrs.normed_all_distances = train_set.genattrs.normed_all_distances[train_set.genattrs.class_labels != OFF_MFLD_LABEL]
            train_set.genattrs.class_labels = train_set.genattrs.class_labels[train_set.genattrs.class_labels != OFF_MFLD_LABEL]

            val_set.genattrs.all_points = val_set.genattrs.all_points[val_set.genattrs.class_labels != OFF_MFLD_LABEL]
            val_set.genattrs.all_distances = val_set.genattrs.all_distances[val_set.genattrs.class_labels != OFF_MFLD_LABEL]
            val_set.genattrs.normed_all_points = val_set.genattrs.normed_all_points[val_set.genattrs.class_labels != OFF_MFLD_LABEL]
            val_set.genattrs.normed_all_distances = val_set.genattrs.normed_all_distances[val_set.genattrs.class_labels != OFF_MFLD_LABEL]
            val_set.genattrs.class_labels = val_set.genattrs.class_labels[val_set.genattrs.class_labels != OFF_MFLD_LABEL]

            test_set.genattrs.all_points = test_set.genattrs.all_points[test_set.genattrs.class_labels != OFF_MFLD_LABEL]
            test_set.genattrs.all_distances = test_set.genattrs.all_distances[test_set.genattrs.class_labels != OFF_MFLD_LABEL]
            test_set.genattrs.normed_all_points = test_set.genattrs.normed_all_points[test_set.genattrs.class_labels != OFF_MFLD_LABEL]
            test_set.genattrs.normed_all_distances = test_set.genattrs.normed_all_distances[test_set.genattrs.class_labels != OFF_MFLD_LABEL]
            test_set.genattrs.class_labels = test_set.genattrs.class_labels[test_set.genattrs.class_labels != OFF_MFLD_LABEL]
        else:
            train_set.all_points = train_set.all_points[train_set.class_labels != OFF_MFLD_LABEL]
            train_set.all_distances = train_set.all_distances[train_set.class_labels != OFF_MFLD_LABEL]
            train_set.normed_all_points = train_set.normed_all_points[train_set.class_labels != OFF_MFLD_LABEL]
            train_set.normed_all_distances = train_set.normed_all_distances[train_set.class_labels != OFF_MFLD_LABEL]
            train_set.class_labels = train_set.class_labels[train_set.class_labels != OFF_MFLD_LABEL]

            val_set.all_points = val_set.all_points[val_set.class_labels != OFF_MFLD_LABEL]
            val_set.all_distances = val_set.all_distances[val_set.class_labels != OFF_MFLD_LABEL]
            val_set.normed_all_points = val_set.normed_all_points[val_set.class_labels != OFF_MFLD_LABEL]
            val_set.normed_all_distances = val_set.normed_all_distances[val_set.class_labels != OFF_MFLD_LABEL]
            val_set.class_labels = val_set.class_labels[val_set.class_labels != OFF_MFLD_LABEL]

            test_set.all_points = test_set.all_points[test_set.class_labels != OFF_MFLD_LABEL]
            test_set.all_distances = test_set.all_distances[test_set.class_labels != OFF_MFLD_LABEL]
            test_set.normed_all_points = test_set.normed_all_points[test_set.class_labels != OFF_MFLD_LABEL]
            test_set.normed_all_distances = test_set.normed_all_distances[test_set.class_labels != OFF_MFLD_LABEL]
            test_set.class_labels = test_set.class_labels[test_set.class_labels != OFF_MFLD_LABEL]

    shuffle = True if train else False

    dataloaders = {
        "train": DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers),
        "val": DataLoader(dataset=val_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers),
        "test": DataLoader(dataset=test_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    }

    return train_set, val_set, test_set, dataloaders



@ex.automain
def main(num_epochs, task, loss_func, lr, warmup,\
     cooldown, cuda, ftname, tgtname, name, save_dir):

    device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and cuda else "cpu")
    train_set, val_set, test_set, dataloaders = data_setup()

    model = initialise_model()
    print(model)
    loss_function = lmd.loss_funcs[loss_func]

    if task == "clf":
        loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler_params = {"warmup": warmup, "cooldown": cooldown}
    lr_sched_factor = lambda epoch: epoch / (scheduler_params["warmup"]) if epoch <= scheduler_params["warmup"] else (1 if epoch > scheduler_params["warmup"] and epoch < scheduler_params["cooldown"] else max(0, 1 + (1 / (scheduler_params["cooldown"] - num_epochs)) * (epoch - scheduler_params["cooldown"])))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_sched_factor)
    
    model, optimizer, train_loss_matrix, val_loss_matrix = lmd.train(model, optimizer, loss_function,\
        dataloaders, device, save_dir, scheduler, feature_name=ftname, target_name=tgtname,\
        num_epochs=num_epochs, task=task, name=name, scheduler_params=scheduler_params, specs_dict=None, debug=False)

# if __name__ == '__main__':
#     # lmd.init()
#     run = ex.run()