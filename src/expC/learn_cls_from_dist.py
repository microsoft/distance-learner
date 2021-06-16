"""
Explores various ways to classify points on spheres (or general manifold)
using the distance learned in Experiment B (see: `learn_mfld_distance.py`)
"""

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

class DistanceBasedClf(object):
    """
        Classifier that uses distances computed by 
        Distance Regressor for manifolds to predict
        class values
    """

    def __init__(self):
        pass



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