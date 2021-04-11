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


class MLP(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_sizes=[512, 512, 512, 512]):
        
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        layers = [("fcn-0", nn.Linear(self.input_size, hidden_sizes[0])), ("relu-0", nn.ReLU())]
        
        for i in range(len(hidden_sizes) - 1):

            layers.append(("fcn-{n}".format(n=i+1), nn.Linear(hidden_sizes[i], hidden_sizes[i+1])))
            layers.append(("relu-{n}".format(n=i+1), nn.ReLU()))
                    
        layers.append(("fcn-" + str(len(hidden_sizes)), nn.Linear(hidden_sizes[-1], output_size)))
        
        self.layers = nn.Sequential(OrderedDict(layers))
        
    def forward(self, X):
        logits = self.layers(X)
        return logits

class ConvNet1(nn.Module):
    """CNN-based architecture"""
    
    
    def __init__(self, input_size, output_size):
        
        super(ConvNet1, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2)
        self.bn_1 = nn.BatchNorm1d(16)
        self.conv_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.bn_2 = nn.BatchNorm1d(16)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=2)
        self.dp_1 = nn.Dropout(0.2)
        
        
        
        
        self.blk_out = nn.Sequential(OrderedDict([
            ("fc-1", nn.Linear(in_features=16 * 124, out_features=1))
        ]))
        
    def forward(self, X):

        X = X.view(X.shape[0], 1, -1)


        logits = self.conv_1(X)
        logits = self.bn_1(logits)
        logits = self.conv_2(logits)
        logits = self.bn_2(logits)
        logits = self.maxpool_1(logits)
        logits = self.dp_1(logits)

        logits = logits.view(logits.shape[0], -1)
        logits = self.blk_out(logits)
        
        return logits

class ConvNet2(nn.Module):
    """CNN-based architecture"""
    
    
    def __init__(self, input_size, output_size):
        
        super(ConvNet2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.conv_11 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self.bn_11 = nn.BatchNorm1d(32)
        self.conv_12 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.bn_12 = nn.BatchNorm1d(32)
        self.maxpool_11 = nn.MaxPool1d(kernel_size=2)
        self.dp_11 = nn.Dropout(0.2)
        
        self.conv_21 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=2)
        self.bn_21 = nn.BatchNorm1d(16)
        self.conv_22 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.bn_22 = nn.BatchNorm1d(16)
        self.maxpool_21 = nn.MaxPool1d(kernel_size=2)
        self.dp_21 = nn.Dropout(0.5)
        
        
        
        
        self.blk_out = nn.Sequential(OrderedDict([
            ("fc-1", nn.Linear(in_features=16 * 15, out_features=128)),
            ("relu-fc-1", nn.ReLU()),
            ("bn-fc-1", nn.BatchNorm1d(128)),
            ("dp-fc-1", nn.Dropout(0.8)),
            ("fc-2", nn.Linear(in_features=128, out_features=1))
        ]))
        
    def forward(self, X):

        X = X.view(X.shape[0], 1, -1)

        logits = F.relu(self.conv_11(X))
        logits = self.bn_11(logits)
        logits = F.relu(self.conv_12(logits))
        logits = self.bn_12(logits)
        logits = self.maxpool_11(logits)
        
        logits = F.relu(self.conv_21(logits))
        logits = self.bn_21(logits)
        logits = F.relu(self.conv_22(logits))
        logits = self.bn_22(logits)
        logits = self.maxpool_21(logits)
        
        logits = logits.view(logits.shape[0], -1)
        logits = self.blk_out(logits)
        
        return logits

class ConvNet3(nn.Module):

    """CNN-based architecture"""
    
    
    def __init__(self, input_size, output_size):
        
        super(ConvNet3, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.conv_11 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2)
        self.bn_11 = nn.BatchNorm1d(16)
        self.conv_12 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.bn_12 = nn.BatchNorm1d(16)
        self.maxpool_11 = nn.MaxPool1d(kernel_size=2)
        self.dp_11 = nn.Dropout(0.2)
        
        self.conv_21 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.bn_21 = nn.BatchNorm1d(32)
        self.conv_22 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.bn_22 = nn.BatchNorm1d(32)
        self.maxpool_21 = nn.MaxPool1d(kernel_size=2)
        self.dp_21 = nn.Dropout(0.5)
        
        
        
        
        self.blk_out = nn.Sequential(OrderedDict([
            ("fc-1", nn.Linear(in_features=32 * 15, out_features=128)),
            ("relu-fc-1", nn.ReLU()),
            ("bn-fc-1", nn.BatchNorm1d(128)),
            ("dp-fc-1", nn.Dropout(0.8)),
            ("fc-2", nn.Linear(in_features=128, out_features=1))
        ]))
        
    def forward(self, X):
        X = X.view(X.shape[0], 1, -1)

        logits = F.relu(self.conv_11(X))
        logits = self.bn_11(logits)
        logits = F.relu(self.conv_12(logits))
        logits = self.bn_12(logits)
        logits = self.maxpool_11(logits)
        logits = self.dp_11(logits)
        
        logits = F.relu(self.conv_21(logits))
        logits = self.bn_21(logits)
        logits = F.relu(self.conv_22(logits))
        logits = self.bn_22(logits)
        logits = self.maxpool_21(logits)
        logits = self.dp_21(logits)
        
        logits = logits.view(logits.shape[0], -1)
        logits = self.blk_out(logits)
        
        return logits


