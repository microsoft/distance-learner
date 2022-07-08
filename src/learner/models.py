# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
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

from expB.ptcifar.models import ResNet18


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes=[512, 512, 512, 512], use_tanh=True, use_relu=True, **kwargs):
        
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.use_tanh = use_tanh
        self.use_relu = use_relu

        layers = [("fcn-0", nn.Linear(self.input_size, hidden_sizes[0])), ("relu-0", nn.ReLU())]
        
        for i in range(len(hidden_sizes) - 1):
            
            
            layers.append(
                ("fcn-{n}".format(n=i+1), nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            )

            layers.append(
                ("relu-{n}".format(n=i+1), nn.ReLU())
            )



            
                    
        layers.append(("fcn-" + str(len(hidden_sizes)), nn.Linear(hidden_sizes[-1], output_size)))
        if not self.use_tanh:
            if not self.use_relu:
                layers.append(("sigmoid-" + str(len(hidden_sizes)), nn.Sigmoid()))
            else:
                layers.append(("relu-" + str(len(hidden_sizes)), nn.ReLU()))
        else:
            layers.append(("tanh-" + str(len(hidden_sizes)), nn.Tanh()))


        self.layers = nn.Sequential(OrderedDict(layers))
        
    def forward(self, X):
        logits = self.layers(X)
        return logits

class MLPwithNormalisation(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_sizes=[512, 512, 512, 512], weight_norm=True, use_tanh=True, use_relu=True, **kwargs):
        
        super(MLPwithNormalisation, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.weight_norm = weight_norm
        self.use_tanh = use_tanh
        self.use_relu = use_relu

        layers = None

        if not self.weight_norm:

            layers = [("fcn-0", nn.Linear(self.input_size, hidden_sizes[0])), ("bn-0", nn.LayerNorm(hidden_sizes[0])), ("relu-0", nn.ReLU())]
        
        else:

            layers =  [("fcn-0", nn.utils.weight_norm(nn.Linear(self.input_size, hidden_sizes[0]))), ("relu-0", nn.ReLU())]


        for i in range(len(self.hidden_sizes) - 1):
            
            

            if not self.weight_norm:
                layers.append(
                    ("fcn-{n}".format(n=i+1), nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                )
            

                layers.append(
                    ("bn-{n}".format(n=i+1), nn.LayerNorm(hidden_sizes[i+1]))
                )

            else:
                layers.append(
                    ("fcn-{n}".format(n=i+1), nn.utils.weight_norm(nn.Linear(hidden_sizes[i], hidden_sizes[i+1])))
                )

            layers.append(
                    ("relu-{n}".format(n=i+1), nn.ReLU())
            )

            

            
        # last layer  
        layers.append(("fcn-" + str(len(hidden_sizes)), nn.Linear(hidden_sizes[-1], output_size)))

        if not self.use_tanh:
            if self.use_relu:
                # my own addition
                layers.append(("relu-" + str(len(hidden_sizes)), nn.ReLU()))
            else:
                layers.append(("sigmoid-" + str(len(hidden_sizes)), nn.Sigmoid()))
        else:
            layers.append(("tanh-" + str(len(hidden_sizes)), nn.Tanh()))

        # last tanh mandatory like the paper
        # layers.append(("th", nn.Tanh()))


        self.layers = nn.Sequential(OrderedDict(layers))
        
    def forward(self, X):
        logits = self.layers(X)
        return logits

class MTMLPwithNormalisation(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_sizes=[512, 512, 512, 512], num_tasks=2, weight_norm=True, use_tanh=True, use_relu=True, **kwargs):
        
        super(MTMLPwithNormalisation, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.num_tasks = num_tasks
        self.weight_norm = weight_norm
        self.use_tanh = use_tanh
        self.use_relu = use_relu

        layers = None

        if not self.weight_norm:

            layers = [("fcn-0", nn.Linear(self.input_size, hidden_sizes[0])), ("bn-0", nn.LayerNorm(hidden_sizes[0])), ("relu-0", nn.ReLU())]
        
        else:

            layers =  [("fcn-0", nn.utils.weight_norm(nn.Linear(self.input_size, hidden_sizes[0]))), ("relu-0", nn.ReLU())]


        for i in range(len(self.hidden_sizes) - 3):

            if not self.weight_norm:
                layers.append(
                    ("fcn-{n}".format(n=i+1), nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                )

                layers.append(
                    ("bn-{n}".format(n=i+1), nn.LayerNorm(hidden_sizes[i+1]))
                )

            else:
                layers.append(
                    ("fcn-{n}".format(n=i+1), nn.utils.weight_norm(nn.Linear(hidden_sizes[i], hidden_sizes[i+1])))
                )

            layers.append(
                    ("relu-{n}".format(n=i+1), nn.ReLU())
            )

        self.shared_layers = nn.Sequential(OrderedDict(layers))
        
        self.task_layers = {}

        for i in range(num_tasks):
            
            layers = list()

            for j in range(len(self.hidden_sizes) - 3, len(self.hidden_sizes) - 1):

                if not self.weight_norm:
                    
                    layers.append(
                        ("fcn-{t}-{n}".format(t=i+1, n=j+1), nn.Linear(hidden_sizes[j], hidden_sizes[j+1]))
                    )

                    layers.append(
                        ("bn-{t}-{n}".format(t=i+1, n=j+1), nn.LayerNorm(hidden_sizes[j+1]))
                    )

                else:
                    layers.append(
                        ("fcn-{t}-{n}".format(t=i+1, n=j+1), nn.utils.weight_norm(nn.Linear(hidden_sizes[j], hidden_sizes[j+1])))
                    )

                layers.append(
                        ("relu-{t}-{n}".format(t=i+1, n=j+1), nn.ReLU())
                )

            # for last layer of the task
            layers.append(("fcn-" + str(self.num_tasks) + "-" + str(len(hidden_sizes)), nn.Linear(hidden_sizes[-1], output_size // self.num_tasks)))
        
            if not self.use_tanh:
                if self.use_relu:
                    # my own addition
                    layers.append(("relu-" + str(self.num_tasks) + "-" + str(len(hidden_sizes)), nn.ReLU()))
                else:
                    layers.append(("sigmoid-" + str(self.num_tasks) + "-" + str(len(hidden_sizes)), nn.Sigmoid()))
            else:
                layers.append(("tanh-" + str(self.num_tasks) + "-" + str(len(hidden_sizes)), nn.Tanh()))

            self.task_layers["task-" + str(i+1)] = nn.Sequential(OrderedDict(layers))
        self.task_layers = nn.ModuleDict(self.task_layers)
        
    def forward(self, X):
        shared_logits = self.shared_layers(X)
        logits = torch.zeros((X.shape[0], self.output_size))
        logits = logits.to(shared_logits.device)

        start_idx = 0

        for i in range(self.num_tasks):
            logits[:, i*(self.output_size // self.num_tasks):(i+1)*(self.output_size // self.num_tasks)] = self.task_layers["task-" + str(i+1)](shared_logits)

        return logits





class ConvNet1(nn.Module):
    """CNN-based architecture"""
    
    
    def __init__(self, input_size, output_size, **kwargs):
        
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
    
    
    def __init__(self, input_size, output_size, **kwargs):
        
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
    
    
    def __init__(self, input_size, output_size, **kwargs):
        
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

class MTLModelForDistanceAndClass(nn.Module):

    

    def __init__(self, input_size, output_size, **kwargs):
        """
            :param input_size: dims of input 
            :type input_size: int
            :param output_size: number of classes 
            :type output_size: int
        """        


        super(MTLModelForDistanceAndClass, self).__init__()    

        self.input_size = input_size
        self.output_size = output_size

        self.shared_resnet = self.ResNet18(num_classes=512)


        self.distance_branch = nn.Sequential(OrderedDict([
            ("dist-linear-00", nn.Linear(512, 256)),
            ("dist-linear-01", nn.Linear(256, 128)),
            ("dist-linear-01", nn.Linear(128, self.output_size)), 
        ]))

        self.class_branch = nn.Sequential(OrderedDict([
            ("class-linear-00", nn.Linear(512, 256)),
            ("class-linear-01", nn.Linear(256, 128)),
            ("class-linear-01", nn.Linear(128, self.output_size)), 
        ]))

    def forward(self, X):

        shared_logits = self.shared_resnet(X)
        distance_logits = self.distance_branch(shared_logits)
        class_logits = self.class_branch(shared_logits)

        return distance_logits, class_logits
