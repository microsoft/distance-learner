"""
Given a model and its data split, and an adversarial attack
this measures the efficacy of the attack at various epsilons
"""
import os
import re
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import json
import copy
import time
import copy
import random

from collections import OrderedDict

import numpy as np

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
from sacred.observers import FileStorageObserver


from datagen import datagen
from expB import learn_mfld_distance as lmd
from expC.model_ingredients import model_ingredient
from attacks import *

ex = Experiment("stdclf_vs_distlearn", ingredients=[model_ingredient])

@ex.config
def config(model):

    cuda = 1

    num_workers = 8
    OFF_MFLD_LABEL = 2

    batch_size = 512

    task = "regression"

    num_classes = model["output_size"]
    input_size = model["input_size"] # dimension in which manifold is embedded

    ftname = "normed_points" # feature from the dataset to train on
    tgtname = "normed_actual_distances" # target values to train against
    if task == "clf":
        tgtname = "classes"


    proj_dir = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/"
    data_tag = "rdm_concspheres_k100n100_noninfdist"

    dump_dir = os.path.join(proj_dir, data_tag)

    runs = ["1", "2"]

    