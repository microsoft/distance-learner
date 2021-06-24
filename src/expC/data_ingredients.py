from ast import NodeTransformer
import os
import sys
import copy

import torch
import numpy as np
from sacred import Ingredient

from datagen.synthetic.single import sphere, swissroll
from datagen.synthetic.multiple import intertwinedswissrolls

from data_configs import sphere_cfg

DATA_CONFIGS = {
    "single-sphere": sphere_cfg
}

DATA_TYPE = {
    "single-sphere": sphere.RandomSphere,
    "single-swissroll": swissroll.RandomSwissRoll,
    "ittw-swissrolls": intertwinedswissrolls.IntertwinedSwissRolls
}

data_ingredient = Ingredient('data')

@data_ingredient.config
def data_cfg():
    mtype = "single-sphere" # manifold type
    generate = True # generate fresh dataset
    save_dir = None # location of the dataset
    data_params = DATA_CONFIGS[mtype]()



@data_ingredient.capture
def initialise_data(data_params, mtype="single-sphere", generate=True,\
    save_dir=None, **kwargs):
    
    data_cfg_dict = dict(copy.deepcopy(data_params))
    train_set = None
    val_set = None
    test_set = None
    if generate:
        train_set, val_set, test_set = DATA_TYPE[mtype].make_train_val_test_splits(data_cfg_dict, save_dir)
    else:
        train_set, val_set, test_set = DATA_TYPE[mtype].load_splits(save_dir)

    return train_set, val_set, test_set
