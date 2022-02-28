from ast import NodeTransformer
import os
import sys
import copy

import torch
import numpy as np
from sacred import Ingredient

from datagen.synthetic.single import sphere, swissroll
from datagen.synthetic.multiple import intertwinedswissrolls, concentricspheres

from data_configs import conc_spheres_cfg, sphere_cfg, ittw_swissrolls_cfg

DATA_CONFIGS = {
    "single-sphere": sphere_cfg,
    "ittw-swissrolls": ittw_swissrolls_cfg,
    "conc-spheres": conc_spheres_cfg
}

DATA_TYPE = {
    "single-sphere": sphere.RandomSphere,
    "single-swissroll": swissroll.RandomSwissRoll,
    "ittw-swissrolls": intertwinedswissrolls.IntertwinedSwissRolls,
    "conc-spheres": concentricspheres.ConcentricSpheres
}

data_ingredient = Ingredient('data')

@data_ingredient.config
def data_cfg():
    mtype = "ittw-swissrolls" # manifold type
    generate = False # generate fresh dataset
    backup_dir = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_concspheres_test/" 
    logdir = "/data/dumps/expC_dist_learner_for_adv_ex/rdm_concspheres_test/" # high-level dump folder
    data_tag = "rdm_swiss_rolls_k2n2" # low-level data directory name
    data_dir = os.path.join(logdir, data_tag, "data") # complete data directory path
    data_params = DATA_CONFIGS[mtype]()

@data_ingredient.capture
def initialise_data(data_params, mtype="single-sphere", generate=True,\
    data_dir=None, data_tag=None, **kwargs):
    
    save_dir = None
    if data_dir is not None:
        save_dir = data_dir
    data_cfg_dict = dict(copy.deepcopy(data_params))
    train_set = None
    val_set = None
    test_set = None

    if generate:
        train_set, val_set, test_set = DATA_TYPE[mtype].make_train_val_test_splits(data_cfg_dict, save_dir)
    else:
        train_set, val_set, test_set = DATA_TYPE[mtype].load_splits(save_dir)

    return train_set, val_set, test_set
