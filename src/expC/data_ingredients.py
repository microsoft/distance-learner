from ast import NodeTransformer
import os
import sys
import copy

import torch
import numpy as np
from sacred import Ingredient

from datagen.synthetic.single import sphere, swissroll
from datagen.synthetic.multiple import intertwinedswissrolls, concentricspheres, wellseparatedspheres
from datagen.real import mnist

from data_configs import conc_spheres_cfg, inf_ittw_swissrolls_cfg, inf_ittw_swissrolls_cfg2, inf_ws_spheres_cfg,\
     sphere_cfg, ittw_swissrolls_cfg, synth_conc_spheres_cfg, inf_conc_spheres_cfg, ws_spheres_cfg, mnist_cfg

DATA_CONFIGS = {
    "single-sphere": sphere_cfg,
    "ittw-swissrolls": ittw_swissrolls_cfg,
    "inf-ittw-swissrolls": inf_ittw_swissrolls_cfg,
    "inf-ittw-swissrolls2": inf_ittw_swissrolls_cfg2,
    "conc-spheres": conc_spheres_cfg,
    "synth-conc-spheres": synth_conc_spheres_cfg,
    "inf-conc-spheres": inf_conc_spheres_cfg,
    "ws-spheres": ws_spheres_cfg,
    "inf-ws-spheres": inf_ws_spheres_cfg,
    "mnist": mnist_cfg
}

DATA_TYPE = {
    "single-sphere": sphere.RandomSphere,
    "single-swissroll": swissroll.RandomSwissRoll,
    "ittw-swissrolls": intertwinedswissrolls.IntertwinedSwissRolls,
    "inf-ittw-swissrolls": intertwinedswissrolls.IntertwinedSwissRolls,
    "inf-ittw-swissrolls2": intertwinedswissrolls.IntertwinedSwissRolls,
    "conc-spheres": concentricspheres.ConcentricSpheres,
    "synth-conc-spheres": concentricspheres.ConcentricSpheres,
    "inf-conc-spheres": concentricspheres.ConcentricSpheres,
    "ws-spheres": wellseparatedspheres.WellSeparatedSpheres,
    "inf-ws-spheres": wellseparatedspheres.WellSeparatedSpheres,
    "mnist": mnist.MNISTManifolds
}

data_ingredient = Ingredient('data')

@data_ingredient.config
def data_cfg():
    mtype = "ittw-swissrolls" # manifold type
    generate = False # generate fresh dataset
    # backup_dir = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_concspheres_test/" 
    # logdir = "/data/dumps/expC_dist_learner_for_adv_ex/rdm_concspheres_test/" # high-level dump folder
    backup_dir = "/azuredrive/deepimage/data2/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex"
    logdir = "/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_concspheres_test"
    # logdir = "/data/t-achetan_2/dumps/expC_dist_learner_for_adv_ex/rdm_concspheres_test"
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
