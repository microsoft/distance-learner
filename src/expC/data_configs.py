import os
import sys
import copy

import numpy as np

def sphere_cfg():

    train_cfg_dict = {
        "N": 100000,
        "num_neg": None,
        "n": 2,
        "k": 2,
        "r": 0.5,
        "max_norm": 0.25,
        "D": 0.2,
        "mu": 0,
        "sigma": 1,
        "seed": 23,
        "normalize": True,
        "gamma": 0.5
    }

    val_cfg_dict = copy.deepcopy(train_cfg_dict)
    val_cfg_dict["seed"] = 101

    test_cfg_dict = copy.deepcopy(train_cfg_dict)
    test_cfg_dict["seed"] = 89

    cfg = {
        "train": train_cfg_dict,
        "val": val_cfg_dict,
        "test": test_cfg_dict
    }

    return cfg

def conc_spheres_cfg():

    train_cfg_dict = {
        "N": 100000,
        "num_neg": None,
        "n": 2,
        "k": 2,
        "r": 0.5,
        "g": 0.7,
        "D": 0.2,
        "max_norm": 0.3,
        "mu": 0,
        "sigma": 1,
        "seed": 23,
        "normalize": True,
        "gamma": 0.5
    }

    val_cfg_dict = copy.deepcopy(train_cfg_dict)
    val_cfg_dict["seed"] = 101

    test_cfg_dict = copy.deepcopy(train_cfg_dict)
    test_cfg_dict["seed"] = 89

    cfg = {
        "train": train_cfg_dict,
        "val": val_cfg_dict,
        "test": test_cfg_dict
    }

    return cfg

def ittw_swissrolls_cfg():

    train_cfg = {
        "N": 100000,
        "num_neg": None,
        "n": 2,
        "k": 2,
        "D": 20,
        "max_norm": 40,
        "contract": 100,
        "mu": 0,
        "sigma": 1,
        "seed": 23,
        "gamma": 0.5,
        "t_min": 150,
        "t_max": 450,
        "num_turns": None,
        "omega": np.pi * 0.01
    }

    val_cfg = copy.deepcopy(train_cfg)
    val_cfg["N"] = 20000
    val_cfg["seed"] = 101
    test_cfg = copy.deepcopy(train_cfg)
    test_cfg["N"] = 20000
    test_cfg["seed"] = 89

    cfg = {
        "train": train_cfg,
        "val": val_cfg,
        "test": test_cfg
    }

    return cfg