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
        "gamma": 0.5,
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

