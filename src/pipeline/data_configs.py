# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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

def synth_conc_spheres_cfg():

    train_cfg_dict = {
        "N": 6500000,
        "num_neg": 6000000,
        "n": 500,
        "k": 50,
        "r": 1.0,
        "g": 0.3,
        "D": 0.07,
        "max_norm": 0.1,
        "mu": 0,
        "sigma": 1,
        "seed": 23,
        "normalize": True,
        "norm_factor": 1,
        "bp": 0.09,
        "M": 1,
        "inferred": False,
        "online": False,
        "off_online": False,
        "augment": False,
        "gamma": 0,
        "cache_dir": "../../data_cache/train/"
    }

    val_cfg_dict = copy.deepcopy(train_cfg_dict)
    val_cfg_dict["N"] = 200000
    val_cfg_dict["num_neg"] = None
    val_cfg_dict["inferred"] = False
    val_cfg_dict["cache_dir"] = "../../data_cache/val/"
    val_cfg_dict["seed"] = 101

    test_cfg_dict = copy.deepcopy(train_cfg_dict)
    test_cfg_dict["N"] = 200000
    test_cfg_dict["num_neg"] = None
    test_cfg_dict["inferred"] = False
    test_cfg_dict["cache_dir"] = "../../data_cache/test/"
    test_cfg_dict["seed"] = 89

    cfg = {
        "train": train_cfg_dict,
        "val": val_cfg_dict,
        "test": test_cfg_dict
    }


def inf_conc_spheres_cfg():

    train_cfg_dict = {
        "N": 20000,
        "num_neg": 10000,
        "n": 2,
        "k": 2,
        "r": 1.0,
        "g": 0.3,
        "D": 0.07,
        "max_norm": 0.1,
        "mu": 0,
        "sigma": 1,
        "seed": 23,
        "normalize": True,
        "norm_factor": 1,
        "bp": 0.09,
        "M": 1,
        "inferred": True,
        "online": False,
        "off_online": False,
        "augment": False,
        "max_t_delta": 1e-3,
        "recomp_tn": False,
        "gamma": 0,
        "cache_dir": "../../data_cache/train/",
        "on_mfld_noise": 0.0
    }

    val_cfg_dict = copy.deepcopy(train_cfg_dict)
    val_cfg_dict["num_neg"] = None
    val_cfg_dict["inferred"] = False
    val_cfg_dict["cache_dir"] = "../../data_cache/val/"
    val_cfg_dict["seed"] = 101

    test_cfg_dict = copy.deepcopy(train_cfg_dict)
    test_cfg_dict["num_neg"] = None
    test_cfg_dict["inferred"] = False
    test_cfg_dict["cache_dir"] = "../../data_cache/test/"
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
        "omega": np.pi * 0.01,
        "online": False,
        "off_online": False,
        "augment": False
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

def inf_ittw_swissrolls_cfg():

    train_cfg = {
        "N": 100000,
        "num_neg": None,
        "n": 2,
        "k": 2,
        "D": 0.2,
        "max_norm": 0.4,
        "contract": 1,
        "mu": 0,
        "sigma": 1,
        "seed": 23,
        "gamma": 0.5,
        "t_min": 1.5,
        "t_max": 4.5,
        "height": 0.21,
        "num_turns": None,
        "omega": np.pi,
        "inferred": True,
        "online": False,
        "off_online": False,
        "augment": False,
        "gamma": 0
    }

    val_cfg = copy.deepcopy(train_cfg)
    val_cfg["N"] = 20000
    val_cfg["seed"] = 101
    val_cfg["num_neg"] = None
    val_cfg["inferred"] = False

    test_cfg = copy.deepcopy(train_cfg)
    test_cfg["N"] = 20000
    test_cfg["seed"] = 89
    test_cfg["num_neg"] = None
    test_cfg["inferred"] = False

    cfg = {
        "train": train_cfg,
        "val": val_cfg,
        "test": test_cfg
    }

    return cfg


def inf_ittw_swissrolls_cfg2():

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
        "height": 21,
        "num_turns": None,
        "omega": np.pi * 0.01,
        "inferred": True,
        "online": False,
        "off_online": False,
        "augment": False,
        "gamma": 0
    }

    val_cfg = copy.deepcopy(train_cfg)
    val_cfg["N"] = 20000
    val_cfg["seed"] = 101
    val_cfg["num_neg"] = None
    val_cfg["inferred"] = False

    test_cfg = copy.deepcopy(train_cfg)
    test_cfg["N"] = 20000
    test_cfg["seed"] = 89
    test_cfg["num_neg"] = None
    test_cfg["inferred"] = False

    cfg = {
        "train": train_cfg,
        "val": val_cfg,
        "test": test_cfg
    }

    return cfg

def inf_ws_spheres_cfg():

    train_cfg_dict = {
        "N": 20000,
        "num_neg": 10000,
        "n": 2,
        "k": 2,
        "r": [1.0, 1.0],
        "D": 0.07,
        "max_norm": 0.14,
        "mu": 0,
        "sigma": 1,
        "seed": 23,
        "normalize": True,
        "norm_factor": 1,
        "bp": 0.09,
        "M": 1,
        "inferred": True,
        "online": False,
        "off_online": False,
        "augment": False,
        "max_t_delta": 1e-3,
        "recomp_tn": False,
        "gamma": 0,
        "c_dist": 2.5,
        "same_rot": False,
        "cache_dir": "../../data_cache/train/"
    }

    val_cfg_dict = copy.deepcopy(train_cfg_dict)
    val_cfg_dict["num_neg"] = None
    val_cfg_dict["inferred"] = False
    val_cfg_dict["cache_dir"] = "../../data_cache/val/"
    val_cfg_dict["seed"] = 101

    test_cfg_dict = copy.deepcopy(train_cfg_dict)
    test_cfg_dict["num_neg"] = None
    test_cfg_dict["inferred"] = False
    test_cfg_dict["cache_dir"] = "../../data_cache/test/"
    test_cfg_dict["seed"] = 89

    cfg = {
        "train": train_cfg_dict,
        "val": val_cfg_dict,
        "test": test_cfg_dict
    }

    return cfg

def ws_spheres_cfg():

    train_cfg_dict = {
        "N": 20000,
        "num_neg": 10000,
        "n": 2,
        "k": 2,
        "r": [1.0, 1.0],
        "D": 0.07,
        "max_norm": 0.14,
        "mu": 0,
        "sigma": 1,
        "seed": 23,
        "normalize": True,
        "norm_factor": 1,
        "bp": 0.09,
        "M": 1,
        "inferred": False,
        "online": False,
        "off_online": False,
        "augment": False,
        "max_t_delta": None,
        "recomp_tn": False,
        "gamma": 0,
        "c_dist": 2.5,
        "cache_dir": "../../data_cache/train/"
    }

    val_cfg_dict = copy.deepcopy(train_cfg_dict)
    val_cfg_dict["num_neg"] = None
    val_cfg_dict["inferred"] = False
    val_cfg_dict["cache_dir"] = "../../data_cache/val/"
    val_cfg_dict["seed"] = 101

    test_cfg_dict = copy.deepcopy(train_cfg_dict)
    test_cfg_dict["num_neg"] = None
    test_cfg_dict["inferred"] = False
    test_cfg_dict["cache_dir"] = "../../data_cache/test/"
    test_cfg_dict["seed"] = 89

    cfg = {
        "train": train_cfg_dict,
        "val": val_cfg_dict,
        "test": test_cfg_dict
    }

    return cfg

def mnist_cfg():

    strategy = "only"
    has_val = False

    train_cfg_dict = {
        "on_mfld_path": "../../data/datasets/MNIST",
        "k": 10,
        "n": 784,
        "use_labels": [1, 8],
        "off_mfld_label": 9,
        "download": True,
        "split": "train",
        "seed": 23,
        "num_neg": 1000000,
        "nn": 50,
        "buf_nn": 2,
        "max_t_delta": 1e-3,
        "D": 7e-2,
        "max_norm": 1e-1,
        "M": 1.0,
        "online": False,
        "off_online": False,
        "augment": False,
        "transform": None,
        "gamma": 0.5
    }

    val_cfg_dict = copy.deepcopy(train_cfg_dict)
    val_cfg_dict["num_neg"] = 0
    val_cfg_dict["split"] = "test" # since has_val == False

    test_cfg_dict = copy.deepcopy(val_cfg_dict)

    cfg_dict = {
        "strategy": "only",
        "has_val": False,
        "train": train_cfg_dict,
        "val": val_cfg_dict,
        "test": test_cfg_dict
    }

    return cfg_dict
