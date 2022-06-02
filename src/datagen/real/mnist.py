import os
from posixpath import split
import re
import sys
import json
import copy
import uuid
import time
import shutil
import random
import inspect
import multiprocessing
from functools import partial
from collections.abc import Iterable

import numpy as np
import scipy as sp
import numpy.linalg as npla
import scipy.linalg as spla

import torch
import torchvision
from torch.data.utils import Dataset, Dataloader

from tqdm import tqdm

from utils import *
from .manifolds import RealWorldManifolds

logger = init_logger(__name__)

class MNISTManifolds(RealWorldManifolds, Dataset):

    def __init__(
        self,
        num_pos,
        on_mfld_path,
        k,
        n=784,
        use_labels=[1, 9],
        off_mfld_label=0,
        download=False,
        split="train",
        N=None,
        num_neg=None,
        nn=30,
        max_t_delta=1e-3,
        max_norm=1e-1,
        M=1.0,
        transform=None,
        **kwargs):

        # MNIST is small. Always load entire data in memory
        self._load_all = True

        super.__init__(num_pos, on_mfld_path, k,\
             n, use_labels, off_mfld_label, download,\
             self._load_all, split, N, num_neg, nn, max_t_delta,\
             max_norm, M, transform, **kwargs)

        if self.transform is None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])


    def load_raw_om_data(self):
        
        train = True if self.split == "train" else False
        logger.info("[{}]: loading MNIST dataset".format(self.__class__.__name__))
        dataset = torchvision.datasets.MNIST(self.on_mfld_path,\
             train=train, download=self.download, transform=self.transform)
        logger.info("[{}]: MNIST data loaded".format(self.__class__.__name__))

        assert len(dataset) == self.num_pos, "num_pos ({}) value invalid. does not match dataset size ({})!".format(self.num_pos, len(dataset))

        tmp = torch.zeros(len(dataset), self.n)
        tmp_cls = torch.zeros(len(dataset)).long()

        batch_size = 8192
        num_workers = 8
        dataloader = Dataloader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

        for idx, X, y in tqdm(enumerate(dataloader)):
            tmp[idx*batch_size:(idx+1)*batch_size] = X.reshape(X.shape[0], -1)
            tmp_cls[idx*batch_size:(idx+1)*batch_size] = y.reshape(y.shape[0])
        
        tmp_cls = tmp_cls.long() # just in case

        dataset_flat = (tmp, tmp_cls)

        return dataset, dataset_flat

    def init_onmfld_pts(self, om_augs=None):
        return super().init_onmfld_pts(om_augs)

    def find_knn(self, X):
        return super().find_knn(X)

    def make_poca_idx(self):
        return super().make_poca_idx()

    def make_inferred_off_mfld(self, pp_chunk_size=5000):
        return super().make_inferred_off_mfld(pp_chunk_size)

    def compute_points(self, om_samples=None, om_class_labels=None):
        return super().compute_points(om_samples, om_class_labels)

    def save_data(self, save_dir):
        return super().save_data(save_dir)

    def load_data(self, dump_dir):
        return super().load_data(dump_dir)

    @classmethod
    def get_demo_cfg_dict(cls, N=60000, num_neg=60000):

        strategy = "only"
        has_val = False

        train_cfg_dict = {
            "num_pos": 60000,
            "on_mfld_path": "./mnist_data",
            "k": 25,
            "n": 784,
            "use_labels": [1, 8],
            "off_mfld_label": 9,
            "load_all": True,
            "split": "train",
            "seed": 23,
            "N": N,
            "num_neg": num_neg,
            "nn": 50,
            "max_t_delta": 1e-3,
            "max_norm": 1e-1,
            "M": 1.0,
            "transform": None
        }

        val_cfg_dict = copy.deepcopy(train_cfg_dict)
        val_cfg_dict["num_pos"] = 10000
        val_cfg_dict["num_neg"] = 0
        val_cfg_dict["N"] = 10000
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


    @classmethod
    def make_train_val_test_splits(cls, cfg_dict=None, save_dir=None):
        return super().make_train_val_test_splits(cfg_dict, save_dir)

    @classmethod
    def save_splits(cls, train_set, val_set, test_set, dump_dir):
        return super().save_splits(train_set, val_set, test_set, dump_dir)

    @classmethod
    def load_splits(cls, dump_dir):
        return super().load_splits(dump_dir)