import os
import re
import sys
import json
import copy
import time
import random
import inspect
from collections.abc import Iterable

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from ..single.sphere import *

class ConcentricSpheres(Dataset):

    def __init__(self, N=1000, num_neg=None, n=100, k=2, D=2.0, max_norm=5.0, bp=1.8, M=50, mu=10,\
                sigma=5, seed=42, r=10.0, g=10.0, x_ck=None, rotation=None, translation=None,\
                normalize=True, norm_factor=None, gamma=0.5, anchor=None, online=False,\
                off_online=False, augment=False, **kwargs):
        """
        :param N: number of samples in the dataset
        :type N: int
        :param num_neg: number of off-manifold samples
        :type num_neg: int
        :param n: dim. of space where data is embedded
        :type n: int
        :param k: dim. of spehere (in reality sphere is the generated sphere is (k-1) dim.)
        :type k: int
        :param D: threshold applied on the distance function used for learning
        :type D: float
        :param max_norm: maximum distance upto which points can be generated
        :type max_norm: float
        :param bp: buffer point for distance
        :type bp: float
        :param M: distance value for far-off manifold
        :type M: float
        :param mu: mean of normal distribution from which we sample
        :type mu: float
        :param sigma: std. dev. of normal distribution from which we sample
        :type sigma: float
        :param seed: seed used for random generator
        :type seed: int
        :param r: radius of the inner sphere in the concentric sphere dataset
        :type r: float
        :param g: difference between radii of inner and outer spheres
        :type g: float
        :param x_ck: center of the sphere in k-dim. space
        :type x_ck: numpy.array
        :param rotation: rotation transform applied to the dataset
        :type rotation: numpy.ndarray
        :param translation: translation transform applied to the dataset
        :type translation: numpy.array
        :param normalize: whether to normalize the dataset or not
        :type normalize: bool
        :param norm_factor: normalization factor used for normalizing the dataset
        :type norm_factor: float
        :param gamma: conservative normalization factor used in normalization
        :type gamma: float
        :param anchor: anchor point used in normalization
        :type anchor: numpy.array
        :param online: whether to sample points on-the-fly
        :type online: bool
        :param off_online: whether only off-manifold samples should be sampled on-the-fly
        :type off_online: bool
        :param augment: whether to treat off-manifold points generated on-the-fly as augmentations
        :type augment: bool
        """
    
        self._N = N
        self._num_neg = num_neg
        self._n = n
        self._k = k
        self._D = D
        self._max_norm = max_norm
        self._bp = bp
        self._M = M
        self._mu = mu
        self._sigma = sigma
        self._seed = seed
        self._r = r
        self._g = g
        self._online = online
        self._off_online = off_online
        self._augment = augment
        self._x_ck = x_ck
        if self._x_ck is None:
            self._x_ck = np.random.normal(self._mu, self._sigma, self._k)
        self._x_cn = None

        self._rotation = rotation
        if self._rotation is None:
            self._rotation = np.random.normal(self._mu, self._sigma, (self._n, self._n))
            self._rotation = np.linalg.qr(self._rotation)[0]
        self._translation = translation
        if self._translation is None:
            self._translation = np.random.normal(self._mu, self._sigma, self._n)

        self._normalize = normalize
        self._norm_factor = norm_factor
        self._gamma = gamma
        self._fix_center = None
        self._anchor = anchor

        self.S1 = None
        self.S2 = None

        self.all_points_k = None
        self.all_points = None
        self.all_distances = None
        self.all_actual_distances = None
        self.all_smooth_distances = None
        self.class_labels = None

        self.normed_all_points = self.all_points
        self.normed_all_distances = self.all_distances
        self.normed_all_actual_distances = self.all_actual_distances
        self.normed_all_smooth_distances = self.all_smooth_distances
        self.norm_factor = norm_factor
        self.gamma = gamma
        self.fix_center = None

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        raise RuntimeError("cannot set `N` after instantiation")

    @property
    def num_neg(self):
        return self._num_neg

    @num_neg.setter
    def num_neg(self, x):
        raise RuntimeError("cannot set `num_neg` after instantiation")

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, x):
        raise RuntimeError("cannot set `n` after instantiation")

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, x):
        raise RuntimeError("cannot set `k` after instantiation")

    @property
    def D(self):
        return self._D

    @D.setter
    def D(self, x):
        raise RuntimeError("cannot set `D` after instantiation")

    @property
    def max_norm(self):
        return self._max_norm

    @max_norm.setter
    def max_norm(self, x):
        raise RuntimeError("cannot set `max_norm` after instantiation")

    @property
    def bp(self):
        return self._bp

    @bp.setter
    def bp(self, x):
        raise RuntimeError("cannpt set `bp` after instantiation")

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, x):
        raise RuntimeError("cannpt set `M` after instantiation")

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, x):
        raise RuntimeError("cannot set `mu` after instantiation")

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, x):
        raise RuntimeError("cannot set `sigma` after instantiation")

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, x):
        raise RuntimeError("cannot set `seed` after instantiation")
    
    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, x):
        raise RuntimeError("cannot set `rotation` after instantiation!")

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def traslation(self, x):
        raise RuntimeError("cannot set `translation` after instantiation!")

    @property
    def x_ck(self):
        return self._x_ck

    @x_ck.setter
    def x_ck(self, x_ck):
        raise RuntimeError("cannot set `x_ck` after instantiation!")

    @property
    def x_cn(self):
        return self._x_cn

    @x_cn.setter
    def x_cn(self, x_cn):
        raise RuntimeError("cannot set `x_cn` after instantiation!")

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        raise RuntimeError("cannot set `r` after instantiation!")

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, g):
        raise RuntimeError("cannot set `g` after instantiation!")

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, anchor):
        raise RuntimeError("cannot set `anchor` after instantiation!")

    @property
    def online(self):
        return self._online

    @online.setter
    def online(self, online):
        raise RuntimeError("cannot set `online` after instantiation!")
    
    @property
    def off_online(self):
        return self._off_online

    @off_online.setter
    def off_online(self, off_online):
        raise RuntimeError("cannot set `off_online` after instantiation!")

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, augment):
        raise RuntimeError("cannot set `augment` after instantiation!")

    def compute_points(self):

        tot_count_per_mfld = self._N // 2
        neg_count_per_mfld = self._num_neg // 2 if self._num_neg is not None and not self.online else None

        s_gamma = 0.5 if self._gamma is 0 else self._gamma # gamma = 0 needed for concentric spheres but throws error with constituent spheres
        self.S1 = RandomSphere(N=tot_count_per_mfld, num_neg=neg_count_per_mfld, n=self._n,\
            k=self._k, r=self._r, D=self._D, max_norm=self._max_norm, mu=self._mu, sigma=self._sigma,\
            seed=self._seed, x_ck=self._x_ck, rotation=self._rotation, translation=self._translation,\
            normalize=True, norm_factor=None, gamma=s_gamma, anchor=None, online=self._online, \
            off_online=self._off_online, augment=self._augment)

        self.S1.compute_points()
        print("[ConcentricSpheres]: Generated S1")
        self._x_cn = self.S1.specattrs.x_cn

        self.S2 = RandomSphere(N=tot_count_per_mfld, num_neg=neg_count_per_mfld, n=self._n,\
            k=self._k, r=self._r + self._g, D=self._D, max_norm=self._max_norm, mu=self._mu, sigma=self._sigma,\
            seed=self._seed, x_ck=self._x_ck, rotation=self._rotation, translation=self._translation,\
            normalize=True, norm_factor=None, gamma=s_gamma, anchor=None, online=self._online, \
            off_online=self._off_online, augment=self._augment)

        self.S2.compute_points()
        assert (self._x_cn == self.S2.specattrs.x_cn).all() == True
        print("[ConcentricSpheres]: Generated S2")

        self.all_points = np.vstack((self.S1.genattrs.points_n.numpy(), self.S2.genattrs.points_n.numpy()))
        
        self.all_distances = np.zeros((self.S1.genattrs.N + self.S2.genattrs.N, 2))
        self.all_distances[:self.S1.genattrs.N, 0] = self.S1.genattrs.distances.reshape(-1)
        self.all_distances[:self.S1.genattrs.N, 1] = self._D
        self.all_distances[self.S1.genattrs.N:, 1] = self.S2.genattrs.distances.reshape(-1)
        self.all_distances[self.S1.genattrs.N:, 0] = self._D

        # giving class labels
        # 2: no manifold; 0: S_1; 1: S_2
        self.class_labels = np.zeros(self.S1.genattrs.N + self.S2.genattrs.N, dtype=np.int64)
        self.class_labels[:self.S1.genattrs.num_neg] = 2
        self.class_labels[self.S1.genattrs.num_neg:self.S1.genattrs.N] = 0
        self.class_labels[self.S1.genattrs.N:self.S1.genattrs.N + self.S2.genattrs.num_neg] = 2
        self.class_labels[self.S1.genattrs.N + self.S2.genattrs.num_neg:] = 1

        # true distances of points in S1 to S2 and vice versa are not available and marked `M`
        self.all_actual_distances = np.zeros((self.S1.genattrs.N + self.S2.genattrs.N, 2))
        self.all_actual_distances[:self.S1.genattrs.N, 0] = self.S1.genattrs.actual_distances.reshape(-1)
        self.all_actual_distances[:self.S1.genattrs.N, 1] = self._M
        self.all_actual_distances[self.S1.genattrs.N:, 1] = self.S2.genattrs.actual_distances.reshape(-1)
        self.all_actual_distances[self.S1.genattrs.N:, 0] = self._M

        # smoothed distances of points
        self.all_smooth_distances = np.copy(self.all_actual_distances)
        within_buffer_mask = (self.all_actual_distances > self._bp) & (self.all_actual_distances <= self._max_norm)
        self.all_smooth_distances[within_buffer_mask] = self._bp + ((self._M - self._bp) * ((self.all_smooth_distances[within_buffer_mask] - self._bp) / (self._max_norm - self._bp)))
        self.all_smooth_distances[self.all_actual_distances > self._max_norm] = self._M  # this is not really needed

        self.all_points = torch.from_numpy(self.all_points).float()
        self.all_distances = torch.from_numpy(self.all_distances).float()
        self.all_actual_distances = torch.from_numpy(self.all_actual_distances).float()
        self.all_smooth_distances = torch.from_numpy(self.all_smooth_distances).float()
        self.class_labels = torch.from_numpy(self.class_labels).long()

        if self._normalize:
            self.norm()
            print("[ConcentricSpheres]: Overall noramalization done")

        self.get_all_points_k()

    def get_all_points_k(self):
        """
        get the k-dim embedding of all the normalised points
        
        (only call after normalization, although in the case of swiss rolls it does not matter)
        """
        if self.normed_all_points is None:
            raise RuntimeError("this function is made for normalised points!")

        k_dim_samples = np.zeros((self.N, self.k))
        start = 0
        for attr in vars(self):
            if len(re.findall(r'S[0-9]+', attr)) > 0:
                N_attr = getattr(self, attr).genattrs.N
                num_neg_attr = getattr(self, attr).genattrs.num_neg
                k_dim_samples[start:start + N_attr - num_neg_attr] = getattr(self, attr).genattrs.points_k
                k_dim_samples[start + N_attr - num_neg_attr:start + N_attr] = getattr(self, attr).genattrs.pre_images_k
                start += N_attr

        self.all_points_k = torch.from_numpy(k_dim_samples).float()
        return k_dim_samples

    def norm(self):
        """normalise points and distances so that the whole setup lies in a unit sphere"""
        if self.norm_factor is None:
            self.norm_factor = max(
                [torch.max(self.all_points[:, i]) - torch.min(self.all_points[:, i]) for i in range(self.n)]
            )
            # min_coord = torch.min(self.all_points).item()
            # max_coord = torch.max(self.all_points).item()
        
        self._anchor = self._x_cn / self.norm_factor
        
        self.normed_all_points = self.all_points / self.norm_factor
        self.normed_all_distances = self.all_distances / self.norm_factor
        self.normed_all_actual_distances = self.all_actual_distances / self.norm_factor
        self.normed_all_smooth_distances = self.all_smooth_distances / self.norm_factor

        self.S1.genattrs.normed_points_n = self.normed_all_points[:self._N//2]
        self.S1.genattrs.normed_distances = self.normed_all_distances[:self._N//2]
        self.S1.genattrs.normed_actual_distances = self.normed_all_actual_distances[:self._N//2]

        self.S2.genattrs.normed_points_n = self.normed_all_points[self._N//2:]
        self.S2.genattrs.normed_distances = self.normed_all_distances[self._N//2:]
        self.S2.genattrs.normed_actual_distances = self.normed_all_actual_distances[self._N//2:]

        # change anchor point to bring it closer to origin (smaller numbers are easier to learn)
        tmp = self.gamma if self.gamma is not None else 1
        self.fix_center = tmp * np.ones(self._n)
        self.normed_all_points = self.normed_all_points - self.anchor + self.fix_center

        self.normed_all_points = self.normed_all_points.float()
        self.normed_all_distances = self.normed_all_distances.float()
        self.normed_all_actual_distances = self.normed_all_actual_distances.float()
        self.normed_all_smooth_distances = self.normed_all_smooth_distances.float()
    
    def invert_points(self, normed_points):
        unnormed_points = normed_points - self.fix_center + self.anchor
        unnormed_points = unnormed_points * self.norm_factor
        return unnormed_points

    def invert_distances(self, normed_distances):
        unnormed_distances = self.norm_factor * normed_distances
        return unnormed_distances

    def online_compute_points(self, batch, idx):
        """
        * in case of online sampling:
            Case I : batch will have only on-manifold points for which we need augmentations 
            Case II: batch will not be needed at all
        * In Case I, we need constituent sphere indices of the points, compose a batch and 
          pass them to the `online_compute_points` method of the sphere
        * In Case II, we need neither `batch` or `idx` as input, but just confirm that 
          passing `None` to `self.S*.online_compute_points` does not blast
        * In either case, retrieved points must be normalised using current class's `norm` method
        
        TODO: Complete this method!
        """

        batch_pts = batch["points"]
        class_labels = batch["classes"]

        tot_count_per_mfld = self._N // 2

        # still a bit queasy about this
        mfld = self.S2 if class_labels == 1 else self.S1 

        # find idx of points in the constituent spheres
        cs_idx = idx - tot_count_per_mfld if class_labels == 1 else idx
        
        # extract corresponding batch from the constituent sphere
        # (this will sample points on-the-fly as needed, since the constituent manifold knows the settings)
        cs_batch = mfld[cs_idx]

        # pass extracted batch to method for extracting on-the-fly samples from constituent sphere
        online_cs_batch = mfld.online_compute_points(cs_batch)

        # re-assemble online constituent sphere batch to resemble container class batch
        online_all_points = online_cs_batch["points_n"]
        online_all_disances = online_cs_batch["distances"]
        online_all_actual_distances = online_cs_batch["actual_distances"]
        online_classes = batch["classes"]

        return None 


    def load_data(self, dump_dir):
        specs_fn = os.path.join(dump_dir, "specs.json")
        data_fn = os.path.join(dump_dir, "data.pkl")

        S1_dump = os.path.join(dump_dir, "S1_dump")
        S2_dump = os.path.join(dump_dir, "S2_dump")

        with open(specs_fn) as f:
            specs_attrs = json.load(f)

        data_attrs = torch.load(data_fn)

        attrs = {**specs_attrs, **data_attrs}

        attr_set = vars(self)
        for attr in attr_set:
            if attr in ["S1", "S2"]:
                continue
            if attr in attrs:
                setattr(self, attr, attrs[attr])

        self.S1 = RandomSphere()
        self.S1.load_data(S1_dump)

        self.S2 = RandomSphere()
        self.S2.load_data((S2_dump))


    def save_data(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        S1_dir = os.path.join(save_dir, "S1_dump")
        S2_dir = os.path.join(save_dir, "S2_dump")

        specs_fn = os.path.join(save_dir, "specs.json")
        data_fn = os.path.join(save_dir, "data.pkl")

        specs_attrs = dict()
        data_attrs = dict()

        attr_set = vars(self)
        for attr in attr_set:
            if attr in ["S1", "S2"]:
                continue
            if not isinstance(attr_set[attr], Iterable):
                specs_attrs[attr] = attr_set[attr]                
            else:
                data_attrs[attr] = attr_set[attr]

        with open(specs_fn, "w+") as f:
            json.dump(specs_attrs, f)

        torch.save(data_attrs, data_fn)

        self.S1.save_data(S1_dir)
        self.S2.save_data(S2_dir)

    @classmethod
    def get_demo_cfg_dict(cls, n=2, k=2):

        train_cfg_dict = {
            "N": 100000,
            "num_neg": None,
            "n": n,
            "k": k,
            "D": 0.07,
            "max_norm": 0.10,
            "bp": 0.09,
            "M": 1.0,
            "r": 1.0,
            "g": 0.3,
            "mu": 0,
            "sigma": 1,
            "seed": 42,
            "gamma": 0.5,
            "norm_factor": 1.0
        }

        val_cfg_dict = copy.deepcopy(train_cfg_dict)
        val_cfg_dict["N"] = 10000
        val_cfg_dict["seed"] = 101

        test_cfg_dict = copy.deepcopy(train_cfg_dict)
        test_cfg_dict["N"] = 10000
        test_cfg_dict["seed"] = 89

        cfg_dict = {
            "train": train_cfg_dict,
            "val": val_cfg_dict,
            "test": test_cfg_dict
        }

        return cfg_dict

    @classmethod
    def make_train_val_test_splits(cls, cfg_dict=None, save_dir=None):

        if cfg_dict is None:
            cfg_dict = cls.get_demo_cfg_dict()

        train_cfg = cfg_dict["train"]
        train_set = cls(**train_cfg)
        train_set.compute_points()

        val_cfg = cfg_dict["val"]
        val_cfg["rotation"] = train_set.rotation
        val_cfg["translation"] = train_set.translation
        val_cfg["x_ck"] = train_set.x_ck
        val_cfg["norm_factor"] = train_set.norm_factor
        val_set = cls(**val_cfg)
        val_set.compute_points()

        test_cfg = cfg_dict["test"]
        test_cfg["rotation"] = train_set.rotation
        test_cfg["translation"] = train_set.translation
        test_cfg["x_ck"] = train_set.x_ck
        test_cfg["norm_factor"] = train_set.norm_factor
        test_set = cls(**test_cfg)
        test_set.compute_points()

        if save_dir is not None:
            cls.save_splits(train_set, val_set, test_set, save_dir)

        return train_set, val_set, test_set

    @classmethod
    def save_splits(cls, train_set, val_set, test_set, save_dir):
        train_dir = os.path.join(save_dir, "train")
        val_dir = os.path.join(save_dir, "val")
        test_dir = os.path.join(save_dir, "test")
    
        if save_dir is not None:
            os.makedirs(train_dir, exist_ok=True)
            train_set.save_data(train_dir)
            os.makedirs(val_dir, exist_ok=True)
            val_set.save_data(val_dir)
            os.makedirs(test_dir, exist_ok=True)
            test_set.save_data(test_dir)

        return train_set, val_set, test_set

    @classmethod
    def load_splits(cls, dump_dir):

        train_dir = os.path.join(dump_dir, "train")
        os.makedirs(train_dir, exist_ok=True)
        train_set = cls()
        train_set.load_data(train_dir)

        val_dir = os.path.join(dump_dir, "val")
        os.makedirs(val_dir, exist_ok=True)
        val_set = cls()
        val_set.load_data(val_dir)

        test_dir = os.path.join(dump_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        test_set = cls()
        test_set.load_data(test_dir)

        return train_set, val_set, test_set


    def __len__(self):
        return self.all_points.shape[0]

    def __getitem__(self, idx):
        batch = {
            "points": self.all_points[idx],
            "distances": self.all_distances[idx],
            "actual_distances": self.all_actual_distances[idx],
            "normed_points": self.normed_all_points[idx],
            "normed_distances": self.normed_all_distances[idx],
            "normed_actual_distances": self.normed_all_actual_distances[idx],
            "classes": self.class_labels[idx]
        }

        if self.all_smooth_distances is not None:
            batch["smooth_distances"] = self.all_smooth_distances[idx]
        if self.normed_all_smooth_distances is not None:
            batch["normed_smooth_distances"] = self.normed_all_smooth_distances[idx]

        return batch

    
        


