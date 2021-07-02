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

# from manifold import GeneralManifoldAttrs, SpecificManifoldAttrs, Manifold
from ..single.swissroll import *
# from datagen.synthetic.single.swissroll import *

class IntertwinedSwissRolls(Dataset):

    def __init__(self, N=1000, num_neg=None, n=100, k=3, D=1.5, max_norm=2, mu=10,\
                 sigma=5, seed=42, t_min=1.5, t_max=4.5, omega=np.pi, num_turns=None, noise=0,\
                 correct=True, scale=None, contract=2, g=identity, d_g=d_identity,\
                 height=21, rotation=None, translation=None, anchor=None, normalize=True, norm_factor=None,\
                 gamma=0.5, **kwargs):

        """
        :param t_min: start of time interval for sampling
        :type t_min: float
        :param t_max: end of time interval for sampling
        :type t_max: float
        :param omega: angular velocity along the swiss roll
        :type omega: float
        :param num_turns: number of total 'turns' in the swiss roll
        :type num_turns: float
        :param noise: noise to be added
        :type noise: float
        :param correct: Swiss roll should look like a swiss roll. If yours is not looking like one, enable this flag for some heuristics to kick in
        :type correct: bool
        :param scale: used when `correct` is 
        :param contract: by how much to contract the amplitude of the inner swiss roll
        :type contract: float
        :param g: "amplitude" function for swiss roll
        :type g: function
        :param d_g: derivative of the "amplitude" function
        :type d_g: function
        :param height: 'height' of the swiss roll in the non-lateral directions
        :type height: float
        """

        self._N = N
        self._num_neg = num_neg
        self._n = n
        self._k = k
        self._D = D
        self._max_norm = max_norm
        self._mu = mu
        self._sigma = sigma
        self._seed = seed
        
        self._t_min = t_min
        self._t_max = t_max
        self._omega = omega
        self._num_turns = num_turns
        self._noise = noise
        self._correct = correct
        self._scale = scale
        self._contract = contract
        self._height = height
        
        self._anchor = anchor

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

        if self._max_norm is None:
            self._max_norm = contract / 2
            """this is to ensure that the off-manifold points of the two 
            swiss rolls do not overlap"""
            self._D = 0.6 * self._max_norm

        self.g = eval(g)
        self.d_g = eval(d_g)
        self.g_text = g
        self.d_g_text = d_g
        self.g_contract = lambda x: self.g(x) - contract
        self.dg_contract = lambda x: self.d_g(x)
        self.g_contract_text = g + " - " + str(contract)
        self.dg_contract_text = d_g + " - " + str(0)

        self.S1 = None
        self.S2 = None

        self.all_points_k = None
        self.all_points = None
        self.all_distances = None
        self.all_actual_distances = None
        self.class_labels = None

        self.normed_all_points = self.all_points
        self.normed_all_distances = self.all_distances
        self.normed_all_actual_distances = self.all_actual_distances
        self.norm_factor = norm_factor
        self.gamma = gamma
        self.fix_center = None

    @property 
    def N(self):
        return self._N
    
    @N.setter
    def N(self, x):
        raise RuntimeError("cannot set `N` after instantiation")

    @property
    def num_neg(self):
        return self._num_neg

    @num_neg.setter
    def num_neg(self, x):
        return RuntimeError("cannot set `num_neg` after instantiation")

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, x):
        return RuntimeError("cannot set `n` after instantiation")

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, x):
        return RuntimeError("cannot set `k` after instantiation")

    @property
    def D(self):
        return self._D

    @D.setter
    def D(self, x):
        return RuntimeError("cannot set `D` after instantiation")

    @property
    def max_norm(self):
        return self._max_norm

    @max_norm.setter
    def max_norm(self, x):
        return RuntimeError("cannot set `max_norm` after instantiation")

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, x):
        return RuntimeError("cannot set `mu` after instantiation")

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, x):
        return RuntimeError("cannot set `sigma` after instantiation")

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, x):
        return RuntimeError("cannot set `seed` after instantiation")

    @property
    def t_min(self):
        return self._t_min

    @t_min.setter
    def t_min(self, n):
        raise RuntimeError("cannot set `t_min` after instantiation!")

    @property
    def t_max(self):
        return self._t_max

    @t_max.setter
    def t_max(self, n):
        raise RuntimeError("cannot set `t_max` after instantiation!")

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, n):
        raise RuntimeError("cannot set `noise` after instantiation!")

    @property
    def num_turns(self):
        return self._num_turns

    @num_turns.setter
    def num_turns(self, n):
        raise RuntimeError("cannot set `num_turns` after instantiation!")

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, n):
        raise RuntimeError("cannot set `omega` after instantiation!")

    @property
    def correct(self):
        return self._correct

    @correct.setter
    def correct(self, n):
        raise RuntimeError("cannot set `correct` after instantiation!")

    @property
    def gap(self):
        return self._gap

    @gap.setter
    def gap(self, n):
        raise RuntimeError("cannot set `gap` after instantiation!")

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, n):
        raise RuntimeError("cannot set `scale` after instantiation!")

    @property
    def contract(self):
        return self._contract
    
    @contract.setter
    def contract(self, x):
        raise RuntimeError("cannot set `contract` after instantiation!")

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, n):
        raise RuntimeError("cannot set `height` after instantiation!")
    
    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, x):
        raise RuntimeError("cannot set `anchor` after instantiation!")


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


    def compute_points(self):

        tot_count_per_mfld = self._N // 2
        neg_count_per_mfld = self._num_neg // 2 if self._num_neg is not None else None

        self.S1 = RandomSwissRoll(N=tot_count_per_mfld, num_neg=neg_count_per_mfld, n=self._n,\
            k=self._k, D=self._D, max_norm=self._max_norm, mu=self._mu, sigma=self._sigma,\
            seed=self._seed, t_min=self._t_min, t_max=self._t_max, omega=self._omega,\
            num_turns=self._num_turns, noise=self._noise, correct=self._correct,\
            scale=self._scale, g=self.g_text, d_g=self.d_g_text, height=self._height, rotation=self._rotation,\
            translation=self._translation, normalize=self._normalize, norm_factor=self._norm_factor,\
            gamma=self._gamma)

        self.S1.compute_points()
        print("[InterTwinedSwissRolls]: Generated S1")

        self.S2 = RandomSwissRoll(N=tot_count_per_mfld, num_neg=neg_count_per_mfld, n=self._n,\
            k=self._k, D=self._D, max_norm=self._max_norm, mu=self._mu, sigma=self._sigma,\
            seed=self._seed, t_min=self._t_min, t_max=self._t_max, omega=self._omega,\
            num_turns=self._num_turns, noise=self._noise, correct=self._correct,\
            scale=self._scale, g=self.g_contract_text, d_g=self.dg_contract_text, height=self._height, rotation=self._rotation,\
            translation=self._translation, normalize=self._normalize, norm_factor=self._norm_factor,\
            gamma=self._gamma)

        self.S2.compute_points()
        print("[InterTwinedSwissRolls]: Generated S2")

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

        # true distances of points in S1 to S2 and vice versa are not available and marked `-1`
        self.all_actual_distances = np.zeros((self.S1.genattrs.N + self.S2.genattrs.N, 2))
        self.all_actual_distances[:self.S1.genattrs.N, 0] = self.S1.genattrs.actual_distances.reshape(-1)
        self.all_actual_distances[:self.S1.genattrs.N, 1] = np.inf
        self.all_actual_distances[self.S1.genattrs.N:, 1] = self.S2.genattrs.actual_distances.reshape(-1)
        self.all_actual_distances[self.S1.genattrs.N:, 0] = np.inf

        self.all_points = torch.from_numpy(self.all_points).float()
        self.all_distances = torch.from_numpy(self.all_distances).float()
        self.all_actual_distances = torch.from_numpy(self.all_actual_distances).float()
        self.class_labels = torch.from_numpy(self.class_labels).long()

        if self._normalize:
            self.norm()
            print("[InterTwinedSwissRolls]: Overall noramalization done")

        self.get_all_points_k()

    def __len__(self):
        return self.all_points.shape[0]

    def __getitem__(self, idx):
        return {
            "points": self.all_points[idx],
            "distances": self.all_distances[idx],
            "actual_distances": self.all_actual_distances[idx],
            "normed_points": self.normed_all_points[idx],
            "normed_distances": self.normed_all_distances[idx],
            "normed_actual_distances": self.normed_all_actual_distances[idx],
            "classes": self.class_labels[idx]
        }

    def norm(self):
        """normalise points and distances so that the whole setup lies in a unit sphere"""
        if self._norm_factor is None:
            min_coord = torch.min(self.all_points).item()
            max_coord = torch.max(self.all_points).item()
            self.norm_factor = max_coord - min_coord
        
        self.normed_all_points = self.all_points / self.norm_factor
        self.normed_all_distances = self.all_distances / self.norm_factor
        self.normed_all_actual_distances = self.all_actual_distances / self.norm_factor

        self.S1.genattrs.normed_points_n = self.normed_all_points[:self._N//2]
        self.S1.genattrs.normed_distances = self.normed_all_distances[:self._N//2]
        self.S1.genattrs.normed_actual_distances = self.normed_all_actual_distances[:self._N//2]

        self.S2.genattrs.normed_points_n = self.normed_all_points[self._N//2:]
        self.S2.genattrs.normed_distances = self.normed_all_distances[self._N//2:]
        self.S2.genattrs.normed_actual_distances = self.normed_all_actual_distances[self._N//2:]


        # change anchor point to bring it closer to origin (smaller numbers are easier to learn)
        tmp = self.gamma if self.gamma is not None else 1
        self.fix_center = tmp * np.ones(self._n)
        if self.anchor is None:
            self._anchor = self.normed_all_points[np.argmin(self.S1.specattrs.t)]
            assert (self.anchor == self._anchor).all()
        self.normed_all_points = self.normed_all_points - self.anchor + self.fix_center

        self.normed_all_points = self.normed_all_points.float()
        self.normed_all_distances = self.normed_all_distances.float()
        self.normed_all_actual_distances = self.normed_all_actual_distances.float()

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
                k_dim_samples[start:start + N_attr//2] = getattr(self, attr).genattrs.points_k
                k_dim_samples[start + N_attr//2:start + N_attr] = getattr(self, attr).genattrs.pre_images_k
                start += N_attr

        self.all_points_k = torch.from_numpy(k_dim_samples).float()
        return k_dim_samples

    def invert_points(self, normed_points):
        """invert normalised points to unnormalised values"""
        anchor = self.normed_all_points[np.argmin(self.S1.specattrs.t)]
        normed_points = normed_points - self.fix_center + anchor
        return normed_points * self.norm_factor

    def invert_distances(self, normed_distances):
        """invert normalised distances to unnormalised values"""
        return normed_distances * self.norm_factor

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
                if attr in ["g", "d_g"]:
                    attrs[attr] = eval(attrs[attr + "_text"])
                elif attr == "g_contract":
                    attrs[attr] = lambda x: eval(attrs["g_text"])(x) - attrs["contract"]
                elif attr == "dg_contract":
                    attrs[attr] = lambda x: eval(attrs["d_g_text"])(x)
                setattr(self, attr, attrs[attr])

        self.S1 = RandomSwissRoll()
        self.S1.load_data(S1_dump)

        self.S2 = RandomSwissRoll()
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
                if attr in ["g", "d_g", "g_contract", "dg_contract"]:
                    continue
                specs_attrs[attr] = attr_set[attr]
                
                    # specs_attrs[attr] = inspect.getsourcelines(specs_attrs[attr])[0][0].split("=")[1].strip()
            else:
                data_attrs[attr] = attr_set[attr]

        with open(specs_fn, "w+") as f:
            json.dump(specs_attrs, f)

        torch.save(data_attrs, data_fn)

        self.S1.save_data(S1_dir)
        self.S2.save_data(S2_dir)

        
    @classmethod
    def get_demo_cfg_dict(cls, n=3, k=2):

        train_cfg_dict = {
            "N": 100000,
            "num_neg": None,
            "n": n,
            "k": k,
            "D": 20,
            "max_norm": 40,
            "contract": 100,
            "mu": 0,
            "sigma": 1,
            "seed": 42,
            "gamma": 0.5,
            "t_min": 150,
            "t_max": 450,
            "num_turns": None,
            "omega": np.pi * 0.01
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
    def make_train_val_test_splits(cls, cfg_dict=None, save_dir=None):

        if cfg_dict is None:
            cfg_dict = cls.get_demo_cfg_dict()

        train_cfg = cfg_dict["train"]
        train_set = cls(**train_cfg)
        train_set.compute_points()

        val_cfg = cfg_dict["val"]
        val_cfg["rotation"] = train_set.rotation
        val_cfg["translation"] = train_set.translation
        val_cfg["anchor"] = train_set.anchor
        val_cfg["norm_factor"] = train_set.norm_factor
        val_set = cls(**val_cfg)
        val_set.compute_points()

        test_cfg = cfg_dict["test"]
        test_cfg["rotation"] = train_set.rotation
        test_cfg["translation"] = train_set.translation
        test_cfg["anchor"] = train_set.anchor
        test_cfg["norm_factor"] = train_set.norm_factor
        test_set = cls(**test_cfg)
        test_set.compute_points()

        if save_dir is not None:
            cls.save_splits(train_set, val_set, test_set, save_dir)

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




    
