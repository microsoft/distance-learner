import os
import sys
import json
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
                 height=21, rotation=None, translation=None, normalize=True, norm_factor=None,\
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
        self.g_contract = lambda x: self.g(x) - contract
        self.dg_contract = lambda x: self.d_g(x)

        self.S1 = None
        self.S2 = None

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
    

    def compute_points(self):

        tot_count_per_mfld = self._N // 2
        neg_count_per_mfld = self._num_neg // 2 if self._num_neg is not None else None

        self.S1 = RandomSwissRoll(N=tot_count_per_mfld, num_neg=neg_count_per_mfld, n=self._n,\
            k=self._k, D=self._D, max_norm=self._max_norm, mu=self._mu, sigma=self._sigma,\
            seed=self._seed, t_min=self._t_min, t_max=self._t_max, omega=self._omega,\
            num_turns=self._num_turns, noise=self._noise, correct=self._correct,\
            scale=self._scale, g=self.g, d_g=self.d_g, height=self._height, rotation=self._rotation,\
            translation=self._translation, normalize=self._normalize, norm_factor=self._norm_factor,\
            gamma=self._gamma)

        self.S1.compute_points()
        print("[InterTwinedSwissRolls]: Generated S1")

        self.S2 = RandomSwissRoll(N=tot_count_per_mfld, num_neg=neg_count_per_mfld, n=self._n,\
            k=self._k, D=self._D, max_norm=self._max_norm, mu=self._mu, sigma=self._sigma,\
            seed=self._seed, t_min=self._t_min, t_max=self._t_max, omega=self._omega,\
            num_turns=self._num_turns, noise=self._noise, correct=self._correct,\
            scale=self._scale, g=self.g_contract, d_g=self.dg_contract, height=self._height, rotation=self._rotation,\
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
        self.all_actual_distances[:self.S1.genattrs.N, 1] = -1
        self.all_actual_distances[self.S1.genattrs.N:, 1] = self.S2.genattrs.actual_distances.reshape(-1)
        self.all_actual_distances[self.S1.genattrs.N:, 0] = -1

        self.all_points = torch.from_numpy(self.all_points).float()
        self.all_distances = torch.from_numpy(self.all_distances).float()
        self.all_actual_distances = torch.from_numpy(self.all_actual_distances).float()
        self.class_labels = torch.from_numpy(self.class_labels).long()

        

        if self._normalize:
            self.norm()
            print("[InterTwinedSwissRolls]: Overall noramalization done")

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
        self.normed_distances = self.all_distances / self.norm_factor
        self.normed_actual_distances = self.all_actual_distances / self.norm_factor

        self.S1.genattrs.normed_points_n = self.normed_all_points[:self._N//2]
        self.S1.genattrs.normed_distances = self.normed_distances[:self._N//2]
        self.S1.genattrs.normed_actual_distances = self.normed_actual_distances[:self._N//2]

        self.S2.genattrs.normed_points_n = self.normed_all_points[self._N//2:]
        self.S2.genattrs.normed_distances = self.normed_all_points[self._N//2:]
        self.S2.genattrs.normed_actual_distances = self.normed_all_points[self._N//2:]


        # change anchor point to bring it closer to origin (smaller numbers are easier to learn)
        tmp = self.gamma if self.gamma is not None else 1
        self.fix_center = tmp * np.ones(self._n)
        anchor = self.normed_all_points[np.argmin(self.S1.specattrs.t)]
        self.normed_all_points = self.normed_all_points - anchor + self.fix_center

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
                if attr in ["g", "d_g", "g_contract", "dg_contract"]:
                    attrs[attr] = eval(attrs[attr])
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
                specs_attrs[attr] = attr_set[attr]
                if attr in ["g", "d_g", "g_contract", "dg_contract"]:
                    specs_attrs[attr] = inspect.getsourcelines(specs_attrs[attr])[0][0].split("=")[1].strip()
            else:
                data_attrs[attr] = attr_set[attr]

        with open(specs_fn, "w+") as f:
            json.dump(specs_attrs, f)

        torch.save(data_attrs, data_fn)

        self.S1.save_data(S1_dir)
        self.S2.save_data(S2_dir)

        




    