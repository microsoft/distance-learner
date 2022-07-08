# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import re
import sys
import json
import copy
import time
import random
import multiprocessing
from collections.abc import Iterable
from functools import partial

import numpy as np

import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
import scipy.linalg as spla
from torch.utils.data import Dataset

# from manifold import GeneralManifoldAttrs, SpecificManifoldAttrs, Manifold
from ..single.swissroll import *
# from datagen.synthetic.single.swissroll import *

from utils import *
from datagen.datagen_utils import *

logger = init_logger(__name__)

class IntertwinedSwissRolls(Dataset):

    def __init__(self, N=1000, num_neg=None, n=100, k=3, D=1.5, max_norm=2, mu=10,\
                 sigma=5, seed=42, t_min=1.5, t_max=4.5, omega=np.pi, num_turns=None, noise=0,\
                 correct=True, scale=None, contract=2, g=identity, d_g=d_identity,\
                 height=21, rotation=None, translation=None, anchor=None, normalize=True, norm_factor=None,\
                 gamma=0.5, online=False, M=1, inferred=False, max_t_delta=1e-3, nn=None, buffer_nbhrs=2, **kwargs):

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
        :param M: distance to be used for farther manifold
        :type M: float
        :param inferred: whether to use inferred manifold for point computation
        :type inferred: bool
        """

        self._N = N
        self._num_neg = num_neg
        if num_neg is None:
            self._num_neg = N // 2
        self._num_pos = None if self._num_neg is None else self._N - self._num_neg
        self._n = n
        self._k = k
        self._D = D
        self._max_norm = max_norm
        self._mu = mu
        self._sigma = sigma
        self._seed = seed
        self._M = M
        self._inferred = inferred
        self._online = online

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

        self.buf_ht = None
        self.buf_t = None

        if "buf_ht" in kwargs:
            self.buf_ht = kwargs["buf_ht"]

        if "buf_t" in kwargs:
            self.buf_t = kwargs["buf_t"]

        self.on_mfld_t_ = None

        ## only relevant when `self._inferred == True`

        self.avoid_io = True # generate points without writing intermediate steps to disk

        self.all_points_trivial_ = None
        self.all_points_tr_ = None
        self.all_points_rot_ = None

        self._nn = nn # neighborhood size for kNN
        if self._nn is None:
            self._nn = k - 1
        self._buf_nn = buffer_nbhrs # no. of buffer neighbors
        
        self.knn = None # Faiss kNN object
        self.new_knn = None # to store new Faiss object after tangential perturbations
        
        self.nn_indices = None
        self.nn_distances = None

        self.new_nn_indices = None
        self.new_nn_distances = None

        self._max_t_delta = max_t_delta # maximum tangential perturbation allowed

        self.poca_idx = None # indices of points of closes approach in on-manifold points
        self.poca_idx_counts = None # number of times the ith on-manifold point was used to make off-manifold points
        self.new_poca = None # points of closest approach after applying tangential perturbations
        self.new_poca_prturb_sizes = None # perturbation size of tangential perturbation

        self.on_mfld_pts_k_ = None # all on-manifold points
        self.on_mfld_pts_trivial_ = None # trivial embeddings of on-manifold points

        self.tang_dir = None
        self.norm_dir = None
        self.new_poca_dir = None
        self.tang_dset = None
        self.norm_dset = None
        self.new_poca_dset = None



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
    def num_pos(self):
        return self._num_pos

    @num_pos.setter
    def num_pos(self, x):
        return RuntimeError("cannot set `num_pos` after instantiation")

    @property
    def online(self):
        return self._online

    @online.setter
    def online(self, x):
        return RuntimeError("cannot set `online` after instantiation")

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

    @property
    def nn(self):
        return self._nn

    @nn.setter
    def nn(self, nn):
        raise RuntimeError("cannot set `nn` after instantiation!")

    @property
    def buf_nn(self):
        return self._buf_nn

    @buf_nn.setter
    def buf_nn(self, buf_nn):
        raise RuntimeError("cannot set `buf_nn` after instantiation!")

    @property
    def max_t_delta(self):
        return self._max_t_delta

    @max_t_delta.setter
    def max_t_delta(self, max_t_delta):
        raise RuntimeError("cannot set `max_t_delta` after instantiation!")

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, M):
        raise RuntimeError("cannot set `M` after instantiation!")


    def compute_points(self):

        tot_count_per_mfld = self._N // 2
        neg_count_per_mfld = self._num_neg // 2 if self._num_neg is not None else None

        self.S1 = RandomSwissRoll(N=tot_count_per_mfld, num_neg=neg_count_per_mfld, n=self._n,\
            k=self._k, D=self._D, max_norm=self._max_norm, mu=self._mu, sigma=self._sigma,\
            seed=self._seed, t_min=self._t_min, t_max=self._t_max, omega=self._omega,\
            num_turns=self._num_turns, noise=self._noise, correct=self._correct,\
            scale=self._scale, g=self.g_text, d_g=self.d_g_text, height=self._height, rotation=self._rotation,\
            translation=self._translation, normalize=self._normalize, norm_factor=self._norm_factor,\
            gamma=self._gamma, inferred=self._inferred)

        self.S1.compute_points()
        logger.info("[IntertwinedSwissRolls]: Generated S1")

        self.S2 = RandomSwissRoll(N=tot_count_per_mfld, num_neg=neg_count_per_mfld, n=self._n,\
            k=self._k, D=self._D, max_norm=self._max_norm, mu=self._mu, sigma=self._sigma,\
            seed=self._seed, t_min=self._t_min, t_max=self._t_max, omega=self._omega,\
            num_turns=self._num_turns, noise=self._noise, correct=self._correct,\
            scale=self._scale, g=self.g_contract_text, d_g=self.dg_contract_text, height=self._height, rotation=self._rotation,\
            translation=self._translation, normalize=self._normalize, norm_factor=self._norm_factor,\
            gamma=self._gamma, inferred=self._inferred)

        self.S2.compute_points()
        logger.info("[IntertwinedSwissRolls]: Generated S2")

        if not self._inferred:
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
            self.all_actual_distances[:self.S1.genattrs.N, 1] = self.M
            self.all_actual_distances[self.S1.genattrs.N:, 1] = self.S2.genattrs.actual_distances.reshape(-1)
            self.all_actual_distances[self.S1.genattrs.N:, 0] = self.M
            # self.all_actual_distances = np.zeros((self.S1.genattrs.N + self.S2.genattrs.N, 2))
            # self.all_actual_distances[:self.S1.genattrs.N, 0] = self.S1.genattrs.actual_distances.reshape(-1)
            # self.all_actual_distances[:self.S1.genattrs.N, 1] = np.inf
            # self.all_actual_distances[self.S1.genattrs.N:, 1] = self.S2.genattrs.actual_distances.reshape(-1)
            # self.all_actual_distances[self.S1.genattrs.N:, 0] = np.inf

        else:
            self.compute_inferred_points()

        self.all_points = torch.from_numpy(self.all_points).float()
        self.all_distances = torch.from_numpy(self.all_distances).float()
        self.all_actual_distances = torch.from_numpy(self.all_actual_distances).float()
        self.class_labels = torch.from_numpy(self.class_labels).long()

        if self._normalize:
            self.norm()
            logger.info("[IntertwinedSwissRolls]: Overall noramalization done")

        # self.get_all_points_k()

    def _make_poca_idx(self):
        self.poca_idx = np.zeros(self.num_neg, dtype=np.int64)
        
        S1_idx_choice = np.arange(self.S1.genattrs.N - self.S1.genattrs.num_neg, dtype=np.int64)
        assert S1_idx_choice.shape[0] == self.S1.specattrs.t.shape[0]
        S2_idx_choice = np.arange(self.S1.genattrs.N - self.S1.genattrs.num_neg, self.num_pos, dtype=np.int64)
        assert S2_idx_choice.shape[0] == self.S2.specattrs.t.shape[0]
        
        if self.buf_ht is None and self.buf_t is None:
            """
            this ensures that the poca indices are from non
            boundary on-manifold points only
            """
            self._collect_on_mfld_t()
            
            num_onmfld_S1 = self.S1.genattrs.N - self.S1.genattrs.num_neg
            non_bdry_idx_S1 = (self.S1.specattrs.t > self.t_min + self.buf_t) & (self.S1.specattrs.t < self.t_max - self.buf_t) & (self.on_mfld_pts_k_[:num_onmfld_S1, 2:] > self.buf_ht).all(axis=1) & (self.on_mfld_pts_k_[:num_onmfld_S1, 2:] < self.height - self.buf_ht).all(axis=1)
            S1_idx_choice = S1_idx_choice[non_bdry_idx_S1]

            non_bdry_idx_S2 = (self.S2.specattrs.t > self.t_min + self.buf_t) & (self.S2.specattrs.t < self.t_max - self.buf_t) & (self.on_mfld_pts_k_[num_onmfld_S1:, 2:] > self.buf_ht).all(axis=1) & (self.on_mfld_pts_k_[num_onmfld_S1:, 2:] < self.height - self.buf_ht).all(axis=1)
            S2_idx_choice = S2_idx_choice[non_bdry_idx_S2]

        self.poca_idx[:self.num_neg // 2] = np.random.choice(S1_idx_choice, size=self.S1.genattrs.num_neg, replace=True).astype(np.int64)
        self.poca_idx[self.num_neg // 2:] = np.random.choice(S2_idx_choice, size=self.S2.genattrs.num_neg, replace=True).astype(np.int64)
        self.poca_idx_counts = np.zeros(self.num_pos).astype(np.int64)
        tmp = np.unique(self.poca_idx, return_counts=True)
        self.poca_idx_counts[tmp[0]] = tmp[1]
        
    def _collect_on_mfld_k(self):
        self.on_mfld_pts_k_ = np.zeros((self.num_pos, self.k))
        num_on_mfld_S1 = self.S1.genattrs.N - self.S1.genattrs.num_neg
        self.on_mfld_pts_k_[:num_on_mfld_S1] = self.S1.genattrs.points_k
        self.on_mfld_pts_k_[num_on_mfld_S1:] = self.S2.genattrs.points_k
        if self.N < 1e+7:
            self.on_mfld_pts_trivial_ = np.zeros((self.num_pos, self.n))
            self.on_mfld_pts_trivial_[:, :self.k] = self.on_mfld_pts_k_

    def _collect_on_mfld_t(self):
        self.on_mfld_t_ = np.zeros(self.num_pos)
        num_on_mfld_S1 = self.S1.genattrs.N - self.S1.genattrs.num_neg
        self.on_mfld_t_[:num_on_mfld_S1] = self.S1.specattrs.t
        self.on_mfld_t_[num_on_mfld_S1:] = self.S2.specattrs.t

    def _inf_setup(self):
        """setting up data for off manifold samples when computing inferred manifold"""
        self._make_poca_idx()
        logger.info("[IntertwinedSwissRolls]: made poca_idx")
        self._collect_on_mfld_k()
        logger.info("[IntertwinedSwissRolls]: collect on-mfld k-dim points from both spheres")
        
        

    def find_knn(self, X, use_new=False):
        """
            get k-nearest neighbors of on-manifold points in n-dims
            if `use_new == True`, then use `self.new_knn` for computation
            that is trained on `self.new_poca`
        """
        logger.info("[IntertwinedSwissRolls]: use_new == {}".format(use_new))
        if not use_new:
            
            if self.knn is None:
                self.knn = FaissKNeighbors(k=self.nn + self.buf_nn)
                to_fit = self.on_mfld_pts_trivial_
                if self.on_mfld_pts_trivial_ is None:
                    to_fit = np.zeros((self.N - self.num_neg, self.n))
                    to_fit[:, :self.k] = self.on_mfld_pts_k_
                logger.info("[IntertwinedSwissRolls]: fitting knn...")
                self.knn.fit(to_fit)
                logger.info("[IntertwinedSwissRolls]: knn fit done")

            logger.info("[IntertwinedSwissRolls]: predicting nbhrs...")
            distances, indices = self.knn.predict(X)
            logger.info("[IntertwinedSwissRolls]: prediction complete...")

        else:
            if self.new_knn is None:
                logger.info("[IntertwinedSwissRolls]: new_knn is None. fitting now...")
                self.new_knn = FaissKNeighbors(k=self.nn + self.buf_nn)
                logger.info("[IntertwinedSwissRolls]: fitting new_knn...")
                self.new_knn.fit(self.new_poca_dset[:])
                logger.info("[IntertwinedSwissRolls]: new_knn fit done")

            logger.info("[IntertwinedSwissRolls]: predicting nbhrs...")
            distances, indices = self.new_knn.predict(X)
            logger.info("[IntertwinedSwissRolls]: prediction complete...")

        return distances, indices

    def make_inferred_off_mfld2(self, pp_chunk_size=50000):
        """
        roll in T&N construction, on mfld perturbation, and off mfld 
        perturbation all into one. avoid I/O

        :param pp_chunk_size: chunk size for parallel processing
        """
        if self.nn_distances is None or self.nn_indices is None:
            logger.info("[IntertwinedSwissRolls]: knn not computed. computing now ...")
            X = None
            if self.on_mfld_pts_trivial_ is None:
                X = np.zeros((self.N - self.num_neg, self.n))
                X[:, :self.k] = self.on_mfld_pts_k_
                self.nn_distances, self.nn_indices = self.find_knn(X, use_new=False)
            else:
                self.nn_distances, self.nn_indices = self.find_knn(self.on_mfld_pts_trivial_, use_new=False)
        
        

        if self.class_labels is None:
            self.class_labels = np.zeros(self.N).astype(np.int64)
        if self.all_actual_distances is None:
            self.all_actual_distances = np.zeros((self.N, 2))
        if self.all_points is None:
            self.all_points = np.zeros((self.N, self.n))
        
        num_offmfld_per_idx = max(self.poca_idx_counts)
        total_num_neg_made = 0
        # print(num_offmfld_per_idx)

        S1_off_mfld_idx = 0
        S2_off_mfld_idx = self.S1.genattrs.N

        num_pos = self.num_pos
        non_bdry_idx = None
        # if self.buf_ht is not None and self.buf_t is not None:
        #     self._collect_on_mfld_t()
        #     non_bdry_idx = (self.on_mfld_t_ > self.t_min + self.buf_t) & (self.on_mfld_t_ < self.t_max - self.buf_t) & (self.on_mfld_pts_k_[:, 2:] > self.buf_ht).all(axis=1) & (self.on_mfld_pts_k_[:, 2:] < self.height - self.buf_ht).all(axis=1) 
        #     num_pos = int(sum(non_bdry_idx))

        for i in tqdm(range(0, int(num_pos), pp_chunk_size), desc="computing off mfld (2)"):
            
            nbhr_indices = None
            # if self.buf_ht is None and self.buf_t is None:
            nbhr_indices = self.nn_indices[i:min(num_pos, i+pp_chunk_size)]
            # else:
            #     nbhr_indices = self.nn_indices[non_bdry_idx][i:min(num_pos, i+pp_chunk_size)]
            
            
            mask = np.equal(nbhr_indices, np.arange(i, min(num_pos, i+pp_chunk_size)).reshape(-1, 1))
            mask[mask.sum(axis=1) == 0, -1] = True
            nbhr_indices = nbhr_indices[~mask].reshape(nbhr_indices.shape[0], nbhr_indices.shape[1] - 1).astype(np.int64)

            on_mfld_pts = None
            nbhrs = None
            if self.on_mfld_pts_trivial_ is None:
                on_mfld_pts = np.zeros((pp_chunk_size, self.n))
                # if self.buf_ht is not None and self.buf_t is not None:
                #     on_mfld_pts[:, :self.k] = self.on_mfld_pts_k_[non_bdry_idx][i:i+pp_chunk_size]
                # else:
                on_mfld_pts[:, :self.k] = self.on_mfld_pts_k_[i:i+pp_chunk_size]

                nbhrs = np.zeros((pp_chunk_size, nbhr_indices.shape[1], self.n))
                nbhrs[:, :, :self.k] = self.on_mfld_pts_k_[nbhr_indices]
            else:
                on_mfld_pts = self.on_mfld_pts_trivial_[i:i+pp_chunk_size]
                nbhrs = self.on_mfld_pts_trivial_[nbhr_indices]
            # nbhrs - > (50000, 50, 500) on_mfld_pts -> (50000, 500)
            nbhr_local_coords = (nbhrs.transpose(1, 0, 2) - on_mfld_pts).transpose(1, 0, 2)
            # print(nbhr_local_coords.shape)
            #
            # torch.save({
            #     "on_mfld_pts": on_mfld_pts,
            #     "nbhrs": nbhrs
            # },"/data/tmp/k{}n{}.pth".format(self.k, self.n))
                       
            _get_tn_for_on_mfld_idx_partial = partial(_get_tn_for_on_mfld_idx, \
                k=self.k,
                return_dirs=True)

            
            with multiprocessing.Pool(processes=24) as pool:
                
                all_tang_and_norms = pool.starmap(
                    _get_tn_for_on_mfld_idx_partial, 
                    zip(
                        range(i, int(min(i + pp_chunk_size, self.num_pos))),
                        nbhr_local_coords
                        )
                    )
            all_tang_and_norms = np.array(all_tang_and_norms)
            if self.N <= 1e+7:
                self.all_tang_and_norms = all_tang_and_norms

            actual_chunk_size = min(pp_chunk_size, self.num_pos - i)

            on_mfld_pb_coeffs = np.random.normal(size=(actual_chunk_size, num_offmfld_per_idx, self.k - 1))
            # print(on_mfld_pb_coeffs.shape, all_tang_and_norms.shape)
            on_mfld_pb = np.zeros((actual_chunk_size, num_offmfld_per_idx, self.n))
            on_mfld_pb_sizes = np.random.uniform(0, self.max_t_delta, size=(actual_chunk_size, num_offmfld_per_idx))
            
            # the next line was giving an error in ipython for some reason. will take a closer look later
            # hint: https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
            # on_mfld_pb = np.dot(on_mfld_pb_coeffs, all_tang_and_norms[:, :self.k - 1, :])
            
            off_mfld_pb_coeffs = np.random.normal(size=(actual_chunk_size, num_offmfld_per_idx, self.n - self.k + 1))
            off_mfld_pb = np.zeros((actual_chunk_size, num_offmfld_per_idx, self.n))
            off_mfld_pb_sizes = np.random.uniform(0, self.max_norm, size=(actual_chunk_size, num_offmfld_per_idx))
            
            if self.N <= 20000:
                self.on_mfld_pb_sizes = on_mfld_pb_sizes
                self.off_mfld_pb_sizes = off_mfld_pb_sizes

            # the next line was giving an error in ipython for some reason. will take a closer look later
            # hint: https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
            # off_mfld_pb = np.dot(off_mfld_pb_coeffs, all_tang_and_norms[:, self.k - 1:, :])

            for j in range(actual_chunk_size):
                on_mfld_pb[j] = np.dot(on_mfld_pb_coeffs[j], all_tang_and_norms[j, :self.k - 1, :])
                off_mfld_pb[j] = np.dot(off_mfld_pb_coeffs[j], all_tang_and_norms[j, self.k - 1:, :])
            
            on_mfld_pb = on_mfld_pb * np.expand_dims(on_mfld_pb_sizes / np.linalg.norm(on_mfld_pb, axis=-1), axis=-1)
            new_pocas_for_chunk = np.expand_dims(on_mfld_pts, axis=1) + on_mfld_pb

            off_mfld_pb = off_mfld_pb * np.expand_dims(off_mfld_pb_sizes / np.linalg.norm(off_mfld_pb, axis=-1), axis=-1)
            off_mfld_pts_for_chunk = new_pocas_for_chunk + off_mfld_pb
            
            if self.N <= 20000:
                self.on_mfld_pb = torch.from_numpy(on_mfld_pb)
                self.off_mfld_pb = torch.from_numpy(off_mfld_pb)
                self.off_mfld_pts_for_chunk = torch.from_numpy(off_mfld_pts_for_chunk)

            indices_to_use = sum([[((j) * num_offmfld_per_idx) + k for k in range(self.poca_idx_counts[i + j])] for j in range(actual_chunk_size)], [])
            # logger.info("indices_to_use: {}".format(indices_to_use[:10]))
            assert len(indices_to_use) == np.sum(self.poca_idx_counts[i:min(i+pp_chunk_size, self.num_pos)])
            total_num_neg_made += np.sum(self.poca_idx_counts[i:min(i+pp_chunk_size, self.num_pos)])
            off_mfld_pts = off_mfld_pts_for_chunk.reshape(-1, self.n)[indices_to_use]
            off_mfld_dists = off_mfld_pb_sizes.reshape(-1)[indices_to_use]

            pre_image_idx = np.array(sum([[0 if j < self.S1.genattrs.N - self.S1.genattrs.num_neg else 1 for k in range(self.poca_idx_counts[j])] for j in range(i, min(i+pp_chunk_size, self.num_pos))], []))
            assert len(pre_image_idx) == np.sum(self.poca_idx_counts[i:min(i+pp_chunk_size, self.num_pos)])
            
            S1_off_mfld_offset = np.count_nonzero(pre_image_idx == 0)
            self.all_points[S1_off_mfld_idx:S1_off_mfld_idx+S1_off_mfld_offset] = off_mfld_pts[pre_image_idx == 0] 
            self.all_actual_distances[S1_off_mfld_idx:S1_off_mfld_idx+S1_off_mfld_offset, 0] = off_mfld_dists[pre_image_idx == 0]
            self.all_actual_distances[S1_off_mfld_idx:S1_off_mfld_idx+S1_off_mfld_offset, 1] = self.M
            self.class_labels[S1_off_mfld_idx:S1_off_mfld_idx+S1_off_mfld_offset] = 2
            S1_off_mfld_idx += S1_off_mfld_offset

            S2_off_mfld_offset = np.count_nonzero(pre_image_idx == 1)
            self.all_points[S2_off_mfld_idx:S2_off_mfld_idx+S2_off_mfld_offset] = off_mfld_pts[pre_image_idx == 1] 
            self.all_actual_distances[S2_off_mfld_idx:S2_off_mfld_idx+S2_off_mfld_offset, 1] = off_mfld_dists[pre_image_idx == 1]
            self.all_actual_distances[S2_off_mfld_idx:S2_off_mfld_idx+S2_off_mfld_offset, 0] = self.M
            self.class_labels[S2_off_mfld_idx:S2_off_mfld_idx+S2_off_mfld_offset] = 2
            S2_off_mfld_idx += S2_off_mfld_offset
            

        # for on-manifold points
        self.all_actual_distances[self.S1.genattrs.num_neg:self.S1.genattrs.N, 1] = self.M
        self.all_actual_distances[self.S1.genattrs.N+self.S2.genattrs.num_neg:, 0] = self.M
        assert S1_off_mfld_idx == self.S1.genattrs.num_neg
        assert S2_off_mfld_idx - self.S1.genattrs.N == self.S2.genattrs.num_neg
        assert total_num_neg_made == self.num_neg

    def compute_inferred_points(self):

        self._inf_setup()
        logger.info("initial setup complete")

        # print("after tn", self.nn_indices)
        if not self.online:
            self.make_inferred_off_mfld2(pp_chunk_size=5000)

            num_on_mfld_S1 = self.S1.genattrs.N - self.S1.genattrs.num_neg
            self.all_points[self.S1.genattrs.num_neg:self.S1.genattrs.N, :self.k] = self.on_mfld_pts_k_[:num_on_mfld_S1]
            self.all_points[self.S1.genattrs.N+self.S2.genattrs.num_neg:, :self.k] = self.on_mfld_pts_k_[num_on_mfld_S1:]

            self.class_labels[self.S1.genattrs.num_neg:self.S1.genattrs.N] = 0
            self.class_labels[self.S1.genattrs.N+self.S2.genattrs.num_neg:] = 1

            assert (self.all_actual_distances[self.S1.genattrs.num_neg:self.S1.genattrs.N, 0] == 0).all()
            assert (self.all_actual_distances[self.S1.genattrs.N+self.S2.genattrs.num_neg:, 1] == 0).all()
            assert (self.class_labels[:self.S1.genattrs.num_neg] == 2).all()
            assert (self.class_labels[self.S1.genattrs.N:self.S1.genattrs.N + self.S2.genattrs.num_neg] == 2).all(), np.unique(self.class_labels[self.S1.genattrs.N:self.S1.genattrs.N + self.S2.genattrs.num_neg], return_counts=True)

            if self.all_distances is None:
                self.all_distances = self.all_actual_distances.copy()
                self.all_distances[self.all_distances >= self.D] = self.D

            self.all_points_trivial_ = None
            self.all_points_tr_ = None
            self.all_points_rot_ = None
            if self.N < 1e+7:
                self.all_points_tr_ = self.all_points + self.translation
                self.all_points_rot_ = np.dot(self.rotation, self.all_points_tr_.T).T
                self.all_points_trivial_ = self.all_points.copy()
                self.all_points = self.all_points_rot_
                
                self.all_points_trivial_ = torch.from_numpy(self.all_points_trivial_).float()
                self.all_points_tr_ = torch.from_numpy(self.all_points_tr_).float()
                self.all_points_rot_ = torch.from_numpy(self.all_points_rot_).float()
            else:
                self.all_points += self.translation
                self.all_points = np.dot(self.rotation, self.all_points.T).T
                
        else:
            # TODO: implement online sampling of off-manifold points from induced manifold
            pass

        # self.all_distances = torch.from_numpy(self.all_distances).float()
        
        




    def __len__(self):
        return self.normed_all_points.shape[0]

    def __getitem__(self, idx):
        item_attr_map = {
            "points": "all_points",
            "distances": "all_distances",
            "actual_distances": "all_actual_distances",
            "smooth_distances": "all_smooth_distances",
            "normed_smooth_distances": "normed_smooth_distances",
            "normed_points": "normed_all_points",
            "normed_distances": "normed_all_distances",
            "normed_actual_distances": "normed_all_actual_distances",
            "pre_classes": "pre_class_labels",
            "classes": "class_labels"
        }

        batch = dict()
        for attr in item_attr_map:
            if hasattr(self, item_attr_map[attr]) and getattr(self, item_attr_map[attr]) is not None:
                batch[attr] = getattr(self, item_attr_map[attr])[idx]

        # if self.pre_class_labels is not None:
        #     batch["pre_classes"] = self.pre_class_labels[idx]
        # if self.all_smooth_distances is not None:
        #     batch["smooth_distances"] = self.all_smooth_distances[idx]
        # if self.normed_all_smooth_distances is not None:
        #     batch["normed_smooth_distances"] = self.normed_all_smooth_distances[idx]

        return batch

    def norm(self):
        """normalise points and distances so that the whole setup lies in a unit sphere"""
        # if self.norm_factor is None:
        #     min_coord = torch.min(self.all_points).item()
        #     max_coord = torch.max(self.all_points).item()
        #     self.norm_factor = max_coord - min_coord
        if self.norm_factor is None:
            min_coords = torch.min(self.all_points, dim=0)[0]
            max_coords = torch.max(self.all_points, dim=0)[0]
            ranges = max_coords - min_coords
            self.norm_factor = torch.max(ranges).item()


        self.normed_all_points = self.all_points / self.norm_factor
        self.normed_all_distances = self.all_distances / self.norm_factor
        self.normed_all_actual_distances = self.all_actual_distances / self.norm_factor
        
        self.normed_all_actual_distances[:self.S1.genattrs.N, 1] = self.M
        self.normed_all_actual_distances[self.S1.genattrs.N:, 0] = self.M 

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
                if attr in ["g", "d_g", "g_contract", "dg_contract"] or "nn" in attr:
                    continue
                print(attr, attr_set[attr], type(attr_set[attr]))
                specs_attrs[attr] = attr_set[attr]
                
                    # specs_attrs[attr] = inspect.getsourcelines(specs_attrs[attr])[0][0].split("=")[1].strip()
            else:
                data_attrs[attr] = attr_set[attr]

        try:
            with open(specs_fn, "w+") as f:
                json.dump(specs_attrs, f)
        except:
            logger.info("[IntertwinedSwissRolls]: Could not save spec_attrs")

        try:
            torch.save(data_attrs, data_fn)
        except:
            logger.info("[IntertwinedSwissRolls]: Could not save data_attrs")

        try:
            self.S1.save_data(S1_dir)
            self.S2.save_data(S2_dir)
        except:
            logger.info("[IntertwinedSwissRolls]: Could not save anything")

        

        
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



def _get_tn_for_on_mfld_idx(idx,nbhr_local_coords, k, \
     tang_dset_per_dir_size=None, norm_dset_per_dir_size=None, tang_dir=None, norm_dir=None, return_dirs=True):
    """compute tangential and normal directions for all on-manifold points"""
    # print("nbhr_local_coords", nbhr_local_coords.shape)
    pca = PCA(n_components=k-1) # manifold is (k-1) dim so tangent space should be same
    pca.fit(nbhr_local_coords)

    tangential_dirs = pca.components_
    normal_dirs = spla.null_space(tangential_dirs).T

    # if (k == 500 or k == 50) and idx < 50:
    #     # assert normal_dirs.shape[0] == 1
    #     true_normal_dir = on_mfld_pt / np.linalg.norm(on_mfld_pt, ord=2)
    #     est_normal_dir = normal_dirs[0] / np.linalg.norm(normal_dirs[0], ord=2)
    #     dot_prod = np.dot(true_normal_dir, est_normal_dir)
    #     torch.save({"on_mfld_pt": on_mfld_pt, "nbhr_local_coords": nbhr_local_coords, "true_normal_dir": true_normal_dir, "est_normal_dir": est_normal_dir, "dot_prod": dot_prod}, "/data/tmp/{}.pth".format(idx))


    # print("tangent", tangential_dirs.shape)
    # print("normal", normal_dirs.shape)

    # tangential_dirs = np.ones((k - 1, 500))
    # normal_dirs = np.ones((500 - k + 1, 500))

    if tang_dir is not None and tang_dset_per_dir_size is not None:
        cur_tang_dir_idx = idx // tang_dset_per_dir_size
        cur_tang_dir_name = os.path.join(tang_dir, str(cur_tang_dir_idx))
        os.makedirs(cur_tang_dir_name, exist_ok=True)
        tang_fn = os.path.join(cur_tang_dir_name, str(idx) + ".pth")
        torch.save(tangential_dirs, tang_fn)

    if norm_dir is not None and norm_dset_per_dir_size is not None:
        cur_norm_dir_idx = idx // norm_dset_per_dir_size
        cur_norm_dir_name = os.path.join(norm_dir, str(cur_norm_dir_idx))
        os.makedirs(cur_norm_dir_name, exist_ok=True)
        norm_fn = os.path.join(cur_norm_dir_name, str(idx) + ".pth")
        torch.save(normal_dirs, norm_fn)
    # print(idx, tang_dset_per_dir_size, norm_dset_per_dir_size, cur_tang_dir_idx)
    t_and_n = np.zeros((tangential_dirs.shape[1], tangential_dirs.shape[1]))
    t_and_n[:k-1] = tangential_dirs
    t_and_n[k-1:] = normal_dirs
    if return_dirs: return t_and_n
    # return (idx, tangential_dirs, normal_dirs)
    
