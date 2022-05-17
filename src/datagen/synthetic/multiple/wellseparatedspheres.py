from functools import partial
import os
import re
import shutil
import sys
import json
import copy
import uuid
import time
import random
import inspect
import multiprocessing
from contextlib import closing
from collections.abc import Iterable

import numpy as np
import scipy.linalg as spla
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import faiss

import torch
from torch.utils.data import Dataset

from tqdm import tqdm

from ..single.sphere import *

from utils import *
from datagen.datagen_utils import *

logger = init_logger(__name__)



class WellSeparatedSpheres(Dataset):

    def __init__(self, N=1000, num_neg=None, n=100, k=2, D=2.0, max_norm=5.0, bp=1.8, M=50, mu=10,\
                sigma=5, seed=42, r=[10, 10], x_ck=None, rotation=None, translation=None,\
                normalize=True, norm_factor=None, gamma=0.5, anchor=None, online=False,\
                off_online=False, augment=False, inferred=False, nn=None, buffer_nbhrs=2,\
                max_t_delta=1e-3, recomp_tn=False, use_new_knn=False, cache_dir="/tmp", c_dist=None, **kwargs):
        """
        :param translation: random translation vector of shape: (n,) [single used for both spheres]
        :type translation: np.array 
        :param rotation: random rotation transform of shape (2, n, n) [one for each sphere]
        :type rotation: np.array 
        :param c_dist: distance between the centres in n-D space
        :type c_dist: float
        """
        if seed is not None: seed_everything(seed)

        self._N = N
        self._num_neg = num_neg
        if num_neg is None:
            self._num_neg = N // 2
        self._num_pos = None if self._num_neg is None else self._N - self._num_neg
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
        self._online = online
        self._off_online = off_online
        self._augment = augment
        self._inferred = inferred
        self._x_ck = x_ck
        self._x_ck = np.random.normal(self._mu, self._sigma, (2, self._k))
        self._c_dist = c_dist
        
        self._x_cn = None

        self._rotation = rotation
        if self._rotation is None:
            self._rotation = np.random.normal(self._mu, self._sigma, (2, self._n, self._n))
            for i in range(2):
                self._rotation[i] = np.linalg.qr(self._rotation[i])[0]
        self._translation = translation
        if self._translation is None:
            self._translation = np.random.normal(self._mu, self._sigma, self._n)

        self._normalize = normalize
        self._norm_factor = norm_factor
        self._gamma = gamma
        self._fix_center = None
        self._anchor = anchor

        
        self._uuid = str(uuid.uuid4())
        self._cache_dir = os.path.join(cache_dir, self._uuid)
        

        self.S1 = None
        self.S2 = None
        self.old_S2 = None
        self.old_S2_data_attrs = None

        self.all_points_k = None
        self.all_points = None
        self.all_distances = None
        self.all_actual_distances = None
        self.all_smooth_distances = None
        self.class_labels = None
        self.pre_class_labels = None

        self.normed_all_points = self.all_points
        self.normed_all_distances = self.all_distances
        self.normed_all_actual_distances = self.all_actual_distances
        self.normed_all_smooth_distances = self.all_smooth_distances
        self.norm_factor = norm_factor
        self.gamma = gamma
        self.fix_center = None

        

        ### only relevant when `self.inferred == True`###

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

        self._recomp_tn = recomp_tn
        self._use_new_knn = use_new_knn

        self.tang_dir = None
        self.norm_dir = None
        self.new_poca_dir = None
        self.tang_dset = None
        self.norm_dset = None
        self.new_poca_dset = None

        if not self.avoid_io:
            os.makedirs(self._cache_dir, exist_ok=True)

        if not self.online and not self.avoid_io:

            self.tang_dir = os.path.join(self.cache_dir, "tangents_at_poca") # cache directory to dump tangents and normals
            self.norm_dir = os.path.join(self.cache_dir, "normals_at_poca") # cache directory to dump tangents and normals

            self.new_poca_dir = os.path.join(self.cache_dir, "new_poca") # cache directory to store new poca

            self.tang_dset = TensorFileDataset(
                root_dir=self.tang_dir,
                total_len=self.num_pos,
                per_dir_size=50000
            )

            self.norm_dset = TensorFileDataset(
                root_dir=self.norm_dir,
                total_len=self.num_pos,
                per_dir_size=50000
            )

            self.new_poca_dset = TensorFileDataset(
                root_dir=self.new_poca_dir,
                total_len=self.num_neg,
                per_dir_size=50000
            )

        # for debugging
        self.all_tang_and_norms = None
        self.on_mfld_pb = None
        self.off_mfld_pb = None
        self.off_mfld_pts_for_chunk = None
        self.on_mfld_pb_sizes = None
        self.off_mfld_pb_sizes = None

    def _reposition_centres(self):
        req_c_dist = (sum(self.r) + 2* (self.max_norm)) * 1.1
        logger.info("[WellSeparatedSpheres]: req. distance between spheres = {}".format(req_c_dist))
        req_c_dist = np.round(req_c_dist, 1)
        logger.info("[WellSeparatedSpheres]: final distance between spheres = {}".format(req_c_dist))
        new_S2_x_ck = np.random.normal(self.mu, self.sigma, self.k)
        new_S2_x_ck = self.x_ck[0] + (req_c_dist * (new_S2_x_ck / np.linalg.norm(new_S2_x_ck, ord=2)))
        self.x_ck[1] = new_S2_x_ck

    def _make_poca_idx(self):
        self.poca_idx = np.zeros(self.num_neg, dtype=np.int64)
        self.poca_idx[:self.num_neg // 2] = np.random.choice(np.arange(self.S1.genattrs.N - self.S1.genattrs.num_neg, dtype=np.int64), size=self.S1.genattrs.num_neg, replace=True).astype(np.int64)
        self.poca_idx[self.num_neg // 2:] = np.random.choice(np.arange(self.S1.genattrs.N - self.S1.genattrs.num_neg, self.num_pos, dtype=np.int64), size=self.S2.genattrs.num_neg, replace=True).astype(np.int64)
        self.poca_idx_counts = np.zeros(self.num_pos).astype(np.int64)
        tmp = np.unique(self.poca_idx, return_counts=True)
        self.poca_idx_counts[tmp[0]] = tmp[1]
        # self.poca_idx =  np.zeros(self.num_neg, dtype=np.int64)
        # tmp = min((self.N // 2) - (self.num_neg // 2), self.num_neg // 2)
        # num_on_mfld_S1 = self.S1.genattrs.N - self.S1.genattrs.num_neg
        # self.poca_idx[:tmp] = np.arange(tmp, dtype=np.int64)
        # self.poca_idx[tmp:2*tmp] = np.arange(num_on_mfld_S1, num_on_mfld_S1 + tmp, dtype=np.int64)
        # # self.poca_idx[:self.num_pos] = np.arange(min(self.num_pos, self.num_neg), dtype=np.int64)
        # self.poca_idx[2*tmp:] = np.random.choice(np.arange(self.num_pos, dtype=np.int64), size=max(0, self.num_neg - 2*tmp), replace=True).astype(np.int64) 

    def _collect_on_mfld_k(self):
        self.on_mfld_pts_k_ = np.zeros((self.num_pos, self.k))
        num_on_mfld_S1 = self.S1.genattrs.N - self.S1.genattrs.num_neg
        self.on_mfld_pts_k_[:num_on_mfld_S1] = self.S1.genattrs.points_k
        self.on_mfld_pts_k_[num_on_mfld_S1:] = self.S2.genattrs.points_k
        if self.N < 1e+7:
            self.on_mfld_pts_trivial_ = np.zeros((self.num_pos, self.n))
            self.on_mfld_pts_trivial_[:, :self.k] = self.on_mfld_pts_k_

    def _inf_setup(self):
        """setting up data for off manifold samples when computing inferred manifold"""
        self._make_poca_idx()
        logger.info("[WellSeparatedSpheres]: made poca_idx")
        self._collect_on_mfld_k()
        logger.info("[WellSeparatedSpheres]: collect on-mfld k-dim points from both spheres")
        self._x_cn = np.zeros((2, self.n))
        self._x_cn[0, :self.k] = self.S1.specattrs.x_ck
        self._x_cn[1, :self.k] = self.S2.specattrs.x_ck
        

    def find_knn(self, X, use_new=False):
        """
            get k-nearest neighbors of on-manifold points in n-dims
            if `use_new == True`, then use `self.new_knn` for computation
            that is trained on `self.new_poca`
        """
        logger.info("[WellSeparatedSpheres]: use_new == {}".format(use_new))
        if not use_new:
            
            if self.knn is None:
                self.knn = FaissKNeighbors(k=self.nn + self.buf_nn)
                to_fit = self.on_mfld_pts_trivial_
                if self.on_mfld_pts_trivial_ is None:
                    to_fit = np.zeros((self.N, self.n))
                    to_fit[:, self.k] = self.on_mfld_pts_k_
                logger.info("[WellSeparatedSpheres]: fitting knn...")
                self.knn.fit(to_fit)
                logger.info("[WellSeparatedSpheres]: knn fit done")

            logger.info("[WellSeparatedSpheres]: predicting nbhrs...")
            distances, indices = self.knn.predict(X)
            logger.info("[WellSeparatedSpheres]: prediction complete...")

        else:
            if self.new_knn is None:
                logger.info("[WellSeparatedSpheres]: new_knn is None. fitting now...")
                self.new_knn = FaissKNeighbors(k=self.nn + self.buf_nn)
                logger.info("[WellSeparatedSpheres]: fitting new_knn...")
                self.new_knn.fit(self.new_poca_dset[:])
                logger.info("[WellSeparatedSpheres]: new_knn fit done")

            logger.info("[WellSeparatedSpheres]: predicting nbhrs...")
            distances, indices = self.new_knn.predict(X)
            logger.info("[WellSeparatedSpheres]: prediction complete...")

        return distances, indices

    def make_inferred_off_mfld2(self, pp_chunk_size=50000):
        """
        roll in T&N construction, on mfld perturbation, and off mfld 
        perturbation all into one. avoid I/O

        :param pp_chunk_size: chunk size for parallel processing
        """
        if self.nn_distances is None or self.nn_indices is None:
            logger.info("[WellSeparatedSpheres]: knn not computed. computing now ...")
            X = None
            if self.on_mfld_pts_trivial_ is None:
                X = np.zeros((self.N, self.n))
                X[:, self.k] = self.on_mfld_pts_k_
                self.nn_distances, self.nn_indices = self.find_knn(X, use_new=False)
            else:
                self.nn_distances, self.nn_indices = self.find_knn(self.on_mfld_pts_trivial_, use_new=False)
        
        

        if self.class_labels is None:
            self.class_labels = np.zeros(self.N).astype(np.int64)
        if self.all_actual_distances is None:
            self.all_actual_distances = np.zeros((self.N, 2))
        if self.all_points is None:
            self.all_points = np.zeros((self.N, self.n))
        # print("here1:", self.all_points.shape)
        num_offmfld_per_idx = max(self.poca_idx_counts)
        total_num_neg_made = 0
        # print(num_offmfld_per_idx)

        S1_off_mfld_idx = 0
        S2_off_mfld_idx = self.S1.genattrs.N

        for i in tqdm(range(0, int(self.num_pos), pp_chunk_size), desc="computing off mfld (2)"):

            nbhr_indices = self.nn_indices[i:min(self.num_pos, i+pp_chunk_size)]
            mask = np.equal(nbhr_indices, np.arange(i, min(self.num_pos, i+pp_chunk_size)).reshape(-1, 1))
            mask[mask.sum(axis=1) == 0, -1] = True
            nbhr_indices = nbhr_indices[~mask].reshape(nbhr_indices.shape[0], nbhr_indices.shape[1] - 1).astype(np.int64)

            on_mfld_pts = None
            nbhrs = None
            if self.on_mfld_pts_trivial_ is None:
                on_mfld_pts = np.zeros((pp_chunk_size, self.n))
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
            if self.N <= 20000:
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
            
        # print("here2", self.all_points.shape)
        # for on-manifold points
        self.all_actual_distances[self.S1.genattrs.num_neg:self.S1.genattrs.N, 1] = self.M
        self.all_actual_distances[self.S1.genattrs.N+self.S2.genattrs.num_neg:, 0] = self.M
        assert S1_off_mfld_idx == self.S1.genattrs.num_neg
        assert S2_off_mfld_idx - self.S1.genattrs.N == self.S2.genattrs.num_neg
        assert total_num_neg_made == self.num_neg

    def get_tn_for_on_mfld_pts(self, pp_chunk_size=50000):
        """
        make tangents and normals and store them in shard tensors for on-manifold points

        :param pp_chunk_size: chunk size for parallel processing
        """

        if self.nn_distances is None or self.nn_indices is None:
            logger.info("[WellSeparatedSpheres]: knn not computed. computing now ...")
            X = None
            if self.on_mfld_pts_trivial_ is None:
                X = np.zeros((self.N, self.n))
                X[:, self.k] = self.on_mfld_pts_k_
                self.nn_distances, self.nn_indices = self.find_knn(X, use_new=False)
            else:
                self.nn_distances, self.nn_indices = self.find_knn(self.on_mfld_pts_trivial_, use_new=False)


        if os.path.exists(self.tang_dir):
            logger.info("[WellSeparatedSpheres]: tang_dir already exists. removing and recreating...")
            shutil.rmtree(self.tang_dir)
            os.makedirs(self.tang_dir)
            logger.info("[WellSeparatedSpheres]: tang_dir recreated at: {}".format(self.tang_dir))
        if os.path.exists(self.norm_dir):
            logger.info("[WellSeparatedSpheres]: norm_dir already exists. removing and recreating...")
            shutil.rmtree(self.norm_dir)
            os.makedirs(self.norm_dir)
            logger.info("[WellSeparatedSpheres]: norm_dir recreated at: {}".format(self.norm_dir))


        for i in tqdm(range(0, int(self.num_pos), pp_chunk_size), desc="computing T&N"):

            nbhr_indices = self.nn_indices[i:min(self.num_pos, i+pp_chunk_size)]
            mask = np.equal(nbhr_indices, np.arange(i, min(self.num_pos, i+pp_chunk_size)).reshape(-1, 1))
            mask[mask.sum(axis=1) == 0, -1] = True
            nbhr_indices = nbhr_indices[~mask].reshape(nbhr_indices.shape[0], nbhr_indices.shape[1] - 1).astype(np.int64)

            on_mfld_pts = None
            nbhrs = None
            if self.on_mfld_pts_trivial_ is None:
                on_mfld_pts = np.zeros((pp_chunk_size, self.n))
                on_mfld_pts[:, :self.k] = self.on_mfld_pts_k_[i:i+pp_chunk_size]

                nbhrs = np.zeros((pp_chunk_size, nbhr_indices.shape[1], self.n))
                nbhrs[:, :, :self.k] = self.on_mfld_pts_k_[nbhr_indices]
            else:
                on_mfld_pts = self.on_mfld_pts_trivial_[i:i+pp_chunk_size]
                nbhrs = self.on_mfld_pts_trivial_[nbhr_indices]
            # nbhrs - > (50000, 50, 500) on_mfld_pts -> (50000, 500)
            nbhr_local_coords = (nbhrs.transpose(1, 0, 2) - on_mfld_pts).transpose(1, 0, 2)
            # print(nbhr_local_coords.shape)
            _get_tn_for_on_mfld_idx_partial = partial(_get_tn_for_on_mfld_idx, \
                k=self.k,
                tang_dset_per_dir_size=self.tang_dset.per_dir_size, \
                norm_dset_per_dir_size=self.norm_dset.per_dir_size, \
                tang_dir=self.tang_dir, \
                norm_dir=self.norm_dir)
            with multiprocessing.Pool(processes=24) as pool:
                
                pool.starmap(
                    _get_tn_for_on_mfld_idx_partial, 
                    zip(
                        range(i, int(min(i + pp_chunk_size, self.num_pos))),
                        nbhr_local_coords
                        )
                    )

            # for j in range(i, min(i + pp_chunk_size, self.num_pos)):

            #     tmp = results[j - i][0]

            #     cur_tang_dir_idx = results[j - i][0] // self.tang_dset.per_dir_size
            #     cur_tang_dir_name = os.path.join(self.tang_dir, str(cur_tang_dir_idx))
            #     if not os.path.exists(cur_tang_dir_name): os.makedirs(cur_tang_dir_name, exist_ok=True)

            #     cur_norm_dir_idx = results[j - i][0] // self.norm_dset.per_dir_size
            #     cur_norm_dir_name = os.path.join(self.norm_dir, str(cur_norm_dir_idx))
            #     if not os.path.exists(cur_norm_dir_name): os.makedirs(cur_norm_dir_name, exist_ok=True)

            #     # if j % self.tang_dset.per_dir_size == 0 and j != 0:
            #     #     cur_tang_dir_idx += 1
            #     #     cur_tang_dir_name = os.path.join(self.tang_dir, str(cur_tang_dir_idx))
            #     #     os.makedirs(cur_tang_dir_name, exist_ok=True)

            #     # if j % self.norm_dset.per_dir_size == 0 and j != 0:
            #     #     cur_norm_dir_idx += 1
            #     #     cur_norm_dir_name = os.path.join(self.norm_dir, str(cur_norm_dir_idx))
            #     #     os.makedirs(cur_norm_dir_name, exist_ok=True)

            #     tang_fn = os.path.join(cur_tang_dir_name, str(int(tmp)) + ".pth")
            #     torch.save(results[j - i][1], tang_fn)
                
            #     norm_fn = os.path.join(cur_norm_dir_name, str(int(tmp)) + ".pth")
            #     torch.save(results[j - i][2], norm_fn)


        # n = None
        # k = None
        # on_mfld_pts_k_ = None
        # on_mfld_pts_trivial_ = None
        # nn_indices = None
        # nn_distances = None

    
    def make_perturbed_poca(self, pp_chunk_size=5000):
        """
        :param pp_chunk_size: chunk size for parallel processing
        """
        # X = self.on_mfld_pts_trivial_[self.poca_idx] if self.on_mfld_pts_trivial_ is not None else None
        # if self.on_mfld_pts_trivial_ is None:
        #     Z = np.zeros((self.N, self.n))
        #     Z[:, self.k] = self.on_mfld_pts_k_
        #     X = Z[self.poca_idx]

        if self.nn_distances is None or self.nn_indices is None:
            Y = self.on_mfld_pts_trivial_ if self.on_mfld_pts_trivial_ is not None else None
            if Y is None:
                Y = np.zeros((self.N, self.n))
                Y[:, self.k] = self.on_mfld_pts_k_
            logger.info("[WellSeparatedSpheres]: knn not computed. computing now ...")
            self.nn_distances, self.nn_indices = self.find_knn(Y)


        if os.path.exists(self.new_poca_dir):
            logger.info("[WellSeparatedSpheres]: new_poca_dir already exists. removing and recreating...")
            shutil.rmtree(self.new_poca_dir)
            os.makedirs(self.new_poca_dir)
            logger.info("[WellSeparatedSpheres]: recreated new_poca_dir at: {}".format(self.new_poca_dir))

        for i in tqdm(range(0, self.num_neg, pp_chunk_size), desc="computing perturbed poca"):
            poca = self.poca[i:min(self.num_neg, i+pp_chunk_size)]
            poca_idx = self.poca_idx[i:min(self.num_neg, i+pp_chunk_size)]
            max_t_delta = self.max_t_delta
            new_poca_dset_per_dir_size = self.new_poca_dset.per_dir_size
            new_poca_dir = self.new_poca_dir

            _make_perturbed_poca_for_idx_partial = partial(_make_perturbed_poca_for_idx,\
                 max_t_delta=max_t_delta, new_poca_dset_per_dir_size=new_poca_dset_per_dir_size,\
                 new_poca_dir=new_poca_dir, tang_dset=self.tang_dset, norm_dset=self.norm_dset)
            
            with multiprocessing.Pool(processes=24) as pool:
                results = pool.starmap(
                    _make_perturbed_poca_for_idx_partial, 
                    zip(
                        range(i, min(i + pp_chunk_size, self.num_neg)),
                        poca,
                        poca_idx
                    )
                )

            # cur_new_poca_dir_idx = 0
            # cur_new_poca_dir_name = os.path.join(self.new_poca_dir, str(cur_new_poca_dir_idx))
            # os.makedirs(cur_new_poca_dir_name, exist_ok=True)

            end_j = min(i + pp_chunk_size, self.num_neg)
            for j in range(i, end_j):
                # self.new_poca[i] = results[i][1]
                # if j == end_j - 1:
                #     flush_it=True 
                # self.new_poca_dset.append(results[j][1], flush_it)
                tmp = results[j - i][0]

                # cur_new_poca_dir_idx = tmp // self.new_poca_dset.per_dir_size
                # cur_new_poca_dir_name = os.path.join(self.new_poca_dir, str(cur_new_poca_dir_idx))
                # if not os.path.exists(cur_new_poca_dir_name):
                #     os.makedirs(cur_new_poca_dir_name, exist_ok=True)
                
                # if (j % self.new_poca_dset.per_dir_size) == 0 and j != 0:
                #     cur_new_poca_dir_idx += 1
                #     cur_new_poca_dir_name = os.path.join(self.new_poca_dir, str(cur_new_poca_dir_idx))
                    

                # new_poca_fn = os.path.join(cur_new_poca_dir_name, str(j) + ".pth")
                # torch.save(results[j - i][1], new_poca_fn)

                # if self.new_poca is None:
                #     self.new_poca = np.zeros((self.S1.genattrs.num_neg + self.S2.genattrs.num_neg, self.n))
                if self.new_poca_prturb_sizes is None:
                    self.new_poca_prturb_sizes = np.zeros(self.S1.genattrs.num_neg + self.S2.genattrs.num_neg) 
                if self.class_labels is None:
                    self.class_labels = np.zeros(self.S1.genattrs.N + self.S2.genattrs.N, dtype=np.int64)
                
                tmp2 = tmp
                if j >= self.S1.genattrs.num_neg:
                    tmp2 += (self.S1.genattrs.N - self.S1.genattrs.num_neg)

                # self.new_poca[i] = results[i][1] 
                self.new_poca_prturb_sizes[tmp] = results[j - i][2]

                self.class_labels[tmp2] = 2
        


    def make_inferred_off_mfld_eg(self, pp_chunk_size=5000):
        """
        :param pp_chunk_size: chunk size for parallel processing
        """
        
        if self.recomp_tn:
            # TODO: This will not work as slice operator for new_poca_dset does not work
            # will implement this later. until then, recomp_tn is False and we need not enter here
            if self.use_new_knn:
                self.new_nn_distances, self.new_nn_indices = self.find_knn(self.new_poca_dset[:], use_new=True)
            else:
                self.new_nn_distances, self.new_nn_indices = self.find_knn(self.new_poca_dset[:], use_new=False)


        if self.all_points is None:
            self.all_points = np.zeros((self.N, self.n))

        s1_off_mfld_idx = 0
        s2_off_mfld_idx = self.S1.genattrs.N
        for i in tqdm(range(0, self.num_neg, pp_chunk_size), desc="computing off mfld"):
            
            k = self.k
            idx = range(i, min(self.num_neg, i+pp_chunk_size))
            poca_idx = self.poca_idx[i:min(self.num_neg, i+pp_chunk_size)]
            new_nbhr_indices = None
            new_nbhrs = None
            
            if not self.recomp_tn:
                new_nbhrs = [None] * pp_chunk_size
            else:
                tmp = idx if self.use_new_knn else poca_idx[idx]
                new_nbhr_indices = self.new_nn_indices[tmp]
                mask = np.equal(new_nbhr_indices, np.array(tmp).reshape(-1, 1))
                mask[mask.sum(axis=1) == 0, :-1] = True
                new_nbhr_indices = new_nbhr_indices[~mask].reshape(new_nbhr_indices.shape[0], new_nbhr_indices.shape[1] - 1)
                new_nbhrs = np.zeros((self.nn + self.buf_nn, self.n))
                if self.use_new_knn:
                    for nbhr_idx in new_nbhr_indices:
                        for k in range(nbhr_idx.shape[0]):
                            new_nbhrs[k] = self.new_poca
                    else:
                        new_nbhrs[:, self.k] = self.on_mfld_pts_k_[new_nbhr_indices]

            _make_off_mfld_eg_for_idx_partial = partial(
                _make_off_mfld_eg_for_idx,
                norm_dset=self.norm_dset,
                new_poca_dset=self.new_poca_dset,
                recomp_tn=self.recomp_tn,
                max_norm=self.max_norm,
                return_all=False

            )
            with multiprocessing.Pool(processes=24) as pool:
                results = pool.starmap(
                    _make_off_mfld_eg_for_idx_partial, 
                    zip(
                        idx,
                        poca_idx,
                        new_nbhrs
                    ))
        
            end_j = min(i + pp_chunk_size, self.num_neg)
            for j in range(i, end_j):
                tmp = results[j - i][0]
                j_to_poca_idx = self.poca_idx[tmp]
                
                col = 0
                row = None
                if j_to_poca_idx >= (self.S1.genattrs.N - self.S1.genattrs.num_neg):
                    col = 1
                    row = s2_off_mfld_idx
                    s2_off_mfld_idx += 1

                else:
                    row = s1_off_mfld_idx
                    s1_off_mfld_idx += 1
                # print(j, tmp, j_to_poca_idx, row, col)
                self.all_points[row] = results[j - i][1]
                    
                if self.all_actual_distances is None:
                    self.all_actual_distances = np.zeros((self.S1.genattrs.N + self.S2.genattrs.N, 2))

                self.all_actual_distances[row, col] = results[j - i][2]
                self.all_actual_distances[row, 1 - col] = self.M

        self.all_actual_distances[self.S1.genattrs.num_neg:self.S1.genattrs.N, 1] = self.M
        self.all_actual_distances[self.S1.genattrs.N + self.S2.genattrs.num_neg:, 0] = self.M


    def compute_inferred_points(self):

        self._inf_setup()
        logger.info("initial setup complete")
        if not self.avoid_io:
            # print("after setup", self.nn_indices)
            self.get_tn_for_on_mfld_pts(pp_chunk_size=5000)

        # print("after tn", self.nn_indices)
        if not self.online:
            if not self.avoid_io:
                self.make_perturbed_poca(pp_chunk_size=5000)
                # print("after perturbation", self.nn_indices)
                self.make_inferred_off_mfld_eg(pp_chunk_size=5000)
                # print("off mfld eg", self.nn_indices)
            else:
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
                self.all_points_rot_ = np.zeros_like(self.all_points_tr_)
                self.all_points_rot_[:self.S1.genattrs.N] = np.dot(self.rotation[0], self.all_points_tr_[:self.S1.genattrs.N].T).T
                self.all_points_rot_[self.S1.genattrs.N:] = np.dot(self.rotation[1], self.all_points_tr_[self.S1.genattrs.N:].T).T
                self.all_points_trivial_ = self.all_points.copy()
                self.all_points = self.all_points_rot_
            else:
                self.all_points += self.translation
                self.all_points[:self.S1.genattrs.N] = np.dot(self.rotation[0], self.all_points[:self.S1.genattrs.N].T).T
                self.all_points[self.S1.genattrs.N:] = np.dot(self.rotation[1], self.all_points[self.S1.genattrs.N:].T).T
                # self.all_points = np.dot(self.rotation, self.all_points.T).T
            
            for mfld_idx in range(self._x_cn.shape[0]):
                self._x_cn[mfld_idx] += self.translation
                self._x_cn[mfld_idx] = np.dot(self.rotation[mfld_idx], self.x_cn[mfld_idx])
            # print("before reposition x_cn:", self._x_cn)
            self._reposition_spheres()

        else:
            # TODO: implement online sampling of off-manifold points from induced manifold
            pass

        # self.all_distances = torch.from_numpy(self.all_distances).float()
        self.all_points_trivial_ = torch.from_numpy(self.all_points_trivial_).float()
        self.all_points_tr_ = torch.from_numpy(self.all_points_tr_).float()
        self.all_points_rot_ = torch.from_numpy(self.all_points_rot_).float()
        self._x_cn = torch.from_numpy(self._x_cn).float()


    def compute_points(self):

        tot_count_per_mfld = self._N // 2
        neg_count_per_mfld = self._num_neg // 2 if self._num_neg is not None else None

        self._num_neg = 2 * neg_count_per_mfld
        self._num_pos = self._N - self._num_neg

        s_gamma = 0.5 if self._gamma is 0 else self._gamma # gamma = 0 needed for concentric spheres but throws error with constituent spheres
        self.S1 = RandomSphere(N=tot_count_per_mfld, num_neg=neg_count_per_mfld, n=self._n,\
            k=self._k, r=self._r[0], D=self._D, max_norm=self._max_norm, mu=self._mu, sigma=self._sigma,\
            seed=self._seed, x_ck=self._x_ck[0], rotation=self._rotation[0], translation=self._translation,\
            normalize=True, norm_factor=None, gamma=s_gamma, anchor=None, online=self._online, \
            off_online=self._off_online, augment=self._augment, inferred=self._inferred)

        self.S1.compute_points()
        logger.info("[WellSeparatedSpheres]: Generated S1")
        self._x_cn = np.zeros((2, self.n))
        if not self.inferred: 
            self._x_cn[0] = self.S1.specattrs.x_cn
        else:
            self._x_cn[:, :self.k] = self._x_ck

        # `seed` is passed as `None` since we need not have same seed between the two spheres
        self.S2 = RandomSphere(N=tot_count_per_mfld, num_neg=neg_count_per_mfld, n=self._n,\
            k=self._k, r=self._r[1], D=self._D, max_norm=self._max_norm, mu=self._mu, sigma=self._sigma,\
            seed=None, x_ck=self._x_ck[1], rotation=self._rotation[1], translation=self._translation,\
            normalize=True, norm_factor=None, gamma=s_gamma, anchor=None, online=self._online, \
            off_online=self._off_online, augment=self._augment, inferred=self._inferred)

        self.S2.compute_points()
        if not self.inferred:
            self._x_cn[1] = self.S2.specattrs.x_cn
            self._reposition_spheres()  
        # print("before compute x_cn:", self._x_cn)
        if not self.inferred: assert (self._x_cn[1] == self.S2.specattrs.x_cn).all() == True
        logger.info("[WellSeparatedSpheres]: Generated S2")

        if not self.inferred:

            self.all_points = np.zeros((self.N, self.n))
            self.all_points[:self.S1.genattrs.N] = self.S1.genattrs.points_n.numpy()
            self.all_points[self.S1.genattrs.N:] = self.S2.genattrs.points_n.numpy()
            
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

            # pre-image class labels
            # 2: no manifold; 0: S_1; 1: S_2
            if self.N < 1e+7:
                self.pre_class_labels = np.zeros(self.S1.genattrs.N + self.S2.genattrs.N, dtype=np.int64)
                self.pre_class_labels[:self.S1.genattrs.num_neg] = 0
                self.pre_class_labels[self.S1.genattrs.num_neg:self.S1.genattrs.N] = 0
                self.pre_class_labels[self.S1.genattrs.N:self.S1.genattrs.N + self.S2.genattrs.num_neg] = 1
                self.pre_class_labels[self.S1.genattrs.N + self.S2.genattrs.num_neg:] = 1

            # true distances of points in S1 to S2 and vice versa are not available and marked `M`
            self.all_actual_distances = np.zeros((self.S1.genattrs.N + self.S2.genattrs.N, 2))
            self.all_actual_distances[:self.S1.genattrs.N, 0] = self.S1.genattrs.actual_distances.reshape(-1)
            self.all_actual_distances[:self.S1.genattrs.N, 1] = self._M
            self.all_actual_distances[self.S1.genattrs.N:, 1] = self.S2.genattrs.actual_distances.reshape(-1)
            self.all_actual_distances[self.S1.genattrs.N:, 0] = self._M

            # smoothed distances of points
            if self.N < 1e+7:
                self.all_smooth_distances = np.copy(self.all_actual_distances)
                within_buffer_mask = (self.all_actual_distances > self._bp) & (self.all_actual_distances <= self._max_norm)
                self.all_smooth_distances[within_buffer_mask] = self._bp + ((self._M - self._bp) * ((self.all_smooth_distances[within_buffer_mask] - self._bp) / (self._max_norm - self._bp)))
                self.all_smooth_distances[self.all_actual_distances > self._max_norm] = self._M  # this is not really needed

        else:
            self.compute_inferred_points()

          
        # print("x_cn:", self._x_cn)
        self.all_points = torch.from_numpy(self.all_points).float()
        self.all_distances = torch.from_numpy(self.all_distances).float()
        self.all_actual_distances = torch.from_numpy(self.all_actual_distances).float()
        if self.N < 1e+7 and self.all_smooth_distances is not None:
            self.all_smooth_distances = torch.from_numpy(self.all_smooth_distances).float()
        self.class_labels = torch.from_numpy(self.class_labels).long()

        if self.N > 1e+7:
            del self.S1
            del self.S2

        if self._normalize:
            self.norm()
            logger.info("[WellSeparatedSpheres]: Overall noramalization done")

        
        
        # self.get_all_points_k()

    def _reposition_spheres(self):
        """repositions S2 so that the 2 spheres are not extremely far apart"""
        req_c_dist = self.c_dist    
        if self.c_dist is None:
            logger.info("[WellSeparatedSpheres]: no c_dist given. using heuristics...")
            req_c_dist = (sum(self.r) + 2*self.max_norm) * 1.1
            self._c_dist = req_c_dist
            logger.info("[WellSeparatedSpheres]: setting c_dist := {}".format(self.c_dist))
        
        logger.info("[WellSeparatedSpheres]: using c_dist = {}".format(self.c_dist))
        shifting_vec = np.random.normal(self.mu, self.sigma, self.n)
        shifting_vec = self.c_dist * (shifting_vec / np.linalg.norm(shifting_vec, ord=2))
        shifting_vec = shifting_vec + self.x_cn[0]

        if not self.inferred:
            
            logger.info("[WellSeparatedSpheres]: re-positioning S2")
            if self.N <= 1e+7:
                self.old_S2 = copy.deepcopy(self.S2)
            self.S2.genattrs.points_n -= self.S2.specattrs.x_cn
            self.S2.genattrs.points_n += shifting_vec
            self.S2.specattrs.x_cn = shifting_vec

        else:
            """
            if working with inferred manifold then S2 only contains points_k.
            re-positioning is done entirely within the concentric spheres class
            """
            logger.info("[WellSeparatedSpheres]: re-positioning n-dim points of S2")
            if self.N <= 1e+7:
                self.old_S2_data_attrs = {
                    "x_cn": self._x_cn[1],
                    "all_points": self.all_points
                }
            # print(self.all_points.shape)
            self.all_points[self.S1.genattrs.N:] -= self.x_cn[1]
            self.all_points[self.S1.genattrs.N:] += shifting_vec

        self._x_cn[1] = shifting_vec
        # print(self._x_cn)
        logger.info("[WellSeparatedSpheres]: S2 re-positioned")    

    def resample_points(self, seed=42, no_op=False):
        if no_op:
            return None
        if seed is None:
            logger.info("[WellSeparatedSpheres]: No seed provided. proceeding with current seed")
        else:
            logger.info("[WellSeparatedSpheres]: Re-sampling points with seed={}".format(seed))
            seed_everything(seed)
        
        logger.info("[WellSeparatedSpheres]: Starting re-sampling from S1")
        self.S1.resample_points()
        logger.info("[WellSeparatedSpheres]: Re-sampling from S1 done")

        logger.info("[WellSeparatedSpheres]: Starting re-sampling from S2")
        self.S2.resample_points()
        logger.info("[WellSeparatedSpheres]: Re-sampling from S2 done")

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

        # pre-image class labels
        # 2: no manifold; 0: S_1; 1: S_2
        self.pre_class_labels = np.zeros(self.S1.genattrs.N + self.S2.genattrs.N, dtype=np.int64)
        self.pre_class_labels[:self.S1.genattrs.num_neg] = 0
        self.pre_class_labels[self.S1.genattrs.num_neg:self.S1.genattrs.N] = 0
        self.pre_class_labels[self.S1.genattrs.N:self.S1.genattrs.N + self.S2.genattrs.num_neg] = 1
        self.pre_class_labels[self.S1.genattrs.N + self.S2.genattrs.num_neg:] = 1

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
            self.norm(resample=True)
            logger.info("[WellSeparatedSpheres]: Re-sampling noramalization done")

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

    def norm(self, resample=False):
        """normalise points and distances so that the whole setup lies in a unit sphere"""
        if self.norm_factor is None:
            self.norm_factor = max(
                [torch.max(self.all_points[:, i]) - torch.min(self.all_points[:, i]) for i in range(self.n)]
            )
            # min_coord = torch.min(self.all_points).item()
            # max_coord = torch.max(self.all_points).item()
        # print("in norm:", self._x_cn)
        if not resample: 
            self._anchor = np.mean(self._x_cn.numpy() if torch.is_tensor(self._x_cn) else self._x_cn, axis=0) / self.norm_factor
        
        self.normed_all_points = self.all_points / self.norm_factor
        self.normed_all_distances = self.all_distances / self.norm_factor
        self.normed_all_actual_distances = self.all_actual_distances / self.norm_factor
            

        if self.N < 1e+7 and self.all_smooth_distances is not None:
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
        self.normed_all_points -= self.anchor
        self.normed_all_points += self.fix_center
        # self.normed_all_points = self.normed_all_points - self.anchor + self.fix_center

        self.normed_all_points = self.normed_all_points.float()
        self.normed_all_distances = self.normed_all_distances.float()
        self.normed_all_actual_distances = self.normed_all_actual_distances.float()
        if self.N < 1e+7 and self.normed_all_smooth_distances is not None:
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
            if attr in ["S1", "S2"] or "dset" in attr:
                continue
            if attr in attrs:
                if type(attrs[attr]) == dict and "is_data_attr" in attrs[attr]:
                    data_attr = None
                    if os.path.exists(attrs[attr]["path"]):
                        data_attr = torch.load(attrs[attr]["path"])
                        logger.info("[WellSeparatedSpheres]: data attribute ({}) loaded from file: {}".format(attr, attrs[attr]["path"]))
                    else:
                        
                        data_fn = os.path.basename(attrs[attr]["path"])
                        path = os.path.join(dump_dir, data_fn)
                        # logger.info("data_dir: {}".format(data_dir))
                        # logger.info("data_fn: {}".format(data_fn))
                        # logger.info("data_fn_split: {}".format(data_fn_split))
                        data_attr = torch.load(path)
                        logger.info("[WellSeparatedSpheres]: data attribute ({}) loaded from file: {}".format(attr, path))
                    setattr(self, attr, data_attr)
                else:
                    setattr(self, attr, attrs[attr])

        if self.tang_dir is not None:
            self.tang_dset = TensorFileDataset(
                root_dir = self.tang_dir,
                total_len = self.num_pos,
                per_dir_size = 50000
            )
        else: self.tang_dset = None
        
        if self.norm_dir is not None:
            self.norm_dset = TensorFileDataset(
                root_dir = self.norm_dir,
                total_len = self.num_pos,
                per_dir_size = 50000
            )
        else: self.norm_dset = None

        if self.new_poca_dir is not None:
            self.new_poca_dset = TensorFileDataset(
                root_dir = self.new_poca_dir,
                total_len = self.num_neg,
                per_dir_size = 50000
            )
        else: self.new_poca_dset = None

        if os.path.exists(S1_dump):
            self.S1 = RandomSphere()
            self.S1.load_data(S1_dump)

        if os.path.exists(S2_dump):
            self.S2 = RandomSphere()
            self.S2.load_data(S2_dump)

    def _get_save_cache_path(self, save_dir):
        cache_dump = os.path.join(save_dir, "cache")
        return cache_dump
        

    def save_data(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        S1_dir = None
        S2_dir = None
        
        if self.N >= 1e+7:
            ## Saving cache dir was taking a ton of memory (of the order of 800 GBs for N=1e+7)
            logger.info("[WellSeparatedSpheres]: deleting cache before saving...")
            shutil.rmtree(self.cache_dir)
            self._cache_dir = None
            self.tang_dir = None
            self.norm_dir = None
            self.new_poca_dir = None
        else:
            cache_dump_path = self._get_save_cache_path(save_dir)
            if os.path.exists(cache_dump_path):
                shutil.rmtree(cache_dump_path)
            if os.path.exists(self.cache_dir) and (not self.avoid_io):
                shutil.move(self.cache_dir, cache_dump_path)
            self._cache_dir = cache_dump_path
            self.tang_dir = os.path.join(self.cache_dir, "tangents_at_poca")
            self.norm_dir = os.path.join(self.cache_dir, "normals_at_poca")
            self.new_poca_dir = os.path.join(self.cache_dir, "new_poca")

        if self.N < 1e+7:
            S1_dir = os.path.join(save_dir, "S1_dump")
            S2_dir = os.path.join(save_dir, "S2_dump")

        specs_fn = os.path.join(save_dir, "specs.json")
        data_fn = os.path.join(save_dir, "data.pkl")

        specs_attrs = dict()
        data_attrs = dict()

        attr_set = vars(self)
        for attr in attr_set:
            # print("dumping this:", attr)
            if attr in ["S1", "S2"] or "dset" in attr or "knn" in attr:
                # S1 and S2 saved separately so need not be handled. dsets like
                # tang_dset and norm_dset can be constructed at loading. need not be
                # dumped
                continue
            if (type(attr_set[attr]) == str) or not isinstance(attr_set[attr], Iterable):
                # print(attr, attr_set[attr])
                specs_attrs[attr] = attr_set[attr]                
            else:
                attr_fn = os.path.join(save_dir, attr + ".pkl")
                torch.save(attr_set[attr], attr_fn)
                logger.info("[WellSeparatedSpheres]: data attribute ({}) saved to: {}".format(attr, attr_fn))
                data_attrs[attr] = {"is_data_attr": True, "path": attr_fn}

        with open(specs_fn, "w+") as f:
            json.dump(specs_attrs, f)

        torch.save(data_attrs, data_fn)
        
        if self.N < 1e+7:
            self.S1.save_data(S1_dir)
            self.S2.save_data(S2_dir)

    @classmethod
    def get_demo_cfg_dict(cls, N=2500000, n=500, k=2):

        train_cfg_dict = {
            "N": N,
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
            "gamma": 0,
            "norm_factor": 1.0,
            "cache_dir": "/data/data_cache"
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

        logger.info("[WellSeparatedSpheres]: starting with split generation")
        if cfg_dict is None:
            cfg_dict = cls.get_demo_cfg_dict()

        logger.info("[WellSeparatedSpheres]: generating train set...")
        train_cfg = cfg_dict["train"]
        train_set = cls(**train_cfg)
        train_set.compute_points()
        logger.info("[WellSeparatedSpheres]: train set generation done!")

        logger.info("[WellSeparatedSpheres]: generating val set...")
        val_cfg = cfg_dict["val"]
        val_cfg["rotation"] = train_set.rotation
        val_cfg["translation"] = train_set.translation
        val_cfg["x_ck"] = train_set.x_ck
        val_cfg["norm_factor"] = train_set.norm_factor
        val_set = cls(**val_cfg)
        val_set.compute_points()
        logger.info("[WellSeparatedSpheres]: val set generation done!")

        logger.info("[WellSeparatedSpheres]: generating test set...")
        test_cfg = cfg_dict["test"]
        test_cfg["rotation"] = train_set.rotation
        test_cfg["translation"] = train_set.translation
        test_cfg["x_ck"] = train_set.x_ck
        test_cfg["norm_factor"] = train_set.norm_factor
        test_set = cls(**test_cfg)
        test_set.compute_points()
        logger.info("[WellSeparatedSpheres]: test set generation done!")


        if save_dir is not None:
            logger.info("[WellSeparatedSpheres]: saving splits at: {}".format(save_dir))
            cls.save_splits(train_set, val_set, test_set, save_dir)
        
        logger.info("[WellSeparatedSpheres]: generated splits!")
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

        try:
            train_set.load_data(train_dir)
        except:
            logger.info("[WellSeparatedSpheres]: could not load train split!")
            train_set = None

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
    def num_pos(self):
        return self._num_pos

    @num_pos.setter
    def num_pos(self, x):
        raise RuntimeError("cannot set `num_pos` after instantiation")

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

    @property
    def inferred(self):
        return self._inferred

    @inferred.setter
    def inferred(self, inferred):
        raise RuntimeError("cannot set `inferred` after instantiation!")

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
    def recomp_tn(self):
        return self._recomp_tn

    @recomp_tn.setter
    def recomp_tn(self, recomp_tn):
        raise RuntimeError("cannot set `recomp_tn` after instantiation!")

    @property
    def use_new_knn(self):
        return self._use_new_knn

    @use_new_knn.setter
    def use_new_knn(self):
        raise RuntimeError("cannot set `use_new_knn` after instantiation!")

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, x):
        raise RuntimeError("cannot set `cache_dir` after instantiation!")

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, x):
        raise RuntimeError("cannot set `uuid` after instantiation!")

    @property
    def c_dist(self):
        return self._c_dist

    @c_dist.setter
    def c_dist(self, x):
        raise RuntimeError("cannot set `c_dist` after instantiation!")

    @property
    def poca(self, idx=None):
        if idx is None:
            x = np.zeros((self.num_neg, self.n))
            x[:, :self.k] = self.on_mfld_pts_k_[self.poca_idx]
            return x

        elif isinstance(idx, Iterable):
            x = np.zeros((len(idx), self.n))
            x[:, :self.k] = self.on_mfld_pts_k_[self.poca_idx[idx]]
            return x

        else:
            x = np.zeros(self.n)
            x[:self.k] = self.on_mfld_pts_k_[self.poca_idx[idx]]
            return x

    # @property
    # def on_mfld_pts_trivial_(self, idx=None):
    #     if idx is None:
    #         x = np.zeros((self.on_mfld_pts_k_.shape[0], self.n))
    #         x[:, :self.k] = self.on_mfld_pts_k_
    #         return x

    #     elif isinstance(idx, Iterable):
    #         x = np.zeros((len(idx), self.n))
    #         x[:, :self.k] = self.on_mfld_pts_k_[idx]
    #         return x

    #     else:
    #         x = np.zeros(self.n)
    #         x[:self.k] = self.on_mfld_pts_k_[idx]
    #         return x




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




def _make_perturbed_poca_for_idx(idx, poca, poca_idx, max_t_delta,\
     tang_dset, norm_dset, new_poca_dset_per_dir_size, new_poca_dir,\
     return_all=False):

    on_mfld_pt = poca

    tangential_dirs = tang_dset[poca_idx]
    normal_dirs = norm_dset[poca_idx]
    
    rdm_coeffs = np.random.normal(0, 1, size=tangential_dirs.shape[0])
    delta = np.sum(rdm_coeffs.reshape(-1, 1) * tangential_dirs, axis=0)

    prturb_size = np.random.uniform(1e-8, max_t_delta)
    delta = (prturb_size / np.linalg.norm(delta, ord=2)) * delta

    prturb_poca = on_mfld_pt + delta

    # prturb_poca = np.ones(500)
    # prturb_size = 1
    # tangential_dirs = None
    # normal_dirs = None

    cur_new_poca_dir_idx = idx // new_poca_dset_per_dir_size
    cur_new_poca_dir_name = os.path.join(new_poca_dir, str(cur_new_poca_dir_idx))
    os.makedirs(cur_new_poca_dir_name, exist_ok=True)
    new_poca_fn = os.path.join(cur_new_poca_dir_name, str(idx) + ".pth")
    torch.save(prturb_poca, new_poca_fn)

    if return_all:
        return (
            idx,
            prturb_poca,
            prturb_size, 
            tangential_dirs,
            normal_dirs
        )
    return (idx, prturb_poca, prturb_size)


def _make_off_mfld_eg_for_idx(idx, poca_idx, new_nbhrs, new_poca_dset,\
     norm_dset, recomp_tn, max_norm, return_all=False):
        
    on_mfld_pt = new_poca_dset[idx]

    # tangential_dirs = None
    normal_dirs = None

    if recomp_tn:
        """
        TODO: new_nn_indices and new_nn_distances are computed by passing
        new_poca_dset as a whole to the self.new_knn. This needs to be batched.

        will implement this later. until then recomp_tn is False, and we need
        not enter here

        """

        # since nn_indices and nn_dists are computed over on_mfld_pts_trivial_
        # therefore we search for indices and dists using poca_idx[idx]
        # tmp = idx if use_new_knn else poca_idx[idx]
        # new_nbhr_indices = new_nn_indices[tmp]

        # if idx < num_pos and poca_idx[idx] in new_nbhr_indices:
        #     new_nbhr_indices = new_nbhr_indices[new_nbhr_indices != poca_idx[idx]]
        # else:
        #     new_nbhr_indices = new_nbhr_indices[:-1]

        # new_nbhrs = np.zeros((nn + buf_nn, n))

        # if use_new_knn:
        #     for nbhr_idx in new_nbhr_indices:
        #         new_nbhrs[nbhr_idx] = new_poca_dset[nbhr_idx]
        # else:
        #     new_nbhrs[:, :k] = on_mfld_pts_k_[new_nbhr_indices]

        # new_nbhr_local_coords = new_nbhrs - on_mfld_pt

        # pca = PCA(n_components=k - 1)
        # pca.fit(new_nbhr_local_coords)

        # tangential_dirs = pca.components_
        # normal_dirs = spla.null_space(tangential_dirs).T

        # tangential_dirs += on_mfld_pt
        # normal_dirs += on_mfld_pt

    else:

        # tangential_dirs = tang_dset[poca_idx[idx]]
        # normal_dirs = norm_dset[poca_idx[idx]]
        
        normal_dirs = norm_dset[poca_idx]
        # normal_dirs = None

    rdm_coeffs = np.random.normal(0, 1, size=normal_dirs.shape[0])
    off_mfld_pt = np.sum(rdm_coeffs.reshape(-1, 1) * normal_dirs, axis=0)
    rdm_norm = np.random.uniform(0, max_norm)
    off_mfld_pt = off_mfld_pt * (rdm_norm / np.linalg.norm(off_mfld_pt))
    off_mfld_pt += on_mfld_pt

    # off_mfld_pt = np.ones(2)
    # rdm_norm = 1


    if return_all:
        return (
            idx, 
            off_mfld_pt,
            rdm_norm,
            normal_dirs
        )
    return (idx, off_mfld_pt, rdm_norm)


def _make_off_mfld_eg_for_idx2(idx, on_mfld_pt, nbhr_local_coords, poca_idx_count, k, max_t_delta, max_norm, return_all=False):
    """
    1. make tangents and normals
    2. do tangential perturbation
    3. do normal perturbation
    4. return final off mfld examples


    :param idx: index of on-manifold point
    :param poca_idx_count: number of times such that self.poca_idx == idx
    :param nbhr_local_coords: local co-ordinates of the neighbors
    :param k: intrinsic dimensions
    :param poca_idx: indices of poca_idx that are equal to idx
    """

    pca = PCA(n_components=k-1) # manifold is (k-1) dim so tangent space should be same
    pca.fit(nbhr_local_coords)

    tangential_dirs = pca.components_
    normal_dirs = spla.null_space(tangential_dirs).T

    on_mfld_pb_coeffs = np.random.normal(size=(poca_idx_count, tangential_dirs.shape[0]))
    on_mfld_pb = np.dot(on_mfld_pb_coeffs, tangential_dirs)
    on_mfld_pb_sizes = np.random.uniform(0, max_t_delta, size=(on_mfld_pb.shape[0]))
    on_mfld_pb = on_mfld_pb * (on_mfld_pb_sizes / np.linalg.norm(on_mfld_pb, axis=0, ord=2)).reshape(-1, 1)

    new_poca_for_idx = on_mfld_pt + on_mfld_pb
    assert new_poca_for_idx.shape[0] == poca_idx_count.shape[0]

    off_mfld_pb_coeffs = random.normal(size=(new_poca_for_idx.shape[0], normal_dirs.shape[0]))
    off_mfld_pb = np.dot(off_mfld_pb_coeffs, normal_dirs)
    off_mfld_pb_sizes = np.random.uniform(0, max_norm, size=(off_mfld_pb.shape[0]))
    off_mfld_pb = off_mfld_pb * (off_mfld_pb_sizes / np.linalg.norm(off_mfld_pb, axis=0, ord=2)).reshape(-1, 1)

    off_mfld_pts_for_idx = new_poca_for_idx + off_mfld_pb

    if return_all:
        return (
            idx,
            on_mfld_pt, 
            nbhr_local_coords,
            k,
            poca_idx_count,
            tangential_dirs, 
            normal_dirs,
            new_poca_for_idx,
            off_mfld_pts_for_idx
        )

    return (
        idx,
        off_mfld_pts_for_idx
    )



