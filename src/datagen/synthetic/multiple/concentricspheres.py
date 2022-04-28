import multiprocessing
import os
import re
import sys
import json
import copy
import uuid
import time
import random
import inspect
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
from datagen_utils import *

logger = init_logger(__name__)

class ConcentricSpheres(Dataset):

    def __init__(self, N=1000, num_neg=None, n=100, k=2, D=2.0, max_norm=5.0, bp=1.8, M=50, mu=10,\
                sigma=5, seed=42, r=10.0, g=10.0, x_ck=None, rotation=None, translation=None,\
                normalize=True, norm_factor=None, gamma=0.5, anchor=None, online=False,\
                off_online=False, augment=False, inferred=False, nn=None, buffer_nbhrs=2,\
                max_t_delta=1e-3, recomp_tn=False, use_new_knn=False, cache_dir="/tmp", **kwargs):
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
        :param inferred: if True, then off-manifold points are generated from the inferred manifold
        :type inferred: bool
        :param nn: number of points to use for k_neighbors (default is `k-1`), when `inferred == True`
        :type nn: int
        :param buffer_nbhrs: buffer neighbors for reasons of PCA, when `inferred == True`
        :type buffer_nbhrs: int
        :param max_t_delta: maximum perturbation allowed in the tangential direction, when `inferred == True`
        :type max_t_delta: float
        :param recomp_tn: if True, then recompute tangents and normals for perturbed on-manifold points, when `inferred == True`
        :type recomp_tn: bool
        :param use_new_knn: if True, use .new_knn for tangent and normal recomputation, when `inferred == True`
        :type use_new_knn: bool
        :param cache_dir: directory to cache auxillary attributes in order to free RAM
        :type cache_dir: str
        """

        if seed is not None: seed_everything(seed)

        self._N = N
        self._num_neg = num_neg
        self._num_pos = None if num_neg is None else N - num_neg
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
        self._inferred = inferred
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

        self._cache_dir = cache_dir
        self._uuid = str(uuid.uuid4())
        os.makedirs(os.path.join(self.cache_dir, self.uuid), exist_ok=True)

        self.S1 = None
        self.S2 = None

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
        self.new_poca = None # points of closest approach after applying tangential perturbations
        self.new_poca_prturb_sizes = None # perturbation size of tangential perturbation

        self.on_mfld_pts_k_ = None # all on-manifold points
        self.on_mfld_pts_trivial_ = None # trivial embeddings of on-manifold points

        self._recomp_tn = recomp_tn
        self._use_new_knn = use_new_knn

        self.tn_dir = None
        self.new_poca_dir = None
        self.tn_dset = None
        self.new_poca_dset = None

        if not self.online:
            self.tang_dir = os.path.join(self.cache_dir, self.uuid, "tangents_at_poca") # cache directory to dump tangents and normals
            self.norm_dir = os.path.join(self.cache_dir, self.uuid, "normals_at_poca") # cache directory to dump tangents and normals

            self.new_poca_dir = os.path.join(self.cache_dir, self.uuid, "new_poca") # cache directory to store new poca

            self.tang_dset = TensorShardsDataset(
                root_dir=self.tang_dir,
                data_shape=(self.k-1, self.n),
                data=None,
                chunk_size=50000
            )

            self.norm_dset = TensorShardsDataset(
                root_dir=self.tang_dir,
                data_shape=(self.n - self.k + 1, self.n),
                data=None,
                chunk_size=50000
            )

            self.new_poca_dset = TensorShardsDataset(
                root_dir=self.new_poca_dir,
                data_shape=(self.n,),
                data=None,
                chunk_size=50000
            )

    def _make_poca_idx(self):
        self.poca_idx =  np.zeros(self.num_neg, dtype=np.int64)
        self.poca_idx[:self.num_pos] = np.arange(min(self.num_pos, self.num_neg), dtype=np.int64)
        self.poca_idx[self.num_pos:] = np.random.choice(np.arange(self.num_pos, dtype=np.int64), size=max(0, self.num_neg - self.num_pos), replace=True).astype(np.int64) 

    def _collect_on_mfld_k(self):
        self.on_mfld_pts_k_ = np.zeros((self.N, self.k))
        num_on_mfld_S1 = self.S1.genattrs.N - self.S1.genattrs.num_neg
        self.on_mfld_pts_k_[:num_on_mfld_S1] = self.S1.genattrs.points_k
        self.on_mfld_pts_k_[num_on_mfld_S1:] = self.S2.genattrs.points_k
        if self.N < 1e+7:
            self.on_mfld_pts_trivial_ = np.zeros((self.n, self.k))
            self.on_mfld_pts_trivial_[:, self.k] = self.on_mfld_pts_k_

    def _inf_setup(self):
        """setting up data for off manifold samples when computing inferred manifold"""
        self._make_poca_idx()
        self._collect_on_mfld_k()
        

    def find_knn(self, X, use_new=False):
        """
            get k-nearest neighbors of on-manifold points in n-dims
            if `use_new == True`, then use `self.new_knn` for computation
            that is trained on `self.new_poca`
        """
        if not use_new:
            if self.knn is None:
                self.knn = FaissKNeighbors(k=self.nn + self.buf_nn)
                to_fit = self.on_mfld_pts_trivial_
                if self.on_mfld_pts_trivial_ is None:
                    to_fit = np.zeros((self.N, self.n))
                    to_fit[:, self.k] = self.on_mfld_pts_k_
                self.knn.fit(to_fit)
            distances, indices = self.knn.predict(X)
        else:
            if self.new_knn is None:
                self.new_knn = FaissKNeighbors(k=self.nn + self.buf_nn)
                self.new_knn.fit(self.new_poca_dset[:])
            distances, indices = self.new_knn.predict(X)

        return distances, indices

    def _get_tn_for_on_mfld_idx(self, idx):
        """compute tangential and normal directions for all on-manifold points"""
        if self.nn_indices is None or self.nn_distances is None:
            raise RuntimeError("knn's not predicted yet!")

        on_mfld_pt = None
        if self.on_mfld_pts_trivial_ is None:
            on_mfld_pt = np.zeros(self.n)
            on_mfld_pt[:, self.k] = self.on_mfld_pts_k_[idx]
        else:
            on_mfld_pt = self.on_mfld_pts_trivial_[idx]
        nbhr_indices = self.nn_indices[idx]
        nbhr_dists = self.nn_distances[idx]

        if idx in nbhr_indices:
            nbhr_indices = nbhr_indices[nbhr_indices != idx]
        else:
            nbhr_indices = nbhr_indices[:-1]

        nbhrs = None
        if self.on_mfld_pts_trivial_ is None:
            nbhrs = np.zeros((nbhr_indices.shape[0], self.n))
            nbhrs[:, self.k] = self.on_mfld_pts_k[nbhr_indices]
        else:
            nbhrs = self.on_mfld_pts_trivial_[nbhr_indices]
        
        nbhr_local_coords = nbhrs - on_mfld_pt

        pca = PCA(n_components=self.k-1) # manifold is (k-1) dim so tangent space should be same
        pca.fit(nbhr_local_coords)

        tangential_dirs = pca.components_
        normal_dirs = spla.null_space(tangential_dirs).T

        tangential_dirs += on_mfld_pt
        normal_dirs += on_mfld_pt

        return (idx, tangential_dirs, normal_dirs)

    def get_tn_for_on_mfld_pts(self, pp_chunk_size=50000):
        """
        make tangents and normals and store them in shard tensors for on-manifold points

        :param pp_chunk_size: chunk size for parallel processing
        """

        if self.nn_distances is None or self.nn_indices is None:
            X = self.on_mfld_pts_trivial_
            if self.on_mfld_pts_trivial_ is None:
                X = np.zeros((self.N, self.n))
                X[:, self.k] = self.on_mfld_pts_k_
            self.nn_distances, self.nn_indices = self.find_knn(X, use_new=False)
        
        for i in tqdm(range(0, self.num_pos, pp_chunk_size)):
            with multiprocessing.Pool(processes=24) as pool:
                results = pool.map(self._get_tn_for_on_mfld_idx, range(i, min(i + pp_chunk_size, self.num_pos)))

            flush_it = False
            for j in range(i, min(i + pp_chunk_size, self.num_pos)):
                if j == min(i + pp_chunk_size, self.num_pos) - 1:
                    flush_it = True
                self.tang_dset.append(results[j][1], flush_it)
                self.norm_dset.append(results[j][2], flush_it)

    def _make_perturbed_poca_for_idx(self, idx, return_all=False):

        prturb_size = 0

        if idx < self.num_pos:
            # one copy of poca should be unperturbed
            return (idx, self.poca[idx], prturb_size)

        on_mfld_pt = self.poca[idx]

        nbhr_indices = None
        nbhr_dists = None
        if self.nn_indices is None or self.nn_distances is None:
            nbhr_dists, nbhr_indices = self.knn.predict(self.poca[idx].reshape(-1, 1))
            nbhr_dists = nbhr_dists[0]
            nbhr_indices = nbhr_indices[0]
        else:
            # since .nn_indices and .nn_dists are computed over on_mfld_pts_trivial_
            # therefore we search for indices and dists using poca_idx[idx]
            nbhr_indices = self.nn_indices[self.poca_idx[idx]]
            nbhr_dists = self.nn_distances[self.poca_idx[idx]]

        if self.poca_idx[idx] in nbhr_indices:
            nbhr_indices = nbhr_indices[nbhr_indices != self.poca_idx[idx]]
        else:
            nbhr_indices = nbhr_indices[:-1]

        nbhrs = np.zeros((nbhr_indices.shape[0], self.n))
        nbhrs[:, self.k] = self.on_mfld_pts_k[nbhr_indices]
        nbhr_local_coords = nbhrs - on_mfld_pt
        

        # pca = PCA(n_components=self.k-1) # manifold is (k-1) dim so tangent space should be same
        # pca.fit(nbhr_local_coords)

        # tangential_dirs = pca.components_
        # normal_dirs = spla.null_space(tangential_dirs).T

        # tangential_dirs += on_mfld_pt
        # normal_dirs += on_mfld_pt

        tangential_dirs = self.tang_dset[self.poca_idx[idx]]
        normal_dirs = self.norm_dset[self.poca_idx[idx]]

        rdm_coeffs = np.random.normal(0, 1, size=tangential_dirs.shape[0])
        delta = np.sum(rdm_coeffs.reshape(-1, 1) * tangential_dirs, axis=0)

        prturb_size = np.random.uniform(1e-8, self.max_t_delta)
        delta = (prturb_size / np.linalg.norm(delta, ord=2)) * delta

        prturb_poca = on_mfld_pt + delta
        if return_all:
            return (
                idx,
                prturb_poca,
                prturb_size, 
                tangential_dirs,
                normal_dirs
            )
        return (idx, prturb_poca, prturb_size)

    def make_perturbed_poca(self, pp_chunk_size=50000):
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
            self.nn_distances, self.nn_indices = self.find_knn(Y)

        

        for i in tqdm(range(0, self.num_neg, pp_chunk_size)):
            with multiprocessing.Pool(processes=24) as pool:
                results = pool.map(self._make_perturbed_poca_for_idx, range(i, min(i + pp_chunk_size, self.num_neg)))
        
            flush_it=False
            end_j = min(i + pp_chunk_size, self.num_neg)
            for j in range(i, end_j):
                # self.new_poca[i] = results[i][1]
                if j == end_j - 1:
                    flush_it=True 
                self.new_poca_dset.append(results[j][1], flush_it)

                # if self.new_poca is None:
                #     self.new_poca = np.zeros((self.S1.genattrs.num_neg + self.S2.genattrs.num_neg, self.n))
                if self.new_poca_prturb_sizes is None:
                    self.new_poca_prturb_sizes = np.zeros(self.S1.genattrs.num_neg + self.S2.genattrs.num_neg) 
                if self.class_labels is None:
                    self.class_labels = np.zeros(self.S1.genattrs.N + self.S2.genattrs.N, dtype=np.int64)
                
                tmp = j
                if j > self.S1.genattrs.num_neg:
                    tmp += (self.S1.genattrs.N - self.S1.genattrs.num_neg)

                # self.new_poca[i] = results[i][1] 
                self.new_poca_prturb_sizes[j] = results[j][2]

                self.class_labels[tmp] = 2

    def _make_off_mfld_eg_for_idx(self, idx, return_all=False):
        
        on_mfld_pt = self.new_poca_dset[idx]

        tangential_dirs = None
        normal_dirs = None

        if self.recomp_tn:

            new_nbhr_indices = None
            new_nbhr_dists = None

            if self.new_nn_distances is None or self.new_nn_indices is None:
                if self.use_new_knn:
                    self.new_nn_distances, self.new_nn_indices = self.find_knn(self.new_poca_dset[:], use_new=True)
                else:
                    self.new_nn_distances, self.new_nn_indices = self.find_knn(self.new_poca_dset[:], use_new=False)

            if self.new_nn_indices is None or self.new_nn_distances is None:
                new_nbhr_dists, new_nbhr_indices = self.knn.predict(self.poca[idx].reshape(-1, 1))
                new_nbhr_dists = new_nbhr_dists[0]
                new_nbhr_indices = new_nbhr_indices[0]
            else:
                # since nn_indices and nn_dists are computed over on_mfld_pts_trivial_
                # therefore we search for indices and dists using poca_idx[idx]
                tmp = idx if self.use_new_knn else self.poca_idx[idx]
                new_nbhr_indices = self.new_nn_indices[tmp]
                new_nbhr_dists = self.new_nn_distances[tmp]

            if idx < self.num_pos and self.poca_idx[idx] in new_nbhr_indices:
                new_nbhr_indices = new_nbhr_indices[new_nbhr_indices != self.poca_idx[idx]]
            else:
                new_nbhr_indices = new_nbhr_indices[:-1]

            new_nbhrs = np.zeros((self.nn + self.buf_nn, self.n))

            if self.use_new_knn:
                for nbhr_idx in new_nbhr_indices:
                    new_nbhrs[nbhr_idx] = self.new_poca_dset[nbhr_idx]
            else:
                new_nbhrs[:, :self.k] = self.on_mfld_pts_k_[new_nbhr_indices]

            new_nbhr_local_coords = new_nbhrs - on_mfld_pt

            pca = PCA(n_components=self.k - 1)
            pca.fit(new_nbhr_local_coords)

            tangential_dirs = pca.components_
            normal_dirs = spla.null_space(tangential_dirs).T

            tangential_dirs += on_mfld_pt
            normal_dirs += on_mfld_pt

        else:

            tangential_dirs = self.tang_dset[self.poca_idx[idx]]
            normal_dirs = self.norm_dset[self.poca_idx[idx]]


        rdm_coeffs = np.random.normal(0, 1, size=normal_dirs.shape[0])
        off_mfld_pt = np.sum(rdm_coeffs.reshape(-1, 1) * normal_dirs, axis=0)
        rdm_norm = np.random.uniform(0, self.max_norm)
        off_mfld_pt = off_mfld_pt * (rdm_norm / np.linalg.norm(off_mfld_pt))
        off_mfld_pt += on_mfld_pt


        if return_all:
            return (
                idx, 
                off_mfld_pt,
                rdm_norm,
                tangential_dirs,
                normal_dirs
            )
        return (idx, off_mfld_pt, rdm_norm)

    def make_inferred_off_mfld_eg(self, pp_chunk_size=50000):
        """
        :param pp_chunk_size: chunk size for parallel processing
        """
        if self.new_nn_distances is None or self.new_nn_indices is None:
            self.new_nn_distances, self.new_nn_indices = self.find_knn(self.new_poca_dset[:])

        for i in tqdm(range(0, self.num_neg, pp_chunk_size)):
            with multiprocessing.Pool(processes=24) as pool:
                results = pool.map(self._make_off_mfld_eg_for_idx, range(i, min(i + pp_chunk_size, self.num_neg)))
        
            end_j = min(i + pp_chunk_size, self.num_neg)
            for j in range(i, end_j):
                self.all_points[j] = results[j][1]
                
                if self.all_actual_distances is None:
                    self.all_actual_distances = np.zeros(self.S1.genattrs.N + self.S2.genattrs.N) 
                
                tmp = j
                if j > self.S1.genattrs.num_neg:
                    tmp += (self.S1.genattrs.N - self.S1.genattrs.num_neg)

                # self.new_poca[i] = results[i][1] 
                self.all_actual_distances[tmp] = results[j][2]

    def compute_inferred_points(self):

        self._inf_setup()
        self.get_tn_for_on_mfld_pts(pp_chunk_size=50000)
        if not self.online:
            self.make_perturbed_poca(pp_chunk_size=50000)
            self.make_inferred_off_mfld_eg(pp_chunk_size=50000)

            num_on_mfld_S1 = self.S1.genattrs.N - self.S1.genattrs.num_neg
            self.all_points[self.S1.genattrs.num_neg:self.S1.genattrs.N, :self.k] = self.on_mfld_pts_k_[:num_on_mfld_S1]
            self.all_points[self.S1.genattrs.N+self.S2.genattrs.num_neg:, :self.k] = self.on_mfld_pts_k_[num_on_mfld_S1:]

            self.class_labels[self.S1.genattrs.num_neg:self.S1.genattrs.N] = 0
            self.class_labels[self.S1.genattrs.N+self.S2.genattrs.num_neg:] = 1

            assert (self.all_actual_distances[self.S1.genattrs.num_neg:self.S1.genattrs.N] == 0).all()
            assert (self.all_actual_distances[self.S1.genattrs.N+self.S2.genattrs.num_neg:] == 0).all()
            assert (self.class_labels[:self.S1.genattrs.num_neg] == 2).all()
            assert (self.class_labels[self.S1.genattrs.N:self.S1.genattrs.N + self.S2.genattrs.num_neg] == 2).all()

            if self.distances is None:
                self.all_distances = np.zeros(self.S1.genattrs.N + self.S2.genattrs.N)
                self.all_distances[:] = self.all_actual_distances[:]
                self.all_distances[self.all_distances >= self.D] = self.D

            self.all_points += self.translation
            self.all_points = np.dot(self.rotation, self.all_points)

        else:
            # TODO: implement online sampling of off-manifold points from induced manifold
            pass

    def compute_points(self):

        tot_count_per_mfld = self._N // 2
        neg_count_per_mfld = self._num_neg // 2 if self._num_neg is not None else None

        self._num_neg = 2 * neg_count_per_mfld
        self._num_pos = self._N - self._num_neg

        s_gamma = 0.5 if self._gamma is 0 else self._gamma # gamma = 0 needed for concentric spheres but throws error with constituent spheres
        self.S1 = RandomSphere(N=tot_count_per_mfld, num_neg=neg_count_per_mfld, n=self._n,\
            k=self._k, r=self._r, D=self._D, max_norm=self._max_norm, mu=self._mu, sigma=self._sigma,\
            seed=self._seed, x_ck=self._x_ck, rotation=self._rotation, translation=self._translation,\
            normalize=True, norm_factor=None, gamma=s_gamma, anchor=None, online=self._online, \
            off_online=self._off_online, augment=self._augment, inferred=self._inferred)

        self.S1.compute_points()
        logger.info("[ConcentricSpheres]: Generated S1")
        self._x_cn = self.S1.specattrs.x_cn

        # `seed` is passed as `None` since we need not have same seed between the two spheres
        self.S2 = RandomSphere(N=tot_count_per_mfld, num_neg=neg_count_per_mfld, n=self._n,\
            k=self._k, r=self._r + self._g, D=self._D, max_norm=self._max_norm, mu=self._mu, sigma=self._sigma,\
            seed=None, x_ck=self._x_ck, rotation=self._rotation, translation=self._translation,\
            normalize=True, norm_factor=None, gamma=s_gamma, anchor=None, online=self._online, \
            off_online=self._off_online, augment=self._augment, inferred=self._inferred)

        self.S2.compute_points()
        assert (self._x_cn == self.S2.specattrs.x_cn).all() == True
        logger.info("[ConcentricSpheres]: Generated S2")

        if not self.inferred:
            self.all_points = np.zeros((self.N, self.n))
            self.all_points[:self.S1.genattrs.n] = self.S1.genattrs.points_n.numpy()
            self.all_points[self.S1.genattrs.n:] = self.S2.genattrs.points_n.numpy()
            
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

        self.all_points = torch.from_numpy(self.all_points).float()
        self.all_distances = torch.from_numpy(self.all_distances).float()
        self.all_actual_distances = torch.from_numpy(self.all_actual_distances).float()
        if self.N < 1e+7:
            self.all_smooth_distances = torch.from_numpy(self.all_smooth_distances).float()
        self.class_labels = torch.from_numpy(self.class_labels).long()

        if self.N > 1e+7:
            del self.S1
            del self.S2

        if self._normalize:
            self.norm()
            logger.info("[ConcentricSpheres]: Overall noramalization done")

        
        
        # self.get_all_points_k()

    def resample_points(self, seed=42, no_op=False):
        if no_op:
            return None
        if seed is None:
            logger.info("[ConcentricSpheres]: No seed provided. proceeding with current seed")
        else:
            logger.info("[ConcentricSpheres]: Re-sampling points with seed={}".format(seed))
            seed_everything(seed)
        
        logger.info("[ConcentricSpheres]: Starting re-sampling from S1")
        self.S1.resample_points()
        logger.info("[ConcentricSpheres]: Re-sampling from S1 done")

        logger.info("[ConcentricSpheres]: Starting re-sampling from S2")
        self.S2.resample_points()
        logger.info("[ConcentricSpheres]: Re-sampling from S2 done")

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
            logger.info("[ConcentricSpheres]: Re-sampling noramalization done")

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
        
        if not resample: self._anchor = self._x_cn / self.norm_factor
        
        self.normed_all_points = self.all_points / self.norm_factor
        self.normed_all_distances = self.all_distances / self.norm_factor
        self.normed_all_actual_distances = self.all_actual_distances / self.norm_factor
            

        if self.N < 1e+7:
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
        if self.N < 1e+7:
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
                if type(attrs[attr]) == dict and "is_data_attr" in attrs[attr]:
                    data_attr = torch.load(attrs[attr]["path"])
                    logger.info("[ConcentricSpheres]: data attribute ({}) loaded from file: {}".format(attr, attrs[attr]["path"]))
                    setattr(self, attr, data_attr)
                else:
                    setattr(self, attr, attrs[attr])

        if os.path.exists(S1_dump):
            self.S1 = RandomSphere()
            self.S1.load_data(S1_dump)

        if os.path.exists(S2_dump):
            self.S2 = RandomSphere()
            self.S2.load_data((S2_dump))


    def save_data(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        S1_dir = None
        S2_dir = None
        if self.N < 1e+7:
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
                attr_fn = os.path.join(save_dir, attr + ".pkl")
                torch.save(attr_set[attr], attr_fn)
                logger.info("[ConcentricSpheres]: data attribute ({}) saved to: {}".format(attr, attr_fn))
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

        logger.info("[ConcentricSpheres]: starting with split generation")
        if cfg_dict is None:
            cfg_dict = cls.get_demo_cfg_dict()

        logger.info("[ConcentricSpheres]: generating train set...")
        train_cfg = cfg_dict["train"]
        train_set = cls(**train_cfg)
        train_set.compute_points()
        logger.info("[ConcentricSpheres]: train set generation done!")

        logger.info("[ConcentricSpheres]: generating val set...")
        val_cfg = cfg_dict["val"]
        val_cfg["rotation"] = train_set.rotation
        val_cfg["translation"] = train_set.translation
        val_cfg["x_ck"] = train_set.x_ck
        val_cfg["norm_factor"] = train_set.norm_factor
        val_set = cls(**val_cfg)
        val_set.compute_points()
        logger.info("[ConcentricSpheres]: val set generation done!")

        logger.info("[ConcentricSpheres]: generating test set...")
        test_cfg = cfg_dict["test"]
        test_cfg["rotation"] = train_set.rotation
        test_cfg["translation"] = train_set.translation
        test_cfg["x_ck"] = train_set.x_ck
        test_cfg["norm_factor"] = train_set.norm_factor
        test_set = cls(**test_cfg)
        test_set.compute_points()
        logger.info("[ConcentricSpheres]: test set generation done!")


        if save_dir is not None:
            logger.info("[ConcentricSpheres]: saving splits at: {}".format(save_dir))
            cls.save_splits(train_set, val_set, test_set, save_dir)
        
        logger.info("[ConcentricSpheres]: generated splits!")
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
        return self.cache_dir

    @cache_dir.setter
    def cache_dir(self, x):
        raise RuntimeError("cannot set `cache_dir` after instantiation!")

    @property
    def uuid(self):
        return self.uuid

    @uuid.setter
    def uuid(self, x):
        raise RuntimeError("cannot set `uuid` after instantiation!")

    @property
    def poca(self, idx=None):
        if idx is None:
            x = np.zeros((self.on_mfld_pts_k_.shape[0], self.n))
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
