import os
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
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import scipy as sp
import numpy.linalg as npla
import scipy.linalg as spla

import torch
import torchvision

from sklearn.decomposition import PCA

from tqdm import tqdm

from utils import *
from datagen.datagen_utils import *


logger = init_logger(__name__)


class RealWorldManifolds(ABC):
    """
    This class acts as an abstract class for all the real-world datasets
    that we will be using. It implements some common methods and attributes 
    that are dataset agnostic. 

    TODO: Have to make dataset agnostic functions such that they can handle 
    datasets that are too large to be loaded in memory all at once. This would 
    involve memory and time efficient ways to do k-NN and implementing ways to
    index data on the disk and not in memory.
    """

    def __init__(
        self,
        num_pos,
        on_mfld_path,
        k,
        n,
        use_labels,
        off_mfld_label,
        seed=23,
        download=False,
        load_all=True,
        split="train",
        N=None,
        num_neg=None,
        nn=30,
        max_t_delta=1e-3,
        max_norm=1e-1,
        M=1.0,
        transform=None,
        **kwargs):
        """
        :param num_pos: number of on-manifold samples in the dataset
        :type num_pos: int
        :param on_mfld_path: path to on-manifold samples
        :type on_mfld_path: str
        :param k: estimate of intrinsic data dimension
        :type k: int
        :param n: embedding dimension
        :type n: int
        :param use_labels: class labels to keep in the dataset
        :type use_labels: list
        :param off_mfld_label: the off-manifold label
        :type off_mfld_label: int
        :param seed: random seed
        :type seed: int
        :param download: whether to download on manifold samples or not
        :type download: bool
        :param load_all: load all samples in memory (helps with slicing)
        :type load_all: bool
        :param split: split to generate - "train", "val"  or "test"
        :type split: str
        :param N: total number of samples (on + off manifold)
        :type N: int
        :param num_neg: number of off-manifold examples
        :type num_neg: int
        :param nn: number of nearest nbhrs to search for
        :type nn: int
        :param max_t_delta: maximum tangential perturbation
        :type max_t_delta: float
        :param max_norm: maximum normal perturbation
        :type max_norm: float
        :param M: high value of distance set for distance to the other manifolds
        :type M: float
        :param transform: transform to apply to samples in the dataset
        :type transform: Optional[Callable]
        """

        self.seed = seed
        seed_everything(seed)

        self.num_pos = num_pos
        if N is None and num_neg is not None:
            N = num_pos + num_neg
        elif N is not None:
            if num_neg is not None:
                assert N == num_pos + num_neg, "incompatible values for `N`, `num_neg`, `num_pos`"
            else:
                num_neg = N - num_pos
        else:
            N = num_pos
            num_neg = 0

        self.N = N
        self.num_neg = num_neg

        self.nn = nn
        self.knn = None

        self.k = k
        self.n = n

        self.load_all = load_all
        self.on_mfld_path = on_mfld_path
        self._download = download # don't want to expose this outside
        self.split = split

        self.use_labels = use_labels
        self.off_mfld_label = off_mfld_label
        self.transform = transform

        self.max_t_delta = max_t_delta
        self.max_norm = max_norm
        self.M = M

        self._num_offmfld_by_class = [self.num_neg // len(self.use_labels)] * len(self.use_labels)
        if sum(self._num_offmfld_by_class) != self.num_neg:
            tmp = self.num_neg - (self._num_offmfld_by_class * len(self.use_labels))
            for i in range(tmp):
                self._num_offmfld_by_class[(tmp) % len(self.use_labels)] += 1
        
        self.dataset = None

        self.dataset_flat = None

        self.on_mfld_pts = None
        self.on_mfld_class_labels = None
        self.onmfld_class_label_counts = None

        self.all_points = None
        self.all_actual_distances = None
        self.class_labels = None
        self.pre_image_labels = None
        
        self.poca_idx = None
        self.poca_idx_counts = None

    @abstractmethod
    def load_raw_om_data(self):
        """dataset specific method for loading raw on-mfld data"""
        raise NotImplementedError("""implement dataset-specific method\
             that returns a Pytorch dataset object and the same object\
             as a (Tensor, Tensor) object containing points and class labels""")

    @abstractmethod
    def init_onmfld_pts(self, om_augs=None):
        """
        initialise the on-manifold images from appropriate data source
        such as torchvision or some external archive and select images 
        whose classes lie in `use_labels`
        """

        # load dataset
        self.dataset, self.dataset_flat = self.load_raw_om_data()

        # load on-mfld pts
        num_om_samples = len(self.dataset)
        if om_augs is not None:
            logger.info("[{}]: on-manifold augmentations provided...".format(self.__class__.__name__))
            num_om_samples = len(self.dataset) + om_augs.shape[0]
            logger.info("[{}]: size of on-mfld set = {}".format(self.__class__.__name__, num_om_samples))


        self.on_mfld_pts = torch.zeros(num_om_samples, self.n)
        self.on_mfld_pts[:len(self.dataset)] = self.dataset_flat[0]
        self.on_mfld_pts[len(self.dataset):] = om_augs[0]
        
        self.on_mfld_class_labels = torch.zeros(num_om_samples)
        self.on_mfld_class_labels[:len(self.dataset)] = self.dataset_flat[1]
        self.on_mfld_class_labels[len(self.dataset):] = om_augs[1]
        self.on_mfld_class_labels = self.on_mfld_class_labels.long()

        use_idx = np.isin(self.on_mfld_class_labels, self.use_labels)
        self.on_mfld_pts = self.on_mfld_pts[use_idx]
        self.on_mfld_class_labels = self.on_mfld_class_labels[use_idx]

        # populate class_label_counts
        tmp = np.unique(self.on_mfld_class_labels, return_counts=True)
        self.onmfld_class_label_counts = dict()
        for i in range(len(tmp[0])):
            self.onmfld_class_label_counts[tmp[0][i]] = self.onmfld_class_label_counts[tmp[1][i]]

    
    @abstractmethod
    def find_knn(self, X):
        """
        find nearest neighbors of input points in n-dim space

        TODO: have to figure out how to index datasets that will not fit in RAM
        """            
        if self.knn is None:
            self.knn = FaissKNeighbors(k=self.nn + self.buf_nn)
            logger.info("[{}]: fitting knn...".format(self.__class__.__name__))
            self.knn.fit(self.on_mfld_pts)
            logger.info("[{}]: knn fit done".format(self.__class__.__name__))

        logger.info("[{}]: predicting nbhrs...".format(self.__class__.__name__))
        distances, indices = self.knn.predict(X)
        logger.info("[{}]: prediction complete...".format(self.__class__.__name__))

        return distances, indices

    @abstractmethod
    def make_poca_idx(self):
        self.poca_idx = np.zeros(self.num_neg, dtype=np.int64)
        end_idx = 0
        for i in range(self.use_labels):
            self.poca_idx[end_idx:end_idx+self._num_offmfld_by_class[i]] = np.random.choice(np.where(self.class_labels == self.use_labels[i]), size=self._num_offmfld_by_class[i], replace=True).astype(np.int64)
            end_idx += self._num_offmfld_by_class[i]
        assert end_idx == self.num_neg
        tmp = np.unique(self.poca_idx, return_counts=True)
        self.poca_idx_counts[tmp[0]] = tmp[1]

    def make_inferred_off_mfld(self, pp_chunk_size=5000):

        if self.nn_distances is None or self.nn_indices is None:
            logger.info("[{}]: knn not computed. computing now ...".format(self.__class__.__name__))
            self.nn_distances, self.nn_indices = self.find_knn(self.on_mfld_pts)
        
        if self.all_actual_distances is None:
            self.all_actual_distances = np.zeros((self.N, len(self.use_labels)))

        if self.all_points is None:
            self.all_points = np.zeros((self.N, self.n))

        if self.class_labels is None:
            self.class_labels = np.zeros(self.N).astype(np.int64)

        if self.pre_image_labels is None:
            self.pre_image_labels = np.zeros(self.N).astype(np.int64)
        
        num_offmfld_per_idx = max(self.poca_idx_counts)
        total_num_neg_made = 0

        for i in tqdm(range(0, int(self.num_pos), pp_chunk_size), desc="computing off mfld (2)"):

            nbhr_indices = self.nn_indices[i:min(self.num_pos, i+pp_chunk_size)]
            mask = np.equal(nbhr_indices, np.arange(i, min(self.num_pos, i+pp_chunk_size)).reshape(-1, 1))
            mask[mask.sum(axis=1) == 0, -1] = True
            nbhr_indices = nbhr_indices[~mask].reshape(nbhr_indices.shape[0], nbhr_indices.shape[1] - 1).astype(np.int64)

            on_mfld_pts = self.on_mfld_pts[i:min(self.num_pos, i+pp_chunk_size)]
            nbhrs = self.on_mfld_pts[nbhr_indices]

            nbhr_local_coords = (nbhrs.transpose(1, 0, 2) - on_mfld_pts).transpose(1, 0, 2)

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
            on_mfld_pb = np.zeros((actual_chunk_size, num_offmfld_per_idx, self.n))
            on_mfld_pb_sizes = np.random.uniform(0, self.max_t_delta, size=(actual_chunk_size, num_offmfld_per_idx))

            off_mfld_pb_coeffs = np.random.normal(size=(actual_chunk_size, num_offmfld_per_idx, self.n - self.k + 1))
            off_mfld_pb = np.zeros((actual_chunk_size, num_offmfld_per_idx, self.n))
            off_mfld_pb_sizes = np.random.uniform(0, self.max_norm, size=(actual_chunk_size, num_offmfld_per_idx))
            
            if self.N <= 20000:
                self.on_mfld_pb_sizes = on_mfld_pb_sizes
                self.off_mfld_pb_sizes = off_mfld_pb_sizes

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

            total_num_neg_made += off_mfld_pts_for_chunk.shape[0]
            
            self.all_points[i:min(self.num_pos, i+pp_chunk_size)] = off_mfld_pts_for_chunk
            self.pre_image_labels[i:min(self.num_pos, i+pp_chunk_size)] = self.on_mfld_class_labels[self.poca_idx[i:min(self.num_pos, i+pp_chunk_size)]]
            self.class_labels[i:min(self.num_pos, i+pp_chunk_size)] = self.off_mfld_label
            self.all_actual_distances[i:min(self.num_pos, i+pp_chunk_size), :] = self.M
            self.all_actual_distances[np.arange(i, min(self.num_pos, i+pp_chunk_size)), self.class_labels[i:min(self.num_pos, i+pp_chunk_size)]] = off_mfld_pb_sizes

        assert total_num_neg_made == self.num_neg


    @abstractmethod
    def compute_points(self, om_augs=None):
        """
        :param om_augs: on-manifold augmentations, (Tensor, Tensor) of samples and classes
        """

        # initialise on-mfld points
        self.init_onmfld_pts(om_augs=om_augs)
        logger.info("[{}]: initialised on-mfld points".format(self.__class__.__name__))

         # make off manifold points
        self.make_inferred_off_mfld(pp_chunk_size=5000)
        logger.info("[{}]: completed off-mfld generation".format(self.__class__.__name__))

        # filling on-mfld points in the main container
        if om_augs is None:
            self.all_points[self.num_neg:] = self.on_mfld_pts
            self.pre_image_labels[self.num_neg:] = self.on_mfld_class_labels
            self.class_labels[self.num_neg:] = self.on_mfld_class_labels
        else:
            self.all_points[self.num_neg:] = self.dataset_flat[0]
            self.pre_image_labels[self.num_neg:] = self.dataset_flat[1]
            self.class_labels[self.num_neg:] = self.dataset_flat[1]
               
        self.all_actual_distances[self.num_neg:, :] = self.M
        self.all_actual_distances[np.arange(self.num_neg, self.N), self.class_labels[self.num_neg:]] = 0

    @abstractmethod
    def save_data(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        specs_fn = os.path.join(save_dir, "specs.json")
        data_fn = os.path.join(save_dir, "data.pkl")

        specs_attrs = dict()
        data_attrs = dict()

        attr_set = vars(self)
        for attr in attr_set:

            if attr in ["dataset", "dataset_flat"]:
                continue
            
            if (type(attr_set[attr]) == str) or not isinstance(attr_set[attr], Iterable):
                specs_attrs[attr] = attr_set[attr]
            else:
                attr_fn = os.path.join(save_dir, attr + ".pkl")
                torch.save(attr_set[attr], attr_fn)
                logger.info("[{}]: data attribute ({}) saved to: {}".format(self.__class__.__name__, attr, attr_fn))
                data_attrs[attr] = {"is_data_attr": True, "path": attr + ".pkl"}
        
        with open(specs_fn, "w+") as f:
            json.dump(specs_attrs, f)

        torch.save(data_attrs, data_fn)

    @abstractmethod
    def load_data(self, dump_dir):
        specs_fn = os.path.join(dump_dir, "specs.json")
        data_fn = os.path.join(dump_dir, "data.pkl")

        with open(specs_fn) as f:
            specs_attrs = json.load(f)

        data_attrs = torch.load(data_fn)

        attrs = {**specs_attrs, **data_attrs}

        attr_set = vars(self)
        for attr in attr_set:
            if attr in ["dataset", "dataset_flat"]:
                continue
            if attr in attrs:
                if type(attrs[attr]) == dict and "is_data_attr" in attrs[attr]:
                    data_attr = None
                    data_fn = os.path.basename(attrs[attr]["path"])
                    path = os.path.join(dump_dir, data_fn)
                    data_attr = torch.load(path)
                    logger.info("[{}]: data attribute ({}) loaded from file: {}".format(self.__class__.__name__, attr, path))
                    setattr(self, attr, data_attr)
                else:
                    setattr(self, attr, attrs[attr])

        self.dataset, self.dataset_flat = self.load_raw_om_data()

    @abstractmethod
    @classmethod
    def get_demo_cfg_dict(cls):
        raise NotImplementedError()

    @abstractmethod
    @classmethod
    def make_train_val_test_splits(cls, cfg_dict=None, save_dir=None):
    
        logger.info("[{}]: starting with split generation".format(cls.__name__))
        if cfg_dict is None:
            cfg_dict = cls.get_demo_cfg_dict()

        # `strategy` stores the strategy for making val, test splits. chosen from ["only", "full"].
        # "only" - means val/test split consist only of the actual raw data points, all with distances 0
        # "full" - means we generate off-manifold points for the split using train set + on-mfld samples of the split
        strategy = cfg_dict["strategy"]
        # `has_val` tells whether dataset has a validation set or not
        has_val = cfg_dict["has_val"]

        logger.info("[{}]: generating train set...".format(cls.__name__))
        train_cfg = cfg_dict["train"]
        train_set = cls(**train_cfg)
        train_set.compute_points()
        logger.info("[{}]: train set generation done!".format(cls.__name__))

        logger.info("[{}]: generating val set...".format(cls.__name__))
        val_cfg = cfg_dict["val"]
        val_set = cls(**val_cfg)
        if strategy == "only":
            val_set.compute_points()
        elif strategy == "full":
            om_augs = train_set.dataset_flat
            val_set.compute_points(om_augs)
        logger.info("[{}]: val set generation done!".format(cls.__name__))

        logger.info("[{}]: generating test set...".format(cls.__name__))
        if has_val:
            test_cfg = cfg_dict["test"]
            test_set = cls(**test_cfg)
            if strategy == "only":
                test_set.compute_points()
            elif strategy == "full":
                om_augs = train_set.dataset_flat
                test_set.compute_points(om_augs)
        else:
            logger.info("[{}]: dataset has no val set. test set will be copy of val set".format(cls.__name__))
            logger.info("[{}]: generating copy...".format(cls.__name__))
            test_set = copy.deepcopy(val_set)
        logger.info("[{}]: test set generation done!".format(cls.__name__))


        if save_dir is not None:
            logger.info("[{}]: saving splits at: {}".format(cls.__name__, save_dir))
            cls.save_splits(train_set, val_set, test_set, save_dir)
        
        logger.info("[{}]: generated splits!".format(cls.__name__))
        return train_set, val_set, test_set


    @abstractmethod
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

    @abstractmethod
    @classmethod
    def load_splits(cls, dump_dir):
        train_dir = os.path.join(dump_dir, "train")
        os.makedirs(train_dir, exist_ok=True)
        train_set = cls()

        try:
            train_set.load_data(train_dir)
        except:
            logger.info("[ConcentricSpheres]: could not load train split!")
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




def _get_tn_for_on_mfld_idx(
    idx,nbhr_local_coords, 
    k,
    tang_dset_per_dir_size=None, 
    norm_dset_per_dir_size=None, 
    tang_dir=None, 
    norm_dir=None, 
    return_dirs=True):
    """compute tangential and normal directions for all on-manifold points"""

    pca = PCA(n_components=k-1) # manifold is (k-1) dim so tangent space should be same
    pca.fit(nbhr_local_coords)

    tangential_dirs = pca.components_
    normal_dirs = spla.null_space(tangential_dirs).T

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

    t_and_n = np.zeros((tangential_dirs.shape[1], tangential_dirs.shape[1]))
    t_and_n[:k-1] = tangential_dirs
    t_and_n[k-1:] = normal_dirs
    if return_dirs: return t_and_n

    