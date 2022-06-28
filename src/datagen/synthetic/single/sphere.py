import os
import sys
import copy
import json
import time
import random
from collections.abc import Iterable

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from utils import *
from .manifold import GeneralManifoldAttrs, SpecificManifoldAttrs, Manifold


logger = init_logger(__name__)

class SpecificSphereAttrs(SpecificManifoldAttrs):

    def __init__(self, mu=0, sigma=1, seed=42, n=100, r=0.5, x_ck=None, **kwargs):
        """
        :param r: radius of the sphere
        :type r: float
        :param x_ck: coordinates of the centre in k-dim space
        :type x_ck: numpy.array
        """

        self._mu = mu
        self._sigma = sigma
        self._seed = seed
        self._n = n
        
        self._r = r
        self.x_ck = x_ck

        self.x_cn_trivial_ = None
        """trivial embedding in n-dims of center"""

        self.x_cn_tr_ = None
        """embedding of center after translation"""

        self.x_cn_rot_ = None
        """embedding of center after rotation"""

        self.x_cn = None
        """centre in the n-dimensional space"""

        self.normed_x_cn = None
        """normalised centre in k-dims"""
        
        self.normed_r = None
        """normalised radius"""

        self.pre_images_k = None
        """k-dimensional on-manifold pre-images of the off-manifold points"""


    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, m):
        raise RuntimeError("cannot set `mu` after instantiation!")

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, m):
        raise RuntimeError("cannot set `sigma` after instantiation!")

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, m):
        raise RuntimeError("cannot set `seed` after instantiation!")

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        raise RuntimeError("cannot set `n` after instantiation!")

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        raise RuntimeError("cannot set `r` after instantiation!")


class RandomSphere(Manifold, Dataset):

    """
    Class for generating dataset of a random
    sphere lying in a low dimensional manifold
    embedded in a higher dimensional space
    """

    def __init__(self, genattrs=None, specattrs=None, N=1000, num_neg=None,\
                 n=100, k=3, r=10.0, D=50.0, max_norm=2.0, mu=10, sigma=5, seed=42,\
                 x_ck=None, rotation=None, translation=None, normalize=True,\
                 norm_factor=None, gamma=0.5, anchor=None, online=False,\
                 off_online=False, augment=False, inferred=False):
        """constructor for class containing a random sphere"""

        ## setting seed
        if seed is not None: seed_everything(seed)

        self._genattrs = genattrs
        self._specattrs = specattrs

        if not isinstance(genattrs, GeneralManifoldAttrs):
            self._genattrs = GeneralManifoldAttrs(N=N, num_neg=num_neg,\
                 n=n, k=k, D=D, max_norm=max_norm, mu=mu, sigma=sigma,\
                 seed=seed, normalize=normalize, norm_factor=norm_factor,\
                 gamma=gamma, rotation=rotation, translation=translation,\
                 anchor=anchor, online=online, off_online=off_online,\
                 augment=augment, inferred=inferred)

        if not isinstance(specattrs, SpecificSphereAttrs):
            self._specattrs = SpecificSphereAttrs(mu=self._genattrs.mu,\
                 sigma=self._genattrs.sigma, seed=self._genattrs.seed,\
                 n=self._genattrs.n, x_ck=x_ck, r=r)

    @property
    def genattrs(self):
        """
        instance of `GeneralManifoldAttrs`
        """
        return self._genattrs

    @genattrs.setter
    def genattrs(self, *args, **kwargs):
        """
        why immutable?: because changing specs without re-computing 
        data is wrong! and recomputing with same object is pointless
        and confusing.
        """
        raise RuntimeError("Cannot set `genattrs` after instantiation!")
        
    @property
    def specattrs(self):
        """
        instance of `SpecificManifoldAttrs`
        """
        return self._specattrs

    @specattrs.setter
    def specattrs(self, *args, **kwargs):
        """
        why immutable?: because changing specs without re-computing 
        data is wrong! and recomputing with same object is pointless
        and confusing.
        """
        raise RuntimeError("Cannot set `specattrs` after instantiation!")

    def __len__(self):
        if self._genattrs.online:
            return self._genattrs.N
        return self._genattrs.points_n.shape[0]

    def online_gen_pre_images(self, batch_pt):
        """generate points/pre-images"""
        batch_pre_image = batch_pt # if `augment` is enabled, then re-use batch points as pre-images
        
        # when `augment` is not enabled, generate pre-images from scratch
        if not self._genattrs.augment:
            batch_pre_image = np.random.normal(self.genattrs.k)
            norm_batch_pre_image = np.linalg.norm(batch_pre_image, axis=1, ord=2)
            batch_pre_image = batch_pre_image / norm_batch_pre_image
            batch_pre_image = self.specattrs.r * batch_pre_image
            batch_pre_image = batch_pre_image + self.specattrs.x_ck
        return batch_pre_image

    def online_compute_normals(self, online_pt):
        
        # normal_vectors_to_mfld_at_p are actually centred at x_ck, but 
        # we can imagine the same vector at $p$, and later adjust the coordinates
        # by adding the position vector of $p$ back.
        #
        # Also note that these negative examples are being generated using the pre-images
        # that we generated and stored in self._genattrs.pre_images_k
        batch_normal_vectors_to_mfld_at_p = online_pt - self._specattrs.x_ck
        batch_embedded_normal_vectors_to_mfld_at_p = np.zeros(self._genattrs.n)
        batch_embedded_normal_vectors_to_mfld_at_p[:self._genattrs.k] = batch_normal_vectors_to_mfld_at_p

        return batch_embedded_normal_vectors_to_mfld_at_p

    def online_make_off_mfld_eg(self, online_batch):
        return super().online_make_off_mfld_eg(online_batch)

    def online_embed_in_n(self, point, is_offline):
        """recieve a point and embed it in n dims"""

        # batch_num_neg = batch_pre_images.shape[0]
        # batch_N = batch_pre_images.shape[0] + batch_pts.shape[0]
        online_point = point
        if is_offline:
            online_point, distance = self.online_make_off_mfld_eg(online_point)

        online_point_n_trivial_ = np.zeros(self._genattrs.n)
        online_point_n_trivial_ = online_point
        online_point_n_tr_ = online_point_n_trivial_ + self._genattrs.translation
        online_point_n_rot_ = np.dot(self._genattrs.rotation, online_point_n_tr_)

        online_point_n = online_point_n_rot_

        online_actual_distance = np.zeros(1)
        if is_offline:
            online_actual_distance[0] = distance
        else:
            online_actual_distance[0] = np.linalg.norm(online_point_n - self._specattrs.x_cn, axis=1, ord=2) - self._specattrs.r
        
        online_distance = np.clip(online_actual_distance, a_min=0, a_max=self._genattrs.D)

        return {
            "points_k": point,
            "points_n": online_point_n,
            "distances": online_distance,
            "actual_distances": online_actual_distance,
        }

    def online_norm(self, online_point_n, online_distance, online_actual_distance):
        """normalise points and distances so that the whole setup lies in a unit sphere"""

        online_normed_point_n = online_point_n / self._genattrs.norm_factor
        online_normed_distance = online_distance / self._genattrs.norm_factor
        online_normed_actual_distance = online_actual_distance / self._genattrs.norm_factor

        # change centre to bring it closer to origin (smaller numbers are easier to learn)
        online_normed_point_n = online_normed_point_n - self._genattrs.anchor + self._genattrs.fix_center

        online_normed_point_n = online_normed_point_n.float()
        online_normed_distance = online_normed_distance.float()
        online_normed_actual_distance = online_normed_actual_distance.float()

        return {
            "normed_points_n": online_normed_point_n,
            "normed_distances": online_normed_distance,
            "normed_actual_distances": online_normed_actual_distance
        }        


    def __getitem__(self, idx):
        # return self._genattrs.points_n[idx], self._genattrs.distances[idx]
        batch = {
            "points_n": self._genattrs.points_n[idx],
            "distances": self._genattrs.distances[idx],
            "actual_distances": self._genattrs.actual_distances[idx],
            "normed_points_n": self._genattrs.normed_points_n[idx],
            "normed_distances": self._genattrs.normed_distances[idx],
            "normed_actual_distances": self._genattrs.normed_actual_distances[idx]
        }
 
        
        # seed_everything(self._genattrs.seed + idx)
        # if self._genattrs.online:
        #     batch["points_k"] = self._genattrs.points_k[idx]
        #     batch = self.online_compute_points(batch)

        return batch

    def gen_center(self):
        """generate a center in lower dimension"""
        if self._specattrs.x_ck is not None:
            return 
        self._specattrs.x_ck = np.random.normal(self._genattrs.mu, \
            self._genattrs.sigma, self._genattrs.k)

    def gen_points(self):
        """
            generating points in k-dim
        
            reference: https://en.wikipedia.org/wiki/N-sphere#Uniformly_at_random_on_the_(n_%E2%88%92_1)-sphere 
        """

        num_onmfld_samples = self._genattrs.N - self._genattrs.num_neg

        points_k = np.random.normal(size=(num_onmfld_samples, self._genattrs.k))
        
        norms = np.linalg.norm(points_k, axis=1, ord=2).reshape(-1, 1)
        points_k = (points_k / norms)
                
        assert (np.round(np.linalg.norm(points_k, axis=1, ord=2)) == 1).all()
        
        points_k = self._specattrs.r * points_k
        
        points_k = points_k + self._specattrs.x_ck
        
        self._genattrs.points_k = points_k

    def gen_pre_images(self, pre_images=None):
        """
        generating on-manifold k-dimensional projections of off-maniofld samples
        
        this is essentially a clone of `self.gen_points` but I am too afraid to fiddle
        with it, in case things break and the earth swallows me whole.

        [20 Jan 2022] EDIT: if either online is not enabled or augment is not enabled
        only then sample fresh points, else this acts as an identity function for input
        parameter `pre_images` which should be points for which augmentations are sought;

        """
        num_off_mfld_ex = self._genattrs.num_neg

        if not (self._genattrs.online and self._genattrs.augment):
            points_k = np.random.normal(size=(num_off_mfld_ex, self._genattrs.k))

            norms = np.linalg.norm(points_k, axis=1, ord=2).reshape(-1, 1)
            points_k = (points_k / norms)
            
            assert (np.round(np.linalg.norm(points_k, axis=1, ord=2)) == 1).all()
            
            points_k = self._specattrs.r * points_k

            points_k = points_k + self._specattrs.x_ck
            
            self._genattrs.pre_images_k = points_k

            return points_k
        elif pre_images is not None:
            random_idx = np.random.randint(self._genattrs.N - self._genattrs.num_neg,\
                size=num_off_mfld_ex)
            # logger.info(random_idx)
            self._genattrs.pre_images_k = pre_images[random_idx]
            return pre_images[random_idx]            

        else:
            raise RuntimeError("expected `pre_images` but got None")


    def compute_normals(self):
        
        # normal_vectors_to_mfld_at_p are actually centred at x_ck, but 
        # we can imagine the same vector at $p$, and later adjust the coordinates
        # by adding the position vector of $p$ back.
        #
        # Also note that these negative examples are being generated using the pre-images
        # that we generated and stored in self._genattrs.pre_images_k
        normal_vectors_to_mfld_at_p = self._genattrs.pre_images_k - self._specattrs.x_ck
        embedded_normal_vectors_to_mfld_at_p = np.zeros((self._genattrs.num_neg, self._genattrs.n))
        embedded_normal_vectors_to_mfld_at_p[:, :self._genattrs.k] = normal_vectors_to_mfld_at_p

        return embedded_normal_vectors_to_mfld_at_p

    def make_off_mfld_eg(self):
        return super().make_off_mfld_eg()

    def embed_in_n(self, resample=False):
        """embedding center and sampled points in `self._genattrs.n`-dims"""
        
        num_neg = self._genattrs.num_neg

        # embedding the center
        if not resample:
            self._specattrs.x_cn_trivial_ = np.zeros(self._genattrs.n)
            self._specattrs.x_cn_trivial_[:self._genattrs.k] = self._specattrs.x_ck
            self._specattrs.x_cn_tr_ = self._specattrs.x_cn_trivial_ + self._genattrs.translation
            self._specattrs.x_cn_rot_ = np.dot(self._genattrs.rotation, self._specattrs.x_cn_tr_)
            self._specattrs.x_cn = self._specattrs.x_cn_rot_
            
        
        # generate the negative examples
        neg_examples, neg_distances = self.make_off_mfld_eg()
        
        #embedding the points
        self._genattrs.points_n_trivial_ = np.zeros((self._genattrs.N, self._genattrs.n))
        self._genattrs.points_n_trivial_[:num_neg] = neg_examples
        
        self._genattrs.points_n_trivial_[num_neg:, :self._genattrs.k] = self._genattrs.points_k
        self._genattrs.points_n_tr_ = self._genattrs.points_n_trivial_ + self._genattrs.translation
        
        self._genattrs.points_n_rot_ = np.dot(self._genattrs.rotation, self._genattrs.points_n_tr_.T).T
        
        self._genattrs.points_n = self._genattrs.points_n_rot_
        
        self._genattrs.actual_distances = np.zeros((self._genattrs.N, 1))
        if not self._genattrs.online: self._genattrs.actual_distances[:num_neg] = neg_distances.reshape(-1, 1)
        self._genattrs.actual_distances[num_neg:] = np.linalg.norm(self._genattrs.points_n[num_neg:] - self._specattrs.x_cn, axis=1, ord=2).reshape(-1, 1) - self._specattrs.r
        self._genattrs.distances = np.clip(self._genattrs.actual_distances, a_min=0, a_max=self._genattrs.D)
        
        # checking that the on-manifold points are still self.r away from center
        # print(np.round(np.linalg.norm(self._genattrs.points_n[self._genattrs.num_neg:] - self._genattrs.x_cn, axis=1, ord=2)))



    def norm(self, resample=False):
        """normalise points and distances so that the whole setup lies in a unit sphere"""
        
        if self._genattrs.norm_factor is None:
            # NOTE: using `_norm_factor` to set `norm_factor` in `self._genattrs`. DO NOT MAKE THIS A HABIT!!!
            self._genattrs._norm_factor = self._genattrs.gamma * np.max(np.linalg.norm(self._genattrs.points_n - self._specattrs.x_cn, ord=2, axis=1)).item()
            assert self._genattrs.norm_factor == self._genattrs._norm_factor

        if (not resample) or (resample and (not self._genattrs.off_online)): 
            # only normalize these when either static or when `off_online` is disabled
            self._genattrs.normed_points_n = self._genattrs.points_n / self._genattrs.norm_factor
        elif resample and self._genattrs.off_online:
            # when not static and `off_online` is enabled
            self._genattrs.normed_points_n[:self._genattrs.num_neg] = self._genattrs.points_n[:self._genattrs.num_neg] / self._genattrs.norm_factor
        if not resample: self._specattrs.normed_x_cn = self._specattrs.x_cn / self._genattrs.norm_factor
        self._genattrs.normed_distances = self._genattrs.distances / self._genattrs.norm_factor
        self._genattrs.normed_actual_distances = self._genattrs.actual_distances / self._genattrs.norm_factor

        # change centre to bring it closer to origin (smaller numbers are easier to learn)
        tmp = self._genattrs.gamma if self._genattrs.gamma is not None else 1
        self._genattrs.fix_center = tmp * np.ones(self._genattrs.n)
        if self._genattrs._anchor is None:
            self._genattrs._anchor = self._specattrs.normed_x_cn
            assert (self._genattrs.anchor == self._genattrs._anchor).all()
        # print("anchor and fix_centers", self._genattrs.anchor, self._genattrs.fix_center)
        self._genattrs.normed_points_n = self._genattrs.normed_points_n - self._genattrs.anchor + self._genattrs.fix_center

        self._genattrs.normed_points_n = self._genattrs.normed_points_n.float()
        self._genattrs.normed_distances = self._genattrs.normed_distances.float()
        self._genattrs.normed_actual_distances = self._genattrs.normed_actual_distances.float()

    def invert_points(self, normed_points):
        """invert normalised points to unnormalised values"""
        normed_points = normed_points - self._genattrs.fix_center + self._specattrs.normed_x_cn
        return normed_points * self._genattrs.norm_factor

    def invert_distances(self, normed_distances):
        """invert normalised distances to unnormalised values"""
        return normed_distances * self._genattrs.norm_factor

    def online_compute_points(self, batch):
        
        online_batch = None

        if self._genattrs.off_online:
            # verify based on biased coin-flip if point is to be off-manifold
            is_offline = np.random.uniform(0, 1) < self._genattrs.num_neg 

            if not is_offline:
                # if not off-manifold, return it as is
                return batch

        # generate on-the-fly pre-images for off-manifold points
        online_pre_images = self.online_gen_pre_images(batch["points_k"])
        
        # embed the on-mfld and off-mfld samples in n-dims
        online_batch = self.online_embed_in_n(online_pre_images, is_offline)

        online_batch = {i: torch.from_numpy(online_batch[i]).float() if not torch.is_tensor(online_batch[i]) else online_batch[i]  for i in online_batch}
        if self._genattrs.normalize:
            online_batch.update(**self.online_norm(
                online_point_n=online_batch["points_n"],
                online_distance=online_batch["distances"],
                online_actual_distance=online_batch["actual_distances"]
            ))
        

        return online_batch

    def compute_points(self):

        self.gen_center()
        logger.info("[RandomSphere]: generated centre")
        self.gen_points()
        logger.info("[RandomSphere]: generated points in k-dim")
        
        if not self.genattrs.inferred:
            pre_images = self._genattrs.points_k if self._genattrs.augment else None
            self.gen_pre_images(pre_images)
            logger.info("[RandomSphere]: pre-images generated")
            self.embed_in_n()
            logger.info("[RandomSphere]: embedded the sphere in n-dim space")

            self._genattrs.points_n = torch.from_numpy(self._genattrs.points_n).float() if not torch.is_tensor(self._genattrs.points_n) else self._genattrs.points_n
            self._genattrs.points_k = torch.from_numpy(self._genattrs.points_k).float() if not torch.is_tensor(self._genattrs.points_k) else self._genattrs.points_k
            self._genattrs.distances = torch.from_numpy(self._genattrs.distances).float() if not torch.is_tensor(self._genattrs.distances) else self._genattrs.distances
            self._genattrs.actual_distances = torch.from_numpy(self._genattrs.actual_distances).float() if not torch.is_tensor(self._genattrs.actual_distances) else self._genattrs.actual_distances

            if self._genattrs.normalize:
                self.norm()
                logger.info("[RandomSphere]: normalization complete")
                    
    def resample_points(self, seed=None):
        """to re-sample points on-the-fly at the end of each epoch"""

        if seed is None:
            logger.info("[RandomSphere]: no seed provided. proceeding with current seed")
        else:
            logger.info("[RandomSphere]: re-sampling points with seed={}".format(seed))
            seed_everything(seed)
        if not self._genattrs.off_online:
            self.gen_points()
            logger.info("[RandomSphere]: generated points in k-dim")
        pre_images = self._genattrs.points_k if self._genattrs.augment else None
        self.gen_pre_images(pre_images)
        logger.info("[RandomSphere]: pre-images generated")
        self.embed_in_n(resample=True)
        logger.info("[RandomSphere]: embedded the sphere in n-dim space")
        
        self._genattrs.points_n = torch.from_numpy(self._genattrs.points_n).float() if not torch.is_tensor(self._genattrs.points_n) else self._genattrs.points_n
        self._genattrs.points_k = torch.from_numpy(self._genattrs.points_k).float() if not torch.is_tensor(self._genattrs.points_k) else self._genattrs.points_k
        self._genattrs.distances = torch.from_numpy(self._genattrs.distances).float() if not torch.is_tensor(self._genattrs.distances) else self._genattrs.distances
        self._genattrs.actual_distances = torch.from_numpy(self._genattrs.actual_distances).float() if not torch.is_tensor(self._genattrs.actual_distances) else self._genattrs.actual_distances


        if self._genattrs.normalize:
            self.norm(resample=True)
            logger.info("[RandomSphere]: normalization complete")


    def viz_test(self, dimX=0, dimY=1, dimZ=2, num_pre_img=5):
        """
            generate plots that act as visual sanity checks
            (only effective for `self._genattrs.n = 3`)
            
            1. distribution of distance labels from manifold
            2. distribution of actual distances from manifold
            3. on-manifold samples - trivial, translated and rotated in input 3 dims
            4. off-manifold samples - pick one, find its coresponding index in self._genattrs.points_k,
               transform it to the on-manifold sample, assert that the distance between the two is equal
               to the distance label and plot this distance

            NOTE: Status - Failing!! Rest of the API has progressed with no updates to this

            TODO: Fix it!
        """
        
        plt.hist(self._genattrs.distances.numpy())
        plt.xlabel("clipped (D={d}) distances from manifold".format(d=self._genattrs.D))
        plt.ylabel("frequency")
        plt.title("distribution of label distances from the manifold")
        plt.show()
        
        plt.hist(self._genattrs.actual_distances.numpy())
        plt.xlabel("actual distances from manifold")
        plt.ylabel("frequency")
        plt.title("distribution of actual distances from the manifold")
        plt.show()
        
        # plot the on-manifold examples
        
        on_mfld_idx = self._genattrs.distances.reshape(-1) == 0
        off_mfld_idx = self._genattrs.distances.reshape(-1) != 0

        # before translation
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self._genattrs.points_n_trivial_[on_mfld_idx, dimX],\
                      self._genattrs.points_n_trivial_[on_mfld_idx, dimY], self._genattrs.points_n_trivial_[on_mfld_idx, dimZ])
        
        plt.title("on-manifold samples (trivial embedding)")
        plt.show()
        
        # before rotation
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self._genattrs.points_n_tr_[on_mfld_idx, dimX],\
                      self._genattrs.points_n_tr_[on_mfld_idx, dimY], self._genattrs.points_n_tr_[on_mfld_idx, dimZ])
        
        plt.title("on-manifold samples (after translation)")
        plt.show()
        
        # after rotation
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self._genattrs.points_n[on_mfld_idx, dimX],\
                      self._genattrs.points_n[on_mfld_idx, dimY], self._genattrs.points_n[on_mfld_idx, dimZ])
        
        plt.title("on-manifold samples (after rotation)")
        plt.show()
        
        
        # plots for off-manifold samples
        
        #indices to show for pre-images
        idx = np.random.choice(np.arange(self._genattrs.num_neg), num_pre_img)
        
        # trivial pre_image
        
        neg_pre_img = self._genattrs.points_k[off_mfld_idx]
        neg_pre_img_trivial_ = np.zeros((self._genattrs.num_neg, self.n))
        neg_pre_img_trivial_[:, :self.k] = neg_pre_img
        
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self._genattrs.points_n_trivial_[on_mfld_idx, dimX],\
              self._genattrs.points_n_trivial_[on_mfld_idx, dimY],\
              self._genattrs.points_n_trivial_[on_mfld_idx, dimZ], color="blue", label="on-manifold", s=1, marker="1")
        
        
        
        ax1.scatter3D(neg_pre_img_trivial_[:, dimX][idx],\
                      neg_pre_img_trivial_[:, dimY][idx],\
                      neg_pre_img_trivial_[:, dimZ][idx], color="green", marker="^", label="trivial pre-image", s=80)
        
        
        
        ax1.scatter3D(self._genattrs.points_n_trivial_[off_mfld_idx, dimX][idx],\
                      self._genattrs.points_n_trivial_[off_mfld_idx, dimY][idx],\
                      self._genattrs.points_n_trivial_[off_mfld_idx, dimZ][idx], color="red", label="off-manifold")
        
        actual_distances_trivial_ = np.linalg.norm(neg_pre_img_trivial_\
                               - self._genattrs.points_n_trivial_[off_mfld_idx], ord=2, axis=1)
        
        for i in idx:
            ax1.plot([neg_pre_img_trivial_[:, dimX][i], self._genattrs.points_n_trivial_[off_mfld_idx, dimX][i]],\
                    [neg_pre_img_trivial_[:, dimY][i], self._genattrs.points_n_trivial_[off_mfld_idx, dimY][i]],\
                    [neg_pre_img_trivial_[:, dimZ][i], self._genattrs.points_n_trivial_[off_mfld_idx, dimZ][i]], color="black")
            ax1.text(np.mean([neg_pre_img_trivial_[:, dimX][i], self._genattrs.points_n_trivial_[off_mfld_idx, dimX][i]]),\
                    np.mean([neg_pre_img_trivial_[:, dimY][i], self._genattrs.points_n_trivial_[off_mfld_idx, dimY][i]]),\
                    np.mean([neg_pre_img_trivial_[:, dimZ][i], self._genattrs.points_n_trivial_[off_mfld_idx, dimZ][i]]),\
                    "{:.2f}".format(actual_distances_trivial_[i].item()))
        
        
        plt.legend()
        plt.title("pre-image samples (trivial embedding)")
        plt.show()
        
        plt.hist(np.linalg.norm(neg_pre_img_trivial_\
                               - self._genattrs.points_n_trivial_[off_mfld_idx], ord=2, axis=1) - self._genattrs.actual_distances[off_mfld_idx].numpy().reshape(-1))
        plt.ylabel("freq")
        plt.xlabel("error value")
        plt.title("error distribution for distance value from manifold (trivial)")
        plt.show()
        
        
        # translated pre_image
        
        
        neg_pre_img_tr_ = neg_pre_img_trivial_ + self.translation
        
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self._genattrs.points_n_tr_[on_mfld_idx, dimX],\
              self._genattrs.points_n_tr_[on_mfld_idx, dimY],\
              self._genattrs.points_n_tr_[on_mfld_idx, dimZ], color="blue", label="on-manifold", s=1, marker="1")
        
        
        ax1.scatter3D(neg_pre_img_tr_[:, dimX][idx],\
                      neg_pre_img_tr_[:, dimY][idx],\
                      neg_pre_img_tr_[:, dimZ][idx], color="green", marker="^", label="translated pre-image", s=80)
        
        
        
        ax1.scatter3D(self._genattrs.points_n_tr_[off_mfld_idx, dimX][idx],\
                      self._genattrs.points_n_tr_[off_mfld_idx, dimY][idx],\
                      self._genattrs.points_n_tr_[off_mfld_idx, dimZ][idx], color="red", label="off-manifold")
        
        for i in idx:
            ax1.plot([neg_pre_img_tr_[:, dimX][i], self._genattrs.points_n_tr_[off_mfld_idx, dimX][i]],\
                    [neg_pre_img_tr_[:, dimY][i], self._genattrs.points_n_tr_[off_mfld_idx, dimY][i]],\
                    [neg_pre_img_tr_[:, dimZ][i], self._genattrs.points_n_tr_[off_mfld_idx, dimZ][i]], color="black")
            ax1.text(np.mean([neg_pre_img_tr_[:, dimX][i], self._genattrs.points_n_tr_[off_mfld_idx, dimX][i]]),\
                    np.mean([neg_pre_img_tr_[:, dimY][i], self._genattrs.points_n_tr_[off_mfld_idx, dimY][i]]),\
                    np.mean([neg_pre_img_tr_[:, dimZ][i], self._genattrs.points_n_tr_[off_mfld_idx, dimZ][i]]),\
                    "{:.2f}".format(self._genattrs.actual_distances[i].item()))
        
        
        plt.legend()
        plt.title("pre-image samples (translated embedding)")
        plt.show()
        
        plt.hist(np.linalg.norm(neg_pre_img_tr_\
                               - self._genattrs.points_n_tr_[off_mfld_idx], ord=2, axis=1) - self._genattrs.actual_distances[off_mfld_idx].numpy().reshape(-1))
        plt.ylabel("freq")
        plt.xlabel("error value")
        plt.title("error distribution for distance value from manifold (tanslated)")
        plt.show()
        
        
        # rotated pre_image
        
        
        neg_pre_img_rot_ = np.dot(self.rotation, neg_pre_img_tr_.T).T
        
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self._genattrs.points_n_rot_[on_mfld_idx, dimX],\
              self._genattrs.points_n_rot_[on_mfld_idx, dimY],\
              self._genattrs.points_n_rot_[on_mfld_idx, dimZ], color="blue", label="on-manifold", s=1, marker="1")
        
        
        ax1.scatter3D(neg_pre_img_rot_[:, dimX][idx],\
                      neg_pre_img_rot_[:, dimY][idx],\
                      neg_pre_img_rot_[:, dimZ][idx], color="green", marker="^", label="rotated pre-image", s=80)
        
        
        
        ax1.scatter3D(self._genattrs.points_n_rot_[off_mfld_idx, dimX][idx],\
                      self._genattrs.points_n_rot_[off_mfld_idx, dimY][idx],\
                      self._genattrs.points_n_rot_[off_mfld_idx, dimZ][idx], color="red", label="off-manifold")
        
        for i in idx:
            ax1.plot([neg_pre_img_rot_[:, dimX][i], self._genattrs.points_n_rot_[off_mfld_idx, dimX][i]],\
                    [neg_pre_img_rot_[:, dimY][i], self._genattrs.points_n_rot_[off_mfld_idx, dimY][i]],\
                    [neg_pre_img_rot_[:, dimZ][i], self._genattrs.points_n_rot_[off_mfld_idx, dimZ][i]], color="black")
            ax1.text(np.mean([neg_pre_img_rot_[:, dimX][i], self._genattrs.points_n_rot_[off_mfld_idx, dimX][i]]),\
                    np.mean([neg_pre_img_rot_[:, dimY][i], self._genattrs.points_n_rot_[off_mfld_idx, dimY][i]]),\
                    np.mean([neg_pre_img_rot_[:, dimZ][i], self._genattrs.points_n_rot_[off_mfld_idx, dimZ][i]]),\
                    "{:.2f}".format(self._genattrs.actual_distances[i].item()))
        
        
        plt.legend()
        plt.title("pre-image samples (rotated embedding)")
        plt.show()
        
        plt.hist(np.linalg.norm(neg_pre_img_rot_\
                               - self._genattrs.points_n_rot_[off_mfld_idx], ord=2, axis=1) - self._genattrs.actual_distances[off_mfld_idx].numpy().reshape(-1))
        plt.ylabel("freq")
        plt.xlabel("error value")
        plt.title("error distribution for distance value from manifold (rotated)")
        plt.show()
        
        min_dist_vals = list()
        
        for i in idx:
            
            neg_ex = self._genattrs.points_n[i]
            min_dist = None
            
            for j in range(self._genattrs.num_neg):
                
                a_pre_img = neg_pre_img_rot_[j]
                dist = np.linalg.norm(neg_ex - a_pre_img, ord=2)
                
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    
            min_dist_vals.append(min_dist)
            
        errors = np.abs(self._genattrs.actual_distances[idx].numpy().reshape(-1) - np.array(min_dist_vals))
        rel_errors = errors / self._genattrs.actual_distances[idx].numpy().reshape(-1)
        logger.info("absolute errors:", errors)
        logger.info("relative errors:", rel_errors)

    def load_data(self, dump_dir):
        """
        `dump_dir` should contain two files - 
        `specs.json` and `data.pkl`. What they contain
        can be learned from `.save_data` method
        """
        specs_fn = os.path.join(dump_dir, "specs.json")
        data_fn = os.path.join(dump_dir, "data.pkl")

        with open(specs_fn) as f:
            specs_attrs = json.load(f)

        data_attrs = torch.load(data_fn)

        attrs = {**specs_attrs, **data_attrs}

        inp_attrs = {i.lstrip("_"): attrs[i] for i in attrs}
        self._genattrs = GeneralManifoldAttrs(**inp_attrs)
        self._specattrs = SpecificSphereAttrs(**inp_attrs)

        for attr_set in [self._genattrs, self._specattrs]:
            for attr in vars(attr_set):
                if attr in attrs:
                    if type(attrs[attr]) == dict and "is_data_attr" in attrs[attr]:
                        if os.path.exists(attrs[attr]["path"]):
                            data_attr = torch.load(attrs[attr]["path"])
                            logger.info("[RandomSphere]: data attribute ({}) loaded from file: {}".format(attr, attrs[attr]["path"]))
                        else:
                            data_fn = os.path.basename(attrs[attr]["path"])
                            path = os.path.join(dump_dir, data_fn)
                            data_attr = torch.load(path)
                            logger.info("[RandomSphere]: data attribute ({}) loaded from file: {}".format(attr, path))
                        setattr(attr_set, attr, data_attr)
                    else:
                        setattr(attr_set, attr, attrs[attr])


    def save_data(self, save_dir):
        super().save_data(save_dir)
        # """
        # based on past mistakes:
        # 1. keep specification variables (typically non-iterables)
        #    separately
        # 2. keep tensors in a dictionary and pickle the dictionary, not
        #    the whole object; helps with serialization errors
        # """
        # os.makedirs(save_dir, exist_ok=True)
        # specs_fn = os.path.join(save_dir, "specs.json")
        # data_fn = os.path.join(save_dir, "data.pkl")

        # specs_attrs = dict()
        # data_attrs = dict()

        # gen_attrs = vars(self._genattrs)
        # sphere_attrs = vars(self._specattrs)

        # for attr_set in [gen_attrs, sphere_attrs]:
        #     for attr in attr_set:
        #         if not isinstance(attr_set[attr], Iterable):
        #             specs_attrs[attr] = attr_set[attr]
        #         else:
        #             attr_fn = os.path.join(save_dir, attr + ".pkl")
        #             torch.save(attr_set[attr], attr_fn)
        #             logger.info("[{}}]: data attribute ({}) saved to: {}".format(self.__class__.__name__, attr, attr_fn))
        #             data_attrs[attr] = {"is_data_attr": True, "path": attr_fn}

        # with open(specs_fn, "w+") as f:
        #     json.dump(specs_attrs, f)

        # torch.save(data_attrs, data_fn)

    @classmethod
    def get_demo_cfg_dict(cls):

        train_cfg_dict = {
            "N": 2500000,
            "num_neg": None,
            "n": 500,
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

        cfg_dict = {
            "train": train_cfg_dict,
            "val": val_cfg_dict,
            "test": test_cfg_dict
        }

        return cfg_dict

    @classmethod
    def save_splits(cls, train_set, val_set, test_set, save_dir):
        return super().save_splits(train_set, val_set, test_set, save_dir)
    
    @classmethod
    def make_train_val_test_splits(cls, cfg_dict=None, save_dir=None):
        
        logger.info("[RandomSphere]: starting with split generation")

        if cfg_dict is None:
            cfg_dict = cls.get_demo_cfg_dict()

        logger.info("[RandomSphere]: generating train set...")
        train_cfg = cfg_dict["train"]
        train_set = cls(**train_cfg)
        train_set.compute_points()
        logger.info("[RandomSphere]: train set generation done!")

        logger.info("[RandomSphere]: generating val set...")
        val_cfg = cfg_dict["val"]
        val_cfg["x_ck"] = train_set.specattrs.x_ck
        val_cfg["rotation"] = train_set.genattrs.rotation
        val_cfg["translation"] = train_set.genattrs.translation
        val_set = cls(**val_cfg)
        val_set.compute_points()
        logger.info("[RandomSphere]: val set generation done!")

        logger.info("[RandomSphere]: generating test set...")
        test_cfg = cfg_dict["test"]
        test_cfg["x_ck"] = train_set.specattrs.x_ck
        test_cfg["rotation"] = train_set.genattrs.rotation
        test_cfg["translation"] = train_set.genattrs.translation
        test_set = cls(**test_cfg)
        test_set.compute_points()
        logger.info("[RandomSphere]: test set generation done!")

        if save_dir is not None:
            logger.info("[RandomSphere]: saving splits at: {}".format(save_dir))
            cls.save_splits(train_set, val_set, test_set, save_dir)
            logger.info("[RandomSphere]: splits saved")

        logger.info("[RandomSphere]: generated splits!")
        return train_set, val_set, test_set
        

    @classmethod
    def load_splits(cls, dump_dir):
        return super().load_splits(dump_dir)
    




if __name__ == '__main__':

    dummy_params = {

        "N": 100000,
        "num_neg": None,
        "n": 2,
        "k": 2,
        "r": 0.5,
        "D": 0.2,
        "max_norm": 0.25,
        "mu": 0,
        "sigma": 1,
        "seed": 42,
        "gamma": 1

    }

    test = RandomSphere(**dummy_params)
    test.compute_points()
    test.save_data("./test")
    b = RandomSphere()
    b.load_data("./test")


