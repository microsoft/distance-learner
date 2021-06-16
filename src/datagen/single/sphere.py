import os
import sys
import json
import time
import random

import numpy as np

import torch
from torch.utils.data import Dataset

from manifold import GeneralManifoldAttrs, SpecificManifoldAttrs, Manifold


class SpecificSphereAttrs(SpecificManifoldAttrs):

    def __init__(self, mu=0, sigma=1, seed=42, n=100, r=0.5, x_ck=None):
        """
        :param r: radius of the sphere
        :type r: float
        :param x_cn: coordinates of the centre in k-dim space
        :type x_cn: numpy.array
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
                 norm_factor=None, gamma=0.5):
        """constructor for class containing a random sphere"""

        self._genattrs = genattrs
        self._specattrs = specattrs

        if not isinstance(genattrs, GeneralManifoldAttrs):
            self._genattrs = GeneralManifoldAttrs(N=N, num_neg=num_neg,\
                 n=n, k=k, D=D, max_norm=max_norm, mu=mu, sigma=sigma,\
                 seed=seed, normalize=normalize, norm_factor=norm_factor,\
                 gamma=gamma, rotation=rotation, translation=translation)

        if not isinstance(specattrs, SpecificSphereAttrs):
            self._specattrs = SpecificSphereAttrs(mu=self._genattrs.mu,\
                 sigma=self._genattrs.sigma, seed=self._genattrs.seed,\
                 n=self._genattrs.n, x_ck=x_ck)

        self.compute_points()

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
        return self._genattrs.points_n.shape[0]

    def __getitem__(self, idx):
        # return self.points_n[idx], self.distances[idx]
        return {
            "points_n": self._genattrs.points_n[idx],
            "distances": self._genattrs.distances[idx],
            "actual_distances": self._genattrs.actual_distances[idx],
            "normed_points_n": self._genattrs.normed_points_n[idx],
            "normed_distances": self._genattrs.normed_distances[idx],
            "normed_actual_distances": self._genattrs.normed_actual_distances[idx]
        }

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
        points_k = np.random.normal(size=(self._genattrs.N - self._genattrs.num_neg, self._genattrs.k))
        
        norms = np.linalg.norm(points_k, axis=1, ord=2).reshape(-1, 1)
        points_k = (points_k / norms)
                
        assert (np.round(np.linalg.norm(points_k, axis=1, ord=2)) == 1).all()
        
        points_k = self._specattrs.r * points_k
        
        points_k = points_k + self._specattrs.x_ck
        
        self._genattrs.points_k = points_k

    def gen_pre_images(self):
        """
        generating on-manifold k-dimensional projections of off-maniofld samples
        
        this is essentially a clone of `self.gen_points` but I am too afraid to fiddle
        with it, in case things break and the earth swallows me whole.
        """

        points_k = np.random.normal(size=(self._genattrs.num_neg, self._genattrs.k))
        
        
        norms = np.linalg.norm(points_k, axis=1, ord=2).reshape(-1, 1)
        points_k = (points_k / norms)
        
        assert (np.round(np.linalg.norm(points_k, axis=1, ord=2)) == 1).all()
        
        points_k = self._specattrs.r * points_k

        points_k = points_k + self._specattrs.x_ck
        
        self._specattrs.pre_images_k = points_k

    def compute_normals(self):
        
        # normal_vectors_to_mfld_at_p are actually centred at x_ck, but 
        # we can imaging the same vector at $p$, and later adjust the coordinates
        # by adding the position vector of $p$ back.
        #
        # Also note that these negative examples are being generated using the pre-images
        # that we generated and stored in self._specattrs.pre_images_k
        normal_vectors_to_mfld_at_p = self._specattrs.pre_images_k - self._specattrs.x_ck
        embedded_normal_vectors_to_mfld_at_p = np.zeros((self._genattrs.num_neg, self._genattrs.n))
        embedded_normal_vectors_to_mfld_at_p[:, :self._genattrs.k] = normal_vectors_to_mfld_at_p

        return embedded_normal_vectors_to_mfld_at_p

    def make_off_mfld_eg(self):
        return super().make_off_mfld_eg()

    def embed_in_n(self):
        
        """embedding center and sampled points in `self._genattrs.n`-dims"""
        
        # embedding the center
        self._specattrs.x_cn_trivial_ = np.zeros(self._genattrs.n)
        self._specattrs.x_cn_trivial_[:self._genattrs.k] = self._specattrs.x_ck
        self._specattrs.x_cn_tr_ = self._specattrs.x_cn_trivial_ + self._genattrs.translation
        self._specattrs.x_cn_rot_ = np.dot(self._genattrs.rotation, self._specattrs.x_cn_tr_)
        self._specattrs.x_cn = self._specattrs.x_cn_rot_
        
        
        # generate the negative examples
        neg_examples, neg_distances = self.make_off_mfld_eg()
        
        #embedding the points
        self._genattrs.points_n_trivial_ = np.zeros((self._genattrs.N, self._genattrs.n))
        self._genattrs.points_n_trivial_[:self._genattrs.num_neg] = neg_examples
        
        self._genattrs.points_n_trivial_[self._genattrs.num_neg:, :self._genattrs.k] = self._genattrs.points_k
        self._genattrs.points_n_tr_ = self._genattrs.points_n_trivial_ + self._genattrs.translation
        
        self._genattrs.points_n_rot_ = np.dot(self._genattrs.rotation, self._genattrs.points_n_tr_.T).T
        
        self._genattrs.points_n = self._genattrs.points_n_rot_
        
        self._genattrs.actual_distances = np.zeros((self._genattrs.N, 1))
        self._genattrs.actual_distances[:self._genattrs.num_neg] = neg_distances.reshape(-1, 1)
        self._genattrs.actual_distances[self._genattrs.num_neg:] = np.linalg.norm(self._genattrs.points_n[self._genattrs.num_neg:] - self._specattrs.x_cn, axis=1, ord=2).reshape(-1, 1) - self._specattrs.r
        self._genattrs.distances = np.clip(self._genattrs.actual_distances, a_min=0, a_max=self._genattrs.D)
        
        # checking that the on-manifold points are still self.r away from center
        # print(np.round(np.linalg.norm(self._genattrs.points_n[self.num_neg:] - self._genattrs.x_cn, axis=1, ord=2)))



    def norm(self):
        """normalise points and distances so that the whole setup lies in a unit sphere"""
        if self._genattrs.norm_factor is None:
            # NOTE: using `_norm_factor` to set `norm_factor` in `self._genattrs`. DO NOT MAKE THIS A HABIT!!!
            self._genattrs._norm_factor = self._genattrs.gamma * np.max(np.linalg.norm(self._genattrs.points_n - self._specattrs.x_cn, ord=2, axis=1))
            assert self._genattrs.norm_factor == self._genattrs._norm_factor
        self._genattrs.normed_points_n = self._genattrs.points_n / self._genattrs.norm_factor
        self._specattrs.normed_x_cn = self._specattrs.x_cn / self._genattrs.norm_factor
        self._genattrs.normed_distances = self._genattrs.distances / self._genattrs.norm_factor
        self._genattrs.normed_actual_distances = self._genattrs.actual_distances / self._genattrs.norm_factor

        # change centre to bring it closer to origin (smaller numbers are easier to learn)
        tmp = self._genattrs.gamma if self._genattrs.gamma is not None else 1
        self._genattrs.fix_center = tmp * np.ones(self._genattrs.n)
        self._genattrs.normed_points_n = self._genattrs.normed_points_n - self._specattrs.normed_x_cn + self._genattrs.fix_center

    def invert_points(self, normed_points):
        """invert normalised points to unnormalised values"""
        normed_points = normed_points - self._genattrs.fix_center + self._specattrs.normed_x_cn
        return normed_points * self._genattrs.norm_factor

    def invert_distances(self, normed_distances):
        """invert normalised distances to unnormalised values"""
        return normed_distances * self._genattrs.norm_factor

    def compute_points(self):
        
        self.gen_center()
        print("[RandomSphere]: generated centre")
        self.gen_points()
        print("[RandomSphere]: generated points in k-dim")
        self.gen_pre_images()
        print("[RandomSphere]: pre-images generated")
        self.embed_in_n()
        print("[RandomSphere]: embedded the sphere in n-dim space")

        self._genattrs.points_n = torch.from_numpy(self._genattrs.points_n).float()
        self._genattrs.points_k = torch.from_numpy(self._genattrs.points_k).float()
        self._genattrs.distances = torch.from_numpy(self._genattrs.distances).float()
        self._genattrs.actual_distances = torch.from_numpy(self._genattrs.actual_distances).float()

        if self._genattrs.normalize:
            self.norm()
            print("[RandomSphere]: normalization complete")

    def viz_test(self):
        return super().viz_test()

    def load_data(self):
        return super().load_data()

    def save_data(self, save_dir):
        return super().save_data(save_dir)

    


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


