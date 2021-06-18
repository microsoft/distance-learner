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

from manifold import GeneralManifoldAttrs, SpecificManifoldAttrs, Manifold


identity = lambda x: x
d_identity = lambda x: 1
contract = lambda x, k=np.pi: x - k
d_contract = lambda x: 1

# def identity(x):
#     """id. functional transform for swissroll"""
#     return x

# def d_identity(x):
#     """derivative of identity w.r.t. input"""
#     return 1

# def contract(x, k=np.pi):
#     return x - k

# def d_contract(x):
#     """derivative of contract w.r.t. input"""
#     return 1

class SpecificSwissRollAttrs(SpecificManifoldAttrs):

    def __init__(self, mu=0, sigma=1, n=100, seed=42, g=identity, d_g=d_identity, t_min=1.5*np.pi,\
         t_max=4.5*np.pi, omega=np.pi * 0.1, num_turns=None, noise=0, correct=True,\
         height=21, **kwargs):
        """
        :param g: "amplitude" function for swiss roll
        :type g: function
        :param d_g: derivative of the "amplitude" function
        :type d_g: function
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
        :param height: 'height' of the swiss roll in the non-lateral directions
        :type height: float
        """

        self._mu = mu
        self._sigma = sigma
        self._seed = seed
        self._n = n

        self._t_min = t_min
        """start of time interval for sampling"""
        self._t_max = t_max
        """end of time interval for sampling"""
        self._noise = noise
        """noise to be added"""

        self._g = g
        self._d_g = d_g
        self._omega = omega
        """angular velocity along the swiss roll"""
        self._num_turns = num_turns
        """number of total 'turns' in the swiss roll"""

        self.t = None
        """The univariate position of the sample according
         to the main dimension of the points in the manifold."""

        self.pre_images_t = None
        """The univariate position of the sample according
         to the main dimension of the points in the manifold
         for the pre-images of the off-manifold points"""

        self._correct = correct
        """Swiss roll should look like a swiss roll. If yours is not looking
        like one, enable this flag for some heuristics to kick in"""
        self._scale = kwargs["scale"] if "scale" in kwargs else None
        """used when `self.correct` is enabled"""

        self._height = height

        self._gap = 2 * np.pi / omega if omega is not None else None
        """gap between successive rings of the swiss roll"""

    def setup(self):
        """helper function for managing input parameters"""

        if (self._t_max is not None) and (self._omega is not None) and (self._num_turns is not None):
            print(self._t_min, self._t_max, self._omega, self._num_turns)
            assert self._num_turns == ((self._t_max - self._t_min) * self._omega) / (2 * np.pi), "`num_turns`, `t_max`, and `omega` are incompatible!"
            
        elif self._t_max is None:
            self._t_max = self._t_min + ((2 * np.pi / self._omega) * self._num_turns)
            
        elif self._omega is None:
            self._omega = (self._num_turns * 2 * np.pi) / (self._t_max - self._t_min)
            
        elif self._num_turns is None:
            self._num_turns = ((self._t_max - self._t_min) * self._omega) / (2 * np.pi)
        
        if self._num_turns < 1:
            print("[RandomSwissRoll]: [warning] num_turns < 1, might not lead to meaningful dataset")
            if self._correct:
                print("[RandomSwissRoll]: [log] `correct` enabled, fixing things")
                self._scale = self._omega / np.pi
                self._t_min = self._t_min / self._scale
                self._t_max = self._t_max / self._scale
                print("[RandomSwissRoll]: [log] tried fixing: new values listed below")
                print("[RandomSwissRoll]: [log]", "t_min:", self._t_min, "t_max:", self._t_max,\
                     "omega:", self._omega, "num_turns:", self._num_turns, "gap:", 2*np.pi/self._omega)

        self._gap = 2 * np.pi / self._omega
        

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
    def g(self):
        return self._g

    @g.setter
    def g(self, n):
        raise RuntimeError("cannot set `g` after instantiation!")

    @property
    def d_g(self):
        return self._d_g

    @d_g.setter
    def d_g(self, n):
        raise RuntimeError("cannot set `d_g` after instantiation!")

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, n):
        raise RuntimeError("cannot set `omega` after instantiation!")

    @property
    def gap(self):
        return self._gap

    @gap.setter
    def gap(self, n):
        raise RuntimeError("cannot set `gap` after instantiation!")

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
    def height(self):
        return self._height

    @height.setter
    def height(self, n):
        raise RuntimeError("cannot set `height` after instantiation!")

class RandomSwissRoll(Manifold, Dataset):

    def __init__(self, genattrs=None, specattrs=None, N=1000, num_neg=None,\
                 n=100, k=3, D=1.5, max_norm=2, mu=10, sigma=5, seed=42,\
                 t_min=1.5, t_max=4.5, omega=np.pi, num_turns=None, noise=0,\
                 correct=True, scale=None, g=identity, height=21, rotation=None,\
                 translation=None, normalize=True, norm_factor=None, gamma=0.5,\
                 **kwargs):
        """constructor for class containing a random swiss roll"""

        self._genattrs = genattrs
        self._specattrs = specattrs

        if not isinstance(genattrs, GeneralManifoldAttrs):
            self._genattrs = GeneralManifoldAttrs(N=N, num_neg=num_neg,\
                 n=n, k=k, D=D, max_norm=max_norm, mu=mu, sigma=sigma,\
                 seed=seed, normalize=normalize, norm_factor=norm_factor,\
                 gamma=gamma, rotation=rotation, translation=translation)

        if not isinstance(specattrs, SpecificSwissRollAttrs):
            self._specattrs = SpecificSwissRollAttrs(mu=self._genattrs.mu,\
                 sigma=self._genattrs.sigma, n=self._genattrs.n, seed=self._genattrs.seed, 
                 g=g, t_min=t_min, t_max=t_max, omega=omega, num_turns=num_turns,\
                 noise=noise, correct=correct, scale=scale, height=height)

        if self._genattrs.max_norm is None:
            self._genattrs._max_norm = self._specattrs.gap / 2
            assert self._genattrs._max_norm == self._genattrs.max_norm

        elif self._genattrs.max_norm > self._specattrs.gap:
            raise RuntimeError("max_norm > gap (2 * np.pi / omega). Please revise values!")


        ## setting seed
        torch.manual_seed(self._genattrs.seed)
        np.random.seed(self._genattrs.seed)

        # self.compute_points()

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
        # return self._genattrs.points_n[idx], self._genattrs.distances[idx]
        return {
            "points_n": self._genattrs.points_n[idx],
            "distances": self._genattrs.distances[idx],
            "actual_distances": self._genattrs.actual_distances[idx],
            "normed_points_n": self._genattrs.normed_points_n[idx],
            "normed_distances": self._genattrs.normed_distances[idx],
            "normed_actual_distances": self._genattrs.normed_actual_distances[idx],
            "t": self._specattrs.t[idx]
        }

    def gen_points(self):
        """
            Making your own swiss roll in k-dim

            references:
            - https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/datasets/_samples_generator.py#L1401
            - https://jlmelville.github.io/smallvis/swisssne.html 
        """

        t_min = self._specattrs.t_min
        t_max = self._specattrs.t_max
        num_samples = self._genattrs.N - self._genattrs.num_neg
        omega = self._specattrs.omega
        noise = self._specattrs.noise
        g = self._specattrs.g

        k = self._genattrs.k        

        t = t_min + (np.random.rand(1, num_samples) * (t_max - t_min))
    
        x = g(t) * np.cos(omega * t)
        y = g(t) * np.sin(omega * t)
        zs = self._specattrs.height * np.random.uniform(size=(k-2, num_samples))

        X = np.concatenate((x, y, zs), axis=0)
        X += noise * np.random.randn(k, num_samples)
        X = X.T

        t = np.squeeze(t)

        self._genattrs.points_k = X
        self._specattrs.t = t

    def gen_pre_images(self):

        """
            Making your own swiss roll in k-dim

            references:
            - https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/datasets/_samples_generator.py#L1401
            - https://jlmelville.github.io/smallvis/swisssne.html 
        """

        t_min = self._specattrs.t_min
        t_max = self._specattrs.t_max
        num_samples = self._genattrs.num_neg
        omega = self._specattrs.omega
        noise = self._specattrs.noise
        g = self._specattrs.g

        k = self._genattrs.k        

        t = t_min + (np.random.rand(1, num_samples) * (t_max - t_min))
    
        x = g(t) * np.cos(omega * t)
        y = g(t) * np.sin(omega * t)
        zs = self._specattrs.height * np.random.uniform(size=(k-2, num_samples))
        
        X = np.concatenate((x, y, zs), axis=0)
        
        X += noise * np.random.randn(k, num_samples)
        X = X.T
        
        t = np.squeeze(t)
        
        self._genattrs.pre_images_k = X
        self._specattrs.pre_images_t = t

    def compute_normals(self):
        """
        analytical expression for the swiss roll is given by:

        $$
        \\vector{v} = 
                    \[
                        g(t) cos(\pi t) \\
                        g(t) sin(\pi t) \\
                            \psi
                    \]
        $$

        In order to find the normal direction to this surface, we first find
        $\partial{\vector{v}}{t}$ and $\partial{\vector{v}}{psi}$ and then take
        their cross-products
        """
        
        g = self._specattrs.g
        d_g = self._specattrs.d_g
        t = self._specattrs.pre_images_t
        omega = self._specattrs.omega
        N = t.shape[0]
        k = self._genattrs.k

        # components of partial of v w.r.t t (apart from x, y all other components are 0!)
        dv_t_x =  (d_g(t) * np.cos(omega * t)) - (omega * g(t) * np.sin(omega * t))
        dv_t_y =  (d_g(t) * np.sin(omega * t)) + (omega * g(t) * np.cos(omega * t))
        
        # taking cross product of dv_t and dv_psi, the net result is as given below
        normal_vectors_to_mfld_at_p = np.zeros((N, k))
        normal_vectors_to_mfld_at_p[:, 0] = dv_t_y
        normal_vectors_to_mfld_at_p[:, 1] = -1 * dv_t_x

        # scaling to self._specattrs.max_norm for ease of visualisation
        scaling_factor = (self._genattrs.max_norm / (np.linalg.norm(normal_vectors_to_mfld_at_p, axis=1, ord=2))).reshape(-1, 1)
        normal_vectors_to_mfld_at_p = scaling_factor * normal_vectors_to_mfld_at_p
    
        embedded_normal_vectors_to_mfld_at_p = np.zeros((self._genattrs.num_neg, self._genattrs.n))
        embedded_normal_vectors_to_mfld_at_p[:, :self._genattrs.k] = normal_vectors_to_mfld_at_p

        return embedded_normal_vectors_to_mfld_at_p

    def make_off_mfld_eg(self):
        return super().make_off_mfld_eg()

    def embed_in_n(self):
        
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
        self._genattrs.actual_distances[self._genattrs.num_neg:] = 0
        self._genattrs.distances = np.clip(self._genattrs.actual_distances, a_min=0, a_max=self._genattrs.D)
        
    def norm(self):
        """normalise points and distances so that the whole setup lies in a unit sphere"""
        if self._genattrs.norm_factor is None:
            min_coord = torch.min(self._genattrs.points_n).item()
            max_coord = torch.max(self._genattrs.points_n).item()
            # NOTE: using `_norm_factor` to set `norm_factor` in `self._genattrs`. DO NOT MAKE THIS A HABIT!!!
            self._genattrs._norm_factor = max_coord - min_coord
            assert self._genattrs.norm_factor == self._genattrs._norm_factor
        
        self._genattrs.normed_points_n = self._genattrs.points_n / self._genattrs.norm_factor
        self._genattrs.normed_distances = self._genattrs.distances / self._genattrs.norm_factor
        self._genattrs.normed_actual_distances = self._genattrs.actual_distances / self._genattrs.norm_factor

        # change anchor point to bring it closer to origin (smaller numbers are easier to learn)
        tmp = self._genattrs.gamma if self._genattrs.gamma is not None else 1
        self._genattrs.fix_center = tmp * np.ones(self._genattrs.n)
        anchor = self._genattrs.normed_points_n[np.argmin(self._specattrs.t)]
        self._genattrs.normed_points_n = self._genattrs.normed_points_n - anchor + self._genattrs.fix_center

    def invert_points(self, normed_points):
        """invert normalised points to unnormalised values"""
        anchor = self._genattrs.normed_points_n[np.argmin(self._specattrs.t)]
        normed_points = normed_points - self._genattrs.fix_center + anchor
        return normed_points * self._genattrs.norm_factor

    def invert_distances(self, normed_distances):
        """invert normalised distances to unnormalised values"""
        return normed_distances * self._genattrs.norm_factor

    def compute_points(self):
        
        self._specattrs.setup()
        print("[RandomSwissRoll]: swiss roll attribute setup done")
        self.gen_points()
        print("[RandomSwissRoll]: generated points in k-dim")
        self.gen_pre_images()
        print("[RandomSwissRoll]: pre-images generated")
        self.embed_in_n()
        print("[RandomSwissRoll]: embedded the sphere in n-dim space")

        self._genattrs.points_n = torch.from_numpy(self._genattrs.points_n).float()
        self._genattrs.points_k = torch.from_numpy(self._genattrs.points_k).float()
        self._genattrs.distances = torch.from_numpy(self._genattrs.distances).float()
        self._genattrs.actual_distances = torch.from_numpy(self._genattrs.actual_distances).float()
        
        if self._genattrs.normalize:
            self.norm()
            print("[RandomSwissRoll]: normalization complete")

    def viz_test(self, dimX=0, dimY=1, dimZ=2, num_pre_img=5):
        """
            generate plots that act as visual sanity checks
            (only effective for `self._genattrs.n = 3`)
            
            1. distribution of clamped distances from manifold
            2. distribution of actual distances from manifold
            3. plotting 3D projection of the points
            4. plotting 2D projection of the points (as it would look with
               camera looking at X-Y plane from the Z-axis)
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

        
        on_mfld_idx = self._genattrs.distances.reshape(-1) == 0
        off_mfld_idx = self._genattrs.distances.reshape(-1) != 0

        plt.figure(figsize=(6,6))
        plt.scatter(self._genattrs.points_n_trivial_[on_mfld_idx, dimX],\
                    self._genattrs.points_n_trivial_[on_mfld_idx, dimY],\
                    label="on-manifold", s=0.01)
        plt.scatter(self._genattrs.points_n_trivial_[off_mfld_idx, dimX],\
                    self._genattrs.points_n_trivial_[off_mfld_idx, dimY],\
                    label="off-manifold", s=0.01)

        # plt.scatter(self._genattrs.pre_images_k[:, dimX],\
        #             self._genattrs.pre_images_k[:, dimY],\
        #             label="on-manifold", s=0.01)
        

        random_idx = np.random.choice(np.arange(self._genattrs.num_neg), 10)

        embedded_normals_at_p = self.compute_normals()
        
        for i in range(random_idx.shape[0]):
            j = random_idx[i]
            plt.arrow(self._genattrs.pre_images_k[j, dimX],\
                self._genattrs.pre_images_k[j, dimY],\
                embedded_normals_at_p[j, dimX],\
                embedded_normals_at_p[j, dimY],\
                head_width=0.1, width=0.01)
        
        plt.xlabel("dimX = {}".format(dimX))
        plt.ylabel("dimY = {}".format(dimY))
        plt.legend(markerscale=100)
        plt.title("2D projection of on-/off-manifold points with some normals shown")
        plt.show()

        

        if self._genattrs.n >= 3:
            
            fig = plt.figure(figsize = (10, 7))
            ax1 = plt.axes(projection ="3d")

            ax1.scatter3D(self._genattrs.points_n_trivial_[on_mfld_idx, dimX],\
                        self._genattrs.points_n_trivial_[on_mfld_idx, dimY],\
                        self._genattrs.points_n_trivial_[on_mfld_idx, dimZ])
            plt.title("on-manifold samples (trivial embedding)")
            plt.show()

             # before rotation
            fig = plt.figure(figsize = (10, 7))
            ax1 = plt.axes(projection ="3d")
            
            ax1.scatter3D(self._genattrs.points_n_tr_[on_mfld_idx, dimX],\
                        self._genattrs.points_n_tr_[on_mfld_idx, dimY],\
                        self._genattrs.points_n_tr_[on_mfld_idx, dimZ])
            
            plt.title("on-manifold samples (after translation)")
            plt.show()
            
            # after rotation
            fig = plt.figure(figsize = (10, 7))
            ax1 = plt.axes(projection ="3d")
            
            ax1.scatter3D(self._genattrs.points_n[on_mfld_idx, dimX],\
                        self._genattrs.points_n[on_mfld_idx, dimY],\
                        self._genattrs.points_n[on_mfld_idx, dimZ])
            
            plt.title("on-manifold samples (after rotation)")
            plt.show()

            # plots for off-manifold samples

            #indices to show for pre-images
            idx = np.random.choice(np.arange(self._genattrs.num_neg), num_pre_img)
            
            # trivial pre_image
            
            neg_pre_img = self._genattrs.pre_images_k[idx]
            neg_pre_img_trivial_ = np.zeros((num_pre_img, self._genattrs.n))
            neg_pre_img_trivial_[:, :self._genattrs.k] = neg_pre_img

            fig = plt.figure(figsize = (10, 7))
            ax1 = plt.axes(projection ="3d")
            
            ax1.scatter3D(self._genattrs.points_n_trivial_[on_mfld_idx, dimX],\
                self._genattrs.points_n_trivial_[on_mfld_idx, dimY],\
                self._genattrs.points_n_trivial_[on_mfld_idx, dimZ], color="blue", label="on-manifold", s=0.1, marker="1")
            
            ax1.scatter3D(neg_pre_img_trivial_[:, dimX],\
                        neg_pre_img_trivial_[:, dimY],\
                        neg_pre_img_trivial_[:, dimZ],\
                        color="green", marker="^", label="trivial pre-image", s=80)

            ax1.scatter3D(self._genattrs.points_n_trivial_[off_mfld_idx, dimX][idx],\
                        self._genattrs.points_n_trivial_[off_mfld_idx, dimY][idx],\
                        self._genattrs.points_n_trivial_[off_mfld_idx, dimZ][idx], color="red", label="off-manifold")
            
            actual_distances_trivial_ = np.linalg.norm(neg_pre_img_trivial_\
                                - self._genattrs.points_n_trivial_[idx], ord=2, axis=1)
            
            for i in range(len(idx)):
                j = idx[i]
                ax1.plot([neg_pre_img_trivial_[:, dimX][i], self._genattrs.points_n_trivial_[off_mfld_idx, dimX][j]],\
                        [neg_pre_img_trivial_[:, dimY][i], self._genattrs.points_n_trivial_[off_mfld_idx, dimY][j]],\
                        [neg_pre_img_trivial_[:, dimZ][i], self._genattrs.points_n_trivial_[off_mfld_idx, dimZ][j]], color="black")
                ax1.text(np.mean([neg_pre_img_trivial_[:, dimX][i], self._genattrs.points_n_trivial_[off_mfld_idx, dimX][j]]),\
                        np.mean([neg_pre_img_trivial_[:, dimY][i], self._genattrs.points_n_trivial_[off_mfld_idx, dimY][j]]),\
                        np.mean([neg_pre_img_trivial_[:, dimZ][i], self._genattrs.points_n_trivial_[off_mfld_idx, dimZ][j]]),\
                        "{:.2f}".format(actual_distances_trivial_[i].item()))

            plt.legend()
            plt.title("pre-image samples (trivial embedding)")
            plt.show()
            

            # translated pre_image

            neg_pre_img_tr_ = neg_pre_img_trivial_ + self._genattrs.translation
            
            fig = plt.figure(figsize = (10, 7))
            ax1 = plt.axes(projection ="3d")
            
            ax1.scatter3D(self._genattrs.points_n_tr_[on_mfld_idx, dimX],\
                self._genattrs.points_n_tr_[on_mfld_idx, dimY],\
                self._genattrs.points_n_tr_[on_mfld_idx, dimZ], color="blue", label="on-manifold", s=1, marker="1")
            
            
            ax1.scatter3D(neg_pre_img_tr_[:, dimX],\
                        neg_pre_img_tr_[:, dimY],\
                        neg_pre_img_tr_[:, dimZ], color="green", marker="^", label="translated pre-image", s=80)
            
            
            
            ax1.scatter3D(self._genattrs.points_n_tr_[off_mfld_idx, dimX][idx],\
                        self._genattrs.points_n_tr_[off_mfld_idx, dimY][idx],\
                        self._genattrs.points_n_tr_[off_mfld_idx, dimZ][idx], color="red", label="off-manifold")
            
            for i in range(len(idx)):
                j = idx[i]
                ax1.plot([neg_pre_img_tr_[:, dimX][i], self._genattrs.points_n_tr_[off_mfld_idx, dimX][j]],\
                        [neg_pre_img_tr_[:, dimY][i], self._genattrs.points_n_tr_[off_mfld_idx, dimY][j]],\
                        [neg_pre_img_tr_[:, dimZ][i], self._genattrs.points_n_tr_[off_mfld_idx, dimZ][j]], color="black")
                ax1.text(np.mean([neg_pre_img_tr_[:, dimX][i], self._genattrs.points_n_tr_[off_mfld_idx, dimX][j]]),\
                        np.mean([neg_pre_img_tr_[:, dimY][i], self._genattrs.points_n_tr_[off_mfld_idx, dimY][j]]),\
                        np.mean([neg_pre_img_tr_[:, dimZ][i], self._genattrs.points_n_tr_[off_mfld_idx, dimZ][j]]),\
                        "{:.2f}".format(self._genattrs.actual_distances[i].item()))

            plt.legend()
            plt.title("pre-image samples (translated embedding)")
            plt.show()
            


            # rotated pre_image
           
            neg_pre_img_rot_ = np.dot(self._genattrs.rotation, neg_pre_img_tr_.T).T
            
            fig = plt.figure(figsize = (10, 7))
            ax1 = plt.axes(projection ="3d")
            
            ax1.scatter3D(self._genattrs.points_n_rot_[on_mfld_idx, dimX],\
                self._genattrs.points_n_rot_[on_mfld_idx, dimY],\
                self._genattrs.points_n_rot_[on_mfld_idx, dimZ], color="blue", label="on-manifold", s=1, marker="1")
            
            
            ax1.scatter3D(neg_pre_img_rot_[:, dimX],\
                        neg_pre_img_rot_[:, dimY],\
                        neg_pre_img_rot_[:, dimZ], color="green", marker="^", label="rotated pre-image", s=80)
            
            
            
            ax1.scatter3D(self._genattrs.points_n_rot_[off_mfld_idx, dimX][idx],\
                        self._genattrs.points_n_rot_[off_mfld_idx, dimY][idx],\
                        self._genattrs.points_n_rot_[off_mfld_idx, dimZ][idx], color="red", label="off-manifold")
            
            for i in range(len(idx)):
                j = idx[i]
                ax1.plot([neg_pre_img_rot_[:, dimX][i], self._genattrs.points_n_rot_[off_mfld_idx, dimX][j]],\
                        [neg_pre_img_rot_[:, dimY][i], self._genattrs.points_n_rot_[off_mfld_idx, dimY][j]],\
                        [neg_pre_img_rot_[:, dimZ][i], self._genattrs.points_n_rot_[off_mfld_idx, dimZ][j]], color="black")
                ax1.text(np.mean([neg_pre_img_rot_[:, dimX][i], self._genattrs.points_n_rot_[off_mfld_idx, dimX][j]]),\
                        np.mean([neg_pre_img_rot_[:, dimY][i], self._genattrs.points_n_rot_[off_mfld_idx, dimY][j]]),\
                        np.mean([neg_pre_img_rot_[:, dimZ][i], self._genattrs.points_n_rot_[off_mfld_idx, dimZ][j]]),\
                        "{:.2f}".format(self._genattrs.actual_distances[i].item()))
            
            
            plt.legend()
            plt.title("pre-image samples (rotated embedding)")
            plt.show()

            min_dist_vals = list()
        
            for i in idx:
                
                neg_ex = self._genattrs.points_n[i]
                min_dist = None
                
                for j in range(self._genattrs.num_neg):

                    a_pre_img = self._genattrs.pre_images_k[j]
                    tmp = np.zeros(self._genattrs.n)
                    tmp[:self._genattrs.k] = a_pre_img
                    a_pre_img = tmp
                    a_pre_img = a_pre_img + self._genattrs.translation
                    a_pre_img = np.dot(self._genattrs.rotation, a_pre_img)

                    dist = np.linalg.norm(neg_ex - a_pre_img, ord=2)
                    
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        
                min_dist_vals.append(min_dist)
                
            errors = np.abs(self._genattrs.actual_distances[idx].numpy().reshape(-1) - np.array(min_dist_vals))
            rel_errors = errors / self._genattrs.actual_distances[idx].numpy().reshape(-1)
            print("absolute errors:", errors)
            print("relative errors:", rel_errors)
            
        
    def load_data(self, dump_dir):

        specs_fn = os.path.join(dump_dir, "specs.json")
        data_fn = os.path.join(dump_dir, "data.pkl")

        with open(specs_fn) as f:
            specs_attrs = json.load(f)

        data_attrs = torch.load(data_fn)

        attrs = {**specs_attrs, **data_attrs}

        self._genattrs = GeneralManifoldAttrs()
        self._specattrs = SpecificSwissRollAttrs()

        for attr_set in [self._genattrs, self._specattrs]:
            for attr in vars(attr_set):
                if attr in attrs:
                    if attr == "_g" or attr == "_d_g":
                        attrs[attr] = eval(attrs[attr])
                    setattr(attr_set, attr, attrs[attr])
                    
                    

    def save_data(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        specs_fn = os.path.join(save_dir, "specs.json")
        data_fn = os.path.join(save_dir, "data.pkl")

        specs_attrs = dict()
        data_attrs = dict()

        gen_attrs = vars(self._genattrs)
        swissroll_attrs = vars(self._specattrs)

        for attr_set in [gen_attrs, swissroll_attrs]:
            for attr in attr_set:
        
                if not isinstance(attr_set[attr], Iterable):
                    specs_attrs[attr] = attr_set[attr]
                    if attr == "_g" or attr == "_d_g":
                        specs_attrs[attr] = inspect.getsourcelines(specs_attrs[attr])[0][0].split("=")[1].strip()
                else:
                    data_attrs[attr] = attr_set[attr]

        with open(specs_fn, "w+") as f:
            json.dump(specs_attrs, f)

        torch.save(data_attrs, data_fn)

if __name__ == '__main__':

    dummy_params = {

        "N": 100000,
        "num_neg": None,
        "n": 3,
        "k": 2,
        "D": 0.2,
        "max_norm": 1.5,
        "mu": 0,
        "sigma": 1,
        "seed": 42,
        "gamma": 1,
        "t_min": 1.5,
        "t_max": 4.5,
        "num_turns": None,
        "omega": np.pi 

    }

    test = RandomSwissRoll(**dummy_params)
    test.compute_points()
    test.save_data("./test")
    b = RandomSwissRoll()
    b.load_data("./test")



        

        


    
        
        


    
    
    

    

    





        
        
