import os
import sys
import time
import copy
import json
import argparse

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets

import PIL

import matplotlib
from mpl_toolkits import mplot3d
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

from livelossplot import PlotLosses

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import IncrementalPCA

from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, classification_report

from tqdm import tqdm

from ptcifar.models import ResNet18



class RandomSphere(Dataset):
    
    """
    Class for generating dataset of a random
    sphere lying in a low dimensional manifold
    embedded in a higher dimensional space
    """
    
    def __init__(self, N=1000, num_neg=None, n=100, k=3, r=10.0,\
                 D=50.0, max_norm=2.0, mu=10, sigma=5, seed=42,\
                 x_ck=None, rotation=None, translation=None, normalize=True,\
                 norm_factor=None, gamma=0.5):
        
        """
        :param N: total number of samples
        :type N: int
        :param k: low (k-1)-dimensional manifold, embedded in k dims 
        :type k: int
        :param n: dimension of manifold in which sphere is embedded
        :type n: int
        :param r: radius of the sphere
        :type r: float
        :param D: clamping limit for negative examples
        :type D: float
        :param max_norm: maximum possible distance of point from manifold can be `r / max_norm`
        :type max_norm: float
        :param mu: mean of normal distribution from which we sample
        :type: float
        :param sigma: standard deviation of normal distribution from which we sample
        :type: float
        :param seed: random seed (default is the answer to the ultimate question!)
        :type: int
        :param x_ck: center of lower dimensional manifold
        :type numpy.array:
        :param normalize: whether to normalize the dataset or not
        :type: bool
        :param rotation: rotation matrix to be used
        :type numpy.ndarray:
        :param translation: translation vector to be used
        :type numpy.array:
        :param norm_factor: factor by which to normalise the coordinates
        :type float:
        :param gamma: conservative scale used in normalisation, only used if `norm_factor` is not provided
        :type float:
        """
        
        self.N = N
        self.num_neg = np.floor(self.N / 2).astype(np.int64)
        if num_neg is not None:
            self.num_neg = num_neg
        self.n = n
        self.k = k
        self.r = r
        self.D = D
        self.max_norm = max_norm
        self.mu = mu
        self.sigma = sigma
        self.seed = seed
        
        ## setting seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.x_ck = None
        """center of the sphere"""
        if x_ck is not None:
            self.x_ck = x_ck
        
        self.x_cn_trivial_ = None
        """center of the sphere in higher dimension after trivial embedding"""
        
        self.x_cn_tr_ = None
        """center of the sphere in higher dimension after translation"""
        
        self.x_cn_rot_ = None
        """center of the sphere in higher dimension after translation and rotation"""
        
        self.x_cn = None
        """center of sphere in higher dimension"""
        
        self.points_k = None
        """points sampled from the sphere in k-dim"""
        
        self.points_n_trivial_ = None
        """sampled points in higher dimension after trivial embedding"""
        
        self.points_n_tr_ = None
        """sampled points in higher dimension after translation"""
        
        self.points_n_rot_ = None
        """sampled points in higher dimension after translation and rotation"""
        
        self.points_n = None
        """embedding of `self.points_k` in n-dim"""
        
        self.actual_distances = None
        """actual distance of points from the sphere's surface"""
        
        self.distances = None
        """clamped distance of the point from the sphere's surface"""
        
        self.translation = None
        """randomly sampled translation transform"""
        if translation is None:
            self.translation = np.random.normal(self.mu, self.sigma, self.n)
        else:
            self.translation = translation
        
        self.rotation = None
        """randomly sampled rotation transform"""
        if rotation is None:
            self.rotation = np.random.normal(self.mu, self.sigma, (self.n, self.n))
            self.rotation = np.linalg.qr(self.rotation)[0]
        else:
            self.rotation = rotation
        
        self.pre_images_k = None
        """k-dimensional on-manifold pre-images of the off-manifold points"""

        self.normed_points_n = self.points_n
        """points normalized so that they lie in a unit cube"""
        self.normed_x_cn = self.x_cn
        """center normalized to lie in a unit cube"""
        self.normed_distances = self.distances
        """normalised distances"""
        self.normed_actual_distances = self.actual_distances
        """actual normalised distances"""
        self.norm_factor = norm_factor
        """factor by which to normalise (use for inverting)"""
        self.gamma = gamma 
        """conservative factor used in normalisation"""
        self.fix_center = self.x_cn
        """center after re-positioning normalised dataset"""
        
        self.gen_center()
        print("center generated")
        self.gen_points()
        print("points generated")
        self.gen_pre_images()
        print("pre-images generated")
        self.embed_in_n()
        print("embedding done")
#         self.compute_distances()
        
        self.points_n = torch.from_numpy(self.points_n).float()
        self.points_k = torch.from_numpy(self.points_k).float()
        self.distances = torch.from_numpy(self.distances).float()
        self.actual_distances = torch.from_numpy(self.actual_distances).float()

        self.normalize = normalize

        if self.normalize:
            self.normalization()
            print("normalization complete")

        
        
    def __len__(self):
        return self.points_n.shape[0]
    
    def __getitem__(self, idx):
        # return self.points_n[idx], self.distances[idx]
        return {
            "points_n": self.points_n[idx],
            "distances": self.distances[idx],
            "actual_distances": self.actual_distances[idx],
            "normed_points_n": self.normed_points_n[idx],
            "normed_distances": self.normed_distances[idx],
            "normed_actual_distances": self.normed_actual_distances[idx]
        }
        

    def gen_center(self):
        """generate a center in lower dimension"""
        if self.x_ck is not None:
            return
        self.x_ck = np.random.normal(self.mu, self.sigma, self.k)
        
    def gen_points(self):
        """
        
            generating points in k-dim and embedding in n-dim

            reference: https://en.wikipedia.org/wiki/N-sphere#Uniformly_at_random_on_the_(n_%E2%88%92_1)-sphere
            
        """
        points_k = np.random.normal(size=(self.N - self.num_neg, self.k))
        
        norms = np.linalg.norm(points_k, axis=1, ord=2).reshape(-1, 1)
        points_k = (points_k / norms)
        
#         print(np.round(np.linalg.norm(points_k, axis=1, ord=2))[np.ceil(np.linalg.norm(points_k, axis=1, ord=2)) > 1])
        
        assert (np.round(np.linalg.norm(points_k, axis=1, ord=2)) == 1).all()
        
        points_k = self.r * points_k
        
#         neg_norms = np.random.uniform(low=1 + np.finfo(np.float).eps,\
#                                       high=self.max_norm, size=np.floor(self.N / 2).astype(np.int64))
        
#         points_k[:np.floor(self.N / 2).astype(np.int64)] = (neg_norms.reshape(-1, 1) / self.r) * points_k[:np.floor(self.N / 2).astype(np.int64)]
        
        points_k = points_k + self.x_ck
        
        self.points_k = points_k

    def gen_pre_images(self):
        """generating on-manifold k-dimensional projections of off-maniofld samples"""

        points_k = np.random.normal(size=(self.num_neg, self.k))
        
        
        norms = np.linalg.norm(points_k, axis=1, ord=2).reshape(-1, 1)
        points_k = (points_k / norms)
        
#         print(np.round(np.linalg.norm(points_k, axis=1, ord=2))[np.ceil(np.linalg.norm(points_k, axis=1, ord=2)) > 1])
        
        assert (np.round(np.linalg.norm(points_k, axis=1, ord=2)) == 1).all()
        
        points_k = self.r * points_k
        
#         neg_norms = np.random.uniform(low=1 + np.finfo(np.float).eps,\
#                                       high=self.max_norm, size=np.floor(self.N / 2).astype(np.int64))
        
#         points_k[:np.floor(self.N / 2).astype(np.int64)] = (neg_norms.reshape(-1, 1) / self.r) * points_k[:np.floor(self.N / 2).astype(np.int64)]
        
        points_k = points_k + self.x_ck
        
        self.pre_images_k = points_k
        
    def make_neg_examples(self):
        """generating negative examples, i.e., points not on the manifold"""
        
        # normal_vectors_to_mfld_at_p are actually centred at x_ck, but 
        # we can imaging the same vector at $p$, and later adjust the coordinates
        # by adding the position vector of $p$ back.
        #
        # Also note that these negative examples are being generated using first
        # half of self.points_k
        
        normal_vectors_to_mfld_at_p = self.pre_images_k - self.x_ck
        embedded_normal_vectors_to_mfld_at_p = np.zeros((self.num_neg, self.n))
        embedded_normal_vectors_to_mfld_at_p[:, :self.k] = normal_vectors_to_mfld_at_p
        
        neg_examples = np.zeros((self.num_neg, self.n))
        neg_distances = np.zeros(self.num_neg)
        
        # canonical basis $e_i$ over leftover dimensions
        remaining_dims = self.n - self.k
        leftover_basis = np.eye(remaining_dims)
        
        # variable storing the remaining spanning set apart from the normal
        remaining_span_set = np.zeros((remaining_dims, self.n))
        remaining_span_set[:, self.k:] = leftover_basis

        # coefficients for the remaining bases vectors
        remaining_coefficients = np.random.normal(self.mu, self.sigma, size=(self.num_neg, self.n))
        # sum of the remaning span set
        sum_span_set = np.sum(remaining_span_set, axis=0)
        # taking advantage of the standard basis, we can form convert the sum to a linear combination
        remaining_linear_combination = remaining_coefficients * sum_span_set

        # coefficients to multiply the normals
        first_coefficients = np.random.normal(self.mu, self.sigma, size=(self.num_neg, 1))
        weighted_normals = first_coefficients * embedded_normal_vectors_to_mfld_at_p
        
        neg_examples = weighted_normals + remaining_linear_combination

        """
        # variable to store spanning set
        span_set = np.zeros((remaining_dims + 1, self.n))
        span_set[1:, self.k:] = leftover_basis
        
        
        for i in tqdm(range(self.num_neg)):

            n_cap = embedded_normal_vectors_to_mfld_at_p[i]
            
            # span set is n_cap and leftover basis
            span_set[0] = n_cap
            

            # sample random coefficients
            # coefficients = np.random.uniform(1 + np.finfo(np.float).eps, self.max_norm, span_set.shape[0])
            coefficients = np.random.normal(self.mu, self.sigma, span_set.shape[0])

            # take linear combination
            neg_examples[i] = np.sum(coefficients.reshape(-1, 1) * span_set, axis=0)
        """

        # re-scale with random norms, sampled from U[\epsilon, self.r/ self.max_norm]
        # neg_norms = np.random.uniform(low=0, high=0, size=self.num_neg)
        neg_norms = np.random.uniform(low=1e-6 + np.finfo(np.float).eps, high=self.r / self.max_norm, size=self.num_neg)
        # neg_norms = np.random.normal(self.mu, self.sigma, size=self.num_neg)
        
        neg_examples = (neg_norms.reshape(-1, 1) / np.linalg.norm(neg_examples, axis=1, ord=2).reshape(-1, 1)) * neg_examples

        # add position vector of $p$ to get origin centered coordinates
        # neg_examples[:, :self.k] = neg_examples[:, :self.k] + self.points_k[self.off_mfld_idx]
        neg_examples[:, :self.k] = neg_examples[:, :self.k] + self.pre_images_k

        # distances from the manifold will be the norms the samples were rescaled by
        neg_distances = neg_norms
        
        return neg_examples, neg_distances
            
    
    
    def embed_in_n(self):
        """embedding center and sampled points in `self.n`-dims"""
        
        # embedding the center
        self.x_cn_trivial_ = np.zeros(self.n)
        self.x_cn_trivial_[:self.k] = self.x_ck
        self.x_cn_tr_ = self.x_cn_trivial_ + self.translation
        self.x_cn_rot_ = np.dot(self.rotation, self.x_cn_tr_)
        self.x_cn = self.x_cn_rot_
        
        
        # generate the negative examples
        neg_examples, neg_distances = self.make_neg_examples()
        
        #embedding the points
        self.points_n_trivial_ = np.zeros((self.N, self.n))
        self.points_n_trivial_[:self.num_neg] = neg_examples
        
        self.points_n_trivial_[self.num_neg:, :self.k] = self.points_k
        self.points_n_tr_ = self.points_n_trivial_ + self.translation
        
        self.points_n_rot_ = np.dot(self.rotation, self.points_n_tr_.T).T
        
        self.points_n = self.points_n_rot_
        
        self.actual_distances = np.zeros((self.N, 1))
        self.actual_distances[:self.num_neg] = neg_distances.reshape(-1, 1)
        self.actual_distances[self.num_neg:] = np.linalg.norm(self.points_n[self.num_neg:] - self.x_cn, axis=1, ord=2).reshape(-1, 1) - self.r
        self.distances = np.clip(self.actual_distances, a_min=0, a_max=self.D)
        
        # checking that the on-manifold points are still self.r away from center
        print(np.round(np.linalg.norm(self.points_n[self.num_neg:] - self.x_cn, axis=1, ord=2)))
        # assert (np.round(np.linalg.norm(self.points_n[self.on_mfld_idx] - self.x_cn, axis=1, ord=2)) == self.r).all()
    
    def invert_distances(self, normed_distances):
        """invert normalised distances to unnormalised values"""
        return normed_distances * self.norm_factor

    def invert_points(self, normed_points):
        """invert normalised points to unnormalised values"""
        normed_points = normed_points - self.fix_center + self.normed_x_cn
        return normed_points * self.norm_factor

    def normalization(self):
        """normalise points and distances so that the whole setup lies in a unit sphere"""
        if self.norm_factor is None:
            self.norm_factor = self.gamma * np.max(np.linalg.norm(self.points_n - self.x_cn, ord=2, axis=1))
        self.normed_points_n = self.points_n / self.norm_factor
        self.normed_x_cn = self.x_cn / self.norm_factor
        self.normed_distances = self.distances / self.norm_factor
        self.normed_actual_distances = self.actual_distances / self.norm_factor

        # change centre to bring it closer to origin (smaller numbers are easier to learn)
        tmp = self.gamma if self.gamma is not None else 1
        self.fix_center = tmp * np.ones(self.n)
        self.normed_points_n = self.normed_points_n - self.normed_x_cn + self.fix_center
        
        


    def viz_test(self, dimX=0, dimY=1, dimZ=2, num_pre_img=5):
        """
            generate plots that act as visual sanity checks
            
            1. distribution of distance labels from manifold
            2. distribution of actual distances from manifold
            3. on-manifold samples - trivial, translated and rotated in input 3 dims
            4. off-manifold samples - pick one, find its coresponding index in self.points_k,
               transform it to the on-manifold sample, assert that the distance between the two is equal
               to the distance label and plot this distance
        """
        
        plt.hist(self.distances.numpy())
        plt.xlabel("clipped (D={d}) distances from manifold".format(d=self.D))
        plt.ylabel("frequency")
        plt.title("distribution of label distances from the manifold")
        plt.show()
        
        plt.hist(self.actual_distances.numpy())
        plt.xlabel("actual distances from manifold")
        plt.ylabel("frequency")
        plt.title("distribution of actual distances from the manifold")
        plt.show()
        
        # plot the on-manifold examples
        
        # before translation
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self.points_n_trivial_[self.on_mfld_idx, dimX],\
                      self.points_n_trivial_[self.on_mfld_idx, dimY], self.points_n_trivial_[self.on_mfld_idx, dimZ])
        
        plt.title("on-manifold samples (trivial embedding)")
        plt.show()
        
        # before rotation
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self.points_n_tr_[self.on_mfld_idx, dimX],\
                      self.points_n_tr_[self.on_mfld_idx, dimY], self.points_n_tr_[self.on_mfld_idx, dimZ])
        
        plt.title("on-manifold samples (after translation)")
        plt.show()
        
        # after rotation
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self.points_n[self.on_mfld_idx, dimX],\
                      self.points_n[self.on_mfld_idx, dimY], self.points_n[self.on_mfld_idx, dimZ])
        
        plt.title("on-manifold samples (after rotation)")
        plt.show()
        
        
        # plots for off-manifold samples
        
        #indices to show for pre-images
        idx = np.random.choice(np.arange(self.num_neg), num_pre_img)
        
        # trivial pre_image
        
        neg_pre_img = self.points_k[self.off_mfld_idx]
        neg_pre_img_trivial_ = np.zeros((self.num_neg, self.n))
        neg_pre_img_trivial_[:, :self.k] = neg_pre_img
        
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self.points_n_trivial_[self.on_mfld_idx, dimX],\
              self.points_n_trivial_[self.on_mfld_idx, dimY],\
              self.points_n_trivial_[self.on_mfld_idx, dimZ], color="blue", label="on-manifold", s=1, marker="1")
        
        
        
        ax1.scatter3D(neg_pre_img_trivial_[:, dimX][idx],\
                      neg_pre_img_trivial_[:, dimY][idx],\
                      neg_pre_img_trivial_[:, dimZ][idx], color="green", marker="^", label="trivial pre-image", s=80)
        
        
        
        ax1.scatter3D(self.points_n_trivial_[self.off_mfld_idx, dimX][idx],\
                      self.points_n_trivial_[self.off_mfld_idx, dimY][idx],\
                      self.points_n_trivial_[self.off_mfld_idx, dimZ][idx], color="red", label="off-manifold")
        
        actual_distances_trivial_ = np.linalg.norm(neg_pre_img_trivial_\
                               - self.points_n_trivial_[self.off_mfld_idx], ord=2, axis=1)
        
        for i in idx:
            ax1.plot([neg_pre_img_trivial_[:, dimX][i], self.points_n_trivial_[self.off_mfld_idx, dimX][i]],\
                    [neg_pre_img_trivial_[:, dimY][i], self.points_n_trivial_[self.off_mfld_idx, dimY][i]],\
                    [neg_pre_img_trivial_[:, dimZ][i], self.points_n_trivial_[self.off_mfld_idx, dimZ][i]], color="black")
            ax1.text(np.mean([neg_pre_img_trivial_[:, dimX][i], self.points_n_trivial_[self.off_mfld_idx, dimX][i]]),\
                    np.mean([neg_pre_img_trivial_[:, dimY][i], self.points_n_trivial_[self.off_mfld_idx, dimY][i]]),\
                    np.mean([neg_pre_img_trivial_[:, dimZ][i], self.points_n_trivial_[self.off_mfld_idx, dimZ][i]]),\
                    "{:.2f}".format(actual_distances_trivial_[i].item()))
        
        
        plt.legend()
        plt.title("pre-image samples (trivial embedding)")
        plt.show()
        
        plt.hist(np.linalg.norm(neg_pre_img_trivial_\
                               - self.points_n_trivial_[self.off_mfld_idx], ord=2, axis=1) - self.actual_distances[self.off_mfld_idx].numpy().reshape(-1))
        plt.ylabel("freq")
        plt.xlabel("error value")
        plt.title("error distribution for distance value from manifold (trivial)")
        plt.show()
        
        
        # translated pre_image
        
        
        neg_pre_img_tr_ = neg_pre_img_trivial_ + self.translation
        
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self.points_n_tr_[self.on_mfld_idx, dimX],\
              self.points_n_tr_[self.on_mfld_idx, dimY],\
              self.points_n_tr_[self.on_mfld_idx, dimZ], color="blue", label="on-manifold", s=1, marker="1")
        
        
        ax1.scatter3D(neg_pre_img_tr_[:, dimX][idx],\
                      neg_pre_img_tr_[:, dimY][idx],\
                      neg_pre_img_tr_[:, dimZ][idx], color="green", marker="^", label="translated pre-image", s=80)
        
        
        
        ax1.scatter3D(self.points_n_tr_[self.off_mfld_idx, dimX][idx],\
                      self.points_n_tr_[self.off_mfld_idx, dimY][idx],\
                      self.points_n_tr_[self.off_mfld_idx, dimZ][idx], color="red", label="off-manifold")
        
        for i in idx:
            ax1.plot([neg_pre_img_tr_[:, dimX][i], self.points_n_tr_[self.off_mfld_idx, dimX][i]],\
                    [neg_pre_img_tr_[:, dimY][i], self.points_n_tr_[self.off_mfld_idx, dimY][i]],\
                    [neg_pre_img_tr_[:, dimZ][i], self.points_n_tr_[self.off_mfld_idx, dimZ][i]], color="black")
            ax1.text(np.mean([neg_pre_img_tr_[:, dimX][i], self.points_n_tr_[self.off_mfld_idx, dimX][i]]),\
                    np.mean([neg_pre_img_tr_[:, dimY][i], self.points_n_tr_[self.off_mfld_idx, dimY][i]]),\
                    np.mean([neg_pre_img_tr_[:, dimZ][i], self.points_n_tr_[self.off_mfld_idx, dimZ][i]]),\
                    "{:.2f}".format(self.actual_distances[i].item()))
        
        
        plt.legend()
        plt.title("pre-image samples (translated embedding)")
        plt.show()
        
        plt.hist(np.linalg.norm(neg_pre_img_tr_\
                               - self.points_n_tr_[self.off_mfld_idx], ord=2, axis=1) - self.actual_distances[self.off_mfld_idx].numpy().reshape(-1))
        plt.ylabel("freq")
        plt.xlabel("error value")
        plt.title("error distribution for distance value from manifold (tanslated)")
        plt.show()
        
        
        # rotated pre_image
        
        
        neg_pre_img_rot_ = np.dot(self.rotation, neg_pre_img_tr_.T).T
        
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self.points_n_rot_[self.on_mfld_idx, dimX],\
              self.points_n_rot_[self.on_mfld_idx, dimY],\
              self.points_n_rot_[self.on_mfld_idx, dimZ], color="blue", label="on-manifold", s=1, marker="1")
        
        
        ax1.scatter3D(neg_pre_img_rot_[:, dimX][idx],\
                      neg_pre_img_rot_[:, dimY][idx],\
                      neg_pre_img_rot_[:, dimZ][idx], color="green", marker="^", label="rotated pre-image", s=80)
        
        
        
        ax1.scatter3D(self.points_n_rot_[self.off_mfld_idx, dimX][idx],\
                      self.points_n_rot_[self.off_mfld_idx, dimY][idx],\
                      self.points_n_rot_[self.off_mfld_idx, dimZ][idx], color="red", label="off-manifold")
        
        for i in idx:
            ax1.plot([neg_pre_img_rot_[:, dimX][i], self.points_n_rot_[self.off_mfld_idx, dimX][i]],\
                    [neg_pre_img_rot_[:, dimY][i], self.points_n_rot_[self.off_mfld_idx, dimY][i]],\
                    [neg_pre_img_rot_[:, dimZ][i], self.points_n_rot_[self.off_mfld_idx, dimZ][i]], color="black")
            ax1.text(np.mean([neg_pre_img_rot_[:, dimX][i], self.points_n_rot_[self.off_mfld_idx, dimX][i]]),\
                    np.mean([neg_pre_img_rot_[:, dimY][i], self.points_n_rot_[self.off_mfld_idx, dimY][i]]),\
                    np.mean([neg_pre_img_rot_[:, dimZ][i], self.points_n_rot_[self.off_mfld_idx, dimZ][i]]),\
                    "{:.2f}".format(self.actual_distances[i].item()))
        
        
        plt.legend()
        plt.title("pre-image samples (rotated embedding)")
        plt.show()
        
        plt.hist(np.linalg.norm(neg_pre_img_rot_\
                               - self.points_n_rot_[self.off_mfld_idx], ord=2, axis=1) - self.actual_distances[self.off_mfld_idx].numpy().reshape(-1))
        plt.ylabel("freq")
        plt.xlabel("error value")
        plt.title("error distribution for distance value from manifold (rotated)")
        plt.show()
        
        min_dist_vals = list()
        
        for i in idx:
            
            neg_ex = self.points_n[i]
            min_dist = None
            
            for j in range(self.num_neg):
                
                a_pre_img = neg_pre_img_rot_[j]
                dist = np.linalg.norm(neg_ex - a_pre_img, ord=2)
                
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    
            min_dist_vals.append(min_dist)
            
        errors = np.abs(self.actual_distances[idx].numpy().reshape(-1) - np.array(min_dist_vals))
        rel_errors = errors / self.actual_distances[idx].numpy().reshape(-1)
        print("absolute errors:", errors)
        print("relative errors:", rel_errors)
        




class TwoRandomSpheres(Dataset):
    """
        Class containing dataset of two spheres
    """
    def __init__(self, S1_config, S2_config, seed=42, new_translation_for_S2=None,\
                 normalize=True, norm_factor=None, gamma=1.1):
        
        """
        :param S1_config: config for the first sphere
        :type dict: 
        :param S2_config: config for the second sphere
        :type dict: 
        :param seed: random seed for any random operations within TwoRandomSpheres
        :type int: 
        :param new_translation_for_S2: if user wants to define a fixed translation for S2
        :type int:
        :param normalize: flag for normalizing the dataset
        :type bool:
        :param norm_factor: factor by which to normalise the points
        :type float:
        :param gamma: conservative scale used in normalisation, only used if `norm_factor` is not provided
        :type float:
        """
        
        self.S1_config = S1_config
        self.S2_config = S2_config
        self.seed = seed

        if self.S1_config["n"] != self.S2_config["n"]:
            raise RuntimeError("higher dimension size `n` should match for both spheres")
        self.n = self.S1_config["n"]

        # we do not normalize now, since all this will be eventually normalized as a complete dataset
        self.S1 = RandomSphere(**S1_config, normalize=False, gamma=None, norm_factor=None)
        self.S2 = RandomSphere(**S2_config, normalize=False, gamma=None, norm_factor=None)

        # distance between centres for no overlap
        self.c_dist = (self.S1.r + self.S2.r + (self.S1.r / self.S1.max_norm) + (self.S2.r / self.S2.max_norm)) * 1.1

        ## setting seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # shift centre of S2 to origin
        S2_points_n_tmp = self.S2.points_n - self.S2.x_cn
        S2_x_cn_tmp = self.S2.x_cn - self.S2.x_cn

        # sample a new center that is `self.c_dist` from `self.S1.x_cn`
        self.new_translation_for_S2 = new_translation_for_S2
        if new_translation_for_S2 is None:
            new_translation = np.random.normal(self.S2.mu, self.S2.sigma, size=self.S2.n)
            new_translation = new_translation - self.S1.x_cn
            new_translation = (new_translation / np.linalg.norm(new_translation, ord=2)) * self.c_dist
            self.new_translation_for_S2 = new_translation + self.S1.x_cn

        # shift center and points of S2 by `self.new_translation_for_S2`
        self.shifted_S2 = copy.deepcopy(self.S2)
        self.shifted_S2.points_n = S2_points_n_tmp + self.new_translation_for_S2
        self.shifted_S2.x_cn = S2_x_cn_tmp + self.new_translation_for_S2
        self.shifted_S2.translation = self.new_translation_for_S2

        # assert (np.linalg.norm(self.shifted_S2.points_n - self.S1.x_cn, ord=2, axis=1) > self.S1.r + self.S1.D).all()
        # assert (np.linalg.norm(self.S1.points_n - self.shifted_S2.x_cn, ord=2, axis=1) > self.S2.r + self.S2.D).all()

        self.all_points = np.vstack((self.S1.points_n.numpy(), self.shifted_S2.points_n.numpy()))
        
        self.all_distances = np.zeros((self.S1.N + self.shifted_S2.N, 2))
        self.all_distances[:self.S1.N, 0] = self.S1.distances.reshape(-1)
        self.all_distances[:self.S1.N, 1] = self.S1.D
        self.all_distances[self.S1.N:, 1] = self.shifted_S2.distances.reshape(-1)
        self.all_distances[self.S1.N:, 0] = self.shifted_S2.D

        # true distances of points in S1 to S2 and vice versa are not available and marked `-1`
        self.all_actual_distances = np.zeros((self.S1.N + self.shifted_S2.N, 2))
        self.all_actual_distances[:self.S1.N, 0] = self.S1.actual_distances.reshape(-1)
        self.all_actual_distances[:self.S1.N, 1] = -1
        self.all_actual_distances[self.S1.N:, 1] = self.shifted_S2.actual_distances.reshape(-1)
        self.all_actual_distances[self.S1.N:, 0] = -1

        # giving class labels
        # 0: no manifold
        # 1: S_1
        # 2: S_2
        self.class_labels = np.zeros(self.S1.N + self.shifted_S2.N, dtype=np.int64)
        self.class_labels[:self.S1.num_neg] = 0
        self.class_labels[self.S1.num_neg:self.S1.N] = 1
        self.class_labels[self.S1.N:self.S1.N + self.S2.num_neg] = 0
        self.class_labels[self.S1.N + self.S2.num_neg:] = 2

        # shuffling the inputs and targets
        # self.perm = np.random.permutation(self.all_points.shape[0])

        # self.all_points = torch.from_numpy(self.all_points[self.perm]).float()
        # self.all_distances = torch.from_numpy(self.all_distances[self.perm]).float()
        # self.all_actual_distances = torch.from_numpy(self.all_actual_distances[self.perm]).float()
        # self.class_labels = torch.from_numpy(self.class_labels[self.perm]).long()

        self.all_points = torch.from_numpy(self.all_points).float()
        self.all_distances = torch.from_numpy(self.all_distances).float()
        self.all_actual_distances = torch.from_numpy(self.all_actual_distances).float()
        self.class_labels = torch.from_numpy(self.class_labels).long()

        self.normalize = normalize
        self.normed_all_points = self.all_points
        self.normed_all_distances = self.all_distances
        self.normed_all_actual_distances = self.all_actual_distances
        self.norm_factor = norm_factor
        self.gamma = gamma
        self.fix_center = None

        if self.normalize:
            self.normalization()


    def __len__(self):
        return self.all_points.shape[0]
    
    def __getitem__(self, idx):
        return {
            "points": self.all_points[idx],
            "distances": self.all_distances[idx],
            "normed_points": self.normed_all_points[idx],
            "normed_distances": self.normed_all_distances[idx],
            "classes": self.class_labels[idx]
        }

        # return self.all_points[idx], self.all_distances[idx], self.class_labels[idx]

    def normalization(self):
        """
            scales down the points to fit in a unit sphere and moves them closer to origin
        """

        if self.norm_factor is None:
            self.norm_factor = self.gamma * (self.c_dist + self.S1.r + self.S2.r + self.S1.max_norm + self.S2.max_norm)
        self.normed_all_points = self.all_points / self.norm_factor
        self.normed_all_distances = self.all_distances / self.norm_factor
        self.normed_all_actual_distances = self.all_distances / self.norm_factor
    
        # change the coordinates of the central point
        self.S1.normed_x_cn = self.S1.x_cn / self.norm_factor
        self.shifted_S2.normed_x_cn = self.shifted_S2.x_cn / self.norm_factor
        
        # pick a central point for the whole dataset
        self.central_pt = 0.5 * (self.S1.normed_x_cn + self.shifted_S2.normed_x_cn) 
        # self.central_pt = torch.mean(self.normed_all_points, axis=1)
        
        # tmp = 1
        self.fix_center = 0.5 * np.ones(self.S1.n)
        self.normed_all_points = self.normed_all_points - self.central_pt + self.fix_center # shift data so that new `central_pt` lies at `self.fix_center`
        
        # normalize the individual spheres
        self.S1.norm_factor = self.norm_factor
        self.S1.normed_points_n = self.normed_all_points[:self.S1.N, :]
        self.S1.normed_distances = self.normed_all_distances[:self.S1.N]
        self.S1.normed_actual_distances = self.normed_all_actual_distances[:self.S1.N]
        self.S1.normed_x_cn = self.S1.normed_x_cn - self.central_pt + self.fix_center

        self.shifted_S2.norm_factor = self.norm_factor
        self.shifted_S2.normed_points_n = self.normed_all_points[self.S1.N:, :]
        self.shifted_S2.normed_distances = self.normed_all_distances[self.S1.N:]
        self.shifted_S2.normed_actual_distances = self.normed_all_actual_distances[self.S1.N:]
        self.shifted_S2.normed_x_cn = self.shifted_S2.normed_x_cn - self.central_pt + self.fix_center

        self.normed_all_points = self.normed_all_points.float()
        self.normed_all_distances = self.normed_all_distances.float()

    def invert_points(self, unnormed_points):
        unnormed_points = unnormed_points - self.fix_center + self.central_pt
        return self.norm_factor * unnormed_points

    def invert_distances(self, unnormed_distances):
        return self.norm_factor * unnormed_distances
        




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--N", type=int, help="number of samples", default=50000)
    parser.add_argument("--num_neg", type=int, help="number of negative samples", default=None)
    parser.add_argument("--n", type=int, help="dimension of the higher dimensional space", default=3 * 32 * 32)
    parser.add_argument("--k", type=int, help="the manifold lies in the (k-1) dimensional space", default=2)
    parser.add_argument("--r", type=float, help="radius of the k-sphere", default=100.0)
    parser.add_argument("--D", type=float, help="maximum clamp distance", default=25.0)
    parser.add_argument("--max_norm", type=float, help="off-manifold samples are upto a distance of `r / max_norm`")
    parser.add_argument("--mu", type=float, help="mean of the normal distribution from which the points are sampled", default=1000)
    parser.add_argument("--sigma", type=float, help="std. deviation of the normal distribution from which points are sampled", default=5000)
    parser.add_argument("--gamma", type=float, help="scaling factor to use in normalisation", default=0.5)
    
    parser.add_argument("--seed", type=int, help="random seed for generating the dataset", default=42)
    parser.add_argument("--val_seed", type=int, help="different seed for validation set if needed", default=None)
    parser.add_argument("--test_seed", type=int, help="different seed for test set if needed", default=None)


    parser.add_argument("--twospheres", action="store_true", help="generate an instance of the two spheres dataset")
    parser.add_argument("--config", type=str, help="config file to use for data generation", default=None)

    parser.add_argument("--save_dir", type=str, help="directory where to save the dataset", default=None)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)


    if args.twospheres:

        if args.config is None:
            raise Exception("Need to specify config file for TwoSpheres")

        else:
            with open(args.config, "r") as f:
                config = json.load(f)
            
            seed = args.seed

            # train set
            S1_config_train = config["train"]["S1"]
            S2_config_train = config["train"]["S2"]

            train_set = TwoRandomSpheres(S1_config_train, S2_config_train, seed)

            # validation set
            S1_config_val = config["val"]["S1"]
            S1_config_val["x_ck"] = train_set.S1.x_ck
            S1_config_val["translation"] = train_set.S1.translation
            S1_config_val["rotation"] = train_set.S1.rotation

            S2_config_val = config["val"]["S2"]
            S2_config_val["x_ck"] = train_set.S2.x_ck
            S2_config_val["translation"] = train_set.S2.translation
            S2_config_val["rotation"] = train_set.S2.rotation

            val_set = TwoRandomSpheres(S1_config_val, S2_config_val, seed)

            # test set
            S1_config_test = config["test"]["S1"]
            S1_config_test["x_ck"] = train_set.S1.x_ck
            S1_config_test["translation"] = train_set.S1.translation
            S1_config_test["rotation"] = train_set.S1.rotation

            S2_config_test = config["val"]["S2"]
            S2_config_test["x_ck"] = train_set.S2.x_ck
            S2_config_test["translation"] = train_set.S2.translation
            S2_config_test["rotation"] = train_set.S2.rotation

            test_set = TwoRandomSpheres(S1_config_test, S2_config_test, seed)

            

    else:
        
        if args.config is None:
            
            # train set
            train_config = {

                "N": args.N,
                "num_neg": args.num_neg,
                "n": args.n,
                "k": args.k,
                "r": args.r,
                "D": args.D,
                "max_norm": args.max_norm,
                "mu": args.mu,
                "sigma": args.sigma,
                "seed": args.seed,
                "gamma": args.gamma

            }

            train_set = RandomSphere(**train_config)

            # validation set
            val_seed = args.val_seed
            if args.val_seed is None:
                val_seed = args.seed

            val_config = {

                "N": args.N,
                "num_neg": args.num_neg,
                "n": args.n,
                "k": args.k,
                "r": args.r,
                "D": args.D,
                "max_norm": args.max_norm,
                "mu": args.mu,
                "sigma": args.sigma,
                "x_ck": train_set.x_ck,
                "translation": train_set.translation,
                "rotation": train_set.rotation,
                "seed": val_seed,
                "norm_factor": train_set.norm_factor,
                "gamma":train_set.gamma

            }

            val_set = RandomSphere(**val_config)

            # test set
            test_seed = args.test_seed
            if args.val_seed is None:
                test_seed = args.seed

            test_config = {

                "N": args.N,
                "num_neg": args.num_neg,
                "n": args.n,
                "k": args.k,
                "r": args.r,
                "D": args.D,
                "max_norm": args.max_norm,
                "mu": args.mu,
                "sigma": args.sigma,
                "x_ck": train_set.x_ck,
                "translation": train_set.translation,
                "rotation": train_set.rotation,
                "seed": test_seed,
                "gamma": train_set.gamma

            }

            test_set = RandomSphere(**test_config)


        else:
            
            with open(args.config, "r") as f:
                config = json.load(f)

            # train set
            train_config = config["train"]
            train_set = RandomSphere(**train_config)
            
            # validation set
            val_config = config["val"]
            val_config["x_ck"] = train_set.x_ck
            val_config["translation"] = train_set.translation
            val_config["rotation"] = train_set.rotation
            val_config["norm_factor"] = train_set.norm_factor
            val_config["gamma"] = train_set.gamma
            val_set = RandomSphere(**val_config)

            # test set
            test_config = config["test"]
            test_config["x_ck"] = train_set.x_ck
            test_config["translation"] = train_set.translation
            test_config["rotation"] = train_set.rotation
            test_config["norm_factor"] = train_set.norm_factor
            test_config["gamma"] = train_set.gamma
            test_set = RandomSphere(**test_config)

    train_set_fn = os.path.join(args.save_dir, "train_set.pt")
    val_set_fn = os.path.join(args.save_dir, "val_set.pt")
    test_set_fn = os.path.join(args.save_dir, "test_set.pt")

    torch.save(train_set, train_set_fn)
    torch.save(val_set, val_set_fn)
    torch.save(test_set, test_set_fn)
            







    # dummy_params = {
    #     "N": 100000,
    #     "num_neg": None,
    #     "n": 3,
    #     "k": 2,
    #     "r": 100.0,
    #     "D": 25.0,
    #     "max_norm": 500.0,
    #     "mu": 1000,
    #     "sigma": 5000,
    #     "seed": 85
    # }

    # dummy_set = RandomSphere(**dummy_params)
    # dummy_set.viz_test()

    # dummy_params_1 = {
    #     "N": 100000,
    #     "num_neg": None,
    #     "n": 3,
    #     "k": 2,
    #     "r": 100.0,
    #     "D": 25.0,
    #     "max_norm": 2.0,
    #     "mu": 1000,
    #     "sigma": 5000,
    #     "seed": 85
    # }

    # dummy_params_2 = {
    #     "N": 100000,
    #     "num_neg": None,
    #     "n": 3,
    #     "k": 2,
    #     "r": 100.0,
    #     "D": 25.0,
    #     "max_norm": 2.0,
    #     "mu": 1000,
    #     "sigma": 5000,
    #     "seed": 91
    # }

    # dummy_set = TwoRandomSpheres(dummy_params_1, dummy_params_2)

     

    