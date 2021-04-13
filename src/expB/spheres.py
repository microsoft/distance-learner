import os
import sys
import time
import copy

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
                 D=50.0, max_norm=100.0, mu=10, sigma=5, seed=42,\
                 x_ck=None, rotation=None, translation=None):
        
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
        :return: points
        :param max_norm: maximum possible norm that a point can have
        :type max_norm: float
        :param mu: mean of normal distribution from which we sample
        :type: float
        :param sigma: standard deviation of normal distribution from which we sample
        :type: float
        :param seed: random seed (default is the answer to the ultimate question!)
        :type: int
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
        
        
        self.gen_center()
        print("center generated")
        self.gen_points()
        print("points generated")
        self.embed_in_n()
        print("embedding done")
#         self.compute_distances()
        
        self.points_n = torch.from_numpy(self.points_n).float()
        self.points_k = torch.from_numpy(self.points_k).float()
        self.distances = torch.from_numpy(self.distances).float()
        self.actual_distances = torch.from_numpy(self.actual_distances).float()
        
    def __len__(self):
        return self.points_n.shape[0]
    
    def __getitem__(self, idx):
        return self.points_n[idx], self.distances[idx]
        
        
    def gen_center(self):
        """generate a center in lower dimension"""
        if self.x_ck is not None:
            return
        self.x_ck = np.random.normal(self.mu, self.sigma, self.k)
        
    def gen_points(self):
        """generating points in k-dim and embedding in n-dim"""
        points_k = np.random.normal(self.mu, self.sigma, (self.N, self.k))
        points_k = points_k - self.x_ck
        
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
        
    def make_neg_examples(self):
        """generating negative examples, i.e., points not on the manifold"""
        
        # normal_vectors_to_mfld_at_p are actually centred at x_ck, but 
        # we can imaging the same vector at $p$, and later adjust the coordinates
        # by adding the position vector of $p$ back.
        #
        # Also note that these negative examples are being generated using first
        # half of self.points_k
        normal_vectors_to_mfld_at_p = self.points_k[:self.num_neg] - self.x_ck
        embedded_normal_vectors_to_mfld_at_p = np.zeros((self.num_neg, self.n))
        embedded_normal_vectors_to_mfld_at_p[:, :self.k] = normal_vectors_to_mfld_at_p
        
        neg_examples = np.zeros((self.num_neg, self.n))
        neg_distances = np.zeros(self.num_neg)
        
        # canonical basis $e_i$ over leftover dimensions
        remaining_dims = self.n - self.k
        leftover_basis = np.eye(remaining_dims)
        
        # variable to store spanning set
        span_set = np.zeros((remaining_dims + 1, self.n))
        span_set[1:, self.k:] = leftover_basis
        
        
        for i in tqdm(range(self.num_neg)):

            n_cap = embedded_normal_vectors_to_mfld_at_p[i]
            
            # span set is n_cap and leftover basis
            span_set[0] = n_cap
            

            # sample random coefficients
#             coefficients = np.random.uniform(1 + np.finfo(np.float).eps, self.max_norm, span_set.shape[0])
            coefficients = np.random.normal(self.mu, self.sigma, span_set.shape[0])

            # take linear combination
            neg_examples[i] = np.sum(coefficients.reshape(-1, 1) * span_set, axis=0)
            
        # re-scale with random norms, sampled from U[\epsilon, self.r/2]
#         neg_norms = np.random.uniform(low=0, high=0, size=self.num_neg)
        neg_norms = np.random.uniform(low=1e-6 + np.finfo(np.float).eps, high=self.r / 2, size=self.num_neg)
#         neg_norms = np.random.normal(self.mu, self.sigma, size=self.num_neg)
        neg_examples = (neg_norms.reshape(-1, 1) / np.linalg.norm(neg_examples, axis=1, ord=2).reshape(-1, 1)) * neg_examples

        # add position vector of $p$ to get origin centered coordinates
        neg_examples[:, :self.k] = neg_examples[:, :self.k] + self.points_k[:self.num_neg]

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
        
        self.points_n_trivial_[self.num_neg:, :self.k] = self.points_k[self.num_neg:]
        self.points_n_tr_ = self.points_n_trivial_ + self.translation
        
        self.points_n_rot_ = np.dot(self.rotation, self.points_n_tr_.T).T
        
        self.points_n = self.points_n_rot_
        
        self.actual_distances = np.zeros((self.N, 1))
        self.actual_distances[:self.num_neg] = neg_distances.reshape(-1, 1)
        self.actual_distances[self.num_neg:] = np.linalg.norm(self.points_n[self.num_neg:] - self.x_cn, axis=1, ord=2).reshape(-1, 1) - self.r
        self.distances = np.clip(self.actual_distances, a_min=0, a_max=self.D)
        
        # checking that the on-manifold points are still self.r away from center
        print(np.round(np.linalg.norm(self.points_n[self.num_neg:] - self.x_cn, axis=1, ord=2)))
        assert (np.round(np.linalg.norm(self.points_n[self.num_neg:] - self.x_cn, axis=1, ord=2)) == self.r).all()
    
    
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
        
        ax1.scatter3D(self.points_n_trivial_[self.num_neg:, dimX],\
                      self.points_n_trivial_[self.num_neg:, dimY], self.points_n_trivial_[self.num_neg:, dimZ])
        
        plt.title("on-manifold samples (trivial embedding)")
        plt.show()
        
        # before rotation
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self.points_n_tr_[self.num_neg:, dimX],\
                      self.points_n_tr_[self.num_neg:, dimY], self.points_n_tr_[self.num_neg:, dimZ])
        
        plt.title("on-manifold samples (after translation)")
        plt.show()
        
        # after rotation
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self.points_n[self.num_neg:, dimX],\
                      self.points_n[self.num_neg:, dimY], self.points_n[self.num_neg:, dimZ])
        
        plt.title("on-manifold samples (after rotation)")
        plt.show()
        
        
        # plots for off-manifold samples
        
        #indices to show for pre-images
        idx = np.random.choice(np.arange(self.num_neg), num_pre_img)
        
        # trivial pre_image
        
        neg_pre_img = self.points_k[:self.num_neg]
        neg_pre_img_trivial_ = np.zeros((self.num_neg, self.n))
        neg_pre_img_trivial_[:, :self.k] = neg_pre_img
        
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self.points_n_trivial_[self.num_neg:, dimX],\
              self.points_n_trivial_[self.num_neg:, dimY],\
              self.points_n_trivial_[self.num_neg:, dimZ], color="blue", label="on-manifold", s=1, marker="1")
        
        
        
        ax1.scatter3D(neg_pre_img_trivial_[:, dimX][idx],\
                      neg_pre_img_trivial_[:, dimY][idx],\
                      neg_pre_img_trivial_[:, dimZ][idx], color="green", marker="^", label="trivial pre-image", s=80)
        
        
        
        ax1.scatter3D(self.points_n_trivial_[:self.num_neg, dimX][idx],\
                      self.points_n_trivial_[:self.num_neg, dimY][idx],\
                      self.points_n_trivial_[:self.num_neg, dimZ][idx], color="red", label="off-manifold")
        
        actual_distances_trivial_ = np.linalg.norm(neg_pre_img_trivial_\
                               - self.points_n_trivial_[:self.num_neg], ord=2, axis=1)
        
        for i in idx:
            ax1.plot([neg_pre_img_trivial_[:, dimX][i], self.points_n_trivial_[:self.num_neg, dimX][i]],\
                    [neg_pre_img_trivial_[:, dimY][i], self.points_n_trivial_[:self.num_neg, dimY][i]],\
                    [neg_pre_img_trivial_[:, dimZ][i], self.points_n_trivial_[:self.num_neg, dimZ][i]], color="black")
            ax1.text(np.mean([neg_pre_img_trivial_[:, dimX][i], self.points_n_trivial_[:self.num_neg, dimX][i]]),\
                    np.mean([neg_pre_img_trivial_[:, dimY][i], self.points_n_trivial_[:self.num_neg, dimY][i]]),\
                    np.mean([neg_pre_img_trivial_[:, dimZ][i], self.points_n_trivial_[:self.num_neg, dimZ][i]]),\
                    "{:.2f}".format(actual_distances_trivial_[i].item()))
        
        
        plt.legend()
        plt.title("pre-image samples (trivial embedding)")
        plt.show()
        
        plt.hist(np.linalg.norm(neg_pre_img_trivial_\
                               - self.points_n_trivial_[:self.num_neg], ord=2, axis=1) - self.actual_distances[:self.num_neg].numpy().reshape(-1))
        plt.ylabel("freq")
        plt.xlabel("error value")
        plt.title("error distribution for distance value from manifold (trivial)")
        plt.show()
        
        
        # translated pre_image
        
        
        neg_pre_img_tr_ = neg_pre_img_trivial_ + self.translation
        
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self.points_n_tr_[self.num_neg:, dimX],\
              self.points_n_tr_[self.num_neg:, dimY],\
              self.points_n_tr_[self.num_neg:, dimZ], color="blue", label="on-manifold", s=1, marker="1")
        
        
        ax1.scatter3D(neg_pre_img_tr_[:, dimX][idx],\
                      neg_pre_img_tr_[:, dimY][idx],\
                      neg_pre_img_tr_[:, dimZ][idx], color="green", marker="^", label="translated pre-image", s=80)
        
        
        
        ax1.scatter3D(self.points_n_tr_[:self.num_neg, dimX][idx],\
                      self.points_n_tr_[:self.num_neg, dimY][idx],\
                      self.points_n_tr_[:self.num_neg, dimZ][idx], color="red", label="off-manifold")
        
        for i in idx:
            ax1.plot([neg_pre_img_tr_[:, dimX][i], self.points_n_tr_[:self.num_neg, dimX][i]],\
                    [neg_pre_img_tr_[:, dimY][i], self.points_n_tr_[:self.num_neg, dimY][i]],\
                    [neg_pre_img_tr_[:, dimZ][i], self.points_n_tr_[:self.num_neg, dimZ][i]], color="black")
            ax1.text(np.mean([neg_pre_img_tr_[:, dimX][i], self.points_n_tr_[:self.num_neg, dimX][i]]),\
                    np.mean([neg_pre_img_tr_[:, dimY][i], self.points_n_tr_[:self.num_neg, dimY][i]]),\
                    np.mean([neg_pre_img_tr_[:, dimZ][i], self.points_n_tr_[:self.num_neg, dimZ][i]]),\
                    "{:.2f}".format(self.actual_distances[i].item()))
        
        
        plt.legend()
        plt.title("pre-image samples (translated embedding)")
        plt.show()
        
        plt.hist(np.linalg.norm(neg_pre_img_tr_\
                               - self.points_n_tr_[:self.num_neg], ord=2, axis=1) - self.actual_distances[:self.num_neg].numpy().reshape(-1))
        plt.ylabel("freq")
        plt.xlabel("error value")
        plt.title("error distribution for distance value from manifold (tanslated)")
        plt.show()
        
        
        # rotated pre_image
        
        
        neg_pre_img_rot_ = np.dot(self.rotation, neg_pre_img_tr_.T).T
        
        fig = plt.figure(figsize = (10, 7))
        ax1 = plt.axes(projection ="3d")
        
        ax1.scatter3D(self.points_n_rot_[self.num_neg:, dimX],\
              self.points_n_rot_[self.num_neg:, dimY],\
              self.points_n_rot_[self.num_neg:, dimZ], color="blue", label="on-manifold", s=1, marker="1")
        
        
        ax1.scatter3D(neg_pre_img_rot_[:, dimX][idx],\
                      neg_pre_img_rot_[:, dimY][idx],\
                      neg_pre_img_rot_[:, dimZ][idx], color="green", marker="^", label="rotated pre-image", s=80)
        
        
        
        ax1.scatter3D(self.points_n_rot_[:self.num_neg, dimX][idx],\
                      self.points_n_rot_[:self.num_neg, dimY][idx],\
                      self.points_n_rot_[:self.num_neg, dimZ][idx], color="red", label="off-manifold")
        
        for i in idx:
            ax1.plot([neg_pre_img_rot_[:, dimX][i], self.points_n_rot_[:self.num_neg, dimX][i]],\
                    [neg_pre_img_rot_[:, dimY][i], self.points_n_rot_[:self.num_neg, dimY][i]],\
                    [neg_pre_img_rot_[:, dimZ][i], self.points_n_rot_[:self.num_neg, dimZ][i]], color="black")
            ax1.text(np.mean([neg_pre_img_rot_[:, dimX][i], self.points_n_rot_[:self.num_neg, dimX][i]]),\
                    np.mean([neg_pre_img_rot_[:, dimY][i], self.points_n_rot_[:self.num_neg, dimY][i]]),\
                    np.mean([neg_pre_img_rot_[:, dimZ][i], self.points_n_rot_[:self.num_neg, dimZ][i]]),\
                    "{:.2f}".format(self.actual_distances[i].item()))
        
        
        plt.legend()
        plt.title("pre-image samples (rotated embedding)")
        plt.show()
        
        plt.hist(np.linalg.norm(neg_pre_img_rot_\
                               - self.points_n_rot_[:self.num_neg], ord=2, axis=1) - self.actual_distances[:self.num_neg].numpy().reshape(-1))
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
#                 print(neg_ex.shape, a_pre_img.shape)
                dist = np.linalg.norm(neg_ex - a_pre_img, ord=2)
                
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    
            min_dist_vals.append(min_dist)
            
        errors = np.abs(self.actual_distances[idx].numpy().reshape(-1) - np.array(min_dist_vals))
        rel_errors = errors / self.actual_distances[idx].numpy().reshape(-1)
        print("absolute errors:", errors)
        print("relative errors:", rel_errors)
        

if __name__ == '__main__':

    dummy_params = {
        "N": 100000,
        "num_neg": None,
        "n": 3,
        "k": 2,
        "r": 100.0,
        "D": 25.0,
        "max_norm": 500.0,
        "mu": 1000,
        "sigma": 5000,
        "seed": 85
    }

    dummy_set = RandomSphere(**dummy_params)
    dummy_set.viz_test()

    