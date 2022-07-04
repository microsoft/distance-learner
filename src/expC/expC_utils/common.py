# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
import json

import numpy as np

import torch

import matplotlib.pyplot as plt

from expB import learn_mfld_distance as lmd
from datagen.synthetic.single import manifold

def load_config(dump_dir):
    config_fn = os.path.join(dump_dir, "config.json")
    with open(config_fn) as f:
        config_dict = json.load(f)
    return config_dict

def load_model(dump_dir, best=True):
    prefix = ""
    if not best:
        prefix = "running_"
    model_fn = os.path.join(dump_dir, "models", prefix + "ckpt.pth")
    config_dict = load_config(dump_dir)

    model_params = config_dict["model"]

    mod_type = model_params.pop("model_type")

    model = lmd.model_type[mod_type](**model_params)
    model.load_state_dict(torch.load(model_fn)["model_state_dict"])

    return model

def get_nplane_samples_for_kmfld(k_dim_samples, dataset, n=3, num_samples=50000):
    """
    returns samples from n-dim plane containing the manifold

    :param k_dim_samples: k-dim embeddings of the points
    :type k_dim_samples: torch.Tensor
    :param dataset: dataset object to get necessary transforms
    :type dataset: torch.util.data.Dataset
    :param n: the dimension of the space in which the plane will be
    :type n: int
    :param num_samples: number of samples to generate
    :type num_samples:: int
    """
    k = k_dim_samples.shape[1]
    if type(k_dim_samples) == torch.Tensor:
        k_dim_samples = k_dim_samples.numpy()

    low = None
    high = None
    gen_kd_grid = None

    if dataset.rotation.shape[0] == 2:
        x_min, x_max = np.min(k_dim_samples[:, 0]) * (1 - np.sign(np.min(k_dim_samples[:, 0])) * 0.1) - 0.2, np.max(k_dim_samples[:, 0]) * (1 + np.sign(np.max(k_dim_samples[:, 0])) * 0.1) + 0.2
        y_min, y_max = np.min(k_dim_samples[:, 1]) * (1 - np.sign(np.min(k_dim_samples[:, 1])) * 0.1) - 0.2, np.max(k_dim_samples[:, 1]) * (1 + np.sign(np.max(k_dim_samples[:, 1])) * 0.1) + 0.2
        low = np.array([x_min, y_min])
        high = np.array([x_max, y_max])
        gen_kd_grid = np.random.uniform(low, high, size=(num_samples, k))
    else:
        low = (1 - np.sign(np.min(k_dim_samples)) * 0.1) * np.min(k_dim_samples) 
        high = (1 + np.sign(np.max(k_dim_samples)) * 0.1) * np.max(k_dim_samples)

        gen_kd_grid = np.random.uniform(low, high, size=(num_samples, k))

    gen_nd_grid = np.zeros((num_samples, n))

    accessor = dataset
    if isinstance(dataset, manifold.Manifold):
        accessor = dataset.genattrs

    gen_nd_grid[:, :k] = gen_kd_grid
    gen_nd_grid = gen_nd_grid + accessor.translation
    rotation = accessor.rotation
    if len(accessor.rotation.shape) == 3 and accessor.rotation.shape[0] == 2:
        rotation = accessor.rotation[0]
    gen_nd_grid = np.dot(rotation, gen_nd_grid.T).T
    gen_nd_grid = gen_nd_grid / accessor.norm_factor
    gen_nd_grid = torch.from_numpy(gen_nd_grid).float()
    gen_nd_grid = gen_nd_grid - accessor.anchor + accessor.fix_center

    gen_kd_grid = torch.from_numpy(gen_kd_grid).float()
    gen_nd_grid = gen_nd_grid.float()

    return gen_kd_grid, gen_nd_grid

def get_coplanar_kdim_samples(dataset):
    """for datasets with single manifold
    or multiple manifolds that are co-planar"""

    if isinstance(dataset, manifold.Manifold):
        return dataset.genattrs.points_k

    if hasattr(dataset, "on_mfld_pts_k_"):
        if dataset.on_mfld_pts_k_ is None:
            dataset._collect_on_mfld_k()
        return dataset.on_mfld_pts_k_
    
    elif hasattr(dataset, "all_points_k"):
        if dataset.all_points_k is None:
            dataset.get_all_points_k()
        return dataset.all_points_k



if __name__ == "__main__":
    model = load_model("/azuredrive/dumps/expC_dist_learner_for_adv_ex/rdm_swrolls/2")
