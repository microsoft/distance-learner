# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import torch
import torchvision

import numpy as np

from sklearn.decomposition import PCA

trfm_map = {
    "default": DefaultTransform,
    "pca": PCATransform
}

class DefaultTransform(object):

    name_to_cstat_map = {
        "mnist": {
            "mean": 0.1307,
            "std": 0.3081
        }
    }

    def __init__(self, name="mnist"):
        self.name = name
        self.cstats = self.name_to_cstat_map[name]
        self.transform = torchvision.transform.compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __call__(self, input):
        return self.transform(input)

class PCATransform(object):

    def __init__(self, output_dim, data):
        self.output_dim = output_dim
        self.pca = None
        self.fit(data)

    def fit(self, data):
        self.pca = PCA(n_components=self.output_dim)
        self.pca.fit(data.reshape(data.shape[0], -1))

    def transform(self, inp):
        return self.pca.transform(inp.reshape(inp.shape[0], -1))

    def __call__(self, inp):
        return self.transform(inp)
        
        

