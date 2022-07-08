# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import copy
import unittest
import logging
# logging.disable(logging.CRITICAL)

import torch
import numpy as np

from utils import seed_everything
from datagen.synthetic.single.sphere import RandomSphere


class RandomSphereTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        cls.dummy_params = {
            "N": 100000,
            "num_neg": 50000,
            "n": 500,
            "k": 2,
            "r": 1,
            "D": 0.2,
            "max_norm": 0.25,
            "mu": 0,
            "sigma": 1,
            "seed": 42,
            "gamma": 1,
            "online": False,
            "off_online": False,
            "augment": False
        }

        cls.split_params = {
            "train": copy.deepcopy(cls.dummy_params),
            "val": copy.deepcopy(cls.dummy_params),
            "test": copy.deepcopy(cls.dummy_params)
        }

        cls.split_params["val"]["N"] = 1000
        cls.split_params["test"]["N"] = 1000

        cls.split_params["val"]["num_neg"] = 500
        cls.split_params["test"]["num_neg"] = 500

        cls.split_params["val"]["seed"] = 89
        cls.split_params["test"]["seed"] = 116

        cls.train_set, cls.val_set, cls.test_set = RandomSphere.make_train_val_test_splits(cls.split_params)

        cls.test_sphere = RandomSphere(**cls.dummy_params)
        cls.test_sphere.compute_points()
        # set system seed to same seed as dummy sphere
        seed_everything(cls.dummy_params["seed"])

    def test_settings(self):
        self.assertEqual(self.test_sphere.genattrs.N, self.dummy_params["N"])
        self.assertEqual(self.test_sphere.genattrs.num_neg, self.dummy_params["num_neg"])
        self.assertEqual(self.test_sphere.genattrs.seed, self.dummy_params["seed"])

    def test_simple_gen(self):

        # set system seed to same seed as dummy sphere
        seed_everything(self.dummy_params["seed"])

        # compare translation
        tmp = np.random.normal(self.dummy_params["mu"], self.dummy_params["sigma"], self.dummy_params["n"])
        self.assertEqual((self.test_sphere.genattrs.translation == tmp).all(), True)

        # # compare rotation
        tmp = np.random.normal(self.dummy_params["mu"], self.dummy_params["sigma"], (self.dummy_params["n"], self.dummy_params["n"]))
        tmp = np.linalg.qr(tmp)[0]
        self.assertEqual((self.test_sphere.genattrs.rotation == tmp).all(), True)

        # # compare centres in k-dim
        tmp = np.random.normal(self.dummy_params["mu"], self.dummy_params["sigma"], self.dummy_params["k"])
        self.assertEqual((self.test_sphere.specattrs.x_ck == tmp).all(), True)
        tmp_x_ck = tmp

        # compare points in k-dim
        tmp = np.random.normal(size=(self.dummy_params["N"] - self.dummy_params["num_neg"], self.dummy_params["k"]))
        tmp = (self.dummy_params["r"] * (tmp / np.linalg.norm(tmp, ord=2, axis=1).reshape(-1, 1))) + tmp_x_ck
        tmp = torch.from_numpy(tmp).float()
        self.assertEqual((self.test_sphere.genattrs.points_k == tmp).all(), True)
        tmp_points_k = tmp

        # testing distances
        tmp = np.linalg.norm(tmp_points_k - tmp_x_ck, axis=1, ord=2).reshape(-1, 1) - self.dummy_params["r"]
        tmp = np.round(torch.from_numpy(tmp).float())
        self.assertEqual((np.round(self.test_sphere.genattrs.actual_distances[self.test_sphere.genattrs.num_neg:]) == tmp).all(), True)
        self.assertEqual((self.test_sphere.genattrs.actual_distances[:self.test_sphere.genattrs.num_neg] > 0).all(), True)
        self.assertEqual((self.test_sphere.genattrs.actual_distances[:self.test_sphere.genattrs.num_neg] < self.dummy_params["max_norm"]).all(), True)

    # TODO: Tests for checking consistency across splits


if __name__ == '__main__':
    unittest.main()
