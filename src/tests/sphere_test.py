import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import unittest

import torch
import numpy as np

from utils import seed_everything
from datagen.synthetic.single.sphere import RandomSphere

class RandomSphereTest(unittest.TestCase):

    dummy_params = {
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

    def test_settings(self):
        test_sphere = RandomSphere(**self.dummy_params)
        self.assertEqual(test_sphere.genattrs.N, self.dummy_params["N"])
        self.assertEqual(test_sphere.genattrs.num_neg, self.dummy_params["num_neg"])
        self.assertEqual(test_sphere.genattrs.seed, self.dummy_params["seed"])
    
    def test_simple_gen(self):
        test_sphere = RandomSphere(**self.dummy_params)
        test_sphere.compute_points()

        seed_everything(self.dummy_params["seed"])

        tmp = np.random.normal(self.dummy_params["mu"], self.dummy_params["sigma"], self.dummy_params["n"])
        self.assertEqual((test_sphere.genattrs.translation == tmp).all(), True)

        tmp = np.random.normal(self.dummy_params["mu"], self.dummy_params["sigma"], (self.dummy_params["n"], self.dummy_params["n"]))
        tmp = np.linalg.qr(tmp)[0]
        self.assertEqual((test_sphere.genattrs.rotation == tmp).all(), True)

        tmp = np.random.normal(self.dummy_params["mu"], self.dummy_params["sigma"], self.dummy_params["k"])
        self.assertEqual((test_sphere.specattrs.x_ck == tmp).all(), True)
        tmp_x_ck = tmp

        tmp = np.random.normal(size=(self.dummy_params["N"] - self.dummy_params["num_neg"], self.dummy_params["k"]))
        tmp = (self.dummy_params["r"] * (tmp / np.linalg.norm(tmp, ord=2, axis=1).reshape(-1, 1))) + tmp_x_ck
        tmp = torch.from_numpy(tmp).float()
        self.assertEqual((test_sphere.genattrs.points_k == tmp).all(), True)

if __name__ == '__main__':
    unittest.main()