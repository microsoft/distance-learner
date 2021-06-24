import numpy as np

from datagen.synthetic.multiple.intertwinedswissrolls import *

dummy_params = {

    "N": 100000,
    "num_neg": None,
    "n": 2,
    "k": 2,
    "D": 80,
    "max_norm": 30,
    "contract": 100,
    "mu": 0,
    "sigma": 1,
    "seed": 42,
    "gamma": 0.5,
    "t_min": 150,
    "t_max": 450,
    "num_turns": None,
    "omega": np.pi * 0.01
}


test = IntertwinedSwissRolls(**dummy_params)
test.compute_points()