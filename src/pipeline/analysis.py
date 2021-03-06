# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse

import numpy as np

import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader

from datagen.synthetic.single import sphere, swissroll
from datagen.synthetic.multiple import intertwinedswissrolls, wellseparatedspheres, concentricspheres
from datagen.real import mnist

from pipeline_utils.common import *
from pipeline_utils import plotter

MFLD_TYPES = {
    "single-sphere": sphere.RandomSphere,
    "single-swissroll": swissroll.RandomSwissRoll,
    "ittw-swissrolls": intertwinedswissrolls.IntertwinedSwissRolls,
    "inf-ittw-swissrolls": intertwinedswissrolls.IntertwinedSwissRolls,
    "inf-ittw-swissrolls2": intertwinedswissrolls.IntertwinedSwissRolls,
    "inf-ws-spheres": wellseparatedspheres.WellSeparatedSpheres,
    "inf-conc-spheres": concentricspheres.ConcentricSpheres,
    "mnist": mnist.MNISTManifolds
}

MFLD_VIZ_BY_TYPE = {
    "ittw-swissrolls": plotter,
    "inf-ittw-swissrolls": plotter,
    "inf-ittw-swissrolls2": plotter,
    "inf-ws-spheres": plotter,
    "inf-conc-spheres": plotter

}


def run_analysis(dump_dir, on="val", num_points=50000, thresh=None):

    if on not in ["train", "test", "val"]:
        raise RuntimeError("`on` can only be one of 'train', 'test', 'val'")

    config_dict = load_config(dump_dir)
    task = config_dict["task"]
    model = load_model(dump_dir)
    data_dir = os.path.join(dump_dir, "../../data")
    data_mfld_type = config_dict["data"]["mtype"]

    train_set, val_set, test_set = MFLD_TYPES[data_mfld_type].load_splits(data_dir)

    data_set = val_set if on == "val" else test_set

    # plot figure 1
    MFLD_VIZ_BY_TYPE[data_mfld_type].make_plots(model, data_set, dump_dir, task, num_points, thresh=thresh)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_dir", type=str, help="directory of result dumps", required=True)
    parser.add_argument("--on", type=str, help="which split to analyse (train, val, test)", default="test")
    parser.add_argument("--num_points", type=int, help="number of samples to render", default=50000)
    parser.add_argument("--thresh", type=float, help="threshold distance to use", default=None)


    args = parser.parse_args()

    run_analysis(args.dump_dir, args.on, args.num_points, args.thresh)
    



