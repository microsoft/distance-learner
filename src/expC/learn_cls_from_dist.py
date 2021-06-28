"""
Explores various ways to classify points on spheres (or general manifold)
using the distance learned in Experiment B (see: `learn_mfld_distance.py`)
"""

import os
import re
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import json
import copy
import time
import copy
import random
import argparse

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


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

from sacred import Experiment
from sacred.observers import FileStorageObserver

from datagen.synthetic.single import manifold, sphere, swissroll
from datagen.synthetic.multiple import intertwinedswissrolls
from expB import learn_mfld_distance as lmd

from model_ingredients import initialise_model, model_cfg, model_ingredient
from data_ingredients import initialise_data, data_cfg, data_ingredient


# class DistanceBasedClf(object):
#     """
#         Classifier that uses distances computed by 
#         Distance Regressor for manifolds to predict
#         class values
#     """

#     def __init__(self):
#         pass



def argmin_dist_clf(inputs, targets, class_labels, on_mfld=True, off_mfld_lbl=2):
    """
        argmin distance based clf for on-manifold points

        :param inputs: logits obtained from a distance regressor
        :type inputs: torch.Tensor 
        :param targets: actual values of distances
        :type targets: torch.Tensor
        :param class_labels: class labels from the dataset
        :type class_labels: torch.Tensor
        :param on_mfld: find stats for on manifold points (only works for well-separate manifolds)
        :type on_mfld: bool
        :param off_mfld_lbl: label for off manifold samples
        :type off_mfld_lbl: int
        :return: minimum and argmin of predicted and target distances
        :rtype: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
    """
    
    OFF_MFLD_LBL = off_mfld_lbl
    # selecting on-manifold points for further computations

    cls_labels = class_labels.detach().clone()

    min_pred_dist = None
    pred_argmin = None

    min_true_dist = None
    true_argmin = None

    if on_mfld:
        on_mfld_targets = targets[cls_labels != OFF_MFLD_LBL]
        on_mfld_inputs = inputs[cls_labels != OFF_MFLD_LBL]
        
        min_pred_dist, pred_argmin = torch.min(on_mfld_inputs, axis=1)
        min_true_dist, true_argmin = torch.min(on_mfld_targets, axis=1)
        print("On Manifold Results\n")
        print(classification_report(true_argmin, pred_argmin, target_names=["S" + str(i + 1) for i in range(inputs.shape[1])]))
        clf_report_dict = classification_report(true_argmin, pred_argmin, target_names=["S" + str(i + 1) for i in range(inputs.shape[1])], output_dict=True)

    else:
        min_pred_dist, pred_argmin = torch.min(inputs, axis=1)
        min_true_dist, true_argmin = torch.min(targets, axis=1)
        print("Results\n")
        print(classification_report(true_argmin, pred_argmin, target_names=["S" + str(i + 1) for i in range(inputs.shape[1])] + ["off-mfld"]))
        clf_report_dict = classification_report(true_argmin, pred_argmin, target_names=["S" + str(i + 1) for i in range(inputs.shape[1])] + ["off-mfld"], output_dict=True)
    
    

    return min_pred_dist, pred_argmin, min_true_dist, true_argmin, clf_report_dict
        

ex = Experiment("stdclf_vs_distlearn", ingredients=[model_ingredient, data_ingredient])

@ex.config
def config(data, model):

    train = True # train flag
    test = False # test flag
    cuda = 0 # GPU device id for training

    num_workers = 8
    OFF_MFLD_LABEL = 2

    batch_size = 512
    task = "regression" # "regression" or "clf"
    train_on_onmfld = True # flag for only training on on-mfld. samples (only useful for stdclf.)
    if task == "regression":
        train_on_onmfld = False # default should be False for regression
    num_epochs = 500
    warmup = 10
    cooldown = 300
    lr = 1e-5

    num_mflds = 2 # number of manifolds
    num_classes = num_mflds if (train_on_onmfld and task == "clf") else num_mflds + 1 # useful for stdclf only
    input_size = data["data_params"]["train"]["n"] # dimension in which manifold is embedded

    loss_func = "masked_mse" # ["std_mse", "masked_mse", "weighted_mse", "cross_entropy"]

    ftname = "normed_points" # feature from the dataset to train on
    tgtname = "normed_actual_distances" # target values to train against

    # TIME_STAMP = time.strftime("%d%m%Y-%H%M%S")

    # name = "stdclf_vs_distlearn"
    name = data["data_tag"]

    logdir = data["logdir"] # directory to store run dumps and input data

    ex.observers.append(FileStorageObserver(logdir))



@ex.capture
def data_setup(task, train, train_on_onmfld, OFF_MFLD_LABEL, batch_size, num_workers, data):

    train_set, val_set, test_set = initialise_data()

    if task == "clf" and train_on_onmfld:
        for dataset in [train_set, val_set, test_set]:
            for attr in ["all_points", "all_distances", "normed_all_points", "normed_all_distances", "class_labels"]:
                if isinstance(dataset, manifold.Manifold):
                    setattr(dataset.genattrs, attr, getattr(dataset.genattrs, attr)[dataset.genattrs.class_labels != OFF_MFLD_LABEL])
                else:
                    setattr(dataset, attr, getattr(dataset, attr)[dataset.class_labels != OFF_MFLD_LABEL])


    # if task == "clf" and train_on_onmfld:
    #     if isinstance(train_set, manifold.Manifold):
            
    #         train_set.genattrs.all_points = train_set.genattrs.all_points[train_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #         train_set.genattrs.all_distances = train_set.genattrs.all_distances[train_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #         train_set.genattrs.normed_all_points = train_set.genattrs.normed_all_points[train_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #         train_set.genattrs.normed_all_distances = train_set.genattrs.normed_all_distances[train_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #         train_set.genattrs.class_labels = train_set.genattrs.class_labels[train_set.genattrs.class_labels != OFF_MFLD_LABEL]

    #         val_set.genattrs.all_points = val_set.genattrs.all_points[val_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #         val_set.genattrs.all_distances = val_set.genattrs.all_distances[val_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #         val_set.genattrs.normed_all_points = val_set.genattrs.normed_all_points[val_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #         val_set.genattrs.normed_all_distances = val_set.genattrs.normed_all_distances[val_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #         val_set.genattrs.class_labels = val_set.genattrs.class_labels[val_set.genattrs.class_labels != OFF_MFLD_LABEL]

    #         test_set.genattrs.all_points = test_set.genattrs.all_points[test_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #         test_set.genattrs.all_distances = test_set.genattrs.all_distances[test_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #         test_set.genattrs.normed_all_points = test_set.genattrs.normed_all_points[test_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #         test_set.genattrs.normed_all_distances = test_set.genattrs.normed_all_distances[test_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #         test_set.genattrs.class_labels = test_set.genattrs.class_labels[test_set.genattrs.class_labels != OFF_MFLD_LABEL]
    #     else:
    #         train_set.all_points = train_set.all_points[train_set.class_labels != OFF_MFLD_LABEL]
    #         train_set.all_distances = train_set.all_distances[train_set.class_labels != OFF_MFLD_LABEL]
    #         train_set.normed_all_points = train_set.normed_all_points[train_set.class_labels != OFF_MFLD_LABEL]
    #         train_set.normed_all_distances = train_set.normed_all_distances[train_set.class_labels != OFF_MFLD_LABEL]
    #         train_set.class_labels = train_set.class_labels[train_set.class_labels != OFF_MFLD_LABEL]

    #         val_set.all_points = val_set.all_points[val_set.class_labels != OFF_MFLD_LABEL]
    #         val_set.all_distances = val_set.all_distances[val_set.class_labels != OFF_MFLD_LABEL]
    #         val_set.normed_all_points = val_set.normed_all_points[val_set.class_labels != OFF_MFLD_LABEL]
    #         val_set.normed_all_distances = val_set.normed_all_distances[val_set.class_labels != OFF_MFLD_LABEL]
    #         val_set.class_labels = val_set.class_labels[val_set.class_labels != OFF_MFLD_LABEL]

    #         test_set.all_points = test_set.all_points[test_set.class_labels != OFF_MFLD_LABEL]
    #         test_set.all_distances = test_set.all_distances[test_set.class_labels != OFF_MFLD_LABEL]
    #         test_set.normed_all_points = test_set.normed_all_points[test_set.class_labels != OFF_MFLD_LABEL]
    #         test_set.normed_all_distances = test_set.normed_all_distances[test_set.class_labels != OFF_MFLD_LABEL]
    #         test_set.class_labels = test_set.class_labels[test_set.class_labels != OFF_MFLD_LABEL]

    shuffle = True if train else False

    dataloaders = {
        "train": DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers),
        "val": DataLoader(dataset=val_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers),
        "test": DataLoader(dataset=test_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    }

    datasets = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }
    return datasets, dataloaders

@ex.capture
def run_training(num_epochs, task, loss_func, lr, warmup,\
     cooldown, cuda, ftname, tgtname, name, save_dir, _log):


    device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and cuda is not None else "cpu")
    
    datasets, dataloaders = data_setup()

    model = initialise_model()
    # print(model)
    loss_function = lmd.loss_funcs[loss_func]

    if task == "clf":
        loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler_params = {"warmup": warmup, "cooldown": cooldown}
    lr_sched_factor = lambda epoch: epoch / (scheduler_params["warmup"]) if epoch <= scheduler_params["warmup"] else (1 if epoch > scheduler_params["warmup"] and epoch < scheduler_params["cooldown"] else max(0, 1 + (1 / (scheduler_params["cooldown"] - num_epochs)) * (epoch - scheduler_params["cooldown"])))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_sched_factor)
    
    model, optimizer, scheduler, _, _ = lmd.train(model, optimizer, loss_function,\
        dataloaders, device, save_dir, scheduler, feature_name=ftname, target_name=tgtname,\
        num_epochs=num_epochs, task=task, name=name, scheduler_params=scheduler_params, specs_dict=None, debug=False)

    return model, optimizer, scheduler, datasets, dataloaders

@ex.capture
def log_clf_dict(_run, clf_report_dict, split="train", on_mfld=True):
    
    _run.log_scalar("acc", clf_report_dict["accuracy"])

    for key in clf_report_dict.keys():
        if type(clf_report_dict[key]) == dict:
            for metric in clf_report_dict[key]:
                metric_tag = "onmfld." + metric if on_mfld else metric
                tag = ".".join([split, key, metric_tag]) if split is not None else ".".join([key, metric_tag])
                _run.log_scalar(tag, clf_report_dict[key][metric])


@ex.capture
def run_eval(data, model, dataloaders, datasets, cuda, task, ftname, tgtname, input_size,\
     num_classes, num_mflds, train_on_onmfld, OFF_MFLD_LABEL, num_workers, batch_size,\
     _run, plotdir, analysis=True, save_dir=None):
    
    # print(save_dir)
    device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and cuda is not None else "cpu")

    if dataloaders is None:
        dataloaders, datasets = data_setup()

    logits_and_targets = dict()

    for split in ["train", "val", "test"]:

        dl = dataloaders[split]
        _, _, all_targets, all_logits = lmd.test(model, dl, device, task, ftname, tgtname, save_dir, name=split)

        if split not in logits_and_targets:
            logits_and_targets[split] = dict()

        logits_and_targets[split]["logits"] = all_logits
        logits_and_targets[split]["targets"] = all_targets

        print("---------------------------------\nResults for {} split\n---------------------------------\n".format(split))

        # below code only works when number of classes 
        # in dataset >= 2 or when dataset class lies in
        # `datagen.synthetic.multiple`
        #
        # TODO: maybe ammend multi-manifold datasets to avoid this?
        if num_classes >= 2:
            if task == "regression":
                _, _, _, _, clf_report_dict = argmin_dist_clf(all_logits, all_targets, datasets[split].class_labels)
                log_clf_dict(_run, clf_report_dict, split, on_mfld=True)
                _, _, _, _, clf_report_dict = argmin_dist_clf(all_logits, all_targets, datasets[split].class_labels, on_mfld=False)
                log_clf_dict(_run, clf_report_dict, split, on_mfld=False)
            elif task == "clf":
                """
                for standard clf. on-mfld. stats are the only stats that can be computed for 
                case where data is trained only on on-mfld. samples. The reason
                is that when you have on-mfld. training, then ONLY on-mfld. class
                labels exist in the dataset. And in such a case, off-mfld. analysis
                does not even make sense.

                However, on-mfld. training is useful in highlighting the difference in
                decision regions in both kinds of training.
                """ 
                def disp_stdclf_results(all_logits, datasets, split, train_on_onmfld, on_mfld=True):

                    msg = "On Manifold Results\n" if on_mfld else "Results\n"
                    pred_labels = torch.max(all_logits, axis=1)[1]
                    true_labels = datasets[split].class_labels

                    labels = ["S" + str(i) for i in range(all_logits.shape[1])]

                    if (not train_on_onmfld) and on_mfld:
                        """
                        if training has been done on on-mfld. points but results are needed over
                        all of the dataset, then off-mfld. points should be removed, since the
                        network cannot pred
                        """
                        pred_labels = pred_labels[true_labels != OFF_MFLD_LABEL]
                        true_labels = true_labels[true_labels != OFF_MFLD_LABEL]
                    
                    if not train_on_onmfld:
                        # training is done on off manifold samples and
                        # then off manifold label can be predicted and 
                        # therefore needs to be included in labels
                        labels = labels + ["off manifold"]
                    

                    clf_report_dict = classification_report(true_labels, pred_labels, target_names=labels, output_dict=True, zero_division=0)
                    print(msg)
                    print(classification_report(true_labels, pred_labels, target_names=labels, zero_division=0))
                    log_clf_dict(_run, clf_report_dict, split, on_mfld)

                
                disp_stdclf_results(all_logits, datasets, split, train_on_onmfld, on_mfld=True)

                if not train_on_onmfld:
                    disp_stdclf_results(all_logits, split, train_on_onmfld, on_mfld=False)



    if analysis and input_size == 2:
        
        # k = data["data_params"]["k"]
        # n = data["data_params"]["n"]

        low = torch.min(datasets["train"].normed_all_points) - 0.1
        high = torch.max(datasets["train"].normed_all_points) + 0.1

        def make_2d_grid_data(low, high):

            gen_2d_data = np.random.uniform(low, high, size=(100000, input_size))
            gen_2d_data = torch.from_numpy(gen_2d_data).float()
            
    
            dummy_labels = torch.from_numpy(np.zeros((100000, num_classes))).float()
            if task == "clf":
                dummy_labels = dummy_labels[:, 0]
            gen_2d_dataset = TensorDataset(gen_2d_data, dummy_labels)

            gen_2d_dl = DataLoader(dataset=gen_2d_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
            
            return gen_2d_dl, gen_2d_dataset, gen_2d_data

        gen_2d_dl, gen_2d_dataset, gen_2d_data = make_2d_grid_data(low, high)

        _, _, _, gen_2d_logits = lmd.test(model, gen_2d_dl, device, feature_name=ftname,\
                                 target_name=tgtname, task=task, debug=False) 
        
        gen_pred_classes = None
        if task == "clf":
            gen_pred_classes = torch.max(gen_2d_logits, axis=1)[1]
        elif task == "regression":
            gen_pred_classes = torch.min(gen_2d_logits, axis=1)[1]            

        THRESH = datasets["train"].S1.genattrs.D / datasets["train"].norm_factor

        if task == "regression": gen_pred_classes[torch.min(gen_2d_logits, axis=1)[0] >= THRESH] = OFF_MFLD_LABEL

        col = ["blue", "green", "yellow"]

        plt.figure(figsize=(6, 6))

        for i in range(len(col)):
            plt.scatter(gen_2d_data[gen_pred_classes.numpy() == i, 0].numpy(), gen_2d_data[gen_pred_classes.numpy() == i, 1].numpy(), s=0.01, c=col[i], label=i)
            if (i < 2):
                plt.scatter(datasets["train"].normed_all_points[datasets["train"].class_labels == i, 0].numpy(), datasets["train"].normed_all_points[datasets["train"].class_labels == i, 1].numpy(), c=col[i], s=0.1)

        plt.legend(markerscale=100)
        plt.title("fig.1: clf labels with off-manifold label (2) for dist regressor")
        plt.savefig(os.path.join(plotdir, "fig-1.png"))
        ex.add_artifact(filename=os.path.join(plotdir, "fig-1.png"))


        
        for i in range(num_mflds):

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            axs[0].scatter(gen_2d_data[gen_2d_logits[:, i] >= THRESH][:, 0],\
                    gen_2d_data[gen_2d_logits[:, i] >= THRESH][:, 1], s=0.01)
            axs[0].set_title("predicted off mfld pts.")


            sc1 = axs[1].scatter(gen_2d_data[:, 0], gen_2d_data[:, 1], s=0.01, c=gen_2d_logits[:, i], cmap="hot")
            fig.colorbar(sc1, ax=axs[1])
            axs[1].set_title("predicted heat map")


            fig.suptitle("fig. 2.{a}: for S{a} (label {b})".format(a=i+1, b=i))
            fig.tight_layout()

            plt.savefig(os.path.join(plotdir, "fig-2-{a}.png".format(a=i+1)))
            ex.add_artifact(filename=os.path.join(plotdir, "fig-2-{a}.png".format(a=i+1)))


    elif analysis and input_size == 3:
        # TODO: add code for visualizing
        # the plane/volume of a 1D/2D manifold in 3D
        pass



    return logits_and_targets



@ex.automain
def main(train, logdir, data, _log, _run):

    model = None
    optimizer = None
    scheduler = None
    datasets = None
    dataloaders = None

    TIME_STAMP = time.strftime("%d%m%Y-%H%M%S")
    
    save_dir = os.path.join(logdir, _run._id)
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if train:
        # first train the model
        model, optimizer, scheduler, datasets, dataloaders = run_training(save_dir=save_dir)
        # generate the logits
        logits_and_targets = run_eval(model=model, dataloaders=dataloaders,\
             datasets=datasets, save_dir=save_dir, plotdir=plot_dir)
        # generate 2D/3D clf decision regions

    else:
        run_eval(model=model, dataloaders=dataloaders, save_dir=save_dir)

    


    

# if __name__ == '__main__':
#     # lmd.init()
#     run = ex.run()