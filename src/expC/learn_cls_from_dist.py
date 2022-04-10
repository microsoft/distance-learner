"""
Explores various ways to classify points on spheres (or general manifold)
using the distance learned in Experiment B (see: `learn_mfld_distance.py`)
"""

from hashlib import new
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
import subprocess

from collections.abc import Iterable
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



def argmin_dist_clf(inputs, targets, class_labels, thresh, on_mfld=True, off_mfld_lbl=2):
    """
        argmin distance based clf for on-manifold points

        this has been coded this way assuming that we will
        later be dealing with manifolds that are not "well-separated"

        :param inputs: logits obtained from a distance regressor
        :type inputs: torch.Tensor 
        :param targets: actual values of distances
        :type targets: torch.Tensor
        :param class_labels: class labels from the dataset
        :type class_labels: torch.Tensor
        :param thresh: threshold to classify a point as off-manifold
        :type thresh: float
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

    target_names = ["S" + str(i + 1) for i in range(inputs.shape[1])] + ["off-mfld"]
    labels = np.arange(inputs.shape[1] + 1)

    if on_mfld:
        on_mfld_targets = targets[cls_labels != OFF_MFLD_LBL]
        on_mfld_inputs = inputs[cls_labels != OFF_MFLD_LBL]
        on_mfld_classes = cls_labels[cls_labels != OFF_MFLD_LBL]

        min_pred_dist, pred_argmin = torch.min(on_mfld_inputs, axis=1)
        min_true_dist, true_argmin = torch.min(on_mfld_targets, axis=1)
        
        pred_argmin[min_pred_dist > thresh] = OFF_MFLD_LBL
        true_argmin[min_true_dist > thresh] = OFF_MFLD_LBL

        print("* On Manifold Results\n")
        print(classification_report(true_argmin, pred_argmin, target_names=target_names, labels=labels, zero_division=0))
        clf_report_dict = classification_report(true_argmin, pred_argmin, target_names=target_names, labels=labels, output_dict=True, zero_division=0)

    else:
        min_pred_dist, pred_argmin = torch.min(inputs, axis=1)
        min_true_dist, true_argmin = torch.min(targets, axis=1)

        pred_argmin[min_pred_dist > thresh] = OFF_MFLD_LBL
        true_argmin[min_true_dist > thresh] = OFF_MFLD_LBL

        print("* Results\n")
        print(classification_report(true_argmin, pred_argmin, target_names=target_names, labels=labels, zero_division=0))
        clf_report_dict = classification_report(true_argmin, pred_argmin, target_names=target_names, labels=labels, output_dict=True, zero_division=0)
    

    

    return min_pred_dist, pred_argmin, min_true_dist, true_argmin, clf_report_dict
        

ex = Experiment("stdclf_vs_distlearn", ingredients=[model_ingredient, data_ingredient])

@ex.config
def config(data, model):

    train = True # train flag
    test = False # test flag
    cuda = 0 # GPU device id for training

    debug = False # Flag for saving epoch wise logits and other debugging stuff

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
    num_classes = num_mflds if (train_on_onmfld and task == "clf") or (task == "regression") else num_mflds + 1 # useful for stdclf only
    input_size = data["data_params"]["train"]["n"] # dimension in which manifold is embedded
    online = data["data_params"]["train"]["online"]
    off_online = data["data_params"]["train"]["off_online"]
    augment = data["data_params"]["train"]["augment"]

    on_mfld_noise = 1e-6
    test_off_mfld = True # test stdclf with off mfld samples


    loss_func = "masked_mse" # ["std_mse", "masked_mse", "weighted_mse", "cross_entropy"]
    if task == "clf":
        loss_func = "cross_entropy"

    ftname = "normed_points" # feature from the dataset to train on
    tgtname = "normed_smooth_distances" # target values to train against
    if task == "clf":
        tgtname = "classes"

    ram_efficient = True # delete unneeded attributes of dataset for freeing up RAM

    # TIME_STAMP = time.strftime("%d%m%Y-%H%M%S")

    # name = "stdclf_vs_distlearn"
    name = data["data_tag"]

    logdir = data["logdir"] # directory to store run dumps and input data
    run_dump_dir = os.path.join(logdir, name)

    backup_dir = data["backup_dir"] # directory on HDD/remote server where you want to keep backup in background

    ex.observers.append(FileStorageObserver(run_dump_dir))


def make_dataloaders(train_set, val_set, test_set, batch_size, num_workers, train):
    shuffle = True if train else False

    dataloaders = {
        "train": DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers),
        "val": DataLoader(dataset=val_set, shuffle=False, batch_size=batch_size, num_workers=num_workers),
        "test": DataLoader(dataset=test_set, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    }

    return dataloaders


@ex.capture
def data_setup(task, train, train_on_onmfld, OFF_MFLD_LABEL, batch_size, num_workers, num_mflds, tgtname, ftname, ram_efficient, on_mfld_noise, test_off_mfld, data):

    train_set, val_set, test_set = initialise_data()
    
    # delete attributes not required in training to save RAM
    attr_name_map = {
        "points": "all_points",
        "distances": "all_distances",
        "actual_distances": "all_actual_distances",
        "normed_points": "normed_all_points",
        "normed_distances": "normed_all_distances",
        "normed_actual_distances": "normed_all_actual_distances",
        "classes": "class_labels"
    }
    tgt_attr = attr_name_map[tgtname]
    ft_attr = attr_name_map[ftname]
    if ram_efficient:
        for dataset in [train_set, val_set, test_set]:
            delete_attrs = list()
            attrs = vars(dataset)
            for attr_name in attrs:
                if isinstance(attrs[attr_name], Iterable) and attr_name not in [tgt_attr, ft_attr, "class_labels", "all_distances", "all_actual_distances", "normed_all_distances", "normed_all_actual_distances"]:
                    delete_attrs.append(attr_name)
                # "S1" and "S2" not used in training when they are present so remove them
                elif "S1" in attr_name or "S2" in attr_name:
                    delete_attrs.append(attr_name)

            for attr_name in delete_attrs:
                # print(attr_name)
                delattr(dataset, attr_name)

    if task == "clf" and train_on_onmfld:
        idx = 0
        for dataset in [train_set, val_set, test_set]:
            for attr in ["all_points", "all_distances", "normed_all_points", "normed_all_distances", "class_labels"]:
                if not hasattr(dataset, attr):
                    continue
                if isinstance(dataset, manifold.Manifold):
                    setattr(dataset.genattrs, attr, getattr(dataset.genattrs, attr)[dataset.genattrs.class_labels != OFF_MFLD_LABEL])
                else:
                    if attr == "normed_all_points" and idx > 0:
                        
                        if not test_off_mfld:
                            noise_mat = torch.randn(getattr(dataset, attr)[dataset.class_labels != OFF_MFLD_LABEL].shape)
                            noise_mat = on_mfld_noise * (noise_mat / torch.norm(noise_mat, p=2, dim=1).reshape(-1, 1))
                            setattr(dataset, attr, getattr(dataset, attr)[dataset.class_labels != OFF_MFLD_LABEL] + noise_mat)
                        else:
                            noise_mat = torch.randn(getattr(dataset, attr).shape)
                            noise_mat = on_mfld_noise * (noise_mat / torch.norm(noise_mat, p=2, dim=1).reshape(-1, 1))
                            setattr(dataset, attr, getattr(dataset, attr) + noise_mat)
                            new_class_labels = getattr(dataset, "class_labels").clone()
                            tmp_idx0 = (new_class_labels == 0).nonzero(as_tuple=True)[0].item()
                            new_class_labels[:tmp_idx0] = 0
                            new_class_labels[new_class_labels == OFF_MFLD_LABEL] = 1
                            setattr(dataset, attr, new_class_labels)
                    else:
                        setattr(dataset, attr, getattr(dataset, attr)[dataset.class_labels != OFF_MFLD_LABEL])
            idx += 1
    elif (task == "regression") or (task == "clf" and not train_on_onmfld):
        for dataset in [train_set, val_set, test_set]:
            if isinstance(dataset, manifold.Manifold):
                
                min_dists = None
                argmin_mfld = None
                if num_mflds > 1:
                    min_dists, argmin_mfld = torch.min(dataset.genattrs.all_distances, axis=1)
                else:
                    # for single manifold
                    min_dists, argmin_mfld = dataset.genattrs.all_distances, torch.zeros_like(dataset.genattrs.all_distances)

                for i in range(dataset.genattrs.N):
                    if min_dists[i] < dataset.genattrs.D:
                        dataset.genattrs.class_labels[i] = argmin_mfld[i]
                    else:
                        dataset.genattrs.class_labels[i] = OFF_MFLD_LABEL
            else:
                
                min_dists = None
                argmin_mfld = None
                distance_attr = getattr(dataset, tgt_attr)
                if num_mflds > 1:
                    min_dists, argmin_mfld = torch.min(distance_attr, axis=1)
                else:
                    # for single manifold
                    min_dists, argmin_mfld = distance_attr, torch.zeros_like(distance_attr)

                for i in range(dataset.N):
                    if min_dists[i] < dataset.D:
                        dataset.class_labels[i] = argmin_mfld[i]
                    else:
                        dataset.class_labels[i] = OFF_MFLD_LABEL

    dataloaders = make_dataloaders(train_set, val_set, test_set, batch_size, num_workers, train)
    # shuffle = True if train else False

    # dataloaders = {
    #     "train": DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers),
    #     "val": DataLoader(dataset=val_set, shuffle=False, batch_size=batch_size, num_workers=num_workers),
    #     "test": DataLoader(dataset=test_set, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    # }

    datasets = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }
    return datasets, dataloaders

@ex.capture
def run_training(num_epochs, task, loss_func, lr, warmup,\
     cooldown, cuda, ftname, tgtname, name, save_dir, debug, _log, online):


    device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and cuda is not None else "cpu")
    
    datasets, dataloaders = data_setup()

    model, scheduler_state_dict, optimizer_state_dict = initialise_model()
    # print(model)
    loss_function = lmd.loss_funcs[loss_func]

    if task == "clf":
        loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler_params = {"warmup": warmup, "cooldown": cooldown}
    lr_sched_factor = lambda epoch: epoch / (scheduler_params["warmup"]) if epoch <= scheduler_params["warmup"] else (1 if epoch > scheduler_params["warmup"] and epoch < scheduler_params["cooldown"] else max(0, 1 + (1 / (scheduler_params["cooldown"] - num_epochs)) * (epoch - scheduler_params["cooldown"])))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_sched_factor)

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
    
    model, optimizer, scheduler, _, _ = lmd.train(model, optimizer, loss_function,\
        dataloaders, device, save_dir, scheduler, feature_name=ftname, target_name=tgtname,\
        num_epochs=num_epochs, task=task, name=name, scheduler_params=scheduler_params, \
        specs_dict=None, debug=debug, online=online)

    return model, optimizer, scheduler, datasets, dataloaders

@ex.capture
def log_clf_dict(_run, clf_report_dict, split="train", on_mfld=True):
    
    if "accurcy" in clf_report_dict:
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
    
    if model is None:
        model = initialise_model()

    if dataloaders is None:
        dataloaders, datasets = data_setup(train=False)

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
        # in dataset >= 2 ,that is, when dataset class lies in
        # `datagen.synthetic.multiple`
        #
        # TODO: maybe ammend multi-manifold datasets to avoid this?
        if num_classes >= 2:
            if task == "regression":
                thresh = datasets[split].D / datasets[split].norm_factor
                _, _, _, _, clf_report_dict = argmin_dist_clf(all_logits, all_targets, datasets[split].class_labels, thresh=thresh)
                log_clf_dict(_run, clf_report_dict, split, on_mfld=True)
                _, _, _, _, clf_report_dict = argmin_dist_clf(all_logits, all_targets, datasets[split].class_labels, thresh=thresh, on_mfld=False)
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
                def disp_stdclf_results(all_logits, all_targets, split, train_on_onmfld, on_mfld=True):

                    msg = "* On Manifold Results\n" if on_mfld else "* Results\n"
                    pred_labels = torch.max(all_logits, axis=1)[1]
                    true_labels = all_targets

                    target_names = ["S" + str(i) for i in range(all_logits.shape[1])]
                    labels = np.arange(all_logits.shape[1])

                    if (not train_on_onmfld) and on_mfld:
                        """
                        if training has been done on on-mfld. points but results are needed over
                        all of the dataset, then off-mfld. points should be removed, since the
                        network cannot predict an off manifold label
                        """
                        pred_labels = pred_labels[true_labels != OFF_MFLD_LABEL]
                        true_labels = true_labels[true_labels != OFF_MFLD_LABEL]
                    
                    if not train_on_onmfld:
                        """
                        training is done on off manifold samples and
                        then off manifold label can be predicted and 
                        therefore needs to be included in labels
                        """
                        target_names = target_names[:-1] + ["off manifold"]
                    

                    clf_report_dict = classification_report(true_labels, pred_labels, target_names=target_names, labels=labels, output_dict=True, zero_division=0)
                    print(msg)
                    print(classification_report(true_labels, pred_labels, target_names=target_names, labels=labels, zero_division=0))
                    log_clf_dict(_run, clf_report_dict, split, on_mfld)

                
                disp_stdclf_results(all_logits, all_targets, split, train_on_onmfld, on_mfld=True)

                if not train_on_onmfld:
                    disp_stdclf_results(all_logits, all_targets, split, train_on_onmfld, on_mfld=False)



    if analysis and input_size == 2:
        
        # k = data["data_params"]["k"]
        # n = data["data_params"]["n"]

        low = torch.min(datasets["test"].normed_all_points) - 0.1
        high = torch.max(datasets["test"].normed_all_points) + 0.1

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

        THRESH = datasets["train"].D / datasets["train"].norm_factor

        if task == "regression": gen_pred_classes[torch.min(gen_2d_logits, axis=1)[0] >= THRESH] = OFF_MFLD_LABEL

        col = ["blue", "green", "yellow"]

        plt.figure(figsize=(6, 6))

        for i in range(len(col)):
            plt.scatter(gen_2d_data[gen_pred_classes.numpy() == i, 0].numpy(), gen_2d_data[gen_pred_classes.numpy() == i, 1].numpy(), s=0.01, c=col[i], label=i)
            if (i < 2):
                plt.scatter(datasets["test"].normed_all_points[datasets["test"].class_labels == i, 0].numpy(), datasets["test"].normed_all_points[datasets["test"].class_labels == i, 1].numpy(), c=col[i], s=0.1)

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
def main(train, logdir, data, name, _log, _run, backup_dir):

    model = None
    optimizer = None
    scheduler = None
    datasets = None
    dataloaders = None

    
    save_dir = os.path.join(logdir, name, _run._id)
    _log.info("Dump being stored in: {}".format(save_dir))
    plot_dir = None
    if data["data_params"]["train"]["n"] in [2, 3]:
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
        logits_and_targets = run_eval(model=model, datasets=datasets, dataloaders=dataloaders, save_dir=save_dir, plotdir=plot_dir)

    
    if backup_dir is not None:
        src_dir = os.path.join(logdir, name)
        dest_dir = os.path.join(backup_dir, name)
        bkup_cmd = "rsync -avzr {src} {dest}".format(src=src_dir, dest=dest_dir)
        bkup_cmd_list = bkup_cmd.split()
        sync_sanity_check_file = os.path.join(save_dir, "rsync_bkup_sanity_check_cout.txt")

        with open(sync_sanity_check_file, "w") as f:
            subprocess.Popen(bkup_cmd_list, stdout=f)


# if __name__ == '__main__':
#     # lmd.init()
#     run = ex.run()