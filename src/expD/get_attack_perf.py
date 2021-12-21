"""
Given a model and its data split, and an adversarial attack
this measures the efficacy of the attack at various epsilons
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
import logging

from itertools import product

import numpy as np

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

from utils import *
from datagen import datagen
from expB import learn_mfld_distance as lmd
from attack_ingredients import attack_ingredient, get_atk
from inpfn_ingredients import get_inp_fn, inpfn_ingredient
from attacks import *

ex = Experiment("get_attack_perfs", ingredients=[attack_ingredient, inpfn_ingredient])

# set up a custom logger
logger = logging.getLogger("expD")
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s -- [%(levelname).1s] %(name)s >> %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

# attach it to the experiment
ex.logger = logger


@ex.config
def config(attack, input_files):

    cuda = 1
    device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and cuda is not None else "cpu")

    num_workers = 8
    OFF_MFLD_LABEL = 2

    batch_size = 512
    use_split = "test" # data split to generate perturbations from

    th_analyze = np.arange(1e-2, 1.6e-1, 1e-2) # if model is distance learner, then thresholds to analyse performance
    th_analyze = np.append(th_analyze, np.inf)

    debug = True

    dump_dir = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/attack_perfs_on_runs"
    ex.observers.append(FileStorageObserver(dump_dir))


def load_run_config(run_dir):

    # load the run config
    run_config_file = os.path.join(run_dir, "config.json")
    with open(run_config_file) as f:
        run_config = json.load(f)
    return run_config

def load_model_for_run(run_dir):
    
    run_config = load_run_config(run_dir)

    # loading the model
    model_ckpt = os.path.join(run_dir, "models", "ckpt.pth")
    model_params = run_config["model"]
    model_class_name = model_params["model_type"]
    model_class = lmd.model_type[model_class_name]
    model_fn = model_class(**model_params)
    model_fn.load_state_dict(torch.load(model_ckpt)["model_state_dict"])

    return model_fn

@ex.capture
def load_data_for_run(run_parent_dir, run_config, batch_size, num_workers):

    # loading the data
    mtype = run_config["data"]["mtype"]
    data_class = datagen.dtype[mtype]
    data_dir = os.path.join(run_parent_dir, "data")
    train_set, val_set, test_set = data_class.load_splits(data_dir)

    dataloaders = {
        "train": DataLoader(dataset=train_set, shuffle=False, batch_size=batch_size, num_workers=num_workers),
        "val": DataLoader(dataset=val_set, shuffle=False, batch_size=batch_size, num_workers=num_workers),
        "test": DataLoader(dataset=test_set, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    }

    return dataloaders

@ex.capture
def attack_and_eval_run(inp_dir, attack, th_analyze, use_split, OFF_MFLD_LABEL, _log, debug, dataloaders=None):
    
    result_container = list()

    run_dir = inp_dir
    run_parent_dir = os.path.abspath(os.path.join(run_dir, os.pardir))
    _log.info("working on: {}".format(run_dir))

    _log.info("loading model for run...")
    model_fn = load_model_for_run(run_dir=run_dir)
    _log.info("model loaded")
    _log.info("loading config for run...")
    run_config = load_run_config(run_dir=run_dir)
    _log.info("config loaded")
    _log.info("is data loaded: {}".format(dataloaders is None))
    if dataloaders is None:
        _log.info("loading data for run from parent directory: {} ...".format(run_parent_dir))
        dataloaders = load_data_for_run(run_parent_dir=run_parent_dir, run_config=run_config)
        _log.info("data loaded")
    else:
        _log.info("data already loaded")

    task = "dist" if run_config["task"] == "regression" else "clf"
    _log.info("task: {}".format(task))

    dataset = dataloaders[use_split].dataset
    data_param_dict = {
        "k": dataset.k,
        "n": dataset.n,
        "max_norm": dataset.max_norm,
        "N": dataset.N
    }

    attack_param_names = list(attack.keys())
    attack_param_vals = list(attack.values())

    result_tag = ""
    for i in range(len(attack_param_names)):    
        result_tag += attack_param_names[i] + "={}"
        if i < len(attack_param_names) - 1:
            result_tag += ","

    for comb in product(*attack_param_vals):

        attack_param_dict = {attack_param_names[i]: comb[i] for i in range(len(attack_param_names))}
        atk_flavor = attack_param_dict["atk_flavor"]
        atk_routine = attack_param_dict["atk_routine"]
        eps = attack_param_dict["eps"]
        eps_iter = attack_param_dict["eps_iter"]
        nb_iter = attack_param_dict["nb_iter"]
        norm = attack_param_dict["norm"]
        verbose = attack_param_dict["verbose"]

        attack_fn, atk_routine = get_atk(atk_flavor, task, atk_routine)
        # so that if input atk_routine is unavailable, the one used is captured
        attack_param_dict["atk_routine"] = atk_routine

        result_parent_dir = os.path.join(run_dir, "attack_perf")
        _log.info(result_tag)
        result_dir = make_new_res_dir(result_parent_dir, result_tag, True, True, atk_flavor, task, atk_routine, eps, eps_iter, nb_iter, norm, verbose)
        _log.info("perturbed ex will be dumped in: {}".format(result_dir))

        logits_of_pb_ex, all_pb_ex, all_deltas, logits_of_raw_ex = attack_model(dataloaders=dataloaders,\
            model_fn=model_fn, attack_fn=attack_fn, task=task, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm, verbose=verbose)

        out_fn = os.path.join(result_dir, "logits_and_advex.pt")
        if not debug:
            torch.save({
                "logits_of_pb_ex": logits_of_pb_ex,
                "all_pb_ex": all_pb_ex,
                "all_deltas": all_deltas,
                "logits_of_raw_ex": logits_of_raw_ex
            }, out_fn)

        result_entries = calc_attack_perf(inp_dir, dataset, all_pb_ex, logits_of_pb_ex, logits_of_raw_ex,\
             th_analyze, OFF_MFLD_LABEL, attack_param_dict, data_param_dict, task)
        result_container.extend(result_entries)

    return result_container
    
@ex.capture
def calc_attack_perf(inp_dir, dataset, all_pb_ex, logits_of_pb_ex, logits_of_raw_ex, th_analyze, OFF_MFLD_LABEL, attack_param_dict, data_param_dict, task):

    results = list()

    onmfld_pts = dataset[dataset.class_labels != OFF_MFLD_LABEL]
    pair_dist_pb_to_raw = torch.cdist(all_pb_ex, onmfld_pts)
    min_dist_pb_to_raw = torch.min(pair_dist_pb_to_raw, dim=1)
    min_dist_pb_to_raw_val = min_dist_pb_to_raw.values
    min_dist_pb_to_raw_idx = min_dist_pb_to_raw.indices

    if task != "dist":
        true_classes = dataset.class_labels[dataset.class_labels != OFF_MFLD_LABEL]
        pred_classes = torch.max(logits_of_raw_ex, dim=1)[1]

        clf_report = classification_report(true_classes, pred_classes, output_dict=True)
        raw_cm = make_general_cm(true_classes, pred_classes, output_dict=False)
        pct_cm = make_general_cm(true_classes, pred_classes, pct=True, output_dict=False)

        adv_true_classes = dataset.class_labels[dataset.class_labels != OFF_MFLD_LABEL][min_dist_pb_to_raw_idx]
        assert (adv_true_classes == true_classes).all()
        adv_pred_classes = torch.max(logits_of_pb_ex, dim=1)[1]

        clf_report = classification_report(true_classes, pred_classes, output_dict=True)
        raw_cm = make_general_cm(true_classes, pred_classes, output_dict=False)
        pct_cm = make_general_cm(true_classes, pred_classes, pct=True, output_dict=False)

        result_entry = attack_param_dict
        result_entry.update(**data_param_dict)
        stat_dict = {
            "inp_dir": inp_dir,
            "task": task,
            "thresh": np.nan,
            "clf_report": clf_report,
            "raw_cm": raw_cm,
            "pct_cm": pct_cm 
        }
        result_entry.update(**stat_dict)
        results.append(result_entry)

    else:

        for th in th_analyze:
            
            adv_true_classes = dataset.class_labels[dataset.class_labels != OFF_MFLD_LABEL][min_dist_pb_to_raw_idx]
            adv_true_classes[min_dist_pb_to_raw_val > th] = OFF_MFLD_LABEL
            pred_classes = torch.min(logits_of_raw_ex, dim=1)[1]

            clf_report = classification_report(adv_true_classes, pred_classes, output_dict=True)
            raw_cm = make_general_cm(adv_true_classes, pred_classes, output_dict=False)
            pct_cm = make_general_cm(adv_true_classes, pred_classes, pct=True, output_dict=False)

            result_entry = attack_param_dict
            result_entry.update(**data_param_dict)
            stat_dict = {
                "task": task,
                "thresh": th,
                "clf_report": clf_report,
                "raw_cm": raw_cm,
                "pct_cm": pct_cm 
            }
            result_entry.update(**stat_dict)
            results.append(result_entry)
    
    return results


@ex.capture
def attack_model(_log, cuda, use_split, OFF_MFLD_LABEL, dataloaders, model_fn, attack_fn, eps, eps_iter, nb_iter, norm, verbose, task):

    _log.info("logging attack parameters")
    _log.info("eps={}".format(eps))
    _log.info("eps_iter={}".format(eps_iter))
    _log.info("nb_iter={}".format(nb_iter))
    _log.info("norm={}".format(norm))
    _log.info("verbose={}".format(verbose))
    _log.info("task={}".format(task))

    device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and cuda is not None else "cpu")
    model_fn.to(device)

    dl = dataloaders[use_split]

    num_neg = np.floor(dl.dataset.N / 2).astype(np.int64).item()
    num_onmfld = dl.dataset.N - num_neg

    num_classes = dl.dataset.class_labels[dl.dataset.class_labels != OFF_MFLD_LABEL].max().item() + 1
    logits_of_raw_ex = torch.zeros(num_onmfld, num_classes)
    logits_of_pb_ex = torch.zeros(num_onmfld, num_classes)
    
    all_deltas = torch.zeros(num_onmfld, dl.dataset.normed_all_points.shape[1])
    all_pb_ex = torch.zeros(num_onmfld, dl.dataset.normed_all_points.shape[1])

    start = end = 0

    for (i, batch) in tqdm(enumerate(dl)):

        inputs = batch["normed_points"]
        true_distances = batch["normed_actual_distances"]
        true_classes = batch["classes"]

        # experiment was performed on points 'exactly' on the manifold.
        # in our dataset, these points are those with class labels != 2
        inputs = inputs[true_classes != OFF_MFLD_LABEL]
        true_distances = true_distances[true_classes != OFF_MFLD_LABEL]
        true_classes = true_classes[true_classes != OFF_MFLD_LABEL]
        end = start + inputs.shape[0]

        if inputs.shape[0] == 0:
            continue
        
        inputs = inputs.to(device)
        true_distances = true_distances.to(device)
        true_classes = true_classes.to(device)

        y = true_classes
        if task == "dist":
            y = true_distances

        x = inputs

        adv_x = attack_fn(model_fn=model_fn, x=x, y=y,\
             eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, verbose=verbose, norm=norm)
        delta = adv_x - x
        
        all_deltas[start:end] = delta
        all_pb_ex[start:end] = adv_x

        with torch.no_grad():
            model_fn.eval()
            logits_x = model_fn(x)
            logits_advx = model_fn(adv_x)
        model_fn.train()

        logits_of_raw_ex[start:end] = logits_x.detach().cpu()
        logits_of_pb_ex[start:end] = logits_advx.detach().cpu()

        start = end

    model_fn.to('cpu')
    return (
        logits_of_pb_ex,
        all_pb_ex,
        all_deltas,
        logits_of_raw_ex
    )

@ex.capture
def attack_on_runs(inp_files, attack, th_analyze, use_split, OFF_MFLD_LABEL, _log):
    all_results = list()
    run_parent_dirs = list()
    dataloaders = None
    for inp_dir in inp_files:
        parent_dir = os.path.abspath(os.path.join(inp_dir, os.pardir))
        if parent_dir not in run_parent_dirs or dataloaders is None:
            run_config = load_run_config(inp_dir)
            _log.info("loading data for run from parent directory: {} ...".format(parent_dir))
            dataloaders = load_data_for_run(parent_dir, run_config)
            _log.info("data loaded")
            run_parent_dirs.append(parent_dir)
        else:
            _log.info("data was loaded for a previous run")
        result_container = attack_and_eval_run(inp_dir, attack, th_analyze, use_split, OFF_MFLD_LABEL, _log, dataloaders=dataloaders)    
        all_results.extend(result_container)
    return all_results

@ex.automain
def main(attack, input_files, th_analyze, use_split, OFF_MFLD_LABEL, dump_dir, _log, _run):

    inp_files = get_inp_fn()
    all_results = attack_on_runs(inp_files, attack, th_analyze, use_split, OFF_MFLD_LABEL, _log)
    _log.info("dump dir: {}".format(dump_dir))
    os.makedirs(dump_dir, exist_ok=True)
    result_fn = os.path.join(dump_dir, _run._id, "all_attack_perfs.json")
    with open(result_fn, "w") as f:
        json.dump(all_results, result_fn)

    _log.info("result file created at: {}".format(result_fn))
    
