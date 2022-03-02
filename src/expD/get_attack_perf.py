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
import shutil
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
logger = init_logger()

# attach it to the experiment
ex.logger = logger


@ex.config
def config(attack, input_files):

    cuda = 0
    device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and cuda is not None else "cpu")

    num_workers = 8
    OFF_MFLD_LABEL = 2

    batch_size = 512
    use_split = "test" # data split to generate perturbations from

    th_analyze = np.arange(1e-2, 1.6e-1, 1e-2) # if model is distance learner, then thresholds to analyse performance
    # th_analyze = np.array([1e-2])
    th_analyze = np.append(th_analyze, np.inf)

    debug = True
    clean = False

    # dump_dir = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/attack_perfs_on_runs"
    dump_dir = "/data/dumps/expC_dist_learner_for_adv_ex/rdm_concspheres_test/attack_perfs_on_runs"
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
def load_data_for_run(run_parent_dir, run_config, batch_size, num_workers, use_split="test"):

    # loading the data
    mtype = run_config["data"]["mtype"]
    data_class = datagen.dtype[mtype]
    data_dir = os.path.join(run_parent_dir, "data")
    # train_set, val_set, test_set = data_class.load_splits(data_dir)
    dataset = data_class()
    dataset.load_data(os.path.join(data_dir, use_split))
    dataloaders = {
        use_split: DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    }

    # dataloaders = {
    #     "val": DataLoader(dataset=val_set, shuffle=False, batch_size=batch_size, num_workers=num_workers),
    #     "test": DataLoader(dataset=test_set, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    # }
    # if train_set is not None:
    #     dataloaders["train"] = DataLoader(dataset=train_set, shuffle=False, batch_size=batch_size, num_workers=num_workers)

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
    _log.info("is data loaded: {}".format(not (dataloaders is None)))
    if dataloaders is None:
        _log.info("loading data for run from parent directory: {} ...".format(run_parent_dir))
        dataloaders = load_data_for_run(run_parent_dir=run_parent_dir, run_config=run_config)
        _log.info("data loaded")
    else:
        _log.info("data already loaded")

    task = "dist" if run_config["task"] == "regression" else "clf"
    _log.info("task: {}".format(task))
    _log.info("ftname: {}".format(run_config["ftname"]))
    _log.info("tgtname: {}".format(run_config["tgtname"]))


    dataset = dataloaders[use_split].dataset
    data_param_dict = {
        "run_tag": run_config["data"]["data_tag"],
        "k": dataset.k,
        "n": dataset.n,
        "max_norm": dataset.max_norm,
        "N": dataset.N
    }

    if "train" in dataloaders:
        data_param_dict["train.N"] = run_config["data"]["train"]["N"]

    attack_param_names = list(attack.keys())
    attack_param_vals = list(attack.values())

    result_tag = ""
    for i in range(len(attack_param_names)):    
        result_tag += attack_param_names[i] + "={}"
        if i < len(attack_param_names) - 1:
            result_tag += ","

    result_tag += ",task={}".format(task)

    for comb in product(*attack_param_vals):

        attack_param_dict = {attack_param_names[i]: comb[i] for i in range(len(attack_param_names))}
        atk_flavor = attack_param_dict["atk_flavor"]
        atk_routine = attack_param_dict["atk_routine"]
        eps = attack_param_dict["eps"]
        eps_iter = attack_param_dict["eps_iter"]
        nb_iter = attack_param_dict["nb_iter"]
        norm = attack_param_dict["norm"]
        verbose = attack_param_dict["verbose"]
        restarts = attack_param_dict["restarts"]

        attack_fn, atk_routine = get_atk(atk_flavor, task, atk_routine)
        # so that if input atk_routine is unavailable, the one used is captured
        attack_param_dict["atk_routine"] = atk_routine

        result_parent_dir = os.path.join(run_dir, "attack_perf")
        # _log.info(result_tag)
        result_dir = make_new_res_dir(result_parent_dir, result_tag, True, True, atk_flavor, atk_routine, eps, eps_iter, nb_iter, norm, restarts, verbose)
        _log.info("perturbed ex will be dumped in: {}".format(result_dir))

        logits_of_pb_ex, all_pb_ex, all_deltas, logits_of_raw_ex, all_targets = attack_model(dataloaders=dataloaders,\
            model_fn=model_fn, attack_fn=attack_fn, atk_flavor=atk_flavor, atk_routine=atk_routine, task=task, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter,\
            norm=norm, verbose=verbose, restarts=restarts, ftname=run_config["ftname"], tgtname=run_config["tgtname"])

        out_fn = os.path.join(result_dir, "logits_and_advex.pt")
        if not debug:
            torch.save({
                "logits_of_pb_ex": logits_of_pb_ex,
                "all_pb_ex": all_pb_ex,
                "all_deltas": all_deltas,
                "logits_of_raw_ex": logits_of_raw_ex,
                "all_targets": all_targets
            }, out_fn)

        result_entries, min_dist_to_pb_raw_vals, min_dist_to_pb_raw_idx = calc_attack_perf(
            inp_dir=inp_dir, \
            dataset=dataset, \
            all_pb_ex=all_pb_ex, \
            all_targets=all_targets, \
            logits_of_pb_ex=logits_of_pb_ex, \
            logits_of_raw_ex=logits_of_raw_ex, \
            th_analyze=th_analyze, \
            OFF_MFLD_LABEL=OFF_MFLD_LABEL, \
            attack_param_dict=attack_param_dict, \
            data_param_dict=data_param_dict, \
            task=task, \
            ftname=run_config["ftname"], \
            tgtname=run_config["tgtname"], \
            result_dir=result_dir)
        result_container.extend(result_entries)

        if not debug:
            out_fn = os.path.join(result_dir, "min_dist_to_pb_raw.pt")
            torch.save({
                "values": min_dist_to_pb_raw_vals,
                "indices": min_dist_to_pb_raw_idx
            }, out_fn)

    return result_container
    
@ex.capture
def calc_attack_perf(inp_dir, dataset, all_pb_ex, all_targets, logits_of_pb_ex, logits_of_raw_ex,\
     th_analyze, OFF_MFLD_LABEL, attack_param_dict, data_param_dict, task, ftname, tgtname, result_dir, use_split, _log):

    results = list()

    if task == "dist":
        distance_scatter_plot_dir = os.path.join(result_dir, "distance_scatter_plots_{}".format(use_split))
        os.makedirs(distance_scatter_plot_dir, exist_ok=True)
    abs_cm_plot_dir = os.path.join(result_dir, "abs_cm_plots_{}".format(use_split))
    os.makedirs(abs_cm_plot_dir, exist_ok=True)
    pct_cm_plot_dir = os.path.join(result_dir, "pct_cm_plots_{}".format(use_split))
    os.makedirs(pct_cm_plot_dir, exist_ok=True)

    onmfld_pts = dataset.normed_all_points[dataset.class_labels != OFF_MFLD_LABEL]

    def get_closest_onmfld_pt(onmfld_pts, all_pb_ex):
        pair_dist_pb_to_raw = torch.cdist(all_pb_ex, onmfld_pts)
        min_dist_pb_to_raw = torch.min(pair_dist_pb_to_raw, dim=1)
        min_dist_pb_to_raw_vals = min_dist_pb_to_raw.values
        min_dist_pb_to_raw_idx = min_dist_pb_to_raw.indices
        return min_dist_pb_to_raw_vals, min_dist_pb_to_raw_idx

    min_dist_pb_to_raw_vals, min_dist_pb_to_raw_idx = get_closest_onmfld_pt(onmfld_pts, all_pb_ex)

    if task != "dist":
        # _log.info("what is task here?: {}".format(task))
        # _log.info("what is task != 'dist': {}".format(task != "dist"))

        # for normal examples (will be helpful for comparison)
        true_classes = dataset.class_labels[dataset.class_labels != OFF_MFLD_LABEL]
        pred_classes = torch.max(logits_of_raw_ex, dim=1)[1]

        clf_report = classification_report(true_classes, pred_classes, output_dict=True)
        abs_cm = make_general_cm(true_classes, pred_classes, output_dict=False)
        abs_cm_fn = os.path.join(abs_cm_plot_dir, "abs_cm_{}.csv".format(use_split))
        abs_cm.to_csv(abs_cm_fn)
        pct_cm = make_general_cm(true_classes, pred_classes, pct=True, output_dict=False)        
        pct_cm_fn = os.path.join(pct_cm_plot_dir, "pct_cm_{}.csv".format(use_split))
        pct_cm.to_csv(pct_cm_fn)


        # for adversarial examples
        adv_true_classes = dataset.class_labels[dataset.class_labels != OFF_MFLD_LABEL][min_dist_pb_to_raw_idx]
        # assert (adv_true_classes == true_classes).all()
        adv_pred_classes = torch.max(logits_of_pb_ex, dim=1)[1]

        adv_clf_report = classification_report(adv_true_classes, adv_pred_classes, output_dict=True)
        adv_abs_cm = make_general_cm(adv_true_classes, adv_pred_classes, output_dict=False)
        adv_abs_cm_fn = os.path.join(abs_cm_plot_dir, "adv_abs_cm_{}.csv".format(use_split))
        adv_abs_cm.to_csv(adv_abs_cm_fn)
        adv_pct_cm = make_general_cm(adv_true_classes, adv_pred_classes, pct=True, output_dict=False)
        adv_pct_cm_fn = os.path.join(pct_cm_plot_dir, "adv_pct_cm_{}.csv".format(use_split))
        adv_pct_cm.to_csv(adv_pct_cm_fn)

        # form the result entry
        result_entry = copy.deepcopy(attack_param_dict)
        result_entry.update(**data_param_dict)

        stat_dict = {
            "inp_dir": inp_dir,
            "task": task,
            "ftname": ftname,
            "tgtname": tgtname,
            "thresh": np.nan,
            "adv_clf_report": adv_clf_report,
            "adv_abs_cm": adv_abs_cm_fn,
            "adv_pct_cm": adv_pct_cm_fn,
            "clf_report": clf_report,
            "abs_cm": abs_cm_fn,
            "pct_cm": pct_cm_fn,
            "distance_sct_plt": None,
            "result_dir": result_dir
        }
        result_entry.update(**stat_dict)
        results.append(result_entry)

        # plot the confusion matrices
        sns.heatmap(abs_cm, annot=True)
        plt.ylabel("True Labels")
        plt.xlabel("Pred Labels")
        plt.title("Abs CM")
        plt.savefig(os.path.join(abs_cm_plot_dir, "abs_cm_{}.png".format(use_split)))
        plt.clf()
        _log.info("abs confusion matrices written to: {}".format(abs_cm_plot_dir))

        sns.heatmap(pct_cm, annot=True)
        plt.ylabel("True Labels")
        plt.xlabel("Pred Labels")
        plt.title("Percentage CM")
        plt.savefig(os.path.join(pct_cm_plot_dir, "pct_cm_{}.png".format(use_split)))
        plt.clf()
        _log.info("pct confusion matrices written to: {}".format(pct_cm_plot_dir))
        
        sns.heatmap(adv_abs_cm, annot=True)
        plt.ylabel("True Labels")
        plt.xlabel("Pred Labels")
        plt.title("Adv Abs CM")
        plt.savefig(os.path.join(abs_cm_plot_dir, "adv_abs_cm_{}.png".format(use_split)))
        plt.clf()
        _log.info("abs confusion matrices written to: {}".format(abs_cm_plot_dir))

        sns.heatmap(adv_pct_cm, annot=True)
        plt.ylabel("True Labels")
        plt.xlabel("Pred Labels")
        plt.title("Adv Percentage CM")
        plt.savefig(os.path.join(pct_cm_plot_dir, "adv_pct_cm_{}.png".format(use_split)))
        plt.clf()
        _log.info("pct confusion matrices written to: {}".format(pct_cm_plot_dir))

    else:

        for th in th_analyze:
            
            # for normal examples (will be helpful for comparison)
            true_classes = dataset.class_labels[dataset.class_labels != OFF_MFLD_LABEL]
            pred_classes = torch.min(logits_of_raw_ex, dim=1)[1]
            pred_classes[torch.min(logits_of_raw_ex, dim=1)[0] > th] = OFF_MFLD_LABEL

            clf_report = classification_report(true_classes, pred_classes, output_dict=True)
            abs_cm = make_general_cm(true_classes, pred_classes, output_dict=False)
            abs_cm_fn = os.path.join(abs_cm_plot_dir, "abs_cm_{}_th={}.csv".format(use_split, th))
            abs_cm.to_csv(abs_cm_fn)
            pct_cm = make_general_cm(true_classes, pred_classes, pct=True, output_dict=False)        
            pct_cm_fn = os.path.join(pct_cm_plot_dir, "pct_cm_{}_th={}.csv".format(use_split, th))
            pct_cm.to_csv(pct_cm_fn)


            # for adversarial examples
            adv_true_classes = dataset.class_labels[dataset.class_labels != OFF_MFLD_LABEL][min_dist_pb_to_raw_idx]
            adv_true_preclasses = adv_true_classes.clone()
            adv_true_classes[min_dist_pb_to_raw_vals > th] = OFF_MFLD_LABEL
            adv_pred_classes = torch.min(logits_of_pb_ex, dim=1)[1]
            adv_pred_classes[torch.min(logits_of_pb_ex, dim=1)[0] > th] = OFF_MFLD_LABEL

            adv_clf_report = classification_report(adv_true_classes, adv_pred_classes, output_dict=True)
            adv_abs_cm = make_general_cm(adv_true_classes, adv_pred_classes, output_dict=False)
            adv_abs_cm_fn = os.path.join(abs_cm_plot_dir, "adv_abs_cm_{}_th={}.csv".format(use_split, th))
            adv_abs_cm.to_csv(adv_abs_cm_fn)
            adv_pct_cm = make_general_cm(adv_true_classes, adv_pred_classes, pct=True, output_dict=False)
            adv_pct_cm_fn = os.path.join(pct_cm_plot_dir, "adv_pct_cm_{}_th={}.csv".format(use_split, th))
            adv_pct_cm.to_csv(adv_pct_cm_fn)

            # forming result entry
            result_entry = copy.deepcopy(attack_param_dict)
            result_entry.update(**data_param_dict)
            stat_dict = {
                "inp_dir": inp_dir,
                "task": task,
                "ftname": ftname,
                "tgtname": tgtname,
                "thresh": th,
                "adv_clf_report": adv_clf_report,
                "adv_abs_cm": adv_abs_cm_fn,
                "adv_pct_cm": adv_pct_cm_fn,
                "clf_report": clf_report,
                "abs_cm": abs_cm_fn,
                "pct_cm": pct_cm_fn,
                "distance_sct_plt": os.path.join(distance_scatter_plot_dir, "pred_vs_gt_dists_all_{}_th={}.png".format(use_split, th)),
                "result_dir": result_dir
            }
            result_entry.update(**stat_dict)
            results.append(result_entry)

            # plot the distance scatter plot

            def plot_distance_scatter_plot(logits, targets, target_name, save_dir, th, tol=5e-2):
                with sns.axes_style("whitegrid"):
                    for idx in range(logits.shape[1]):
                        mask = np.abs(targets.numpy()[:, idx] - logits.numpy()[:, idx]) >= tol
                        plt.scatter(targets.numpy()[mask, idx], logits.numpy()[mask, idx], s=0.01, c="red")
                        plt.scatter(targets.numpy()[np.logical_not(mask), idx], logits.numpy()[np.logical_not(mask), idx], s=0.01, c="green")
                        # plt.plot(targets.numpy()[:, idx], targets.numpy()[:, idx])
                        # plt.plot(targets.numpy()[:, idx], targets.numpy()[:, idx] + th)
                        # plt.plot(targets.numpy()[:, idx], targets.numpy()[:, idx] - th)
                        if th < np.inf:
                            plt.axvline(x = th, color = 'b', label = 'th={}'.format(th))
                            plt.axhline(y = th, color = 'b', label = 'th={}'.format(th))
                        plt.gca().set_ylim(bottom=0)
                        plt.gca().set_xticks(np.arange(0, 1.1, 0.1))
                        plt.gca().set_yticks(np.arange(0, 1.1, 0.1))
                        plt.xlabel("gt distance ({})".format(target_name))
                        plt.ylabel("pred distance")
                        plt.title("gt vs. pred {}".format(target_name))
                        plt.savefig(os.path.join(save_dir, "pred_vs_gt_dists_S{}_{}_th={}.png".format(idx + 1, use_split, th)))
                        plt.clf()

                    mask = np.abs(targets.numpy().ravel() - logits.numpy().ravel()) >= 5e-2
                    plt.scatter(targets.numpy().ravel()[mask], logits.numpy().ravel()[mask], s=0.01, c="red")
                    plt.scatter(targets.numpy().ravel()[np.logical_not(mask)], logits.numpy().ravel()[np.logical_not(mask)], s=0.01, c="green")
                    # plt.plot(targets.numpy().ravel(), targets.numpy().ravel())
                    # plt.plot(targets.numpy().ravel(), targets.numpy().ravel() + th)
                    # plt.plot(targets.numpy().ravel(), targets.numpy().ravel() - th)
                    if th < np.inf:
                        plt.axvline(x = th, color = 'b', label = 'th={}'.format(th))
                        plt.axhline(y = th, color = 'b', label = 'th={}'.format(th))
                    plt.gca().set_ylim(bottom=0)
                    plt.gca().set_xticks(np.arange(0, 1.1, 0.1))
                    plt.gca().set_yticks(np.arange(0, 1.1, 0.1))
                    plt.xlabel("gt distance ({})".format(target_name))
                    plt.ylabel("pred distance")
                    plt.title("gt vs. pred {}, th={}".format(target_name, th))
                    plt.savefig(os.path.join(save_dir, "pred_vs_gt_dists_all_{}_th={}.png".format(use_split, th)))
                    plt.clf()

            targets_of_pb_ex = torch.zeros_like(logits_of_pb_ex)
            targets_of_pb_ex[np.arange(targets_of_pb_ex.shape[0]), adv_true_preclasses] = min_dist_pb_to_raw_vals
            targets_of_pb_ex[np.arange(targets_of_pb_ex.shape[0]), ~adv_true_preclasses] = dataset.M
            plot_distance_scatter_plot(logits_of_pb_ex, targets_of_pb_ex, tgtname, distance_scatter_plot_dir, th, tol=5e-2)
            _log.info("distance scatter plots written to: {}".format(distance_scatter_plot_dir))

            # plot the confusion matrices
            sns.heatmap(abs_cm, annot=True)
            plt.ylabel("True Labels")
            plt.xlabel("Pred Labels")
            plt.title("Abs CM th={}".format(th))
            plt.savefig(os.path.join(abs_cm_plot_dir, "abs_cm_{}_th={}.png".format(use_split, th)))
            plt.clf()
            _log.info("abs confusion matrices written to: {}".format(abs_cm_plot_dir))

            sns.heatmap(adv_pct_cm, annot=True)
            plt.ylabel("True Labels")
            plt.xlabel("Pred Labels")
            plt.title("Percentage CM th={}".format(th))
            plt.savefig(os.path.join(pct_cm_plot_dir, "pct_cm_{}_th={}.png".format(use_split, th)))
            plt.clf()
            _log.info("pct confusion matrices written to: {}".format(pct_cm_plot_dir))

            sns.heatmap(adv_abs_cm, annot=True)
            plt.ylabel("True Labels")
            plt.xlabel("Pred Labels")
            plt.title("Raw CM th={}".format(th))
            plt.savefig(os.path.join(abs_cm_plot_dir, "adv_raw_cm_{}_th={}.png".format(use_split, th)))
            plt.clf()
            _log.info("adv abs confusion matrices written to: {}".format(abs_cm_plot_dir))

            sns.heatmap(adv_pct_cm, annot=True)
            plt.ylabel("True Labels")
            plt.xlabel("Pred Labels")
            plt.title("Percentage CM th={}".format(th))
            plt.savefig(os.path.join(pct_cm_plot_dir, "adv_pct_cm_{}_th={}.png".format(use_split, th)))
            plt.clf()
            _log.info("adv pct confusion matrices written to: {}".format(pct_cm_plot_dir))
    
    return results, min_dist_pb_to_raw_vals, min_dist_pb_to_raw_idx


@ex.capture
def attack_model(_log, cuda, use_split, OFF_MFLD_LABEL, dataloaders, model_fn, attack_fn, atk_routine, atk_flavor, eps, eps_iter, nb_iter, norm, verbose, task, restarts, ftname, tgtname):

    _log.info("logging attack parameters")
    _log.info("atk_flavor={}".format(atk_flavor))
    _log.info("atk_routine={}".format(atk_routine))
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

    all_targets = torch.zeros(num_onmfld, num_classes)
    if task == "clf":
        all_targets = torch.zeros(num_onmfld).long()

    start = end = 0

    for (i, batch) in tqdm(enumerate(dl)):

        inputs = batch[ftname]
        targets = batch[tgtname]
        true_classes = batch["classes"]

        # experiment was performed on points 'exactly' on the manifold.
        # in our dataset, these points are those with class labels != 2
        inputs = inputs[true_classes != OFF_MFLD_LABEL]
        targets = targets[true_classes != OFF_MFLD_LABEL]
        true_classes = true_classes[true_classes != OFF_MFLD_LABEL]
        end = start + inputs.shape[0]

        if inputs.shape[0] == 0:
            continue
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        true_classes = true_classes.to(device)

        x = inputs
        y = targets

        adv_x = None
        if atk_routine == "chans":
            adv_x = attack_fn(model_fn=model_fn, x=x, y=y,\
             eps=eps, eps_iter=eps_iter, nb_iter=nb_iter,\
             norm=norm)
        else:
            adv_x = attack_fn(model_fn=model_fn, x=x, y=y,\
                eps=eps, eps_iter=eps_iter, nb_iter=nb_iter,\
                verbose=verbose, norm=norm, restarts=restarts)

        delta = adv_x - x
        # if task == "clf" and atk_routine == "chans":
        #     _log.info("shape of all_pb_ex: {}".format(all_pb_ex.shape))
        #     _log.info("shape of adv_x: {}".format(adv_x.shape))
        all_deltas[start:end] = delta
        all_pb_ex[start:end] = adv_x
        all_targets[start:end] = targets

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
        logits_of_raw_ex,
        all_targets
    )

@ex.capture
def attack_on_runs(inp_files, attack, th_analyze, use_split, OFF_MFLD_LABEL, dump_dir, _log):
    
    sep_results_for_all_runs_dir = os.path.join(dump_dir, "all_attack_perfs")
    os.makedirs(sep_results_for_all_runs_dir, exist_ok=True)
    
    all_results = list()
    run_parent_dirs = list()
    dataloaders = None
    for inp_dir in inp_files:
        parent_dir = os.path.abspath(os.path.join(inp_dir, os.pardir))
        run_config = load_run_config(inp_dir)
        if parent_dir not in run_parent_dirs or dataloaders is None:
            _log.info("loading data for run from parent directory: {} ...".format(parent_dir))
            dataloaders = load_data_for_run(parent_dir, run_config)
            _log.info("data loaded")
            run_parent_dirs.append(parent_dir)
        else:
            _log.info("data was loaded for a previous run")
        result_container = attack_and_eval_run(inp_dir, attack, th_analyze, use_split, OFF_MFLD_LABEL, _log, dataloaders=dataloaders)    
        
        run_task = run_config["task"]
        run_data_tag = run_config["data"]["data_tag"]
        run_result_tag = run_data_tag + "-" + run_task + ".json"
        result_for_run_fn = os.path.join(sep_results_for_all_runs_dir, run_result_tag)
        
        _log.info("saving result for run in: {}".format(result_for_run_fn))
        if os.path.exists(result_for_run_fn):
            _log.info("result file for run exists. loading...")
            with open(result_for_run_fn) as f:
                _log.info("result file for run loaded. adding new results...")
                result_container.extend(json.load(f))
        with open(result_for_run_fn, "w") as f:
            json.dump(result_container, f)
            _log.info("result file for run saved!")
        all_results.extend(result_container)
    return all_results

@ex.capture
def clean_incorrect_dumps(_log, inp_files):
    for inp_dir in inp_files:
        attack_perf_result_dir = os.path.join(inp_dir, "attack_perf")
        try:
            shutil.rmtree(attack_perf_result_dir)
            _log.info("removed directory: {}".format(attack_perf_result_dir))
        except:
            _log.info("could not remove directory: {}".format(attack_perf_result_dir))
       

@ex.automain
def main(attack, th_analyze, use_split, OFF_MFLD_LABEL, dump_dir, clean, _log, _run):

    inp_files = get_inp_fn()
    if not clean:
        all_results = attack_on_runs(inp_files, attack, th_analyze, use_split, OFF_MFLD_LABEL, dump_dir, _log)
        _log.info("dump dir: {}".format(dump_dir))
        os.makedirs(dump_dir, exist_ok=True)
        result_fn = os.path.join(dump_dir, _run._id, "all_attack_perfs_collated.json")
        with open(result_fn, "w") as f:
            json.dump(all_results, f)

        _log.info("result file created at: {}".format(result_fn))
    else:
        clean_incorrect_dumps(_log=_log, inp_files=inp_files)

    
