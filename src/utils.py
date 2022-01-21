import os
import random
import logging

import torch
import numpy as np


def seed_everything(manual_seed):
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    return manual_seed

def init_logger(name):
    logger = logging.getLogger(name)
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s -- [%(levelname).1s] %(name)s >> %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("INFO")
    return logger

def make_new_res_dir(parent_path, fmt_str, exist_ok=True, duplicate=True, *args):
    new_dir_name = fmt_str.format(*args)
    new_dir_base_path = os.path.join(parent_path, new_dir_name)
    new_dir_path = new_dir_base_path
    uniq = 1
    if not duplicate and os.path.exists(new_dir_path):
        raise RuntimeError("directory {} exists: either enable duplicate or delete directory".format(new_dir_path))
    while os.path.exists(new_dir_path):
        new_dir_path = new_dir_base_path + "_{}".format(uniq)
        uniq+=1
    os.makedirs(new_dir_path, exist_ok=exist_ok)
    return new_dir_path


def make_general_cm(true_labels, pred_labels, pct=False, output_dict=False):
    """
    returns general confusion matrix, rows are true labels

    in dictionary formatfirst-level keys are the true labels, 
    second-level keys are the pred labels

    :param true_labels: array containing true labels 
    :type true_labels: numpy.ndarray
    :param pred_labels: array containing pred_labels 
    :type pred_labels: numpy.ndarray
    :param pct: return values as a percentage
    :type pct: bool
    :param output_dict: return output as a dict
    :type output_dict: bool
    """
    uniq_true_labels = np.unique(true_labels).astype(np.int64)
    uniq_pred_labels = np.unique(pred_labels).astype(np.int64)

    cmatrix = np.zeros((uniq_true_labels.shape[0], uniq_pred_labels.shape[0]))
    cm = dict()
    for tlab_idx in range(uniq_true_labels.shape[0]):
        tlab = uniq_true_labels[tlab_idx]
        tidx = true_labels == tlab
        if tlab not in cm:
            cm[tlab] = dict()
        for plab_idx in range(uniq_pred_labels.shape[0]):
            plab = uniq_pred_labels[plab_idx]
            pidx = pred_labels[tidx] == plab
            if plab not in cm[tlab]:
                cm[tlab][plab] = sum(pidx).float().item()
                cmatrix[tlab_idx, plab_idx] = sum(pidx).float().item()
                if pct:
                    cm[tlab][plab]/=tidx.shape[0]
                    cmatrix[tlab_idx][plab_idx]/=tidx.shape[0]
    if output_dict:
        return cm
    return cmatrix

    