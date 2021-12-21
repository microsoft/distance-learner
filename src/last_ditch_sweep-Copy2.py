import os
os.chdir("/data/adv_geom/src/")
print(os.path.dirname(os.path.realpath(__file__)))
import sys
import json
import pickle
import random

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from tqdm import tqdm

from datagen.synthetic.multiple.concentricspheres import ConcentricSpheres
from expB.myNNs import MTMLPwithNormalisation, MLP, MLPwithNormalisation
from expD.attacks import pgd_l2_mfld_clf_attack, pgd_l2_mfld_dist_attack, pgd_l2_cls, pgd_dist, pgd_linf_rand



# data_dir = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/rdm_concspheres_k2n500/data/"
data_dir = "/data/k500n500_testsplits/"
# data_dir = "/data/k100n100_testsplits/"


# train_set, val_set, test_set = ConcentricSpheres.load_splits(data_dir)
test_set = ConcentricSpheres()
test_set.load_data(os.path.join(data_dir, "test"))


shuffle = False
# batch_size = 1
batch_size = 512
num_workers = 8

dataloaders = {
#     "train": DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers),
#     "val": DataLoader(dataset=val_set, shuffle=False, batch_size=batch_size, num_workers=num_workers),
    "test": DataLoader(dataset=test_set, shuffle=False, batch_size=batch_size, num_workers=num_workers)
}



input_size = 500
output_size = 2

hidden_sizes = [1024] * 4

use_tanh = False
use_relu = False
weight_norm = False

cuda = 0
device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and cuda is not None else "cpu")


# Standard Classifier model
# clf_init_wts = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/rdm_concspheres_k2n500/8/models/ckpt.pth"
clf_init_wts = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/rdm_concspheres_k500n500_noninfdist/2/models/ckpt.pth"
# clf_init_wts = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/rdm_concspheres_k100n100_noninfdist/2/models/ckpt.pth"



# stdclf_model = MTMLPwithNormalisation(input_size=input_size,\
#          output_size=output_size, hidden_sizes=hidden_sizes,\
#          use_tanh=use_tanh, use_relu=use_relu, weight_norm=weight_norm)
# hidden_sizes = [1024] * 2
# stdclf_model = MLP(input_size=input_size,\
#          output_size=output_size, hidden_sizes=hidden_sizes,\
#          use_tanh=use_tanh, use_relu=use_relu, weight_norm=weight_norm)

hidden_sizes = [1024] * 2
stdclf_model = MLPwithNormalisation(input_size=input_size,\
         output_size=output_size, hidden_sizes=hidden_sizes,\
         use_tanh=use_tanh, use_relu=use_relu, weight_norm=weight_norm)


stdclf_model.load_state_dict(torch.load(clf_init_wts)["model_state_dict"])
stdclf_model.to(device)


# Distance Learner model
hidden_sizes = [512] * 4
# distlearn_init_wts = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/rdm_concspheres_k100n100_noninfdist/1/models/ckpt.pth"
distlearn_init_wts = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/rdm_concspheres_k500n500_noninfdist/1/models/ckpt.pth"


distlearn_model = MTMLPwithNormalisation(input_size=input_size,\
         output_size=output_size, hidden_sizes=hidden_sizes,\
         use_tanh=use_tanh, use_relu=use_relu, weight_norm=weight_norm)


distlearn_model.load_state_dict(torch.load(distlearn_init_wts)["model_state_dict"])
distlearn_model.to(device)


num_neg = np.floor(dataloaders["test"].dataset.N / 2).astype(np.int64).item()
num_onmfld = dataloaders["test"].dataset.N - num_neg

stdclf_pred = torch.zeros(num_onmfld, 2)
dl_pred = torch.zeros(num_onmfld, 2)
stdclf_pred_ma = torch.zeros(num_onmfld, 2)
dl_pred_ma = torch.zeros(num_onmfld, 2)
stdclf_pred_ma_dl = torch.zeros(num_onmfld, 2)
dl_pred_ma_dl = torch.zeros(num_onmfld, 2)

all_deltas_ma = torch.zeros(num_onmfld, dataloaders["test"].dataset.n)
all_deltas_dl_ma = torch.zeros(num_onmfld, dataloaders["test"].dataset.n)




eps = 0.1
eps_iter = 5e-3
nb_iter = 100
norm = 2

sweep_logs = {}

for eps_iter in np.arange(5e-3, 0.081, 5e-3):
    
    stdclf_pred = torch.zeros(num_onmfld, 2)
    dl_pred = torch.zeros(num_onmfld, 2)
    stdclf_pred_ma = torch.zeros(num_onmfld, 2)
    dl_pred_ma = torch.zeros(num_onmfld, 2)
    stdclf_pred_ma_dl = torch.zeros(num_onmfld, 2)
    dl_pred_ma_dl = torch.zeros(num_onmfld, 2)

    all_deltas_ma = torch.zeros(num_onmfld, dataloaders["test"].dataset.n)
    all_deltas_dl_ma = torch.zeros(num_onmfld, dataloaders["test"].dataset.n)
    
    start = 0
    end = 0
    all_losses = list()
    all_losses_dl = list() 
    
    for (i, batch) in tqdm(enumerate(dataloaders["test"])):

        inputs = batch["normed_points"]
        true_distances = batch["normed_actual_distances"]
        true_classes = batch["classes"]

        # experiment was performed on points 'exactly' on the manifold.
        # in our dataset, these points are those with class labels != 2
        inputs = inputs[true_classes != 2].to(device)
        true_distances = true_distances[true_classes != 2].to(device)
        true_classes = true_classes[true_classes != 2].to(device)
        end = start + inputs.shape[0]

        if inputs.shape[0] == 0:
            continue

        center = torch.from_numpy(test_set.fix_center).float().to(device)
    #     deltas_ma, losses = pgd_l2_mfld_clf_attack(stdclf_model, inputs - center, true_classes, greedy=True)
    #     deltas_ma_dl, losses_dl = pgd_l2_mfld_dist_attack(distlearn_model, inputs - center, true_distances, greedy=True)
    #     deltas_ma, losses = pgd_l2_cls(stdclf_model, inputs, true_classes, epsilon=0.1, num_iter=200)
        adv_eg = projected_gradient_descent(model_fn=stdclf_model,\
                               x=inputs, \
                               eps=eps, \
                               eps_iter=eps_iter, \
                               nb_iter=nb_iter, \
                               norm=norm, y=true_classes)
    #     adv_eg = pgd_l2_mfld_clf_attack(stdclf_model, inputs, true_classes, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm)
        deltas_ma = adv_eg - inputs
        deltas_ma_dl = pgd_dist(distlearn_model, inputs, true_distances, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm) 
    #     deltas_ma_dl = pgd_l2_mfld_dist_attack(distlearn_model, inputs, true_distances, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm)
    #     deltas_ma = all_deltas_ma[start:end].to(device)
    #     all_losses.append(losses)
    #     all_losses_dl.append(losses_dl)
    #     inputs_ma = inputs + deltas_ma
        inputs_ma = adv_eg
    #     inputs_ma_dl = inputs + deltas_ma_dl
        inputs_ma_dl = deltas_ma_dl
        all_deltas_ma[start:end] = deltas_ma
        all_deltas_dl_ma[start:end] = deltas_ma_dl - inputs

    #     x_fgm = fast_gradient_method(net, x, FLAGS.eps, np.inf)
    #     x_pgd = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
        with torch.no_grad():
            stdclf_model.eval()
            y_clf_pred = stdclf_model(inputs)
            y_clf_pred_ma = stdclf_model(inputs_ma)
            y_clf_pred_ma_dl = stdclf_model(inputs_ma_dl)

            distlearn_model.eval()
            y_dl_pred = distlearn_model(inputs)
            y_dl_pred_ma = distlearn_model(inputs_ma)
            y_dl_pred_ma_dl = distlearn_model(inputs_ma_dl)



        stdclf_model.train()
        distlearn_model.train()

        stdclf_pred[start:end] = y_clf_pred.detach().cpu()
        stdclf_pred_ma[start:end] = y_clf_pred_ma.detach().cpu()
        stdclf_pred_ma_dl[start:end] = y_clf_pred_ma_dl.detach().cpu()
        dl_pred[start:end] = y_dl_pred.detach().cpu()
        dl_pred_ma[start:end] = y_dl_pred_ma.detach().cpu()
        dl_pred_ma_dl[start:end] = y_dl_pred_ma_dl.detach().cpu()

        start = end

    if eps_iter not in sweep_logs:
        sweep_logs[eps_iter] = dict()
    
    print("eps_iter:", eps_iter)
    dict_for_epsiter = dict()
    print("Std. Clf. on actual test set")
    stdclf_raw_report = classification_report(test_set.class_labels[test_set.class_labels != 2], torch.max(stdclf_pred, 1)[1], output_dict=True)
    dict_for_epsiter["stdclf_raw_report"] = stdclf_raw_report
    
    print("Std. Clf. on perturbed test set by CLF attack")
    stdclf_clfadv_report = classification_report(test_set.class_labels[test_set.class_labels != 2], torch.max(stdclf_pred_ma, 1)[1], output_dict=True)
    dict_for_epsiter["stdclf_clfadv_report"] = stdclf_clfadv_report
    
    print("Std. Clf. on perturbed test set by DL attack")
    stdclf_dladv_report = classification_report(test_set.class_labels[test_set.class_labels != 2], torch.max(stdclf_pred_ma_dl, 1)[1], output_dict=True)
    dict_for_epsiter["stdclf_dladv_report"] = stdclf_dladv_report
    
    print("Dist. Learner on actual test set")
    dl_pred_labels = torch.min(dl_pred, 1)[1]

    # dl_pred_labels[torch.min(dl_pred, 1)[0] > (test_set.D / test_set.norm_factor)] = 2
    distlearn_raw_report = classification_report(test_set.class_labels[test_set.class_labels != 2], dl_pred_labels, output_dict=True)
    dict_for_epsiter["distlearn_raw_report"] = distlearn_raw_report
    
    print("Dist. Learner on perturbed test set by CLF attack")
    # dl_labels_on_perturbed_set = 
    dl_pred_labels_ma = torch.min(dl_pred_ma, 1)[1]
    # dl_pred_labels_ma[torch.min(dl_pred_ma, 1)[0] > 0.07] = 2

    # dl_pred_labels_ma[torch.min(dl_pred_ma, 1)[0] > (test_set.D / test_set.norm_factor)] = 2
    distlearn_clfadv_report = classification_report(test_set.class_labels[test_set.class_labels != 2], dl_pred_labels_ma, output_dict=True)
    dict_for_epsiter["distlearn_clfadv_report"] = distlearn_clfadv_report
    
    print("Dist. Learner on perturbed test set by DL attack")
    # dl_labels_on_perturbed_set = 
    dl_pred_labels_ma_dl = torch.min(dl_pred_ma_dl, 1)[1]
    # dl_pred_labels_ma_dl[torch.min(dl_pred_ma_dl, 1)[0] > 0.08] = 2
    # dl_pred_labels_ma_dl[torch.min(dl_pred_ma_dl, 1)[0] > (test_set.D / test_set.norm_factor)] = 2
    distlearn_dladv_report = classification_report(test_set.class_labels[test_set.class_labels != 2], dl_pred_labels_ma_dl, output_dict=True)
    dict_for_epsiter["distlearn_dladv_report"] = distlearn_dladv_report
    
    sweep_logs[eps_iter] = dict_for_epsiter

    
with open("sweep_logs2_100iters.json", "w") as f:
    json.dump(sweep_logs, f)