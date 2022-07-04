# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import re
import sys
import json
import copy
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

import torch

import seaborn as sns



import matplotlib
font = {'family' : 'sans-serif',
        'size'   : 14}

matplotlib.rc('font', **font)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "-res_dir", type=str, required=True, help="directory with adversarial attack analysis logs")
parser.add_argument('-f','--req_files', nargs='+', help='log file names', required=True)
parser.add_argument('-l','--labels', nargs='+', help='labels to identify each run', required=False, default=[
    r"SC",
    r"RC ($\eta=5\mathrm{e}-2$)",
    r"RC ($\eta=8\mathrm{e}-2$)",
    r"DL"
 ])
parser.add_argument('-m','--markers', nargs='+', help='markers to use for plotting', required=False, default=["o", "X", "s", "D"])
parser.add_argument('-t','--thresh', help='threshold to use for distance learner analysis', required=False, default=0.14, type=float)
parser.add_argument('-e','--eps_iter', help='step size of adversarial attacks', required=False, type=float, default=5e-03)
parser.add_argument('-c','--count_offmfld_pred_corr', help='count off-manifold predictions as correct', action="store_true")
parser.add_argument('-d','--discard_offmfld_pred', help='discard off manifold samples in result prediction', action="store_true")
parser.add_argument('-o','--off_mfld_label', help='off-manifold label', type=str, default="2")
parser.add_argument('-r', '--result_file', help='path to plot generated', type=str, default='./adv_robustness.pdf')

args = parser.parse_args()

RES_DIR = args.res_dir

all_run_files = [i for i in os.listdir(RES_DIR) if i.endswith(".json")]




req_files_k50n500 = args.req_files

req_labels_k50n500 = args.labels

markers = args.markers



master_df = None
for f in req_files_k50n500:
    res_file = os.path.join(RES_DIR, f)
    if master_df is None:
        master_df = pd.read_json(res_file)
    else:
        tmp = pd.read_json(res_file)
        master_df = pd.concat([master_df, tmp], ignore_index=True)
columns = master_df.columns.tolist()
drop_dup_by_columns = [i for i in columns if "clf_report" not in i]
master_df.drop_duplicates(drop_dup_by_columns, inplace=True)



#### setting hyperparameters which will be fixed!!!

ths = [args.thresh]
eps_iter = args.eps_iter
count_offmfld_pred_corr = args.count_offmfld_pred_corr # count off manifold predictions as accurate
discard_offmfld_pred = args.discard_offmfld_pred
off_mfld_label = args.off_mfld_label



fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharey=True)

for k in range(len(ths)):
    thresh = ths[k]
    for j in range(len(req_files_k50n500)):
        file = req_files_k50n500[j]
        full_fn = os.path.join(RES_DIR, file)
        df = pd.read_json(full_fn)

        eps_arr = df.eps.unique()[:-1]

        task = df.task.unique()[0]
        run_tag = None
        run_id = None
        try:
            run_tag = df.run_tag.unique()[0]
            run_id = df.run_id.unique()[0]
        except:
            run_tag = "rdm_concspheres_k50n500_noninfdist"
            run_id = 1
        perf = np.zeros(eps_arr.shape)
        for i in range(eps_arr.shape[0]):
            eps = eps_arr[i]
            if task == "dist":
                path_to_load = df[(np.round(df.thresh, 2) == thresh) & (df.eps == eps) & (np.round(df.eps_iter, 3) == eps_iter)].adv_pct_cm.tolist()[0]
                
                adv_pct_cm_df = pd.read_csv(path_to_load, index_col=0)
                adv_pct_cm = adv_pct_cm_df.to_numpy()
                perf[i] = np.trace(adv_pct_cm)
                if count_offmfld_pred_corr:
                    if off_mfld_label in adv_pct_cm_df.columns: 
                        off_mfld_stats = adv_pct_cm_df["2"].to_numpy()
                        if off_mfld_stats.shape[0] == adv_pct_cm.shape[0]:
                            perf[i] += np.sum(off_mfld_stats[:-1])
                        else:
                            perf[i] += np.sum(off_mfld_stats[:])
                elif discard_offmfld_pred:
                    end_row = adv_pct_cm.shape[0]
                    end_col = adv_pct_cm.shape[1]
                    if off_mfld_label in adv_pct_cm_df.columns:
                        end_col = -1
                    if end_row == int(off_mfld_label) + 1:
                        end_row = -1
                    perf[i] = np.trace(adv_pct_cm[:end_row, :end_col]) / np.sum(adv_pct_cm[:end_row, :end_col])

            elif task == "clf":
                path_to_load = df[(df.eps == eps) & (np.round(df.eps_iter, 3) == eps_iter)].adv_pct_cm.tolist()[0]
                
                adv_pct_cm = pd.read_csv(path_to_load, index_col=0).to_numpy()
                perf[i] = np.trace(adv_pct_cm)
    

        label = req_labels_k50n500[j]
        axs.plot(eps_arr, perf, label=label, marker=markers[j], markersize=8, linewidth=2)

axs.legend(fancybox=True, shadow=True)

axs.set_xlabel(r"$\epsilon$", fontsize=18)
axs.set_ylabel("Accuracy", fontsize=18)
plt.savefig(args.result_file, bbox_inches="tight")
plt.show()

