"""
Script written to help with forming commands to call for n Vs. k analysis
of learn_cls_from_dist.py (Expt. 3a). 

This script only requests usual changes in the experiment run parameters from
the user and forms a command based on the provided arguments

Why is this needed?

Changing arguments within files is unsavory. Changing arguments of the main script
in the command line is tiresome and prone to catastrophic errors which then require
removals. Having this in place gives a non-explosive way to generate commands and 
then copy-paste and run them.

Initial values are set to default. Should be fine as such.
"""
import os
import sys

mkdown_text = """
# Commands for Experiment 3a
"""

# mtype = "ittw-swissrolls"
mtype = "conc-spheres"
# logdir = "/azuredrive/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/"
logdir = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_concspheres/"
# data_tag_prefix = "rdm_swrolls_"
data_tag_prefix = "rdm_concspheres"
cuda = 0
dims = [2, 3, 5, 10, 50, 100, 500, 1000]

expt_cmd_prefix = "python3 learn_cls_from_dist.py with cuda={} data.logdir={} data.mtype={} ".format(cuda, logdir, mtype)
analysis_cmd_prefix = "python3 analysis.py --dump_dir={} --on=test --num_points=50000"
expt_scenarios = ["task=regression", "task=clf train_on_onmfld=True", "model.output_size=3 task=clf train_on_onmfld=False"]
scenario_texts = ["### Command to train Distance-learner", "### Command to train Standard Classifier", "### Command to train Standard Classifier with Off-manifold label"]

out_file = "./commands_nVSk_concspheres.md"

for kidx in range(len(dims)):
    for nidx in range(kidx, len(dims)):

        k = dims[kidx]
        n = dims[nidx]
        mkdown_text += """
## k = {}, n = {}

""".format(k, n)

        for i in range(len(expt_scenarios)):

            expt_cmd = expt_cmd_prefix 

            

            data_tag = data_tag_prefix + "k" + str(k) + "n" + str(n)
            expt_cmd += "data.data_tag=" + data_tag + " "

            for split in ["train", "val", "test"]:
                expt_cmd += "data.data_params." + split + ".k=" + str(k) + " "
                expt_cmd += "data.data_params." + split + ".n=" + str(n) + " "

            expt_cmd += "model.input_size=" + str(n) + " "

            if i == 0:
                expt_cmd += "data.generate=True "
            
            expt_cmd += expt_scenarios[i]

            mkdown_text += """
{}
""".format(scenario_texts[i])

            mkdown_text += """
```bash
{}
""".format(expt_cmd)

            analysis_input_dir = os.path.join(logdir, data_tag, str(i+1))
            analysis_cmd = analysis_cmd_prefix.format(analysis_input_dir)
            mkdown_text += """
{}
```
""".format(analysis_cmd)

with open(out_file, "w") as f:
    f.write(mkdown_text)







