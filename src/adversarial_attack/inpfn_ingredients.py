# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
from sacred import Ingredient


inpfn_ingredient = Ingredient('input_files')

@inpfn_ingredient.config
def inpfn_config():

    proj_dir = "../../data/rdm_concspheres_test/"

    # providing input files as a dictionary
    # with keys as data_tag and value as a list of
    # runs to process
    settings_type = "dict"
    settings_to_analyze = {
        "rdm_concspheres_m100n100": ["1", "2"],
        "rdm_concspheres_m50n500": ["1", "2"]
    }

    # providing input files as a list of 
    # directories containing runs to process
    #
    # settings_type = "list"
    # settings_to_analyze = [
    #     "rdm_concspheres_m100n100/1",
    #     "rdm_concspheres_m100n100/2",
    #     "rdm_concspheres_m50n500/1",
    #     "rdm_concspheres_m50n500/2",
    # ]
    
    settings_fn = None
    if settings_fn is not None:
        with open(settings_fn) as f:
            settings = json.load(settings_fn)
            settings_to_analyze = settings["analyze"]
            proj_dir = settings["proj_dir"]

@inpfn_ingredient.capture
def get_inp_fn(proj_dir, settings_type, settings_to_analyze):
    inp_files = []
    if settings_type == "dict":
        for data_tag in settings_to_analyze:
            for run in settings_to_analyze[data_tag]:
                inp_files.append(os.path.join(proj_dir, data_tag, run))
    else:
        for run in settings_to_analyze:
            inp_files.append(os.path.join(proj_dir, run))
    return sorted(inp_files)




