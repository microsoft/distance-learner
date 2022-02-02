import os
import json

from sacred import Ingredient

inpfn_ingredient = Ingredient('input_files')

@inpfn_ingredient.config
def inpfn_config():


    # proj_dir = "/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/"
    # settings_to_analyze = {
    #     "rdm_concspheres_k100n100_noninfdist": ["1", "2"],
    #     "rdm_concspheres_k500n500_noninfdist": ["1", "2"]
    # }
    proj_dir = "/data/dumps/expC_dist_learner_for_adv_ex/rdm_concspheres_test/"
    settings_to_analyze = {
        "rdm_concspheres_k2n500_noninfdist": ["1", "2"],
        "rdm_concspheres_k500n500_noninfdist": ["1", "2"],
        "rdm_concspheres_k50n500_noninfdist": ["1", "2"]
    }
    settings_fn = None
    if settings_fn is not None:
        with open(settings_fn) as f:
            settings = json.load(settings_fn)
            settings_to_analyze = settings["analyze"]
            proj_dir = settings["proj_dir"]

@inpfn_ingredient.capture
def get_inp_fn(proj_dir, settings_to_analyze):
    inp_files = []
    for data_tag in settings_to_analyze:
        for run in settings_to_analyze[data_tag]:
            inp_files.append(os.path.join(proj_dir, data_tag, run))
    return sorted(inp_files)




