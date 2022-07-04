# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from sacred import Ingredient

from expB import learn_mfld_distance as lmd

model_ingredient = Ingredient('model')

@model_ingredient.config
def model_cfg():
    
    model_type = "mt-mlp-norm" # see the model_type dictionary in expB/learn_manifold_distance.py for possible values.
    input_size = 2
    output_size = 2
    hidden_sizes = [512] * 4
    weight_norm = False
    use_tanh = False
    use_relu = False

    init_wts = None

@model_ingredient.capture
def initialise_model(model_type="mt-mlp-norm", input_size=3,\
     output_size=2, hidden_sizes=[512]*5, weight_norm=False,\
     use_tanh=False, use_relu=False, init_wts=None, **kwargs):
    
    model = lmd.model_type[model_type](input_size=input_size,\
         output_size=output_size, hidden_sizes=hidden_sizes,\
         use_tanh=use_tanh, use_relu=use_relu, weight_norm=weight_norm)

    scheduler_state_dict = None
    optimizer_state_dict = None

    if init_wts is not None:
        ckpt = torch.load(init_wts)
        model.load_state_dict(ckpt["model_state_dict"])
        scheduler_state_dict = ckpt["scheduler_state_dict"]
        optimizer_state_dict = ckpt["optimizer_state_dict"]

    return model, scheduler_state_dict, optimizer_state_dict

