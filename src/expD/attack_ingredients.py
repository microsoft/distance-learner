import torch
import numpy as np
from sacred import Ingredient

attack_ingredient = Ingredient('attack')

@attack_ingredient.config
def attack_cfg():

    atk_name = "pgd_l2_mfld_clf"

    