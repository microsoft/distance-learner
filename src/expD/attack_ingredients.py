import torch
import numpy as np
from sacred import Ingredient

from attacks import attacks

attack_ingredient = Ingredient('attack')

@attack_ingredient.config
def attack_cfg():

    atk_flavor = ["std_pgd", "onmfld_pgd"] # ["std_pgd", "onmfld_pgd"]
    atk_routine = ["chans"] # ["chans", "my"] if given option is not available, it defaults to "my"

    eps = np.arange(1e-2, 1.1e-1, 1e-2)
    eps_iter = np.arange(5e-3, 1e-2, 1e-3)
    nb_iter = [40, 100]
    norm = [2, np.inf] # [2, np.inf]
    restarts = [1]

    thresh = np.arange(1e-2, 1.6e-1, 1e-2) # if model is distance learner, then thresholds to analyse performance
    thresh = np.append(thresh, np.inf)

@attack_ingredient.capture
def get_atk(atk_flavor, task, atk_routine):
    if atk_routine not in attacks[atk_flavor][task]:
        return attacks[atk_flavor][task]["my"], "my"
    return attacks[atk_flavor][task][atk_routine], atk_routine





