# `adversarial_attack`

This directory contains scripts to generated adversarial examples for models and evaluate them against generated examples.

- `attack_ingredients.py`: used for providing settings for attacks against which we want to evaluate the models
- `inp_fn_ingredients.py`: used for providing locations of `pipeline` runs that we want to evaluate against adversarial attacks
- `attacks.py`: code for various attacks that models can be evaluated on
- `get_attack_perf.py`: end-to-end script that accepts directories containing `pipeline` runs, attacks them using adversarial attacks, and evaluates the performance of the models on these attacks

