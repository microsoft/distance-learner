# `pipeline`

This directory contains the pipeline code for synthesizing data and training our models (distance learner, standard classifier and robust classifier)

- `data_ingredients.py`: contains settings for synthesizing dataset, like specifying where it will be dumped, what manifold type to use, etc.
- `data_configs.py`: contains dictionaries containing specific parameters for synthesizing each manifold type; used by `data_ingredients.py`
- `model_ingredients.py`: used to specify settings for the model
- `learn_cls_from_dist.py`: main pipeline script that synthesizes the data and trains the model
- `analysis.py`: analysis script that can plot decision regions for lower dimensional data
- `pipeline_utils/`: contains common utility functions useful for all experimental settings
  - `common.py`: contains functions for loading models, experimental configs and generating synthetic data for decision region plotting
  - `plotter.py`: contains plotting scripts for decision regions for low-dimensional manifolds
