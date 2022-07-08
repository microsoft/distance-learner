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


## Running `pipeline` with custom settings

1. First go into `pipeline/data_ingredients.py` and change the `logdir` and `data_tag` appropriately in the `data_cfg` function.
    - You might also need to change `mtype`. Refer to `datagen/datagen.py` for which `mtype` or manifold type maps to which manifold class.
    - From the command line (using `with` from `sacred`), these can be changed by `with data.logdir=<BLAH> data.data_tag=<BLEH> data.mtype=<BLAH>`.
    - Possible values of `mtype` are keys in the `DATA_TYPE` variable in `pipeline/data_ingredients.py`.

2. Next go to `pipeline/data_configs.py` and if you need to, then change the parameters of the appropriate dictionary in the appropriate function depending on `mtype` from the previous point.
    - Again this can be done from the command line also.
    - Say you want to change the number of samples in the dataset to 10000 (usually key value `N` in the dictionary). You do: `with data.data_params.N=10000`.

3. Next go to `pipeline/model_ingredients.py`. Here the value of input_size must be same as the value of `n` in the dictionary in `pipeline/data_configs.py`. Rest of parameters can also be changed as needed. 
    - For changing things from command line you can prepend the config param with `model`, eg. `with model.model_type=vanilla-mlp`


4. We have typically done 3 types of experiments together with same data. Here are the commands to run them after settings have been changed as described above:
    - Training Distance Learner: ```python3 learn_cls_from_dist.py with task=regression data.generate=True```
    - Training Standard Classifier: ```python3 learn_cls_from_dist.py with task=clf train_on_onmfld=True on_mfld_noise=0 adv_train=False test_off_mfld=False```
    - Training Robust Classifier: `python3 learn_cls_from_dist.py with task=clf train_on_onmfld=False train_on_onmfld=True on_mfld_noise=0 adv_train=True adv_train_params.atk_eps=<INSERT_ATTACK_EPSILON> test_off_mfld=False`

5. For a fixed manifold type, we try the above steps in order, in the same `logdir` and `data_tag`.

6. We run them on variations of `n` and `k`. The changes have been described in points 1-3 and summarised below:
    - Edit `data_tag` and `logdir` in data_ingredients.py (or from command line using `with data.data_tag, data.log_dir`)
    - Edit `n`, `k` in `data_config.py` (or from command line using `with data.data_params.{n,k}`)
    - Edit `input_size` in `model_config.py` (or from command line using `with model.input_size`)

7. If you want to edit parameters used for robust training, refer to the `adv_train_params` dictionary in the `config` function in `learn_cls_from_dist.py`.
    - To edit them from command line, use `with adv_train_params.<PARAM_NAME>=<PARAM_VALUE>`
    - For details on what the parameters mean, see [this document](../adversarial_attack/README.md).

8. After a run, we generate a few plots using `pipeline/analysis.py`. Command is: `analysis.py --dump_dir <PATH_TO_SACRED_RUN_DUMP> --on <SPLIT_TO_ANALYSE> --num_points <NUM_POINTS_TO_RENDER>`