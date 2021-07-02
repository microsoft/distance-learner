# Adversarial Geometry

Below I have summarised briefly what is happening in each of the important files/folders and some notes on how to run them.

1. `datagen` - Data Generation

This package holds the classes that are used for creating our synthetic manifold datasets. Later I also intend to use it to house the code that we use for generating our versions of real-world datasets.

- `datagen/synthetic`: For generating synthetic datasets.
    - `datagen/synthetic/single`: For single manifold datasets, primarily used by us for veifying that we can learn distances.
    - `datagen/synthetic/multiple`: For multiple manifold datasets.
    - I have tried to give each file in there its own self-contained `main` method to test things out. 
    - Copy imports in new files from the existing files.

2. `expB`: This folder was used to house the code for learning distance from manifold. Basically, we were trying to see whether a network can actually learn distance from single or multiple manifolds.
    - `expB/ptcifar.py`: Clone of [this repo](https://github.com/kuangliu/pytorch-cifar). Used for its ResNet18 code mostly.
    - `expB/myNNs.py`: contains code for the models that we have been using in our experiments, that were developed by us.
    - `expB/spheres_v{1,2}.py`: Initial code for random sphere and two random spheres dataset. How to run it using given options is self-explanatory. Usually prefer the configuration JSON for generation. Examples can be found in `expB/data_configs`. Command for running is: 
    ```python3 spheres_v2.py --twospheres --config <CONFIG_FILE> --save_dir <SAVE_DIR>```
        - `v1`: pre-images not generated separately
        - `v2`: pre-images generated separately (latest and mostly use this)
    - `expB/learn_mfld_distance.py`: Script for learning distance from a manifold. Contains training and testing code. 
    For training, while it has options that are self-explanatory, I have found using a specifications file much easier. Examples of specification files for train and test are in `expB/expt_specs`. Command used is:
        - For training: `python3 learn_mfld_distance.py --train --specs <PATH_TO_SPECS_FILE>`
        - For testing: `python3 learn_mfld_distance.py --test --specs <PATH_TO_SPECS_FILE>`
    - `expB/workspace.py`: This is used by `learn_mfld_distance.py`. Any new parameter added to this file needs to be updated here.

3. `expC`: This folder contains the code for Experiment C, that is, learning classes from distance learner, and comparing it with Standard Classifier on normal (done) and adversarial (to be done) examples.
    - `expC/learn_cls_from_dist.py`: Runner code for learning class from distance learner. Built using [`sacred`](https://sacred.readthedocs.io/en/stable/index.html). It can train both distance learner and standard classifier with/without off manifold points.
    - `expC/data_configs.py`: Basically a container script for storing functions that return dictionaries with parameters for data generation.
    - `expC/data_ingredients.py`: [`sacred` ingredient](https://sacred.readthedocs.io/en/stable/ingredients.html) script, used by `expC/learn_cls_from_dist.py`. It contains the code for generating the data (if needed), dumping it or loading it.
    - `expC/model_ingredients.py`: `sacred` ingredient script used by `expC/learn_cls_from_dist.py`. It contains code for initialising the model, dumping it and loading it if needed.
    - `expC/analysis.py`: Functions to generate some analysis plots on specified data split and with given number of samples.
    - `expC/utils`: `common.py` has common utility functions for different experimental settings, and each manifold type and setting gets a separate file for making its figures, taking help from `common.py`

# Running `expC` with tweaks

Typically, you would want to run `expC` for different datasets on different parameters. Right now it is written keeping only synthetic multiple manifold datasets in mind and not tested for other kinds of datasets. 

## How to proceed?

1. First go into `expC/data_ingredients.py` and change the `logdir` and `data_tag` appropriately in the `data_cfg` function.
    - You might also need to change `mtype`
    - From the command line (using `with` from `sacred`), these can be changed by `with data.logdir=<BLAH> data.data_tag=<BLEH> data.mtype=<BLAH>`.
    - Possible values of `mtype` are keys in the `DATA_TYPE` variable in `expC/data_ingredients.py`.

2. Next go to `expC/data_configs.py` and if you need to, then change the parameters of the appropriate dictionary in the appropriate function depending on `mtype` from the previous point.
    - Again from the command line also this can be done.
    - Say you want to change the number of samples in the dataset to 10000 (usually key value `N` in the dictionary). You do: `with data.data_params.N=10000`.

3. Next go to `expC/model_ingredients.py`. Here the value of input_size must be same as the value of `n` in the dictionary in `expC/data_configs.py`. Rest of parameters can also be changed as needed. 
    - For changing things from command line you can prepend the config param with `model`, eg. `with model.model_type=vanilla-mlp`

4. As of 2nd July 2021, we have typically done 3 types of experiments together with same settings (except for the necessary one's which decide the experiment type). Here they are with commands to run them after settings have been changed as described above:
    - Training Distance Learner: ```python3 learn_cls_from_dist.py with task=regression data.generate=True```
    - Training Standard Classifier: ```python3 learn_cls_from_dist.py with task=clf train_on_onmfld=True```
    - Training Standard Classifier with Off-manifold labels: ```python3 learn_cls_from_dist.py with task=clf train_on_onmfld=False model.output_size=3```

5. For a fixed manifold type, we try the above steps in order, in the same `logdir` and `data_tag`.
6. We run them on variations of `n` and `k`. The changes have been described in points 1-3 and summarised below:
    - Edit `data_tag` and `logdir` in data_ingredients.py (`with data.data_tag, data.log_dir`)
    - Edit `n`, `k` in `data_config.py` (`with data.data_params.{n,k}`)
    - Edit `input_size` in `model_config.py` (`with model.input_size`)


7. After a run, we generate a few plots (recall `fig. 1`, `fig. 2-{1, 2}`) using `expC/analysis.py`. Command is: `analysis.py --dump_dir <PATH_TO_SACRED_RUN_DUMP> --on <SPLIT_TO_ANALYSE> --num_points <NUM_POINTS_TO_RENDER>`


**Note:**: `../notebooks/Experiment C - Distance Learner for Adversarial Examples.ipynb` has a decent amount of code that shows a lot of these script in action. Refer to it when in need.