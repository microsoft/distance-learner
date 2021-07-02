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


**Note:**: `../notebooks/Experiment C - Distance Learner for Adversarial Examples.ipynb` has a decent amount of code that shows a lot of these script in action. Refer to it when in need.