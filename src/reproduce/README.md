# Reproducing Results

To reproduce results given in the paper, follow the below steps:

1. Generate Models: Run any of the `.sh` files in this directory to create model and data dumps for a data setting. The files have been named `run_D_mMnN.sh` where `M` is the manifold dimension and `N` is the embedding dimension, and `D` is an identifier for the dataset type (`cs`: Concentric Spheres, `sw`: Intertwined Swiss Rolls, `ws`: Separated Spheres).

2. Decision Region & Heatmap Plots: Navigate to  `ppr_decreg_and_heatmaps.py`. Provide the path to the distance learner and standard classifier dumps, and an identifier for the plot file names and run the script.

3. Out-of-domain Confidence Plot: Navigate to  `ppr_confidence.py`. Provide the path to the distance learner and standard classifier dumps, and an identifier for the plot file names and run the script.

4. Adversarial Robustness Plot: Make sure you have run `run_cs_m50n500.sh` and `run_cs_m25n500.sh`. Navigate to `adv_robustness.py`. Provide a path to the locations of adversarial performance dumps (`.json` files created when you run the bash scripts), and run the script.
