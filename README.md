# Distance Learner: Incorporating Manifold Prior to Model Training

This repository contains the official implementation for:

>[Distance Learner: Incorporating Manifold Prior to Model Training](https://arxiv.org/abs/2207.06888). *[Aditya Chetan](http://justachetan.github.io/)*, *[Nipun Kwatra](https://www.microsoft.com/en-us/research/people/nkwatra/)*

## About

The manifold hypothesis (real world data concentrates near low-dimensional manifolds) is suggested as the principle behind the effectiveness of machine learning algorithms in very high dimensional problems that are common in domains such as vision and speech. Multiple methods have been proposed to explicitly incorporate the manifold hypothesis as a prior in modern Deep Neural Networks (DNNs), with varying success. In this paper, we propose a new method, Distance Learner, to incorporate this prior for DNN-based classifiers. Distance Learner is trained to predict the distance of a point from the underlying manifold of each class, rather than the class label. For classification, Distance Learner then chooses the class corresponding to the closest predicted class manifold. Distance Learner can also identify points as being out of distribution (belonging to neither class), if the distance to the closest manifold is higher than a threshold. We evaluate our method on multiple synthetic datasets and show that Distance Learner learns much more meaningful classification boundaries compared to a standard classifier. We also evaluate our method on the task of adversarial robustness, and find that it not only outperforms standard classifier by a large margin, but also performs at par with classifiers trained via state-of-the-art adversarial training.

## Setup and Usage

### Dependencies

```
* Python 3.6+
* scikit-learn==0.22.1
* scipy==1.4.1
* numpy==1.18.1
* torch==1.11.0
* torchvision==0.12.0
* faiss==1.7.2
* cleverhans==4.0.0
* livelossplot==0.5.5
* matplotlib==3.1.3
* plotly==5.8.0
* seaborn==0.10.0
* tensorboard==2.9.0
* tqdm==4.42.1
* sacred==0.8.2
```

All of these can be installed using any package manager such as `pip` or Conda. We recommend using a virtual environment before installing these packages. For installing Faiss, please refer to [instructions](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) on their official repository.

### Sample Code

The code in this project is used to run a pipeline as follows:

1. Data Synthesis: Synthesizes the data used for training the models. This includes on-manifold point generation for synthetic datasets, as well as off-manifold augmentations.
2. Distance Learner Training: Involves training the distance learner on the generated dataset.
3. Standard Classifier Training: Involves training the standard classifier on the generated dataset.
4. Robust Classifier Training: To create a classifier generated under adversarial training [[Madry et. al]](https://arxiv.org/abs/1706.06083)
5. Testing on Adversarial Attacks: We attack the models trained in 2-4 with PGD-based $l_2$-norm attacks.

The complete pipeline trains and compares Distance Learner, Standard Classifier and Robust Classifier on adversarial attacks. As an example, let us say that we want to run this pipeline on a dataset consisting of two concentric 50-spheres (50 dimensional spheres) embedded in 500-dimensional space. The following commands would be required to run this pipeline:

```bash
# Switch to directory with training code
cd ./src/pipeline

# Steps 1 & 2: Data Synthesis and Distance Learner training

python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1.5e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-conc-spheres data.logdir="../../dumps/rdm_concspheres_test/" \
 data.data_tag=rdm_concspheres_m50n500 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.k=51 \
 data.data_params.train.n=500 \
 data.data_params.train.max_t_delta=1e-3 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.val.N=200000 \
 data.data_params.val.num_neg=100000 \
 data.data_params.val.k=51 \
 data.data_params.val.n=500 \
 data.data_params.val.max_norm=0.14 \
 data.data_params.test.N=200000 \
 data.data_params.test.num_neg=100000 \
 data.data_params.test.k=51 \
 data.data_params.test.n=500 \
 data.data_params.test.max_norm=0.14 \
 model.input_size=500 \
 data.generate=True \
 task=regression
 
# Step 3: Standard Classifier training

python3 learn_cls_from_dist.py with cuda=3 num_epochs=1000 cooldown=700 warmup=10 lr=8e-5 batch_size=4096 debug=False data.mtype=inf-conc-spheres \
 data.logdir="../../dumps/rdm_concspheres_test/" \
 data.data_tag=rdm_concspheres_m50n500 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.k=51 \
 data.data_params.train.n=500 \
 data.data_params.train.max_t_delta=1e-3 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.val.N=200000 \
 data.data_params.val.num_neg=100000 \
 data.data_params.val.k=51 \
 data.data_params.val.n=500 \
 data.data_params.val.max_norm=0.14 \
 data.data_params.test.N=200000 \
 data.data_params.test.num_neg=100000 \
 data.data_params.test.k=51 \
 data.data_params.test.n=500 \
 data.data_params.test.max_norm=0.14 \
 model.input_size=500 \
 on_mfld_noise=0 \
 adv_train=False \
 test_off_mfld=False \
 data.generate=False \
 task=clf
 
 # Step 4: Robust Classifier training

python3 learn_cls_from_dist.py with cuda=3 num_epochs=1000 cooldown=700 warmup=10 lr=8e-5 batch_size=4096 debug=False data.mtype=inf-conc-spheres \
 data.logdir="../../dumps/rdm_concspheres_test/" \
 data.data_tag=rdm_concspheres_m50n500 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.k=51 \
 data.data_params.train.n=500 \
 data.data_params.train.max_t_delta=1e-3 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.val.N=200000 \
 data.data_params.val.num_neg=100000 \
 data.data_params.val.k=51 \
 data.data_params.val.n=500 \
 data.data_params.val.max_norm=0.14 \
 data.data_params.test.N=200000 \
 data.data_params.test.num_neg=100000 \
 data.data_params.test.k=51 \
 data.data_params.test.n=500 \
 data.data_params.test.max_norm=0.14 \
 model.input_size=500 \
 on_mfld_noise=0 \
 adv_train=True \
 adv_train_params.atk_eps=8e-2 \
 test_off_mfld=False \
 data.generate=False \
 task=clf
 
# Switch to directory with adversarial attack analysis
cd ../adversarial_attack
 
# Step 5: Adversarial Attack analysis

python3 get_attack_perf.py with debug=False "attack.atk_routine=['my']" \
 input_files.settings_type=list \
 input_files.proj_dir="../dumps/rdm_concspheres_test/" \
 input_files.settings_to_analyze="['rdm_concspheres_m50n500/1','rdm_concspheres_m50n500/2', 'rdm_concspheres_m50n500/3']" \
 dump_dir="../../dumps/rdm_concspheres_test/attack_perfs_on_runs" \
```

### About the code

This section describes the purpose of relevant files in the project.

- `./src/datagen/`: Module for generating data for experiments

- `./src/pipeline/`: Contains the code for our models and traning loop
  - `models.py`: Contains the code for all our models
  - `learn_mfld_distance.py`: Training and test loop for our models

- `./src/pipeline/`: Contains pipeline code for data synthesis and model training
  - `pipeline_utils/`: Contains some utility functions for the pipeline
    - `common.py`: Some common utility functions for all kinds of synthetic manifold datasets
    - `plot_ittwswrolls.py`: Plotting functions for synthetic manifold datasets
  - `data_configs.py`: Configuration values for synthetic datasets
  - `data_ingredients.py`: Data ingredient for the pipeline; Collates data parameters and synthesizes the dataset
  - `model_ingredients.py`: Model ingredient for the pipeline; Loads/Initialized the model to be used for training
  - `learn_cls_from_dist.py`: Runs the data synthesis and model training pipeline end-to-end

- `./src/adversarial_attack/`: Contains pipeline code for adversarial attacks on trained models
  - `attack_ingredients.py`: Settings for attacks that models have to be evaluated on
  - `inpfn_ingredients.py`: Settings for models and datasets that have to be evaluated with adversarial attacks
  - `attacks.py`: Code for adversarial attacks to the models can be evaluated on
  - `get_attack_perf.py`: Runner script that loads the models and data, generates adversarial examples and evaluated the models

- `./src/reproduce/`: Contains steps to reproduce the results in our work 

### Results

In order to generate results given in the paper, follow the instructions given [here](./src/reproduce/README.md).

## Cite

If you find this code useful in your projects, please consider citing our work:

```
@misc{chetan2022distance,
    title={Distance Learner: Incorporating Manifold Prior to Model Training},
    author={Aditya Chetan and Nipun Kwatra},
    year={2022},
    eprint={2207.06888},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
