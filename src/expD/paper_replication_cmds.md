# Commands for replicating dataset for "The Relationship Between High-Dimensional Geometry and Adversarial Examples"

First run the Distance Learner for the dataset that is similar to the one given in tha paper:

```bash
python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 lr=5e-5 debug=False loss_func=std_mse\
 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/ \
 data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k500n500_noninfsmoothdist \
 data.data_params.train.N=1000000 \
 data.data_params.train.k=500 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.g=0.3 \
 data.data_params.train.max_norm=0.1 \
 data.data_params.train.bp=0.09 \
 data.data_params.train.M=1.0 \
 data.data_params.train.D=0.07 \
 data.data_params.train.norm_factor=1 \
 data.data_params.val.k=500 \
 data.data_params.val.n=500 \
 data.data_params.val.r=1 \
 data.data_params.val.g=0.3 \
 data.data_params.val.N=200000 \
 data.data_params.val.max_norm=0.1 \
 data.data_params.val.bp=0.09 \
 data.data_params.val.M=1.0 \
 data.data_params.val.D=0.07 \
 data.data_params.val.norm_factor=1 \
 data.data_params.test.k=500 \
 data.data_params.test.n=500 \
 data.data_params.test.r=1 \
 data.data_params.test.g=0.3 \
 data.data_params.test.N=200000 \
 data.data_params.test.max_norm=0.1 \
 data.data_params.test.bp=0.09 \
 data.data_params.test.M=1.0 \
 data.data_params.test.D=0.07 \
 data.data_params.test.norm_factor=1 \
 model.input_size=500 \
 data.generate=False \
 task=regression
```

```bash
python3 learn_cls_from_dist.py with cuda=0 num_epochs=300 cooldown=299 lr=1e-5 model.hidden_sizes=[1024,1024] model.model_type=mlp-norm \
 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/ \
 data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k500n500_noninfdist_moreoffmfld \
 data.data_params.train.N=1000000 \
 data.data_params.train.num_neg=600000 \
 data.data_params.train.k=500 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.g=0.3 \
 data.data_params.train.gamma=0 \
 data.data_params.train.norm_factor=1 \
 data.data_params.train.max_norm=0.1 \
 data.data_params.train.D=0.07 \
 data.data_params.val.k=500 \
 data.data_params.val.n=500 \
 data.data_params.val.r=1 \
 data.data_params.val.g=0.3 \
 data.data_params.val.N=200000 \
 data.data_params.val.gamma=0 \
 data.data_params.val.norm_factor=1 \
 data.data_params.val.max_norm=0.1 \
 data.data_params.val.D=0.07 \
 data.data_params.test.k=500 \
 data.data_params.test.n=500 \
 data.data_params.test.r=1 \
 data.data_params.test.g=0.3 \
 data.data_params.test.N=200000 \
 data.data_params.test.gamma=0 \
 data.data_params.test.norm_factor=1 \
 data.data_params.test.max_norm=0.1 \
 data.data_params.test.D=0.07 \
 model.input_size=500 \
 data.generate=False \
 task=clf \
 train_on_onmfld=True
```






```bash
python3 learn_cls_from_dist.py with cuda=0\
 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/ \
 model.hidden_sizes=[256,256,256,256] \
 data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k500n500 \
 data.data_params.train.N=200000 \
 data.data_params.train.k=500 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.g=0.3 \
 data.data_params.train.gamma=0 \
 data.data_params.train.norm_factor=1 \
 data.data_params.train.max_norm=0.15 \
 data.data_params.train.D=0.075 \
 data.data_params.val.k=500 \
 data.data_params.val.n=500 \
 data.data_params.val.r=1 \
 data.data_params.val.g=0.3 \
 data.data_params.val.N=200000 \
 data.data_params.val.gamma=0 \
 data.data_params.val.norm_factor=1 \
 data.data_params.val.max_norm=0.15 \
 data.data_params.val.D=0.075 \
 data.data_params.test.k=500 \
 data.data_params.test.n=500 \
 data.data_params.test.r=1 \
 data.data_params.test.g=0.3 \
 data.data_params.test.N=200000 \
 data.data_params.test.gamma=0 \
 data.data_params.test.norm_factor=1 \
 data.data_params.test.max_norm=0.15 \
 data.data_params.test.D=0.075 \
 model.input_size=500 \
 data.generate=False \
 task=regression
```

```bash
python3 learn_cls_from_dist.py with cuda=0\
 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/ \
 model.hidden_sizes=[1024,1024] \
 data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k500n500 \
 data.data_params.train.N=200000 \
 data.data_params.train.k=500 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.g=0.3 \
 data.data_params.train.gamma=0 \
 data.data_params.train.norm_factor=1 \
 data.data_params.train.max_norm=0.15 \
 data.data_params.train.D=0.075 \
 data.data_params.val.k=500 \
 data.data_params.val.n=500 \
 data.data_params.val.r=1 \
 data.data_params.val.g=0.3 \
 data.data_params.val.N=200000 \
 data.data_params.val.gamma=0 \
 data.data_params.val.norm_factor=1 \
 data.data_params.val.max_norm=0.15 \
 data.data_params.val.D=0.075 \
 data.data_params.test.k=500 \
 data.data_params.test.n=500 \
 data.data_params.test.r=1 \
 data.data_params.test.g=0.3 \
 data.data_params.test.N=200000 \
 data.data_params.test.gamma=0 \
 data.data_params.test.norm_factor=1 \
 data.data_params.test.max_norm=0.15 \
 data.data_params.test.D=0.075 \
 model.input_size=500 \
 data.generate=False \
 task=clf \
 train_on_onmfld=True
```


```bash
python3 learn_cls_from_dist.py with cuda=0\
 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/ \
 data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k5n500 \
 data.data_params.train.N=200000 \
 data.data_params.train.k=5 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.g=0.3 \
 data.data_params.train.gamma=0 \
 data.data_params.train.norm_factor=1 \
 data.data_params.train.max_norm=0.15 \
 data.data_params.train.D=0.075 \
 data.data_params.val.k=5 \
 data.data_params.val.n=500 \
 data.data_params.val.r=1 \
 data.data_params.val.g=0.3 \
 data.data_params.val.N=200000 \
 data.data_params.val.gamma=0 \
 data.data_params.val.norm_factor=1 \
 data.data_params.val.max_norm=0.15 \
 data.data_params.val.D=0.075 \
 data.data_params.test.k=5 \
 data.data_params.test.n=500 \
 data.data_params.test.r=1 \
 data.data_params.test.g=0.3 \
 data.data_params.test.N=200000 \
 data.data_params.test.gamma=0 \
 data.data_params.test.norm_factor=1 \
 data.data_params.test.max_norm=0.15 \
 data.data_params.test.D=0.075 \
 model.input_size=500 \
 data.generate=True \
 task=regression
```

```bash
python3 learn_cls_from_dist.py with cuda=0\
 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/ \
 data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k5n500 \
 data.data_params.train.N=200000 \
 data.data_params.train.k=5 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.g=0.3 \
 data.data_params.train.gamma=0 \
 data.data_params.train.norm_factor=1 \
 data.data_params.train.max_norm=0.15 \
 data.data_params.train.D=0.075 \
 data.data_params.val.k=5 \
 data.data_params.val.n=500 \
 data.data_params.val.r=1 \
 data.data_params.val.g=0.3 \
 data.data_params.val.N=200000 \
 data.data_params.val.gamma=0 \
 data.data_params.val.norm_factor=1 \
 data.data_params.val.max_norm=0.15 \
 data.data_params.val.D=0.075 \
 data.data_params.test.k=5 \
 data.data_params.test.n=500 \
 data.data_params.test.r=1 \
 data.data_params.test.g=0.3 \
 data.data_params.test.N=200000 \
 data.data_params.test.gamma=0 \
 data.data_params.test.norm_factor=1 \
 data.data_params.test.max_norm=0.15 \
 data.data_params.test.D=0.075 \
 model.input_size=500 \
 data.generate=False \
 task=clf \
 train_on_onmfld=True
```

```bash
python3 learn_cls_from_dist.py with cuda=0\
 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/ \
 data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k10n500 \
 data.data_params.train.N=200000 \
 data.data_params.train.k=10 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.g=0.3 \
 data.data_params.train.gamma=0 \
 data.data_params.train.norm_factor=1 \
 data.data_params.train.max_norm=0.15 \
 data.data_params.train.D=0.075 \
 data.data_params.val.k=10 \
 data.data_params.val.n=500 \
 data.data_params.val.r=1 \
 data.data_params.val.g=0.3 \
 data.data_params.val.N=200000 \
 data.data_params.val.gamma=0 \
 data.data_params.val.norm_factor=1 \
 data.data_params.val.max_norm=0.15 \
 data.data_params.val.D=0.075 \
 data.data_params.test.k=10 \
 data.data_params.test.n=500 \
 data.data_params.test.r=1 \
 data.data_params.test.g=0.3 \
 data.data_params.test.N=200000 \
 data.data_params.test.gamma=0 \
 data.data_params.test.norm_factor=1 \
 data.data_params.test.max_norm=0.15 \
 data.data_params.test.D=0.075 \
 model.input_size=500 \
 data.generate=True \
 task=regression
```

```bash
python3 learn_cls_from_dist.py with cuda=0\
 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/ \
 data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k10n500 \
 data.data_params.train.N=200000 \
 data.data_params.train.k=10 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.g=0.3 \
 data.data_params.train.gamma=0 \
 data.data_params.train.norm_factor=1 \
 data.data_params.train.max_norm=0.15 \
 data.data_params.train.D=0.075 \
 data.data_params.val.k=10 \
 data.data_params.val.n=500 \
 data.data_params.val.r=1 \
 data.data_params.val.g=0.3 \
 data.data_params.val.N=200000 \
 data.data_params.val.gamma=0 \
 data.data_params.val.norm_factor=1 \
 data.data_params.val.max_norm=0.15 \
 data.data_params.val.D=0.075 \
 data.data_params.test.k=10 \
 data.data_params.test.n=500 \
 data.data_params.test.r=1 \
 data.data_params.test.g=0.3 \
 data.data_params.test.N=200000 \
 data.data_params.test.gamma=0 \
 data.data_params.test.norm_factor=1 \
 data.data_params.test.max_norm=0.15 \
 data.data_params.test.D=0.075 \
 model.input_size=500 \
 data.generate=False \
 task=clf \
 train_on_onmfld=True
```



=====

```bash
python3 learn_cls_from_dist.py with cuda=0 num_epochs=300 cooldown=299 lr=1e-5 debug=False loss_func=std_mse\
 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expD_distlearner_against_adv_eg/rdm_concspheres/ \
 data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k100n500_noninfdist_moreoffmfld \
 data.data_params.train.N=1000000 \
 data.data_params.train.num_neg=600000 \
 data.data_params.train.k=100 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.g=0.3 \
 data.data_params.train.gamma=0 \
 data.data_params.train.norm_factor=1 \
 data.data_params.train.max_norm=0.1 \
 data.data_params.train.D=0.07 \
 data.data_params.val.k=100 \
 data.data_params.val.n=500 \
 data.data_params.val.r=1 \
 data.data_params.val.g=0.3 \
 data.data_params.val.N=200000 \
 data.data_params.val.gamma=0 \
 data.data_params.val.norm_factor=1 \
 data.data_params.val.max_norm=0.1 \
 data.data_params.val.D=0.07 \
 data.data_params.test.k=100 \
 data.data_params.test.n=500 \
 data.data_params.test.r=1 \
 data.data_params.test.g=0.3 \
 data.data_params.test.N=200000 \
 data.data_params.test.gamma=0 \
 data.data_params.test.norm_factor=1 \
 data.data_params.test.max_norm=0.1 \
 data.data_params.test.D=0.07 \
 model.input_size=500 \
 data.generate=True \
 task=regression
 ```