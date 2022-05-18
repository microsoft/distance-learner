# Commands for replicating dataset for "The Relationship Between High-Dimensional Geometry and Adversarial Examples"

First run the Distance Learner for the dataset that is similar to the one given in the paper:


```bash
python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1.5e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k50n500_noninfdist_moreoffmfldv4_bs4096_highmn40 \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.N=12500000 \
 data.data_params.train.num_neg=12000000 \
 data.data_params.train.g=0.3 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.train.bp=0.09 \
 data.data_params.train.M=1.0 \
 data.data_params.train.D=0.07 \
 data.data_params.train.norm_factor=1 \
 data.data_params.train.gamma=0 \
 data.data_params.train.online=False \
 data.data_params.train.off_online=False \
 data.data_params.train.augment=False \
 data.data_params.val.k=50 \
 data.data_params.val.n=500 \
 data.data_params.val.r=1 \
 data.data_params.val.g=0.3 \
 data.data_params.val.N=200000 \
 data.data_params.val.max_norm=0.14 \
 data.data_params.val.bp=0.09 \
 data.data_params.val.M=1.0 \
 data.data_params.val.D=0.07 \
 data.data_params.val.norm_factor=1 \
 data.data_params.val.gamma=0 \
 data.data_params.val.online=False \
 data.data_params.val.off_online=False \
 data.data_params.val.augment=False \
 data.data_params.test.k=50 \
 data.data_params.test.n=500 \
 data.data_params.test.r=1 \
 data.data_params.test.g=0.3 \
 data.data_params.test.N=200000 \
 data.data_params.test.max_norm=0.14 \
 data.data_params.test.bp=0.09 \
 data.data_params.test.M=1.0 \
 data.data_params.test.D=0.07 \
 data.data_params.test.norm_factor=1 \
 data.data_params.test.gamma=0 \
 data.data_params.test.online=False \
 data.data_params.test.off_online=False \
 data.data_params.test.augment=False \
 model.input_size=500 \
 data.generate=True \
 task=regression
```

```bash
python3 learn_cls_from_dist.py with num_epochs=1000 cooldown=700 lr=8e-5 debug=False cuda=1 \
 data.mtype=conc-spheres batch_size=4096 \
 data.data_tag=rdm_concspheres_k50n500_noninfdist_moreoffmfld_advtraindebug_bs4096_eps=1e-1 \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.N=2500000 \
 data.data_params.train.num_neg=2000000 \
 data.data_params.train.g=0.3 \
 data.data_params.train.max_norm=0.1 \
 data.data_params.train.bp=0.09 \
 data.data_params.train.M=1.0 \
 data.data_params.train.D=0.07 \
 data.data_params.train.norm_factor=1 \
 data.data_params.train.gamma=0 \
 data.data_params.train.online=False \
 data.data_params.train.off_online=False \
 data.data_params.train.augment=False \
 data.data_params.val.k=50 \
 data.data_params.val.n=500 \
 data.data_params.val.r=1 \
 data.data_params.val.g=0.3 \
 data.data_params.val.N=200000 \
 data.data_params.val.max_norm=0.1 \
 data.data_params.val.bp=0.09 \
 data.data_params.val.M=1.0 \
 data.data_params.val.D=0.07 \
 data.data_params.val.norm_factor=1 \
 data.data_params.val.gamma=0 \
 data.data_params.val.online=False \
 data.data_params.val.off_online=False \
 data.data_params.val.augment=False \
 data.data_params.test.k=50 \
 data.data_params.test.n=500 \
 data.data_params.test.r=1 \
 data.data_params.test.g=0.3 \
 data.data_params.test.N=200000 \
 data.data_params.test.max_norm=0.1 \
 data.data_params.test.bp=0.09 \
 data.data_params.test.M=1.0 \
 data.data_params.test.D=0.07 \
 data.data_params.test.norm_factor=1 \
 data.data_params.test.gamma=0 \
 data.data_params.test.online=False \
 data.data_params.test.off_online=False \
 data.data_params.test.augment=False \
 model.input_size=500 \
 data.generate=True \
 adv_train=True \
 adv_train_params.atk_eps=1e-1 \
 on_mfld_noise=0 \
 test_off_mfld=False \
 task=clf
```

## Inferred Manifold commands

### Concentric Spheres Dataset

```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=1 num_epochs=1000 cooldown=700 warmup=10 lr=1e-5 batch_size=512 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-conc-spheres \
 data.data_tag=rdm_concspheres_k500n500_noninfdist_moreoffmfld_inferred_maxtdelta_1e-4 \
 data.data_params.train.N=2500000 \
 data.data_params.train.num_neg=2000000 \
 data.data_params.train.k=500 \
 data.data_params.train.n=500 \
 data.data_params.train.max_t_delta=1e-4 \
 data.data_params.train.max_norm=0.1 \
 data.data_params.val.N=200000 \
 data.data_params.val.num_neg=100000 \
 data.data_params.val.k=500 \
 data.data_params.val.n=500 \
 data.data_params.val.max_norm=0.1 \
 data.data_params.test.N=200000 \
 data.data_params.test.num_neg=100000 \
 data.data_params.test.k=500 \
 data.data_params.test.n=500 \
 data.data_params.test.max_norm=0.1 \
 model.input_size=500 \
 data.generate=True \
 task=regression
```

### Intertwined Swiss Rolls

#### Distance Learner

```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=1 num_epochs=1000 cooldown=700 warmup=10 lr=1e-6 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-ittw-swissrolls \
 data.data_tag=rdm_swrolls_k2n50_noninfdist_moreoffmfld_inferred_maxtdelta=1e=3 \
 data.logdir=/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_swrolls_test \
 data.backup_dir=/azuredrive/deepimage/data2/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rdm_swrolls_test \
 data.data_params.train.k=2 \
 data.data_params.train.n=50 \
 data.data_params.train.N=100000 \
 data.data_params.train.num_neg=50000 \
 data.data_params.val.k=2 \
 data.data_params.val.n=50 \
 data.data_params.test.k=2 \
 data.data_params.test.n=50 \
 model.input_size=50 \
 data.generate=False \
 task=regression
```

#### Standard Classifier


```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=1 num_epochs=1000 cooldown=700 warmup=10 lr=1e-6 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-ittw-swissrolls \
 data.data_tag=rdm_swrolls_k2n50_noninfdist_moreoffmfld_inferred_maxtdelta=1e=3 \
 data.logdir=/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_swrolls_test \
 data.backup_dir=/azuredrive/deepimage/data2/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rdm_swrolls_test \
 data.data_params.train.k=2 \
 data.data_params.train.n=50 \
 data.data_params.train.N=100000 \
 data.data_params.train.num_neg=50000 \
 data.data_params.val.k=2 \
 data.data_params.val.n=50 \
 data.data_params.test.k=2 \
 data.data_params.test.n=50 \
 model.input_size=50 \
 data.generate=False \
 task=clf
```



### Well-separated spheres

```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=2 num_epochs=1000 cooldown=700 warmup=10 lr=1e-5 batch_size=512 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-ws-spheres \
 data.data_tag=rdm_wsspheres_samerot_k2n50_noninfdist_moreoffmfld_inferred_maxtdelta=1e=3 \
 data.logdir=/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_wsspheres_test \
 data.backup_dir=/azuredrive/deepimage/data2/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rdm_wsspheres_test/VM2 \
 data.data_params.train.k=2 \
 data.data_params.train.n=50 \
 data.data_params.train.N=1050000 \
 data.data_params.train.num_neg=1000000 \
 data.data_params.train.same_rot=True \
 data.data_params.val.k=2 \
 data.data_params.val.n=50 \
 data.data_params.val.same_rot=True \
 data.data_params.test.k=2 \
 data.data_params.test.n=50 \
 data.data_params.test.same_rot=True \
 model.input_size=50 \
 data.generate=True \
 task=regression
```






### for fast debugging
```bash
python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1.5e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-conc-spheres \
 data.data_tag=rdm_concspheres_k2n2_noninfdist_bs4096_inferred_debug \
 data.data_params.train.N=20000 \
 data.data_params.train.num_neg=10000 \
 data.data_params.train.k=2 \
 data.data_params.train.n=2 \
 data.data_params.val.N=20 \
 data.data_params.val.num_neg=10 \
 data.data_params.val.k=2 \
 data.data_params.val.n=2 \
 data.data_params.test.N=20 \
 data.data_params.test.num_neg=10 \
 data.data_params.test.k=2 \
 data.data_params.test.n=2 \
 model.input_size=2 \
 data.generate=True \
 task=regression
```

### for reloading from checkpoint

```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=1 num_epochs=1000 cooldown=-1 warmup=-1 lr=1.5e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-conc-spheres \
 data.data_tag=rdm_concspheres_k50n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_5e-3 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.max_t_delta=0.5e-3 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.val.N=200000 \
 data.data_params.val.num_neg=100000 \
 data.data_params.val.k=50 \
 data.data_params.val.n=500 \
 data.data_params.val.max_norm=0.14 \
 data.data_params.test.N=200000 \
 data.data_params.test.num_neg=100000 \
 data.data_params.test.k=50 \
 data.data_params.test.n=500 \
 data.data_params.test.max_norm=0.14 \
 model.input_size=500 \
 data.generate=True \
 task=regression
```