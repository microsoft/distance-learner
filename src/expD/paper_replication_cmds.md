# Commands for replicating dataset for "The Relationship Between High-Dimensional Geometry and Adversarial Examples"

First run the Distance Learner for the dataset that is similar to the one given in the paper:


```bash
python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1.5e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k50n500_noninfdist_moreoffmfldv5_bs4096_highmn40 \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.N=20500000 \
 data.data_params.train.num_neg=20000000 \
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
 data.data_tag=rdm_concspheres_k50n500_noninfdist_moreoffmfld_advtraindebug_bs4096_eps=7e-2 \
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
 data.generate=False \
 adv_train=True \
 adv_train_params.atk_eps=7e-2 \
 on_mfld_noise=0 \
 test_off_mfld=False \
 task=clf
```

## Inferred Manifold commands

```bash
python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1.5e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-conc-spheres \
 data.data_tag=rdm_concspheres_k50n500_noninfdist_moreoffmfld_bs4096_inferred \
 data.data_params.train.N=1000000 \
 data.data_params.train.num_neg=500000 \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.val.N=200000 \
 data.data_params.val.num_neg=100000 \
 data.data_params.val.k=50 \
 data.data_params.val.n=500 \
 data.data_params.test.N=200000 \
 data.data_params.test.num_neg=100000 \
 data.data_params.test.k=50 \
 data.data_params.test.n=500 \
 model.input_size=500 \
 data.generate=True \
 task=regression
```