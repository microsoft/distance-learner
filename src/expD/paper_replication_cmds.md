# Commands for replicating dataset for "The Relationship Between High-Dimensional Geometry and Adversarial Examples"

First run the Distance Learner for the dataset that is similar to the one given in the paper:


```bash
python3 learn_cls_from_dist.py with cuda=2 num_epochs=18 cooldown=1 warmup=1 lr= batch_size=512 debug=False loss_func=std_mse tgtname=normed_actual_distances \ 
 "init_wts=/data/dumps/expC_dist_learner_for_adv_ex/rdm_concspheres_test/rdm_concspheres_k50n500_noninfdist_moreoffmfldv4/1/models/ckpt.pth" \
 data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k50n500_noninfdist_moreoffmfldv4 \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.r=1 \
 data.data_params.train.N=12500000 \
 data.data_params.train.num_neg=12000000 \
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
 task=regression
```

```bash
python3 learn_cls_from_dist.py with num_epochs=1000 cooldown=700 lr=1e-5 debug=False cuda=0 \
 data.mtype=conc-spheres \
 data.data_tag=rdm_concspheres_k400n500_noninfdist_moreoffmfld \
 data.data_params.train.k=400 \
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
 data.data_params.val.k=400 \
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
 data.data_params.test.k=400 \
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
 task=clf
```
