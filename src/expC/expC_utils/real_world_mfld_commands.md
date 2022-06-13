# Real-world Manifolds

## Distance Learner

```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=2 num_epochs=1000 cooldown=700 warmup=10 lr=1e-5 batch_size=4096 debug=False loss_func=std_mse ftname=normed_points tgtname=normed_actual_distances model.hidden_sizes="[512, 512, 512, 512, 512, 512, 512, 512]" OFF_MFLD_LABEL=9 data.mtype=mnist data.logdir="/mnt/t-achetan/expC_dist_learner_for_adv_ex/mnist_test/" \
 data.data_tag=mnist_18_val_only_moreoffmfldv3_nn10_highmn \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.nn=10 \
 data.data_params.train.max_norm=2 \
 data.data_params.val.nn=10 \
 data.data_params.val.max_norm=2 \
 data.data_params.test.nn=10 \
 data.data_params.test.max_norm=2 \
 model.input_size=784 \
 data.generate=True \
 task=regression 
```

## Standard Classifier

```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=2 num_epochs=1000 cooldown=700 warmup=10 lr=1e-6 batch_size=4096 debug=False loss_func=std_mse ftname=points OFF_MFLD_LABEL=9 data.mtype=mnist data.logdir="/mnt/t-achetan/expC_dist_learner_for_adv_ex/mnist_test/" \
 data.data_tag=mnist_18_val_only_moreoffmfldv3_nn10 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.nn=10 \
 data.data_params.val.nn=10 \
 data.data_params.test.nn=10 \
 model.input_size=784 \
 data.generate=False \
 task=clf 
```