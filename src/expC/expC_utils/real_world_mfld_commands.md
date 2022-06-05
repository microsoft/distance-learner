# Real-world Manifolds

## Distance Learner

```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1e-5 batch_size=4096 debug=False loss_func=std_mse ftname=points tgtname=actual_distances OFF_MFLD_LABEL=9 data.mtype=mnist data.logdir="/mnt/t-achetan/expC_dist_learner_for_adv_ex/mnist_test/" \
 data.data_tag=mnist_18_val_only_moreoffmfldv3 \
 data.data_params.train.num_neg=6000000 \
 model.input_size=784 \
 data.generate=True \
 task=regression 
```