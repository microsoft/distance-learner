# Real-world Manifolds

## Distance Learner

```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=3 num_epochs=1000 cooldown=700 warmup=10 lr=5e-4 batch_size=4096 debug=False loss_func=std_mse ftname=points tgtname=normed_actual_distances model.hidden_sizes="[512, 512, 512, 512, 512, 512, 512, 512]" OFF_MFLD_LABEL=9 data.mtype=mnist data.logdir="/mnt/t-achetan/expC_dist_learner_for_adv_ex/mnist_test/" \
 data.data_tag=mnist_18_val_only_moreoffmfldv3_nn10_highmn6_nonormom \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.nn=10 \
 data.data_params.train.max_norm=6 \
 data.data_params.train.transform=None \
 data.data_params.val.nn=10 \
 data.data_params.val.max_norm=6 \
 data.data_params.val.transform=None \
 data.data_params.test.nn=10 \
 data.data_params.test.max_norm=6 \
 data.data_params.test.transform=None \
 model.input_size=784 \
 data.generate=True \
 task=regression 
```

## Standard Classifier

```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=2 num_epochs=1000 cooldown=700 warmup=10 lr=1e-6 batch_size=4096 debug=False ftname=points tgtname=class_idx OFF_MFLD_LABEL=2 data.mtype=mnist data.logdir="/mnt/t-achetan/expC_dist_learner_for_adv_ex/mnist_test/" \
 data.data_tag=mnist_18_val_only_moreoffmfldv3_nn10_highmn_nonormpts \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.max_norm=2 \
 data.data_params.train.nn=10 \
 data.data_params.val.nn=10 \
 data.data_params.val.max_norm=2 \
 data.data_params.test.nn=10 \
 data.data_params.test.max_norm=2 \
 model.input_size=784 \
 data.generate=False \
 test_off_mfld=False \
 adv_train=False \
 on_mfld_noise=0 \
 task=clf 
```

```
python3 get_attack_perf.py with dump_dir="/data/t-achetan/dumps/expC_dist_learner_for_adv_ex/mnist_test/attack_perfs_on_runs" true_cls_attr_name="class_idx" true_cls_batch_attr_name="class_idx" "th_analyze=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]" "attack.eps=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]" input_files.proj_dir="/mnt/t-achetan/expC_dist_learner_for_adv_ex/mnist_test/" input_files.settings_type=list "input_files.settings_to_analyze=['mnist_18_val_only_moreoffmfldv3_nn10_highmn_nonormpts/3']" "attack.atk_routine=['my']"

python3 get_attack_perf.py with dump_dir="/data/t-achetan/dumps/expC_dist_learner_for_adv_ex/mnist_test/attack_perfs_on_runs" true_cls_attr_name="class_idx" true_cls_batch_attr_name="class_idx" "th_analyze=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]" "attack.eps=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]" input_files.proj_dir="/mnt/t-achetan/expC_dist_learner_for_adv_ex/mnist_test/" input_files.settings_type=list "input_files.settings_to_analyze=['mnist_18_val_only_moreoffmfldv3_nn10_highmn_nonormpts/3']" "attack.atk_routine=['my']"
```