### Well-separated spheres Adversarial Attacks 


#### Distance Learner

/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=2 num_epochs=1000 cooldown=700 warmup=10 lr=1.5e-5 batch_size=4096 debug=False data.mtype=inf-ws-spheres tgtname=normed_actual_distances loss_func=std_mse \
 data.data_tag=rdm_wsspheres_k50n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3 \
 data.logdir=/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_wsspheres_test \
 data.backup_dir=/azuredrive/deepimage/data2/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rdm_wsspheres_test/VM3 \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.train.same_rot=False \
 data.data_params.val.k=50 \
 data.data_params.val.n=500 \
 data.data_params.val.N=200000 \
 data.data_params.val.max_norm=0.14 \
 data.data_params.val.inferred=False \
 data.data_params.val.same_rot=False \
 data.data_params.test.k=50 \
 data.data_params.test.n=500 \
 data.data_params.test.N=200000 \
 data.data_params.test.max_norm=0.14 \
 data.data_params.test.inferred=False \
 data.data_params.test.same_rot=False \
 model.input_size=500 \
 data.generate=True \
 task=regression



#### Standard classifier
```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=1 num_epochs=1000 cooldown=700 warmup=10 lr=8e-5 batch_size=4096 debug=False data.mtype=inf-ws-spheres \
 data.data_tag=rdm_wsspheres_k50n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3 \
 data.logdir=/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_wsspheres_test \
 data.backup_dir=/azuredrive/deepimage/data2/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rdm_wsspheres_test/VM3 \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.train.same_rot=False \
 data.data_params.val.k=50 \
 data.data_params.val.n=500 \
 data.data_params.val.max_norm=0.14 \
 data.data_params.val.same_rot=False \
 data.data_params.test.k=50 \
 data.data_params.test.n=500 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.test.same_rot=False \
 model.input_size=500 \
 data.generate=False \
 adv_train=False \
 on_mfld_noise=0 \
 test_off_mfld=False \
 task=clf
```

#### Robust  classifier
```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=1 num_epochs=1000 cooldown=700 warmup=10 lr=8e-5 batch_size=4096 debug=False data.mtype=inf-ws-spheres \
 data.data_tag=rdm_wsspheres_samerot_k50n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3 \
 data.logdir=/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_wsspheres_test \
 data.backup_dir=/azuredrive/deepimage/data2/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rdm_wsspheres_test/VM3 \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.train.same_rot=True \
 data.data_params.train.same_rot=True \
 data.data_params.val.k=50 \
 data.data_params.val.n=500 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.train.same_rot=True \
 data.data_params.val.same_rot=True \
 data.data_params.test.k=50 \
 data.data_params.test.n=500 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.train.same_rot=True \
 data.data_params.test.same_rot=True \
 model.input_size=500 \
 data.generate=False \
 adv_train=True \
 adv_train_params.atk_eps=8e-2 \
 on_mfld_noise=0 \
 test_off_mfld=False \
 task=clf
```


### Adversarial attack command

```bash
/root/anaconda3/bin/python3 get_attack_perf.py with debug=False "attack.atk_routine=['my']" cuda=0 input_files.settings_type=list input_files.proj_dir="/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_wsspheres_test/" dump_dir="/data/t-achetan/dumps/expC_dist_learner_for_adv_ex/rdm_wsspheres_test/attack_perfs_on_runs" "input_files.settings_to_analyze=['rdm_wsspheres_samerot_k50n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3/2', 'rdm_wsspheres_samerot_k50n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3/3', 'rdm_wsspheres_samerot_k50n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3/4', 'rdm_wsspheres_samerot_k50n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3/5']"
```