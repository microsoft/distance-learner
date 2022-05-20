## Concentric Spheres

#### Distance Learner

```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=1 num_epochs=1000 cooldown=700 warmup=10 lr=1.5e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-conc-spheres \
 data.data_tag=rdm_concspheres_k2n50_noninfdist_moreoffmfld_inferred_maxtdelta_1e-3 \
 data.data_params.train.N=1500000 \
 data.data_params.train.num_neg=1000000 \
 data.data_params.train.k=2 \
 data.data_params.train.n=50 \
 data.data_params.train.max_t_delta=1e-3 \
 data.data_params.train.max_norm=0.1 \
 data.data_params.val.N=200000 \
 data.data_params.val.num_neg=100000 \
 data.data_params.val.k=2 \
 data.data_params.val.n=50 \
 data.data_params.val.max_norm=0.1 \
 data.data_params.test.N=200000 \
 data.data_params.test.num_neg=100000 \
 data.data_params.test.k=2 \
 data.data_params.test.n=50 \
 data.data_params.test.max_norm=0.1 \
 model.input_size=50 \
 data.generate=True \
 task=regression
 ```

#### Standard Classifier


```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=1 num_epochs=1000 cooldown=700 warmup=10 lr=1.5e-5 batch_size=4096 debug=False data.mtype=inf-conc-spheres \
 data.data_tag=rdm_concspheres_k2n50_noninfdist_moreoffmfld_inferred_maxtdelta_1e-3 \
 data.data_params.train.N=1500000 \
 data.data_params.train.num_neg=1000000 \
 data.data_params.train.k=2 \
 data.data_params.train.n=50 \
 data.data_params.train.max_t_delta=1e-3 \
 data.data_params.train.max_norm=0.1 \
 data.data_params.val.N=200000 \
 data.data_params.val.num_neg=100000 \
 data.data_params.val.k=2 \
 data.data_params.val.n=50 \
 data.data_params.val.max_norm=0.1 \
 data.data_params.test.N=200000 \
 data.data_params.test.num_neg=100000 \
 data.data_params.test.k=2 \
 data.data_params.test.n=50 \
 data.data_params.test.max_norm=0.1 \
 model.input_size=50 \
 on_mfld_noise=0 \
 adv_train=True \
 adv_train_params.atk_eps=5e-2 \
 test_off_mfld=False \
 data.generate=False \
 task=clf
 ```

```
 python3 get_attack_perf.py with debug=False "attack.atk_routine=['my']" input_files.settings_type=list
 ```

 ```
 python3 get_attack_perf.py with debug=False "attack.atk_routine=['my']" input_files.settings_type=list input_files.proj_dir="/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_wsspheres_test/" dump_dir="/data/t-achetan/dumps/expC_dist_learner_for_adv_ex/rdm_wsspheres_test/attack_perfs_on_runs" "input_files.settings_to_analyze=['rdm_wsspheres_samerot_k2n500_noninfdist_moreoffmfld_inferred_maxtdelta=1e=3/5', 'rdm_wsspheres_samerot_k2n500_noninfdist_moreoffmfld_inferred_maxtdelta=1e=3/4', 'rdm_wsspheres_samerot_k2n500_noninfdist_moreoffmfld_inferred_maxtdelta=1e=3/6', 'rdm_wsspheres_samerot_k2n500_noninfdist_moreoffmfld_inferred_maxtdelta=1e=3/7']"
 ```