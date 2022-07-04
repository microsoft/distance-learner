## Concentric Spheres

#### Distance Learner

```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=0 model_seed=8296456 num_epochs=1000 cooldown=700 warmup=10 lr=1.5e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-conc-spheres data.logdir="/data/t-achetan/dumps/expC_dist_learner_for_adv_ex/rdm_concspheres_test/" \
 data.data_tag=rdm_concspheres_k50n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.max_t_delta=1e-3 \
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
 data.generate=False \
 task=regression
 ```


 ```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with model_seed=8296456 cuda=1 num_epochs=1000 cooldown=700 warmup=10 lr=1.5e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-conc-spheres \
 data.data_tag=rdm_concspheres_k25n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.k=25 \
 data.data_params.train.n=500 \
 data.data_params.train.max_t_delta=1e-3 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.val.N=200000 \
 data.data_params.val.num_neg=100000 \
 data.data_params.val.k=25 \
 data.data_params.val.n=500 \
 data.data_params.val.max_norm=0.14 \
 data.data_params.test.N=200000 \
 data.data_params.test.num_neg=100000 \
 data.data_params.test.k=25 \
 data.data_params.test.n=500 \
 data.data_params.test.max_norm=0.14 \
 model.input_size=500 \
 data.generate=False \
 task=regression
 ```

#### Standard Classifier


```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=3 num_epochs=1000 cooldown=700 warmup=10 lr=8e-5 batch_size=4096 debug=False data.mtype=inf-conc-spheres \
 data.data_tag=rdm_concspheres_k25n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.train.k=25 \
 data.data_params.train.n=500 \
 data.data_params.train.max_t_delta=1e-3 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.val.N=200000 \
 data.data_params.val.num_neg=100000 \
 data.data_params.val.k=25 \
 data.data_params.val.n=500 \
 data.data_params.val.max_norm=0.14 \
 data.data_params.test.N=200000 \
 data.data_params.test.num_neg=100000 \
 data.data_params.test.k=25 \
 data.data_params.test.n=500 \
 data.data_params.test.max_norm=0.14 \
 model.input_size=500 \
 on_mfld_noise=0 \
 adv_train=True \
 adv_train_params.atk_eps=8e-2 \
 test_off_mfld=False \
 data.generate=False \
 task=clf
 ```

```bash
/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=3 num_epochs=1000 cooldown=700 warmup=10 lr=8e-5 batch_size=4096 debug=False data.mtype=inf-conc-spheres \
 data.data_tag=rdm_concspheres_k3n50_noninfdist_moreoffmfld_inferred_maxtdelta_1e-3 \
 data.data_params.train.N=2500000 \
 data.data_params.train.num_neg=2000000 \
 data.data_params.train.k=3 \
 data.data_params.train.n=50 \
 data.data_params.train.max_t_delta=1e-3 \
 data.data_params.train.max_norm=0.14 \
 data.data_params.val.N=200000 \
 data.data_params.val.num_neg=100000 \
 data.data_params.val.k=3 \
 data.data_params.val.n=50 \
 data.data_params.val.max_norm=0.14 \
 data.data_params.test.N=200000 \
 data.data_params.test.num_neg=100000 \
 data.data_params.test.k=3 \
 data.data_params.test.n=50 \
 data.data_params.test.max_norm=0.14 \
 model.input_size=50 \
 on_mfld_noise=0 \
 adv_train=True \
 adv_train_params.atk_eps=8e-2 \
 test_off_mfld=False \
 data.generate=False \
 task=clf
```



```bash
 python3 get_attack_perf.py with debug=False "attack.atk_routine=['my']" input_files.settings_type=list
 ```

 ```bash
 /root/anaconda3/bin/python3 get_attack_perf.py with debug=False "attack.atk_routine=['my']" input_files.settings_type=list input_files.proj_dir="/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_concspheres_test/" dump_dir="/data/t-achetan/dumps/expC_dist_learner_for_adv_ex/rdm_concspheres_test/attack_perfs_on_runs" "input_files.settings_to_analyze=['rdm_concspheres_k25n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3/5','rdm_concspheres_k25n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3/6']"
 ```

  ```bash
 /root/anaconda3/bin/python3 get_attack_perf.py with debug=False "attack.atk_routine=['my']" input_files.settings_type=list input_files.proj_dir="/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_swrolls_test/" dump_dir="/data/t-achetan/dumps/expC_dist_learner_for_adv_ex/rdm_swrolls_test/attack_perfs_on_runs" "input_files.settings_to_analyze=['rdm_swrolls_k50n500_noninfdist_moreoffmfldv3_inferred_maxtdelta_1e-3/1', 'rdm_wsspheres_k50n500_noninfdist_moreoffmfldv3_bs4096_highmn40_inferred_maxtdelta_1e-3/6']" attack.eps="[0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01 , 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02 , 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03 , 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039]" th_analyze="[float('inf'), 0.004, 0.008, 0.012, 0.016, 0.02 , 0.024, 0.028, 0.032, 0.036, 0.04 , 0.044, 0.048, 0.052, 0.056]"
 ```

 ```bash
 python3 get_attack_perf.py with debug=False input_files.settings_type=list input_files.proj_dir="/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_swrolls_test/" dump_dir="/data/t-achetan/dumps/expC_dist_learner_for_adv_ex/rdm_swrolls_test/attack_perfs_on_runs/" "input_files.settings_to_analyze=['rdm_swrolls_k50n500_noninfdist_moreoffmfldv3/1']" attack.eps="[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]" th_analyze="np.array([float('inf'), 0.025, 0.05, 0.1, 0.125, 0.15, 0.2, 0.225, 0.25, 0.3, 0.325, 0.35])"
 ```