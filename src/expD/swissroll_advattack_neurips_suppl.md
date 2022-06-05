### Intertwined Swiss Roll Adversarial Attacks


#### Distance Learner

/root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=1 num_epochs=1000 cooldown=700 warmup=10 lr=1e-5 batch_size=512 debug=False loss_func=masked_mse tgtname=normed_actual_distances data.mtype=inf-ittw-swissrolls2 \
 data.data_tag=rdm_swrolls_k50n500_noninfdist_moreoffmfldv3_inferred_cfg2_maxtdelta_1e-3_maskedloss \
 data.logdir=/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_swrolls_test \
 data.backup_dir=/azuredrive/deepimage/data2/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rdm_swrolls_test \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.val.k=50 \
 data.data_params.val.n=500 \
 data.data_params.test.k=50 \
 data.data_params.test.n=500 \
 model.input_size=500 \
 data.generate=True \
 task=regression


 /root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=1 num_epochs=1000 cooldown=700 warmup=10 lr=1e-5 batch_size=512 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-ittw-swissrolls \
 data.data_tag=rdm_swrolls_k50n500_noninfdist_moreoffmfldv3 \
 data.logdir=/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_swrolls_test \
 data.backup_dir=/azuredrive/deepimage/data2/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rdm_swrolls_test \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.val.k=50 \
 data.data_params.val.n=500 \
 data.data_params.test.k=50 \
 data.data_params.test.n=500 \
 model.input_size=500 \
 data.generate=False \
 task=regression


 #### Standard Classifier

```bash
 /root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=2 num_epochs=1000 cooldown=700 warmup=10 lr=5e-6 batch_size=4096 debug=False data.mtype=inf-ittw-swissrolls \
 data.data_tag=rdm_swrolls_k50n500_noninfdist_moreoffmfldv3_inferred_maxtdelta_1e-3 \
 data.logdir=/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_swrolls_test \
 data.backup_dir=/azuredrive/deepimage/data2/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rdm_swrolls_test \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.val.k=50 \
 data.data_params.val.n=500 \
 data.data_params.test.k=50 \
 data.data_params.test.n=500 \
 model.input_size=500 \
 data.generate=False \
 adv_train=False \
 on_mfld_noise=0 \
 test_off_mfld=False \
 on_mfld_noise=0 \
 test_off_mfld=False \
 task=clf
 ```


 #### Adversarial Classifier

```bash
 /root/anaconda3/bin/python3 learn_cls_from_dist.py with cuda=1 num_epochs=1000 cooldown=700 warmup=10 lr=8e-5 batch_size=4096 debug=False data.mtype=inf-ittw-swissrolls \
 data.data_tag=rdm_swrolls_k50n500_noninfdist_moreoffmfldv3_inferred_maxtdelta_1e-3 \
 data.logdir=/mnt/t-achetan/expC_dist_learner_for_adv_ex/rdm_swrolls_test \
 data.backup_dir=/azuredrive/deepimage/data2/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rdm_swrolls_test \
 data.data_params.train.k=50 \
 data.data_params.train.n=500 \
 data.data_params.train.N=6500000 \
 data.data_params.train.num_neg=6000000 \
 data.data_params.val.k=50 \
 data.data_params.val.n=500 \
 data.data_params.test.k=50 \
 data.data_params.test.n=500 \
 model.input_size=500 \
 data.generate=False \
 adv_train=True \
 adv_train_params.atk_eps=8e-2 \
 on_mfld_noise=0 \
 test_off_mfld=False \
 on_mfld_noise=0 \
 task=clf
 ```