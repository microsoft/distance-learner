# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# directory to store results
LOG_DIR="../../data"

cd ../pipeline

python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1e-5 batch_size=512 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-ittw-swissrolls \
 data.data_tag=rdm_swrolls_m1n50 \
 data.logdir=$LOG_DIR \
 data.backup_dir=$LOG_DIR \
 data.data_params.train.k=2 \
 data.data_params.train.n=50 \
 data.data_params.train.N=100000 \
 data.data_params.train.num_neg=50000 \
 data.data_params.val.k=2 \
 data.data_params.val.n=50 \
 data.data_params.test.k=2 \
 data.data_params.test.n=50 \
 model.input_size=50 \
 data.generate=True \
 task=regression

python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1e-5 batch_size=512 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-ittw-swissrolls \
 data.data_tag=rdm_swrolls_m1n50 \
 data.logdir=$LOG_DIR \
 data.backup_dir=$LOG_DIR \
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
 on_mfld_noise=0 \
 test_off_mfld=False \
 task=clf
