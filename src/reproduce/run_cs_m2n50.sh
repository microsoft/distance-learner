# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# directory to store results
LOG_DIR="../../data"

cd ../pipeline

# train distlearner
python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1.5e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-conc-spheres data.logdir=$LOG_DIR \
 data.data_tag=rdm_concspheres_m2n50 \
 data.data_params.train.N=1500000 \
 data.data_params.train.num_neg=500000 \
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
 data.generate=True \
 task=regression

# train stdclf
python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=8e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-conc-spheres data.logdir=$LOG_DIR \
 data.data_tag=rdm_concspheres_m2n50 \
 data.data_params.train.N=1500000 \
 data.data_params.train.num_neg=500000 \
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
 adv_train=False \
 test_off_mfld=False \
 data.generate=False \
 task=clf


