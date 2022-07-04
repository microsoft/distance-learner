# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
cd ../expC


python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1e-5 batch_size=512 debug=False data.mtype=inf-ws-spheres tgtname=normed_actual_distances loss_func=std_mse \
 data.data_tag=rdm_wsspheres_m2n500 \
 data.logdir="./data" \
 data.backup_dir="./data" \
 data.data_params.train.k=3 \
 data.data_params.train.n=500 \
 data.data_params.train.N=1050000 \
 data.data_params.train.num_neg=1000000 \
 data.data_params.train.same_rot=True \
 data.data_params.val.k=3 \
 data.data_params.val.n=500 \
 data.data_params.val.same_rot=True \
 data.data_params.test.k=3 \
 data.data_params.test.n=500 \
 data.data_params.test.same_rot=True \
 model.input_size=500 \
 data.generate=True \
 task=regression


 python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1e-5 batch_size=512 debug=False data.mtype=inf-ws-spheres tgtname=normed_actual_distances loss_func=std_mse \
 data.data_tag=rdm_wsspheres_m2n500 \
 data.logdir="./data" \
 data.backup_dir="./data" \
 data.data_params.train.k=3 \
 data.data_params.train.n=500 \
 data.data_params.train.N=1050000 \
 data.data_params.train.num_neg=1000000 \
 data.data_params.train.same_rot=True \
 data.data_params.val.k=3 \
 data.data_params.val.n=500 \
 data.data_params.val.same_rot=True \
 data.data_params.test.k=3 \
 data.data_params.test.n=500 \
 data.data_params.test.same_rot=True \
 model.input_size=500 \
 on_mfld_noise=0 \
 adv_train=False \
 test_off_mfld=False \
 data.generate=False \
 task=clf
