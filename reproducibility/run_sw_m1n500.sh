cd ../src/expC

python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1e-6 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-ittw-swissrolls \
 data.data_tag=rdm_swrolls_m1n500 \
 data.logdir="./data" \
 data.backup_dir="./data" \
 data.data_params.train.k=2 \
 data.data_params.train.n=500 \
 data.data_params.train.N=150000 \
 data.data_params.train.num_neg=50000 \
 data.data_params.val.k=2 \
 data.data_params.val.n=500 \
 data.data_params.test.k=2 \
 data.data_params.test.n=500 \
 model.input_size=500 \
 data.generate=True \
 task=regression

python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=8e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-ittw-swissrolls \
 data.data_tag=rdm_swrolls_m1n500 \
 data.logdir="./data" \
 data.backup_dir="./data" \
 data.data_params.train.k=2 \
 data.data_params.train.n=500 \
 data.data_params.train.N=150000 \
 data.data_params.train.num_neg=50000 \
 data.data_params.val.k=2 \
 data.data_params.val.n=500 \
 data.data_params.test.k=2 \
 data.data_params.test.n=500 \
 model.input_size=500 \
 data.generate=False \
 on_mfld_noise=0 \
 test_off_mfld=False \
 task=clf
