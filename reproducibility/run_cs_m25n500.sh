cd ../src/expC

python3 learn_cls_from_dist.py with cuda=0 num_epochs=1000 cooldown=700 warmup=10 lr=1.5e-5 batch_size=4096 debug=False loss_func=std_mse tgtname=normed_actual_distances data.mtype=inf-conc-spheres data.logdir="./data" \
 data.data_tag=rdm_concspheres_m25n500 \
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
 data.generate=True \
 task=regression

python3 learn_cls_from_dist.py with cuda=3 num_epochs=1000 cooldown=700 warmup=10 lr=8e-5 batch_size=4096 debug=False data.mtype=inf-conc-spheres  data.logdir="./data" \
 data.data_tag=rdm_concspheres_m25n500 \
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
 adv_train=False \
 test_off_mfld=False \
 data.generate=False \
 task=clf

python3 learn_cls_from_dist.py with cuda=3 num_epochs=1000 cooldown=700 warmup=10 lr=8e-5 batch_size=4096 debug=False data.mtype=inf-conc-spheres  data.logdir="./data" \
 data.data_tag=rdm_concspheres_m25n500 \
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
 adv_train_params.atk_eps=5e-2 \
 test_off_mfld=False \
 data.generate=False \
 task=clf


python3 learn_cls_from_dist.py with cuda=3 num_epochs=1000 cooldown=700 warmup=10 lr=8e-5 batch_size=4096 debug=False data.mtype=inf-conc-spheres  data.logdir="./data" \
 data.data_tag=rdm_concspheres_m25n500 \
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
 
 cd ../expD

python3 get_attack_perf.py with debug=False "attack.atk_routine=['my']" input_files.settings_type=list input_files.proj_dir="./data/rdm_concspheres_test/" dump_dir="./data/rdm_concspheres_test/attack_perfs_on_runs" "input_files.settings_to_analyze=['rdm_concspheres_m25n500/1', 'rdm_concspheres_m25n500/2', 'rdm_concspheres_m25n500/3', 'rdm_concspheres_m25n500/4']"
