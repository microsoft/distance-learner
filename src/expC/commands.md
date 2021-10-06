# Commands for Experiment 3a

## Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py print_config with cuda=0 data.logdir=/azuredrive/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.data_tag=rdm_swiss_rolls_k2n3 data.data_params.train.k=2 data.data_params.val.k=2 data.data_params.test.k=2 data.data_params.train.n=3 data.data_params.val.n=3 data.data_params.test.n=3 data.generate=True model.input_size=3 task=regression
```

## Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py print_config with cuda=0 data.logdir=/azuredrive/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.data_tag=rdm_swiss_rolls_k2n3 data.data_params.train.k=2 data.data_params.val.k=2 data.data_params.test.k=2 data.data_params.train.n=3 data.data_params.val.n=3 data.data_params.test.n=3 model.input_size=3 task=clf train_on_onmfld=True
```

## Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py print_config with cuda=0 data.logdir=/azuredrive/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.data_tag=rdm_swiss_rolls_k2n3 data.data_params.train.k=2 data.data_params.val.k=2 data.data_params.test.k=2 data.data_params.train.n=3 data.data_params.val.n=3 data.data_params.test.n=3 model.input_size=3 model.output_size=3 task=clf train_on_onmfld=False 
```
