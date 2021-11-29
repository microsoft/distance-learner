
# Commands for Experiment 3a

## k = 2, n = 2


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n2 data.data_params.train.k=2 data.data_params.train.n=2 data.data_params.val.k=2 data.data_params.val.n=2 data.data_params.test.k=2 data.data_params.test.n=2 model.input_size=2 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n2/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n2 data.data_params.train.k=2 data.data_params.train.n=2 data.data_params.val.k=2 data.data_params.val.n=2 data.data_params.test.k=2 data.data_params.test.n=2 model.input_size=2 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n2/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n2 data.data_params.train.k=2 data.data_params.train.n=2 data.data_params.val.k=2 data.data_params.val.n=2 data.data_params.test.k=2 data.data_params.test.n=2 model.input_size=2 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n2/3 --on=test --num_points=50000
```

## k = 2, n = 3


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n3 data.data_params.train.k=2 data.data_params.train.n=3 data.data_params.val.k=2 data.data_params.val.n=3 data.data_params.test.k=2 data.data_params.test.n=3 model.input_size=3 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n3/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n3 data.data_params.train.k=2 data.data_params.train.n=3 data.data_params.val.k=2 data.data_params.val.n=3 data.data_params.test.k=2 data.data_params.test.n=3 model.input_size=3 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n3/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n3 data.data_params.train.k=2 data.data_params.train.n=3 data.data_params.val.k=2 data.data_params.val.n=3 data.data_params.test.k=2 data.data_params.test.n=3 model.input_size=3 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n3/3 --on=test --num_points=50000
```

## k = 2, n = 5


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n5 data.data_params.train.k=2 data.data_params.train.n=5 data.data_params.val.k=2 data.data_params.val.n=5 data.data_params.test.k=2 data.data_params.test.n=5 model.input_size=5 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n5/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n5 data.data_params.train.k=2 data.data_params.train.n=5 data.data_params.val.k=2 data.data_params.val.n=5 data.data_params.test.k=2 data.data_params.test.n=5 model.input_size=5 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n5/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n5 data.data_params.train.k=2 data.data_params.train.n=5 data.data_params.val.k=2 data.data_params.val.n=5 data.data_params.test.k=2 data.data_params.test.n=5 model.input_size=5 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n5/3 --on=test --num_points=50000
```

## k = 2, n = 10


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n10 data.data_params.train.k=2 data.data_params.train.n=10 data.data_params.val.k=2 data.data_params.val.n=10 data.data_params.test.k=2 data.data_params.test.n=10 model.input_size=10 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n10/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n10 data.data_params.train.k=2 data.data_params.train.n=10 data.data_params.val.k=2 data.data_params.val.n=10 data.data_params.test.k=2 data.data_params.test.n=10 model.input_size=10 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n10/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n10 data.data_params.train.k=2 data.data_params.train.n=10 data.data_params.val.k=2 data.data_params.val.n=10 data.data_params.test.k=2 data.data_params.test.n=10 model.input_size=10 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n10/3 --on=test --num_points=50000
```

## k = 2, n = 50


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n50 data.data_params.train.k=2 data.data_params.train.n=50 data.data_params.val.k=2 data.data_params.val.n=50 data.data_params.test.k=2 data.data_params.test.n=50 model.input_size=50 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n50/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n50 data.data_params.train.k=2 data.data_params.train.n=50 data.data_params.val.k=2 data.data_params.val.n=50 data.data_params.test.k=2 data.data_params.test.n=50 model.input_size=50 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n50/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n50 data.data_params.train.k=2 data.data_params.train.n=50 data.data_params.val.k=2 data.data_params.val.n=50 data.data_params.test.k=2 data.data_params.test.n=50 model.input_size=50 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n50/3 --on=test --num_points=50000
```

## k = 2, n = 100


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n100 data.data_params.train.k=2 data.data_params.train.n=100 data.data_params.val.k=2 data.data_params.val.n=100 data.data_params.test.k=2 data.data_params.test.n=100 model.input_size=100 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n100/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n100 data.data_params.train.k=2 data.data_params.train.n=100 data.data_params.val.k=2 data.data_params.val.n=100 data.data_params.test.k=2 data.data_params.test.n=100 model.input_size=100 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n100/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n100 data.data_params.train.k=2 data.data_params.train.n=100 data.data_params.val.k=2 data.data_params.val.n=100 data.data_params.test.k=2 data.data_params.test.n=100 model.input_size=100 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n100/3 --on=test --num_points=50000
```

## k = 2, n = 500


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n500 data.data_params.train.k=2 data.data_params.train.n=500 data.data_params.val.k=2 data.data_params.val.n=500 data.data_params.test.k=2 data.data_params.test.n=500 model.input_size=500 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n500/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n500 data.data_params.train.k=2 data.data_params.train.n=500 data.data_params.val.k=2 data.data_params.val.n=500 data.data_params.test.k=2 data.data_params.test.n=500 model.input_size=500 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n500/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n500 data.data_params.train.k=2 data.data_params.train.n=500 data.data_params.val.k=2 data.data_params.val.n=500 data.data_params.test.k=2 data.data_params.test.n=500 model.input_size=500 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n500/3 --on=test --num_points=50000
```

## k = 2, n = 1000


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n1000 data.data_params.train.k=2 data.data_params.train.n=1000 data.data_params.val.k=2 data.data_params.val.n=1000 data.data_params.test.k=2 data.data_params.test.n=1000 model.input_size=1000 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n1000/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n1000 data.data_params.train.k=2 data.data_params.train.n=1000 data.data_params.val.k=2 data.data_params.val.n=1000 data.data_params.test.k=2 data.data_params.test.n=1000 model.input_size=1000 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n1000/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k2n1000 data.data_params.train.k=2 data.data_params.train.n=1000 data.data_params.val.k=2 data.data_params.val.n=1000 data.data_params.test.k=2 data.data_params.test.n=1000 model.input_size=1000 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k2n1000/3 --on=test --num_points=50000
```

## k = 3, n = 3


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n3 data.data_params.train.k=3 data.data_params.train.n=3 data.data_params.val.k=3 data.data_params.val.n=3 data.data_params.test.k=3 data.data_params.test.n=3 model.input_size=3 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n3/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n3 data.data_params.train.k=3 data.data_params.train.n=3 data.data_params.val.k=3 data.data_params.val.n=3 data.data_params.test.k=3 data.data_params.test.n=3 model.input_size=3 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n3/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n3 data.data_params.train.k=3 data.data_params.train.n=3 data.data_params.val.k=3 data.data_params.val.n=3 data.data_params.test.k=3 data.data_params.test.n=3 model.input_size=3 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n3/3 --on=test --num_points=50000
```

## k = 3, n = 5


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n5 data.data_params.train.k=3 data.data_params.train.n=5 data.data_params.val.k=3 data.data_params.val.n=5 data.data_params.test.k=3 data.data_params.test.n=5 model.input_size=5 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n5/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n5 data.data_params.train.k=3 data.data_params.train.n=5 data.data_params.val.k=3 data.data_params.val.n=5 data.data_params.test.k=3 data.data_params.test.n=5 model.input_size=5 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n5/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n5 data.data_params.train.k=3 data.data_params.train.n=5 data.data_params.val.k=3 data.data_params.val.n=5 data.data_params.test.k=3 data.data_params.test.n=5 model.input_size=5 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n5/3 --on=test --num_points=50000
```

## k = 3, n = 10


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n10 data.data_params.train.k=3 data.data_params.train.n=10 data.data_params.val.k=3 data.data_params.val.n=10 data.data_params.test.k=3 data.data_params.test.n=10 model.input_size=10 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n10/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n10 data.data_params.train.k=3 data.data_params.train.n=10 data.data_params.val.k=3 data.data_params.val.n=10 data.data_params.test.k=3 data.data_params.test.n=10 model.input_size=10 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n10/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n10 data.data_params.train.k=3 data.data_params.train.n=10 data.data_params.val.k=3 data.data_params.val.n=10 data.data_params.test.k=3 data.data_params.test.n=10 model.input_size=10 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n10/3 --on=test --num_points=50000
```

## k = 3, n = 50


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n50 data.data_params.train.k=3 data.data_params.train.n=50 data.data_params.val.k=3 data.data_params.val.n=50 data.data_params.test.k=3 data.data_params.test.n=50 model.input_size=50 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n50/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n50 data.data_params.train.k=3 data.data_params.train.n=50 data.data_params.val.k=3 data.data_params.val.n=50 data.data_params.test.k=3 data.data_params.test.n=50 model.input_size=50 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n50/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n50 data.data_params.train.k=3 data.data_params.train.n=50 data.data_params.val.k=3 data.data_params.val.n=50 data.data_params.test.k=3 data.data_params.test.n=50 model.input_size=50 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n50/3 --on=test --num_points=50000
```

## k = 3, n = 100


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n100 data.data_params.train.k=3 data.data_params.train.n=100 data.data_params.val.k=3 data.data_params.val.n=100 data.data_params.test.k=3 data.data_params.test.n=100 model.input_size=100 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n100/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n100 data.data_params.train.k=3 data.data_params.train.n=100 data.data_params.val.k=3 data.data_params.val.n=100 data.data_params.test.k=3 data.data_params.test.n=100 model.input_size=100 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n100/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n100 data.data_params.train.k=3 data.data_params.train.n=100 data.data_params.val.k=3 data.data_params.val.n=100 data.data_params.test.k=3 data.data_params.test.n=100 model.input_size=100 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n100/3 --on=test --num_points=50000
```

## k = 3, n = 500


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n500 data.data_params.train.k=3 data.data_params.train.n=500 data.data_params.val.k=3 data.data_params.val.n=500 data.data_params.test.k=3 data.data_params.test.n=500 model.input_size=500 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n500/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n500 data.data_params.train.k=3 data.data_params.train.n=500 data.data_params.val.k=3 data.data_params.val.n=500 data.data_params.test.k=3 data.data_params.test.n=500 model.input_size=500 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n500/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n500 data.data_params.train.k=3 data.data_params.train.n=500 data.data_params.val.k=3 data.data_params.val.n=500 data.data_params.test.k=3 data.data_params.test.n=500 model.input_size=500 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n500/3 --on=test --num_points=50000
```

## k = 3, n = 1000


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n1000 data.data_params.train.k=3 data.data_params.train.n=1000 data.data_params.val.k=3 data.data_params.val.n=1000 data.data_params.test.k=3 data.data_params.test.n=1000 model.input_size=1000 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n1000/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n1000 data.data_params.train.k=3 data.data_params.train.n=1000 data.data_params.val.k=3 data.data_params.val.n=1000 data.data_params.test.k=3 data.data_params.test.n=1000 model.input_size=1000 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n1000/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k3n1000 data.data_params.train.k=3 data.data_params.train.n=1000 data.data_params.val.k=3 data.data_params.val.n=1000 data.data_params.test.k=3 data.data_params.test.n=1000 model.input_size=1000 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k3n1000/3 --on=test --num_points=50000
```

## k = 5, n = 5


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n5 data.data_params.train.k=5 data.data_params.train.n=5 data.data_params.val.k=5 data.data_params.val.n=5 data.data_params.test.k=5 data.data_params.test.n=5 model.input_size=5 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n5/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n5 data.data_params.train.k=5 data.data_params.train.n=5 data.data_params.val.k=5 data.data_params.val.n=5 data.data_params.test.k=5 data.data_params.test.n=5 model.input_size=5 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n5/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n5 data.data_params.train.k=5 data.data_params.train.n=5 data.data_params.val.k=5 data.data_params.val.n=5 data.data_params.test.k=5 data.data_params.test.n=5 model.input_size=5 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n5/3 --on=test --num_points=50000
```

## k = 5, n = 10


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n10 data.data_params.train.k=5 data.data_params.train.n=10 data.data_params.val.k=5 data.data_params.val.n=10 data.data_params.test.k=5 data.data_params.test.n=10 model.input_size=10 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n10/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n10 data.data_params.train.k=5 data.data_params.train.n=10 data.data_params.val.k=5 data.data_params.val.n=10 data.data_params.test.k=5 data.data_params.test.n=10 model.input_size=10 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n10/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n10 data.data_params.train.k=5 data.data_params.train.n=10 data.data_params.val.k=5 data.data_params.val.n=10 data.data_params.test.k=5 data.data_params.test.n=10 model.input_size=10 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n10/3 --on=test --num_points=50000
```

## k = 5, n = 50


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n50 data.data_params.train.k=5 data.data_params.train.n=50 data.data_params.val.k=5 data.data_params.val.n=50 data.data_params.test.k=5 data.data_params.test.n=50 model.input_size=50 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n50/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n50 data.data_params.train.k=5 data.data_params.train.n=50 data.data_params.val.k=5 data.data_params.val.n=50 data.data_params.test.k=5 data.data_params.test.n=50 model.input_size=50 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n50/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n50 data.data_params.train.k=5 data.data_params.train.n=50 data.data_params.val.k=5 data.data_params.val.n=50 data.data_params.test.k=5 data.data_params.test.n=50 model.input_size=50 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n50/3 --on=test --num_points=50000
```

## k = 5, n = 100


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n100 data.data_params.train.k=5 data.data_params.train.n=100 data.data_params.val.k=5 data.data_params.val.n=100 data.data_params.test.k=5 data.data_params.test.n=100 model.input_size=100 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n100/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n100 data.data_params.train.k=5 data.data_params.train.n=100 data.data_params.val.k=5 data.data_params.val.n=100 data.data_params.test.k=5 data.data_params.test.n=100 model.input_size=100 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n100/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n100 data.data_params.train.k=5 data.data_params.train.n=100 data.data_params.val.k=5 data.data_params.val.n=100 data.data_params.test.k=5 data.data_params.test.n=100 model.input_size=100 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n100/3 --on=test --num_points=50000
```

- - -

## k = 5, n = 500


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n500 data.data_params.train.k=5 data.data_params.train.n=500 data.data_params.val.k=5 data.data_params.val.n=500 data.data_params.test.k=5 data.data_params.test.n=500 model.input_size=500 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n500/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n500 data.data_params.train.k=5 data.data_params.train.n=500 data.data_params.val.k=5 data.data_params.val.n=500 data.data_params.test.k=5 data.data_params.test.n=500 model.input_size=500 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n500/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n500 data.data_params.train.k=5 data.data_params.train.n=500 data.data_params.val.k=5 data.data_params.val.n=500 data.data_params.test.k=5 data.data_params.test.n=500 model.input_size=500 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n500/3 --on=test --num_points=50000
```

## k = 5, n = 1000


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n1000 data.data_params.train.k=5 data.data_params.train.n=1000 data.data_params.val.k=5 data.data_params.val.n=1000 data.data_params.test.k=5 data.data_params.test.n=1000 model.input_size=1000 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n1000/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n1000 data.data_params.train.k=5 data.data_params.train.n=1000 data.data_params.val.k=5 data.data_params.val.n=1000 data.data_params.test.k=5 data.data_params.test.n=1000 model.input_size=1000 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n1000/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k5n1000 data.data_params.train.k=5 data.data_params.train.n=1000 data.data_params.val.k=5 data.data_params.val.n=1000 data.data_params.test.k=5 data.data_params.test.n=1000 model.input_size=1000 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k5n1000/3 --on=test --num_points=50000
```

## k = 10, n = 10


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n10 data.data_params.train.k=10 data.data_params.train.n=10 data.data_params.val.k=10 data.data_params.val.n=10 data.data_params.test.k=10 data.data_params.test.n=10 model.input_size=10 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n10/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n10 data.data_params.train.k=10 data.data_params.train.n=10 data.data_params.val.k=10 data.data_params.val.n=10 data.data_params.test.k=10 data.data_params.test.n=10 model.input_size=10 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n10/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n10 data.data_params.train.k=10 data.data_params.train.n=10 data.data_params.val.k=10 data.data_params.val.n=10 data.data_params.test.k=10 data.data_params.test.n=10 model.input_size=10 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n10/3 --on=test --num_points=50000
```

## k = 10, n = 50


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n50 data.data_params.train.k=10 data.data_params.train.n=50 data.data_params.val.k=10 data.data_params.val.n=50 data.data_params.test.k=10 data.data_params.test.n=50 model.input_size=50 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n50/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n50 data.data_params.train.k=10 data.data_params.train.n=50 data.data_params.val.k=10 data.data_params.val.n=50 data.data_params.test.k=10 data.data_params.test.n=50 model.input_size=50 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n50/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n50 data.data_params.train.k=10 data.data_params.train.n=50 data.data_params.val.k=10 data.data_params.val.n=50 data.data_params.test.k=10 data.data_params.test.n=50 model.input_size=50 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n50/3 --on=test --num_points=50000
```

## k = 10, n = 100


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n100 data.data_params.train.k=10 data.data_params.train.n=100 data.data_params.val.k=10 data.data_params.val.n=100 data.data_params.test.k=10 data.data_params.test.n=100 model.input_size=100 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n100/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n100 data.data_params.train.k=10 data.data_params.train.n=100 data.data_params.val.k=10 data.data_params.val.n=100 data.data_params.test.k=10 data.data_params.test.n=100 model.input_size=100 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n100/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n100 data.data_params.train.k=10 data.data_params.train.n=100 data.data_params.val.k=10 data.data_params.val.n=100 data.data_params.test.k=10 data.data_params.test.n=100 model.input_size=100 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n100/3 --on=test --num_points=50000
```

- - -

## k = 10, n = 500


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n500 data.data_params.train.k=10 data.data_params.train.n=500 data.data_params.val.k=10 data.data_params.val.n=500 data.data_params.test.k=10 data.data_params.test.n=500 model.input_size=500 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n500/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n500 data.data_params.train.k=10 data.data_params.train.n=500 data.data_params.val.k=10 data.data_params.val.n=500 data.data_params.test.k=10 data.data_params.test.n=500 model.input_size=500 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n500/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n500 data.data_params.train.k=10 data.data_params.train.n=500 data.data_params.val.k=10 data.data_params.val.n=500 data.data_params.test.k=10 data.data_params.test.n=500 model.input_size=500 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n500/3 --on=test --num_points=50000
```

## k = 10, n = 1000


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n1000 data.data_params.train.k=10 data.data_params.train.n=1000 data.data_params.val.k=10 data.data_params.val.n=1000 data.data_params.test.k=10 data.data_params.test.n=1000 model.input_size=1000 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n1000/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n1000 data.data_params.train.k=10 data.data_params.train.n=1000 data.data_params.val.k=10 data.data_params.val.n=1000 data.data_params.test.k=10 data.data_params.test.n=1000 model.input_size=1000 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n1000/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k10n1000 data.data_params.train.k=10 data.data_params.train.n=1000 data.data_params.val.k=10 data.data_params.val.n=1000 data.data_params.test.k=10 data.data_params.test.n=1000 model.input_size=1000 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k10n1000/3 --on=test --num_points=50000
```

## k = 50, n = 50


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k50n50 data.data_params.train.k=50 data.data_params.train.n=50 data.data_params.val.k=50 data.data_params.val.n=50 data.data_params.test.k=50 data.data_params.test.n=50 model.input_size=50 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k50n50/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k50n50 data.data_params.train.k=50 data.data_params.train.n=50 data.data_params.val.k=50 data.data_params.val.n=50 data.data_params.test.k=50 data.data_params.test.n=50 model.input_size=50 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k50n50/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k50n50 data.data_params.train.k=50 data.data_params.train.n=50 data.data_params.val.k=50 data.data_params.val.n=50 data.data_params.test.k=50 data.data_params.test.n=50 model.input_size=50 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k50n50/3 --on=test --num_points=50000
```

## k = 50, n = 100


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k50n100 data.data_params.train.k=50 data.data_params.train.n=100 data.data_params.val.k=50 data.data_params.val.n=100 data.data_params.test.k=50 data.data_params.test.n=100 model.input_size=100 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k50n100/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k50n100 data.data_params.train.k=50 data.data_params.train.n=100 data.data_params.val.k=50 data.data_params.val.n=100 data.data_params.test.k=50 data.data_params.test.n=100 model.input_size=100 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k50n100/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k50n100 data.data_params.train.k=50 data.data_params.train.n=100 data.data_params.val.k=50 data.data_params.val.n=100 data.data_params.test.k=50 data.data_params.test.n=100 model.input_size=100 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k50n100/3 --on=test --num_points=50000
```

## k = 50, n = 500


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k50n500 data.data_params.train.k=50 data.data_params.train.n=500 data.data_params.val.k=50 data.data_params.val.n=500 data.data_params.test.k=50 data.data_params.test.n=500 model.input_size=500 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k50n500/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k50n500 data.data_params.train.k=50 data.data_params.train.n=500 data.data_params.val.k=50 data.data_params.val.n=500 data.data_params.test.k=50 data.data_params.test.n=500 model.input_size=500 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k50n500/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k50n500 data.data_params.train.k=50 data.data_params.train.n=500 data.data_params.val.k=50 data.data_params.val.n=500 data.data_params.test.k=50 data.data_params.test.n=500 model.input_size=500 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k50n500/3 --on=test --num_points=50000
```

## k = 50, n = 1000


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k50n1000 data.data_params.train.k=50 data.data_params.train.n=1000 data.data_params.val.k=50 data.data_params.val.n=1000 data.data_params.test.k=50 data.data_params.test.n=1000 model.input_size=1000 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k50n1000/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k50n1000 data.data_params.train.k=50 data.data_params.train.n=1000 data.data_params.val.k=50 data.data_params.val.n=1000 data.data_params.test.k=50 data.data_params.test.n=1000 model.input_size=1000 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k50n1000/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k50n1000 data.data_params.train.k=50 data.data_params.train.n=1000 data.data_params.val.k=50 data.data_params.val.n=1000 data.data_params.test.k=50 data.data_params.test.n=1000 model.input_size=1000 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k50n1000/3 --on=test --num_points=50000
```

## k = 100, n = 100


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k100n100 data.data_params.train.k=100 data.data_params.train.n=100 data.data_params.val.k=100 data.data_params.val.n=100 data.data_params.test.k=100 data.data_params.test.n=100 model.input_size=100 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k100n100/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k100n100 data.data_params.train.k=100 data.data_params.train.n=100 data.data_params.val.k=100 data.data_params.val.n=100 data.data_params.test.k=100 data.data_params.test.n=100 model.input_size=100 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k100n100/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k100n100 data.data_params.train.k=100 data.data_params.train.n=100 data.data_params.val.k=100 data.data_params.val.n=100 data.data_params.test.k=100 data.data_params.test.n=100 model.input_size=100 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k100n100/3 --on=test --num_points=50000
```

## k = 100, n = 500


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k100n500 data.data_params.train.k=100 data.data_params.train.n=500 data.data_params.val.k=100 data.data_params.val.n=500 data.data_params.test.k=100 data.data_params.test.n=500 model.input_size=500 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k100n500/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k100n500 data.data_params.train.k=100 data.data_params.train.n=500 data.data_params.val.k=100 data.data_params.val.n=500 data.data_params.test.k=100 data.data_params.test.n=500 model.input_size=500 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k100n500/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k100n500 data.data_params.train.k=100 data.data_params.train.n=500 data.data_params.val.k=100 data.data_params.val.n=500 data.data_params.test.k=100 data.data_params.test.n=500 model.input_size=500 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k100n500/3 --on=test --num_points=50000
```

## k = 100, n = 1000


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k100n1000 data.data_params.train.k=100 data.data_params.train.n=1000 data.data_params.val.k=100 data.data_params.val.n=1000 data.data_params.test.k=100 data.data_params.test.n=1000 model.input_size=1000 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k100n1000/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k100n1000 data.data_params.train.k=100 data.data_params.train.n=1000 data.data_params.val.k=100 data.data_params.val.n=1000 data.data_params.test.k=100 data.data_params.test.n=1000 model.input_size=1000 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k100n1000/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k100n1000 data.data_params.train.k=100 data.data_params.train.n=1000 data.data_params.val.k=100 data.data_params.val.n=1000 data.data_params.test.k=100 data.data_params.test.n=1000 model.input_size=1000 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k100n1000/3 --on=test --num_points=50000
```

- - -

## k = 500, n = 500


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k500n500 data.data_params.train.k=500 data.data_params.train.n=500 data.data_params.val.k=500 data.data_params.val.n=500 data.data_params.test.k=500 data.data_params.test.n=500 model.input_size=500 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k500n500/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k500n500 data.data_params.train.k=500 data.data_params.train.n=500 data.data_params.val.k=500 data.data_params.val.n=500 data.data_params.test.k=500 data.data_params.test.n=500 model.input_size=500 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k500n500/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k500n500 data.data_params.train.k=500 data.data_params.train.n=500 data.data_params.val.k=500 data.data_params.val.n=500 data.data_params.test.k=500 data.data_params.test.n=500 model.input_size=500 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k500n500/3 --on=test --num_points=50000
```

## k = 500, n = 1000


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k500n1000 data.data_params.train.k=500 data.data_params.train.n=1000 data.data_params.val.k=500 data.data_params.val.n=1000 data.data_params.test.k=500 data.data_params.test.n=1000 model.input_size=1000 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k500n1000/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k500n1000 data.data_params.train.k=500 data.data_params.train.n=1000 data.data_params.val.k=500 data.data_params.val.n=1000 data.data_params.test.k=500 data.data_params.test.n=1000 model.input_size=1000 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k500n1000/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k500n1000 data.data_params.train.k=500 data.data_params.train.n=1000 data.data_params.val.k=500 data.data_params.val.n=1000 data.data_params.test.k=500 data.data_params.test.n=1000 model.input_size=1000 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k500n1000/3 --on=test --num_points=50000
```

## k = 1000, n = 1000


### Command to train Distance-learner

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k1000n1000 data.data_params.train.k=1000 data.data_params.train.n=1000 data.data_params.val.k=1000 data.data_params.val.n=1000 data.data_params.test.k=1000 data.data_params.test.n=1000 model.input_size=1000 data.generate=True task=regression

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k1000n1000/1 --on=test --num_points=50000
```

### Command to train Standard Classifier

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k1000n1000 data.data_params.train.k=1000 data.data_params.train.n=1000 data.data_params.val.k=1000 data.data_params.val.n=1000 data.data_params.test.k=1000 data.data_params.test.n=1000 model.input_size=1000 task=clf train_on_onmfld=True

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k1000n1000/2 --on=test --num_points=50000
```

### Command to train Standard Classifier with Off-manifold label

```bash
python3 learn_cls_from_dist.py with cuda=0 data.logdir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/ data.mtype=ittw-swissrolls data.data_tag=rdm_swrolls_k1000n1000 data.data_params.train.k=1000 data.data_params.train.n=1000 data.data_params.val.k=1000 data.data_params.val.n=1000 data.data_params.test.k=1000 data.data_params.test.n=1000 model.input_size=1000 model.output_size=3 task=clf train_on_onmfld=False

python3 analysis.py --dump_dir=/azuredrive/deepimage/data1/t-achetan/adv_geom_dumps/dumps/expC_dist_learner_for_adv_ex/rf_expts/rdm_swrolls/rdm_swrolls_k1000n1000/3 --on=test --num_points=50000
```
