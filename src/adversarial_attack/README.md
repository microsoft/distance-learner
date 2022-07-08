# `adversarial_attack`

This directory contains scripts to generated adversarial examples for models and evaluate them against generated examples.

- `attack_ingredients.py`: used for providing settings for attacks against which we want to evaluate the models
- `inp_fn_ingredients.py`: used for providing locations of `pipeline` runs that we want to evaluate against adversarial attacks
- `attacks.py`: code for various attacks that models can be evaluated on
- `get_attack_perf.py`: end-to-end script that accepts directories containing `pipeline` runs, attacks them using adversarial attacks, and evaluates the performance of the models on these attacks


## Running `adversarial_attack` with custom settings

1. First go to `attack_ingredients.py`. 
    - You can set magnitude of attack perturbations you want to evaluate on by editing `eps` in `attack_cfg` function. Note that you need to provide an **list** of values.
    - Similarly you can set other parameters. These include:
        - `eps_iter`: step size used during adversarial attacks
        - `nb_iter`: number of iterations for PGD-based attacks
        - `norm`: norm to use for PGD attack
        - `atk_routine`: whether to use our routine or to use the attack implemented in [`cleverhans`](https://github.com/cleverhans-lab/cleverhans)
    - All settings in this file have to be in a **list**. 
    - Specifying from command-line in this case is possible but can be cumbersome especially when using `numpy` based functions to create the list or using `numpy.inf` in the `norm` list for example.
    - If you absolutely have to specify from command line, an example of how to do it is: `with attack.atk_routine="['my']"` when you want to specify `atk_routine`

2. Next, go to `inpfn_ingredients.py`:
    - You can specify the directory where the script should look for the run directory to be processed
    - `settings_type` can be edited if you want to provide file names as a `"list"` or `"dict"`. The `list` option was included in case you want to provide edits from command-line.
    - edit `settings_fn` if you want to provide edits from a JSON file. The format of the JSON file should be similar to the dictionary format given in the script.

3. Lastly, go to `get_attack_perf.py`:
    - Most of the parameters in this file are self-explanatory.
    - `dump_dir` is the directory where the adversarial attack runs and the compiled JSON file with all the adversarial attack results will be stored.
    - You might need to edit `true_cls_batch_attr_name` and `true_cls_attr_name`:
        - `true_cls_attr_name`: the name of the class attribute in the dataset class that contains the class labels (usually this is `"class_labels"` and only for real-world datasets it is `"class_idx"`)
        - `true_cls_batch_attr_name`: the name of the key in the batch dictionary returned by the dataset iterator that stores the class labels. Look at the `__getitem___` method of the dataset class you are using and see the key in the dictionary that stores the `true_cls_attr_name` attribute. (again, usually this is `"class_labels"` and only for real-world datasets it is `"class_idx"`)
