import json

def load_specs(specs_dict):
    """populate experiment variables with specifications dict"""

    BATCH_SIZE = specs_dict["batch_size"]
    CUDA = specs_dict["cuda"]
    TASK = specs_dict["task"]
    NUM_EPOCHS = specs_dict["num_epochs"]
    SAVE_DIR = specs_dict["save_dir"]
    NAME = specs_dict["name"]

    WARMUP = specs_dict["warmup"]
    COOLDOWN = specs_dict["cooldown"]
    LR = specs_dict["lr"]

    INIT_WTS = specs_dict["init_wts"]
    NUM_CLASSES = specs_dict["num_classes"]
    INPUT_SIZE = specs_dict["input_size"]

    TRAIN_FN = specs_dict["train_fn"]
    VAL_FN = specs_dict["val_fn"]
    TEST_FN = specs_dict["test_fn"]

    FTNAME = specs_dict["ftname"]
    TGTNAME = specs_dict["tgtname"]

    MODEL_TYPE = specs_dict["model_type"]
    loss_func = specs_dict["loss_func"]

    return BATCH_SIZE, CUDA, TASK, NUM_EPOCHS, SAVE_DIR,\
        NAME, WARMUP, COOLDOWN, LR, INIT_WTS, NUM_CLASSES, INPUT_SIZE, TRAIN_FN,\
            VAL_FN, TEST_FN, FTNAME, TGTNAME, MODEL_TYPE, loss_func