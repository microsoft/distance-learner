import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import json
import time
import copy
import random
import argparse

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets

import PIL

import matplotlib
from mpl_toolkits import mplot3d
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

from livelossplot import PlotLosses

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import IncrementalPCA

from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, classification_report

from tqdm import tqdm


from expB.spheres_v2 import RandomSphere, TwoRandomSpheres
from expB.ptcifar.models import ResNet18
from expB.myNNs import *
from expB.workspace import *

from datagen.synthetic.multiple.intertwinedswissrolls import IntertwinedSwissRolls

from utils import *


SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

logger = init_logger("blah")


def weighted_mse_loss(inp, target, weight):
        return (weight.reshape(-1) * (inp - target) ** 2).mean()

def masked_mse_loss_for_far_mfld(logits, targets):
    """
        mask loss caused by points that are far from the target mfld.
        in ground truth. 

        :param logits: logit tensor with dim (num_samples x num_classes)
        :type logits: torch.Tensor(num_samples x num_classes)
        :param targets: logit tensor with dim (num_samples x num_classes)
        :type targets: torch.Tensor(num_samples x num_classes)
    """
    if np.inf not in targets:
        gt_far_classes = torch.max(targets, axis=1)[1]
        mask = torch.zeros_like(targets, requires_grad=False)
        mask[torch.arange(0, mask.shape[0]), gt_far_classes] = 1
        mask = mask.to(logits.device)
        # targets[torch.arange(0, mask.shape[0]), gt_far_classes] = 0
        # logits[torch.arange(0, mask.shape[0]), gt_far_classes] = 0
        # print(targets, gt_far_classes, mask, mask.shape)
        # loss = ((logits - targets)**2).mean()
        loss = ((1 - mask) * ((logits - targets) ** 2)).sum() / (1 - mask).sum()
        # print(targets, (1 - mask) * ((logits - targets) ** 2))
        # print(loss)

    else:
        unmask = targets != np.inf
        unmask = unmask.to(logits.device)
        # print(unmask.device, logits.device, targets.device)
        loss = ((logits[unmask] - targets[unmask]) ** 2).sum() / unmask.sum()
    return loss


loss_funcs = {
    "std_mse": nn.MSELoss(),
    "weighted_mse": weighted_mse_loss,
    "masked_mse": masked_mse_loss_for_far_mfld,
    "cross_entropy": nn.CrossEntropyLoss()
}

model_type = {
    "mlp-vanilla": MLP,
    "mlp-norm": MLPwithNormalisation,
    "mt-mlp-norm": MTMLPwithNormalisation
}


def custom_histogram_adder(writer, model, current_epoch):
       
        # iterating through all parameters
        for name,params in model.named_parameters():
          
            writer.add_histogram(name,params,current_epoch)

def init():
    print("import successful")

def train(model, optimizer, loss_func, dataloaders, device, save_dir, scheduler,\
          feature_name="normed_points", target_name="normed_distances",\
          num_epochs=500, task="regression", name="MLP_512x4_in1000",\
          scheduler_params={"warmup": 10, "cooldown": 300}, specs_dict=None, debug=False, online=False):
    """
        Function to train the model. Also dumps the best model.
        
        Returns the best model and optimizers

    """
    
    # storing the start_lr
    start_lr = optimizer.param_groups[0]['lr']
    
    train_loss_matrix = torch.zeros((num_epochs, len(dataloaders["train"])))
    val_loss_matrix = torch.zeros((num_epochs, len(dataloaders["val"])))
    

    # save_dir = os.path.join(save_dir, name)
    model_dir = os.path.join(save_dir, "models")
    plot_dir = os.path.join(save_dir, "tensorboard")
    specs_dump_fn = os.path.join(save_dir, "specs.json")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    with open(specs_dump_fn, "w+") as f:
        json.dump(specs_dict, f)
    
    plot_fn = os.path.join(plot_dir, name)

    writer = SummaryWriter(plot_fn)
    # liveloss = PlotLosses(fig_path=plot_fn)
    
    def plotDataIn3D(points_n):
        
        if points_n.shape[1] == 2:
            plt.figure(figsize = (10,7))
            fig_plot = plt.scatter(points_n[:, 0], points_n[:, 1], s=0.1).get_figure()
            plt.close(fig_plot)
        else:
            fig = plt.figure(figsize = (10, 7))
            ax1 = plt.axes(projection ="3d")
            fig_plot = ax1.scatter3D(points_n[:, 0], points_n[:, 1], points_n[:, 2], s=0.1).get_figure()
            plt.close(fig_plot)
        return fig_plot

    if hasattr(dataloaders["train"].dataset, "genattrs") and hasattr(dataloaders["train"].dataset.genattrs, "distances"):
        writer.add_histogram("train/distances", dataloaders["train"].dataset.genattrs.distances)
        writer.add_histogram("val/distances", dataloaders["val"].dataset.genattrs.distances)

        # plt.figure(figsize = (10,7))
        # fig_train = plt.scatter(dataloaders["train"].dataset.points_n[:, 0], dataloaders["train"].dataset.points_n[:, 1], s=0.01).get_figure()
        # plt.close(fig_train)
        fig_train = plotDataIn3D(dataloaders["train"].dataset.genattrs.points_n)
        writer.add_figure("train/points_n", fig_train)
        writer.add_text("train/data/params", str(vars(dataloaders["train"].dataset.genattrs)))

        # plt.figure(figsize = (10,7))
        # fig_val = plt.scatter(dataloaders["val"].dataset.points_n[:, 0], dataloaders["val"].dataset.points_n[:, 1], s=0.01).get_figure()
        # plt.close(fig_val)
        fig_val = plotDataIn3D(dataloaders["val"].dataset.genattrs.points_n)
        writer.add_figure("val/points_n", fig_val)
        writer.add_text("train/val/params", str(vars(dataloaders["val"].dataset.genattrs)))

        writer.add_graph(model, dataloaders["train"].dataset.genattrs.points_n[:dataloaders["train"].batch_size])

    elif hasattr(dataloaders["train"].dataset, "all_distances"):
        
        for phase in ["train", "val"]:

            writer.add_histogram(phase + "/S1/actual_distances", dataloaders[phase].dataset.all_actual_distances[:, 0])
            writer.add_histogram(phase + "/S2/actual_distances", dataloaders[phase].dataset.all_actual_distances[:, 1])

            writer.add_histogram(phase + "/S1/normed_distances", dataloaders[phase].dataset.normed_all_distances[:, 0])
            writer.add_histogram(phase + "/S2/normed_distances", dataloaders[phase].dataset.normed_all_distances[:, 1])

            writer.add_histogram(phase + "/S1/normed_actual_distances", dataloaders[phase].dataset.normed_all_actual_distances[:, 0])
            writer.add_histogram(phase + "/S2/normed_actual_distances", dataloaders[phase].dataset.normed_all_actual_distances[:, 1])

            writer.add_histogram(phase + "/S1/normed_actual_distances", dataloaders[phase].dataset.normed_all_actual_distances[:, 0])
            writer.add_histogram(phase + "/S2/normed_actual_distances", dataloaders[phase].dataset.normed_all_actual_distances[:, 1])

        
        capture_attrs =  ["S1_config", "S2_config", "n", "seed"]

        for phase in ["train", "val"]:
            if hasattr(dataloaders[phase].dataset, "all_points"):
                fig = plotDataIn3D(dataloaders[phase].dataset.all_points)
                writer.add_figure(phase + "/all_points", fig)
            data_vars = vars(dataloaders[phase].dataset)
            writer.add_text(phase + "/data/params", str({i: data_vars[i] for i in data_vars if i in capture_attrs}))
        

        writer.add_graph(model, dataloaders["train"].dataset.normed_all_points[:dataloaders["train"].batch_size])


    else:
        writer.add_histogram("train/distances", dataloaders["train"].dataset.tensors[1])
        writer.add_histogram("val/distances", dataloaders["val"].dataset.tensors[1])

        # plt.figure(figsize = (10,7))
        # fig_train = plt.scatter(dataloaders["train"].dataset.tensors[0][:, 0], dataloaders["train"].dataset.tensors[0][:, 1], s=0.01).get_figure()
        # plt.close(fig_train)
        fig_train = plotDataIn3D(dataloaders["train"].dataset.tensors[0])
        writer.add_figure("train/points_n", fig_train)

        # plt.figure(figsize = (10,7))
        # fig_val = plt.scatter(dataloaders["val"].dataset.tensors[0][:, 0], dataloaders["val"].dataset.tensors[0][:, 1], s=0.01).get_figure()
        # plt.close(fig_val)
        fig_val = plotDataIn3D(dataloaders["val"].dataset.tensors[0])
        writer.add_figure("val/points_n", fig_val)

        writer.add_graph(model, dataloaders["train"].dataset.tensors[0][:dataloaders["train"].batch_size])    

    phase = "train"
    
    last_best_epoch_loss = None
    
    losses = {"train_losses": [], "val_losses": []}

    epoch_wise_logits = None
    epoch_wise_targets = None

    if debug:
        epoch_wise_logits = torch.zeros(num_epochs, dataloaders["val"].dataset.normed_all_actual_distances.shape[0], dataloaders["val"].dataset.normed_all_actual_distances.shape[1])
        epoch_wise_targets = torch.zeros(num_epochs, dataloaders["val"].dataset.normed_all_actual_distances.shape[0], dataloaders["val"].dataset.normed_all_actual_distances.shape[1])
    
    for epoch in tqdm(range(num_epochs)):
        
        logs = {
            "loss": 0,
            "val_loss": 0,
            "lr": start_lr
        }

        for phase in ["train", "val"]:
            
            all_logits = None
            all_targets = None

            dl = dataloaders[phase]

            if online and (phase == "train") and (epoch > 0):
                dl.dataset.resample_points((10 * epoch) + (100 * num_epochs), no_op=True)
                
                # # for sanity check: ensuring resampling in a way to generate same samples everytime
                # seed_everything(dl.dataset.seed)
                # logger.info("seed set={}".format(dl.dataset.seed))
                # # dummy translation
                # np.random.normal(dl.dataset.mu, dl.dataset.sigma, dl.dataset.n)
                # # dummy rotation
                # tmp = np.random.normal(dl.dataset.mu, dl.dataset.sigma, size=(dl.dataset.n, dl.dataset.n))
                # tmp = np.linalg.qr(tmp)[0]
                # # dummy center
                # np.random.normal(dl.dataset.mu, dl.dataset.sigma, dl.dataset.k)
                # dl.dataset.resample_points(None)

            pred_classes = None
            target_classes = None
            
            if task == "clf":
                N = dl.dataset.N if hasattr(dl.dataset, "N") else dl.dataset.genattrs.N
                pred_classes = torch.zeros(N)
                target_classes = torch.zeros(N)
            
            if phase == "train":
                torch.set_grad_enabled(True)
                model.train()
            else:
                torch.set_grad_enabled(False)
                model.eval()
            
            num_batches = 0
            
            prefix = ""
            if phase == "val":
                prefix = "val_"
            
            for (i, batch) in enumerate(dataloaders[phase]):
                
                def get_data(batch_dict, feature_name=feature_name, target_name=target_name):
                    return batch_dict[feature_name], batch_dict[target_name]
                
                points, targets = None, None
                if type(batch) == dict:
                    points, targets = get_data(batch, feature_name=feature_name, target_name=target_name)
                else:
                    points, targets = batch[0], batch[1]

                points = points.to(device)
                targets = targets.to(device)
                # classes = batch[2].to(device)
                # print("where is input?", points.device)
                

                model = model.to(device)
                # print("where is model?", next(model.parameters()).is_cuda)
                logits = None
                if phase == "val":
                    with torch.no_grad():
                        logits = model(points)
                elif phase == "train":
                    logits = model(points)
                    # print("where are logits?", logits.device)
                # print(points.shape, distances.shape, logits.shape)

                # weights in case of weighted loss
                # weights = distances.clone().detach()
                # weights[weights == 0] = 1 / torch.max(weights[weights != 0])
                # weights[weights != 0] = 1 / weights[weights != 0]

                # loss = loss_func(logits, distances, weights)

                loss = loss_func(logits, targets)
                
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                logs[prefix + "loss"] += loss.detach().cpu().item()

                if phase == "train":
                    train_loss_matrix[epoch, i] = loss.detach().cpu().item()
                elif phase == "val":
                    val_loss_matrix[epoch, i] = loss.detach().cpu().item()
                
                if all_logits is None:
                    all_logits = torch.zeros(len(dataloaders[phase].dataset), logits.shape[1])
                all_logits[i*points.shape[0]:(i+1)*points.shape[0]] = logits.detach().cpu()

                if all_targets is None:
                    if len(targets.shape) == 1:
                        all_targets = torch.zeros(len(dataloaders[phase].dataset))
                    else:
                        all_targets = torch.zeros(len(dataloaders[phase].dataset), targets.shape[1])
                all_targets[i*points.shape[0]:(i+1)*points.shape[0]] = targets.detach().cpu()

                num_batches += 1
                
                points = points.cpu()
                targets = targets.cpu()
                model = model.cpu()
                logits = logits.cpu()
                
                if task == "clf":
                    pred_classes[i*points.shape[0]:(i+1)*points.shape[0]] = torch.max(logits, axis=1)[1]
                    target_classes[i*points.shape[0]:(i+1)*points.shape[0]] = targets
            
            if task == "clf":
                average = "binary" if model.output_size == 2 else "macro"
                f1 = f1_score(target_classes, pred_classes, average=average)
                acc = accuracy_score(target_classes, pred_classes)

                logs[prefix + "f1"] = f1
                logs[prefix + "acc"] = acc

                writer.add_scalar(phase + "/f1", logs[prefix + "f1"], epoch)
                writer.add_scalar(phase + "/acc", logs[prefix + "acc"], epoch)

            
            
            # dividing by the number of batches
            logs[prefix + "loss"] /= num_batches
            writer.add_scalar(phase + "/loss", logs[prefix + "loss"], epoch)
            if phase == "train":
                custom_histogram_adder(writer, model, epoch)
            
            if phase == "val" and debug:
                epoch_wise_logits[epoch] = all_logits
                epoch_wise_targets[epoch] = all_targets


            if all_logits.shape[1] == 1:
                writer.add_histogram(phase + "/logits", all_logits.reshape(-1), epoch)
            else:
                for idx in range(all_logits.shape[1]):
                    writer.add_histogram(phase + "/S" + str(idx+1) + "/logits", all_logits[:, idx].reshape(-1), epoch)
            
            if epoch == num_epochs - 1 and task == "regression":
                for idx in range(all_logits.shape[1]):
                    mask = np.abs(all_targets.numpy()[:, idx] - all_logits.numpy()[:, idx]) >= 5e-2
                    plt.scatter(all_targets.numpy()[mask, idx], all_logits.numpy()[mask, idx], s=0.01, c="red")
                    plt.scatter(all_targets.numpy()[np.logical_not(mask), idx], all_logits.numpy()[np.logical_not(mask), idx], s=0.01, c="green")
                    plt.xlabel("gt distance ({})".format(target_name))
                    plt.ylabel("pred distance")
                    plt.title("gt vs. pred {}".format(target_name))
                    plt.savefig(os.path.join(save_dir, "pred_vs_gt_dists_S{}_{}.png".format(idx + 1, phase)))
                    plt.clf()

                mask = np.abs(all_targets.numpy().ravel() - all_logits.numpy().ravel()) >= 5e-2
                plt.scatter(all_targets.numpy().ravel()[mask], all_logits.numpy().ravel()[mask], s=0.01, c="red")
                plt.scatter(all_targets.numpy().ravel()[np.logical_not(mask)], all_logits.numpy().ravel()[np.logical_not(mask)], s=0.01, c="green")
                plt.xlabel("gt distance ({})".format(target_name))
                plt.ylabel("pred distance")
                plt.title("gt vs. pred {}".format(target_name))
                plt.savefig(os.path.join(save_dir, "pred_vs_gt_dists_all_{}.png".format(phase)))
                plt.clf()

        check = last_best_epoch_loss is None or logs["val_loss"] < last_best_epoch_loss or epoch == 100
        stat = "val_loss"
        if task == "clf":
            check = last_best_epoch_loss is None or logs["val_f1"] > last_best_epoch_loss
            stat = "val_f1"
            
        if check:
            last_best_epoch_loss = logs[stat]
            dump = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': logs['val_loss'],
                'loss': logs["loss"],
                'scheduler_params': scheduler_params,
                'scheduler_state_dict': scheduler.state_dict()
            }
            
            if task == "clf":
                dump["acc"] = logs["acc"]
                dump["val_acc"] = logs["val_acc"]
                dump["f1"] = logs["f1"]
                dump["val_f1"] = logs["val_f1"]

            if debug:   
                torch.save(dump, os.path.join(model_dir, name + "_val_loss_" + str(logs["val_loss"]) + "_epoch_" + str(epoch) + ".pth"))
            else:
                torch.save(dump, os.path.join(model_dir, "ckpt" + ".pth"))

        
        logs["lr"] = optimizer.param_groups[0]["lr"]

        writer.add_scalar("lr", logs["lr"], epoch)

        # liveloss.update(logs)
        # liveloss.draw()
        
        scheduler.step()
        writer.flush()

    writer.close()

    if debug:
        torch.save(epoch_wise_logits, os.path.join(save_dir, "epoch_wise_val_logits.pt"))
        torch.save(epoch_wise_targets, os.path.join(save_dir, "epoch_wise_val_targets.pt"))

    return model, optimizer, scheduler, train_loss_matrix, val_loss_matrix

                    
            

def test(model, dataloader, device, task="regression",\
     feature_name="normed_points", target_name="normed_distances",\
     save_dir=None, specs_dict=None, name="data", debug=False):
    
    model.eval()
    
    all_logits = None
    all_targets = None
    # all_true_distances = None
    # all_classes = None

    
    with torch.no_grad():
        
        for batch in dataloader:
            # print(len(batch))
            def get_data(batch_dict, feature_name=feature_name, target_name=target_name):
                return batch_dict[feature_name], batch_dict[target_name]
            
            points, targets = None, None
            if type(batch) == dict:
                points, targets = get_data(batch, feature_name=feature_name, target_name=target_name)
            else:
                points, targets = batch[0], batch[1]

            points = points.to(device)
            targets = targets.to(device)

            # points = batch[0].to(device)
            # distances = batch[1].to(device)
            # classes = batch[2].to(device)
            
            model.zero_grad()
            model = model.to(device)

            logits = model(points).detach().cpu()

            points = points.cpu()
            targets = targets.cpu()
            model = model.cpu()
            
            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits))
            
            if all_targets is None:
                all_targets = targets
            else:
                all_targets = torch.cat((all_targets, targets))

            # if task == "classification":
            #     if all_classes is None:
            #         all_classes = classes
            #     else:
            #         all_classes = torch.cat((all_classes, classes))
            
    if debug: print("name:", name)

    targets_fn, logits_fn, specs_fn = None, None, None

    if save_dir is not None:
        
        # TIME_STAMP = time.strftime("%d%m%Y-%H%M%S")
        save_dir = os.path.join(save_dir, "logits", name)

        os.makedirs(save_dir, exist_ok=True)

        targets_fn = os.path.join(save_dir, "targets.pt")
        logits_fn = os.path.join(save_dir, "logits.pt")
        specs_fn = os.path.join(save_dir, "specs.json")

        
        with open(specs_fn, "w+") as f:
            json.dump(specs_dict, f)

    
    # print(targets_fn)
    if task == "regression":
        masked_targets = all_targets.clone().detach()
        masked_targets[all_targets == np.inf] = 0
        masked_logits = all_logits.clone().detach()
        masked_logits[all_targets == np.inf] = 0
        mse = mean_squared_error(masked_targets, masked_logits)
        mse_on_mfld = mean_squared_error(masked_targets[np.round(all_targets) == 0], masked_logits[np.round(all_targets) == 0])

        if save_dir is not None:
            torch.save(all_targets, targets_fn)
            torch.save(all_logits, logits_fn)


        if debug:
            print("MSE for the learned distances:", mse)
            print("MSE for the learned distances (on-manifold):", mse_on_mfld)
        return mse, mse_on_mfld, all_targets, all_logits
    
    elif task == "clf":
        y_pred = torch.max(all_logits, axis=1)[1]
        if debug:
            print(classification_report(all_targets.reshape(-1), y_pred))
        
        acc = accuracy_score(all_targets.reshape(-1), y_pred)
        average = "binary" if model.output_size == 2 else "macro"
        f1 = f1_score(all_targets.reshape(-1), y_pred, average=average)
        
        if save_dir is not None:
            torch.save(all_targets, targets_fn)
            torch.save(all_logits, logits_fn)




        return acc, f1, all_targets, all_logits
    
    

    if debug: print("\n")

    





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="enable train mode", required=False)
    parser.add_argument("--test", action="store_true", help="compute and store predictions on all splits", required=False)


    parser.add_argument("--specs", type=str, help="specifications file for the experiment, removes need for all other options", default=None)

    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--cuda", help="use GPUs", default=None)    
    parser.add_argument("--task", type=str, help="'classification' or 'regression'", default="regression")
    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=500)
    parser.add_argument("--save_dir", type=str, help="save directory path")
    parser.add_argument("--name", type=str, help="name of experiment")
    parser.add_argument("--warmup", type=int, help="number of warmup steps", default=10)
    parser.add_argument("--cooldown", type=int, help="epoch after which to cooldown", default=300)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    # parser.add_argument("--init_wts", type=str, help="path to initial weights", default="/azuredrive/dumps/expB_learning_distance_from_mfld/init_model_weights.pt")
    parser.add_argument("--init_wts", type=str, help="path to initial weights", default=None)
    parser.add_argument("--num_classes", type=int, help="number of manifolds", default=1)
    parser.add_argument("--input_size", type=int, help="input size", default=2)
    
    parser.add_argument("--model_type", type=str, help="model type to use for training")
    parser.add_argument("--loss_func", type=str, help="loss function to use: 'std_mse', 'weighted_mse', and 'masked_mse'")


    parser.add_argument("--train_fn", type=str, help="path to train data") 
    parser.add_argument("--val_fn", type=str, help="path to val data") 
    parser.add_argument("--test_fn", type=str, help="path to test data")
    parser.add_argument("--data_fn", type=str, help="path to any other data")
    parser.add_argument("--ftname", type=str, help="named attribute for features when fetching dataset", default="normed_points") 
    parser.add_argument("--tgtname", type=str, help="named attribute for labels when fetching dataset", default="normed_distances") 

    args = parser.parse_args()


    TRAIN_FLAG = args.train
    TEST_FLAG = args.test
    BATCH_SIZE = args.batch_size
    CUDA = args.cuda
    TASK = args.task
    NUM_EPOCHS = args.num_epochs
    SAVE_DIR = args.save_dir
    NAME = args.name

    WARMUP = args.warmup
    COOLDOWN = args.cooldown
    LR = args.lr

    INIT_WTS = args.init_wts
    NUM_CLASSES = args.num_classes
    INPUT_SIZE = args.input_size

    TRAIN_FN = args.train_fn
    VAL_FN = args.val_fn
    TEST_FN = args.test_fn
    DATA_FN = args.data_fn

    FTNAME = args.ftname
    TGTNAME = args.tgtname

    MODEL_TYPE = args.model_type
    loss_func = args.loss_func

    specs_dict = {
        "batch_size": args.batch_size,
        "cuda": args.cuda,
        "task": args.task,
        "num_epochs": args.num_epochs,
        "save_dir": args.save_dir,
        "name": args.name,
        "warmup": args.warmup,
        "cooldown": args.cooldown,
        "lr": args.lr,
        "init_wts": args.init_wts,
        "num_classes": args.num_classes,
        "input_size": args.input_size,
        "train_fn": args.train_fn,
        "val_fn": args.val_fn,
        "test_fn": args.test_fn,
        "ftname": args.ftname,
        "tgtname": args.tgtname,
        "model_type": args.model_type,
        "loss_func": args.loss_func
    }

    # if specs file is provided, override all past values
    if args.specs is not None:

        specs_dict = json.load(open(args.specs))
        BATCH_SIZE, CUDA, TASK, NUM_EPOCHS, SAVE_DIR, NAME,\
            WARMUP, COOLDOWN, LR, INIT_WTS, NUM_CLASSES, INPUT_SIZE,\
            TRAIN_FN,VAL_FN, TEST_FN, FTNAME, TGTNAME, MODEL_TYPE, loss_func = load_specs(specs_dict)


    if TRAIN_FLAG:
        
        train_set = torch.load(TRAIN_FN)
        val_set = torch.load(VAL_FN)

        # train_set = IntertwinedSwissRolls()
        # train_set.load_data(TRAIN_FN)
        # val_set = IntertwinedSwissRolls()
        # val_set.load_data(VAL_FN)
        # train_set, val_set, test_set = IntertwinedSwissRolls.make_train_val_test_splits(save_dir=os.path.join(SAVE_DIR, "data"))
        
        # train_perm = torch.randperm(train_set.N)
        # val_perm = torch.randperm(val_set.N)

        # print("train_perm:", train_perm[:20])
        # print("val_perm:", val_perm[:20])

        # train_set.points_n = train_set.points_n[train_perm]
        # train_set.distances = train_set.distances[train_perm]

        # val_set.points_n = val_set.points_n[val_perm]
        # val_set.distances = val_set.distances[val_perm]

        # train_set.points_n = train_set.points_n
        # train_set.distances = train_set.distances

        # val_set.points_n = val_set.points_n
        # val_set.distances = val_set.distances

        NUM_WORKERS = 8

        # this is specific to the `TwoSpheres` case; done  
        # so that we only train on on-manifold samples
        if TASK == "clf":
            OFF_MFLD_LABEL = 2

            train_set.all_points = train_set.all_points[train_set.class_labels != OFF_MFLD_LABEL]
            train_set.all_distances = train_set.all_distances[train_set.class_labels != OFF_MFLD_LABEL]
            train_set.normed_all_points = train_set.normed_all_points[train_set.class_labels != OFF_MFLD_LABEL]
            train_set.normed_all_distances = train_set.normed_all_distances[train_set.class_labels != OFF_MFLD_LABEL]
            train_set.class_labels = train_set.class_labels[train_set.class_labels != OFF_MFLD_LABEL]

            val_set.all_points = val_set.all_points[val_set.class_labels != OFF_MFLD_LABEL]
            val_set.all_distances = val_set.all_distances[val_set.class_labels != OFF_MFLD_LABEL]
            val_set.normed_all_points = val_set.normed_all_points[val_set.class_labels != OFF_MFLD_LABEL]
            val_set.normed_all_distances = val_set.normed_all_distances[val_set.class_labels != OFF_MFLD_LABEL]
            val_set.class_labels = val_set.class_labels[val_set.class_labels != OFF_MFLD_LABEL]


        dataloaders = {
            "train": DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, worker_init_fn=seed_worker),
            "val": DataLoader(dataset=val_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
        }  

        device = torch.device("cuda:{}".format(CUDA) if torch.cuda.is_available() and CUDA else "cpu")

        model = model_type[MODEL_TYPE](input_size=INPUT_SIZE, output_size=NUM_CLASSES, hidden_sizes=[512] * 4, weight_norm=False, use_tanh=False, use_relu=False)
        # model = MLPwithNormalisation(input_size=INPUT_SIZE, output_size=NUM_CLASSES, hidden_sizes=[512] * 4, weight_norm=False, use_tanh=False, use_relu=False)
        # model = MLP(input_size=INPUT_SIZE, output_size=NUM_CLASSES, hidden_sizes=[512] * 4, use_tanh=False)
        # model = ResNet18(num_classes=args.num_classes)
        if INIT_WTS is not None:
            model.load_state_dict(torch.load(INIT_WTS))
        
        
        loss_func = loss_funcs[loss_func]
        # loss_func = weighted_mse_loss
        if TASK == "clf":
            loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

        NUM_EPOCHS = NUM_EPOCHS 
        scheduler_params = {"warmup": WARMUP, "cooldown": COOLDOWN}
        lr_sched_factor = lambda epoch: epoch / (scheduler_params["warmup"]) if epoch <= scheduler_params["warmup"] else (1 if epoch > scheduler_params["warmup"] and epoch < scheduler_params["cooldown"] else max(0, 1 + (1 / (scheduler_params["cooldown"] - NUM_EPOCHS)) * (epoch - scheduler_params["cooldown"])))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_sched_factor)




        model, optimizer, scheduler, train_loss_matrix, val_loss_matrix = train(model=model, optimizer=optimizer, loss_func=loss_func, dataloaders=dataloaders,\
                         device=device, task=TASK, num_epochs=NUM_EPOCHS, save_dir=SAVE_DIR, feature_name=FTNAME, target_name=TGTNAME,\
                         name=NAME, scheduler=scheduler, scheduler_params=scheduler_params, specs_dict=specs_dict)

    elif TEST_FLAG:

        
        
        phases = ["train", "val", "test"]
        data_fns = [TRAIN_FN, VAL_FN, TEST_FN]

        splits = {i[0]: {"fn": i[1], "name": i[0]} for i in zip(phases, data_fns)}
        


        for split in splits:
            dataset = torch.load(splits[split]["fn"])
            NUM_WORKERS = 8

            if TASK == "clf":
                OFF_MFLD_LABEL = 2

                dataset.all_points = dataset.all_points[dataset.class_labels != OFF_MFLD_LABEL]
                dataset.all_distances = dataset.all_distances[dataset.class_labels != OFF_MFLD_LABEL]
                dataset.normed_all_points = dataset.normed_all_points[dataset.class_labels != OFF_MFLD_LABEL]
                dataset.normed_all_distances = dataset.normed_all_distances[dataset.class_labels != OFF_MFLD_LABEL]
                dataset.class_labels = dataset.class_labels[dataset.class_labels != OFF_MFLD_LABEL]

            # shuffle not needed here. makes things easier if post-processing is needed
            dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)

            device = torch.device("cuda:{}".format(CUDA) if torch.cuda.is_available() and CUDA else "cpu")

            model = model_type[MODEL_TYPE](input_size=INPUT_SIZE, output_size=NUM_CLASSES, hidden_sizes=[512] * 4, weight_norm=False, use_tanh=False, use_relu=False)
            # model = MLPwithNormalisation(input_size=INPUT_SIZE, output_size=NUM_CLASSES, hidden_sizes=[512] * 4, weight_norm=False, use_tanh=False, use_relu=False)

            if INIT_WTS is None:
                raise RuntimeError("INIT_WTS needed for testing!")
            
            model.load_state_dict(torch.load(INIT_WTS)["model_state_dict"])

            _, _, all_targets, all_logits = test(model=model, dataloader=dataloader,\
                device=device, task=TASK, feature_name=FTNAME, target_name=TGTNAME, name=splits[split]["name"], save_dir=SAVE_DIR)


    else:

        dataset = torch.load(DATA_FN)
        NUM_WORKERS = 8

        dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)

        device = torch.device("cuda:{}".format(CUDA) if torch.cuda.is_available() and CUDA else "cpu")

        model = MLPwithNormalisation(input_size=INPUT_SIZE, output_size=NUM_CLASSES, hidden_sizes=[512] * 4, weight_norm=False, use_tanh=False, use_relu=False)

        if INIT_WTS is None:
            raise Exception("INIT_WTS needed for testing!")
        
        model.load_state_dict(torch.load(INIT_WTS)["model_state_dict"])

        _, _, all_targets, all_logits = test(model=model, dataloader=dataloader,\
             device=device, task=TASK, feature_name=FTNAME, target_name=TGTNAME, name=NAME)



