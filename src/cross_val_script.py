import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

import openslide

from utils import processing, model

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print(torch.cuda.is_available())


################################################################################
################################################################################

DATASET_FLAG = "TCGA"  # set either TCGA or CAMELYON

if DATASET_FLAG == "TCGA":
    ## Load TCGA data

    SLIDE_INFO_DIR = "/path/to/TCGA-RCC-labels.csv"  # CHANGE ACCORDINGLY
    FEATURE_DIR = "/path/to/tcga-rcc/clam/features/h5_files/"  # CHANGE ACCORDINGLY
    SUBTYPES = [
        "KICH",
        "KIRC",
        "KIRP",
    ]  # chromophobe, clear cell, papillary (TCGA-RCC subtypes)

    data, labels = processing.load_tcga_data(SUBTYPES, SLIDE_INFO_DIR)

    features, labels, coords, filenames = processing.load_features(
        data, labels, FEATURE_DIR, experiment="tcga"
    )

elif DATASET_FLAG == "CAMELYON":
    SLIDE_INFO_DIR = "/path/to/camelyon/"  # CHANGE ACCORDINGLY
    FEATURE_DIR = "/path/to/camelyon/clam/features/h5_files/"  # CHANGE ACCORDINGLY

    # load both camelyon datasets (16&17)
    data_16, coords_16, labels_16, filenames_16 = processing.load_camelyon_data(
        SLIDE_INFO_DIR, FEATURE_DIR, mode="16"
    )
    data_17, coords_17, labels_17, filenames_17 = processing.load_camelyon_data(
        SLIDE_INFO_DIR, FEATURE_DIR, mode="17"
    )

    # cat camleyon datasets
    features = data_16 + data_17
    coords = coords_16 + coords_17
    labels = labels_16 + labels_17
    filenames = filenames_16 + filenames_17

print(np.unique(labels, return_counts=True))
print(len(features))
print(np.mean([x.shape[0] for x in features]))
print(np.median([x.shape[0] for x in features]))
print(min([x.shape[0] for x in features]))
print(max([x.shape[0] for x in features]))


features = [processing.preprocess_features(i, pca=-1) for i in features]


prototypes, cluster_labels, kms = processing.cluster_dataset(
    features, k_clusters=5, method="kmeans"
)

# build dataset
domain_labels = [domain for domain in range(len(prototypes))]
dataset = []
for x, y, z, fn in zip(prototypes, labels, domain_labels, filenames):
    x = torch.from_numpy(x)
    dataset.append((x, y, z, fn))

## BAG dataset building (e.g. for standard ABMIL/CLAM...)
# domain_labels = [domain for domain in range(len(features))]
# dataset = []
# assert len(features) == len(labels)
# for x, y, z, fn in zip(features, labels, domain_labels, filenames):
#    x = torch.from_numpy(x)
#    dataset.append((x, y, z, fn))

################################################################################
################################################################################

# Define Model and run Cross-validation


model = model.ProtoMixer(
    token_dim=1024,
    num_tokens=8,
    hidden_dim=2048,
    num_layers=12,
    num_classes=2,
    dropout=0.5,
    pool="mean",
    domain_num=len(domain_labels),
)

# w/o domain adversarial
# model = model.ProtoMixer(token_dim=1024, num_tokens=5, hidden_dim=2048,
#  num_layers=8, num_classes=3, dropout=0.5, pool='mean,' domain_num=0)

# ABMIL model
# model = model.ABMIL(input_size=features[0].shape[1], num_classes=3)

## CLAM model
# model = model.CLAM_SB(
#    gate=True,
#    size_arg="small",
#    dropout=False,
#    k_sample=8,
#    n_classes=3,
#    instance_loss_fn=nn.CrossEntropyLoss(),
#    subtyping=True,
# )

# count parameters
print(
    "Number of trainable params: "
    + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
)


def reset_all_weights(model: nn.Module) -> None:
    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# train/eval loop
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    f1_score,
)


def eval_ans(y_hat, label):
    true_label = int(label)
    if y_hat == true_label:
        ans = 1
    if y_hat != true_label:
        ans = 0
    return ans


def train(model, device, loss_fn, optimizer, train_dl, DA_rate):
    model = model.to(device)
    model.train()  # train mode
    train_class_loss = 0.0
    train_domain_loss = 0.0
    correct_num = 0

    for i, data in enumerate(train_dl, 0):
        # loading data
        input_tensor, class_label, domain_label, fn = data
        # to device
        input_tensor = input_tensor.to(device)
        class_label = class_label.to(device)
        domain_label = domain_label.to(device)

        optimizer.zero_grad()  # initialize gradient
        class_prob, domain_prob, A = model(input_tensor, mode="train", DA_rate=DA_rate)
        # class_prob, domain_prob, A = model(input_tensor[torch.randperm(input_tensor.shape[0])], mode='train', DA_rate=DA_rate) # with randshuffletokens
        # calculate loss
        class_loss = loss_fn(class_prob, class_label)
        domain_loss = loss_fn(domain_prob, domain_label)
        total_loss = class_loss + domain_loss

        train_class_loss += class_loss.item()
        train_domain_loss += domain_loss.item()

        total_loss.backward()  # backpropagation
        optimizer.step()  # renew parameters
        class_hat = torch.argmax(F.softmax(class_prob, dim=1).squeeze(0), dim=0)
        correct_num += eval_ans(class_hat, class_label)

    train_class_loss = train_class_loss / len(train_dl.dataset)
    train_domain_loss = train_domain_loss / len(train_dl.dataset)
    train_acc = correct_num / len(train_dl.dataset)
    return train_class_loss, train_domain_loss, train_acc


def train_clam(model, device, loss_fn, optimizer, train_dl):
    model = model.to(device)
    model.train()  # train mode
    train_class_loss = 0.0
    train_domain_loss = 0
    correct_num = 0

    for i, data in enumerate(train_dl, 0):
        input_tensor, class_label, domain_label, fn = data
        input_tensor = input_tensor.to(device)
        class_label = class_label.to(device)
        domain_label = domain_label.to(device)
        optimizer.zero_grad()  # initialize gradient
        class_prob, A = model(input_tensor)
        # calculate loss
        class_loss = loss_fn(class_prob, class_label)

        train_class_loss += class_loss.item()
        class_loss.backward()  # backpropagation
        optimizer.step()  # renew parameters
        class_hat = torch.argmax(F.softmax(class_prob, dim=1).squeeze(0), dim=0)
        correct_num += eval_ans(class_hat, class_label)

    train_class_loss = train_class_loss / len(train_dl.dataset)
    train_acc = correct_num / len(train_dl.dataset)
    return train_class_loss, train_domain_loss, train_acc


def test(model, device, loss_fn, test_dl, output_file=None):
    model.eval()  # test mode
    correct_num = 0
    y_probs = []
    y_true = []
    y_preds = []
    for i, data in enumerate(test_dl):
        # load data
        input_tensor, class_label, domain_label, fn = data
        input_tensor = input_tensor.to(device)
        class_label = class_label.to(device)

        with torch.no_grad():
            # class_prob, A = model(input_tensor, mode="test", DA_rate=0)
            # class_prob, A = model(input_tensor[torch.randperm(input_tensor.shape[0])], mode='test', DA_rate=0) # with randshuffletokens
            class_prob, A = model(input_tensor)  # CLAM

        class_softmax = F.softmax(class_prob, dim=1).squeeze(0)
        class_hat = torch.argmax(class_softmax, dim=0)
        correct_num += eval_ans(class_hat, class_label)
        val_loss = loss_fn(class_prob, class_label)
        # write predicton results for bag and attention weights for each patches
        if output_file:
            f = open(output_file, "a")
            f_writer = csv.writer(f, lineterminator="\n")
            f_writer.writerow(
                [
                    fold,
                    epoch,
                    class_label.detach().cpu().numpy(),
                    class_hat.detach().cpu().numpy(),
                    class_softmax.detach().cpu().numpy(),
                    "/",
                    "/",
                    "/",
                ]
            )
            f.close()
        y_probs.append(class_softmax.detach().cpu().numpy())
        y_preds.append(class_hat.detach().cpu().numpy())
        y_true.append(class_label.detach().cpu().numpy())

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true=y_true, y_pred=y_preds)
    print(cm)

    print("Matthews Corr Coeff:")
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_preds)
    print(mcc)

    print("Patientwise AUROC is:")
    y_probs = np.stack(y_probs)
    if len(np.unique(y_true)) < 3:
        y_probs = y_probs[:, 1]
    try:
        auc = roc_auc_score(
            y_true=y_true,
            y_score=y_probs,
            multi_class="ovr",
        )
        print(str(auc))
    except:
        print("can't compute")

    print("F1 Score is:")
    f1 = f1_score(y_true, y_preds, average="macro")
    print(f1)

    print("Accuracy: " + str(correct_num / len(test_dl.dataset)))
    return correct_num / len(test_dl.dataset), val_loss.detach().cpu().numpy(), f1, auc


# RUN TRAINING


def run_training(model, epochs, train_dl, val_dl, fold, type):
    # device = "cuda:0" # only for training cost benchmarking
    device = "cuda"
    DA_rate = 0.001
    # DA_rate = 0 # for w/o domain adversarial training

    torch.backends.cudnn.benchmark = True  # cudnn benchmark mode

    reset_all_weights(model)

    loss_fn = nn.CrossEntropyLoss()
    if type == "Mixer":
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.0001,
            momentum=0.9,
            weight_decay=0.0,
            nesterov=False,
        )
    elif type == "CLAM":
        optimizer = optim.Adam(
            model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )

    best_acc = 0
    history = []
    global epoch

    for epoch in range(epochs):
        # calculate domain regularization parameter
        if DA_rate != 0:
            p = ((epoch + 1) / (epochs)) * DA_rate
            lamda = (2 / (1 + np.exp(-10 * p))) - 1
        else:
            lamda = 0  # w/o domain adversarial branch
        # training
        if type == "Mixer":
            class_loss, domain_loss, train_acc = train(
                model, device, loss_fn, optimizer, train_dl, lamda
            )
        elif type == "CLAM":
            class_loss, domain_loss, train_acc = train_clam(
                model, device, loss_fn, optimizer, train_dl
            )

        # validation run
        val_acc, val_loss, val_f1, val_auc = test(
            model, device, loss_fn, val_dl, val_log
        )

        # write log
        f = open(log, "a")
        f_writer = csv.writer(f, lineterminator="\n")
        f_writer.writerow(
            [epoch, class_loss, domain_loss, val_loss, train_acc, val_acc]
        )
        f.close()

        # save parameters
        is_best = bool(val_acc >= best_acc)
        best_acc = max(best_acc, val_acc)
        if is_best:
            torch.save(model.state_dict(), model_params.split(".")[0] + f"_fold_{fold}")
        # save metrics
        result = {
            "train_loss": class_loss,
            "train_acc": train_acc,
            "domain_loss": domain_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(result)
        print(
            "Fold [{}], Epoch [{}] : Train Loss {:.4f}, Train Acc {:.4f}, Domain Loss {:.4f} Val Loss {:.4f}, Val Acc {:.4f}".format(
                fold,
                epoch,
                result["train_loss"],
                result["train_acc"],
                result["domain_loss"],
                result["val_loss"],
                result["val_acc"],
            )
        )
    # validation / test after all epochs
    f = open(val_log, "a")
    model.load_state_dict(torch.load(model_params.split(".")[0] + f"_fold_{fold}"))
    val_acc, val_loss, val_f1, val_auc = test(model, device, loss_fn, val_dl, val_log)
    f_writer = csv.writer(f, lineterminator="\n")
    f_writer.writerow([fold, "FINAL", "/", "/", "/", val_acc, val_f1, val_auc])
    f.close()
    print(f"Final val acc: {val_acc}")
    print(f"Final val f1: {val_f1}")
    print(f"Final val AUC: {val_auc}")
    history.append(
        f"Fold {fold} final results: val_acc={val_acc}, val_f1={val_f1}, val_AUC={val_auc}"
    )
    return history


# CROSS VALIDATION


def cross_val(model, dataset, folds, epochs, type):
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader

    BATCH_SIZE = 1

    all_histories = {}

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1337)
    Y = [y[1] for y in dataset]  # get wsi labels
    global fold
    for fold, (train_index, val_index) in enumerate(
        skf.split(np.zeros(len(dataset)), Y)
    ):
        print(f"Fold {fold}:")
        print(f"  Train index={train_index}")
        print(f"  Val index={val_index}")
        train_ds = [dataset[idx] for idx in train_index]
        val_ds = [dataset[idx] for idx in val_index]

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        history = run_training(model, epochs, train_dl, val_dl, fold=fold, type="Mixer")
        all_histories["Fold_" + str(fold)] = history

    return all_histories


################################################################################

EPOCHS = 100
RUNS = 5
TYPE = "Mixer"
target_folder = "logs/"
log_name = f"5_fold_cv_{EPOCHS}_epochs_EXPERIMENT_NAME.csv"

for run in range(RUNS):
    # output files
    log = f"{target_folder}run_{run}_train_{log_name}"
    val_log = f"{target_folder}run_{run}_val_{log_name}"
    model_params = f"model_params/model_run_{run}_{log_name}"
    f = open(log, "w")
    f_writer = csv.writer(f, lineterminator="\n")
    csv_header = [
        "epoch",
        "train_loss",
        "domain_loss",
        "val_loss",
        "train_acc",
        "val_acc",
    ]
    f_writer.writerow(csv_header)
    f.close()

    f2 = open(val_log, "w")
    csv_header = [
        "fold",
        "epoch",
        "target_label",
        "predicted_label",
        "raw_output",
        "val_acc",
        "val_f1",
        "val_auc",
    ]
    f_writer = csv.writer(f2, lineterminator="\n")
    f_writer.writerow(csv_header)
    f2.close()
    # run
    print(f"RUN {run}")
    cv_history = cross_val(model, dataset, folds=5, epochs=EPOCHS, type=TYPE)
    np.save(f"logs/cv_history_{log_name.split('.')[0]}_run_{run}", cv_history)
