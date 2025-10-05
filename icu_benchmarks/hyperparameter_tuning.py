import os
import copy
import logging
import argparse
from copy import deepcopy
import torch.nn.functional as F


import gin
import json
import hashlib
import pandas as pd
import polars as pl
from pathlib import Path
import pickle
from timeit import default_timer as timer
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit
from icu_benchmarks.data.preprocessor import Preprocessor, PandasClassificationPreprocessor, PolarsClassificationPreprocessor
from icu_benchmarks.constants import RunMode
from icu_benchmarks.run_utils import check_required_keys
from icu_benchmarks.data.constants import DataSplit as Split, DataSegment as Segment, VarType as Var

from icu_benchmarks.data.split_process_data import *
from icu_benchmarks.cross_validation import execute_repeated_cv  # adjust if path is different
from icu_benchmarks.run import *
from icu_benchmarks.models.dl_models.bat import * 

from icu_benchmarks.models.train import load_model
from pathlib import Path

import torch
import random
import numpy as np

parser = argparse.ArgumentParser(description="Fine-tuning hyperparameter tuning script")

parser.add_argument('--size', '-s', type=int, required=True, help='Size of the dataset')
parser.add_argument('--fine_tuning_dataset', '-d', type=str, required=True, help='Name of the fine-tuning dataset')
parser.add_argument('--fine_tune_head', '-f', type=str, required=True, help='Fine-tune only the head (True/False)')

args = parser.parse_args()

# Assign variables
size = args.size
fine_tuning_dataset = args.fine_tuning_dataset
fine_tune_head = args.fine_tune_head

vars_dict = {
    "GROUP": "stay_id",
    "SEQUENCE": "time",
    "LABEL": "label",
    "DYNAMIC": ["alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun", "ca", "cai", "ck", "ckmb", "cl",
        "crea", "crp", "dbp", "fgn", "fio2", "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc", "mcv",
        "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt", "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine",
        "wbc"],
    "STATIC": ["age", "sex", "height", "weight"],
}

# Load the gin config
gin.parse_config_file("/work3/s185395/YAIB/configs/tasks/BinaryClassification.gin")


if fine_tuning_dataset == 'eicu': 
    # Pre-trained on pooled mimic + miiv
    model_path = Path("/work3/s185395/yaib_logs/mimic_miiv/LOS/SSL_BAT_tuned_mimic_miiv/2025-08-27T10-33-15/repetition_0/fold_0/model.ckpt")

elif fine_tuning_dataset == 'miiv': 
    # Pre-trained on pooled eicu + mimic
    model_path = Path("/work3/s185395/yaib_logs/eicu_mimic/LOS/SSL_BAT_tuned_eicu_mimic/2025-08-28T01-27-19/repetition_0/fold_0/model.ckpt")

elif fine_tuning_dataset == 'mimic': 
    # Pre-trained on pooled eicu + miiv
    model_path = Path("/work3/s185395/yaib_logs/eicu_miiv/LOS/SSL_BAT_tuned_eicu_miiv/2025-08-27T23-57-28/repetition_0/fold_0/model.ckpt")

ckpt = torch.load(model_path, map_location="cpu")
hparams = ckpt.get("hyper_parameters", {})

# Instantiate the model class (init args can be anything required)
model = SSL_BAT(**hparams) 

# Load only encoder weights
encoder_state_dict = {k.replace("model.encoder_class.", ""): v
                      for k, v in ckpt["state_dict"].items()
                      if k.startswith("model.encoder_class.")}

model.model.encoder_class.load_state_dict(encoder_state_dict)

# Extract encoder from SSL_BAT
pretrained_encoder = model.model.encoder_class

# Create classification model using the pretrained encoder
classification_model = EncoderPrediction(
    encoder_class=pretrained_encoder,
    prediction_head=BinaryClassificationHead,
    prediction_head_kwargs={"num_classes": 2}
)

def run_experiment(bz, lr, model_path, fine_tune_head, num_epochs = 200):

    # Setting seed for reproducibility (Only want variability in the subset datasets)
    seed = 42

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)

    bz = bz
    lr = lr

    ckpt = torch.load(model_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    # Reset model parameters 
    # Instantiate the model class (init args can be anything required)
    model = SSL_BAT(**hparams) 

    # Load only encoder weights
    encoder_state_dict = {k.replace("model.encoder_class.", ""): v
                        for k, v in ckpt["state_dict"].items()
                        if k.startswith("model.encoder_class.")}

    model.model.encoder_class.load_state_dict(encoder_state_dict)

    # Extract encoder from SSL_BAT
    pretrained_encoder = model.model.encoder_class

    # Create classification model using the pretrained encoder
    classification_model = EncoderPrediction(
        encoder_class=pretrained_encoder,
        prediction_head=BinaryClassificationHead,
        prediction_head_kwargs={"num_classes": 2}
    )

    finetune_train_loader = DataLoader(finetune_train_set, batch_size=bz, shuffle=True, generator=g, 
                                    collate_fn=finetune_train_set.collate_fn_pad_to_longest_in_batch())
    finetune_val_loader = DataLoader(finetune_val_set, batch_size=bz, shuffle=True, generator=g, 
                                    collate_fn=finetune_val_set.collate_fn_pad_to_longest_in_batch())
    eval_loader = DataLoader(finetune_test_set, batch_size=bz, shuffle=False,
                            collate_fn=finetune_test_set.collate_fn_pad_to_longest_in_batch())

    print(f'Finetuening training dataset length: {len(finetune_train_set)}')
    print(f'Finetuening validation dataset length: {len(finetune_val_set)}')
    print(f'Finetuening test dataset length: {len(finetune_test_set)}')

    # Automatically select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"🔧 Using device: {device}")

    if fine_tune_head:
        for param in classification_model.parameters():
            param.requires_grad = False
        for param in classification_model.head.parameters():
            param.requires_grad = True
    else:
        for param in classification_model.parameters():
            param.requires_grad = True

    classification_model.to(device)
    optimizer = torch.optim.Adam(classification_model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    loss_fn = torch.nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_aurocs, val_aurocs = [], []
    train_auprcs, val_auprcs = [], []

    # Set early stopping parameters
    patience = 6
    best_val_auprc = 0
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        #current_lr = optimizer.param_groups[0]['lr']
        #print(f"📉 Current LR after epoch {epoch+1}: {current_lr:.6f}")
        # ======== TRAINING ========
        classification_model.train()
        total_train_loss = 0
        all_train_labels = []
        all_train_probs = []

        loop = tqdm(finetune_train_loader, desc=f"🔧 Fine-tuning Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            x, mask, label, times, static, *_ = batch
            x = x.to(device).float()
            mask = mask.to(device).float()
            times = times.to(device).float()
            static = static.to(device).float()
            label = label.to(device).long()

            optimizer.zero_grad()
            logits = classification_model(x, static=static, time=times, sensor_mask=mask)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            probs = F.softmax(logits, dim=1)[:, 1]  # Probabilities for class 1

            all_train_labels.extend(label.cpu().numpy())
            all_train_probs.extend(probs.detach().cpu().numpy())
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(finetune_train_loader)
        train_losses.append(avg_train_loss)

        train_auroc = roc_auc_score(all_train_labels, all_train_probs)
        train_auprc = average_precision_score(all_train_labels, all_train_probs)
        train_aurocs.append(train_auroc)
        train_auprcs.append(train_auprc)

        #print(f"✅ Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | AUROC: {train_auroc:.4f} | AUPRC: {train_auprc:.4f}")

        # ======== VALIDATION ========
        classification_model.eval()
        total_val_loss = 0
        all_val_labels = []
        all_val_probs = []

        with torch.no_grad():
            for batch in finetune_val_loader:
                x, mask, label, times, static, *_ = batch
                x = x.to(device).float()
                mask = mask.to(device).float()
                times = times.to(device).float()
                static = static.to(device).float()
                label = label.to(device).long()

                logits = classification_model(x, static=static, time=times, sensor_mask=mask)
                loss = loss_fn(logits, label)
                total_val_loss += loss.item()

                probs = F.softmax(logits, dim=1)[:, 1]
                all_val_labels.extend(label.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())

        avg_val_loss = total_val_loss / len(finetune_val_loader)
        val_losses.append(avg_val_loss)

        val_auroc = roc_auc_score(all_val_labels, all_val_probs)
        val_auprc = average_precision_score(all_val_labels, all_val_probs)
        val_aurocs.append(val_auroc)
        val_auprcs.append(val_auprc)

        #print(f"🧪 Validation — Loss: {avg_val_loss:.4f} | AUROC: {val_auroc:.4f} | AUPRC: {val_auprc:.4f}")

        # ======== EARLY STOPPING & BEST MODEL SAVE ========
        scheduler.step()  # update learning rate based on val AUPRC

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_model_state = deepcopy(classification_model.state_dict())
            epochs_without_improvement = 0
            #print(f"📌 New best AUPRC: {best_val_auprc:.4f} — model checkpoint saved")
        else:
            epochs_without_improvement += 1
            #print(f"⏳ No AUPRC improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= patience:
            #print(f"🛑 Early stopping triggered after {patience} epochs without improvement.")
            break

    # Restore best model after training
    classification_model.load_state_dict(best_model_state)

    # ======== TESTING ========
    #print("\n🚀 Starting evaluation on test set...")

    classification_model.eval()
    total_test_loss = 0
    all_test_labels = []
    all_test_probs = []

    with torch.no_grad():
        test_loop = tqdm(eval_loader, desc="🧪 Evaluating on Test Set")
        for batch in test_loop:
            x, mask, label, times, static, *_ = batch
            x = x.to(device).float()
            mask = mask.to(device).float()
            times = times.to(device).float()
            static = static.to(device).float()
            label = label.to(device).long()

            logits = classification_model(x, static=static, time=times, sensor_mask=mask)
            loss = loss_fn(logits, label)
            total_test_loss += loss.item()

            probs = F.softmax(logits, dim=1)[:, 1]
            all_test_labels.extend(label.cpu().numpy())
            all_test_probs.extend(probs.cpu().numpy())

            test_loop.set_postfix(loss=loss.item())

    avg_test_loss = total_test_loss / len(eval_loader)
    test_auroc = roc_auc_score(all_test_labels, all_test_probs)
    test_auprc = average_precision_score(all_test_labels, all_test_probs)

    #print(f"\n🎯 Test Set Results:")
    #print(f"   Loss : {avg_test_loss:.4f}")
    #print(f"   AUROC: {test_auroc:.4f}")
    #print(f"   AUPRC: {test_auprc:.4f}")

    return {'bz': bz, 'lr': lr,'avg_test_loss': avg_test_loss, 'test_auroc': test_auroc, 'test_auprc': test_auprc}

import polars as pl
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import random_split
from icu_benchmarks.data.loader import *
from torch.utils.data import DataLoader


hp_results = []
seeds = [42, 84, 126, 168, 210] 
for seed in seeds: 

    dataset = fine_tuning_dataset # eicu, miiv, mimic
    #size = 500 # 100, 500, 1000, 2000, 3000, 5000, 7000, 9000. 9506
    subset_path = f"/work3/s185395/YAIB/icu_benchmarks/data/preprocessed_data/{dataset}/{size}_{seed}" # !Change dataset subset here! 

    # Set the directory where your Parquet files are saved
    dir = Path(subset_path)

    # Create the data dictionary in the format returned by preprocess_data()
    data = {}

    for split in ["train", "val", "test"]:
        outcome_path = dir / f"{split}_OUTCOME.parquet"
        features_path = dir / f"{split}_FEATURES.parquet"

        if outcome_path.exists() and features_path.exists():
            data[split] = {
                "OUTCOME": pl.read_parquet(outcome_path),
                "FEATURES": pl.read_parquet(features_path),
            }
            print(f"✅ Loaded {split} data")
        else:
            print(f"⚠️ Missing files for split '{split}'")

    finetune_train_set = BATPolarsDataset(data=data, split="train", ram_cache=False, runmode=RunMode.classification, vars=vars_dict)
    finetune_val_set = BATPolarsDataset(data=data, split="val", ram_cache=False, runmode=RunMode.classification,vars=vars_dict)
    finetune_test_set = BATPolarsDataset(data=data, split="test", ram_cache=False, runmode=RunMode.classification, vars=vars_dict)


    # Learning rates and batch sizes to test
    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    batch_sizes = [64]

    # Store results
    results = []

    # Run the experiment for each (bz, lr) pair
    for bz in batch_sizes:
        for lr in lrs:
            print(f"\n🚀 Running experiment with batch_size={bz}, learning_rate={lr}")
            result = run_experiment(bz, lr, model_path, fine_tune_head, num_epochs = 200) 
            results.append(result)

    hp_results.append(results)
    print(f'Finished results from seed: {seed}')


# Assuming hp_results is your list of rounds with dicts inside
metrics_by_lr = {}

for round_result in hp_results:
    for entry in round_result:
        lr = entry['lr']
        if lr not in metrics_by_lr:
            metrics_by_lr[lr] = {
                'bz': entry['bz'],  # assuming batch size constant per lr
                'test_auroc': [],
                'test_auprc': []
            }
        metrics_by_lr[lr]['test_auroc'].append(entry['test_auroc'])
        metrics_by_lr[lr]['test_auprc'].append(entry['test_auprc'])

# Calculate means and stds and track best lr by highest mean AUPRC
best_lr_info = None
best_auprc_mean = -np.inf

for lr, data in metrics_by_lr.items():
    auroc_mean = np.mean(data['test_auroc'])
    auroc_sd = np.std(data['test_auroc'])
    auprc_mean = np.mean(data['test_auprc'])
    auprc_sd = np.std(data['test_auprc'])

    if auprc_mean > best_auprc_mean:
        best_auprc_mean = auprc_mean
        best_lr_info = {
            'lr': lr,
            'bz': data['bz'],
            'test_auroc_mean': round(auroc_mean, 4),
            'test_auroc_sd': round(auroc_sd, 4),
            'test_auprc_mean': round(auprc_mean, 4),
            'test_auprc_sd': round(auprc_sd, 4),
        }

# Build the filename
if fine_tune_head == True:
    filename = f"/work3/s185395/YAIB/icu_benchmarks/tuning/hp_tuning_results/head/fine_tuning_hp_tuning_{fine_tuning_dataset}_{size}.txt"
elif fine_tune_head == False:
    filename = f"/work3/s185395/YAIB/icu_benchmarks/tuning/hp_tuning_results/full/fine_tuning_hp_tuning_{fine_tuning_dataset}_{size}.txt"
else:
    print('Cannot save file, select either finetuning for head or full model weights in command line argument')

# Save using plain text
with open(filename, 'w') as f:
    f.write(str(best_lr_info))

print(f"Saved to {filename}")
