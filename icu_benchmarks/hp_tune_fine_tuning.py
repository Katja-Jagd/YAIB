#!/usr/bin/env python
# hyperparameter_tuning_bat.py — synchronized with finetune_bat.py

import os
import json
import argparse
from pathlib import Path
from copy import deepcopy

import gin
import torch
import random
import numpy as np
import polars as pl
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

# ICU Benchmarks
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.loader import BATPolarsDataset
from icu_benchmarks.models.dl_models.bat import SSL_BAT, EncoderPrediction, BinaryClassificationHead
from icu_benchmarks.models.dl_models.grud import SSL_GRUD, GRUDEncoderPrediction
from icu_benchmarks.data.preprocessor import *
from icu_benchmarks.models.train import load_model


# ----------------------------------------------------
# Variable map
# ----------------------------------------------------
VARS_DICT = {
    "GROUP": "stay_id",
    "SEQUENCE": "time",
    "LABEL": "label",
    "DYNAMIC": [
        "alb","alp","alt","ast","be","bicar","bili","bili_dir","bnd","bun","ca","cai","ck","ckmb","cl",
        "crea","crp","dbp","fgn","fio2","glu","hgb","hr","inr_pt","k","lact","lymph","map","mch",
        "mchc","mcv","methb","mg","na","neut","o2sat","pco2","ph","phos","plt","po2","ptt","resp",
        "sbp","temp","tnt","urine","wbc"
    ],
    "STATIC": ["age", "sex", "height", "weight"],
}

# ----------------------------------------------------
# Utilities
# ----------------------------------------------------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_subset(dataset, task, size, seed, subset_root):
    path = Path(subset_root) / task / dataset / f"{size}_{seed}"
    data = {}
    for split in ["train", "val", "test"]:
        o = path / f"{split}_OUTCOME.parquet"
        f = path / f"{split}_FEATURES.parquet"
        if not o.exists() or not f.exists():
            raise FileNotFoundError(f"Missing required files for {split} in {path}")
        data[split] = {"OUTCOME": pl.read_parquet(o), "FEATURES": pl.read_parquet(f)}
    return data


def build_datasets(data):
    return (
        BATPolarsDataset(data=data, split="train", ram_cache=False, runmode=RunMode.classification, vars=VARS_DICT),
        BATPolarsDataset(data=data, split="val",   ram_cache=False, runmode=RunMode.classification, vars=VARS_DICT),
        BATPolarsDataset(data=data, split="test",  ram_cache=False, runmode=RunMode.classification, vars=VARS_DICT),
    )

MODEL_REGISTRY = {
    "bat": {
        "ssl_class": SSL_BAT,
        "prediction_wrapper": EncoderPrediction,
    },
    "grud": {
        "ssl_class": SSL_GRUD,
        "prediction_wrapper": GRUDEncoderPrediction,
    },
}

def load_pretrained_model(ckpt_path, model_type):
    """
    Load a pretrained SSL model (BAT or GRUD) and wrap it
    with a binary classification head for mortality prediction.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    ssl_class = MODEL_REGISTRY[model_type]["ssl_class"]
    wrapper_class = MODEL_REGISTRY[model_type]["prediction_wrapper"]

    # Instantiate SSL model
    ssl_model = ssl_class(**hparams)

    # Load encoder weights from checkpoint
    encoder_state = {
        k.replace("model.encoder_class.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.encoder_class.")
    }

    ssl_model.model.encoder_class.load_state_dict(encoder_state)

    # Wrap encoder with classification head
    model = wrapper_class(
        encoder_class=ssl_model.model.encoder_class,
        prediction_head=BinaryClassificationHead,
        prediction_head_kwargs={"num_classes": 2},
    )

    return model


# ----------------------------------------------------
# Training + Validation + Testing
# ----------------------------------------------------
def run_single_experiment(
    dataset, task, size, seed, model_path, model_type, lr, batch_size, fine_tune_head, num_epochs, subset_root
):
    # ------------------------------------------------
    # EXACT SAME SEEDING BEHAVIOR AS SCRIPT 2
    # ------------------------------------------------
    set_seeds(42)

    # load dataset
    data = load_subset(dataset, task, size, seed, subset_root)
    train_set, val_set, test_set = build_datasets(data)

    g = torch.Generator().manual_seed(42)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=g,
        collate_fn=train_set.collate_fn_pad_to_longest_in_batch()
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=True, generator=g,
        collate_fn=val_set.collate_fn_pad_to_longest_in_batch()
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=test_set.collate_fn_pad_to_longest_in_batch()
    )

    # model
    model = load_pretrained_model(model_path, model_type)

    # freeze/unfreeze exactly like Script 2
    if fine_tune_head:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_fn = torch.nn.CrossEntropyLoss()

    patience = 3
    best_val_auprc = 0
    best_state = None
    no_imp = 0

    # ------------------------------------------------
    # TRAINING LOOP — COPY OF SCRIPT 2
    # ------------------------------------------------
    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        tot_loss = 0
        all_lab, all_prob = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} train")
        for batch in pbar:
            x, mask, label, times, static, *_ = batch
            x = x.float().to(device)
            mask = mask.float().to(device)
            times = times.float().to(device)
            static = static.float().to(device)
            label = label.long().to(device)

            optimizer.zero_grad()
            logits = model(x, static=static, time=times, sensor_mask=mask)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            probs = F.softmax(logits, 1)[:, 1]

            all_lab.extend(label.cpu().numpy())
            all_prob.extend(probs.detach().cpu().numpy())
            pbar.set_postfix(loss=loss.item())

        train_auroc = roc_auc_score(all_lab, all_prob)
        train_auprc = average_precision_score(all_lab, all_prob)

        # ---- val ----
        model.eval()
        all_lab, all_prob = [], []
        vloss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, mask, label, times, static, *_ = batch
                x = x.float().to(device)
                mask = mask.float().to(device)
                times = times.float().to(device)
                static = static.float().to(device)
                label = label.long().to(device)

                logits = model(x, static=static, time=times, sensor_mask=mask)
                vloss += loss_fn(logits, label).item()

                probs = F.softmax(logits, dim=1)[:, 1]
                all_lab.extend(label.cpu().numpy())
                all_prob.extend(probs.cpu().numpy())

        val_auroc = roc_auc_score(all_lab, all_prob)
        val_auprc = average_precision_score(all_lab, all_prob)
        scheduler.step()

        print(f"Epoch {epoch+1}: val_auprc={val_auprc:.4f}")

        # early stopping (same as Script 2)
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stopping after {patience} epochs with no improvement.")
                break

    # restore best
    model.load_state_dict(best_state)

    # ----- test -----
    model.eval()
    all_lab, all_prob = [], []
    tloss = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            x, mask, label, times, static, *_ = batch
            x = x.float().to(device)
            mask = mask.float().to(device)
            times = times.float().to(device)
            static = static.float().to(device)
            label = label.long().to(device)

            logits = model(x, static=static, time=times, sensor_mask=mask)
            tloss += loss_fn(logits, label).item()

            probs = F.softmax(logits, dim=1)[:, 1]
            all_lab.extend(label.cpu().numpy())
            all_prob.extend(probs.cpu().numpy())

    test_auroc = roc_auc_score(all_lab, all_prob)
    test_auprc = average_precision_score(all_lab, all_prob)

    return {
        "lr": lr,
        "batch_size": batch_size,
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
    }


# ----------------------------------------------------
# MAIN LOGIC
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True, type=str, default='Mortality24')
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lrs", nargs="+", type=float, required=True)
    parser.add_argument("--fine_tune_head", action="store_true")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["bat", "grud"], help="Which pretrained SSL model to fine-tune")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--subset_root", default="/work3/s185395/YAIB/icu_benchmarks/data/preprocessed_data")
    args = parser.parse_args()

    results = []
    for lr in args.lrs:
        print(f"\n=== Running LR = {lr} === Dataset = {args.dataset} === Seed = {args.seed} === Size = {args.size} ===")
        res = run_single_experiment(
            dataset=args.dataset,
            task=args.task,
            size=args.size,
            seed=args.seed,
            model_path=args.model_path,
            model_type=args.model_type,
            lr=lr,
            batch_size=args.batch_size,
            fine_tune_head=args.fine_tune_head,
            num_epochs=args.num_epochs,
            subset_root=args.subset_root,
        )
        
        res["Dataset"] = args.dataset
        res["Size"] = args.size
        res["Fine_tune_head"] = args.fine_tune_head
        results.append(res)
        print(res)

    print("\n=== Summary ===")
    for r in results:
        print(r)
