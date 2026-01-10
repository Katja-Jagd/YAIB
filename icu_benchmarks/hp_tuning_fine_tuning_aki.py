#!/usr/bin/env python
# hyperparameter_tuning_aki_timestep_classification.py
#
# AKI-only: per-timestep classification (NO label shrinking).
# Uses AutoregressiveEncoderCrossParallel for BAT (no pooling) and GRUD pooling="none".

import argparse
from pathlib import Path
from copy import deepcopy
import random
import numpy as np
import polars as pl
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.loader import BATPolarsDataset

# BAT
from icu_benchmarks.models.dl_models.bat import (
    AutoregressiveEncoderCrossParallel,
    EncoderPrediction,
    TimeseriesClassificationHead,
)

# GRU-D
from icu_benchmarks.models.dl_models.grud import (
    GRUDEncoder,
    GRUDEncoderPrediction,
)

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
    # AKI is classification, but labels are per-timestep (B,T)
    return (
        BATPolarsDataset(data=data, split="train", ram_cache=False, runmode=RunMode.classification, vars=VARS_DICT),
        BATPolarsDataset(data=data, split="val",   ram_cache=False, runmode=RunMode.classification, vars=VARS_DICT),
        BATPolarsDataset(data=data, split="test",  ram_cache=False, runmode=RunMode.classification, vars=VARS_DICT),
    )


def derive_obs_mask_from_sensor_mask(sensor_mask: torch.Tensor, label_T: int) -> torch.Tensor:
    """
    Create a [B,T] boolean mask of valid timesteps from sensor_mask.
    Handles sensor_mask layouts [B, D, T] or [B, T, D].
    """
    if sensor_mask.ndim != 3:
        raise ValueError(f"Expected sensor_mask rank-3, got {sensor_mask.shape}")

    if sensor_mask.shape[-1] == label_T:
        # [B, D, T]
        valid_t = sensor_mask.bool().any(dim=1)
    elif sensor_mask.shape[1] == label_T:
        # [B, T, D]
        valid_t = sensor_mask.bool().any(dim=2)
    else:
        raise ValueError(
            f"Cannot infer time axis for sensor_mask {sensor_mask.shape} vs label_T={label_T}"
        )
    return valid_t


def flatten_valid_timesteps(logits_bt2: torch.Tensor, labels_bt: torch.Tensor, obs_mask_bt: torch.Tensor):
    """
    logits_bt2: [B,T,2]
    labels_bt:  [B,T] (bool or {0,1})
    obs_mask_bt:[B,T] bool
    Returns logits_use [N,2], labels_use [N]
    """
    if labels_bt.dtype == torch.bool:
        labels_bt = labels_bt.long()
    else:
        labels_bt = labels_bt.long()

    B, T, C = logits_bt2.shape
    logits_flat = logits_bt2.reshape(B * T, C)
    labels_flat = labels_bt.reshape(B * T)
    mask_flat = obs_mask_bt.reshape(B * T)

    logits_use = logits_flat[mask_flat]
    labels_use = labels_flat[mask_flat]
    return logits_use, labels_use


# ----------------------------------------------------
# Model building from SSL checkpoint
# ----------------------------------------------------
def build_aki_timestep_classification_model_from_ckpt(model_path: str, model_type: str) -> torch.nn.Module:
    """
    Builds per-timestep classification model and loads encoder weights from SSL checkpoint:
      - expects keys under 'model.encoder_class.*'
      - returns model whose forward outputs logits [B,T,2]
    """
    ckpt = torch.load(model_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    encoder_state = {
        k.replace("model.encoder_class.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.encoder_class.")
    }
    if not encoder_state:
        raise KeyError("No encoder keys found under 'model.encoder_class.*' in checkpoint")

    sensors_count = hparams["input_size"][1]
    max_timepoint_count = hparams["input_size"][2]
    static_count = hparams.get("static_count", 4)

    if model_type == "bat":
        encoder = AutoregressiveEncoderCrossParallel(
            device="cpu",
            value_embed_size=hparams["value_embed_size"],
            layers=hparams["layers"],
            heads=hparams["heads"],
            dropout=hparams["dropout"],
            attn_dropout=hparams["attn_dropout"],
            use_mask=hparams["use_mask"],
            sensors_count=sensors_count,
            max_timepoint_count=max_timepoint_count,
            static_count=static_count,
        )
        encoder.load_state_dict(encoder_state, strict=False)

        model = EncoderPrediction(
            encoder_class=encoder,
            prediction_head=TimeseriesClassificationHead,
            prediction_head_kwargs={"num_classes": 2},
        )
        return model

    if model_type == "grud":
        encoder = GRUDEncoder(
            device="cpu",
            pooling="none",  # IMPORTANT: keep per-timestep representations
            sensors_count=sensors_count,
            max_timepoint_count=max_timepoint_count,
            static_count=static_count,
            recurrent_n_units=hparams["recurrent_n_units"],
            dropout=hparams["dropout"],
            recurrent_dropout=hparams["recurrent_dropout"],
            use_static=hparams.get("use_static", True),
            obs_strategy=hparams.get("obs_strategy", "both"),
            x_imputation=hparams.get("x_imputation", "zero"),
            input_decay=hparams.get("input_decay", "exp_relu"),
            hidden_decay=hparams.get("hidden_decay", "exp_relu"),
            activation=hparams.get("activation", "tanh"),
            recurrent_activation=hparams.get("recurrent_activation", "hardsigmoid"),
            use_decay_bias=hparams.get("use_decay_bias", True),
            feed_masking=hparams.get("feed_masking", True),
            masking_decay=hparams.get("masking_decay", None),
        )
        encoder.load_state_dict(encoder_state, strict=False)

        model = GRUDEncoderPrediction(
            encoder_class=encoder,
            prediction_head=TimeseriesClassificationHead,
            prediction_head_kwargs={"num_classes": 2},
        )
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


# ----------------------------------------------------
# One LR experiment
# ----------------------------------------------------
def run_single_experiment(
    dataset, task, size, seed,
    model_path, model_type,
    lr, batch_size, fine_tune_head,
    num_epochs, subset_root,
    patience=3,
):
    set_seeds(42)

    data = load_subset(dataset, task, size, seed, subset_root)
    train_set, val_set, test_set = build_datasets(data)

    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=g,
        collate_fn=train_set.collate_fn_pad_to_longest_in_batch()
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=val_set.collate_fn_pad_to_longest_in_batch()
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=test_set.collate_fn_pad_to_longest_in_batch()
    )

    model = build_aki_timestep_classification_model_from_ckpt(model_path, model_type)

    # freeze/unfreeze
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

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_auprc = -float("inf")
    best_state = None
    no_imp = 0

    # -------------------------
    # Train loop
    # -------------------------
    for epoch in range(num_epochs):
        model.train()
        train_probs, train_labels = [], []
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} train")
        for batch in pbar:
            # Some dataset versions return extra fields; be flexible
            # Expected first 5: x, sensor_mask, label, times, static, ...
            x, sensor_mask, label, times, static, *rest = batch

            x = x.float().to(device)
            sensor_mask = sensor_mask.float().to(device)
            times = times.float().to(device)
            static = static.float().to(device)
            label = label.to(device)

            # Try to get obs_mask from batch if present; else derive from sensor_mask
            obs_mask = None
            for item in rest:
                if torch.is_tensor(item) and item.dtype in (torch.bool, torch.uint8):
                    # Heuristic: obs_mask is usually [B,T] bool
                    if item.ndim == 2 and label.ndim >= 2 and item.shape[-1] == label.shape[-1]:
                        obs_mask = item.to(device).bool()
                        break

            # Normalize label to [B,T]
            if label.ndim == 3 and label.shape[-1] == 1:
                label = label.squeeze(-1)
            if label.ndim != 2:
                raise ValueError(f"Expected AKI label shape [B,T], got {tuple(label.shape)}")

            if obs_mask is None:
                obs_mask = derive_obs_mask_from_sensor_mask(sensor_mask, label_T=label.shape[1]).to(device)

            optimizer.zero_grad()

            logits = model(x, static=static, time=times, sensor_mask=sensor_mask)
            # Normalize logits to [B,T,2]
            if logits.ndim == 2:
                raise RuntimeError(
                    f"Model returned logits {tuple(logits.shape)} but AKI needs per-timestep logits [B,T,2]. "
                    f"Ensure encoder is non-pooled and TimeseriesClassificationHead is used."
                )
            if logits.ndim == 4 and logits.shape[-1] == 2:
                # uncommon: [B,?,T,2] -> try squeeze
                logits = logits.squeeze(1)

            if logits.ndim != 3 or logits.shape[-1] != 2:
                raise ValueError(f"Expected logits [B,T,2], got {tuple(logits.shape)}")

            logits_use, labels_use = flatten_valid_timesteps(logits, label, obs_mask)
            if labels_use.numel() == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = loss_fn(logits_use, labels_use)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

            probs_pos = F.softmax(logits_use, dim=-1)[:, 1]
            train_probs.extend(probs_pos.detach().cpu().numpy().tolist())
            train_labels.extend(labels_use.detach().cpu().numpy().tolist())

        # Metrics
        train_auroc = roc_auc_score(train_labels, train_probs) if len(set(train_labels)) > 1 else float("nan")
        train_auprc = average_precision_score(train_labels, train_probs) if len(set(train_labels)) > 1 else float("nan")

        # -------------------------
        # Validation (early stop on AUPRC)
        # -------------------------
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, sensor_mask, label, times, static, *rest = batch
                x = x.float().to(device)
                sensor_mask = sensor_mask.float().to(device)
                times = times.float().to(device)
                static = static.float().to(device)
                label = label.to(device)

                obs_mask = None
                for item in rest:
                    if torch.is_tensor(item) and item.dtype in (torch.bool, torch.uint8):
                        if item.ndim == 2 and label.ndim >= 2 and item.shape[-1] == label.shape[-1]:
                            obs_mask = item.to(device).bool()
                            break

                if label.ndim == 3 and label.shape[-1] == 1:
                    label = label.squeeze(-1)
                if label.ndim != 2:
                    raise ValueError(f"Expected AKI label shape [B,T], got {tuple(label.shape)}")

                if obs_mask is None:
                    obs_mask = derive_obs_mask_from_sensor_mask(sensor_mask, label_T=label.shape[1]).to(device)

                logits = model(x, static=static, time=times, sensor_mask=sensor_mask)
                if logits.ndim != 3 or logits.shape[-1] != 2:
                    raise ValueError(f"Expected logits [B,T,2], got {tuple(logits.shape)}")

                logits_use, labels_use = flatten_valid_timesteps(logits, label, obs_mask)
                if labels_use.numel() == 0:
                    continue

                probs_pos = F.softmax(logits_use, dim=-1)[:, 1]
                val_probs.extend(probs_pos.detach().cpu().numpy().tolist())
                val_labels.extend(labels_use.detach().cpu().numpy().tolist())

        val_auroc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else float("nan")
        val_auprc = average_precision_score(val_labels, val_probs) if len(set(val_labels)) > 1 else float("nan")

        print(
            f"Epoch {epoch+1}: "
            f"train_auprc={train_auprc:.4f} train_auroc={train_auroc:.4f} | "
            f"val_auprc={val_auprc:.4f} val_auroc={val_auroc:.4f}"
        )

        scheduler.step()

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stopping after {patience} epochs without val AUPRC improvement.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # -------------------------
    # Test
    # -------------------------
    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            x, sensor_mask, label, times, static, *rest = batch
            x = x.float().to(device)
            sensor_mask = sensor_mask.float().to(device)
            times = times.float().to(device)
            static = static.float().to(device)
            label = label.to(device)

            obs_mask = None
            for item in rest:
                if torch.is_tensor(item) and item.dtype in (torch.bool, torch.uint8):
                    if item.ndim == 2 and label.ndim >= 2 and item.shape[-1] == label.shape[-1]:
                        obs_mask = item.to(device).bool()
                        break

            if label.ndim == 3 and label.shape[-1] == 1:
                label = label.squeeze(-1)
            if label.ndim != 2:
                raise ValueError(f"Expected AKI label shape [B,T], got {tuple(label.shape)}")

            if obs_mask is None:
                obs_mask = derive_obs_mask_from_sensor_mask(sensor_mask, label_T=label.shape[1]).to(device)

            logits = model(x, static=static, time=times, sensor_mask=sensor_mask)
            if logits.ndim != 3 or logits.shape[-1] != 2:
                raise ValueError(f"Expected logits [B,T,2], got {tuple(logits.shape)}")

            logits_use, labels_use = flatten_valid_timesteps(logits, label, obs_mask)
            if labels_use.numel() == 0:
                continue

            probs_pos = F.softmax(logits_use, dim=-1)[:, 1]
            test_probs.extend(probs_pos.detach().cpu().numpy().tolist())
            test_labels.extend(labels_use.detach().cpu().numpy().tolist())

    test_auroc = roc_auc_score(test_labels, test_probs) if len(set(test_labels)) > 1 else float("nan")
    test_auprc = average_precision_score(test_labels, test_probs) if len(set(test_labels)) > 1 else float("nan")

    return {
        "lr": lr,
        "batch_size": batch_size,
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
    }


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True, type=str)  # should be AKI task folder name
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lrs", nargs="+", type=float, required=True)
    parser.add_argument("--fine_tune_head", action="store_true")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", required=True, choices=["bat", "grud"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--subset_root", default="/work3/s185395/YAIB/icu_benchmarks/data/preprocessed_data")
    args = parser.parse_args()

    results = []
    for lr in args.lrs:
        print(
            f"\n=== Running LR={lr} === Dataset={args.dataset} "
            f"Seed={args.seed} Size={args.size} Task={args.task} ==="
        )

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
            patience=args.patience,
        )

        res["Dataset"] = args.dataset
        res["Task"] = args.task
        res["Size"] = args.size
        res["Seed"] = args.seed
        res["Fine_tune_head"] = args.fine_tune_head
        results.append(res)
        print(res)

    print("\n=== Summary ===")
    for r in results:
        print(r)
