#!/usr/bin/env python
# hyperparameter_tuning_timestep_classification.py
#
# Per-timestep classification with random hyperparameter sweeps.
# Supports:
#   - BAT via AutoregressiveEncoderCrossParallel + TimeseriesClassificationHead
#   - GRUD via GRUDEncoder(pooling="none") + TimeseriesClassificationHead
#
# Sweepable hyperparameters:
#   - lr
#   - weight_decay
#   - dropout
#   - attn_dropout (BAT only; ignored for GRUD if not applicable)
#   - num_sweeps
#
# Output:
#   - meta_<sweep_id>.json
#   - runs_<sweep_id>.jsonl
#   - summary_<sweep_id>.csv

import os
import json
import math
import hashlib
import argparse
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import numpy as np
import polars as pl
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.loader import BATPolarsDataset

from icu_benchmarks.fine_tuning_utils import (
    VARS_DICT,
    set_seeds,
    load_subset_as_data_dict,
    build_datasets as build_datasets_shared,
    parse_int_list,
)

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
# Registry
# ----------------------------------------------------
MODEL_REGISTRY = {
    "bat": {
        "prediction_wrapper": EncoderPrediction,
        "supports_attn_dropout": True,
    },
    "grud": {
        "prediction_wrapper": GRUDEncoderPrediction,
        "supports_attn_dropout": False,
    },
}


# ----------------------------------------------------
# Utilities
# ----------------------------------------------------
def build_datasets(
    data: Dict[str, Dict[str, pl.DataFrame]]
) -> Tuple[BATPolarsDataset, BATPolarsDataset, BATPolarsDataset]:
    """Wrapper for classification datasets - timestep labels remain [B,T]."""
    return build_datasets_shared(data, runmode=RunMode.classification, vars_dict=VARS_DICT)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def sample_log_uniform(rng: random.Random, low: float, high: float) -> float:
    if low <= 0 or high <= 0:
        raise ValueError(f"log-uniform bounds must be > 0, got low={low}, high={high}")
    if low > high:
        raise ValueError(f"log-uniform requires low <= high, got low={low}, high={high}")
    return 10 ** rng.uniform(math.log10(low), math.log10(high))


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


def flatten_valid_timesteps(
    logits_bt2: torch.Tensor,
    labels_bt: torch.Tensor,
    obs_mask_bt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits_bt2: [B,T,2]
    labels_bt:  [B,T]
    obs_mask_bt:[B,T] bool
    Returns:
      logits_use [N,2], labels_use [N]
    """
    labels_bt = labels_bt.long()

    B, T, C = logits_bt2.shape
    logits_flat = logits_bt2.reshape(B * T, C)
    labels_flat = labels_bt.reshape(B * T)
    mask_flat = obs_mask_bt.reshape(B * T)

    logits_use = logits_flat[mask_flat]
    labels_use = labels_flat[mask_flat]
    return logits_use, labels_use


def safe_binary_metrics(labels: List[int], probs: List[float]) -> Tuple[float, float]:
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(labels, probs), average_precision_score(labels, probs)


# ----------------------------------------------------
# Model building from SSL checkpoint
# ----------------------------------------------------
def build_timestep_classification_model_from_ckpt(
    ckpt_path: Path,
    model_type: str,
    dropout_override: Optional[float] = None,
    attn_dropout_override: Optional[float] = None,
) -> torch.nn.Module:
    """
    Build per-timestep classification model from SSL checkpoint.
    Output logits are expected to be [B,T,2].

    Supports overriding dropout-related hyperparameters before reconstruction.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(MODEL_REGISTRY.keys())}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {}).copy()

    if dropout_override is not None and "dropout" in hparams:
        hparams["dropout"] = dropout_override
    if attn_dropout_override is not None and "attn_dropout" in hparams:
        hparams["attn_dropout"] = attn_dropout_override

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

        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        print(
            f"[INFO] Loaded BAT encoder weights: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"(dropout={hparams.get('dropout')}, attn_dropout={hparams.get('attn_dropout')})"
        )

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

        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        print(
            f"[INFO] Loaded GRUD encoder weights: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"(dropout={hparams.get('dropout')}, recurrent_dropout={hparams.get('recurrent_dropout', 'n/a')})"
        )

        model = GRUDEncoderPrediction(
            encoder_class=encoder,
            prediction_head=TimeseriesClassificationHead,
            prediction_head_kwargs={"num_classes": 2},
        )
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


# ----------------------------------------------------
# Config / result dataclasses
# ----------------------------------------------------
@dataclass
class RunConfig:
    dataset: str
    task: str
    size: int
    seed: int
    model_path: str
    model_type: str
    fine_tune_head: bool
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float
    attn_dropout: float
    num_epochs: int
    patience: int
    subset_root: str
    output_dir: str
    sweep_idx: int = 0


@dataclass
class RunResult:
    dataset: str
    task: str
    size: int
    seed: int
    sweep_idx: int
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float
    attn_dropout: float
    num_epochs: int
    fine_tune_head: bool
    model_path: str
    best_val_auroc: float
    best_val_auprc: float
    best_epoch: int
    test_auroc: float
    test_auprc: float


# ----------------------------------------------------
# Train / eval one run
# ----------------------------------------------------
def train_eval_one(config: RunConfig) -> RunResult:
    # Deterministic training, only subset variability changes
    set_seeds(42)

    subset_path = Path(config.subset_root) / config.task / config.dataset / f"{config.size}_{config.seed}"
    data = load_subset_as_data_dict(subset_path)
    train_set, val_set, test_set = build_datasets(data)

    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        generator=g,
        collate_fn=train_set.collate_fn_pad_to_longest_in_batch(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=val_set.collate_fn_pad_to_longest_in_batch(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=test_set.collate_fn_pad_to_longest_in_batch(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_timestep_classification_model_from_ckpt(
        Path(config.model_path),
        model_type=config.model_type,
        dropout_override=config.dropout,
        attn_dropout_override=config.attn_dropout,
    )

    if config.fine_tune_head:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True

    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_auprc = -float("inf")
    best_val_auroc = float("nan")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0

    # -------------------------
    # Train / validation loop
    # -------------------------
    for epoch in range(config.num_epochs):
        model.train()
        train_probs: List[float] = []
        train_labels: List[int] = []
        total_train_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Sweep {config.sweep_idx} | Epoch {epoch+1}/{config.num_epochs} (train)"
        )

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
                raise ValueError(f"Expected timestep label shape [B,T], got {tuple(label.shape)}")

            if obs_mask is None:
                obs_mask = derive_obs_mask_from_sensor_mask(sensor_mask, label_T=label.shape[1]).to(device)

            optimizer.zero_grad()

            logits = model(x, static=static, time=times, sensor_mask=sensor_mask)

            if logits.ndim == 4 and logits.shape[-1] == 2:
                logits = logits.squeeze(1)

            if logits.ndim != 3 or logits.shape[-1] != 2:
                raise ValueError(
                    f"Expected logits [B,T,2], got {tuple(logits.shape)}. "
                    f"Ensure encoder is non-pooled and TimeseriesClassificationHead is used."
                )

            logits_use, labels_use = flatten_valid_timesteps(logits, label, obs_mask)

            if labels_use.numel() == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = loss_fn(logits_use, labels_use)

            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

            if labels_use.numel() > 0:
                probs_pos = F.softmax(logits_use, dim=-1)[:, 1]
                train_probs.extend(probs_pos.detach().cpu().numpy().tolist())
                train_labels.extend(labels_use.detach().cpu().numpy().tolist())

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_auroc, train_auprc = safe_binary_metrics(train_labels, train_probs)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_probs: List[float] = []
        val_labels: List[int] = []
        total_val_loss = 0.0

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
                    raise ValueError(f"Expected timestep label shape [B,T], got {tuple(label.shape)}")

                if obs_mask is None:
                    obs_mask = derive_obs_mask_from_sensor_mask(sensor_mask, label_T=label.shape[1]).to(device)

                logits = model(x, static=static, time=times, sensor_mask=sensor_mask)

                if logits.ndim == 4 and logits.shape[-1] == 2:
                    logits = logits.squeeze(1)

                if logits.ndim != 3 or logits.shape[-1] != 2:
                    raise ValueError(f"Expected logits [B,T,2], got {tuple(logits.shape)}")

                logits_use, labels_use = flatten_valid_timesteps(logits, label, obs_mask)

                if labels_use.numel() == 0:
                    continue

                loss = loss_fn(logits_use, labels_use)
                total_val_loss += float(loss.item())

                probs_pos = F.softmax(logits_use, dim=-1)[:, 1]
                val_probs.extend(probs_pos.detach().cpu().numpy().tolist())
                val_labels.extend(labels_use.detach().cpu().numpy().tolist())

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_auroc, val_auprc = safe_binary_metrics(val_labels, val_probs)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={avg_train_loss:.4f} auroc={train_auroc:.4f} auprc={train_auprc:.4f} | "
            f"val_loss={avg_val_loss:.4f} auroc={val_auroc:.4f} auprc={val_auprc:.4f}"
        )

        scheduler.step()

        if not np.isnan(val_auprc) and val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after {config.patience} epochs without val AUPRC improvement.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # -------------------------
    # Test
    # -------------------------
    model.eval()
    test_probs: List[float] = []
    test_labels: List[int] = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Sweep {config.sweep_idx} | Testing")
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
                raise ValueError(f"Expected timestep label shape [B,T], got {tuple(label.shape)}")

            if obs_mask is None:
                obs_mask = derive_obs_mask_from_sensor_mask(sensor_mask, label_T=label.shape[1]).to(device)

            logits = model(x, static=static, time=times, sensor_mask=sensor_mask)

            if logits.ndim == 4 and logits.shape[-1] == 2:
                logits = logits.squeeze(1)

            if logits.ndim != 3 or logits.shape[-1] != 2:
                raise ValueError(f"Expected logits [B,T,2], got {tuple(logits.shape)}")

            logits_use, labels_use = flatten_valid_timesteps(logits, label, obs_mask)
            if labels_use.numel() == 0:
                continue

            probs_pos = F.softmax(logits_use, dim=-1)[:, 1]
            test_probs.extend(probs_pos.detach().cpu().numpy().tolist())
            test_labels.extend(labels_use.detach().cpu().numpy().tolist())

    test_auroc, test_auprc = safe_binary_metrics(test_labels, test_probs)

    print(
        "\nBEST VAL RESULTS "
        f"(dataset={config.dataset}, task={config.task}, size={config.size}, "
        f"seed={config.seed}, sweep={config.sweep_idx}): "
        f"epoch={best_epoch} val_auroc={best_val_auroc:.4f} val_auprc={best_val_auprc:.4f} | "
        f"test_auroc={test_auroc:.4f} test_auprc={test_auprc:.4f}"
    )

    return RunResult(
        dataset=config.dataset,
        task=config.task,
        size=config.size,
        seed=config.seed,
        sweep_idx=config.sweep_idx,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        num_epochs=config.num_epochs,
        fine_tune_head=config.fine_tune_head,
        model_path=config.model_path,
        best_val_auroc=best_val_auroc,
        best_val_auprc=best_val_auprc,
        best_epoch=best_epoch,
        test_auroc=test_auroc,
        test_auprc=test_auprc,
    )


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Per-timestep classification fine-tuning with random hyperparameter sweeps."
    )

    parser.add_argument("--model_path", required=True, type=str, help="Path to pretrained checkpoint .ckpt")
    parser.add_argument(
        "--model_type",
        required=True,
        choices=["bat", "grud"],
        help="Which pretrained SSL backbone to fine-tune",
    )
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name (eicu, miiv, mimic, etc.)")
    parser.add_argument("--task", required=True, type=str, help="Timestep classification task name, e.g. AKI")
    parser.add_argument("--sizes", default="9506", type=str, help='e.g. "100,500,1000" or "100:9000:100"')
    parser.add_argument("--seeds", default="42", type=str, help='e.g. "42,84,126"')
    parser.add_argument("--fine_tune_head", action="store_true", help="Only fine-tune the classification head")
    parser.add_argument("--bz", default=64, type=int, help="Batch size")
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument(
        "--subset_root",
        default="icu_benchmarks/data/preprocessed_data",
        type=str,
        help="Root path that contains {task}/{dataset}/{size}_{seed}/ parquet files",
    )

    # Random sweep args
    parser.add_argument("--num_sweeps", type=int, default=1, help="Number of random hyperparameter sweeps to run")
    parser.add_argument("--sweep_seed", type=int, default=123, help="Random seed for hyperparameter sampling")

    # LR
    parser.add_argument("--use_fixed_lr", action="store_true", help="Use --lr as fixed LR for all sweeps")
    parser.add_argument("--lr", default=None, type=float, help="Fixed learning rate if --use_fixed_lr is set")
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=1e-2)

    # Weight decay
    parser.add_argument("--weight_decay_min", type=float, default=1e-4)
    parser.add_argument("--weight_decay_max", type=float, default=1e-1)

    # Dropout choices
    parser.add_argument("--dropout_choices", type=str, default="0,0.2,0.4,0.6")
    parser.add_argument("--attn_dropout_choices", type=str, default="0,0.2,0.4,0.6")

    args = parser.parse_args()

    sizes = parse_int_list(args.sizes)
    seeds = parse_int_list(args.seeds)

    if args.num_sweeps <= 0:
        raise ValueError("--num_sweeps must be >= 1")

    if args.use_fixed_lr and args.lr is None:
        raise ValueError("--use_fixed_lr requires --lr to be provided.")

    dropout_choices = parse_float_list(args.dropout_choices)
    attn_dropout_choices = parse_float_list(args.attn_dropout_choices)

    rng = random.Random(args.sweep_seed)
    mode_str = "head" if args.fine_tune_head else "full"

    output_dir = Path(
        f"finetuning_results/pretrained_{args.model_type.upper()}/{args.task}/{args.dataset}/{mode_str}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_combos = []
    for _ in range(args.num_sweeps):
        hp = {
            "dropout": rng.choice(dropout_choices),
            "attn_dropout": rng.choice(attn_dropout_choices),
            "weight_decay": sample_log_uniform(rng, args.weight_decay_min, args.weight_decay_max),
            "lr": args.lr if args.use_fixed_lr else sample_log_uniform(rng, args.lr_min, args.lr_max),
        }
        sampled_combos.append(hp)

    sweep_meta = {
        "model_type": args.model_type,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "task": args.task,
        "sizes": sizes,
        "seeds": seeds,
        "fine_tune_head": args.fine_tune_head,
        "bz": args.bz,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "subset_root": args.subset_root,
        "num_sweeps": args.num_sweeps,
        "sweep_seed": args.sweep_seed,
        "dropout_choices": dropout_choices,
        "attn_dropout_choices": attn_dropout_choices,
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "weight_decay_min": args.weight_decay_min,
        "weight_decay_max": args.weight_decay_max,
        "sampled_combos": sampled_combos,
    }

    sweep_id = hashlib.md5(json.dumps(sweep_meta, sort_keys=True).encode()).hexdigest()[:10]

    per_run_log = output_dir / f"runs_{sweep_id}.jsonl"
    csv_path = output_dir / f"summary_{sweep_id}.csv"
    meta_path = output_dir / f"meta_{sweep_id}.json"

    with meta_path.open("w") as f:
        json.dump(
            {
                "sweep_id": sweep_id,
                "args": vars(args),
                "sizes": sizes,
                "seeds": seeds,
                "sampled_combos": sampled_combos,
            },
            f,
            indent=2,
        )

    print(f"[INFO] Sweep ID: {sweep_id}")
    print(f"[INFO] Writing metadata to: {meta_path}")
    print(f"[INFO] Number of sampled sweeps: {len(sampled_combos)}")

    all_results: List[RunResult] = []

    for sweep_idx, hp in enumerate(sampled_combos, start=1):
        print(
            f"\n{'=' * 100}\n"
            f"[SWEEP {sweep_idx}/{len(sampled_combos)}] "
            f"dropout={hp['dropout']} | attn_dropout={hp['attn_dropout']} | "
            f"weight_decay={hp['weight_decay']:.6g} | lr={hp['lr']:.6g}\n"
            f"{'=' * 100}"
        )

        for size in sizes:
            for seed in seeds:
                run_cfg = RunConfig(
                    dataset=args.dataset,
                    task=args.task,
                    size=size,
                    seed=seed,
                    model_path=args.model_path,
                    model_type=args.model_type,
                    fine_tune_head=bool(args.fine_tune_head),
                    batch_size=args.bz,
                    lr=hp["lr"],
                    weight_decay=hp["weight_decay"],
                    dropout=hp["dropout"],
                    attn_dropout=hp["attn_dropout"],
                    num_epochs=args.num_epochs,
                    patience=args.patience,
                    subset_root=args.subset_root,
                    output_dir=str(output_dir),
                    sweep_idx=sweep_idx,
                )

                try:
                    result = train_eval_one(run_cfg)
                except Exception as e:
                    print(
                        f"[ERROR] sweep={sweep_idx} size={size} seed={seed} "
                        f"dropout={hp['dropout']} attn_dropout={hp['attn_dropout']} "
                        f"weight_decay={hp['weight_decay']:.6g} lr={hp['lr']:.6g}: {e}"
                    )
                    continue

                all_results.append(result)
                with per_run_log.open("a") as f:
                    f.write(json.dumps(asdict(result)) + "\n")

    if all_results:
        cols = list(asdict(all_results[0]).keys())
        lines = [",".join(cols)]
        for r in all_results:
            row = [str(asdict(r)[c]) for c in cols]
            lines.append(",".join(row))
        csv_path.write_text("\n".join(lines))

        print(f"\n✅ Wrote summary CSV: {csv_path}")
        print(f"🧾 Per-run JSONL:     {per_run_log}")

        valid_results = [r for r in all_results if not np.isnan(r.best_val_auprc)]
        best_result = max(valid_results, key=lambda r: r.best_val_auprc, default=None)

        if best_result is not None:
            print("\n🏆 Best run by validation AUPRC:")
            print(json.dumps(asdict(best_result), indent=2))
    else:
        print("\nNo successful runs to summarize.")


if __name__ == "__main__":
    main()
