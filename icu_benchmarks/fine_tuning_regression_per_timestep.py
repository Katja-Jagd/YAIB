#!/usr/bin/env python
# finetune_ssl_timestep_regression.py

import os
import json
import hashlib
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import torch
import random
import numpy as np
import polars as pl
from tqdm import tqdm
from torch.utils.data import DataLoader

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.loader import BATPolarsDataset

# Import shared utilities
from icu_benchmarks.fine_tuning_utils import (
    VARS_DICT,
    set_seeds,
    load_subset_as_data_dict,
    build_datasets as build_datasets_shared,
    parse_int_list,
)

# [DEBUG]
def state_dict_transfer_report(model: torch.nn.Module, loaded_state: dict) -> dict:
    """
    Reports how much of `loaded_state` is used by `model` when loading with strict=False.
    Returns counts and parameter coverage by numel.
    """
    model_sd = model.state_dict()
    # keys that match name AND shape
    matched = {k for k, v in loaded_state.items() if k in model_sd and model_sd[k].shape == v.shape}
    missing = [k for k in model_sd.keys() if k not in matched]
    unexpected = [k for k in loaded_state.keys() if k not in model_sd]

    matched_numel = sum(model_sd[k].numel() for k in matched)
    total_numel = sum(v.numel() for v in model_sd.values())

    return {
        "matched_keys": len(matched),
        "model_keys": len(model_sd),
        "unexpected_keys": len(unexpected),
        "missing_keys": len(missing),
        "matched_numel": int(matched_numel),
        "total_numel": int(total_numel),
        "matched_param_frac": float(matched_numel / max(1, total_numel)),
    }
# [DEBUG]

# [DEBUG]
def _collect_observed_labels(loader, device) -> np.ndarray:
    """Collects all observed labels (label[obs_mask]) from a dataloader."""
    ys = []
    with torch.no_grad():
        for batch in loader:
            x, mask, label, times, static, delta, obs_mask = batch
            label = label.to(device).float()
            obs_mask = obs_mask.to(device).bool()
            valid_label = label[obs_mask]
            if valid_label.numel() > 0:
                ys.append(valid_label.detach().cpu().numpy())
    if not ys:
        return np.array([], dtype=np.float64)
    return np.concatenate(ys).astype(np.float64)


def eval_constant_predictor(
    loader,
    device,
    constant_value: float,
    loss_fn: torch.nn.Module,
) -> Tuple[float, float, float]:
    """
    Evaluates a constant predictor y_hat = constant_value on a split.
    Returns: (avg_loss_over_batches, rmse, mae) computed on observed points only.
    """
    total_loss = 0.0
    n_batches = 0
    ytrue_all = []
    ypred_all = []

    with torch.no_grad():
        for batch in loader:
            x, mask, label, times, static, delta, obs_mask = batch
            label = label.to(device).float()
            obs_mask = obs_mask.to(device).bool()

            valid_label = label[obs_mask]
            if valid_label.numel() == 0:
                # keep batch accounting consistent with your training loop
                loss = torch.tensor(0.0, device=device)
            else:
                pred = torch.full_like(valid_label, float(constant_value))
                loss = loss_fn(pred, valid_label)

                ytrue_all.extend(valid_label.detach().cpu().numpy().tolist())
                ypred_all.extend(pred.detach().cpu().numpy().tolist())

            total_loss += float(loss.item())
            n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    rmse, mae = rmse_mae(np.array(ytrue_all), np.array(ypred_all))
    return avg_loss, rmse, mae
# [DEBUG]

"""
# [DEBUG]
def summarize_labels_from_parquet(data_split: Dict[str, pl.DataFrame], split_name: str):
    df = data_split["OUTCOME"]
    if "label" not in df.columns:
        print(f"[{split_name}] OUTCOME columns: {df.columns}")
        raise KeyError("Expected OUTCOME to contain 'label'")

    s = df.select(pl.col("label").cast(pl.Float64)).to_series()
    # drop nulls/nans
    s = s.drop_nulls()
    arr = s.to_numpy()
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        print(f"[{split_name}] No finite labels found.")
        return

    q = np.quantile(arr, [0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0])
    uniq = np.unique(arr)
    print(f"\n[{split_name}] label summary:")
    print(f"  n={arr.size}")
    print(f"  mean={arr.mean():.6f} std={arr.std():.6f}")
    print(f"  min={arr.min():.6f} max={arr.max():.6f}")
    print(f"  quantiles: p0={q[0]:.6f} p1={q[1]:.6f} p5={q[2]:.6f} p10={q[3]:.6f} "
          f"p50={q[4]:.6f} p90={q[5]:.6f} p95={q[6]:.6f} p99={q[7]:.6f} p100={q[8]:.6f}")
    print(f"  unique_count={uniq.size} (first 20 unique: {uniq[:20]})")
# [DEBUG]
"""
# [DEBUG]
PREVIEW_EPOCHS = {1, 5, 10}   # human epoch numbers (not zero-based)
PREVIEW_K = 20

def preview_predictions(
    split_name: str,
    pred: torch.Tensor,      # (B,T)
    label: torch.Tensor,     # (B,T)
    obs_mask: torch.Tensor,  # (B,T) bool
    k: int = 20,
):
    """
    Prints k random (y_true, y_pred, error) points among observed timesteps.
    """
    with torch.no_grad():
        # flatten only observed points
        vp = pred[obs_mask].detach().cpu().flatten()
        vl = label[obs_mask].detach().cpu().flatten()

        n = vl.numel()
        if n == 0:
            print(f"[{split_name}] preview: no observed points in this batch")
            return

        k = min(k, n)
        idx = torch.randperm(n)[:k]
        rows = []
        for i in idx.tolist():
            yt = float(vl[i])
            yp = float(vp[i])
            rows.append((yt, yp, yp - yt, abs(yp - yt)))

        # sort by absolute error so you can immediately see if it's “way off”
        rows.sort(key=lambda r: r[3], reverse=True)

        print(f"\n[{split_name}] prediction preview (top-{k} by |error| in this batch):")
        for yt, yp, err, aerr in rows:
            print(f"  y_true={yt:8.4f}  y_pred={yp:8.4f}  err={err:8.4f}  |err|={aerr:8.4f}")
# [DEBUG]

# BAT
from icu_benchmarks.models.dl_models.bat import (
    SSL_BAT,
    EncoderPrediction,
    RegressionHead,
    AutoregressiveEncoderCrossParallel,
)

# GRU-D
from icu_benchmarks.models.dl_models.grud import (
    SSL_GRUD,
    GRUDEncoderPrediction,
    GRUDEncoder,
)


# Note: VARS_DICT, set_seeds, load_subset_as_data_dict, parse_int_list imported from fine_tuning_utils

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


# -------------------------
# Utilities
# -------------------------
def build_datasets(
    data: Dict[str, Dict[str, pl.DataFrame]],
) -> Tuple[BATPolarsDataset, BATPolarsDataset, BATPolarsDataset]:
    """Wrapper around shared build_datasets with regression mode."""
    return build_datasets_shared(data, runmode=RunMode.regression, vars_dict=VARS_DICT)


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    if y_true.size == 0:
        return float("nan"), float("nan")
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae


# -------------------------
# Model builder
# -------------------------
def build_model_from_ckpt_timestep_regression(
    ckpt_path: Path,
    model_type: str,
) -> torch.nn.Module:
    """
    Builds a per-timestep regression model (outputs (B,T) when output_dim=1)
    and loads SSL encoder weights from ckpt keys under 'model.encoder_class.*'.
    """

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(MODEL_REGISTRY.keys())}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    encoder_state_dict = {
        k.replace("model.encoder_class.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.encoder_class.")
    }
    if not encoder_state_dict:
        raise KeyError(
            "No encoder keys found under 'model.encoder_class.*'. "
            "Update prefix stripping to match your checkpoint."
        )

    sensors_count = hparams["input_size"][1]
    max_timepoint_count = hparams["input_size"][2]
    static_count = hparams.get("static_count", 4)

    if model_type == "bat":
        # IMPORTANT: per-timestep requires autoregressive BAT encoder
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
       
        report = state_dict_transfer_report(encoder, encoder_state_dict) # [DEBUG]
        encoder.load_state_dict(encoder_state_dict, strict=False)
        print(
            "[BAT SSL transfer] "
            f"matched_keys={report['matched_keys']}/{report['model_keys']} | "
            f"matched_params={report['matched_param_frac']:.1%}"
        ) # [DEBUG]

        model = EncoderPrediction(
            encoder_class=encoder,
            prediction_head=RegressionHead,
            prediction_head_kwargs={"output_dim": 1},
        )
        return model

    if model_type == "grud":
        # Use your newly-supported pooling="none" to return (B,T,H)
        encoder = GRUDEncoder(
            device="cpu",
            pooling="none",  # <-- this is the change you added in grud.py
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
        encoder.load_state_dict(encoder_state_dict, strict=False)

        model = GRUDEncoderPrediction(
            encoder_class=encoder,
            prediction_head=RegressionHead,
            prediction_head_kwargs={"output_dim": 1},
        )
        return model

    raise ValueError(f"Unhandled model_type: {model_type}")


# -------------------------
# Config + Results
# -------------------------
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
    num_epochs: int
    patience: int
    subset_root: str
    output_dir: str
    debug_pause: bool = False


@dataclass
class RunResult:
    dataset: str
    task: str
    size: int
    seed: int
    batch_size: int
    lr: float
    num_epochs: int
    fine_tune_head: bool
    model_path: str
    avg_test_loss: float
    test_rmse: float
    test_mae: float


# -------------------------
# Train/Eval
# -------------------------
def train_eval_one(config: RunConfig) -> RunResult:
    set_seeds(42)

    subset_path = Path(config.subset_root) / config.task / config.dataset / f"{config.size}_{config.seed}"
    data = load_subset_as_data_dict(subset_path)

    #summarize_labels_from_parquet(data["train"], "train")
    #summarize_labels_from_parquet(data["val"], "val")
    #summarize_labels_from_parquet(data["test"], "test")

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

    # [DEBUG]
    loss_fn = torch.nn.MSELoss()
    # -------------------------
    # Mean-constant baseline
    # -------------------------
    train_obs_labels = _collect_observed_labels(train_loader, device)
    if train_obs_labels.size == 0:
        mean_const = float("nan")
        print("\n[BASELINE] No observed labels in train split; cannot compute mean baseline.")
    else:
        mean_const = float(train_obs_labels.mean())
        print(f"\n[BASELINE] Mean-constant predictor (computed on TRAIN observed points): {mean_const:.6f}")

        b_train_loss, b_train_rmse, b_train_mae = eval_constant_predictor(
            train_loader, device, mean_const, loss_fn
        )
        b_val_loss, b_val_rmse, b_val_mae = eval_constant_predictor(
            val_loader, device, mean_const, loss_fn
        )
        b_test_loss, b_test_rmse, b_test_mae = eval_constant_predictor(
            test_loader, device, mean_const, loss_fn
        )

        print(
            "[BASELINE] "
            f"train_loss={b_train_loss:.6f} rmse={b_train_rmse:.4f} mae={b_train_mae:.4f} | "
            f"val_loss={b_val_loss:.6f} rmse={b_val_rmse:.4f} mae={b_val_mae:.4f} | "
            f"test_loss={b_test_loss:.6f} rmse={b_test_rmse:.4f} mae={b_test_mae:.4f}"
        )
    # [DEBUG]
    
    model = build_model_from_ckpt_timestep_regression(
        Path(config.model_path),
        model_type=config.model_type,
    )

    # Freeze/unfreeze
    if config.fine_tune_head:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True

    model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_fn = torch.nn.MSELoss()
    #loss_fn = torch.nn.L1Loss()

    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    best_state: Optional[dict] = None

    for epoch in range(config.num_epochs):
        # [DEBUG]
        do_preview = (epoch + 1) in PREVIEW_EPOCHS
        did_preview_train = False
        did_preview_val = False
        # [DEBUG]

        # TRAIN
        model.train()
        total_train_loss = 0.0
        all_train_ytrue: List[float] = []
        all_train_ypred: List[float] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} (train)")
        for batch in pbar:
            x, mask, label, times, static, delta, obs_mask = batch

            x = x.to(device).float()
            mask = mask.to(device).float()
            times = times.to(device).float()
            static = static.to(device).float()
            label = label.to(device).float()
            obs_mask = obs_mask.to(device).bool()

            optimizer.zero_grad()

            pred = model(x, static=static, time=times, sensor_mask=mask)

            # pred expected: (B,T) for output_dim=1
            if pred.dim() == 3 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            if pred.dim() != 2:
                raise ValueError(f"Expected per-timestep predictions (B,T), got: {pred.shape}")
            if label.dim() != 2:
                raise ValueError(f"Expected per-timestep labels (B,T), got: {label.shape}")

            valid_pred = pred[obs_mask]
            valid_label = label[obs_mask]

            # [DEBUG]
            if do_preview and (not did_preview_train):
                preview_predictions("train", pred, label, obs_mask, k=PREVIEW_K)
                did_preview_train = True
            # [DEBUG]

            if valid_label.numel() == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = loss_fn(valid_pred, valid_label)

            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

            all_train_ytrue.extend(valid_label.detach().cpu().numpy().tolist())
            all_train_ypred.extend(valid_pred.detach().cpu().numpy().tolist())

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_rmse, train_mae = rmse_mae(np.array(all_train_ytrue), np.array(all_train_ypred))

        # VAL
        model.eval()
        total_val_loss = 0.0
        all_val_ytrue: List[float] = []
        all_val_ypred: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                x, mask, label, times, static, delta, obs_mask = batch

                x = x.to(device).float()
                mask = mask.to(device).float()
                times = times.to(device).float()
                static = static.to(device).float()
                label = label.to(device).float()
                obs_mask = obs_mask.to(device).bool()

                pred = model(x, static=static, time=times, sensor_mask=mask)
                if pred.dim() == 3 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
                if pred.dim() != 2:
                    raise ValueError(f"Expected per-timestep predictions (B,T), got: {pred.shape}")
                if label.dim() != 2:
                    raise ValueError(f"Expected per-timestep labels (B,T), got: {label.shape}")

                valid_pred = pred[obs_mask]
                valid_label = label[obs_mask]

                # [DEBUG]
                if do_preview and (not did_preview_val):
                    preview_predictions("val", pred, label, obs_mask, k=PREVIEW_K)
                    did_preview_val = True
                # [DEBUG]

                if valid_label.numel() == 0:
                    loss = torch.tensor(0.0, device=device)
                else:
                    loss = loss_fn(valid_pred, valid_label)

                total_val_loss += float(loss.item())
                all_val_ytrue.extend(valid_label.detach().cpu().numpy().tolist())
                all_val_ypred.extend(valid_pred.detach().cpu().numpy().tolist())

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_rmse, val_mae = rmse_mae(np.array(all_val_ytrue), np.array(all_val_ypred))

        print(
        f"Epoch {epoch+1}: "
        f"train_loss={avg_train_loss:.6f} rmse={train_rmse:.4f} mae={train_mae:.4f} | "
        f"val_loss={avg_val_loss:.6f} rmse={val_rmse:.4f} mae={val_mae:.4f} "
        f"(mae*336={val_mae * 336:.2f})"
        )


        scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after {config.patience} epochs without val RMSE improvement.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # TEST
    model.eval()
    total_test_loss = 0.0
    all_test_ytrue: List[float] = []
    all_test_ypred: List[float] = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            x, mask, label, times, static, delta, obs_mask = batch

            x = x.to(device).float()
            mask = mask.to(device).float()
            times = times.to(device).float()
            static = static.to(device).float()
            label = label.to(device).float()
            obs_mask = obs_mask.to(device).bool()

            pred = model(x, static=static, time=times, sensor_mask=mask)
            if pred.dim() == 3 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)

            valid_pred = pred[obs_mask]
            valid_label = label[obs_mask]

            if valid_label.numel() == 0:
                loss = torch.tensor(0.0, device=device)
            else:
                loss = loss_fn(valid_pred, valid_label)

            total_test_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

            all_test_ytrue.extend(valid_label.detach().cpu().numpy().tolist())
            all_test_ypred.extend(valid_pred.detach().cpu().numpy().tolist())

    avg_test_loss = total_test_loss / max(1, len(test_loader))
    test_rmse, test_mae = rmse_mae(np.array(all_test_ytrue), np.array(all_test_ypred))

    print(
        "\nTEST RESULTS "
        f"(dataset={config.dataset}, task={config.task}, size={config.size}, seed={config.seed}): "
        f"loss={avg_test_loss:.6f} rmse={test_rmse:.4f} mae={test_mae:.4f}"
    )

    return RunResult(
        dataset=config.dataset,
        task=config.task,
        size=config.size,
        seed=config.seed,
        batch_size=config.batch_size,
        lr=config.lr,
        num_epochs=config.num_epochs,
        fine_tune_head=config.fine_tune_head,
        model_path=config.model_path,
        avg_test_loss=avg_test_loss,
        test_rmse=test_rmse,
        test_mae=test_mae,
    )


# Removed: Using imported parse_int_list from fine_tuning_utils


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SSL_BAT / SSL_GRUD for per-timestep regression.")
    parser.add_argument("--debug_pause", action="store_true")
    parser.add_argument("--model_path", required=True, type=str, help="Path to pretrained SSL checkpoint .ckpt")
    parser.add_argument("--model_type", required=True, choices=["bat", "grud"], help="Which SSL backbone to fine-tune")
    parser.add_argument("--dataset", default="mimic", type=str)
    parser.add_argument("--task", default="LengthOfStay", type=str, help="Used for folder structure under subset_root")
    parser.add_argument("--sizes", default="1000", type=str)
    parser.add_argument("--seeds", default="42", type=str)
    parser.add_argument("--fine_tune_head", action="store_true", help="Only fine-tune the regression head")
    parser.add_argument("--bz", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--subset_root", default="icu_benchmarks/data/preprocessed_data", type=str,
                        help="Root containing {task}/{dataset}/{size}_{seed}/ parquet files")

    args = parser.parse_args()

    sizes = parse_int_list(args.sizes)
    seeds = parse_int_list(args.seeds)

    mode_str = "head" if args.fine_tune_head else "full"
    output_dir = Path(f"finetuning_results_regression/pretrained_{args.model_type.upper()}/{args.task}/{args.dataset}/{mode_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_id = hashlib.md5(json.dumps({
        "model_type": args.model_type,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "task": args.task,
        "sizes": sizes,
        "seeds": seeds,
        "fine_tune_head": args.fine_tune_head,
        "bz": args.bz,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "subset_root": args.subset_root,
    }, sort_keys=True).encode()).hexdigest()[:10]

    per_run_log = output_dir / f"runs_{sweep_id}.jsonl"
    csv_path = output_dir / f"summary_{sweep_id}.csv"

    with (output_dir / f"meta_{sweep_id}.json").open("w") as f:
        json.dump({
            "sweep_id": sweep_id,
            "args": vars(args),
            "sizes": sizes,
            "seeds": seeds
        }, f, indent=2)

    all_results: List[RunResult] = []
    for size in sizes:
        for seed in seeds:
            run_cfg = RunConfig(
                dataset=args.dataset,
                task=args.task,
                model_type=args.model_type,
                size=size,
                seed=seed,
                model_path=args.model_path,
                fine_tune_head=bool(args.fine_tune_head),
                batch_size=args.bz,
                lr=args.lr,
                num_epochs=args.num_epochs,
                patience=args.patience,
                subset_root=args.subset_root,
                output_dir=str(output_dir),
                debug_pause=bool(args.debug_pause),
            )
            try:
                result = train_eval_one(run_cfg)
            except Exception as e:
                print(f"[ERROR] size={size} seed={seed}: {e}")
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
    else:
        print("\nNo successful runs to summarize.")


if __name__ == "__main__":
    main()
