#!/usr/bin/env python
# fine_tuning_aki_timestep_classification.py
#
# Fine-tuning script for AKI (per-timestep binary classification).
# Uses BAT autoregressive encoder (no pooling) or GRU-D pooling="none".
#
# Metrics: AUROC/AUPRC computed over ALL valid timesteps across the split (flattened),
# masking padded/unobserved timesteps via obs_mask if present, otherwise derived from sensor_mask.

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
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
import torch.nn.functional as F

# ICU Benchmarks
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

# -------------------------
# Configurable variable map
# -------------------------
VARS_DICT = {
    "GROUP": "stay_id",
    "SEQUENCE": "time",
    "LABEL": "label",
    "DYNAMIC": [
        "alb","alp","alt","ast","be","bicar","bili","bili_dir","bnd","bun","ca","cai","ck","ckmb","cl",
        "crea","crp","dbp","fgn","fio2","glu","hgb","hr","inr_pt","k","lact","lymph","map","mch","mchc","mcv",
        "methb","mg","na","neut","o2sat","pco2","ph","phos","plt","po2","ptt","resp","sbp","temp","tnt","urine","wbc"
    ],
    "STATIC": ["age", "sex", "height", "weight"],
}

# -------------------------
# Utilities
# -------------------------
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_subset_as_data_dict(base_dir: Path) -> Dict[str, Dict[str, pl.DataFrame]]:
    data = {}
    for split in ["train", "val", "test"]:
        outcome_path = base_dir / f"{split}_OUTCOME.parquet"
        features_path = base_dir / f"{split}_FEATURES.parquet"
        if not outcome_path.exists() or not features_path.exists():
            raise FileNotFoundError(
                f"Missing files for split '{split}'. Expected:\n  {outcome_path}\n  {features_path}"
            )
        data[split] = {
            "OUTCOME": pl.read_parquet(outcome_path),
            "FEATURES": pl.read_parquet(features_path),
        }
    return data


def build_datasets(data: Dict[str, Dict[str, pl.DataFrame]]) -> Tuple[BATPolarsDataset, BATPolarsDataset, BATPolarsDataset]:
    train_set = BATPolarsDataset(data=data, split="train", ram_cache=False, runmode=RunMode.classification, vars=VARS_DICT)
    val_set   = BATPolarsDataset(data=data, split="val",   ram_cache=False, runmode=RunMode.classification, vars=VARS_DICT)
    test_set  = BATPolarsDataset(data=data, split="test",  ram_cache=False, runmode=RunMode.classification, vars=VARS_DICT)
    return train_set, val_set, test_set


def parse_int_list(arg: str) -> List[int]:
    s = arg.strip()
    if ":" in s:
        start, stop, step = [int(x) for x in s.split(":")]
        return list(range(start, stop + (1 if step > 0 else -1), step))
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]


def maybe_extract_obs_mask(rest: List[object], label_T: int, device: torch.device) -> Optional[torch.Tensor]:
    """
    Some dataset variants return an obs_mask (bool) as an extra tensor in the batch.
    We try to detect it robustly.
    """
    for item in rest:
        if torch.is_tensor(item) and item.dtype in (torch.bool, torch.uint8):
            if item.ndim == 2 and item.shape[1] == label_T:
                return item.to(device).bool()
    return None


def derive_obs_mask_from_sensor_mask(sensor_mask: torch.Tensor, label_T: int) -> torch.Tensor:
    """
    sensor_mask can be [B,D,T] or [B,T,D]. Return [B,T] boolean valid timestep mask.
    """
    if sensor_mask.ndim != 3:
        raise ValueError(f"Expected sensor_mask rank-3, got {tuple(sensor_mask.shape)}")
    if sensor_mask.shape[-1] == label_T:
        # [B,D,T]
        return sensor_mask.bool().any(dim=1)
    if sensor_mask.shape[1] == label_T:
        # [B,T,D]
        return sensor_mask.bool().any(dim=2)
    raise ValueError(f"Cannot infer time axis: sensor_mask {tuple(sensor_mask.shape)} vs label_T={label_T}")


def flatten_valid_timesteps(logits: torch.Tensor, label: torch.Tensor, obs_mask: torch.Tensor):
    """
    logits: [B,T,2] or [B,T] or [B,T,1]
    label:  [B,T] (bool or 0/1)
    obs_mask: [B,T] bool
    Returns:
      logits_use: [N,2] for CE (if logits had 2 classes)
      labels_use: [N] long
    """
    # normalize label
    if label.ndim == 3 and label.shape[-1] == 1:
        label = label.squeeze(-1)
    if label.dtype == torch.bool:
        label = label.long()
    else:
        label = label.long()

    # normalize logits to [B,T,2]
    if logits.ndim == 4 and logits.shape[-1] == 2:
        logits = logits.squeeze(1)
    if logits.ndim == 2:
        raise RuntimeError(f"Got logits {tuple(logits.shape)}; AKI per-timestep expects [B,T,2].")
    if logits.ndim == 3 and logits.shape[-1] == 1:
        # treat as single-logit binary; convert to 2-class logits
        logits = torch.cat([-logits, logits], dim=-1)
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise RuntimeError(f"Expected logits [B,T,2], got {tuple(logits.shape)}")

    B, T, C = logits.shape
    logits_flat = logits.reshape(B * T, C)
    label_flat = label.reshape(B * T)
    mask_flat = obs_mask.reshape(B * T)

    logits_use = logits_flat[mask_flat]
    labels_use = label_flat[mask_flat]
    return logits_use, labels_use


def build_model_from_ckpt(ckpt_path: Path, model_type: str) -> torch.nn.Module:
    """
    Loads encoder weights from SSL checkpoint (model.encoder_class.*) and builds a per-timestep
    classification wrapper.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    encoder_state_dict = {
        k.replace("model.encoder_class.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.encoder_class.")
    }
    if not encoder_state_dict:
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
        encoder.load_state_dict(encoder_state_dict, strict=False)

        model = EncoderPrediction(
            encoder_class=encoder,
            prediction_head=TimeseriesClassificationHead,
            prediction_head_kwargs={"num_classes": 2},
        )
        return model

    if model_type == "grud":
        encoder = GRUDEncoder(
            device="cpu",
            pooling="none",
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
            prediction_head=TimeseriesClassificationHead,
            prediction_head_kwargs={"num_classes": 2},
        )
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


@dataclass
class RunConfig:
    dataset: str
    task: str
    sizes: List[int]
    seeds: List[int]
    model_path: str
    model_type: str
    fine_tune_head: bool
    batch_size: int
    lr: float
    num_epochs: int
    patience: int
    subset_root: str
    output_dir: str


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
    model_type: str
    avg_test_loss: float
    test_auroc: float
    test_auprc: float


def train_eval_one(dataset: str, task: str, size: int, seed: int,
                   model_path: str, model_type: str,
                   fine_tune_head: bool, batch_size: int, lr: float,
                   num_epochs: int, patience: int, subset_root: str) -> RunResult:

    set_seeds(42)

    subset_path = Path(subset_root) / task / dataset / f"{size}_{seed}"
    data = load_subset_as_data_dict(subset_path)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_ckpt(Path(model_path), model_type=model_type)
    if fine_tune_head:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True

    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_auprc = -float("inf")
    best_state = None
    epochs_without_improvement = 0

    # --------- Training loop ----------
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_probs: List[float] = []
        train_labels: List[int] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (train)")
        for batch in pbar:
            x, sensor_mask, label, times, static, *rest = batch
            x = x.to(device).float()
            sensor_mask = sensor_mask.to(device).float()
            times = times.to(device).float()
            static = static.to(device).float()
            label = label.to(device)

            # normalize label to [B,T]
            if label.ndim == 3 and label.shape[-1] == 1:
                label = label.squeeze(-1)
            if label.ndim != 2:
                raise ValueError(f"Expected AKI label [B,T], got {tuple(label.shape)}")

            obs_mask = maybe_extract_obs_mask(rest, label_T=label.shape[1], device=device)
            if obs_mask is None:
                obs_mask = derive_obs_mask_from_sensor_mask(sensor_mask, label_T=label.shape[1]).to(device)

            optimizer.zero_grad()
            logits = model(x, static=static, time=times, sensor_mask=sensor_mask)

            logits_use, labels_use = flatten_valid_timesteps(logits, label, obs_mask)
            if labels_use.numel() == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = loss_fn(logits_use, labels_use)

            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

            probs_pos = F.softmax(logits_use, dim=-1)[:, 1]
            train_probs.extend(probs_pos.detach().cpu().numpy().tolist())
            train_labels.extend(labels_use.detach().cpu().numpy().tolist())

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_auroc = roc_auc_score(train_labels, train_probs) if len(set(train_labels)) > 1 else float("nan")
        train_auprc = average_precision_score(train_labels, train_probs) if len(set(train_labels)) > 1 else float("nan")

        # --------- Validation ----------
        model.eval()
        total_val_loss = 0.0
        val_probs: List[float] = []
        val_labels: List[int] = []

        with torch.no_grad():
            for batch in val_loader:
                x, sensor_mask, label, times, static, *rest = batch
                x = x.to(device).float()
                sensor_mask = sensor_mask.to(device).float()
                times = times.to(device).float()
                static = static.to(device).float()
                label = label.to(device)

                if label.ndim == 3 and label.shape[-1] == 1:
                    label = label.squeeze(-1)
                if label.ndim != 2:
                    raise ValueError(f"Expected AKI label [B,T], got {tuple(label.shape)}")

                obs_mask = maybe_extract_obs_mask(rest, label_T=label.shape[1], device=device)
                if obs_mask is None:
                    obs_mask = derive_obs_mask_from_sensor_mask(sensor_mask, label_T=label.shape[1]).to(device)

                logits = model(x, static=static, time=times, sensor_mask=sensor_mask)
                logits_use, labels_use = flatten_valid_timesteps(logits, label, obs_mask)
                if labels_use.numel() == 0:
                    continue

                loss = loss_fn(logits_use, labels_use)
                total_val_loss += float(loss.item())

                probs_pos = F.softmax(logits_use, dim=-1)[:, 1]
                val_probs.extend(probs_pos.detach().cpu().numpy().tolist())
                val_labels.extend(labels_use.detach().cpu().numpy().tolist())

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_auroc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else float("nan")
        val_auprc = average_precision_score(val_labels, val_probs) if len(set(val_labels)) > 1 else float("nan")

        print(
            f"\nEpoch {epoch+1}: "
            f"train_loss={avg_train_loss:.4f} auprc={train_auprc:.4f} auroc={train_auroc:.4f} | "
            f"val_loss={avg_val_loss:.4f} auprc={val_auprc:.4f} auroc={val_auroc:.4f}"
        )

        scheduler.step()

        # early stopping by VAL AUPRC
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without val AUPRC improvement.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --------- TEST ----------
    model.eval()
    total_test_loss = 0.0
    test_probs: List[float] = []
    test_labels: List[int] = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            x, sensor_mask, label, times, static, *rest = batch
            x = x.to(device).float()
            sensor_mask = sensor_mask.to(device).float()
            times = times.to(device).float()
            static = static.to(device).float()
            label = label.to(device)

            if label.ndim == 3 and label.shape[-1] == 1:
                label = label.squeeze(-1)
            if label.ndim != 2:
                raise ValueError(f"Expected AKI label [B,T], got {tuple(label.shape)}")

            obs_mask = maybe_extract_obs_mask(rest, label_T=label.shape[1], device=device)
            if obs_mask is None:
                obs_mask = derive_obs_mask_from_sensor_mask(sensor_mask, label_T=label.shape[1]).to(device)

            logits = model(x, static=static, time=times, sensor_mask=sensor_mask)
            logits_use, labels_use = flatten_valid_timesteps(logits, label, obs_mask)
            if labels_use.numel() == 0:
                continue

            loss = loss_fn(logits_use, labels_use)
            total_test_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

            probs_pos = F.softmax(logits_use, dim=-1)[:, 1]
            test_probs.extend(probs_pos.detach().cpu().numpy().tolist())
            test_labels.extend(labels_use.detach().cpu().numpy().tolist())

    avg_test_loss = total_test_loss / max(1, len(test_loader))
    test_auroc = roc_auc_score(test_labels, test_probs) if len(set(test_labels)) > 1 else float("nan")
    test_auprc = average_precision_score(test_labels, test_probs) if len(set(test_labels)) > 1 else float("nan")

    print(
        "\nTEST RESULTS "
        f"(dataset={dataset}, task={task}, size={size}, seed={seed}): "
        f"loss={avg_test_loss:.4f} auprc={test_auprc:.4f} auroc={test_auroc:.4f}"
    )

    return RunResult(
        dataset=dataset,
        task=task,
        size=size,
        seed=seed,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        fine_tune_head=fine_tune_head,
        model_path=model_path,
        model_type=model_type,
        avg_test_loss=avg_test_loss,
        test_auroc=test_auroc,
        test_auprc=test_auprc,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained SSL model on AKI per-timestep classification")
    parser.add_argument("--model_path", required=True, type=str, help="Path to pretrained checkpoint .ckpt")
    parser.add_argument("--model_type", required=True, choices=["bat", "grud"], help="Which pretrained SSL model")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name (e.g., mimic)")
    parser.add_argument("--task", default="AKI", type=str, help="Task folder name under subset_root (default: AKI)")
    parser.add_argument("--sizes", default="9506", type=str, help='e.g. "100,500,1000" or "100:9000:100"')
    parser.add_argument("--seeds", default="42", type=str, help='e.g. "42,84,126"')
    parser.add_argument("--fine_tune_head", action="store_true", help="Only fine-tune the prediction head")
    parser.add_argument("--bz", default=64, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--subset_root", default="icu_benchmarks/data/preprocessed_data", type=str,
                        help="Root path containing {task}/{dataset}/{size}_{seed}/ parquet files")
    args = parser.parse_args()

    sizes = parse_int_list(args.sizes)
    seeds = parse_int_list(args.seeds)

    mode_str = "head" if args.fine_tune_head else "full"

    output_dir = Path(f"finetuning_results/AKI/{args.model_type}/{args.dataset}/{mode_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_id = hashlib.md5(json.dumps({
        "model_path": args.model_path,
        "model_type": args.model_type,
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
            try:
                result = train_eval_one(
                    dataset=args.dataset,
                    task=args.task,
                    size=size,
                    seed=seed,
                    model_path=args.model_path,
                    model_type=args.model_type,
                    fine_tune_head=bool(args.fine_tune_head),
                    batch_size=args.bz,
                    lr=args.lr,
                    num_epochs=args.num_epochs,
                    patience=args.patience,
                    subset_root=args.subset_root,
                )
            except Exception as e:
                print(f"[ERROR] size={size} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
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
