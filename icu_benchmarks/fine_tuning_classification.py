#!/usr/bin/env python
# finetune_bat.py
import os
import json
import hashlib
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Tuple

# --- Third-party / project imports ---
import gin
import torch
import random
import numpy as np
import polars as pl
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

# ICU Benchmarks (your repo)
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.loader import BATPolarsDataset
from icu_benchmarks.models.dl_models.bat import (
    SSL_BAT,
    EncoderPrediction,
    BinaryClassificationHead,
    TimeseriesClassificationHead,
)
from icu_benchmarks.models.dl_models.grud import SSL_GRUD, GRUDEncoderPrediction
from icu_benchmarks.models.dl_models.radv_transformer import SSL_RadVTransformer
from icu_benchmarks.models.dl_models.itransformer import SSL_iTransformer, EncoderPredictionInverted
from icu_benchmarks.models.dl_models.ip_nets import SSL_IPNets, IPNetsEncoderPrediction
from icu_benchmarks.models.dl_models.deep_set_attention import SSL_DeepSetAttention, DeepSetAttentionEncoderPrediction
from icu_benchmarks.cross_validation import execute_repeated_cv
from icu_benchmarks.constants import RunMode
from icu_benchmarks.run import get_mode
from icu_benchmarks.data.preprocessor import (
    Preprocessor,
    PandasClassificationPreprocessor,
    PolarsClassificationPreprocessor,
)
from icu_benchmarks.models.train import load_model

# -------------------------
# Import shared utilities
# -------------------------
from icu_benchmarks.fine_tuning_utils import (
    VARS_DICT,
    set_seeds,
    load_subset_as_data_dict,
    build_datasets as build_datasets_shared,
    parse_int_list,
)

MODEL_REGISTRY = {
    "bat": {
        "ssl_class": SSL_BAT,
        "prediction_wrapper": EncoderPrediction,
        # For timestep tasks you already use a BAT-specific autoregressive encoder
        "supports_timestep_tasks": True,
    },
    "grud": {
        "ssl_class": SSL_GRUD,
        "prediction_wrapper": GRUDEncoderPrediction,
        # Unless your GRUD pipeline is explicitly designed for timestep logits
        "supports_timestep_tasks": False,
    },
    "radv_transformer": {
        "ssl_class": SSL_RadVTransformer,
        "prediction_wrapper": EncoderPrediction,
        "supports_timestep_tasks": False,
    },
    "itransformer": {
        "ssl_class": SSL_iTransformer,
        "prediction_wrapper": EncoderPredictionInverted,
        "supports_timestep_tasks": False,
    },
    "ipnets": {
        "ssl_class": SSL_IPNets,
        "prediction_wrapper": IPNetsEncoderPrediction,
        "supports_timestep_tasks": False,
    },
    "seft": {
        "ssl_class": SSL_DeepSetAttention,
        "prediction_wrapper": DeepSetAttentionEncoderPrediction,
        "supports_timestep_tasks": False,
    },
}

# -------------------------
# Utilities
# -------------------------
def parse_gin_config(gin_path: str):
    """Parse a gin config and make relative `include` paths work."""
    gin.clear_config()
    p = Path(gin_path).resolve()

    # Likely roots
    tasks_dir   = p.parent                           # .../configs/tasks
    configs_dir = tasks_dir.parent                   # .../configs
    repo_root   = configs_dir.parent                 # .../YAIB

    # Add search roots so includes like "configs/.../X.gin" resolve
    gin.add_config_file_search_path(str(tasks_dir))
    gin.add_config_file_search_path(str(configs_dir))
    gin.add_config_file_search_path(str(repo_root))              # crucial for "configs/..."
    gin.add_config_file_search_path(str(repo_root / "configs"))  # extra safety
    gin.add_config_file_search_path(str(repo_root / "configs" / "tasks"))

    # (Optional) also set CWD to repo root to help any other relative references
    try:
        os.chdir(str(repo_root))
    except Exception:
        pass

    # Helpful sanity check (won't crash if missing)
    include_probe = repo_root / "configs" / "tasks" / "common" / "Imports.gin"
    if not include_probe.exists():
        print(f"[WARN] Could not find expected include at: {include_probe}")
        print("[WARN] Current gin search paths:")
        for sp in gin.config._CONFIG_DIR:  # type: ignore[attr-defined]
            print("       -", sp)

    gin.parse_config_file(str(p))


# Removed: set_seeds, load_subset_as_data_dict, parse_int_list
# Now imported from icu_benchmarks.fine_tuning_utils

def build_datasets(data: Dict[str, Dict[str, pl.DataFrame]]) -> Tuple[BATPolarsDataset, BATPolarsDataset, BATPolarsDataset]:
    """Wrapper around shared build_datasets with classification mode."""
    return build_datasets_shared(data, runmode=RunMode.classification, vars_dict=VARS_DICT)


def build_model_from_ckpt(
    ckpt_path: Path,
    model_type: str,
    task: str = "Mortality24",
):
    """
    Build model (BAT or GRUD) from checkpoint with task-appropriate prediction head.

    - Patient-level tasks: BAT or GRUD supported
    - Timestep-level tasks (e.g. Sepsis): BAT supported via AutoregressiveEncoderCrossParallel
      (GRUD timestep support is repo-dependent; default is to block it to avoid silent misuse.)
    """
    TIMESTEP_TASKS = {"Sepsis", "AKI"}  # extend if needed

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(MODEL_REGISTRY.keys())}")

    is_timestep_task = task in TIMESTEP_TASKS

    if is_timestep_task and not MODEL_REGISTRY[model_type]["supports_timestep_tasks"]:
        raise NotImplementedError(
            f"Task '{task}' is treated as timestep-level, but model_type='{model_type}' "
            f"is not configured for timestep-level heads in this script."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    ssl_class = MODEL_REGISTRY[model_type]["ssl_class"]
    wrapper_class = MODEL_REGISTRY[model_type]["prediction_wrapper"]

    # Load encoder state dict (works for both if checkpoint uses this prefix)
    encoder_state_dict = {
        k.replace("model.encoder_class.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.encoder_class.")
    }

    if is_timestep_task:
        # ---- BAT-only timestep path (your existing logic) ----
        from icu_benchmarks.models.dl_models.bat import AutoregressiveEncoderCrossParallel

        sensors_count = hparams["input_size"][1]
        max_timepoint_count = hparams["input_size"][2]
        static_count = hparams.get("static_count", 4)

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

        missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
        # print(f"[INFO] Loaded encoder weights (timestep): missing={len(missing)} unexpected={len(unexpected)}")

        model = wrapper_class(
            encoder_class=encoder,
            prediction_head=TimeseriesClassificationHead,
            prediction_head_kwargs={"num_classes": 2},
        )
        return model

    # ---- Patient-level path (BAT or GRUD) ----
    ssl_model = ssl_class(**hparams)

    # Note: this assumes both BAT and GRUD checkpoints store encoder weights at `model.encoder_class.*`
    ssl_model.model.encoder_class.load_state_dict(encoder_state_dict)
    # print(f"[INFO] Loaded pretrained encoder weights for model_type={model_type}")

    model = wrapper_class(
        encoder_class=ssl_model.model.encoder_class,
        prediction_head=BinaryClassificationHead,
        prediction_head_kwargs={"num_classes": 2},
    )
    return model


@dataclass
class RunConfig:
    dataset: str
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
    task: str = "Mortality24"
    debug_pause: bool = False


@dataclass
class RunResult:
    dataset: str
    size: int
    seed: int
    batch_size: int
    lr: float
    num_epochs: int
    fine_tune_head: bool
    model_path: str
    avg_test_loss: float
    test_auroc: float
    test_auprc: float


def train_eval_one(config: RunConfig) -> RunResult:
    # fixed seed for training procedure (you asked to keep subset variability only)
    set_seeds(42)

    # data paths
    subset_path = Path(config.subset_root) / config.task / config.dataset / f"{config.size}_{config.seed}"
    data = load_subset_as_data_dict(subset_path)

    # datasets & loaders
    train_set, val_set, test_set = build_datasets(data)

    if config.debug_pause:
        input("Press Enter to continue...")

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
        shuffle=True,
        generator=g,
        collate_fn=val_set.collate_fn_pad_to_longest_in_batch(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=test_set.collate_fn_pad_to_longest_in_batch(),
    )

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine if this is a timestep-level task
    TIMESTEP_TASKS = {"Sepsis"}
    is_timestep_task = config.task in TIMESTEP_TASKS

    # model
    model = build_model_from_ckpt(
        Path(config.model_path),
        model_type=config.model_type,
        task=config.task,
    )

    if config.fine_tune_head:
        # freeze all, unfreeze head
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_fn = torch.nn.CrossEntropyLoss()

    patience = config.patience
    # print(f"\n[INFO] Early stopping patience: {patience} epochs")
    best_val_auprc = 0.0
    epochs_without_improvement = 0
    best_state = None

    # --------- Training loop ----------
    for epoch in range(config.num_epochs):
        # TRAIN
        model.train()
        total_train_loss = 0.0
        all_train_labels: List[int] = []
        all_train_probs: List[float] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} (train)")
        batch_idx = 0
        for batch in pbar:
            batch_idx += 1
            x, mask, label, times, static, delta, obs_mask = batch
            x = x.to(device).float()
            mask = mask.to(device).float()
            times = times.to(device).float()
            static = static.to(device).float()
            label = label.to(device).long()
            obs_mask = obs_mask.to(device).bool()

            optimizer.zero_grad()
            try:
                logits = model(x, static=static, time=times, sensor_mask=mask)
            except Exception as e:
                print(f"\n[ERROR] Model forward pass failed: {e}")
                print(f"Input shapes: x={x.shape}, static={static.shape}, times={times.shape}, mask={mask.shape}")
                raise

            # Handle label format and compute loss based on task type
            if is_timestep_task:
                # Timestep-level prediction (e.g., Sepsis)
                # logits: (B, T, num_classes), label: (B, T), obs_mask: (B, T)
                B, T, C = logits.shape

                # Reshape for cross-entropy: (B*T, num_classes) and (B*T,)
                logits_flat = logits.reshape(B * T, C)
                label_flat = label.reshape(B * T)
                obs_mask_flat = obs_mask.reshape(B * T)

                # Only compute loss on valid (non-padded) positions
                if obs_mask_flat.sum() > 0:
                    loss = loss_fn(logits_flat[obs_mask_flat], label_flat[obs_mask_flat])
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

                # For metrics: collect all valid timestep predictions and labels
                probs = F.softmax(logits, dim=-1)[:, :, 1]  # (B, T) - prob of class 1
                valid_probs = probs[obs_mask]
                valid_labels = label[obs_mask]

                # Print predictions vs labels for first batch of first epoch
                if epoch == 0 and batch_idx == 1:
                    if config.debug_pause:
                        input("Press Enter to continue...")

                all_train_labels.extend(valid_labels.detach().cpu().numpy())
                all_train_probs.extend(valid_probs.detach().cpu().numpy())
            else:
                # Patient-level prediction (e.g., Mortality24)
                # logits: (B, num_classes), label: (B,) or (B, T)
                if label.dim() > 1:
                    # Take last valid timestep for patient-level tasks
                    label = label[:, -1]

                loss = loss_fn(logits, label)
                probs = F.softmax(logits, dim=1)[:, 1]
                all_train_labels.extend(label.detach().cpu().numpy())
                all_train_probs.extend(probs.detach().cpu().numpy())

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_auroc = roc_auc_score(all_train_labels, all_train_probs)
        train_auprc = average_precision_score(all_train_labels, all_train_probs)

        # VAL
        model.eval()
        total_val_loss = 0.0
        all_val_labels: List[int] = []
        all_val_probs: List[float] = []

        # Store first batch info for epoch 0 checkpoint
        first_val_batch_data = None

        # Store patient-level data for validation checkpoints every 5 epochs
        val_patient_data = [] if epoch % 5 == 0 else None

        with torch.no_grad():
            val_batch_idx = 0
            for batch in val_loader:
                val_batch_idx += 1
                x, mask, label, times, static, delta, obs_mask = batch
                x = x.to(device).float()
                mask = mask.to(device).float()
                times = times.to(device).float()
                static = static.to(device).float()
                label = label.to(device).long()
                obs_mask = obs_mask.to(device).bool()

                logits = model(x, static=static, time=times, sensor_mask=mask)

                # Save first validation batch for inspection after epoch 0
                if epoch == 0 and val_batch_idx == 1:
                    first_val_batch_data = {
                        'logits': logits.clone(),
                        'label': label.clone(),
                        'obs_mask': obs_mask.clone(),
                        'x': x.clone(),
                        'static': static.clone()
                    }

                # Handle label format and compute loss based on task type
                if is_timestep_task:
                    # Timestep-level prediction
                    B, T, C = logits.shape
                    logits_flat = logits.reshape(B * T, C)
                    label_flat = label.reshape(B * T)
                    obs_mask_flat = obs_mask.reshape(B * T)

                    if obs_mask_flat.sum() > 0:
                        loss = loss_fn(logits_flat[obs_mask_flat], label_flat[obs_mask_flat])
                    else:
                        loss = torch.tensor(0.0, device=device)

                    probs = F.softmax(logits, dim=-1)[:, :, 1]
                    valid_probs = probs[obs_mask]
                    valid_labels = label[obs_mask]
                    all_val_labels.extend(valid_labels.cpu().numpy())
                    all_val_probs.extend(valid_probs.cpu().numpy())

                    # Store patient-level data for checkpoint every 5 epochs
                    if val_patient_data is not None:
                        for b in range(B):
                            patient_labels = label[b][obs_mask[b]]
                            patient_probs = probs[b][obs_mask[b]]
                            if len(patient_labels) > 0:
                                has_positive = (patient_labels == 1).any().item()
                                val_patient_data.append({
                                    'labels': patient_labels.cpu(),
                                    'probs': patient_probs.cpu(),
                                    'has_positive': has_positive
                                })
                else:
                    # Patient-level prediction
                    if label.dim() > 1:
                        label = label[:, -1]

                    loss = loss_fn(logits, label)
                    probs = F.softmax(logits, dim=1)[:, 1]
                    all_val_labels.extend(label.cpu().numpy())
                    all_val_probs.extend(probs.cpu().numpy())

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_auroc = roc_auc_score(all_val_labels, all_val_probs)
        val_auprc = average_precision_score(all_val_labels, all_val_probs)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={avg_train_loss:.4f} auroc={train_auroc:.4f} auprc={train_auprc:.4f} | "
            f"val_loss={avg_val_loss:.4f} auroc={val_auroc:.4f} auprc={val_auprc:.4f}"
        )


        scheduler.step()

        # early stopping by AUPRC
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without val AUPRC improvement.")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # TEST
    model.eval()
    total_test_loss = 0.0
    all_test_labels: List[int] = []
    all_test_probs: List[float] = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            x, mask, label, times, static, delta, obs_mask = batch
            x = x.to(device).float()
            mask = mask.to(device).float()
            times = times.to(device).float()
            static = static.to(device).float()
            label = label.to(device).long()
            obs_mask = obs_mask.to(device).bool()

            logits = model(x, static=static, time=times, sensor_mask=mask)

            # Handle label format and compute loss based on task type
            if is_timestep_task:
                B, T, C = logits.shape
                logits_flat = logits.reshape(B * T, C)
                label_flat = label.reshape(B * T)
                obs_mask_flat = obs_mask.reshape(B * T)

                if obs_mask_flat.sum() > 0:
                    loss = loss_fn(logits_flat[obs_mask_flat], label_flat[obs_mask_flat])
                else:
                    loss = torch.tensor(0.0, device=device)

                probs = F.softmax(logits, dim=-1)[:, :, 1]
                valid_probs = probs[obs_mask]
                valid_labels = label[obs_mask]
                all_test_labels.extend(valid_labels.cpu().numpy())
                all_test_probs.extend(valid_probs.cpu().numpy())
            else:
                if label.dim() > 1:
                    label = label[:, -1]

                loss = loss_fn(logits, label)
                probs = F.softmax(logits, dim=1)[:, 1]
                all_test_labels.extend(label.cpu().numpy())
                all_test_probs.extend(probs.cpu().numpy())

            total_test_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    avg_test_loss = total_test_loss / max(1, len(test_loader))
    test_auroc = roc_auc_score(all_test_labels, all_test_probs)
    test_auprc = average_precision_score(all_test_labels, all_test_probs)

    print(
        "\nTEST RESULTS "
        f"(dataset={config.dataset}, size={config.size}, seed={config.seed}): "
        f"loss={avg_test_loss:.4f} auroc={test_auroc:.4f} auprc={test_auprc:.4f}"
    )


    return RunResult(
        dataset=config.dataset,
        size=config.size,
        seed=config.seed,
        batch_size=config.batch_size,
        lr=config.lr,
        num_epochs=config.num_epochs,
        fine_tune_head=config.fine_tune_head,
        model_path=config.model_path,
        avg_test_loss=avg_test_loss,
        test_auroc=test_auroc,
        test_auprc=test_auprc,
    )


# Note: Using imported parse_int_list from fine_tuning_utils
# This script can also handle colon syntax if needed locally, but imported version is sufficient


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SSL_BAT on ICU subsets and aggregate results.")
    parser.add_argument("--debug_pause", action="store_true")
    parser.add_argument("--model_path", required=True, type=str, help="Path to pretrained checkpoint .ckpt")
    parser.add_argument("--model_type", required=True, choices=["bat", "grud", "radv_transformer", "itransformer", "ipnets", "seft"], help="Which pretrained SSL backbone to fine-tune")
    parser.add_argument("--dataset", default="mimic", type=str, help="Dataset name (eicu, miiv, mimic, or custom)")
    parser.add_argument("--task", default="Mortality24", type=str,
                        choices=["Mortality24", "Sepsis", "AKI", "Mortality"],
                        help="Task name: determines prediction head and label handling")
    parser.add_argument("--sizes", default="9506", type=str, help='e.g. "100,500,1000" or "100:9000:100"')
    parser.add_argument("--seeds", default="42", type=str, help='e.g. "42,84,126"')
    parser.add_argument("--fine_tune_head", action="store_true", help="Only fine-tune the classification head")
    parser.add_argument("--bz", default=32, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--patience", default=None, type=int,
                        help="Early stopping patience (epochs without improvement). Default: 3")
    parser.add_argument("--subset_root", default="icu_benchmarks/data/preprocessed_data", type=str,
                        help="Root path that contains {dataset}/{size}_{seed}/ parquet files")

    args = parser.parse_args()

    sizes = parse_int_list(args.sizes)
    seeds = parse_int_list(args.seeds)

    # Set patience: default to 3 if not specified
    patience = args.patience if args.patience is not None else 3

    # Map flag -> mode for path naming
    mode_str = "head" if args.fine_tune_head else "full"

    output_dir = Path(f"finetuning_results/pretrained_{args.model_type.upper()}/{args.dataset}/{mode_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_id = hashlib.md5(json.dumps({
        "model_type": args.model_type,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "sizes": sizes,
        "seeds": seeds,
        "fine_tune_head": args.fine_tune_head,
        "bz": args.bz,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
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
                model_type=args.model_type,
                size=size,
                seed=seed,
                model_path=args.model_path,
                fine_tune_head=bool(args.fine_tune_head),
                batch_size=args.bz,
                lr=args.lr,
                num_epochs=args.num_epochs,
                patience=patience,
                subset_root=args.subset_root,
                output_dir=str(output_dir),
                task=args.task,
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
