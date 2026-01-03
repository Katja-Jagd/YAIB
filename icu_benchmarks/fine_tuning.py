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


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_subset_as_data_dict(base_dir: Path) -> Dict[str, Dict[str, pl.DataFrame]]:
    """
    Expects layout:
      base_dir/
        train_OUTCOME.parquet
        train_FEATURES.parquet
        val_OUTCOME.parquet
        val_FEATURES.parquet
        test_OUTCOME.parquet
        test_FEATURES.parquet
    Returns a dict compatible with BATPolarsDataset.
    """
    data = {}
    for split in ["train", "val", "test"]:
        outcome_path = base_dir / f"{split}_OUTCOME.parquet"
        features_path = base_dir / f"{split}_FEATURES.parquet"

        if not outcome_path.exists() or not features_path.exists():
            raise FileNotFoundError(
                f"Missing files for split '{split}'. "
                f"Expected:\n  {outcome_path}\n  {features_path}"
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

    # print("\n" + "="*80)
    # print("DATASET INFORMATION")
    # print("="*80)
    # print(f"Training set size: {len(train_set)}")
    # print(f"Validation set size: {len(val_set)}")
    # print(f"Test set size: {len(test_set)}")
    # print(f"Task: {config.task}")
    # print(f"Is timestep task: {config.task in {'Sepsis'}}")
    # print("="*80 + "\n")
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

    # print(f"\n[DEBUG] Model built:")
    # print(f"  Encoder type: {type(model.encoder_class).__name__}")
    # print(f"  Is autoregressive: {getattr(model, 'is_autoregressive', False)}")
    # print(f"  Prediction head type: {type(model.head).__name__}")
    # print()

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

            # Print detailed info for first batch of first epoch
            if epoch == 0 and batch_idx == 1:
                # print("\n" + "="*80)
                # print("FIRST BATCH DATA SHAPES (TRAINING)")
                # print("="*80)
                # print(f"Time series data (x): {x.shape}")
                # print(f"Sensor mask: {mask.shape}")
                # print(f"Labels: {label.shape}")
                # print(f"Times: {times.shape}")
                # print(f"Static features: {static.shape}")
                # print(f"Observation mask: {obs_mask.shape}")
                # print("\nSTATIC FEATURES SAMPLE (first patient):")
                # print(f"  Values: {static[0].cpu().numpy()}")
                # print("\nTIME SERIES SAMPLE (first patient, first 5 timesteps):")
                # print(f"  Values shape: {x[0, :5].shape}")
                # print(f"  Times: {times[0, :5].cpu().numpy()}")
                # print(f"  Obs mask: {obs_mask[0, :5].cpu().numpy()}")
                # print("\nLABEL STATISTICS:")
                # if label.dim() > 1:
                #     print(f"  Labels shape: {label.shape}")
                #     print(f"  First patient labels (first 10 timesteps): {label[0, :10].cpu().numpy()}")
                #     print(f"  Valid timesteps for first patient: {obs_mask[0].sum().item()}")
                #     print(f"  Positive labels in batch: {label[obs_mask].sum().item()} / {obs_mask.sum().item()}")
                # else:
                #     print(f"  Labels shape: {label.shape}")
                #     print(f"  First 5 labels: {label[:5].cpu().numpy()}")
                #     print(f"  Positive labels in batch: {label.sum().item()} / {len(label)}")
                # print("="*80 + "\n")
                if config.debug_pause:
                    input("Press Enter to continue...")

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
                    # print("\n" + "="*80)
                    # print("MODEL PREDICTIONS VS LABELS (FIRST BATCH)")
                    # print("="*80)
                    # print(f"Logits shape: {logits.shape}")
                    # print(f"Probabilities shape: {probs.shape}")
                    #
                    # print(f"\nBATCH-LEVEL STATISTICS:")
                    # print(f"  Total valid timesteps: {obs_mask_flat.sum().item()}")
                    # print(f"  Positive labels (sepsis): {label_flat[obs_mask_flat].sum().item()}")
                    # print(f"  Negative labels (no sepsis): {(label_flat[obs_mask_flat] == 0).sum().item()}")
                    # print(f"  Mean predicted probability: {valid_probs.mean().item():.4f}")
                    # print(f"  Min predicted probability: {valid_probs.min().item():.4f}")
                    # print(f"  Max predicted probability: {valid_probs.max().item():.4f}")
                    # print(f"  Loss: {loss.item():.4f}")
                    #
                    # # Find patients with positive and negative labels
                    # # For each patient, check if they have any positive labels
                    # print(f"\nSAMPLE PATIENTS WITH PREDICTIONS:")
                    #
                    # positive_patients = []
                    # negative_patients = []
                    #
                    # for b in range(B):
                    #     patient_labels = label[b][obs_mask[b]]
                    #     patient_probs = probs[b][obs_mask[b]]
                    #
                    #     if len(patient_labels) > 0:
                    #         has_positive = (patient_labels == 1).any().item()
                    #         if has_positive and len(positive_patients) < 5:
                    #             positive_patients.append((b, patient_labels, patient_probs))
                    #         elif not has_positive and len(negative_patients) < 5:
                    #             negative_patients.append((b, patient_labels, patient_probs))
                    #
                    # print(f"\nPOSITIVE PATIENTS (with sepsis labels):")
                    # for i, (patient_idx, patient_labels, patient_probs) in enumerate(positive_patients):
                    #     print(f"\n  Patient {patient_idx} (from batch):")
                    #     labels_np = patient_labels.detach().cpu().numpy()
                    #     probs_np = patient_probs.detach().cpu().numpy()
                    #     print(f"    Labels:       {labels_np}")
                    #     print(f"    Predictions:  {np.round(probs_np, 4)}")
                    #     n_correct = ((probs_np > 0.5) == labels_np).sum()
                    #     print(f"    Accuracy: {n_correct}/{len(labels_np)} ({100*n_correct/len(labels_np):.1f}%)")
                    #
                    # if len(positive_patients) == 0:
                    #     print("  No patients with positive labels in this batch")
                    #
                    # print(f"\nNEGATIVE PATIENTS (no sepsis labels):")
                    # for i, (patient_idx, patient_labels, patient_probs) in enumerate(negative_patients):
                    #     print(f"\n  Patient {patient_idx} (from batch):")
                    #     labels_np = patient_labels.detach().cpu().numpy()
                    #     probs_np = patient_probs.detach().cpu().numpy()
                    #     print(f"    Labels:       {labels_np}")
                    #     print(f"    Predictions:  {np.round(probs_np, 4)}")
                    #     n_correct = ((probs_np > 0.5) == labels_np).sum()
                    #     print(f"    Accuracy: {n_correct}/{len(labels_np)} ({100*n_correct/len(labels_np):.1f}%)")
                    #
                    # if len(negative_patients) == 0:
                    #     print("  No patients with all negative labels in this batch")
                    #
                    # print("="*80 + "\n")
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

        # Print training metrics summary
        if epoch == 0:
            # print("\n" + "="*80)
            # print(f"EPOCH {epoch+1} TRAINING METRICS SUMMARY")
            # print("="*80)
            # print(f"Total training samples (valid timesteps): {len(all_train_labels)}")
            # print(f"Positive labels: {sum(all_train_labels)}")
            # print(f"Class balance: {sum(all_train_labels)/len(all_train_labels):.4f}")
            # print(f"Average loss: {avg_train_loss:.4f}")
            # print(f"AUROC: {train_auroc:.4f}")
            # print(f"AUPRC: {train_auprc:.4f}")
            # print("="*80 + "\n")
            if config.debug_pause:
                input("Press Enter to continue...")

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

        # Print detailed output shapes and values for first sample after epoch 0
        if epoch == 0 and first_val_batch_data is not None:
            # print("\n" + "="*80)
            # print("FIRST SAMPLE PREDICTION DETAILS (AFTER EPOCH 1)")
            # print("="*80)
            #
            # logits_first = first_val_batch_data['logits']
            # label_first = first_val_batch_data['label']
            # obs_mask_first = first_val_batch_data['obs_mask']
            #
            # print(f"\nOUTPUT SHAPES:")
            # print(f"  Logits: {logits_first.shape}")
            # print(f"  Labels: {label_first.shape}")
            # print(f"  Observation mask: {obs_mask_first.shape}")
            #
            # if is_timestep_task:
            #     print(f"\nFIRST SAMPLE (patient 0):")
            #     probs_first = F.softmax(logits_first, dim=-1)[0, :, 1]
            #     labels_first = label_first[0]
            #     obs_mask_first_sample = obs_mask_first[0]
            #
            #     valid_mask = obs_mask_first_sample.cpu().numpy()
            #     n_valid = valid_mask.sum()
            #
            #     print(f"  Total timesteps: {len(labels_first)}")
            #     print(f"  Valid timesteps: {n_valid}")
            #
            #     valid_timestep_indices = np.where(valid_mask)[0]
            #     n_show = min(20, len(valid_timestep_indices))
            #
            #     print(f"\n  First {n_show} valid timesteps:")
            #     print(f"  {'Timestep':<10} {'Label':<8} {'Prob(Sepsis)':<15} {'Logit[0]':<12} {'Logit[1]':<12}")
            #     print(f"  {'-'*10} {'-'*8} {'-'*15} {'-'*12} {'-'*12}")
            #
            #     for i in range(n_show):
            #         t_idx = valid_timestep_indices[i]
            #         label_val = labels_first[t_idx].item()
            #         prob_val = probs_first[t_idx].item()
            #         logit_0 = logits_first[0, t_idx, 0].item()
            #         logit_1 = logits_first[0, t_idx, 1].item()
            #         marker = "✓" if (prob_val > 0.5 and label_val == 1) or (prob_val <= 0.5 and label_val == 0) else "✗"
            #         print(f"  {t_idx:<10} {label_val:<8} {prob_val:<15.4f} {logit_0:<12.4f} {logit_1:<12.4f} {marker}")
            # else:
            #     print(f"\nFIRST SAMPLE (patient 0):")
            #     probs_first = F.softmax(logits_first, dim=-1)[0]
            #     label_first_val = label_first[0]
            #
            #     print(f"  Label: {label_first_val.item()}")
            #     print(f"  Logits: {logits_first[0].cpu().numpy()}")
            #     print(f"  Probabilities: {probs_first.cpu().numpy()}")
            #     print(f"  Predicted class: {1 if probs_first[1] > 0.5 else 0}")
            #
            # print("="*80 + "\n")
            if config.debug_pause:
                input("Press Enter to continue...")

        # Print validation prediction details every 5 epochs
        if epoch % 5 == 0 and val_patient_data is not None:
            # print("\n" + "-"*80)
            # print(f"VALIDATION PREDICTIONS SAMPLE (Epoch {epoch+1})")
            # print("-"*80)
            # print(f"Total validation samples: {len(all_val_labels)}")
            # print(f"Positive labels (sepsis): {sum(all_val_labels)}")
            # print(f"Negative labels (no sepsis): {len(all_val_labels) - sum(all_val_labels)}")
            # print(f"Prediction statistics:")
            # print(f"  Mean probability: {np.mean(all_val_probs):.4f}")
            # print(f"  Std probability: {np.std(all_val_probs):.4f}")
            # print(f"  Min probability: {np.min(all_val_probs):.4f}")
            # print(f"  Max probability: {np.max(all_val_probs):.4f}")
            #
            # positive_patients = [p for p in val_patient_data if p['has_positive']]
            # negative_patients = [p for p in val_patient_data if not p['has_positive']]
            #
            # print(f"\nPOSITIVE PATIENTS (with sepsis labels):")
            # n_show_pos = min(5, len(positive_patients))
            # for i in range(n_show_pos):
            #     patient = positive_patients[i]
            #     print(f"\n  Patient {i+1}:")
            #     labels_np = patient['labels'].numpy()
            #     probs_np = patient['probs'].numpy()
            #     print(f"    Labels:       {labels_np}")
            #     print(f"    Predictions:  {np.round(probs_np, 4)}")
            #     n_correct = ((probs_np > 0.5) == labels_np).sum()
            #     print(f"    Accuracy: {n_correct}/{len(labels_np)} ({100*n_correct/len(labels_np):.1f}%)")
            #
            # if n_show_pos == 0:
            #     print("  No patients with positive labels in validation set")
            #
            # print(f"\nNEGATIVE PATIENTS (no sepsis labels):")
            # n_show_neg = min(5, len(negative_patients))
            # for i in range(n_show_neg):
            #     patient = negative_patients[i]
            #     print(f"\n  Patient {i+1}:")
            #     labels_np = patient['labels'].numpy()
            #     probs_np = patient['probs'].numpy()
            #     print(f"    Labels:       {labels_np}")
            #     print(f"    Predictions:  {np.round(probs_np, 4)}")
            #     n_correct = ((probs_np > 0.5) == labels_np).sum()
            #     print(f"    Accuracy: {n_correct}/{len(labels_np)} ({100*n_correct/len(labels_np):.1f}%)")
            #
            # if n_show_neg == 0:
            #     print("  No patients with all negative labels in validation set")
            #
            # print("-"*80 + "\n")
            if config.debug_pause:
                input("Press Enter to continue...")

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

    # print("\n" + "="*80)
    # print("FINAL TEST PREDICTIONS SAMPLE")
    # print("="*80)
    # print(f"Total test samples: {len(all_test_labels)}")
    # print(f"Positive labels (sepsis): {sum(all_test_labels)}")
    # print(f"Negative labels (no sepsis): {len(all_test_labels) - sum(all_test_labels)}")
    # print(f"Prediction statistics:")
    # print(f"  Mean probability: {np.mean(all_test_probs):.4f}")
    # print(f"  Std probability: {np.std(all_test_probs):.4f}")
    # print(f"  Min probability: {np.min(all_test_probs):.4f}")
    # print(f"  Max probability: {np.max(all_test_probs):.4f}")
    #
    # test_labels_arr = np.array(all_test_labels)
    # test_probs_arr = np.array(all_test_probs)
    # pos_indices = np.where(test_labels_arr == 1)[0]
    # neg_indices = np.where(test_labels_arr == 0)[0]
    #
    # print(f"\nPOSITIVE EXAMPLES (SEPSIS):")
    # if len(pos_indices) > 0:
    #     n_show = min(15, len(pos_indices))
    #     for i in range(n_show):
    #         idx = pos_indices[i]
    #         marker = "✓" if test_probs_arr[idx] > 0.5 else "✗"
    #         print(f"  {marker} Label: 1, Pred prob: {test_probs_arr[idx]:.4f}")
    # else:
    #     print("  No positive labels in test set")
    #
    # print(f"\nNEGATIVE EXAMPLES (NO SEPSIS):")
    # if len(neg_indices) > 0:
    #     n_show = min(15, len(neg_indices))
    #     step = len(neg_indices) // n_show if n_show > 0 else 1
    #     for i in range(n_show):
    #         idx = neg_indices[i * step]
    #         marker = "✓" if test_probs_arr[idx] <= 0.5 else "✗"
    #         print(f"  {marker} Label: 0, Pred prob: {test_probs_arr[idx]:.4f}")
    #
    # print("="*80 + "\n")
    if config.debug_pause:
        input("Press Enter to continue...")

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


def parse_int_list(arg: str) -> List[int]:
    # supports "100,500,1000" or "100:1000:100" (start:stop:step) or single "1000"
    s = arg.strip()
    if ":" in s:
        start, stop, step = [int(x) for x in s.split(":")]
        return list(range(start, stop + (1 if step > 0 else -1), step))
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SSL_BAT on ICU subsets and aggregate results.")
    parser.add_argument("--debug_pause", action="store_true")
    parser.add_argument("--model_path", required=True, type=str, help="Path to pretrained checkpoint .ckpt")
    parser.add_argument("--model_type", required=True, choices=["bat", "grud"], help="Which pretrained SSL backbone to fine-tune")
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