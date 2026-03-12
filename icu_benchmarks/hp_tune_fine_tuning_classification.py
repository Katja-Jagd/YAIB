#!/usr/bin/env python
# finetune_bat.py
import os
import json
import math
import hashlib
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

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
from icu_benchmarks.models.dl_models.deep_set_attention import (
    SSL_DeepSetAttention,
    DeepSetAttentionEncoderPrediction,
)

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
        "supports_timestep_tasks": True,
    },
    "grud": {
        "ssl_class": SSL_GRUD,
        "prediction_wrapper": GRUDEncoderPrediction,
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

    tasks_dir = p.parent
    configs_dir = tasks_dir.parent
    repo_root = configs_dir.parent

    gin.add_config_file_search_path(str(tasks_dir))
    gin.add_config_file_search_path(str(configs_dir))
    gin.add_config_file_search_path(str(repo_root))
    gin.add_config_file_search_path(str(repo_root / "configs"))
    gin.add_config_file_search_path(str(repo_root / "configs" / "tasks"))

    try:
        os.chdir(str(repo_root))
    except Exception:
        pass

    include_probe = repo_root / "configs" / "tasks" / "common" / "Imports.gin"
    if not include_probe.exists():
        print(f"[WARN] Could not find expected include at: {include_probe}")
        print("[WARN] Current gin search paths:")
        for sp in gin.config._CONFIG_DIR:  # type: ignore[attr-defined]
            print("       -", sp)

    gin.parse_config_file(str(p))


def build_datasets(
    data: Dict[str, Dict[str, pl.DataFrame]]
) -> Tuple[BATPolarsDataset, BATPolarsDataset, BATPolarsDataset]:
    """Wrapper around shared build_datasets with classification mode."""
    return build_datasets_shared(data, runmode=RunMode.classification, vars_dict=VARS_DICT)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def sample_log_uniform(rng: random.Random, low: float, high: float) -> float:
    """Sample from a log-uniform distribution on [low, high]."""
    if low <= 0 or high <= 0:
        raise ValueError(f"log-uniform bounds must be > 0, got low={low}, high={high}")
    if low > high:
        raise ValueError(f"log-uniform requires low <= high, got low={low}, high={high}")
    return 10 ** rng.uniform(math.log10(low), math.log10(high))


def build_model_from_ckpt(
    ckpt_path: Path,
    model_type: str,
    task: str = "Mortality24",
    dropout_override: Optional[float] = None,
    attn_dropout_override: Optional[float] = None,
):
    """
    Build model from checkpoint with task-appropriate prediction head.

    Supports overriding dropout-related hyperparameters before SSL model reconstruction.
    """
    TIMESTEP_TASKS = {"Sepsis", "AKI"}

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(MODEL_REGISTRY.keys())}")

    is_timestep_task = task in TIMESTEP_TASKS

    if is_timestep_task and not MODEL_REGISTRY[model_type]["supports_timestep_tasks"]:
        raise NotImplementedError(
            f"Task '{task}' is treated as timestep-level, but model_type='{model_type}' "
            f"is not configured for timestep-level heads in this script."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {}).copy()

    if dropout_override is not None and "dropout" in hparams:
        hparams["dropout"] = dropout_override
    if attn_dropout_override is not None and "attn_dropout" in hparams:
        hparams["attn_dropout"] = attn_dropout_override

    ssl_class = MODEL_REGISTRY[model_type]["ssl_class"]
    wrapper_class = MODEL_REGISTRY[model_type]["prediction_wrapper"]

    encoder_state_dict = {
        k.replace("model.encoder_class.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.encoder_class.")
    }

    if is_timestep_task:
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
        print(
            f"[INFO] Loaded timestep encoder weights: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"(dropout={hparams.get('dropout')}, attn_dropout={hparams.get('attn_dropout')})"
        )

        model = wrapper_class(
            encoder_class=encoder,
            prediction_head=TimeseriesClassificationHead,
            prediction_head_kwargs={"num_classes": 2},
        )
        return model

    ssl_model = ssl_class(**hparams)
    ssl_model.model.encoder_class.load_state_dict(encoder_state_dict)

    print(
        f"[INFO] Loaded pretrained encoder for model_type={model_type} "
        f"(dropout={hparams.get('dropout', 'n/a')}, attn_dropout={hparams.get('attn_dropout', 'n/a')})"
    )

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
    weight_decay: float
    dropout: float
    attn_dropout: float
    num_epochs: int
    patience: int
    subset_root: str
    output_dir: str
    task: str = "Mortality24"
    debug_pause: bool = False
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
    best_val_loss: float
    best_val_auroc: float
    best_val_auprc: float
    best_epoch: int


def safe_binary_metrics(labels: List[int], probs: List[float]) -> Tuple[float, float]:
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(labels, probs), average_precision_score(labels, probs)


def train_eval_one(config: RunConfig) -> RunResult:
    # Keep subset variability only; make training procedure deterministic
    set_seeds(42)

    subset_path = Path(config.subset_root) / config.task / config.dataset / f"{config.size}_{config.seed}"
    data = load_subset_as_data_dict(subset_path)

    train_set, val_set, _ = build_datasets(data)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TIMESTEP_TASKS = {"Sepsis", "AKI"}
    is_timestep_task = config.task in TIMESTEP_TASKS

    model = build_model_from_ckpt(
        Path(config.model_path),
        model_type=config.model_type,
        task=config.task,
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
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(config.num_epochs):
        # -------------------------
        # Train
        # -------------------------
        model.train()
        total_train_loss = 0.0
        all_train_labels: List[int] = []
        all_train_probs: List[float] = []

        pbar = tqdm(
            train_loader,
            desc=f"Sweep {config.sweep_idx} | Epoch {epoch+1}/{config.num_epochs} (train)"
        )

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

            if is_timestep_task:
                B, T, C = logits.shape
                logits_flat = logits.reshape(B * T, C)
                label_flat = label.reshape(B * T)
                obs_mask_flat = obs_mask.reshape(B * T)

                if obs_mask_flat.sum() > 0:
                    loss = loss_fn(logits_flat[obs_mask_flat], label_flat[obs_mask_flat])
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

                probs = F.softmax(logits, dim=-1)[:, :, 1]
                valid_probs = probs[obs_mask]
                valid_labels = label[obs_mask]

                if epoch == 0 and batch_idx == 1 and config.debug_pause:
                    input("Press Enter to continue...")

                all_train_labels.extend(valid_labels.detach().cpu().numpy().tolist())
                all_train_probs.extend(valid_probs.detach().cpu().numpy().tolist())

            else:
                if label.dim() > 1:
                    label = label[:, -1]

                loss = loss_fn(logits, label)
                probs = F.softmax(logits, dim=1)[:, 1]

                all_train_labels.extend(label.detach().cpu().numpy().tolist())
                all_train_probs.extend(probs.detach().cpu().numpy().tolist())

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_auroc, train_auprc = safe_binary_metrics(all_train_labels, all_train_probs)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        total_val_loss = 0.0
        all_val_labels: List[int] = []
        all_val_probs: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                x, mask, label, times, static, delta, obs_mask = batch
                x = x.to(device).float()
                mask = mask.to(device).float()
                times = times.to(device).float()
                static = static.to(device).float()
                label = label.to(device).long()
                obs_mask = obs_mask.to(device).bool()

                logits = model(x, static=static, time=times, sensor_mask=mask)

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
                    all_val_labels.extend(valid_labels.cpu().numpy().tolist())
                    all_val_probs.extend(valid_probs.cpu().numpy().tolist())

                else:
                    if label.dim() > 1:
                        label = label[:, -1]

                    loss = loss_fn(logits, label)
                    probs = F.softmax(logits, dim=1)[:, 1]
                    all_val_labels.extend(label.cpu().numpy().tolist())
                    all_val_probs.extend(probs.cpu().numpy().tolist())

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_auroc, val_auprc = safe_binary_metrics(all_val_labels, all_val_probs)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={avg_train_loss:.4f} auroc={train_auroc:.4f} auprc={train_auprc:.4f} | "
            f"val_loss={avg_val_loss:.4f} auroc={val_auroc:.4f} auprc={val_auprc:.4f}"
        )

        scheduler.step()

        if not np.isnan(val_auprc) and val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_val_auroc = val_auroc
            best_val_loss = avg_val_loss
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

    print(
        "\nBEST VAL RESULTS "
        f"(dataset={config.dataset}, task={config.task}, size={config.size}, "
        f"seed={config.seed}, sweep={config.sweep_idx}): "
        f"epoch={best_epoch} loss={best_val_loss:.4f} "
        f"auroc={best_val_auroc:.4f} auprc={best_val_auprc:.4f}"
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
        best_val_loss=best_val_loss,
        best_val_auroc=best_val_auroc,
        best_val_auprc=best_val_auprc,
        best_epoch=best_epoch,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SSL models on ICU subsets with validation-only random hyperparameter sweeps."
    )

    parser.add_argument("--debug_pause", action="store_true")
    parser.add_argument("--model_path", required=True, type=str, help="Path to pretrained checkpoint .ckpt")
    parser.add_argument(
        "--model_type",
        required=True,
        choices=["bat", "grud", "radv_transformer", "itransformer", "ipnets", "seft"],
        help="Which pretrained SSL backbone to fine-tune",
    )
    parser.add_argument("--dataset", default="mimic", type=str, help="Dataset name (eicu, miiv, mimic, or custom)")
    parser.add_argument(
        "--task",
        default="Mortality24",
        type=str,
        choices=["Mortality24", "Sepsis", "AKI", "Mortality"],
        help="Task name: determines prediction head and label handling",
    )
    parser.add_argument("--sizes", default="9506", type=str, help='e.g. "100,500,1000" or "100:9000:100"')
    parser.add_argument("--seeds", default="42", type=str, help='e.g. "42,84,126"')
    parser.add_argument("--fine_tune_head", action="store_true", help="Only fine-tune the classification head")
    parser.add_argument("--bz", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="Fixed learning rate. Ignored unless --use_fixed_lr is set.",
    )
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument(
        "--patience",
        default=None,
        type=int,
        help="Early stopping patience (epochs without improvement). Default: 3",
    )
    parser.add_argument(
        "--subset_root",
        default="icu_benchmarks/data/preprocessed_data",
        type=str,
        help="Root path that contains {task}/{dataset}/{size}_{seed}/ parquet files",
    )

    # Random sweep args
    parser.add_argument("--num_sweeps", type=int, default=1, help="Number of random hyperparameter sweeps to run")
    parser.add_argument("--sweep_seed", type=int, default=123, help="Random seed for hyperparameter sampling")
    parser.add_argument("--use_fixed_lr", action="store_true", help="Use --lr as fixed LR for all sweeps")

    # Dropout categorical defaults
    parser.add_argument("--attn_dropout_choices", type=str, default="0,0.2,0.4,0.6")
    parser.add_argument("--dropout_choices", type=str, default="0,0.2,0.4,0.6")

    # Log-uniform defaults
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=1e-2)
    parser.add_argument("--weight_decay_min", type=float, default=1e-4)
    parser.add_argument("--weight_decay_max", type=float, default=1e-1)

    args = parser.parse_args()

    sizes = parse_int_list(args.sizes)
    seeds = parse_int_list(args.seeds)
    patience = args.patience if args.patience is not None else 3
    mode_str = "head" if args.fine_tune_head else "full"

    output_dir = Path(
        f"finetuning_results/pretrained_{args.model_type.upper()}/{args.task}/{args.dataset}/{mode_str}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    attn_dropout_choices = parse_float_list(args.attn_dropout_choices)
    dropout_choices = parse_float_list(args.dropout_choices)

    rng = random.Random(args.sweep_seed)

    if args.num_sweeps <= 0:
        raise ValueError("--num_sweeps must be >= 1")

    if args.use_fixed_lr and args.lr is None:
        raise ValueError("--use_fixed_lr requires --lr to be provided.")

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
        "patience": patience,
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
                    model_type=args.model_type,
                    size=size,
                    seed=seed,
                    model_path=args.model_path,
                    fine_tune_head=bool(args.fine_tune_head),
                    batch_size=args.bz,
                    lr=hp["lr"],
                    weight_decay=hp["weight_decay"],
                    dropout=hp["dropout"],
                    attn_dropout=hp["attn_dropout"],
                    num_epochs=args.num_epochs,
                    patience=patience,
                    subset_root=args.subset_root,
                    output_dir=str(output_dir),
                    task=args.task,
                    debug_pause=bool(args.debug_pause),
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