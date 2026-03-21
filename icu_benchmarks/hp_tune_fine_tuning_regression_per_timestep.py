#!/usr/bin/env python
# finetune_regression_timestep_full.py

import os
import json
import math
import hashlib
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple, Optional

import gin
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from icu_benchmarks.cross_validation import execute_repeated_cv
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.loader import BATPolarsDataset
from icu_benchmarks.data.constants import DataSplit as Split

from icu_benchmarks.models.dl_models.bat import (
    SSL_BAT,
    AutoregressiveEncoderCrossParallel,
    EncoderPrediction,
    RegressionHead,
)
from icu_benchmarks.models.dl_models.grud import (
    SSL_GRUD,
    GRUDEncoder,
    GRUDEncoderPrediction,
)
from icu_benchmarks.models.dl_models.radv_transformer import SSL_RadVTransformer
from icu_benchmarks.models.dl_models.itransformer import (
    SSL_iTransformer,
    EncoderPredictionInverted,
)
from icu_benchmarks.models.dl_models.ip_nets import (
    SSL_IPNets,
    IPNetsEncoderPrediction,
)
from icu_benchmarks.models.dl_models.deep_set_attention import (
    SSL_DeepSetAttention,
    DeepSetAttentionEncoderPrediction,
)

from icu_benchmarks.fine_tuning_utils import set_seeds
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.run_utils import get_task_gin_and_name


MODEL_REGISTRY = {
    "bat": {
        "ssl_class": SSL_BAT,
        "prediction_wrapper": EncoderPrediction,
        "supports_timestep_tasks": True,
    },
    "grud": {
        "ssl_class": SSL_GRUD,
        "prediction_wrapper": GRUDEncoderPrediction,
        "supports_timestep_tasks": True,
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


def parse_gin_config(gin_path: str):
    """Parse a gin config and make relative include paths work."""
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


def assure_minimum_length(dataset):
    if len(dataset) < 2:
        return [dataset[0], dataset[0]]
    return dataset


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def sample_log_uniform(rng: random.Random, low: float, high: float) -> float:
    """Sample from a log-uniform distribution on [low, high]."""
    if low <= 0 or high <= 0:
        raise ValueError(f"log-uniform bounds must be > 0, got low={low}, high={high}")
    if low > high:
        raise ValueError(f"log-uniform requires low <= high, got low={low}, high={high}")
    return 10 ** rng.uniform(math.log10(low), math.log10(high))


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    if y_true.size == 0:
        return float("nan"), float("nan")
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae


def build_model_from_ckpt(
    ckpt_path: Path,
    model_type: str,
    task: str,
    dropout_override: Optional[float] = None,
    attn_dropout_override: Optional[float] = None,
):
    """
    Build model from checkpoint with task-appropriate prediction head.

    This script supports timestep regression tasks only.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(MODEL_REGISTRY.keys())}")

    is_timestep_task = True

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

    wrapper_class = MODEL_REGISTRY[model_type]["prediction_wrapper"]

    encoder_state_dict = {
        k.replace("model.encoder_class.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.encoder_class.")
    }

    if model_type == "bat":
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

    elif model_type == "grud":
        sensors_count = hparams["input_size"][1]
        max_timepoint_count = hparams["input_size"][2]
        static_count = hparams.get("static_count", 4)

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

        missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
        print(
            f"[INFO] Loaded timestep encoder weights: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"(dropout={hparams.get('dropout')}, recurrent_dropout={hparams.get('recurrent_dropout', 'n/a')})"
        )

    else:
        raise NotImplementedError(
            f"Timestep regression is not implemented for model_type='{model_type}' in this script."
        )

    model = wrapper_class(
        encoder_class=encoder,
        prediction_head=RegressionHead,
        prediction_head_kwargs={"output_dim": 1},
    )
    return model


@gin.configurable("Run")
def get_mode(mode: gin.REQUIRED):
    assert RunMode(mode)
    return RunMode(mode)


@dataclass
class RunConfig:
    data_dir: str
    dataset: str
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
    output_dir: str
    task: str
    debug_pause: bool = False
    sweep_idx: int = 0

    split_seed: int = 2222
    cv_repetitions: int = 1
    cv_folds: int = 5
    fold_index: int = 0
    repetition_index: int = 0
    debug: bool = False
    load_cache: bool = False
    generate_cache: bool = False


@dataclass
class RunResult:
    dataset: str
    task: str
    sweep_idx: int
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float
    attn_dropout: float
    num_epochs: int
    fine_tune_head: bool
    model_path: str
    data_dir: str
    best_val_loss: float
    best_val_rmse: float
    best_val_mae: float
    best_epoch: int


def train_eval_one(config: RunConfig) -> RunResult:
    set_seeds(42)

    data = preprocess_data(
        Path(config.data_dir),
        seed=config.split_seed,
        debug=config.debug,
        load_cache=config.load_cache,
        generate_cache=config.generate_cache,
        cv_repetitions=config.cv_repetitions,
        repetition_index=config.repetition_index,
        train_size=None,
        cv_folds=config.cv_folds,
        fold_index=config.fold_index,
        pretrained_imputation_model=None,
        runmode=RunMode.regression,
        complete_train=False,
    )

    train_set = BATPolarsDataset(data, split=Split.train, ram_cache=False, name=f"{config.dataset}_train")
    val_set = BATPolarsDataset(data, split=Split.val, ram_cache=False, name=f"{config.dataset}_val")

    train_collate = train_set.collate_fn_pad_to_longest_in_batch()
    val_collate = val_set.collate_fn_pad_to_longest_in_batch()

    train_set = assure_minimum_length(train_set)
    val_set = assure_minimum_length(val_set)

    if config.debug_pause:
        input("Press Enter to continue...")

    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        generator=g,
        drop_last=True,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=val_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    loss_fn = torch.nn.MSELoss()

    best_val_rmse = float("inf")
    best_val_mae = float("inf")
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0.0
        all_train_labels: List[float] = []
        all_train_preds: List[float] = []

        pbar = tqdm(
            train_loader,
            desc=f"Sweep {config.sweep_idx} | Epoch {epoch+1}/{config.num_epochs} (train)"
        )

        for batch in pbar:
            x, mask, label, times, static, delta, obs_mask = batch
            x = x.to(device).float()
            mask = mask.to(device).float()
            times = times.to(device).float()
            static = static.to(device).float()
            label = label.to(device).float()
            obs_mask = obs_mask.to(device).bool()

            optimizer.zero_grad()

            try:
                pred = model(x, static=static, time=times, sensor_mask=mask)
            except Exception as e:
                print(f"\n[ERROR] Model forward pass failed: {e}")
                print(f"Input shapes: x={x.shape}, static={static.shape}, times={times.shape}, mask={mask.shape}")
                raise

            if pred.dim() == 3 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)

            if pred.dim() != 2:
                raise ValueError(f"Expected pred (B,T), got {pred.shape}")
            if label.dim() != 2:
                raise ValueError(f"Expected label (B,T), got {label.shape}")

            valid_pred = pred[obs_mask]
            valid_label = label[obs_mask]

            if valid_label.numel() > 0:
                loss = loss_fn(valid_pred, valid_label)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

            all_train_labels.extend(valid_label.detach().cpu().numpy().tolist())
            all_train_preds.extend(valid_pred.detach().cpu().numpy().tolist())

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_rmse, train_mae = rmse_mae(np.array(all_train_labels), np.array(all_train_preds))

        model.eval()
        total_val_loss = 0.0
        all_val_labels: List[float] = []
        all_val_preds: List[float] = []

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
                    raise ValueError(f"Expected pred (B,T), got {pred.shape}")
                if label.dim() != 2:
                    raise ValueError(f"Expected label (B,T), got {label.shape}")

                valid_pred = pred[obs_mask]
                valid_label = label[obs_mask]

                if valid_label.numel() > 0:
                    loss = loss_fn(valid_pred, valid_label)
                    total_val_loss += loss.item()

                all_val_labels.extend(valid_label.detach().cpu().numpy().tolist())
                all_val_preds.extend(valid_pred.detach().cpu().numpy().tolist())

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_rmse, val_mae = rmse_mae(np.array(all_val_labels), np.array(all_val_preds))

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={avg_train_loss:.4f} rmse={train_rmse:.4f} mae={train_mae:.4f} | "
            f"val_loss={avg_val_loss:.4f} rmse={val_rmse:.4f} mae={val_mae:.4f}"
        )

        scheduler.step()

        if not np.isnan(val_rmse) and val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_mae = val_mae
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after {config.patience} epochs without val RMSE improvement.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(
        "\nBEST VAL RESULTS "
        f"(dataset={config.dataset}, task={config.task}, sweep={config.sweep_idx}): "
        f"epoch={best_epoch} loss={best_val_loss:.4f} "
        f"rmse={best_val_rmse:.4f} mae={best_val_mae:.4f}"
    )

    return RunResult(
        dataset=config.dataset,
        task=config.task,
        sweep_idx=config.sweep_idx,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        num_epochs=config.num_epochs,
        fine_tune_head=config.fine_tune_head,
        model_path=config.model_path,
        data_dir=config.data_dir,
        best_val_loss=best_val_loss,
        best_val_rmse=best_val_rmse,
        best_val_mae=best_val_mae,
        best_epoch=best_epoch,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SSL models on full ICU datasets for timestep regression with validation-only random hyperparameter sweeps."
    )
    parser.add_argument("--debug_pause", action="store_true")
    parser.add_argument("--model_path", required=True, type=str, help="Path to pretrained checkpoint .ckpt")
    parser.add_argument(
        "--model_type",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Which pretrained SSL backbone to fine-tune",
    )
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        help="Task name for timestep regression",
    )
    parser.add_argument("--fine_tune_head", action="store_true", help="Only fine-tune the regression head")
    parser.add_argument("--bz", default=32, type=int, help="Batch size")
    parser.add_argument("--lr", default=None, type=float, help="Fixed learning rate. Ignored unless --use_fixed_lr is set.")
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--patience", default=50, type=int, help="Early stopping patience (epochs without improvement).")

    parser.add_argument(
        "-d",
        "--data_dir",
        required=True,
        type=str,
        help="Path to the full/raw dataset directory",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="Dataset name for logging only. If omitted, inferred from data_dir name.",
    )
    parser.add_argument("--split_seed", default=2222, type=int, help="Seed used for full-dataset split generation")
    parser.add_argument("--cv_repetitions", default=1, type=int, help="Number of CV repetitions used by preprocess_data")
    parser.add_argument("--cv_folds", default=5, type=int, help="Number of CV folds used by preprocess_data")
    parser.add_argument("--fold_index", default=0, type=int, help="Which fold to use")
    parser.add_argument("--repetition_index", default=0, type=int, help="Which CV repetition to use")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--load_cache", action="store_true")
    parser.add_argument("--generate_cache", action="store_true")

    parser.add_argument("--num_sweeps", type=int, default=1, help="Number of random hyperparameter sweeps to run")
    parser.add_argument("--sweep_seed", type=int, default=123, help="Random seed for hyperparameter sampling")
    parser.add_argument("--use_fixed_lr", action="store_true", help="Use --lr as fixed LR for all sweeps")

    parser.add_argument("--attn_dropout_choices", type=str, default="0,0.2,0.4,0.6")
    parser.add_argument("--dropout_choices", type=str, default="0,0.2,0.4,0.6")

    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=1e-2)
    parser.add_argument("--weight_decay_min", type=float, default=1e-4)
    parser.add_argument("--weight_decay_max", type=float, default=1e-1)

    args = parser.parse_args()
    dataset_name = args.dataset if args.dataset is not None else Path(args.data_dir).resolve().name
    patience = args.patience

    task_gin, task_name = get_task_gin_and_name(args.task)
    gin.parse_config_file(f"configs/tasks/{task_gin}.gin")

    mode_str = "head" if args.fine_tune_head else "full"

    output_dir = Path(
        f"finetuning_results/pretrained_{args.model_type.upper()}/{args.task}/{dataset_name}/{mode_str}"
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
        "dataset": dataset_name,
        "data_dir": args.data_dir,
        "task": args.task,
        "fine_tune_head": args.fine_tune_head,
        "bz": args.bz,
        "num_epochs": args.num_epochs,
        "patience": patience,
        "split_seed": args.split_seed,
        "cv_repetitions": args.cv_repetitions,
        "cv_folds": args.cv_folds,
        "fold_index": args.fold_index,
        "repetition_index": args.repetition_index,
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
                "dataset": dataset_name,
                "data_dir": args.data_dir,
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

        run_cfg = RunConfig(
            data_dir=args.data_dir,
            dataset=dataset_name,
            model_type=args.model_type,
            model_path=args.model_path,
            fine_tune_head=bool(args.fine_tune_head),
            batch_size=args.bz,
            lr=hp["lr"],
            weight_decay=hp["weight_decay"],
            dropout=hp["dropout"],
            attn_dropout=hp["attn_dropout"],
            num_epochs=args.num_epochs,
            patience=patience,
            output_dir=str(output_dir),
            task=args.task,
            debug_pause=bool(args.debug_pause),
            sweep_idx=sweep_idx,
            split_seed=args.split_seed,
            cv_repetitions=args.cv_repetitions,
            cv_folds=args.cv_folds,
            fold_index=args.fold_index,
            repetition_index=args.repetition_index,
            debug=bool(args.debug),
            load_cache=bool(args.load_cache),
            generate_cache=bool(args.generate_cache),
        )

        try:
            result = train_eval_one(run_cfg)
        except Exception as e:
            print(
                f"[ERROR] sweep={sweep_idx} "
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

        valid_results = [r for r in all_results if not np.isnan(r.best_val_rmse)]
        best_result = min(valid_results, key=lambda r: r.best_val_rmse, default=None)

        if best_result is not None:
            print("\n🏆 Best run by validation RMSE:")
            print(json.dumps(asdict(best_result), indent=2))
    else:
        print("\nNo successful runs to summarize.")


if __name__ == "__main__":
    main()