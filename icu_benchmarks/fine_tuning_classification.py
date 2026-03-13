#!/usr/bin/env python
# finetune_bat.py
import os
import json
import csv
import hashlib
import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional

# --- Third-party / project imports ---
import gin
import torch
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


def parse_scalar(value: str) -> Any:
    """Parse CLI string into bool/int/float/None/JSON/str."""
    v = value.strip()

    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() == "none":
        return None

    if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
        try:
            return json.loads(v)
        except Exception:
            pass

    try:
        return int(v)
    except ValueError:
        pass

    try:
        return float(v)
    except ValueError:
        pass

    return v


def parse_kv_list(items: List[str]) -> Dict[str, Any]:
    """
    Parse repeated CLI args like:
      --model_hparam dropout=0.2 --model_hparam heads=8
    """
    result: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected key=value format, got: {item}")
        key, value = item.split("=", 1)
        result[key.strip()] = parse_scalar(value)
    return result


def inspect_checkpoint_hparams(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    if not hparams:
        print("[WARN] No hyper_parameters found in checkpoint.")
        return

    print("\n[INFO] hyper_parameters found in checkpoint:")
    for k in sorted(hparams.keys()):
        print(f"  {k}: {hparams[k]}")

    print("\n[INFO] CLI override examples:")
    for k in sorted(hparams.keys()):
        print(f"  --model_hparam {k}={hparams[k]}")
    print()


def resolve_training_params(
    ckpt_path: str,
    cli_lr: Optional[float] = None,
    cli_weight_decay: Optional[float] = None,
    cli_batch_size: Optional[int] = None,
) -> Tuple[float, float, int]:
    """
    Resolve training parameters with priority:
      CLI value > checkpoint hyper_parameters > hardcoded fallback

    For batch size, we try common names first. As a last resort, if the checkpoint
    stores `input_size` as something like torch.Size([B, ...]), we use input_size[0].
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    lr = cli_lr if cli_lr is not None else hparams.get("lr", 1e-3)
    weight_decay = (
        cli_weight_decay if cli_weight_decay is not None else hparams.get("weight_decay", 0.0)
    )

    if cli_batch_size is not None:
        batch_size = cli_batch_size
    else:
        batch_size = (
            hparams.get("batch_size")
            or hparams.get("bz")
            or hparams.get("train_batch_size")
            or hparams.get("batch_sz")
        )

        if batch_size is None:
            input_size = hparams.get("input_size")
            if isinstance(input_size, (tuple, list, torch.Size)) and len(input_size) > 0:
                try:
                    batch_size = int(input_size[0])
                    print(
                        f"[INFO] Recovering batch size from checkpoint input_size[0]: {batch_size}"
                    )
                except Exception:
                    batch_size = None

        if batch_size is None:
            batch_size = 32

    print(
        "[INFO] Resolved training params: "
        f"lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}"
    )
    return float(lr), float(weight_decay), int(batch_size)


def build_datasets(
    data: Dict[str, Dict[str, pl.DataFrame]]
) -> Tuple[BATPolarsDataset, BATPolarsDataset, BATPolarsDataset]:
    """Wrapper around shared build_datasets with classification mode."""
    return build_datasets_shared(data, runmode=RunMode.classification, vars_dict=VARS_DICT)


def build_model_from_ckpt(
    ckpt_path: Path,
    model_type: str,
    task: str = "Mortality24",
    model_hparams: Optional[Dict[str, Any]] = None,
):
    """
    Build model from checkpoint with optional CLI hyperparameter overrides.

    Notes:
    - If model_hparams is empty, checkpoint hyper_parameters are used unchanged.
    - If you override architecture-defining params (e.g. layers, heads, hidden sizes),
      checkpoint loading may fail or only partially load.
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

    if model_hparams:
        print("\n[INFO] Applying model hyperparameter overrides:")
        for k, v in model_hparams.items():
            old = hparams.get(k, "<MISSING>")
            print(f"  {k}: {old} -> {v}")
            hparams[k] = v
        print()

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
        if missing or unexpected:
            print(
                f"[WARN] Timestep encoder load_state_dict(strict=False): "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

        model = wrapper_class(
            encoder_class=encoder,
            prediction_head=TimeseriesClassificationHead,
            prediction_head_kwargs={"num_classes": 2},
        )
        return model

    ssl_model = ssl_class(**hparams)

    try:
        ssl_model.model.encoder_class.load_state_dict(encoder_state_dict, strict=True)
    except RuntimeError as e:
        print(f"[WARN] Strict encoder weight loading failed: {e}")
        print("[WARN] Retrying with strict=False. Some weights may remain randomly initialized.")
        missing, unexpected = ssl_model.model.encoder_class.load_state_dict(
            encoder_state_dict, strict=False
        )
        print(
            f"[WARN] Encoder load_state_dict(strict=False): "
            f"missing={len(missing)} unexpected={len(unexpected)}"
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
    num_epochs: int
    patience: int
    subset_root: str
    output_dir: str
    task: str = "Mortality24"
    debug_pause: bool = False
    model_hparams: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    dataset: str
    size: int
    seed: int
    batch_size: int
    lr: float
    weight_decay: float
    num_epochs: int
    fine_tune_head: bool
    model_path: str
    model_hparams: Dict[str, Any]
    avg_test_loss: float
    test_auroc: float
    test_auprc: float


def safe_roc_auc_score(y_true: List[int], y_score: List[float]) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def safe_average_precision_score(y_true: List[int], y_score: List[float]) -> float:
    if len(set(y_true)) < 2:
        # average_precision_score can still run sometimes, but this avoids noisy crashes/behavior
        return float("nan")
    return average_precision_score(y_true, y_score)


def train_eval_one(config: RunConfig) -> RunResult:
    # Fixed seed for training procedure: subset variability stays in the data split seed
    set_seeds(42)

    subset_path = Path(config.subset_root) / config.task / config.dataset / f"{config.size}_{config.seed}"
    data = load_subset_as_data_dict(subset_path)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TIMESTEP_TASKS = {"Sepsis"}
    is_timestep_task = config.task in TIMESTEP_TASKS

    model = build_model_from_ckpt(
        Path(config.model_path),
        model_type=config.model_type,
        task=config.task,
        model_hparams=config.model_hparams,
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_auprc = -float("inf")
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

            if is_timestep_task:
                # logits: (B, T, num_classes), label: (B, T), obs_mask: (B, T)
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
                # Patient-level prediction
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
        train_auroc = safe_roc_auc_score(all_train_labels, all_train_probs)
        train_auprc = safe_average_precision_score(all_train_labels, all_train_probs)

        # VAL
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
        val_auroc = safe_roc_auc_score(all_val_labels, all_val_probs)
        val_auprc = safe_average_precision_score(all_val_labels, all_val_probs)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={avg_train_loss:.4f} auroc={train_auroc:.4f} auprc={train_auprc:.4f} | "
            f"val_loss={avg_val_loss:.4f} auroc={val_auroc:.4f} auprc={val_auprc:.4f}"
        )

        scheduler.step()

        # Early stopping by AUPRC
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after {config.patience} epochs without val AUPRC improvement.")
                break

    # Restore best
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
                all_test_labels.extend(valid_labels.cpu().numpy().tolist())
                all_test_probs.extend(valid_probs.cpu().numpy().tolist())
            else:
                if label.dim() > 1:
                    label = label[:, -1]

                loss = loss_fn(logits, label)
                probs = F.softmax(logits, dim=1)[:, 1]
                all_test_labels.extend(label.cpu().numpy().tolist())
                all_test_probs.extend(probs.cpu().numpy().tolist())

            total_test_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    avg_test_loss = total_test_loss / max(1, len(test_loader))
    test_auroc = safe_roc_auc_score(all_test_labels, all_test_probs)
    test_auprc = safe_average_precision_score(all_test_labels, all_test_probs)

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
        weight_decay=config.weight_decay,
        num_epochs=config.num_epochs,
        fine_tune_head=config.fine_tune_head,
        model_path=config.model_path,
        model_hparams=config.model_hparams,
        avg_test_loss=avg_test_loss,
        test_auroc=test_auroc,
        test_auprc=test_auprc,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained SSL models on ICU subsets and aggregate results."
    )
    parser.add_argument("--debug_pause", action="store_true")
    parser.add_argument("--model_path", required=True, type=str, help="Path to pretrained checkpoint .ckpt")
    parser.add_argument(
        "--model_type",
        required=True,
        choices=["bat", "grud", "radv_transformer", "itransformer", "ipnets", "seft"],
        help="Which pretrained SSL backbone to fine-tune",
    )
    parser.add_argument("--inspect_hparams", action="store_true", help="Print checkpoint hyper_parameters and exit")
    parser.add_argument("--dataset", default="mimic", type=str, help="Dataset name (eicu, miiv, mimic, hirid, or custom)")
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
    parser.add_argument("--bz", default=None, type=int, help="Batch size. If omitted, recover from checkpoint if possible.")
    parser.add_argument("--lr", default=None, type=float, help="Learning rate. If omitted, recover from checkpoint if possible.")
    parser.add_argument(
        "--weight_decay",
        default=None,
        type=float,
        help="Adam weight decay. If omitted, recover from checkpoint if possible.",
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
    parser.add_argument(
        "--model_hparam",
        action="append",
        default=[],
        help='Model-specific override in key=value form. Repeatable, e.g. '
             '--model_hparam dropout=0.1 --model_hparam heads=4',
    )

    args = parser.parse_args()

    if args.inspect_hparams:
        inspect_checkpoint_hparams(args.model_path)
        return

    sizes = parse_int_list(args.sizes)
    seeds = parse_int_list(args.seeds)
    model_hparams = parse_kv_list(args.model_hparam)

    patience = args.patience if args.patience is not None else 3

    resolved_lr, resolved_weight_decay, resolved_bz = resolve_training_params(
        ckpt_path=args.model_path,
        cli_lr=args.lr,
        cli_weight_decay=args.weight_decay,
        cli_batch_size=args.bz,
    )

    mode_str = "head" if args.fine_tune_head else "full"

    output_dir = Path(f"finetuning_results/pretrained_{args.model_type.upper()}/{args.dataset}/{mode_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_id = hashlib.md5(
        json.dumps(
            {
                "model_type": args.model_type,
                "model_path": args.model_path,
                "dataset": args.dataset,
                "task": args.task,
                "sizes": sizes,
                "seeds": seeds,
                "fine_tune_head": args.fine_tune_head,
                "bz": resolved_bz,
                "lr": resolved_lr,
                "weight_decay": resolved_weight_decay,
                "num_epochs": args.num_epochs,
                "subset_root": args.subset_root,
                "model_hparams": model_hparams,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:10]

    per_run_log = output_dir / f"runs_{sweep_id}.jsonl"
    csv_path = output_dir / f"summary_{sweep_id}.csv"

    with (output_dir / f"meta_{sweep_id}.json").open("w") as f:
        json.dump(
            {
                "sweep_id": sweep_id,
                "args": vars(args),
                "resolved_training_params": {
                    "lr": resolved_lr,
                    "weight_decay": resolved_weight_decay,
                    "batch_size": resolved_bz,
                },
                "sizes": sizes,
                "seeds": seeds,
                "model_hparams": model_hparams,
            },
            f,
            indent=2,
            default=str,
        )

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
                batch_size=resolved_bz,
                lr=resolved_lr,
                weight_decay=resolved_weight_decay,
                num_epochs=args.num_epochs,
                patience=patience,
                subset_root=args.subset_root,
                output_dir=str(output_dir),
                task=args.task,
                debug_pause=bool(args.debug_pause),
                model_hparams=model_hparams,
            )
            try:
                result = train_eval_one(run_cfg)
            except Exception as e:
                print(f"[ERROR] size={size} seed={seed}: {e}")
                continue

            all_results.append(result)
            with per_run_log.open("a") as f:
                f.write(json.dumps(asdict(result), default=str) + "\n")

    if all_results:
        rows = []
        for r in all_results:
            d = asdict(r)
            d["model_hparams"] = json.dumps(d["model_hparams"], sort_keys=True)
            rows.append(d)

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n✅ Wrote summary CSV: {csv_path}")
        print(f"🧾 Per-run JSONL:     {per_run_log}")
    else:
        print("\nNo successful runs to summarize.")


if __name__ == "__main__":
    main()