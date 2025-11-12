#!/usr/bin/env python
# fine_tuning_regression.py - Fine-tuning script for regression tasks (e.g., Length of Stay)
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

# ICU Benchmarks (your repo)
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.loader import BATPolarsDataset
from icu_benchmarks.models.dl_models.bat import SSL_BAT, EncoderPrediction, RegressionHead

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
    train_set = BATPolarsDataset(data=data, split="train", ram_cache=False, runmode=RunMode.regression, vars=VARS_DICT)
    val_set   = BATPolarsDataset(data=data, split="val",   ram_cache=False, runmode=RunMode.regression, vars=VARS_DICT)
    test_set  = BATPolarsDataset(data=data, split="test",  ram_cache=False, runmode=RunMode.regression, vars=VARS_DICT)
    return train_set, val_set, test_set


def build_model_from_ckpt(ckpt_path: Path) -> EncoderPrediction:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    # Instantiate SSL_BAT
    model = SSL_BAT(**hparams)

    # Load only encoder weights
    encoder_state_dict = {
        k.replace("model.encoder_class.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.encoder_class.")
    }
    model.model.encoder_class.load_state_dict(encoder_state_dict, strict=False)

    # Build regression wrapper with pretrained encoder
    regression_model = EncoderPrediction(
        encoder_class=model.model.encoder_class,
        prediction_head=RegressionHead,
        prediction_head_kwargs={"output_dim": 1},
    )
    return regression_model


@dataclass
class RunConfig:
    dataset: str
    size: int
    seed: int
    model_path: str
    fine_tune_head: bool
    batch_size: int
    lr: float
    num_epochs: int
    subset_root: str
    output_dir: str
    gin_config: str = ""  # optional; leave empty to skip


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
    test_mse: float
    test_mae: float
    test_r2: float


def train_eval_one(config: RunConfig) -> RunResult:
    # optional gin
    if config.gin_config:
        parse_gin_config(config.gin_config)

    # fixed seed for training procedure
    set_seeds(42)

    # data paths
    subset_path = Path(config.subset_root) / config.dataset / f"{config.size}_{config.seed}"
    data = load_subset_as_data_dict(subset_path)

    # datasets & loaders
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

    # model
    model = build_model_from_ckpt(Path(config.model_path))
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
    loss_fn = torch.nn.MSELoss()

    patience = 3
    best_val_mse = float('inf')
    epochs_without_improvement = 0
    best_state = None

    # --------- Training loop ----------
    for epoch in range(config.num_epochs):
        # TRAIN
        model.train()
        total_train_loss = 0.0
        all_train_labels: List[float] = []
        all_train_preds: List[float] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} (train)")
        for batch in pbar:
            x, mask, label, times, static, *_ = batch
            x = x.to(device).float()
            mask = mask.to(device).float()
            times = times.to(device).float()
            static = static.to(device).float()
            label = label.to(device).float()

            optimizer.zero_grad()
            pred = model(x, static=static, time=times, sensor_mask=mask)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            all_train_labels.extend(label.detach().cpu().numpy())
            all_train_preds.extend(pred.detach().cpu().numpy())
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_mse = mean_squared_error(all_train_labels, all_train_preds)
        train_mae = mean_absolute_error(all_train_labels, all_train_preds)

        # VAL
        model.eval()
        total_val_loss = 0.0
        all_val_labels: List[float] = []
        all_val_preds: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                x, mask, label, times, static, *_ = batch
                x = x.to(device).float()
                mask = mask.to(device).float()
                times = times.to(device).float()
                static = static.to(device).float()
                label = label.to(device).float()

                pred = model(x, static=static, time=times, sensor_mask=mask)
                loss = loss_fn(pred, label)
                total_val_loss += loss.item()
                all_val_labels.extend(label.cpu().numpy())
                all_val_preds.extend(pred.cpu().numpy())

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_mse = mean_squared_error(all_val_labels, all_val_preds)
        val_mae = mean_absolute_error(all_val_labels, all_val_preds)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={avg_train_loss:.4f} mse={train_mse:.4f} mae={train_mae:.4f} | "
            f"val_loss={avg_val_loss:.4f} mse={val_mse:.4f} mae={val_mae:.4f}"
        )

        scheduler.step()

        # early stopping by MSE
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without val MSE improvement.")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # TEST
    model.eval()
    total_test_loss = 0.0
    all_test_labels: List[float] = []
    all_test_preds: List[float] = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            x, mask, label, times, static, *_ = batch
            x = x.to(device).float()
            mask = mask.to(device).float()
            times = times.to(device).float()
            static = static.to(device).float()
            label = label.to(device).float()

            pred = model(x, static=static, time=times, sensor_mask=mask)
            loss = loss_fn(pred, label)
            total_test_loss += loss.item()

            # if label.shape is empty, turn into tensor
            if label.dim() == 0:
                label = label.unsqueeze(0)
            all_test_labels.extend(label.cpu().numpy())
            all_test_preds.extend(pred.cpu().numpy())
            pbar.set_postfix(loss=loss.item())

    avg_test_loss = total_test_loss / max(1, len(test_loader))
    test_mse = mean_squared_error(all_test_labels, all_test_preds)
    test_mae = mean_absolute_error(all_test_labels, all_test_preds)
    test_r2 = r2_score(all_test_labels, all_test_preds)

    print(
        "\nTEST RESULTS "
        f"(dataset={config.dataset}, size={config.size}, seed={config.seed}): "
        f"loss={avg_test_loss:.4f} mse={test_mse:.4f} mae={test_mae:.4f} r2={test_r2:.4f}"
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
        test_mse=test_mse,
        test_mae=test_mae,
        test_r2=test_r2,
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
    parser = argparse.ArgumentParser(description="Fine-tune SSL_BAT on regression tasks (e.g., Length of Stay)")
    parser.add_argument("--model_path", required=True, type=str, help="Path to pretrained checkpoint .ckpt")
    parser.add_argument("--dataset", default="mimic_los", type=str,
                        choices=["eicu_los", "miiv_los", "mimic_los", "mimic_los_regression", "p19_los"],
                        help="Dataset name (e.g., mimic_los, mimic_los_regression)")
    parser.add_argument("--sizes", default="9506", type=str, help='e.g. "100,500,1000" or "100:9000:100"')
    parser.add_argument("--seeds", default="42", type=str, help='e.g. "42,84,126"')
    parser.add_argument("--fine_tune_head", action="store_true", help="Only fine-tune the regression head")
    parser.add_argument("--bz", default=32, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--subset_root", default="icu_benchmarks/data/preprocessed_data", type=str,
                        help="Root path that contains {dataset}/{size}_{seed}/ parquet files (relative or absolute)")
    parser.add_argument("--gin_config", default="", type=str,
                        help="Optional gin config file (not required for fine-tuning); leave empty to skip")

    args = parser.parse_args()

    sizes = parse_int_list(args.sizes)
    seeds = parse_int_list(args.seeds)

    # Map flag -> mode for path naming
    mode_str = "head" if args.fine_tune_head else "full"

    # Build automatic output_dir
    output_dir = Path(f"finetuning_results/pretrained_BAT/{args.dataset}/{mode_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # run ID for this sweep
    sweep_id = hashlib.md5(json.dumps({
        "model_path": args.model_path,
        "dataset": args.dataset,
        "sizes": sizes,
        "seeds": seeds,
        "fine_tune_head": args.fine_tune_head,
        "bz": args.bz,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "subset_root": args.subset_root,
        "gin_config": args.gin_config,
    }, sort_keys=True).encode()).hexdigest()[:10]

    per_run_log = output_dir / f"runs_{sweep_id}.jsonl"
    csv_path = output_dir / f"summary_{sweep_id}.csv"

    # write header-ish metadata
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
                size=size,
                seed=seed,
                model_path=args.model_path,
                fine_tune_head=bool(args.fine_tune_head),
                batch_size=args.bz,
                lr=args.lr,
                num_epochs=args.num_epochs,
                subset_root=args.subset_root,
                output_dir=str(output_dir),
                gin_config=args.gin_config or "",
            )
            try:
                result = train_eval_one(run_cfg)
            except Exception as e:
                print(f"[ERROR] size={size} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
                continue

            all_results.append(result)
            # append JSONL row
            with per_run_log.open("a") as f:
                f.write(json.dumps(asdict(result)) + "\n")

    # aggregate to CSV
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
