#!/usr/bin/env python
# hyperparameter_tuning_regression_timestep.py

import argparse
from pathlib import Path
from copy import deepcopy
import random
import numpy as np
import polars as pl
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.loader import BATPolarsDataset

# BAT
from icu_benchmarks.models.dl_models.bat import (
    AutoregressiveEncoderCrossParallel,
    EncoderPrediction,
    RegressionHead,
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


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    if y_true.size == 0:
        return float("nan"), float("nan")
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae


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
    return (
        BATPolarsDataset(data=data, split="train", ram_cache=False, runmode=RunMode.regression, vars=VARS_DICT),
        BATPolarsDataset(data=data, split="val",   ram_cache=False, runmode=RunMode.regression, vars=VARS_DICT),
        BATPolarsDataset(data=data, split="test",  ram_cache=False, runmode=RunMode.regression, vars=VARS_DICT),
    )


def state_dict_transfer_report(model: torch.nn.Module, loaded_state: dict) -> dict:
    model_sd = model.state_dict()
    matched = {k for k, v in loaded_state.items() if k in model_sd and model_sd[k].shape == v.shape}
    matched_numel = sum(model_sd[k].numel() for k in matched)
    total_numel = sum(v.numel() for v in model_sd.values())
    return {
        "matched_keys": len(matched),
        "model_keys": len(model_sd),
        "matched_param_frac": matched_numel / max(1, total_numel),
    }


def build_timestep_regression_model_from_ckpt(model_path: str, model_type: str) -> torch.nn.Module:
    """
    Builds per-timestep regression model and loads encoder weights from SSL checkpoint:
      - expects keys under 'model.encoder_class.*'
      - returns model whose forward outputs (B,T) for output_dim=1
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

        #report = state_dict_transfer_report(encoder, encoder_state) # [DEBUG] because pretraine dBAT was not on autoregressive model 
        encoder.load_state_dict(encoder_state, strict=False)
        #print(
        #    "[BAT SSL transfer] "
        #    f"matched_keys={report['matched_keys']}/{report['model_keys']} | "
        #    f"matched_params={report['matched_param_frac']:.1%}"
        #) # [DEBUG] because pretraine dBAT was not on autoregressive model 

        model = EncoderPrediction(
            encoder_class=encoder,
            prediction_head=RegressionHead,
            prediction_head_kwargs={"output_dim": 1},
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
        encoder.load_state_dict(encoder_state, strict=False)

        model = GRUDEncoderPrediction(
            encoder_class=encoder,
            prediction_head=RegressionHead,
            prediction_head_kwargs={"output_dim": 1},
        )
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


# ----------------------------------------------------
# Training + Val + Test for one LR
# ----------------------------------------------------
def run_single_experiment(
    dataset, task, size, seed,
    model_path, model_type,
    lr, batch_size, fine_tune_head,
    num_epochs, subset_root,
    patience=10,
):
    # match your previous behavior: fixed training seed
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

    model = build_timestep_regression_model_from_ckpt(model_path, model_type)

    # freeze/unfreeze like before
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
    loss_fn = torch.nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    no_imp = 0

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        all_val_ytrue = None  # not used here
        all_train_ytrue, all_train_ypred = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} train")
        for batch in pbar:
            x, mask, label, times, static, delta, obs_mask = batch
            x = x.float().to(device)
            mask = mask.float().to(device)
            times = times.float().to(device)
            static = static.float().to(device)
            label = label.float().to(device)
            obs_mask = obs_mask.bool().to(device)

            optimizer.zero_grad()
            pred = model(x, static=static, time=times, sensor_mask=mask)

            # ensure (B,T)
            if pred.dim() == 3 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            if pred.dim() != 2:
                raise ValueError(f"Expected pred (B,T), got {pred.shape}")
            if label.dim() != 2:
                raise ValueError(f"Expected label (B,T), got {label.shape}")

            valid_pred = pred[obs_mask]
            valid_label = label[obs_mask]

            if valid_label.numel() == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = loss_fn(valid_pred, valid_label)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

            all_train_ytrue.extend(valid_label.detach().cpu().numpy().tolist())
            all_train_ypred.extend(valid_pred.detach().cpu().numpy().tolist())

        train_rmse, train_mae = rmse_mae(np.array(all_train_ytrue), np.array(all_train_ypred))

        # -------------------------
        # Validation (early stop on RMSE)
        # -------------------------
        model.eval()
        all_val_ytrue, all_val_ypred = [], []
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x, mask, label, times, static, delta, obs_mask = batch
                x = x.float().to(device)
                mask = mask.float().to(device)
                times = times.float().to(device)
                static = static.float().to(device)
                label = label.float().to(device)
                obs_mask = obs_mask.bool().to(device)

                pred = model(x, static=static, time=times, sensor_mask=mask)
                if pred.dim() == 3 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)

                valid_pred = pred[obs_mask]
                valid_label = label[obs_mask]

                if valid_label.numel() > 0:
                    val_loss_sum += float(loss_fn(valid_pred, valid_label).item())

                all_val_ytrue.extend(valid_label.detach().cpu().numpy().tolist())
                all_val_ypred.extend(valid_pred.detach().cpu().numpy().tolist())

        val_rmse, val_mae = rmse_mae(np.array(all_val_ytrue), np.array(all_val_ypred))
        val_loss = val_loss_sum / max(1, len(val_loader))

        print(
            f"Epoch {epoch+1}: "
            f"train_rmse={train_rmse:.4f} | "
            f"val_rmse={val_rmse:.4f} | "
            f"val_mae*336={val_mae * 336:.2f}"
        )



        scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stopping after {patience} epochs without val RMSE improvement.")
                break

    model.load_state_dict(best_state)

    # -------------------------
    # Test
    # -------------------------
    model.eval()
    all_test_ytrue, all_test_ypred = [], []
    test_loss_sum = 0.0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            x, mask, label, times, static, delta, obs_mask = batch
            x = x.float().to(device)
            mask = mask.float().to(device)
            times = times.float().to(device)
            static = static.float().to(device)
            label = label.float().to(device)
            obs_mask = obs_mask.bool().to(device)

            pred = model(x, static=static, time=times, sensor_mask=mask)
            if pred.dim() == 3 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)

            valid_pred = pred[obs_mask]
            valid_label = label[obs_mask]

            if valid_label.numel() > 0:
                test_loss_sum += float(loss_fn(valid_pred, valid_label).item())

            all_test_ytrue.extend(valid_label.detach().cpu().numpy().tolist())
            all_test_ypred.extend(valid_pred.detach().cpu().numpy().tolist())

    test_rmse, test_mae = rmse_mae(np.array(all_test_ytrue), np.array(all_test_ypred))
    test_loss = test_loss_sum / max(1, len(test_loader))

    return {
        "lr": lr,
        "batch_size": batch_size,
        "test_loss": test_loss,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    }


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True, type=str)
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
            f"\n=== Running LR = {lr} === Dataset={args.dataset} "
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
