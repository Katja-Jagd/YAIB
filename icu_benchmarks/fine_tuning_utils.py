"""
Common utilities for fine-tuning scripts.

This module contains shared functionality used across:
- fine_tuning_classification.py
- fine_tuning_regression.py
- fine_tuning_regression_per_timestep.py
- fine_tuning_classification_per_timestep.py

Extracted to avoid code duplication and ensure consistent behavior.
"""

import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import polars as pl
import torch

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.loader import BATPolarsDataset


# Default variable mapping for ICU data
# This matches the feature set used in fine-tuning scripts
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


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed to use for all libraries

    Example:
        >>> set_seeds(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_subset_as_data_dict(base_dir: Path) -> Dict[str, Dict[str, pl.DataFrame]]:
    """Load train/val/test data from parquet files.

    Expected directory structure:
        base_dir/
            train_OUTCOME.parquet
            train_FEATURES.parquet
            val_OUTCOME.parquet
            val_FEATURES.parquet
            test_OUTCOME.parquet
            test_FEATURES.parquet

    Args:
        base_dir: Path to directory containing parquet files

    Returns:
        Dictionary with structure:
        {
            "train": {"OUTCOME": df, "FEATURES": df},
            "val": {"OUTCOME": df, "FEATURES": df},
            "test": {"OUTCOME": df, "FEATURES": df}
        }

    Raises:
        FileNotFoundError: If required parquet files are missing

    Example:
        >>> data = load_subset_as_data_dict(Path("data/mimic/1000_42"))
        >>> train_outcome = data["train"]["OUTCOME"]
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


def load_subset(dataset, task, size, seed, subset_root):
    """Load subset from HP tuning directory structure.

    Used by HP tuning scripts. Expects: subset_root/task/dataset/size_seed/
    """
    path = Path(subset_root) / task / dataset / f"{size}_{seed}"
    data = {}
    for split in ["train", "val", "test"]:
        o = path / f"{split}_OUTCOME.parquet"
        f = path / f"{split}_FEATURES.parquet"
        if not o.exists() or not f.exists():
            raise FileNotFoundError(f"Missing required files for {split} in {path}")
        data[split] = {"OUTCOME": pl.read_parquet(o), "FEATURES": pl.read_parquet(f)}
    return data


def build_datasets(
    data: Dict[str, Dict[str, pl.DataFrame]],
    runmode: RunMode = RunMode.classification,
    vars_dict: Dict = None,
) -> Tuple[BATPolarsDataset, BATPolarsDataset, BATPolarsDataset]:
    """Build train/val/test datasets from loaded data.

    Args:
        data: Data dictionary from load_subset_as_data_dict()
        runmode: Task type (classification or regression)
        vars_dict: Variable mapping dictionary. Uses VARS_DICT if None.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Example:
        >>> data = load_subset_as_data_dict(Path("data/mimic/1000_42"))
        >>> train_ds, val_ds, test_ds = build_datasets(data, RunMode.classification)
    """
    if vars_dict is None:
        vars_dict = VARS_DICT

    train_set = BATPolarsDataset(
        data=data,
        split="train",
        ram_cache=False,
        runmode=runmode,
        vars=vars_dict,
    )
    val_set = BATPolarsDataset(
        data=data,
        split="val",
        ram_cache=False,
        runmode=runmode,
        vars=vars_dict,
    )
    test_set = BATPolarsDataset(
        data=data,
        split="test",
        ram_cache=False,
        runmode=runmode,
        vars=vars_dict,
    )
    return train_set, val_set, test_set


def parse_int_list(arg: str) -> list[int]:
    """Parse comma-separated integers, ranges, or step syntax.

    Supports multiple formats:
    - Single value: "1000" → [1000]
    - Comma-separated: "100,500,1000" → [100, 500, 1000]
    - Colon step syntax: "100:1000:100" → [100, 200, 300, ..., 1000]
    - Hyphen range: "100-500" → [100, 200, 300, 400, 500]

    Args:
        arg: String with integers in supported formats

    Returns:
        List of integers

    Example:
        >>> parse_int_list("100,500,1000")
        [100, 500, 1000]
        >>> parse_int_list("100:500:100")
        [100, 200, 300, 400, 500]
        >>> parse_int_list("100-500")
        [100, 200, 300, 400, 500]
    """
    s = arg.strip()

    # Handle colon syntax: start:stop:step
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            start, stop, step = [int(x) for x in parts]
            return list(range(start, stop + (1 if step > 0 else -1), step))
        else:
            raise ValueError(f"Colon syntax requires exactly 3 parts (start:stop:step), got: {s}")

    # Handle comma-separated values (may include ranges)
    if "," in s:
        result = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                # Hyphen range syntax
                start, end = part.split("-")
                result.extend(range(int(start), int(end) + 1, 100))
            else:
                result.append(int(part))
        return result

    # Single value
    return [int(s)]
