#!/usr/bin/env python
"""
Prepare LOS (Length of Stay) data for fine-tuning.

This script converts the demo LOS data (dyn.parquet, sta.parquet, outc.parquet)
into the format expected by the fine-tuning script:
  - {split}_OUTCOME.parquet
  - {split}_FEATURES.parquet

Usage:
    python icu_benchmarks/data/prepare_los_data.py \
        --input_dir demo_data/los/mimic_demo \
        --output_dir icu_benchmarks/data/preprocessed_data/mimic_los \
        --size 9506 \
        --seed 42 \
        --train_ratio 0.7 \
        --val_ratio 0.15
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import polars as pl
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_demo_data(input_dir: Path) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load demo data files."""
    dyn_df = pl.read_parquet(input_dir / "dyn.parquet")
    sta_df = pl.read_parquet(input_dir / "sta.parquet")
    outc_df = pl.read_parquet(input_dir / "outc.parquet")

    logging.info(f"Loaded dynamic data: {dyn_df.shape}")
    logging.info(f"Loaded static data: {sta_df.shape}")
    logging.info(f"Loaded outcome data: {outc_df.shape}")

    return dyn_df, sta_df, outc_df


def merge_features(dyn_df: pl.DataFrame, sta_df: pl.DataFrame, add_missing_indicators: bool = True) -> pl.DataFrame:
    """Merge dynamic and static features and optionally add missing indicators."""
    # Join static features with dynamic features
    # Static features will be repeated for each time point
    features_df = dyn_df.join(sta_df, on="stay_id", how="left")

    # Convert time from Duration to numeric (hours) for PyTorch compatibility
    if features_df["time"].dtype == pl.Duration:
        logging.info("Converting time from Duration to numeric hours...")
        features_df = features_df.with_columns(
            (pl.col("time").dt.total_milliseconds() / (1000 * 60 * 60)).alias("time")
        )
        logging.info(f"Time column converted to Float64 (hours)")

    # Encode categorical variables (e.g., sex) to numeric
    if "sex" in features_df.columns and features_df["sex"].dtype == pl.Utf8:
        logging.info("Encoding 'sex' column: Male=1, Female=0...")
        features_df = features_df.with_columns(
            pl.when(pl.col("sex") == "Male")
            .then(1.0)
            .when(pl.col("sex") == "Female")
            .then(0.0)
            .otherwise(None)
            .alias("sex")
        )
        logging.info(f"Sex column converted to Float64")

    if add_missing_indicators:
        # Add missing indicator columns for all dynamic features
        # These are expected by the BATPolarsDataset
        dynamic_cols = [col for col in dyn_df.columns if col not in ["stay_id", "time"]]

        for col in dynamic_cols:
            # Create MissingIndicator column: 1 if missing, 0 if present
            features_df = features_df.with_columns(
                pl.col(col).is_null().cast(pl.Int32).alias(f"MissingIndicator_{col}")
            )

        logging.info(f"Merged features shape: {features_df.shape}")
        logging.info(f"Added {len(dynamic_cols)} missing indicator columns")

        # CRITICAL: Impute missing values after creating indicators
        # Following the pattern from PolarsClassificationPreprocessor._process_dynamic
        logging.info("Imputing missing values in dynamic features...")

        # Forward fill within each stay_id group
        features_df = features_df.sort(["stay_id", "time"])
        for col in dynamic_cols:
            features_df = features_df.with_columns(
                pl.col(col).forward_fill().over("stay_id")
            )

        # Zero fill any remaining missing values
        for col in dynamic_cols:
            features_df = features_df.with_columns(
                pl.col(col).fill_null(0.0)
            )

        # Zero fill static features
        static_cols = [col for col in sta_df.columns if col != "stay_id"]
        for col in static_cols:
            features_df = features_df.with_columns(
                pl.col(col).fill_null(0.0)
            )

        logging.info("Missing value imputation complete")
    else:
        logging.info(f"Merged features shape: {features_df.shape}")

    return features_df


def split_data(
    features_df: pl.DataFrame,
    outc_df: pl.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[dict, dict]:
    """Split data into train, val, test sets."""
    # Convert time in outcome data if needed
    if outc_df["time"].dtype == pl.Duration:
        logging.info("Converting outcome time from Duration to numeric hours...")
        outc_df = outc_df.with_columns(
            (pl.col("time").dt.total_milliseconds() / (1000 * 60 * 60)).alias("time")
        )

    # Get unique stay_ids
    unique_stays = features_df["stay_id"].unique().to_list()

    # Set random seed for reproducibility
    np.random.seed(seed)
    np.random.shuffle(unique_stays)

    # Calculate split indices
    n_total = len(unique_stays)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_stays = unique_stays[:n_train]
    val_stays = unique_stays[n_train:n_train + n_val]
    test_stays = unique_stays[n_train + n_val:]

    logging.info(f"Split sizes - Train: {len(train_stays)}, Val: {len(val_stays)}, Test: {len(test_stays)}")

    # Create splits
    splits = {
        "train": {"stays": train_stays},
        "val": {"stays": val_stays},
        "test": {"stays": test_stays}
    }

    features_splits = {}
    outcome_splits = {}

    for split_name, split_info in splits.items():
        stay_ids = split_info["stays"]

        # Filter features and outcomes for this split
        split_features = features_df.filter(pl.col("stay_id").is_in(stay_ids))
        split_outcome = outc_df.filter(pl.col("stay_id").is_in(stay_ids))

        features_splits[split_name] = split_features
        outcome_splits[split_name] = split_outcome

        logging.info(f"{split_name.upper()} - Features: {split_features.shape}, Outcomes: {split_outcome.shape}")

    return features_splits, outcome_splits


def save_splits(
    features_splits: dict,
    outcome_splits: dict,
    output_dir: Path,
    size: int,
    seed: int
):
    """Save splits to parquet files."""
    # Create output directory
    split_dir = output_dir / f"{size}_{seed}"
    split_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        # Save FEATURES
        features_path = split_dir / f"{split_name}_FEATURES.parquet"
        features_splits[split_name].write_parquet(features_path)
        logging.info(f"Saved {features_path}")

        # Save OUTCOME
        outcome_path = split_dir / f"{split_name}_OUTCOME.parquet"
        outcome_splits[split_name].write_parquet(outcome_path)
        logging.info(f"Saved {outcome_path}")

    logging.info(f"\n✅ Data preparation complete! Output directory: {split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning (classification or regression)")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing dyn.parquet, sta.parquet, outc.parquet"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for preprocessed data"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["classification", "regression"],
        help="Task type: classification or regression (default: regression)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=9506,
        help="Dataset size identifier (for output path naming)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Proportion of data for training (default: 0.7)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proportion of data for validation (default: 0.15)"
    )
    parser.add_argument(
        "--add_missing_indicators",
        action="store_true",
        default=True,
        help="Add missing indicator columns (required for BAT models, default: True)"
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    required_files = ["dyn.parquet", "sta.parquet", "outc.parquet"]
    for file in required_files:
        if not (input_dir / file).exists():
            raise FileNotFoundError(f"Required file not found: {input_dir / file}")

    # Load data
    logging.info(f"Loading data from {input_dir} (task_type={args.task_type})")
    dyn_df, sta_df, outc_df = load_demo_data(input_dir)

    # Merge features
    logging.info(f"Merging features (add_missing_indicators={args.add_missing_indicators})...")
    features_df = merge_features(dyn_df, sta_df, add_missing_indicators=args.add_missing_indicators)

    # Split data
    logging.info(f"Splitting data (train={args.train_ratio}, val={args.val_ratio})...")
    features_splits, outcome_splits = split_data(
        features_df, outc_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Save splits
    logging.info(f"Saving splits to {output_dir}")
    save_splits(features_splits, outcome_splits, output_dir, args.size, args.seed)


if __name__ == "__main__":
    main()
