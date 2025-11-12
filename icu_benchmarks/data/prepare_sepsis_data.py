#!/usr/bin/env python
"""
Prepare Sepsis data for fine-tuning.

This script converts the Physionet 2019 sepsis data (NumPy format) into the format
expected by YAIB and the fine-tuning script:
  - {split}_OUTCOME.parquet
  - {split}_FEATURES.parquet

The input data is in NumPy format with the following structure per sample:
  - ts_values: (timesteps, 34) - dynamic features
  - ts_indicators: (timesteps, 34) - missing indicators
  - ts_times: (timesteps,) - time values
  - static: (4,) - static features
  - labels: (timesteps, 1) - binary sepsis labels

Usage:
    python icu_benchmarks/data/prepare_sepsis_data.py \
        --input_dir /isdata/winthergrp/gsn245/scratch/Patient_Journey_Classification/P19data/split_1 \
        --output_dir icu_benchmarks/data/preprocessed_data/p19 \
        --seed 42 \
        --train_ratio 0.7 \
        --val_ratio 0.15
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Feature names based on Physionet 2019 Sepsis Challenge
# See: https://physionet.org/content/challenge-2019/1.0.0/
DYNAMIC_FEATURE_NAMES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets'
]

STATIC_FEATURE_NAMES = ['Age', 'Gender', 'Unit1', 'Unit2']


def load_numpy_data(input_dir: Path) -> List[dict]:
    """Load and combine train, val, test NumPy files."""
    all_data = []

    for split_name in ['train', 'validation', 'test']:
        file_path = input_dir / f"{split_name}_physionet2019_1.npy"

        if not file_path.exists():
            logging.warning(f"File not found: {file_path}, skipping...")
            continue

        logging.info(f"Loading {file_path}...")
        data = np.load(file_path, allow_pickle=True)
        all_data.extend(data)
        logging.info(f"  Loaded {len(data)} samples from {split_name}")

    logging.info(f"Total samples loaded: {len(all_data)}")
    return all_data


def convert_to_polars(data: List[dict]) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Convert NumPy data to Polars DataFrames (FEATURES and OUTCOME format)."""

    # Lists to accumulate data
    feature_rows = []
    outcome_rows = []

    for stay_idx, sample in enumerate(data):
        stay_id = stay_idx  # Use integer stay_id like in the LOS data

        # Extract data from sample
        ts_values = sample['ts_values']  # (timesteps, 34)
        ts_indicators = sample['ts_indicators']  # (timesteps, 34) - True means missing
        ts_times = sample['ts_times']  # (timesteps,)
        static = sample['static']  # (4,)
        labels = sample['labels']  # (timesteps, 1)

        n_timesteps = len(ts_times)

        # Process each timestep
        for t in range(n_timesteps):
            # Build feature row
            feat_row = {
                'stay_id': stay_id,
                'time': float(ts_times[t])
            }

            # Add dynamic features and missing indicators
            for feat_idx, feat_name in enumerate(DYNAMIC_FEATURE_NAMES):
                value = ts_values[t, feat_idx]
                is_missing = bool(ts_indicators[t, feat_idx])

                # Store value as None if missing, otherwise as float
                feat_row[feat_name] = None if is_missing else float(value)

                # Add missing indicator column (1 if missing, 0 if present)
                feat_row[f'MissingIndicator_{feat_name}'] = 1 if is_missing else 0

            # Add static features (repeated for each timestep)
            for feat_idx, feat_name in enumerate(STATIC_FEATURE_NAMES):
                feat_row[feat_name] = float(static[feat_idx])

            feature_rows.append(feat_row)

            # Build outcome row
            outc_row = {
                'stay_id': stay_id,
                'time': float(ts_times[t]),
                'label': int(labels[t, 0])  # Binary classification label (0 or 1)
            }
            outcome_rows.append(outc_row)

        if (stay_idx + 1) % 5000 == 0:
            logging.info(f"  Processed {stay_idx + 1}/{len(data)} samples...")

    # Convert to Polars DataFrames
    features_df = pl.DataFrame(feature_rows)
    outcome_df = pl.DataFrame(outcome_rows)

    logging.info(f"Created DataFrames:")
    logging.info(f"  Features: {features_df.shape}")
    logging.info(f"  Outcome: {outcome_df.shape}")

    return features_df, outcome_df


def add_prediction_window(outcome_df: pl.DataFrame, window_hours: int = 6) -> pl.DataFrame:
    """
    Add a prediction window for sepsis labels.

    If sepsis occurs at hour T, label hours (T-window_hours) through (T-1) as positive.
    This allows the model to predict sepsis up to `window_hours` hours early without penalty.

    Args:
        outcome_df: DataFrame with columns ['stay_id', 'time', 'label']
        window_hours: Number of hours before sepsis onset to also label as positive

    Returns:
        Modified outcome_df with updated labels
    """
    logging.info(f"Adding {window_hours}-hour prediction window to labels...")

    # Sort by stay_id and time
    outcome_df = outcome_df.sort(["stay_id", "time"])

    # For each patient, find first sepsis occurrence and backfill labels
    modified_rows = []

    for stay_id in outcome_df['stay_id'].unique():
        stay_data = outcome_df.filter(pl.col('stay_id') == stay_id)
        times = stay_data['time'].to_list()
        labels = stay_data['label'].to_list()

        # Find first sepsis occurrence (label == 1)
        sepsis_times = [t for t, l in zip(times, labels) if l == 1]

        if sepsis_times:
            first_sepsis_time = min(sepsis_times)

            # Update labels for window_hours before first sepsis
            for i, (time, label) in enumerate(zip(times, labels)):
                if first_sepsis_time - window_hours <= time < first_sepsis_time:
                    labels[i] = 1  # Mark as positive within prediction window

        # Reconstruct rows for this stay
        for time, label in zip(times, labels):
            modified_rows.append({
                'stay_id': stay_id,
                'time': time,
                'label': label
            })

    # Create new DataFrame with updated labels
    updated_outcome_df = pl.DataFrame(modified_rows)

    # Log statistics
    original_positive = outcome_df['label'].sum()
    updated_positive = updated_outcome_df['label'].sum()
    logging.info(f"  Original positive labels: {original_positive}")
    logging.info(f"  Updated positive labels: {updated_positive}")
    logging.info(f"  Added {updated_positive - original_positive} labels in prediction window")

    return updated_outcome_df


def impute_features(features_df: pl.DataFrame) -> pl.DataFrame:
    """Impute missing values in dynamic features following YAIB preprocessing pattern."""
    logging.info("Imputing missing values in dynamic features...")

    # Sort by stay_id and time for proper forward filling
    features_df = features_df.sort(["stay_id", "time"])

    # Forward fill within each stay_id group, then zero fill remaining
    for col in DYNAMIC_FEATURE_NAMES:
        features_df = features_df.with_columns(
            pl.col(col).forward_fill().over("stay_id")
        )
        features_df = features_df.with_columns(
            pl.col(col).fill_null(0.0)
        )

    # Zero fill static features (if any are missing)
    for col in STATIC_FEATURE_NAMES:
        features_df = features_df.with_columns(
            pl.col(col).fill_null(0.0)
        )

    logging.info(f"Imputed features shape: {features_df.shape}")
    return features_df


def split_data(
    features_df: pl.DataFrame,
    outcome_df: pl.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[dict, dict]:
    """Split data into train, val, test sets."""

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
    features_splits = {}
    outcome_splits = {}

    for split_name, stay_ids in [("train", train_stays), ("val", val_stays), ("test", test_stays)]:
        # Filter features and outcomes for this split
        split_features = features_df.filter(pl.col("stay_id").is_in(stay_ids))
        split_outcome = outcome_df.filter(pl.col("stay_id").is_in(stay_ids))

        features_splits[split_name] = split_features
        outcome_splits[split_name] = split_outcome

        logging.info(f"{split_name.upper()} - Features: {split_features.shape}, Outcomes: {split_outcome.shape}")

    return features_splits, outcome_splits


def save_splits(
    features_splits: dict,
    outcome_splits: dict,
    output_dir: Path,
    seed: int
):
    """Save splits to parquet files."""
    # Get total size from combined splits
    total_size = sum(len(features_splits[split]["stay_id"].unique()) for split in features_splits)

    # Create output directory
    split_dir = output_dir / f"{total_size}_{seed}"
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
    parser = argparse.ArgumentParser(description="Prepare Physionet 2019 Sepsis data for YAIB fine-tuning")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing train/val/test .npy files (e.g., .../P19data/split_1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for preprocessed data"
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
        "--prediction_window",
        type=int,
        default=6,
        help="Hours before sepsis onset to label as positive for early prediction (default: 6)"
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Load NumPy data
    logging.info(f"Loading NumPy data from {input_dir}...")
    data = load_numpy_data(input_dir)

    # Convert to Polars DataFrames
    logging.info("Converting to Polars DataFrames...")
    features_df, outcome_df = convert_to_polars(data)

    # Impute missing values
    logging.info("Imputing missing values...")
    features_df = impute_features(features_df)

    # Add prediction window for sepsis labels
    outcome_df = add_prediction_window(outcome_df, window_hours=args.prediction_window)

    # Split data
    logging.info(f"Splitting data (train={args.train_ratio}, val={args.val_ratio})...")
    features_splits, outcome_splits = split_data(
        features_df, outcome_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Save splits
    logging.info(f"Saving splits to {output_dir}")
    save_splits(features_splits, outcome_splits, output_dir, args.seed)


if __name__ == "__main__":
    main()
