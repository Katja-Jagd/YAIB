import json
from datetime import datetime
import logging
import gin
from pathlib import Path
from pytorch_lightning import seed_everything
from icu_benchmarks.wandb_utils import wandb_log
from icu_benchmarks.run_utils import aggregate_results
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.models.train import train_common
from icu_benchmarks.models.utils import JsonResultLoggingEncoder
from icu_benchmarks.run_utils import log_full_line
from icu_benchmarks.constants import RunMode

import os # Added to extract dataset name 

@gin.configurable
def execute_repeated_cv(
    data_dir: Path,
    log_dir: Path,
    seed: int,
    eval_only: bool = False,
    train_size: int = None,
    load_weights: bool = False,
    source_dir: Path = None,
    cv_repetitions: int = 5,
    cv_repetitions_to_train: int = None,
    cv_folds: int = 5,
    cv_folds_to_train: int = None,
    reproducible: bool = True,
    debug: bool = False,
    generate_cache: bool = False,
    load_cache: bool = False,
    test_on: str = "test",
    mode: str = RunMode.classification,
    pretrained_imputation_model: object = None,
    cpu: bool = False,
    verbose: bool = False,
    wandb: bool = False,
    complete_train: bool = False,
    enable_subset_train: bool = False, # ADDED FOR SUBSET
    subset_train_size: int = 1000, # ADDED FOR SUBSET
    subset_train_seed: int = 42, # ADDED FOR SUBSET
    task_name: str = None, # ADDED FOR ORGANIZING PREPROCESSED DATA BY TASK
    dataset_name: str = None, # ADDED FOR ORGANIZING PREPROCESSED DATA BY DATASET NAME
    stop_after_first_fold: bool = False,
) -> float:
    """Preprocesses data and trains a model for each fold.

    Args:

        complete_train: Use the full data for training instead of held out test splits.
        wandb: Use wandb for logging.
        data_dir: Path to the data directory.
        log_dir: Path to the log directory.
        seed: Random seed.
        eval_only: Whether to only evaluate the model.
        train_size: Fixed size of train split (including validation data).
        load_weights: Whether to load weights from source_dir.
        source_dir: Path to the source directory.
        cv_folds: Number of folds for cross validation.
        cv_folds_to_train: Number of folds to use during training. If None, all folds are trained on.
        cv_repetitions: Amount of cross validation repetitions.
        cv_repetitions_to_train: Amount of training repetitions. If None, all repetitions are trained on.
        reproducible: Whether to make torch reproducible.
        debug: Whether to load less data and enable more logging.
        generate_cache: Whether to generate and save cache.
        load_cache: Whether to load previously cached data.
        test_on: Dataset to test on. Can be "test" or "val" (e.g. for hyperparameter tuning).
        mode: Run mode. Can be one of the values of RunMode
        pretrained_imputation_model: Use a pretrained imputation model.
        cpu: Whether to run on CPU.
        verbose: Enable detailed logging.
    Returns:
        The average loss of all folds.
    """
    if not cv_repetitions_to_train:
        cv_repetitions_to_train = cv_repetitions
    if not cv_folds_to_train:
        cv_folds_to_train = cv_folds
    agg_loss = 0
    seed_everything(seed, reproducible)
    if complete_train:
        logging.info("Will train full model without cross validation.")
        cv_repetitions_to_train = 1
        cv_folds_to_train = 1

    else:
        logging.info(f"Starting nested CV with {cv_repetitions_to_train} repetitions of {cv_folds_to_train} folds.")
    # Train model for each repetition (a manner of splitting the folds)
    for repetition in range(cv_repetitions_to_train):
        # Train model for each fold configuration (i.e, one fold is test fold and the rest are train/val folds)
        for fold_index in range(cv_folds_to_train):

            # ------------  SPECIFY ONE FOLD AND REP TO USE FOR SUBSET TRAINING ---------------- # # the one with the lowest loss during pre-training
            #rep_subset = 0
            #fold_subset = 0
            #if (repetition, fold_index) != (rep_subset, fold_subset):
            #    continue
            # ------------------------------------------------------------ #  

            repetition_fold_dir = log_dir / f"repetition_{repetition}" / f"fold_{fold_index}"
            repetition_fold_dir.mkdir(parents=True, exist_ok=True)

            start_time = datetime.now()
            data = preprocess_data(
                data_dir,
                seed=seed,
                debug=debug,
                load_cache=load_cache,
                generate_cache=generate_cache,
                cv_repetitions=cv_repetitions,
                repetition_index=repetition,
                train_size=train_size,
                cv_folds=cv_folds,
                fold_index=fold_index,
                pretrained_imputation_model=pretrained_imputation_model,
                runmode=mode,
                complete_train=complete_train,
            )

            # Added function to save subsets used for fine-tuning experiment
            import polars as pl
            import os

            def downsample_preserving_balance(
                df: pl.DataFrame,
                label_col: str,
                total_samples: int,
                seed: int = None
            ) -> pl.DataFrame:
                """
                Downsample a Polars DataFrame to total_samples.

                For classification (one row per stay): preserves class distribution.
                For regression (multiple rows per stay): samples stays, not individual rows.
                """
                # Check if this is a multi-row-per-stay scenario (regression with timesteps)
                rows_per_stay = df.group_by("stay_id").len()
                max_rows_per_stay = rows_per_stay.select(pl.col("len").max()).item()

                if max_rows_per_stay > 1:
                    # Regression task: each stay has multiple timesteps
                    # Sample at the stay level to avoid row duplication bug
                    logging.info(f"Detected regression task (max {max_rows_per_stay} rows/stay). Sampling at stay level.")
                    unique_stays = df.select("stay_id").unique()
                    n_stays_to_sample = min(total_samples, len(unique_stays))
                    sampled_stays = unique_stays.sample(n=n_stays_to_sample, with_replacement=False, seed=seed)
                    selected_stay_ids = sampled_stays.select("stay_id").to_series().to_list()
                    result = df.filter(pl.col("stay_id").is_in(selected_stay_ids))
                    logging.info(f"Sampled {n_stays_to_sample} stays, resulting in {len(result)} total rows.")
                    return result

                # Classification task: one row per stay, preserve class balance
                logging.info(f"Detected classification task (1 row/stay). Preserving class balance.")

                # Step 1: Count classes
                labels = df[label_col].unique().to_list()
                label_counts = {}
                total_original = 0

                for label in labels:
                    count = len(df.filter(pl.col(label_col) == label))
                    label_counts[label] = count
                    total_original += count

                # Step 2: Compute target number of samples per class
                label_to_n_samples = {
                    label: int(round((count / total_original) * total_samples))
                    for label, count in label_counts.items()
                }

                # Step 3: Sample per class
                samples = []
                for label, n_label in label_to_n_samples.items():
                    df_label = df.filter(pl.col(label_col) == label)
                    if len(df_label) < n_label:
                        raise ValueError(f"Not enough data for label {label}: requested {n_label}, found {len(df_label)}")
                    sampled = df_label.sample(n=n_label, with_replacement=False, seed=seed)
                    samples.append(sampled)

                # Step 4: Combine and shuffle
                combined = pl.concat(samples)

                # Step 5: Preserve original ordering of stay_ids
                selected_ids = combined.select("stay_id").to_series().to_list()
                original_order = (
                    df.filter(pl.col("stay_id").is_in(selected_ids))
                    .select("stay_id")
                )
                combined = original_order.join(combined, on="stay_id", how="left")

                return combined.sample(n=len(combined), with_replacement=False, seed=seed)

            # ======================= #
            # Perform downsampling if enabled
            # ======================= #
            if enable_subset_train:
                print("\n\n\n")
                print(f"🔍 Subsetting training data to {subset_train_size} samples (seed={subset_train_seed})...")
                print("\n\n\n")

                # Define path to save the preprocessed (and downsampled) data
                REPO_ROOT = Path(__file__).resolve().parents[2] # Detect the YAIB repository root
                subset_root = REPO_ROOT / "icu_benchmarks" / "data" / "preprocessed_data" # preprocessed subset root
                ds_name = dataset_name if dataset_name else os.path.basename(data_dir) # Use dataset_name parameter if provided, otherwise fall back to data_dir basename
                task_folder = task_name if task_name else "default_task" # Include task_name in path to organize by task
                folder_path = subset_root / task_folder / str(ds_name) / f"{subset_train_size}_{subset_train_seed}"
                folder_path.mkdir(parents=True, exist_ok=True)
                
                # List of expected files for each split and key
                expected_files = [
                    ("train", "OUTCOME"),
                    ("train", "FEATURES"),
                    ("val", "OUTCOME"),
                    ("val", "FEATURES"),
                    ("test", "OUTCOME"),
                    ("test", "FEATURES"),
                ]

                # Check if all expected .parquet files already exist
                all_exist = True
                for split, key in expected_files:
                    file_path = os.path.join(folder_path, f"{split}_{key}.parquet")
                    if not os.path.exists(file_path):
                        all_exist = False
                        break

                # ---------------------------------------------------
                # CASE 1: Files already exist → load them and skip processing
                # ---------------------------------------------------
                if all_exist:
                    print(f"✅ Preprocessed subset already exists. Loading from:\n  {folder_path}")

                    for split, key in expected_files:
                        file_path = os.path.join(folder_path, f"{split}_{key}.parquet")
                        try:
                            data[split][key] = pl.read_parquet(file_path)
                            print(f"Loaded existing {split}_{key} from: {file_path}")
                        except Exception as e:
                            print(f"❌ Failed to load {file_path}: {e}")
                            raise RuntimeError("Existing preprocessed dataset is incomplete or corrupted.")

                    print("\n✅ DONE — Using cached preprocessed subset.\n")

                # ---------------------------------------------------
                # CASE 2: Files do NOT exist → perform downsampling and save
                # ---------------------------------------------------
                else:
                    print("⚠️ Subset data does not exist — performing downsampling...")

                    original_train_outcome = data["train"]["OUTCOME"]
                    original_train_features = data["train"]["FEATURES"]

                    # Downsample
                    downsampled_outcome = downsample_preserving_balance(
                        df=original_train_outcome,
                        label_col="label",
                        total_samples=subset_train_size,
                        seed=subset_train_seed
                    )

                    selected_ids = downsampled_outcome.select("stay_id").to_series().to_list()
                    downsampled_features = original_train_features.filter(pl.col("stay_id").is_in(selected_ids))

                    # Replace train split
                    data["train"]["OUTCOME"] = downsampled_outcome
                    data["train"]["FEATURES"] = downsampled_features

                # Save all splits to disk
                for split, split_data in data.items():
                    for key, df in split_data.items():
                        file_path = os.path.join(folder_path, f"{split}_{key}.parquet")
                        try:
                            df.write_parquet(file_path)
                            print(f"Saved {split}_{key} DataFrame as: {file_path}")
                        except Exception as e:
                            print(f"Failed to save {split}_{key} to {file_path}: {e}")

                    print(f"\n✅ PREPROCESSED DATA SAVED to: {folder_path}\n")
            # ======================= #
    
            preprocess_time = datetime.now() - start_time
            start_time = datetime.now()
            agg_loss += train_common(
                data,
                log_dir=repetition_fold_dir,
                eval_only=eval_only,
                load_weights=load_weights,
                source_dir=source_dir,
                reproducible=reproducible,
                test_on=test_on,
                mode=mode,
                cpu=cpu,
                verbose=verbose,
                use_wandb=wandb,
                train_only=complete_train,
            )
            train_time = datetime.now() - start_time

            
            # Stop after repetition 0 and fold 0
            if stop_after_first_fold:
                logging.info("Stopping after repetition 0, fold 0.")
                return agg_loss
            
            
            log_full_line(
                f"FINISHED FOLD {fold_index}| PREPROCESSING DURATION {preprocess_time}| PROCEDURE DURATION {train_time}",
                level=logging.INFO,
            )
            durations = {"preprocessing_duration": preprocess_time, "train_duration": train_time}

            with open(repetition_fold_dir / "durations.json", "w") as f:
                json.dump(durations, f, cls=JsonResultLoggingEncoder)
            if wandb:
                wandb_log({"Iteration": repetition * cv_folds_to_train + fold_index})
            if repetition * cv_folds_to_train + fold_index > 1:
                try:
                    aggregate_results(log_dir)
                except Exception as e:
                    logging.error(f"Failed to aggregate results: {e}")
        log_full_line(f"FINISHED CV REPETITION {repetition}", level=logging.INFO, char="=", num_newlines=3)

    return agg_loss / (cv_repetitions_to_train * cv_folds_to_train)

