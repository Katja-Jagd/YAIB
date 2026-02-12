import importlib
import sys
import warnings
from math import sqrt

import gin
import torch
import json
from argparse import ArgumentParser, BooleanOptionalAction as BOA
from datetime import datetime, timedelta
import logging
from pathlib import Path
import scipy.stats as stats
import shutil
from statistics import mean, pstdev
from icu_benchmarks.models.utils import JsonResultLoggingEncoder
from icu_benchmarks.wandb_utils import wandb_log
import polars as pl
from typing import Optional

# Mapping from task names to their gin configuration files
TASK_TO_GIN_MAPPING = {
    # Binary Classification tasks
    "Mortality": "BinaryClassification",
    "Mortality24": "BinaryClassification",
    "AKI": "TimestepBinaryClassification",
    "TimestepBinaryClassification": "TimestepBinaryClassification",
    "Sepsis": "TimestepBinaryClassification",
    # Regression tasks
    "KidneyFunction": "Regression",
    "LengthOfStay": "LengthOfStay",  # Has its own specialized gin
    "LOS": "LengthOfStay",  # Has its own specialized gin
    # Imputation
    "Imputation": "DatasetImputation",
    # Allow direct gin file names for backwards compatibility
    "BinaryClassification": "BinaryClassification",
    "Regression": "Regression",
    "DatasetImputation": "DatasetImputation",
}


def get_task_gin_and_name(task: str) -> tuple[str, str]:
    """Maps a task name to its gin config file and folder name.

    Args:
        task: Task name provided by user (e.g., 'Mortality24', 'LengthOfStay')

    Returns:
        Tuple of (gin_config_name, folder_name)
        - gin_config_name: The gin file to load (e.g., 'BinaryClassification')
        - folder_name: The directory name to use for organizing outputs (e.g., 'Mortality24')

    Raises:
        ValueError: If task is not recognized
    """
    if task not in TASK_TO_GIN_MAPPING:
        available_tasks = ", ".join(sorted(TASK_TO_GIN_MAPPING.keys()))
        raise ValueError(
            f"Unknown task '{task}'. Available tasks: {available_tasks}"
        )

    gin_config = TASK_TO_GIN_MAPPING[task]

    # For specific task names (Mortality24, AKI, etc.), use the task name for folders
    # For generic gin names (BinaryClassification, Regression), keep them as-is for backward compatibility
    folder_name = task

    return gin_config, folder_name


def build_parser() -> ArgumentParser:
    """Builds an ArgumentParser for the command line.

    Returns:
        The configured ArgumentParser.
    """
    parser = ArgumentParser(description="Framework for benchmarking ML/DL models on ICU data")

    parser.add_argument("-d", "--data-dir", required=True, type=Path, help="Path to the parquet data directory.")
    parser.add_argument("-pd", "--prepro-dir", required=False, type=Path, default = None, help="Path to the preprocessed data directory for subsets.")
    parser.add_argument(
        "-t",
        "--task",
        default="Mortality24",
        required=True,
        help="Task name (e.g., Mortality24, AKI, Sepsis, KidneyFunction, LengthOfStay, Imputation). "
             "Determines both the gin config and folder structure. "
             "Generic names (BinaryClassification, Regression, DatasetImputation) still supported."
    )
    parser.add_argument("-n", "--name", help="Name of the (target) dataset.")
    parser.add_argument("-m", "--model", default="LGBMClassifier", help="Name of the model gin.")
    parser.add_argument("-e", "--experiment", help="Name of the experiment gin.")
    parser.add_argument("-l", "--log-dir", default=Path("../yaib_logs/"), type=Path, help="Log directory for model weights.")
    parser.add_argument("-s", "--seed", default=1234, type=int, help="Random seed for processing, tuning and training.")
    parser.add_argument("-v", "--verbose", default=False, action=BOA, help="Set to log verbosly. Disable for clean logs.")
    parser.add_argument("--cpu", default=False, action=BOA, help="Set to use CPU.")
    parser.add_argument("-db", "--debug", default=False, action=BOA, help="Set to load less data.")
    parser.add_argument("--reproducible", default=True, action=BOA, help="Make torch reproducible.")
    parser.add_argument("-lc", "--load_cache", default=False, action=BOA, help="Set to load generated data cache.")
    parser.add_argument("-gc", "--generate_cache", default=False, action=BOA, help="Set to generate data cache.")
    parser.add_argument("-p", "--preprocessor", type=Path, help="Load custom preprocessor from file.")
    parser.add_argument("-pl", "--plot", action=BOA, help="Generate common plots.")
    parser.add_argument("-wd", "--wandb-sweep", action="store_true", help="Activates wandb hyper parameter sweep.")
    parser.add_argument("-imp", "--pretrained-imputation", type=str, help="Path to pretrained imputation model.")
    parser.add_argument("-hp", "--hyperparams", nargs="+", help="Hyperparameters for model.")
    parser.add_argument("--tune", default=False, action=BOA, help="Find best hyperparameters.")
    parser.add_argument("--hp-checkpoint", type=Path, help="Use previous hyperparameter checkpoint.")
    parser.add_argument("--eval", default=False, action=BOA, help="Only evaluate model, skip training.")
    parser.add_argument("--complete-train", default=False, action=BOA, help="Use all data to train model, skip testing.")
    parser.add_argument("-ft", "--fine-tune", default=None, type=int, help="Finetune model with amount of train data.")
    parser.add_argument("-sn", "--source-name", type=Path, help="Name of the source dataset.")
    parser.add_argument("--source-dir", type=Path, help="Directory containing gin and model weights.")
    parser.add_argument("-sa", "--samples", type=int, default=None, help="Number of samples to use for evaluation.")
    parser.add_argument(
        "-mo",
        "--modalities",
        nargs="+",
        help="Optional modality selection to use. Specify multiple modalities separated by spaces.",
    )
    parser.add_argument("--label", type=str, help="Label to use for evaluation in case of multiple labels.", default=None)
    return parser


def create_run_dir(log_dir: Path, randomly_searched_params: str = None) -> Path:
    """Creates a log directory with the current time as name.

    Also creates a file in the log directory, if any parameters were randomly searched.
    The filename contains the fixed hyperparameters to check against in future runs.

    Args:
        log_dir: Parent directory to create run directory in.
        randomly_searched_params: String representing the randomly searched params.

    Returns:
        Path to the created run log directory.
    """
    log_dir_run = log_dir / str(datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    while log_dir_run.exists():
        log_dir_run = log_dir / str(datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f"))
    log_dir_run.mkdir(parents=True)
    if randomly_searched_params:
        (log_dir_run / randomly_searched_params).touch()
    return log_dir_run


def import_preprocessor(preprocessor_path: str):
    # Import custom supplied preprocessor
    log_full_line(f"Importing custom preprocessor from {preprocessor_path}.", logging.INFO)
    try:
        spec = importlib.util.spec_from_file_location("CustomPreprocessor", preprocessor_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["preprocessor"] = module
        spec.loader.exec_module(module)
        gin.bind_parameter("preprocess.preprocessor", module.CustomPreprocessor)
    except Exception as e:
        logging.error(f"Could not import custom preprocessor from {preprocessor_path}: {e}")


def aggregate_results(log_dir: Path, execution_time: timedelta = None):
    """Aggregates results from all folds and writes to JSON file.

    Args:
        log_dir: Path to the log directory.
        execution_time: Overall execution time.
    """
    aggregated = {}
    shap_values_test = []
    for repetition in log_dir.iterdir():
        if repetition.is_dir():
            aggregated[repetition.name] = {}
            for fold_iter in repetition.iterdir():
                aggregated[repetition.name][fold_iter.name] = {}
                if (fold_iter / "test_metrics.json").is_file():
                    with open(fold_iter / "test_metrics.json", "r") as f:
                        result = json.load(f)
                        aggregated[repetition.name][fold_iter.name].update(result)
                elif (fold_iter / "val_metrics.csv").is_file():
                    with open(fold_iter / "val_metrics.csv", "r") as f:
                        result = json.load(f)
                        aggregated[repetition.name][fold_iter.name].update(result)
                # Add durations to metrics
                if (fold_iter / "durations.json").is_file():
                    with open(fold_iter / "durations.json", "r") as f:
                        result = json.load(f)
                        aggregated[repetition.name][fold_iter.name].update(result)
                if (fold_iter / "test_shap_values.parquet").is_file():
                    shap_values_test.append(pl.read_parquet(fold_iter / "test_shap_values.parquet"))

    if shap_values_test:
        shap_values = pl.concat(shap_values_test)
        shap_values.write_parquet(log_dir / "aggregated_shap_values.parquet")

    try:
        shap_values = pl.concat(shap_values_test)
        shap_values.write_parquet(log_dir / "aggregated_shap_values.parquet")
    except Exception as e:
        logging.error(f"Error aggregating or writing SHAP values: {e}")
    # Aggregate results per metric
    list_scores = {}
    for repetition, folds in aggregated.items():
        for fold, result in folds.items():
            for metric, score in result.items():
                if isinstance(score, (float, int)):
                    list_scores[metric] = list_scores.setdefault(metric, [])
                    list_scores[metric].append(score)

    # Compute statistical metric over aggregated results
    averaged_scores = {metric: (mean(list)) for metric, list in list_scores.items()}

    # Calculate the population standard deviation over aggregated results over folds/iterations
    # Divide by sqrt(n) to get standard deviation.
    std_scores = {metric: (pstdev(list) / sqrt(len(list))) for metric, list in list_scores.items()}

    confidence_interval = {
        metric: (stats.t.interval(0.95, len(list) - 1, loc=mean(list), scale=stats.sem(list)))
        for metric, list in list_scores.items()
    }

    accumulated_metrics = {
        "avg": averaged_scores,
        "std": std_scores,
        "CI_0.95": confidence_interval,
        "execution_time": execution_time.total_seconds() if execution_time is not None else 0.0,
    }

    with open(log_dir / "aggregated_test_metrics.json", "w") as f:
        json.dump(aggregated, f, cls=JsonResultLoggingEncoder)

    with open(log_dir / "accumulated_test_metrics.json", "w") as f:
        json.dump(accumulated_metrics, f, cls=JsonResultLoggingEncoder)

    logging.info(f"Accumulated results: {accumulated_metrics}")

    wandb_log(json.loads(json.dumps(accumulated_metrics, cls=JsonResultLoggingEncoder)))


def name_datasets(train="default", val="default", test="default"):
    """Names the datasets for logging (optional)."""
    gin.bind_parameter("train_common.dataset_names", {"train": train, "val": val, "test": test})


def log_full_line(msg: str, level: int = logging.INFO, char: str = "-", num_newlines: int = 0):
    """Logs a full line of a given character with a message centered.

    Args:
        msg: Message to log.
        level: Logging level.
        char: Character to use for the line.
        num_newlines: Number of newlines to append.
    """
    terminal_size = shutil.get_terminal_size((80, 20))
    reserved_chars = len(logging.getLevelName(level)) + 28
    logging.log(
        level,
        "{0:{char}^{width}}{1}".format(msg, "\n" * num_newlines, char=char, width=terminal_size.columns - reserved_chars),
    )


def load_pretrained_imputation_model(use_pretrained_imputation):
    """Loads a pretrained imputation model.

    Args:
        use_pretrained_imputation: Path to the pretrained imputation model.
    """
    if use_pretrained_imputation is not None and not Path(use_pretrained_imputation).exists():
        logging.warning("The specified pretrained imputation model does not exist.")
        use_pretrained_imputation = None

    if use_pretrained_imputation is not None:
        logging.info("Using pretrained imputation from" + str(use_pretrained_imputation))
        pretrained_imputation_model_checkpoint = torch.load(use_pretrained_imputation, map_location=torch.device("cpu"))
        if isinstance(pretrained_imputation_model_checkpoint, dict):
            imputation_model_class = pretrained_imputation_model_checkpoint["class"]
            pretrained_imputation_model = imputation_model_class(**pretrained_imputation_model_checkpoint["hyper_parameters"])
            pretrained_imputation_model.set_trained_columns(pretrained_imputation_model_checkpoint["trained_columns"])
            pretrained_imputation_model.load_state_dict(pretrained_imputation_model_checkpoint["state_dict"])
        else:
            pretrained_imputation_model = pretrained_imputation_model_checkpoint
        pretrained_imputation_model = pretrained_imputation_model.to("cuda" if torch.cuda.is_available() else "cpu")
        try:
            logging.info(f"imputation model device: {next(pretrained_imputation_model.parameters()).device}")
            pretrained_imputation_model.device = next(pretrained_imputation_model.parameters()).device
        except Exception as e:
            logging.debug(f"Could not set device of imputation model: {e}")
    else:
        pretrained_imputation_model = None

    return pretrained_imputation_model


def setup_logging(date_format, log_format, verbose):
    """
    Set up all loggers to use the same format and date format.

    Args:
        date_format: Format for the date.
        log_format: Format for the log.
        verbose: Whether to log debug messages.
    """
    logging.basicConfig(format=log_format, datefmt=date_format)
    loggers = ["pytorch_lightning", "lightning_fabric"]
    for logger in loggers:
        logger_obj = logging.getLogger(logger)
        if logger_obj.handlers:  # Only configure if handlers exist
            logger_obj.handlers[0].setFormatter(logging.Formatter(log_format, datefmt=date_format))

    if not verbose:
        logging.getLogger().setLevel(logging.INFO)
        for logger in loggers:
            logging.getLogger(logger).setLevel(logging.INFO)
        warnings.filterwarnings("ignore")
    else:
        logging.getLogger().setLevel(logging.DEBUG)
        for logger in loggers:
            logging.getLogger(logger).setLevel(logging.DEBUG)
        warnings.filterwarnings("default")


def get_config_files(config_dir: Path):
    """
    Get all task and model config files in the specified directory.
    Args:
        config_dir: Name of the directory containing the config gin files.

    Returns:
        tasks: List of task names
        models: List of model names
    """
    try:
        tasks = list((config_dir / "tasks").glob("*"))
        models = list((config_dir / "prediction_models").glob("*"))
        tasks = [task.stem for task in tasks if task.is_file()]
        models = [model.stem for model in models if model.is_file()]
    except Exception as e:
        logging.error(f"Error retrieving config files: {e}")
        return [], []
    if "common" in tasks:
        tasks.remove("common")
    if "common" in models:
        models.remove("common")
    logging.info(f"Found tasks: {tasks}")
    logging.info(f"Found models: {models}")
    return tasks, models


def check_required_keys(vars, required_keys):
    """
    Checks if all required keys are present in the vars dictionary.

    Args:
        vars (dict): The dictionary to check.
        required_keys (list): The list of required keys.

    Raises:
        KeyError: If any required key is missing.
    """
    missing_keys = [key for key in required_keys if key not in vars]
    if missing_keys:
        raise KeyError(f"Missing required keys in vars: {', '.join(missing_keys)}")

SHIFT = 1_000_000_000  # used to make duplicated stay occurrences unique

def _make_unique_int64_by_offset(
    df_in: pl.DataFrame,
    id_col: str = "stay_id",
    original_id_col: str = "stay_id_original",
    dup_ix_col: str = "_dup_ix",
    shift: int = SHIFT,
) -> pl.DataFrame:
    return df_in.with_columns(
        pl.when(pl.col(dup_ix_col) == 0)
        .then(pl.col(original_id_col))
        .otherwise(pl.col(original_id_col) + pl.col(dup_ix_col) * shift)
        .alias(id_col)
    )


def _top_up_round_robin(
    df_one_row_per_id: pl.DataFrame,
    id_col: str,
    n_needed: int,
    seed: Optional[int],
) -> pl.DataFrame:
    """
    Create n_needed extra samples by cycling through unique ids (shuffled),
    so no id gets a 2nd extra copy until all ids got 1 extra copy.
    Assumes df_one_row_per_id has exactly 1 row per id_col.
    Returns duplicated rows (via join).
    """
    if n_needed <= 0:
        return df_one_row_per_id.head(0)

    ids = df_one_row_per_id.select(id_col).to_series().to_list()
    if len(ids) == 0:
        return df_one_row_per_id.head(0)

    import random
    rng = random.Random(seed)
    rng.shuffle(ids)

    reps = (n_needed + len(ids) - 1) // len(ids)
    picked = (ids * reps)[:n_needed]

    picked_df = pl.DataFrame({id_col: picked})
    return picked_df.join(df_one_row_per_id, on=id_col, how="left")


def downsample_binary_classification(
    df: pl.DataFrame,
    label_col: str,
    total_samples: int,
    seed: Optional[int] = None,
    id_col: str = "stay_id",
    shift: int = SHIFT,
) -> pl.DataFrame:
    """
    Mortality24: single-row-per-stay classification.
    - Preserves class distribution
    - If total_samples > available, oversamples via round-robin
    - Makes oversampled copies get unique stay_id via SHIFT offsets
    """
    if df.height == 0:
        return df

    labels = df[label_col].unique().to_list()
    if len(labels) == 0:
        return df

    # Count classes
    label_counts = {}
    total_original = 0
    for label in labels:
        c = df.filter(pl.col(label_col) == label).height
        label_counts[label] = c
        total_original += c

    # Target samples per class
    label_to_n = {
        label: int(round((count / total_original) * total_samples))
        for label, count in label_counts.items()
    }

    # Fix rounding to sum exactly total_samples
    current = sum(label_to_n.values())
    diff = total_samples - current
    if diff != 0:
        largest_label = max(label_to_n, key=label_to_n.get)
        label_to_n[largest_label] += diff

    # Sample per class
    samples = []
    for label, n_label in label_to_n.items():
        df_label = df.filter(pl.col(label_col) == label)
        available = df_label.height

        n_no_replace = min(n_label, available)
        n_needed = max(0, n_label - available)

        if n_no_replace > 0:
            samples.append(df_label.sample(n=n_no_replace, with_replacement=False, seed=seed))

        if n_needed > 0:
            logging.warning(
                f"[Mortality24/CLS] Oversampling label={label}: requested {n_label}, available {available}. "
                f"Top-up {n_needed} via round-robin."
            )
            samples.append(_top_up_round_robin(df_label, id_col=id_col, n_needed=n_needed, seed=seed))

    combined = pl.concat(samples)

    # Make duplicate occurrences unique
    combined = combined.with_columns(
        pl.col(id_col).alias("stay_id_original")
    ).with_columns(
        pl.cum_count("stay_id_original").over("stay_id_original").alias("_dup_ix")
    ).with_columns(
        (pl.col("_dup_ix") - pl.col("_dup_ix").min().over("stay_id_original")).alias("_dup_ix")
    )

    combined = _make_unique_int64_by_offset(
        combined,
        id_col=id_col,
        original_id_col="stay_id_original",
        dup_ix_col="_dup_ix",
        shift=shift,
    ).drop(["_dup_ix", "stay_id_original"])

    # Shuffle output
    return combined.sample(n=combined.height, with_replacement=False, seed=seed)


def downsample_aki_classification(
    df: pl.DataFrame,
    label_col: str,
    total_stays: int,
    seed: Optional[int] = None,
    id_col: str = "stay_id",
    shift: int = SHIFT,
) -> pl.DataFrame:
    """
    AKI: multi-row-per-stay with boolean label (timestep-level).
    - Computes stay-level label = any(label_col)
    - Samples stays preserving stay-level label distribution
    - Oversamples stays via round-robin if needed
    - Expands back to rows by join
    - Makes oversampled stay occurrences unique via SHIFT offsets
    """
    if df.height == 0:
        return df

    # Guard: label should be boolean for AKI
    if df.schema[label_col] != pl.Boolean:
        raise ValueError(f"AKI expects boolean label_col='{label_col}', got {df.schema[label_col]}")

    stay_level = (
        df.group_by(id_col)
        .agg(pl.col(label_col).max().alias("stay_label"))
    )
    if stay_level.height == 0:
        return df

    n_stays_to_sample = min(total_stays, stay_level.height)

    labels = stay_level["stay_label"].unique().to_list()
    label_counts = {}
    total_original = 0
    for lab in labels:
        c = stay_level.filter(pl.col("stay_label") == lab).height
        label_counts[lab] = c
        total_original += c

    label_to_n = {
        lab: int(round((count / total_original) * n_stays_to_sample))
        for lab, count in label_counts.items()
    }

    # Fix rounding
    current = sum(label_to_n.values())
    diff = n_stays_to_sample - current
    if diff != 0:
        largest_label = max(label_to_n, key=label_to_n.get)
        label_to_n[largest_label] += diff

    samples = []
    for lab, n_lab in label_to_n.items():
        df_lab = stay_level.filter(pl.col("stay_label") == lab)
        available = df_lab.height

        n_no_replace = min(n_lab, available)
        n_needed = max(0, n_lab - available)

        if n_no_replace > 0:
            samples.append(df_lab.sample(n=n_no_replace, with_replacement=False, seed=seed))

        if n_needed > 0:
            logging.warning(
                f"[AKI] Oversampling stay_label={lab}: requested {n_lab}, available {available}. "
                f"Top-up {n_needed} via round-robin."
            )
            samples.append(_top_up_round_robin(df_lab, id_col=id_col, n_needed=n_needed, seed=seed))

    sampled_stays = pl.concat(samples)

    # duplicate index per stay occurrence
    sampled_stays = sampled_stays.with_columns(
        pl.cum_count(id_col).over(id_col).alias("_dup_ix")
    ).with_columns(
        (pl.col("_dup_ix") - pl.col("_dup_ix").min().over(id_col)).alias("_dup_ix")
    )

    # expand to rows
    result = sampled_stays.join(df, on=id_col, how="left").with_columns(
        pl.col(id_col).alias("stay_id_original")
    )

    # shift ids for duplicates
    result = _make_unique_int64_by_offset(
        result,
        id_col=id_col,
        original_id_col="stay_id_original",
        dup_ix_col="_dup_ix",
        shift=shift,
    )

    return result.drop(["_dup_ix", "stay_label", "stay_id_original"])


def downsample_los_regression(
    df: pl.DataFrame,
    total_stays: int,
    seed: Optional[int] = None,
    id_col: str = "stay_id",
) -> pl.DataFrame:
    """
    LOS: regression, multi-row-per-stay (timesteps).
    - Samples stays uniformly (no oversampling)
    """
    if df.height == 0:
        return df

    unique_stays = df.select(id_col).unique()
    if unique_stays.height == 0:
        return df

    n = min(total_stays, unique_stays.height)
    sampled = unique_stays.sample(n=n, with_replacement=False, seed=seed)
    stay_ids = sampled.select(id_col).to_series().to_list()
    return df.filter(pl.col(id_col).is_in(stay_ids))

def downsample_outcome_by_task(
    df_outcome: pl.DataFrame,
    task_name: str,
    subset_size: int,
    subset_seed: int,
) -> pl.DataFrame:

    if task_name == "Mortality24":
        return downsample_binary_classification(
            df=df_outcome,
            label_col="label",
            total_samples=subset_size,
            seed=subset_seed,
        )
    elif task_name == "AKI":
        return downsample_aki_classification(
            df=df_outcome,
            label_col="label",
            total_stays=subset_size,
            seed=subset_seed,
        )
    elif task_name in ("LOS", "LengthOfStay"):
        return downsample_los_regression(
            df=df_outcome,
            total_stays=subset_size,
            seed=subset_seed,
        )
    else:
        raise ValueError(
            f"Unknown task_name='{task_name}'. Expected one of: Mortality24, AKI, LOS."
        )

