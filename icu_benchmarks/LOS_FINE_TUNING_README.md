# Length of Stay (LOS) Prediction 

This document describes the new training and fine-tuning tasks for Length of Stay (LOS) prediction, which is a regression task that predicts the length (in hours) of a patient's ICU stay.

## Relevant Files

### Configuration Files
- **`configs/tasks/LengthOfStay.gin`**: Task configuration for LOS prediction (works for both main YAIB pipeline and fine-tuning)

### Scripts
- **`icu_benchmarks/data/prepare_fine_tuning_data.py`**: General-purpose data preprocessing script for both classification and regression tasks
- **`icu_benchmarks/fine_tuning_regression.py`**: Fine-tuning script specifically for regression tasks like LOS

### Model Updates
- **`icu_benchmarks/models/dl_models/bat.py`**:
  - Added `RegressionHead` class for regression predictions (line 24-30)

### Data Loader Updates
- **`icu_benchmarks/data/loader.py`**:
  - Added handling for numeric time columns (line 51-53)
  - Added handling for numeric time in dataset __getitem__ (line 664-672)
  - Added explicit type casting for features to avoid numpy object arrays (line 581-596)
  - Added regression-specific label handling (single value per stay) (line 561-570)

## Finetuning Quick Start

### 1. Prepare the Data

The demo LOS data is already available in `demo_data/los/mimic_demo/`. Convert it to the fine-tuning format:

```bash
python icu_benchmarks/data/prepare_fine_tuning_data.py \
    --input_dir demo_data/los/mimic_demo \
    --output_dir icu_benchmarks/data/preprocessed_data/mimic_los_regression \
    --task_type regression \
    --size 100 \
    --seed 42 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --add_missing_indicators
```

**Parameters:**
- `--input_dir`: Directory containing `dyn.parquet`, `sta.parquet`, `outc.parquet`
- `--output_dir`: Where to save preprocessed data
- `--task_type`: `classification` or `regression`
- `--size`: Dataset size identifier (for path naming)
- `--train_ratio`: Proportion of data for training (default: 0.7)
- `--val_ratio`: Proportion for validation (default: 0.15)
- `--add_missing_indicators`: Add missing indicator columns required by some models

**Output Structure:**
```
icu_benchmarks/data/preprocessed_data/mimic_los_regression/100_42/
├── train_FEATURES.parquet
├── train_OUTCOME.parquet
├── val_FEATURES.parquet
├── val_OUTCOME.parquet
├── test_FEATURES.parquet
└── test_OUTCOME.parquet
```

### 2. Download Pretrained Model

Download a pretrained BAT model from HuggingFace. To use the model pretrained on eICU and MIMIC-IV, use the following command:

```bash
mkdir -p pretrained_checkpoints
cd pretrained_checkpoints
huggingface-cli download Katja-Jagd/bat-pretrained-eicu-mimiciv-base model.ckpt \
    --local-dir . --local-dir-use-symlinks False
cd ..
```

For other pretrained models, visit: https://huggingface.co/Katja-Jagd/models

### 3. Fine-Tune the Model

Run fine-tuning with your chosen hyperparameters:

```bash
# IMPORTANT: Replace with your actual path or use relative path from YAIB root
python icu_benchmarks/fine_tuning_regression.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset mimic_los_regression \
    --sizes 100 \
    --seeds 42 \
    --bz 64 \
    --lr 0.0001 \
    --num_epochs 200 \
    --subset_root icu_benchmarks/data/preprocessed_data  # Relative path (recommended)
    # OR use absolute path:
    # --subset_root /your/actual/path/to/YAIB/icu_benchmarks/data/preprocessed_data
```

**Output:**
Results are saved to `finetuning_results/pretrained_BAT/{dataset}/{mode}/`:
- `runs_{sweep_id}.jsonl`: Per-run results
- `summary_{sweep_id}.csv`: Aggregated results table
- `meta_{sweep_id}.json`: Run metadata


## Using with Main YAIB Pipeline

While the fine-tuning script is standalone, you can also use the full YAIB pipeline with the LOS task:

```bash
icu-benchmarks \
    -d demo_data/los \
    -n mimic_demo \
    -t Regression \
    -tn LengthOfStay \
    -m LGBMRegressor \
    -gc \
    -lc \
    -s 42 \
    -l yaib_logs
```

For hyperparameter tuning:
```bash
icu-benchmarks train \
    -d demo_data/los \
    -n mimic_demo \
    -t Regression \
    -tn LengthOfStay \
    -m LGBMRegressor \
    --tune
```

## Comparison: Classification vs Regression Tasks

### Classification (Mortality24)
```bash
# Data prep
python icu_benchmarks/data/prepare_fine_tuning_data.py \
    --input_dir demo_data/mortality24/mimic_demo \
    --output_dir icu_benchmarks/data/preprocessed_data/mimic_mortality \
    --task_type classification \
    --size 100 --seed 42 --add_missing_indicators

# Fine-tune
python icu_benchmarks/fine_tuning.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset mimic_mortality \
    --sizes 100 --seeds 42 --bz 64 --lr 0.0001 --num_epochs 200 \
    --subset_root /absolute/path/to/icu_benchmarks/data/preprocessed_data
```

### Regression (LengthOfStay)
```bash
# Data prep
python icu_benchmarks/data/prepare_fine_tuning_data.py \
    --input_dir demo_data/los/mimic_demo \
    --output_dir icu_benchmarks/data/preprocessed_data/mimic_los_regression \
    --task_type regression \
    --size 100 --seed 42 --add_missing_indicators

# Fine-tune
python icu_benchmarks/fine_tuning_regression.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset mimic_los_regression \
    --sizes 100 --seeds 42 --bz 64 --lr 0.0001 --num_epochs 200 \
    --subset_root /absolute/path/to/icu_benchmarks/data/preprocessed_data
```