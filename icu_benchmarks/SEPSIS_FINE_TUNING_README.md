## Overview

The Sepsis fine-tuning workflow follows the standard YAIB pipeline:

1. Start with cohort data in standard YAIB format (`dyn.parquet`, `sta.parquet`, `outc.parquet`)
2. Generate preprocessed subsets by running baseline training with subset flags
3. Fine-tune pretrained models using the preprocessed subsets

## Relevant Files

### Configuration Files
- **`configs/tasks/BinaryClassification.gin`**: Task configuration for binary classification tasks 

### Scripts
- **`icu_benchmarks/fine_tuning.py`**: Fine-tuning script for classification tasks like Sepsis

### Data Loader Updates
- **`icu_benchmarks/data/loader.py`**:
  - Added explicit type casting for features to avoid numpy object arrays (lines 583-597) - **Bug fix**
  - Improved label padding to handle both timestep-level (sepsis, LOS) and patient-level (mortality) tasks (lines 512-520) - **Improvement**


## Standard YAIB Workflow

### Step 1: Generate Preprocessed Subsets

Generate preprocessed training subsets by running baseline training with the `subset_train_size` flag. For example:

```bash
# Loop through sizes and seeds to create all subsets
for size in 100 500 1000 2000 3000 5000; do
  for seed in 42 84 126 168 210; do
    icu-benchmarks train \
      -d path/to/data \
      -n mimic_sepsis \
      -t BinaryClassification \
      -tn Sepsis \
      -m LGBMClassifier \
      -gc \
      -lc \
      -s 2222 \
      -l yaib_logs/ \
      -hp execute_repeated_cv.subset_train_size=$size \
          execute_repeated_cv.subset_train_seed=$seed
  done
done
```

### Step 3: Fine-Tune the Pretrained Model

Now fine-tune using the preprocessed subsets generated in Step 1:

```bash
python icu_benchmarks/fine_tuning.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset mimic_sepsis \
    --sizes 100,500,1000,2000,3000,5000 \
    --seeds 42,84,126,168,210 \
    --bz 64 \
    --lr 0.0001 \
    --num_epochs 200 \
    --subset_root icu_benchmarks/data/preprocessed_data \
    --gin_config configs/tasks/BinaryClassification.gin
```

## Using the Full YAIB Pipeline (Alternative)

If you want to train a model from scratch (not fine-tuning), you can use the full YAIB pipeline:

```bash
# Train from scratch
icu-benchmarks train \
    -d data/sepsis_from_katja/mimic \
    -n mimic_sepsis \
    -t BinaryClassification \
    -tn Sepsis \
    -m BAT \
    -gc \
    -lc \
    -s 42 \
    -l yaib_logs/


