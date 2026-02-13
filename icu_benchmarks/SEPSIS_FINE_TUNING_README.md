## Overview

The Sepsis fine-tuning workflow uses a standalone fine-tuning script that supports timestep-level prediction for sepsis. 

Workflow:
1. Start with cohort data in standard YAIB format (`dyn.parquet`, `sta.parquet`, `outc.parquet`)
2. Generate preprocessed subsets by running baseline training with subset flags
3. Fine-tune pretrained models using the task-aware fine-tuning script

## Relevant Files

### Scripts
- **[`icu_benchmarks/fine_tuning.py`](fine_tuning.py)**: Task-aware fine-tuning script supporting both timestep-level (Sepsis) and patient-level (Mortality) classification
  - **Key features:**
    - `--task` parameter selects appropriate prediction head (`TimeseriesClassificationHead` for Sepsis, `BinaryClassificationHead` for Mortality)
    - Proper masking using `obs_mask` to handle variable-length sequences
    - Timestep-level loss and metrics for Sepsis (evaluates every timestep prediction)
    - Patient-level loss and metrics for Mortality

### Model Architecture
- **[`icu_benchmarks/models/dl_models/bat.py`](models/dl_models/bat.py)**:
  - `TimeseriesClassificationHead` (lines 33-46): Outputs predictions at every timestep `(B, T, num_classes)`
  - `BinaryClassificationHead` (lines 14-21): Outputs single prediction per patient `(B, num_classes)`
  - `AutoregressiveEncoderCrossParallel` (lines 202-461): Encoder that returns per-timestep representations for sequential prediction

### Data Loader
- **[`icu_benchmarks/data/loader.py`](data/loader.py)**:
  - `BATPolarsDataset` with collate function that returns `obs_mask` (line 523-528) to identify valid (non-padded) timesteps
  - Handles both timestep-level labels `(B, T)` for Sepsis and patient-level labels `(B,)` for Mortality (lines 512-520)


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
      -m Transformer_tuned_mimic \
      -gc \
      -lc \
      -s 2222 \
      -l yaib_logs/ \
      -hp execute_repeated_cv.subset_train_size=$size \
          execute_repeated_cv.subset_train_seed=$seed
  done
done
```

### Step 2: Fine-Tune the Pretrained Model

Now fine-tune using the preprocessed subsets generated in Step 1. **Important:** Use the `--task` flag to specify the task type:

```bash
# Sepsis fine-tuning (timestep-level prediction)
python icu_benchmarks/fine_tuning.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset mimic \
    --task Sepsis \
    --sizes 100,500,1000,2000,3000,5000 \
    --seeds 42,84,126,168,210 \
    --bz 64 \
    --lr 0.0001 \
    --num_epochs 200 \
    --subset_root icu_benchmarks/data/preprocessed_data

# Mortality fine-tuning (patient-level prediction, for comparison)
python icu_benchmarks/fine_tuning.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset mimic \
    --task Mortality24 \
    --sizes 100,500,1000,2000,3000,5000 \
    --seeds 42,84,126,168,210 \
    --bz 64 \
    --lr 0.0001 \
    --num_epochs 200 \
    --subset_root icu_benchmarks/data/preprocessed_data
```

**Arguments:**
- `--task`: Task type - determines prediction head and label handling
  - `Sepsis`, `AKI`: Timestep-level prediction using `TimeseriesClassificationHead` (hourly predictions within 6H window)
  - `Mortality24`, `Mortality`: Patient-level prediction using `BinaryClassificationHead`
- `--fine_tune_head`: (Optional) Only fine-tune the classification head, freeze encoder
- `--dataset`: Dataset name (e.g., `mimic`, `eicu`, `miiv`)
- `--sizes`: Training set sizes (comma-separated or range format `100:5000:100`)
- `--seeds`: Random seeds for subset generation

## How Timestep-Level Prediction Works for Sepsis

### Example: Single Patient Trajectory

For a patient with a 7-hour ICU stay where sepsis onset occurs at hour 4:

```
Timesteps:     [t0,  t1,  t2,  t3,  t4,  t5,  t6,  pad, pad]
Labels:        [ 0,   0,   0,   0,   1,   1,   1,   -,   - ]
obs_mask:      [ 1,   1,   1,   1,   1,   1,   1,   0,   0 ]
Model outputs: [(B,T,C)] -> predictions at each timestep
```

**Key behaviors:**
1. **Prediction head:** `TimeseriesClassificationHead` outputs `(B, T, 2)` logits (2 classes: no sepsis, sepsis)
2. **Loss computation:** Only computed on valid timesteps (where `obs_mask=1`), ignoring padded positions
3. **Metrics:** Evaluates predictions at **every valid timestep**, not just patient-level max
   - For the example above, model makes 7 predictions (one per timestep)
   - Metrics (AUROC, AUPRC) computed across all valid timestep predictions
   - This captures the model's ability to detect sepsis onset timing

### Comparison: Patient-Level vs Timestep-Level

| Aspect | Mortality (Patient-Level) | Sepsis (Timestep-Level) |
|--------|---------------------------|-------------------------|
| **Prediction head** | `BinaryClassificationHead` | `TimeseriesClassificationHead` |
| **Output shape** | `(B, 2)` - one per patient | `(B, T, 2)` - one per timestep |
| **Labels** | Single value per patient | Sequence of values over time |
| **Loss** | Cross-entropy on patient outcome | Cross-entropy on all valid timesteps |
| **Metrics** | Patient-level AUROC/AUPRC | Timestep-level AUROC/AUPRC |
| **Clinical interpretation** | "Will patient die?" | "When does sepsis onset occur?" |

## Technical Implementation Details

### Code Changes in `fine_tuning.py`

**1. Task parameter determines model architecture:**
```python
# In build_model_from_ckpt()
TIMESTEP_TASKS = {"Sepsis", "AKI"}  # Line 152

if task in TIMESTEP_TASKS:
    # Sepsis: TimeseriesClassificationHead
    classification_model = EncoderPrediction(
        encoder_class=encoder,
        prediction_head=TimeseriesClassificationHead,
        prediction_head_kwargs={"num_classes": 2},
    )
else:
    # Mortality: BinaryClassificationHead
    classification_model = EncoderPrediction(
        encoder_class=encoder,
        prediction_head=BinaryClassificationHead,
        prediction_head_kwargs={"num_classes": 2},
    )
```

**2. Proper masking using `obs_mask`:**
```python
# Training loop (lines 255-305)
x, mask, label, times, static, delta, obs_mask = batch
obs_mask = obs_mask.to(device).bool()  # (B, T) - identifies valid timesteps

if is_timestep_task:
    # Flatten for loss computation
    logits_flat = logits.reshape(B * T, C)  # (B*T, num_classes)
    label_flat = label.reshape(B * T)       # (B*T,)
    obs_mask_flat = obs_mask.reshape(B * T) # (B*T,)

    # Only compute loss on valid positions
    loss = loss_fn(logits_flat[obs_mask_flat], label_flat[obs_mask_flat])

    # Collect all valid timestep predictions for metrics
    probs = F.softmax(logits, dim=-1)[:, :, 1]  # (B, T)
    valid_probs = probs[obs_mask]
    valid_labels = label[obs_mask]
```

**3. Metrics evaluate every timestep:**
- For Sepsis: AUROC/AUPRC computed across all valid timestep predictions
- Correctly captures temporal dynamics of sepsis onset
- No aggregation to patient-level (unlike the old buggy version)



## Using the Full YAIB Pipeline (Alternative)

If you want to train a model from scratch (not fine-tuning), you can use the full YAIB pipeline:

```bash
# Train from scratch using BinaryClassification task
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


