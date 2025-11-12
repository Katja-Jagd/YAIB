# Sepsis Prediction Fine-Tuning Task

This document describes the new classification fine-tuning task for predicting sepsis in ICU patients using pretrained BAT models with Physionet 2019 Challenge data.

## Overview

Sepsis prediction is a binary classification task that identifies patients at risk of developing sepsis. This implementation allows you to fine-tune pretrained BAT (Bi-Attentive Transformer) models on the Physionet 2019 Sepsis Challenge dataset, which contains hourly clinical measurements and binary sepsis labels.

**Dataset**: Physionet 2019 Sepsis Challenge (p19)
**Task Type**: Binary Classification
**Prediction Horizon**: 24 hours
**Total Patients**: 40,333
**Class Balance**: Highly imbalanced (~1.8% positive rate)

## Files Created

### Configuration Files
- **`configs/tasks/BinaryClassification.gin`**: Standard binary classification configuration (used for Sepsis, Mortality, AKI, etc.)

### Scripts
- **`icu_benchmarks/data/prepare_sepsis_data.py`**: Data preprocessing script to convert Physionet 2019 NumPy format to YAIB parquet format

### Preprocessed Data
- **`icu_benchmarks/data/preprocessed_data/p19/40333_42/`**: Preprocessed parquet files ready for training

## Quick Start

### 1. Data Preparation (Already Done)

The sepsis data has already been preprocessed from the NumPy format. The preprocessed data is located at:
```
icu_benchmarks/data/preprocessed_data/p19/40333_42/
```

If you need to re-run preprocessing or use different split ratios:

```bash
python icu_benchmarks/data/prepare_sepsis_data.py \
    --input_dir /isdata/winthergrp/gsn245/scratch/Patient_Journey_Classification/P19data/split_1 \
    --output_dir icu_benchmarks/data/preprocessed_data/p19 \
    --seed 42 \
    --train_ratio 0.7 \
    --val_ratio 0.15
```

**Parameters:**
- `--input_dir`: Directory containing train/val/test NumPy files (train_physionet2019_1.npy, etc.)
- `--output_dir`: Where to save preprocessed parquet files
- `--seed`: Random seed for reproducible train/val/test splits
- `--train_ratio`: Proportion of data for training (default: 0.7)
- `--val_ratio`: Proportion for validation (default: 0.15)

**Output Structure:**
```
icu_benchmarks/data/preprocessed_data/p19/40333_42/
├── train_FEATURES.parquet  # 28,233 patients, 1,086,289 timesteps
├── train_OUTCOME.parquet
├── val_FEATURES.parquet    # 6,049 patients, 231,906 timesteps
├── val_OUTCOME.parquet
├── test_FEATURES.parquet   # 6,051 patients, 233,898 timesteps
└── test_OUTCOME.parquet
```

### 2. Download Pretrained Model

Download the pretrained BAT model from HuggingFace:

```bash
mkdir -p pretrained_checkpoints
cd pretrained_checkpoints
huggingface-cli download Katja-Jagd/bat-pretrained-eicu-mimiciv-base model.ckpt \
    --local-dir . --local-dir-use-symlinks False
cd ..
```

Or visit: https://huggingface.co/Katja-Jagd/bat-pretrained-eicu-mimiciv-base

### 3. Fine-Tune the Model

Run fine-tuning with your desired hyperparameters:

```bash
# Basic example (adjust CUDA_VISIBLE_DEVICES for your GPU)
CUDA_VISIBLE_DEVICES=0 python icu_benchmarks/fine_tuning.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset p19 \
    --sizes 40333 \
    --seeds 42 \
    --bz 32 \
    --lr 0.0001 \
    --num_epochs 100 \
    --subset_root icu_benchmarks/data/preprocessed_data \
    --gin_config configs/tasks/BinaryClassification.gin
```

**Note on Feature Mapping**: The script automatically maps P19's 34 features to MIMIC's 48-feature space:
- 32/34 P19 features map to corresponding MIMIC features (HR→hr, SBP→sbp, etc.)
- 2 P19 features (EtCO2, Hct) are dropped as they don't exist in MIMIC
- 17 MIMIC features are zero-padded (not present in P19 data)
- This allows using the pretrained MIMIC/eICU model without retraining from scratch

**Key Parameters:**
- `--model_path`: Path to pretrained checkpoint (.ckpt file)
- `--dataset`: Dataset name (must match directory under subset_root: `p19`)
- `--sizes`: Training sizes to test (e.g., "100,500,1000" or "100:1000:100")
- `--seeds`: Random seeds for multiple runs (e.g., "42,84,126")
- `--fine_tune_head`: Flag to only fine-tune the classification head (freeze encoder)
- `--bz`: Batch size
- `--lr`: Learning rate
- `--num_epochs`: Maximum number of epochs (early stopping enabled)
- `--subset_root`: Root directory containing preprocessed data
- `--gin_config`: Optional gin config (can specify `configs/tasks/BinaryClassification.gin`)

**Output:**
Results are saved to `finetuning_results/pretrained_BAT/p19/{mode}/`:
- `runs_{sweep_id}.jsonl`: Per-run results
- `summary_{sweep_id}.csv`: Aggregated results table
- `meta_{sweep_id}.json`: Run metadata

## Dataset Information

### Physionet 2019 Sepsis Challenge Data

The dataset contains hourly clinical measurements from ICU patients:

**34 Dynamic Features (Time-Varying):**
- **Vital Signs**: HR (Heart Rate), O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2
- **Laboratory Values**:
  - Blood Gases: BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2
  - Liver Function: AST, Alkalinephos, Bilirubin_direct, Bilirubin_total
  - Kidney Function: BUN, Creatinine
  - Electrolytes: Calcium, Chloride, Magnesium, Phosphate, Potassium
  - Other: Glucose, Lactate, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets

**4 Static Features (Patient-Level):**
- Age (normalized)
- Gender (binary)
- Unit1, Unit2 (ICU unit indicators)

**Labels:**
- Binary classification: 0 (no sepsis) vs 1 (sepsis)
- Label at each timestep indicating sepsis status
- Highly imbalanced: ~98.2% negative, ~1.8% positive

**Missing Indicators:**
- Each dynamic feature has a corresponding `MissingIndicator_<feature>` column
- Indicates whether the feature was measured at that timestep
- Required by the BAT model architecture

### Data Statistics

| Split | Patients | Timesteps | Sepsis Cases | Positive Rate |
|-------|----------|-----------|--------------|---------------|
| Train | 28,233 | 1,086,289 | ~19,380 | ~1.8% |
| Val | 6,049 | 231,906 | ~4,100 | ~1.8% |
| Test | 6,051 | 233,898 | ~4,120 | ~1.8% |

## Detailed Usage

### Multiple Training Sizes

To test different training sizes, you'll first need to create subsets. For now, use the full dataset (40333 patients):

```bash
CUDA_VISIBLE_DEVICES=0 python icu_benchmarks/fine_tuning.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset p19 \
    --sizes 40333 \
    --seeds 42 \
    --bz 32 \
    --lr 0.0001 \
    --num_epochs 100 \
    --subset_root icu_benchmarks/data/preprocessed_data \
    --gin_config configs/tasks/BinaryClassification.gin
```

**Note**: Creating smaller training subsets (100, 500, 1000 patients) requires additional preprocessing steps not yet implemented.

### Multiple Random Seeds

Run with multiple random seeds for statistical robustness:

```bash
CUDA_VISIBLE_DEVICES=0 python icu_benchmarks/fine_tuning.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset p19 \
    --sizes 40333 \
    --seeds "42,84,126" \
    --bz 32 \
    --lr 0.0001 \
    --num_epochs 100 \
    --subset_root icu_benchmarks/data/preprocessed_data \
    --gin_config configs/tasks/BinaryClassification.gin
```

### Head-Only Fine-Tuning

Fine-tune only the classification head (faster, less memory):

```bash
CUDA_VISIBLE_DEVICES=0 python icu_benchmarks/fine_tuning.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset p19 \
    --sizes 40333 \
    --seeds 42 \
    --fine_tune_head \
    --bz 64 \
    --lr 0.001 \
    --num_epochs 50 \
    --subset_root icu_benchmarks/data/preprocessed_data \
    --gin_config configs/tasks/BinaryClassification.gin
```

### Full Model Fine-Tuning (Recommended)

Fine-tune the entire model for best performance:

```bash
CUDA_VISIBLE_DEVICES=0 python icu_benchmarks/fine_tuning.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset p19 \
    --sizes 40333 \
    --seeds 42,84,126 \
    --bz 32 \
    --lr 0.0001 \
    --num_epochs 100 \
    --subset_root icu_benchmarks/data/preprocessed_data \
    --gin_config configs/tasks/BinaryClassification.gin
```

## Using with Main YAIB Pipeline

You can also use the full YAIB pipeline for training from scratch or with other models:

```bash
icu-benchmarks \
    -d icu_benchmarks/data/preprocessed_data/p19/40333_42 \
    -n sepsis_experiment \
    -t BinaryClassification \
    -tn Sepsis \
    -m LGBMClassifier \
    -gc \
    -lc \
    -s 42 \
    -l yaib_logs
```

For hyperparameter tuning:
```bash
icu-benchmarks train \
    -d icu_benchmarks/data/preprocessed_data/p19/40333_42 \
    -n sepsis_experiment \
    -t BinaryClassification \
    -tn Sepsis \
    -m LGBMClassifier \
    --tune
```

## Using AutoregressiveBAT for Time Series Sepsis Prediction

**NEW**: The repository now includes `AutoregressiveBAT`, a causal variant of BAT designed specifically for autoregressive time series prediction. This model is useful when you want per-timestep predictions rather than single patient-level outcomes.

### What is AutoregressiveBAT?

AutoregressiveBAT is an autoregressive version of the BAT (Bi-Attentive Transformer) model that uses **causal attention** for time series prediction:

- **Causal Time Attention**: At each timestep t, the model only attends to past timesteps (t' ≤ t), ensuring predictions don't "peek" into the future
- **Sensor Attention**: Still operates non-causally within each timestep (across different sensors/features)
- **Per-Timestep Predictions**: Outputs predictions for every timestep rather than a single aggregated prediction
- **Architecture**: Uses `Decoder` layers for time attention (causal) and `Encoder` layers for sensor attention (non-causal)

**When to use AutoregressiveBAT:**
- Predicting sepsis risk at **every hour** during patient stay
- Early warning systems that need continuous risk monitoring
- Analyzing temporal progression of sepsis risk
- Tasks requiring interpretable per-timestep predictions

**When to use regular BAT:**
- Predicting overall patient outcome (will sepsis occur during stay?)
- Binary patient-level classification
- Standard mortality or readmission prediction

### Training with AutoregressiveBAT

**IMPORTANT**: AutoregressiveBAT requires a BAT-specific configuration. You CANNOT use the standard `Regression.gin` config with AutoregressiveBAT (that config is for other models like Transformer, LSTM, etc.).

#### Training from Scratch

```bash
icu-benchmarks \
    -d icu_benchmarks/data/preprocessed_data/p19/40333_42 \
    -n sepsis_autoregressive \
    -t Regression \
    -tn Sepsis_Timeseries \
    -m AutoregressiveBAT \
    -gc \
    -lc \
    -s 42 \
    -l yaib_logs \
    --model-config configs/prediction_models/AutoregressiveBAT_Sepsis.gin
```

**Key Parameters:**
- `-m AutoregressiveBAT`: Use the autoregressive BAT model
- `--model-config configs/prediction_models/AutoregressiveBAT_Sepsis.gin`: **Required** BAT-specific config
- `-t Regression`: Task type (AutoregressiveBAT outputs continuous values per timestep)

#### Fine-Tuning a Pretrained Model

AutoregressiveBAT is designed for training from scratch on regression tasks. For fine-tuning pretrained models on classification, use the standard `fine_tuning.py` script with regular BAT.

### AutoregressiveBAT vs Regular BAT

| Feature | Regular BAT | AutoregressiveBAT |
|---------|-------------|-------------------|
| **Attention Type** | Bidirectional (time & sensors) | Causal (time), Bidirectional (sensors) |
| **Output Shape** | (N,) or (N, classes) | (N, T) or (N, T, classes) |
| **Prediction Type** | Single per patient | Per-timestep predictions |
| **Use Case** | Patient-level outcomes | Continuous risk monitoring |
| **Training Data** | Classification/Regression | Regression with temporal labels |
| **Config File** | Standard task configs | `AutoregressiveBAT_Sepsis.gin` (BAT-specific) |
| **Compatible with** | `fine_tuning.py` | Train from scratch via `icu-benchmarks` CLI |

### AutoregressiveBAT Architecture Details

The model consists of two main components:

1. **AutoregressiveEncoderCrossParallel**:
   - Sensor attention layers: `Encoder` (non-causal, within timestep)
   - Time attention layers: `Decoder` (causal, across timesteps)
   - Returns (N, T, E) representations where each timestep's features only depend on past information

2. **Prediction Head**:
   - `RegressionHead`: For continuous sepsis risk scores per timestep
   - `TimeseriesClassificationHead`: For binary predictions per timestep
   - Applied to each timestep's representation independently

### Configuration File

The `configs/prediction_models/AutoregressiveBAT_Sepsis.gin` file contains BAT-specific settings:

```gin
# BAT-specific dataloader (required for BAT models)
BATPolarsDataset.runmode = "regression"

# AutoregressiveBAT model
train_common.model = @AutoregressiveBAT
train_common.dataset_class = @BATPolarsDataset

# Model hyperparameters
model/hyperparameter.value_embed_size = [8, 16, 32, 64]
model/hyperparameter.layers = [2, 6, 8, 12]
model/hyperparameter.heads = (1, 2)
model/hyperparameter.use_mask = True  # Handle missing values
model/hyperparameter.prediction_head = @RegressionHead
```

**Why this config is BAT-specific:**
- Uses `BATPolarsDataset` (optimized for BAT's data format)
- Configures BAT-specific hyperparameters (value_embed_size, sensor encoding)
- Sets up the cross-parallel attention architecture

**Other models** (Transformer, LSTM, GRU) should use the standard `configs/tasks/Regression.gin` which uses different data loaders and model configurations.

### Example: Comparing Regular BAT vs AutoregressiveBAT

```bash
# Regular BAT: Single patient-level prediction
# "Will this patient develop sepsis during their stay?" → Output: 0 or 1
icu-benchmarks \
    -d icu_benchmarks/data/preprocessed_data/p19/40333_42 \
    -n sepsis_patient_level \
    -t BinaryClassification \
    -tn Sepsis \
    -m BAT \
    -gc -lc -s 42 -l yaib_logs

# AutoregressiveBAT: Per-timestep predictions
# "What is the sepsis risk at each hour?" → Output: risk score for each hour
icu-benchmarks \
    -d icu_benchmarks/data/preprocessed_data/p19/40333_42 \
    -n sepsis_timeseries \
    -t Regression \
    -tn Sepsis_Timeseries \
    -m AutoregressiveBAT \
    -gc -lc -s 42 -l yaib_logs \
    --model-config configs/prediction_models/AutoregressiveBAT_Sepsis.gin
```

### Output Format

AutoregressiveBAT predictions have shape `(batch_size, num_timesteps)`:

```python
# Example: Batch of 32 patients, each with up to 72 timesteps
predictions.shape  # (32, 72)

# Each row is a patient's risk trajectory over time
patient_0_risk = predictions[0, :]  # [0.1, 0.15, 0.2, 0.3, 0.5, ...]
# Risk increases from 10% at hour 0 to 50% at hour 4
```

This allows for:
- Visualizing risk trajectories
- Early warning alerts when risk crosses threshold
- Analyzing temporal patterns in sepsis development
- Per-timestep evaluation metrics

### Testing Causality

To verify the model respects causality (doesn't use future information), run:

```bash
python test_autoregressive_bat.py
```

This test confirms that predictions at timestep t are unaffected by modifications to data at timesteps t+k.

## Training Configuration

### Early Stopping
- Monitors validation AUPRC (Area Under Precision-Recall Curve)
- Patience: 10 epochs without improvement
- Automatically restores best model weights
- AUPRC is preferred over AUROC for imbalanced datasets

### Learning Rate Scheduling
- ReduceLROnPlateau with patience of 5 epochs
- Factor: 0.5 (halves learning rate when plateau detected)
- Monitors validation AUPRC

### Evaluation Metrics
- **AUPRC** (Area Under Precision-Recall Curve): Primary metric for imbalanced classification
- **AUROC** (Area Under ROC Curve): Secondary metric
- **Balanced Accuracy**: Accounts for class imbalance
- **F1 Score**: Harmonic mean of precision and recall

### Class Weighting
- Uses `weight = "balanced"` to handle class imbalance
- Automatically adjusts loss function to penalize misclassification of minority class

## Preparing Your Own Sepsis Data

To prepare sepsis data from a different source:

1. **If starting from NumPy format** (like Physionet 2019):

   Your data should have dictionaries with these keys per sample:
   - `ts_values`: (timesteps, n_features) - dynamic feature values
   - `ts_indicators`: (timesteps, n_features) - missing value indicators (bool)
   - `ts_times`: (timesteps,) - time values
   - `static`: (n_static_features,) - static feature values
   - `labels`: (timesteps, 1) - binary labels

   Then run:
   ```bash
   python icu_benchmarks/data/prepare_sepsis_data.py \
       --input_dir /path/to/your/numpy/data \
       --output_dir icu_benchmarks/data/preprocessed_data/your_sepsis_dataset \
       --seed 42
   ```

2. **If starting from parquet format** (dyn.parquet, sta.parquet, outc.parquet):

   Use the general preprocessing script:
   ```bash
   python icu_benchmarks/data/prepare_fine_tuning_data.py \
       --input_dir /path/to/your/parquet/data \
       --output_dir icu_benchmarks/data/preprocessed_data/your_sepsis_dataset \
       --task_type classification \
       --size <your_size> \
       --seed 42 \
       --add_missing_indicators
   ```

3. **Required data format:**
   - Binary labels (0 or 1)
   - Timestamped observations
   - Consistent patient identifiers (stay_id)
   - All dynamic features should have corresponding missing indicators

## Technical Implementation Details

### P19 to MIMIC Feature Mapping

The fine-tuning script automatically handles the feature count mismatch between P19 (34 features) and MIMIC (48 features):

**Feature Mapping (`icu_benchmarks/p19_to_mimic_feature_map.py`)**:
- **Exact matches** (12): HR→hr, SBP→sbp, MAP→map, DBP→dbp, Temp→temp, etc.
- **Semantic matches** (20): BaseExcess→be, HCO3→bicar, Creatinine→crea, Glucose→glu, etc.
- **Dropped features** (2): EtCO2, Hct (not in MIMIC dataset)
- **Zero-padded** (17): MIMIC features not in P19 (alb, alt, na, urine, etc.)

This transformation happens automatically in the data loader collate function, allowing pretrained MIMIC/eICU models to work with P19 data.

### Data Processing Pipeline

1. **NumPy to Polars Conversion**:
   - Loads train/val/test NumPy files
   - Combines all splits into single dataset
   - Converts to Polars DataFrames for efficiency

2. **Missing Value Handling**:
   - Creates `MissingIndicator_<feature>` columns before imputation
   - Forward fills within each patient stay
   - Zero-fills remaining missing values
   - Preserves information about missingness

3. **Feature Processing**:
   - Dynamic features stored with null values where missing
   - Static features replicated across all timesteps for each patient
   - All features converted to Float64 for PyTorch compatibility
   - **Feature mapping applied during batching** to transform 34→48 features

4. **Data Splitting**:
   - Patient-level splitting (not timestep-level)
   - Ensures all timesteps from a patient are in the same split
   - Reproducible with random seed

### Classification-Specific Details

1. **Balanced Weighting**: Addresses severe class imbalance (98.2% negative)
2. **Cross Entropy Loss**: Standard loss for binary classification
3. **AUPRC Metric**: More informative than AUROC for imbalanced data
4. **Timestep to Patient-Level Aggregation**:
   - Raw P19 data has labels at each timestep (hourly sepsis status)
   - Model predicts patient-level outcome (will patient develop sepsis during stay?)
   - Aggregation: If sepsis occurs at ANY timestep, patient labeled as positive
   - This is standard formulation for sepsis prediction tasks

### Data Preprocessing Script Features

The `prepare_sepsis_data.py` script:
- Automatically extracts feature names from Physionet 2019 standard
- Creates stay_id from sequential indexing
- Handles missing indicators explicitly
- Performs forward-fill and zero-fill imputation
- Splits at patient level with specified ratios

## File Locations Summary

| Component | Location |
|-----------|----------|
| Task config (classification) | `configs/tasks/BinaryClassification.gin` (standard binary classification) |
| Task config (regression) | `configs/tasks/Regression.gin` (for non-BAT models) |
| Model config (AutoregressiveBAT) | `configs/prediction_models/AutoregressiveBAT_Sepsis.gin` (BAT-specific) |
| BAT model code | `icu_benchmarks/models/dl_models/bat.py` |
| Data preprocessing script | `icu_benchmarks/data/prepare_sepsis_data.py` |
| Fine-tuning script | `icu_benchmarks/fine_tuning.py` (standard classification) |
| Test script | `test_autoregressive_bat.py` |
| Source data (NumPy) | `/isdata/winthergrp/gsn245/scratch/Patient_Journey_Classification/P19data/split_1/` |
| Preprocessed data | `icu_benchmarks/data/preprocessed_data/p19/40333_42/` |
| Pretrained models | `pretrained_checkpoints/` |
| Results | `finetuning_results/pretrained_BAT/p19/` |

## Troubleshooting

### Issue: Unbalanced Classes Warning
```
UserWarning: Class balance is highly skewed (1.8% positive)
```
**Solution**: This is expected for sepsis data. The code uses balanced weighting to handle this. Consider:
- Using AUPRC as primary metric (already default)
- Adjusting classification threshold based on precision-recall tradeoff
- Using `--fine_tune_head` for faster experimentation

### Issue: Out of Memory (OOM)
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Reduce batch size: `--bz 16` or `--bz 8`
- Use head-only fine-tuning: `--fine_tune_head`
- Use gradient accumulation (modify script)
- Train on CPU (slower but works)

### Issue: Poor AUPRC Score
```
Validation AUPRC is very low (<0.1)
```
**Solution**:
- Ensure balanced weighting is enabled (default)
- Try different learning rates: `--lr 0.00001` or `--lr 0.001`
- Increase training data size: `--sizes 10000` instead of `--sizes 100`
- Train for more epochs: `--num_epochs 300`
- Check data preprocessing output for class balance

### Issue: Data Loading Errors
```
FileNotFoundError: FEATURES.parquet not found
```
**Solution**:
- Verify preprocessing completed successfully
- Check that `--subset_root` points to correct directory
- Ensure dataset name matches directory name: `p19`
- Re-run preprocessing script if needed

### Issue: Feature Count Mismatch
```
RuntimeError: Expected 74 features but got different number
```
**Solution**: The Physionet 2019 data should have:
- 34 dynamic features
- 34 missing indicators
- 4 static features
- 2 metadata columns (stay_id, time)
- Total: 74 columns in FEATURES

If you see different counts, check the preprocessing script output.

## Performance Tips

1. **Handling Class Imbalance**:
   - Use AUPRC as primary metric (default)
   - Balance weighting is already configured
   - Consider adjusting decision threshold after training

2. **Batch Size**:
   - Start with 32-64 for full fine-tuning
   - Can use 64-128 for head-only fine-tuning
   - Reduce if OOM errors occur

3. **Learning Rate**:
   - 0.0001 works well for full fine-tuning
   - 0.001 for head-only fine-tuning
   - Use learning rate warmup for large batches

4. **Early Stopping**:
   - Patience of 10 epochs prevents overfitting
   - Monitors AUPRC (better for imbalanced data than loss)

5. **Data Efficiency**:
   - Pretrained models work well with <1000 samples
   - Full dataset (28K patients) recommended for best performance
   - Head-only fine-tuning faster for small datasets

## Expected Performance

Based on Physionet 2019 Challenge benchmarks:

| Training Size | Expected AUPRC | Expected AUROC | Notes |
|--------------|----------------|----------------|-------|
| 100 patients | 0.10-0.20 | 0.60-0.70 | Transfer learning helps |
| 1,000 patients | 0.20-0.35 | 0.70-0.80 | Reasonable performance |
| 5,000 patients | 0.30-0.45 | 0.75-0.85 | Good performance |
| Full dataset | 0.40-0.55 | 0.80-0.90 | Best performance |

*Note: These are rough estimates. Actual performance depends on hyperparameters, random seed, and model architecture.*

## Comparison with Other Tasks

### Sepsis (Classification)
```bash
python icu_benchmarks/fine_tuning.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset p19 \
    --sizes 5000 --seeds 42 --bz 64 --lr 0.0001 --num_epochs 200
```
- **Metrics**: AUPRC, AUROC, F1, Balanced Accuracy
- **Loss**: Cross Entropy with balanced weighting
- **Challenge**: Severe class imbalance

### Length of Stay (Regression)
```bash
python icu_benchmarks/fine_tuning_regression.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset mimic_los_regression \
    --sizes 5000 --seeds 42 --bz 64 --lr 0.0001 --num_epochs 200
```
- **Metrics**: MSE, MAE, R²
- **Loss**: Mean Squared Error
- **Challenge**: Predicting continuous values

### Mortality (Classification)
```bash
python icu_benchmarks/fine_tuning.py \
    --model_path pretrained_checkpoints/model.ckpt \
    --dataset mimic_mortality \
    --sizes 5000 --seeds 42 --bz 64 --lr 0.0001 --num_epochs 200
```
- **Metrics**: AUPRC, AUROC, F1, Balanced Accuracy
- **Loss**: Cross Entropy (may or may not need balanced weighting)
- **Challenge**: More balanced than sepsis

## Next Steps

1. **Baseline Comparison**: Compare BAT performance against LGBM/XGBoost baselines
2. **Hyperparameter Tuning**: Grid search over learning rates, batch sizes, architectures
3. **Feature Engineering**: Analyze which features are most predictive
4. **Temporal Analysis**: Study how predictions change over patient stay
5. **Clinical Validation**: Evaluate against clinical sepsis criteria (SOFA, qSOFA)
6. **Early Detection**: Focus on predicting sepsis hours before onset
7. **Multi-Dataset**: Combine with MIMIC-III/IV for larger training set

## Citation

If you use this code or the Physionet 2019 dataset, please cite:

```bibtex
@article{reyna2019early,
  title={Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019},
  author={Reyna, Matthew A and Josef, Chris and Jeter, Russell and others},
  journal={Critical Care Medicine},
  volume={48},
  number={2},
  pages={210--217},
  year={2020}
}

@article{yaib2023,
  title={YAIB: Yet Another ICU Benchmark},
  author={Van de Water, R. and others},
  journal={...},
  year={2023}
}
```

## Related Resources

- **Physionet 2019 Challenge**: https://physionet.org/content/challenge-2019/1.0.0/
- **YAIB Main Repository**: https://github.com/rvandewater/YAIB
- **Pretrained BAT Models**: https://huggingface.co/Katja-Jagd/bat-pretrained-eicu-mimiciv-base
- **YAIB Documentation**: Check CLAUDE.md for general YAIB usage
- **Sepsis-3 Guidelines**: https://jamanetwork.com/journals/jama/fullarticle/2492881

---

*Created: November 8, 2025*
*Last Updated: November 11, 2025*

## Summary of Changes Made

This sepsis prediction task required the following additions to the YAIB codebase:

1. **New Files**:
   - `icu_benchmarks/data/prepare_sepsis_data.py` - NumPy to parquet data preprocessing
   - `icu_benchmarks/models/dl_models/bat.py` - Added `AutoregressiveEncoderCrossParallel` and `AutoregressiveBAT` classes
   - `configs/prediction_models/AutoregressiveBAT_Sepsis.gin` - BAT-specific configuration for autoregressive modeling
   - `test_autoregressive_bat.py` - Test suite for AutoregressiveBAT implementation
   - `SEPSIS_FINE_TUNING_README.md` - This documentation

   **Note**: Uses existing `configs/tasks/BinaryClassification.gin` for regular BAT (no new config needed)

2. **Preprocessed Data**:
   - `icu_benchmarks/data/preprocessed_data/p19/40333_42/` - Ready-to-use parquet files

3. **New Model Architecture**:
   - **AutoregressiveBAT**: Causal attention-based BAT for per-timestep predictions
   - **AutoregressiveEncoderCrossParallel**: Encoder using causal time attention (Decoder) and non-causal sensor attention (Encoder)
   - **Updated EncoderPrediction**: Now supports both regular and autoregressive encoders
   - **Existing prediction heads**: Reuses `RegressionHead` and `TimeseriesClassificationHead`

4. **Reused Components**:
   - Uses standard `icu_benchmarks/fine_tuning.py` for classification
   - Leverages existing BAT model and classification head for patient-level prediction
   - Follows same structure as Mortality and LengthOfStay tasks

5. **Key Differences**:
   - **Regular BAT**: Bidirectional attention, single patient-level prediction, uses standard configs
   - **AutoregressiveBAT**: Causal time attention, per-timestep predictions, requires BAT-specific config

These changes are fully compatible with existing YAIB functionality and follow the established patterns for adding new prediction tasks. The AutoregressiveBAT implementation enables continuous risk monitoring and temporal analysis of sepsis progression.
