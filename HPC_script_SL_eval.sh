
#!/bin/sh 
### General options  
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J YAIB
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 4GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 02:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s185395@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 

### -- specify the output and error file inside the run folder -- 
#BSUB -o /work3/s185395/YAIB/hpc_output/SL_output_%J.out
#BSUB -e /work3/s185395/YAIB/hpc_output/SL_output_%J.err

# Activate venv  and load modules 
module load python3/3.10.16
source yaib_venv/bin/activate

# ------------------------
# CONFIGURABLE PARAMETERS
# ------------------------
SOURCE_NAME="miiv"                        # Training source dataset
EVAL_DATASETS=("mimic" "eicu")             # Evaluation datasets
EXPERIMENT_FOLDER="2025-06-04T14-43-52"   # Folder name in logs

# ------------------------
# EVALUATE ALL FOLDS
# ------------------------

for FOLD in 0 1 2 3 4; do
  echo "Evaluating fold $FOLD"

  for DATASET_NAME in "${EVAL_DATASETS[@]}"; do
    echo "  Using evaluation dataset: $DATASET_NAME"

    icu-benchmarks \
      --eval \
      -d "/work3/s185395/YAIB-cohorts/data/mortality24/${DATASET_NAME}" \
      -n "${SOURCE_NAME}" \
      -t BinaryClassification \
      -tn Mortality24 \
      -m BAT_eval \
      --generate_cache \
      --load_cache \
      -s 2222 \
      -l ../yaib_logs \
      -sn "${SOURCE_NAME}" \
      --source-dir "/work3/s185395/yaib_logs/${SOURCE_NAME}/Mortality24/BAT/${EXPERIMENT_FOLDER}/repetition_0/fold_${FOLD}"
  done
done
