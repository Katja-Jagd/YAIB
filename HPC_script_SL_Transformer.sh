
#!/bin/sh 
### General options  
### -- specify queue -- 
#BSUB -q gpua100
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
#BSUB -W 48:00 
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

# Activate venv and load modules 
module load python3/3.10.16
source yaib_venv/bin/activate

# Define train sizes and seeds
train_sizes="100 500 1000 2000 3000 5000 7000 9000 9506"
seeds="42 84 126 168 210"

# Loop over train sizes and seeds
for train_size in $train_sizes; do
    for seed in $seeds; do
        echo "Running with train size $train_size and seed $seed"
        icu-benchmarks train \
            -d /work3/s185395/YAIB-cohorts/data/mortality24/mimic \
            -n mimic \
            -t BinaryClassification \
            -tn Mortality24 \
            -m Transformer_tuned \
            -gc \
            -lc \
            -s 2222 \
            -l ../yaib_logs/ \
            -hp execute_repeated_cv.subset_train_size=$train_size execute_repeated_cv.subset_train_seed=$seed
    done
done
