
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
#BSUB -R "rusage[mem=12GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 12GB
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
#BSUB -o /work3/s185395/YAIB/hpc_output/SLL_output_%J.out
#BSUB -e /work3/s185395/YAIB/hpc_output/SLL_output_%J.err

# Activate venv  and load modules 
module load python3/3.10.16
source yaib_venv/bin/activate

# Execute command

icu-benchmarks train     -d /work3/s185395/YAIB-cohorts/data/los/eicu_miiv_None     -n eicu_miiv     -t Regression     -tn LOS     -m SSL_BAT_no_tune     -gc     -lc
  -s 2222     -l ../yaib_logs/ --wandb-sweep

