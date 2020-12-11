#!/bin/bash

#SBATCH -o ./logs/%A/slurm_logs/%a.out
#SBATCH --time=06:00:00
#SBATCH --job-name=silently_flexible
#SBATCH --nodes=3
#SBATCH -c1
#SBATCH --partition fiete
#SBATCH --mail-user=apiccato@mit.edu


CONFIG_NAME=$1

echo ${SLURM_ARRAY_JOB_ID}
echo ${SLURM_ARRAY_TASK_ID}
echo ${CONFIG_NAME}

log_dir=logs/${SLURM_ARRAY_JOB_ID}

echo ${log_dir}

mkdir -p ${log_dir}

python openmind_run.py --config ${CONFIG_NAME} --log_directory ${log_dir}