#!/bin/bash
#

array_launch=$(\
    sbatch --array=0-1 openmind_task.sh "$1"
  )

job_id=${array_launch##*' '}

echo "Launched job ${job_id}"

mkdir -p logs/${job_id}
mkdir -p logs/${job_id}/slurm_logs