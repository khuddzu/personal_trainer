#!/bin/bash
#SBATCH --job-name=CO_MBIS_net
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem-per-cpu=3GB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# Change to this job's submit directory
cd $SLURM_SUBMIT_DIR

python trainer.py 
