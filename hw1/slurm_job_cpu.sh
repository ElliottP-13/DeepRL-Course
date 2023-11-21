#!/bin/bash

#SBATCH --time=15:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=32000   # memory (128GB)
#SBATCH -J ds6559_%A   # job name
#SBATCH -o ./logs/hw3_%A.out
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -A sds_ds6559_fl23

CONDA_ENV_PATH="/home/hyy8sc/.conda/envs/ds6559/"

PIP_EX="$CONDA_ENV_PATH/bin/pip"
PYTHON_EX="$CONDA_ENV_PATH/bin/python"

$PIP_EX list

$PYTHON_EX $@

echo "All done"
