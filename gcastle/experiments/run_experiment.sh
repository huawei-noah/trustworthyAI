#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00

eval "$(conda shell.bash hook)"
conda activate cdrl

python run_experimennt.py "$@"