#!/bin/bash
#SBATCH --job-name=reviewkd_26x_26n
#SBATCH --output=logs/reviewkd_%j.out
#SBATCH --error=logs/reviewkd_%j.err
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

source /work/biraj/miniforge3/bin/activate python_env
cd "/work/biraj/neuromorphic/knowledge distillation"


# Recommended: Prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_offline.py
