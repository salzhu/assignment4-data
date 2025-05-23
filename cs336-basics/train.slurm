#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=a4-batch
#SBATCH --qos=a4-batch-qos
#SBATCH --gpus=2              # 2 GPUs for torchrun
#SBATCH --cpus-per-task=4         # 4 CPUs for this job (max allowed)
#SBATCH --mem=32G                # up to 100GB
#SBATCH --time=12:00:00           # max run time
#SBATCH --output=x_%j.out   # logs/cs336-train_<jobid>.out

# Run your training command
uv run torchrun --standalone --nproc_per_node=2 scripts/train.py --config-name=experiment/your_data