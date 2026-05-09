#!/bin/bash
#SBATCH --job-name=pssa-vla
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source ~/.bashrc
conda activate pssa-vla

export NR_DATA=${NR_DATA:-/data/pssa-vla}
export WANDB_PROJECT=${WANDB_PROJECT:-pssa-vla}
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/../.."

CONFIG=${1:-pssa_libero_long}

srun python scripts/train.py --config-name "$CONFIG"
