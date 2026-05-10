#!/usr/bin/env bash
# PSSA Phase-2 training launcher.
# Single GPU:
#   bash run_pssa_training.sh
# Dual GPU FSDP:
#   USE_FSDP=1 bash run_pssa_training.sh
set -euo pipefail
ulimit -n 65536

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pssa-vla
source /root/autodl-tmp/.hf_env

cd /root/autodl-tmp/pssa-vla

USE_FSDP="${USE_FSDP:-0}"
EXTRA_ARGS=("$@")

if [ "$USE_FSDP" = "1" ]; then
    accelerate launch \
        --num_processes 2 --multi_gpu \
        --use_fsdp \
        --fsdp_sharding_strategy FULL_SHARD \
        --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
        --mixed_precision bf16 \
        -m pssa.train "${EXTRA_ARGS[@]}"
else
    python -m pssa.train "${EXTRA_ARGS[@]}"
fi
