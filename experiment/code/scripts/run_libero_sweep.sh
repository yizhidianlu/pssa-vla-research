#!/usr/bin/env bash
# Phase-1 sweep: OpenVLA-7B-finetuned-libero-spatial across all 10 tasks
# of libero_spatial, 50 rollouts each. Saves per-task JSON.
#
# Usage:
#   bash run_libero_sweep.sh <out_dir> [n_rollouts]
#
# Defaults: n_rollouts=50.
set -euo pipefail
# Robosuite leaks fds; per-task even with env reuse we want headroom for
# 10 task creations + processor downloads + model checkpoint fds.
ulimit -n 65536
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pssa-vla
source /root/autodl-tmp/.hf_env

OUT="${1:?usage: run_libero_sweep.sh <out_dir> [n_rollouts]}"
N_ROLLOUTS="${2:-50}"
# All three default to libero_spatial; override via env vars for LIBERO-LONG / Goal / Object
SUITE="${LIBERO_SUITE:-libero_spatial}"
MODEL_ID="${LIBERO_MODEL_ID:-openvla/openvla-7b-finetuned-libero-spatial}"
UNNORM_KEY="${LIBERO_UNNORM_KEY:-libero_spatial}"
mkdir -p "$OUT"
echo "==> sweep out: $OUT"
echo "==> suite:    $SUITE"
echo "==> model:    $MODEL_ID"
echo "==> unnorm:   $UNNORM_KEY"
echo "==> rollouts/task: $N_ROLLOUTS"

START_TS=$(date +%s)
for TID in 0 1 2 3 4 5 6 7 8 9; do
    TASK_OUT="$OUT/task_${TID}.json"
    if [ -f "$TASK_OUT" ]; then
        echo "==> task $TID already done — skip"
        continue
    fi
    echo "==> task $TID starting at $(date +%H:%M:%S)"
    python experiment/code/scripts/run_libero_eval.py \
        --suite "$SUITE" --task-id "$TID" \
        --rollouts "$N_ROLLOUTS" --max-steps 200 \
        --model-id "$MODEL_ID" \
        --unnorm-key "$UNNORM_KEY" \
        --libero-action-fix --libero-image-fix \
        --out "$TASK_OUT"
    DONE_TS=$(date +%s)
    ELAPSED=$((DONE_TS - START_TS))
    echo "==> task $TID done; total elapsed ${ELAPSED}s"
done

echo "==> all 10 tasks complete in $(( $(date +%s) - START_TS ))s"
echo "==> per-task JSONs:"
ls -lh "$OUT"
