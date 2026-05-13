#!/bin/bash
# A1 — train PSSA v2c with SAM-2 + XTC on LIBERO-Spatial, then eval.
# Target (action_guide §6 A1): SR >= 73% (OpenVLA-native matched ceiling at n=10/task).
# Decision gate: SR < 60% -> diagnose PSE vs §3 description gap.
#
# Prereq: SAM-2 mask precompute (A0.5) completed; cache at /root/autodl-tmp/sam2_cache.
#
# Usage:
#   SEED=1 bash scripts/a1_train_eval.sh
#
# Wall-clock estimate: ~3 hr training (dual A800) + ~50 min eval (10 tasks × 10 rollouts)
# Outputs:
#   experiment/runs/a1_sam2_spatial_seed{SEED}-{ts}/
#     ├── train.log
#     ├── config.yaml
#     ├── checkpoints/step_003000/
#     │   ├── pssa_modules.pt
#     │   └── lora/
#     └── eval/
#         ├── task_0.json ... task_9.json
#         ├── summary.json     ← median SR + per-task SRs
#         └── eval.log
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/root/autodl-tmp/pssa-vla}
cd "$REPO_ROOT/experiment/code"

PY=/root/miniconda3/envs/pssa-vla/bin/python
ACCELERATE=/root/miniconda3/envs/pssa-vla/bin/accelerate
export PATH=/root/miniconda3/envs/pssa-vla/bin:$PATH
TS=$(date +%Y%m%d-%H%M%S)
SEED=${SEED:-1}
OUT_DIR="$REPO_ROOT/experiment/runs/a1_sam2_spatial_seed${SEED}-${TS}"
mkdir -p "$OUT_DIR/eval"

export NR_DATA=${NR_DATA:-/root/autodl-tmp/datasets}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf}

# ======================== Training ========================
echo "=== A1 training (seed=$SEED) ==="
$ACCELERATE launch \
    --num_processes 2 \
    --num_machines 1 \
    --multi_gpu \
    --mixed_precision bf16 \
    -m pssa.train \
    out_dir="$OUT_DIR" \
    seed="$SEED" \
    train.lr=2.0e-5 \
    train.max_steps=3000 \
    train.batch_size=2 \
    train.log_every=50 \
    train.save_every=1000 \
    model.pse_position=after_image \
    model.pse.n_entities=8 \
    +model.pse.zero_init_output=true \
    model.lora.enable=true \
    model.lora.r=32 \
    model.lora.alpha=64 \
    model.lambda_xtc=0.1 \
    data.use_sam2_masks=true \
    data.sam2_cache_dir=/root/autodl-tmp/sam2_cache \
    2>&1 | tee "$OUT_DIR/train.log"

CKPT_DIR=$(ls -td "$OUT_DIR/checkpoints"/step_* | head -1)
if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: no checkpoint found under $OUT_DIR/checkpoints"
    exit 1
fi
echo "=== A1 training done, using ckpt: $CKPT_DIR ==="

# ======================== Eval ========================
# 10 tasks × 10 rollouts on libero_spatial, single GPU (GPU 0).
# Why single GPU: run_pssa_eval is per-task, can't easily parallelize within
# one ckpt across GPUs (different envs use different EGL contexts). Run
# sequentially; ~5 min per task × 10 tasks = ~50 min.
echo "=== A1 eval start ==="
EVAL_DIR="$OUT_DIR/eval"
for TASK in 0 1 2 3 4 5 6 7 8 9; do
    echo "--- eval task $TASK ---"
    CUDA_VISIBLE_DEVICES=0 $PY -m scripts.run_pssa_eval \
        --ckpt "$CKPT_DIR" \
        --suite libero_spatial \
        --task-id "$TASK" \
        --rollouts 10 \
        --resolution 128 \
        --n-init-frames 4 \
        --n-entities 8 \
        --pse-position after_image \
        --out "$EVAL_DIR/task_${TASK}.json" \
        2>&1 | tee -a "$EVAL_DIR/eval.log"
done

# ======================== Aggregate ========================
$PY - <<EOF
import json
from pathlib import Path
eval_dir = Path("$EVAL_DIR")
results = {}
total_succ = 0
total_n = 0
for t in range(10):
    p = eval_dir / f"task_{t}.json"
    if not p.is_file():
        results[str(t)] = None
        continue
    d = json.loads(p.read_text())
    sr = d.get("success_rate", d.get("sr", 0))
    n = d.get("n_rollouts", d.get("n", 0))
    n_succ = d.get("n_success", int(sr * n))
    results[str(t)] = {"sr": sr, "n_success": n_succ, "n_rollouts": n}
    total_succ += n_succ
    total_n += n
mean_sr = total_succ / total_n if total_n else 0
summary = {
    "seed": $SEED,
    "ckpt": "$CKPT_DIR",
    "suite": "libero_spatial",
    "per_task": results,
    "total_success": total_succ,
    "total_rollouts": total_n,
    "mean_sr": mean_sr,
    "a1_target": 0.73,
    "a1_pass": mean_sr >= 0.73,
}
out_path = eval_dir / "summary.json"
out_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
EOF

echo "=== A1 done. Summary: $EVAL_DIR/summary.json ==="
