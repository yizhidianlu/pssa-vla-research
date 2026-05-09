#!/usr/bin/env bash
# Smoke-tier LIBERO eval on AutoDL.
# Pre-requisites (already done in earlier installs):
#   - pssa-vla conda env with torch / transformers / OpenVLA loadable
#   - LIBERO + robosuite + bddl + mujoco installed
#   - libegl1 / libglvnd0 system packages installed
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pssa-vla
source /root/autodl-tmp/.hf_env

cd /root/autodl-tmp/pssa-vla
git pull --ff-only origin main 2>&1 | tail -3

TS=$(date +%Y%m%d-%H%M%S)
OUT="experiment/runs/libero-smoke-$TS"
mkdir -p "$OUT"

echo "==> running 1 task × 5 rollouts on libero_spatial:0"
python experiment/code/scripts/run_libero_eval.py \
    --suite libero_spatial --task-id 0 \
    --rollouts 5 --max-steps 200 \
    --unnorm-key bridge_orig \
    --out "$OUT/02_libero_metrics.json" 2>&1 | tee "$OUT/02_libero.log"

# Env fingerprint
{
    echo "## GPU"; nvidia-smi -L
    echo; echo "## key versions"
    pip show libero robosuite bddl mujoco transformers timm tokenizers torch 2>/dev/null | grep -E "^(Name|Version)"
} > "$OUT/03_env.txt"

echo "==> outputs:"
ls -lh "$OUT"
echo "==> dir: $OUT"
