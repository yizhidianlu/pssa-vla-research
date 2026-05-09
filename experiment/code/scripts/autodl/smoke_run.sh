#!/usr/bin/env bash
# Smoke run on AutoDL — validates the PSSA-VLA pipeline end-to-end on a
# real GPU using OpenVLA-7B as the backbone. Does NOT fine-tune; runs
# eval-only with frozen weights against a synthetic stepping environment.
# Budget target: 30-60 minutes on A100 80GB.
#
# Run from repo root on AutoDL:
#   bash experiment/code/scripts/autodl/smoke_run.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"
source activate pssa-vla 2>/dev/null || conda activate pssa-vla

export NR_DATA=${NR_DATA:-/root/autodl-tmp/data}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf}
export PYTHONPATH="$REPO_ROOT/experiment/code:${PYTHONPATH:-}"

OUT_DIR="$REPO_ROOT/experiment/runs/smoke-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"
echo "==> output: $OUT_DIR"

# 1. Stub-backbone smoke (no GPU needed, ~10 sec)
echo "==> [1/3] stub smoke"
python experiment/code/tests/smoke.py | tee "$OUT_DIR/01_stub_smoke.log"

# 2. End-to-end smoke with real OpenVLA-7B + PSSA wrapper, synthetic env
echo "==> [2/3] e2e smoke with OpenVLA-7B"
python experiment/code/tests/smoke_e2e.py \
    --rollouts 5 --episode-len 50 \
    --out "$OUT_DIR/02_e2e_metrics.json" 2>&1 | tee "$OUT_DIR/02_e2e.log"

# 3. Capture environment fingerprint
echo "==> [3/3] env fingerprint"
{
    echo "## GPU"
    nvidia-smi -L
    echo
    echo "## CUDA"
    nvcc --version 2>/dev/null || echo "no nvcc"
    echo
    echo "## Python"
    python -V
    echo
    echo "## pip freeze"
    pip freeze
} > "$OUT_DIR/03_env.txt"

echo "==> smoke_run.sh OK — results in $OUT_DIR"
