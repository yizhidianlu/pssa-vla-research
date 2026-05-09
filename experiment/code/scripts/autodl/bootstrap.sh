#!/usr/bin/env bash
# AutoDL bootstrap — env install + model download.
# Idempotent: safe to re-run after a partial failure.
#
# Run from the repo root on AutoDL:
#   cd /root/autodl-tmp/pssa-vla
#   bash experiment/code/scripts/autodl/bootstrap.sh
#
# Required env vars (export before running):
#   HF_TOKEN=hf_...   # huggingface.co/settings/tokens  (Read scope)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
DATA_ROOT=${NR_DATA:-/root/autodl-tmp/data}
HF_HOME_DIR=${HF_HOME:-/root/autodl-tmp/hf}
mkdir -p "$DATA_ROOT" "$HF_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_CACHE="$HF_HOME_DIR"
export NR_DATA="$DATA_ROOT"

echo "==> Repo root:   $REPO_ROOT"
echo "==> Data root:   $DATA_ROOT"
echo "==> HF cache:    $HF_HOME_DIR"

# 1. AutoDL preset usually has conda + python3.10. Check.
which python || { echo "FATAL: no python in PATH"; exit 1; }
python -V

# 2. Use a dedicated env to avoid clobbering the AutoDL preset
if ! conda env list 2>/dev/null | grep -q '^pssa-vla '; then
    echo "==> Creating conda env pssa-vla"
    conda create -y -n pssa-vla python=3.10
fi
source activate pssa-vla 2>/dev/null || conda activate pssa-vla

# 3. Install deps (smoke-tier — leaves out LIBERO/CALVIN/sim packages)
pip install --upgrade pip
pip install --no-cache-dir \
    "torch==2.4.*" "torchvision==0.19.*" --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir \
    "transformers>=4.45" "accelerate>=0.34" "timm>=1.0" \
    "einops" "sentencepiece" "safetensors" \
    "hydra-core>=1.3" "omegaconf" "draccus" \
    "wandb" "rich" "tqdm" "opencv-python-headless"

# 4. Login to HF if HF_TOKEN is set
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "==> Logging in to HuggingFace"
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    echo "WARN: HF_TOKEN not set — OpenVLA download will fail."
    echo "      Run: export HF_TOKEN=hf_xxx  before bootstrap."
fi

# 5. Pre-download OpenVLA-7B (smoke uses it eval-only, no fine-tune)
python - <<'PY'
import os
from transformers import AutoModelForVision2Seq, AutoProcessor
mid = "openvla/openvla-7b"
print(f"==> Downloading {mid} into {os.environ.get('HF_HOME')}")
AutoProcessor.from_pretrained(mid, trust_remote_code=True)
AutoModelForVision2Seq.from_pretrained(mid, trust_remote_code=True, torch_dtype="auto")
print("==> OpenVLA-7B ready")
PY

# 6. GPU sanity
python - <<'PY'
import torch
print(f"CUDA: {torch.cuda.is_available()}  count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"VRAM:  {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
PY

echo "==> bootstrap.sh OK"
