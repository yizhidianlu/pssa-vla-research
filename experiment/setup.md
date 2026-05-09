# Setup — PSSA-VLA

Session: 20260509000456c3a8
Stage: setup
Date: 2026-05-09
Status: doc-only (compute deferred per user instruction)

## 1. Hardware target

- Minimum: 2× NVIDIA A100 80GB (training) + 1× A100 40GB (eval rollouts).
- Acceptable fallback: 4× RTX 4090 with ZeRO-3 / FSDP CPU offload.
- Storage: 1.5 TB SSD (Open X subset 600 GB, LIBERO 80 GB, CALVIN 220 GB, baselines weights ≈ 200 GB, checkpoints ≈ 400 GB).
- SLURM: provided sample script in `experiment/code/scripts/slurm/`.

## 2. Software stack

| Layer | Choice | Notes |
|-------|--------|-------|
| OS | Ubuntu 22.04 | |
| CUDA | 12.4 | matches OpenVLA ckpt |
| Python | 3.10 | |
| PyTorch | 2.4.x | |
| Lightning / Accelerate | Accelerate 0.34 | FSDP-friendly |
| Vision | torchvision 0.19, timm 1.0.x | |
| 3D | PyTorch3D 0.7.x, gsplat 1.0.x | for Persistent Gaussian Splat |
| SAM | SAM-2 (`segment-anything-2`) | for entity masks at episode init |
| Sim | LIBERO-1.0, CALVIN, VLABench | |
| Logging | wandb | |
| Reproducibility | hydra 1.3 + omegaconf | configs in `experiment/code/configs/` |

`experiment/code/requirements.txt` will pin minimal versions; full lock will be `experiment/code/requirements.lock` after first install.

## 3. Datasets

| Dataset | Size | Use | Source |
|---------|------|-----|--------|
| LIBERO-Spatial / Object / Goal / 100 / LONG | 80 GB | RQ1 train + eval | LIBERO repo |
| LIBERO-Plus, LIBERO-PRO | 12 GB | RQ2 OOD eval | upstream releases |
| CALVIN ABC-D | 220 GB | RQ1 + RQ3 chained tasks | CALVIN release |
| VLABench (long-horizon track) | 60 GB | secondary RQ1 | ICCV 2025 release |
| Open X-Embodiment subset (bridge + rt-1 + furniturebench) | 600 GB | PSE-Tok pretrain warmup | Open X release |

All data downloaded into `$NR_DATA/` (env var). Loader uses `lerobot` format where possible.

## 4. Baseline checkpoints to obtain

| Baseline | Source | License-check |
|----------|--------|---------------|
| OpenVLA-7B | HF `openvla/openvla-7b` | MIT + reproduction OK |
| π0 (open release) | HF `lerobot/pi0` | Apache-2.0 — confirm before use |
| Long-VLA | author release (arXiv 2508.19958) | check repo |
| Seer | ICLR 2025 author release | check repo |
| VLA-in-the-Loop | OpenReview repo | check repo |
| SAM-2 | Meta release | Apache-2.0 |

If any author release is gated or missing, that baseline is reported as "cited number, not reproduced" with a footnote.

## 5. Directory layout produced by setup

```
experiment/
├── setup.md                  ← this file
├── env/
│   ├── conda.yml             ← conda env spec
│   └── pip-requirements.txt  ← pip install -r
├── code/                     ← created in CODING stage
│   ├── pssa/                 ← our method
│   ├── baselines/            ← thin wrappers over external repos
│   ├── data/                 ← LIBERO/CALVIN loaders
│   ├── configs/              ← hydra configs
│   ├── scripts/              ← train.sh, eval.sh, slurm/*.sh
│   └── tests/                ← smoke + unit tests
└── runs/                     ← created in EXECUTION stage
```

## 6. Smoke-test plan (gate before EXECUTION)

In CODING we will produce `experiment/code/tests/smoke.py` that:
1. Loads SAM-2, builds a 2-frame Persistent Gaussian Splat from a fake RGB pair → checks PSE-Tok produces ≥ N entity tokens.
2. Loads OpenVLA-7B head with a 32-token PSE prefix → forward / backward on dummy input.
3. Runs 1 LIBERO-LONG rollout in dry-run mode (env stepping only, action zeros) to confirm sim wiring.
4. Logs latency per step; passes if < 200 ms with batch=1 on a single GPU.

Failure of (1) → fall back to point-track-only PSE-Tok variant (decision-gate from blueprint §9).

## 7. Open dependencies blocked by hardware

- Real GPU not available in this Claude Code session → CODING stage will produce code + tests but skip the live smoke run; reported as `smoke_status: skipped_no_gpu` in the manifest.
- EXECUTION / ANALYSIS / FIGURE_GEN-quantitative-plots stages remain pending until compute is available.
