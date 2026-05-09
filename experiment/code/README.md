# PSSA-VLA — Reference Implementation Scaffold

Persistent Scene-Spatial Alignment for Vision-Language-Action Models.

## Status

This is a **scaffold** — code structure, model definitions, training and eval entry points, configs, and a smoke test. It has not been executed against real GPUs in this session. Run the smoke test (§4) first; it gates the full training runs.

## Layout

```
code/
├── pssa/
│   ├── __init__.py
│   ├── pse_tok.py          # Persistent Scene-Entity Tokenizer (POGS-style)
│   ├── xtc_loss.py         # Cross-Time Consistency Loss
│   ├── cred.py             # Consistency-Residual Error Detector
│   └── model.py            # PSSAVLA wrapper around an OpenVLA-style backbone
├── baselines/
│   ├── openvla_wrapper.py
│   └── pi0_wrapper.py
├── data/
│   └── libero_dataset.py   # minimal LIBERO + CALVIN loader using lerobot
├── configs/
│   ├── default.yaml
│   └── pssa_libero_long.yaml
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── slurm/train.slurm
└── tests/
    └── smoke.py            # gate before EXECUTION (blueprint §9)
```

## Quickstart (when GPU available)

```bash
conda env create -f ../env/conda.yml
conda activate pssa-vla
pip install -r ../env/pip-requirements.txt

export NR_DATA=/path/to/datasets
export WANDB_PROJECT=pssa-vla

# 1. Smoke test
python tests/smoke.py

# 2. Train PSSA on LIBERO-LONG
python scripts/train.py --config-name pssa_libero_long

# 3. Eval all benchmarks
python scripts/eval.py --config-name pssa_libero_long ckpt=runs/latest
```

## Reference

- Persistent Object Gaussian Splat — autolab.berkeley.edu/.../POGS-CRv5.pdf (ICRA 2025)
- 4D Gaussian Splatting — arXiv 2310.08528
- OpenVLA — github.com/openvla/openvla
- LIBERO / CALVIN benchmarks — see `experiment/setup.md` §4
