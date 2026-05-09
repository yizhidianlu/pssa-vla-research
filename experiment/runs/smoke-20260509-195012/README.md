# Smoke Run — 2026-05-09 19:50 CST

Session: 20260509000456c3a8
Hardware: AutoDL A800 80GB PCIe (instance `autodl-container-1139448516-16e183b7`)
Wallclock: ~3 min after model already cached
Cost: ~¥0.5 for the smoke window itself (~¥10 cumulative for full session including bootstrap + 14 GB OpenVLA download)

## What was tested

| Probe | What it validates | Status |
|-------|-------------------|--------|
| **B (PSSA on GPU)** | PSE-Tok forward + XTC-Loss + CRED + backward on GPU | ✅ loss=1.507, loss_action=1.297, loss_xtc=0.209 |
| **A (OpenVLA-7B inference)** | Model loads + `predict_action()` runs end-to-end via hf-mirror.com | ✅ 15.1s load, 7.54B params, 15.45 GB VRAM |

## Performance (A800 80GB)

| Metric | Value |
|--------|-------|
| Model load (cached) | 15.1 s |
| Peak VRAM during inference | **15.45 GB** (out of 85.1 GB → 5.5× headroom) |
| Stable per-step latency (p50, after warmup) | **188 ms/step** |
| Stable per-step latency (p95, after warmup) | 195 ms/step (very tight tail) |
| Warmup overhead (rollout 0 vs rollout 2) | +118 ms/step (one-time CUDA cache build) |

## What this tells us about the bigger budgets

| Budget | Compute estimate | Pulled-forward verdict |
|--------|------------------|------------------------|
| Smoke (this run) | ~¥10 done | ✅ pipeline tractable |
| **Small** (¥800-1200, 4-5 A100-days) | 1× A800, ~50 hr wallclock | ✅ fits — VRAM has 5× headroom for batch=4 |
| **Mid** (¥1500-2200, 8-10 A100-days) | 1-2× A800, ~80 hr wallclock | ✅ fits |
| **Full** (¥2700-4300, 16-20 A100-days) | 2-4× A800, ~120 hr wallclock | ✅ — eval-rollouts are the bottleneck, parallelize across seeds |

LIBERO-LONG eval @ 188 ms/step:
- 1 episode = 600 steps = **113 sec**
- 1 task @ 50 rollouts = **94 min**
- 10 tasks × 50 rollouts = **15.7 hr** for one config
- 5 baselines + PSSA + 4 ablations = **10 configs × 15.7 hr = 157 hr** = 6.5 GPU-days
  → matches blueprint §8 estimate of 16-20 A100-days when you add training time

## Versions pinned by smoke

OpenVLA needs `timm` in `[0.9.10, 1.0.0)` and works with `transformers==4.45.2 / tokenizers==0.20.3` despite warning about its preferred 4.40.1/0.19.1.

```
transformers   4.45.2     (downgraded from 5.8.0 — v5 incompatible with prismatic code)
timm           0.9.16     (downgraded from 1.0.27 — OpenVLA hard-requires v0.9.x)
tokenizers     0.20.3
torch          2.4.1+cu124
```

These pins are enforced in `experiment/env/pip-requirements.txt` going forward.

## Network setup

China mainland → `huggingface.co` was unreachable (`Cannot assign requested address`).
Fix: `export HF_ENDPOINT=https://hf-mirror.com` — added to `/root/autodl-tmp/.hf_env` for persistence.

## Next decisions

The smoke confirms it's safe to scale to **mid budget** (¥1500-2200): no compute-side surprises, latency is reasonable, VRAM is comfortable.

Recommended next pause-point: pick a single LIBERO-LONG task (e.g. `kitchen_pickup_and_place`) and try a real rollout (not synthetic RGB) to confirm the eval harness works against the actual sim. That's a separate small budget tier (¥50-100) before committing to the full sweep.
