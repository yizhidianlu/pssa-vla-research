# LIBERO Smoke Run — 2026-05-09 22:48 CST

Session: 20260509000456c3a8
Hardware: AutoDL A800 80GB PCIe
Suite: libero_spatial, task 0
Task: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`
Model: `openvla/openvla-7b` (generalist, **no** LIBERO fine-tune)

## Result

| Metric | Value |
|--------|-------|
| Rollouts | 5 |
| Success rate | **0 / 5 (0%)** |
| Wallclock | ~3.2 min |
| Stable per-step latency (p50) | **190 ms** (matches synthetic e2e exactly) |
| Stable per-step latency (p95) | **199 ms** |
| Peak VRAM | **15.45 GB** (out of 85.1 GB) |

**SR=0% is 100% expected.** OpenVLA-7B generalist was trained on Bridge V2,
not LIBERO. With `unnorm_key="bridge_orig"`, predicted action magnitudes
don't match LIBERO's action space, so the robot just twitches without
completing tasks. The smoke confirms the **pipeline** works end-to-end —
sim env stepping, MuJoCo EGL rendering, OpenVLA inference, action
application all succeed without error or memory leak across 200-step
episodes.

## Pipeline-side validations passed

| Layer | Validation |
|-------|------------|
| Network | hf-mirror.com replaces unreachable huggingface.co |
| GitHub access | gh-proxy.com mirror used for clone + pull |
| OpenVLA load | 15.5s from cache, 7.54B params, 15.45 GB VRAM |
| MuJoCo EGL | libglvnd/libegl1 (apt) + NVIDIA libEGL_nvidia.so.0, render OK |
| LIBERO env init | OffScreenRenderEnv loads BDDL, returns obs after dummy steps |
| LIBERO env step | 200 steps × 5 rollouts = 1000 sim steps, no crash |
| OpenVLA predict_action | 1000 inference calls, deterministic, no NaN |
| env.close() | Clean teardown, no MjRenderContext.__del__ errors |

## Sim-side fixes required (recorded for reproducibility)

1. **Network**: `HF_ENDPOINT=https://hf-mirror.com` to bypass GFW
2. **GitHub**: `git config --global url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/"` for clones
3. **transformers**: pin to 4.45.2 (5.x breaks OpenVLA `infer_schema`)
4. **timm**: pin to 0.9.16 (OpenVLA hard-requires timm < 1.0)
5. **LIBERO packaging**: created missing `/root/autodl-tmp/LIBERO/libero/__init__.py` (find_packages requires it; clone-only setup didn't have it)
6. **EGL on AutoDL**: `apt install libegl1 libgl1` (libglvnd dispatcher to NVIDIA's libEGL_nvidia.so.0); `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`
7. **Misc deps**: `matplotlib` (LIBERO's env_wrapper.py imports it for vis)

## Cost (cumulative AutoDL session)

| Phase | Time | Cost |
|-------|------|------|
| Initial bootstrap + OpenVLA download | 30 min | ~¥4 |
| Synthetic e2e smoke + push | 5 min | ~¥0.7 |
| LIBERO install + EGL fixing | 60 min (mostly debug) | ~¥8 |
| LIBERO smoke run + finalize | 10 min | ~¥1.3 |
| **Total** | **~105 min** | **~¥14** |

Still well below the ¥20-50 budget set for B-tier.

## Path forward

**For real LIBERO numbers** (next budget tier):

1. Download `openvla/openvla-7b-finetuned-libero-spatial` from HF mirror (~14 GB, +¥3-4)
2. Set `unnorm_key="libero_spatial"` (matches that ckpt's action stats)
3. Re-run with the same `run_libero_eval.py` — expected SR ~70-90% on this task

**For PSSA-VLA proper** (mid budget ¥1500-2200):

1. Use `experiment/code/scripts/run_libero_eval.py` as the eval harness skeleton
2. Add PSSA fine-tune of action head (OpenVLA frozen) on LIBERO demos
3. Sweep across LIBERO-LONG + LIBERO-Plus
4. Ablation: PSSA vs −PSE-Tok vs −XTC-Loss vs −CRED

The current scaffold and runbook are sufficient — no architectural changes needed before scaling up.
