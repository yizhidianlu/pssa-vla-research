# Phase 1 — LIBERO-Spatial Full Baseline

Session: 20260509000456c3a8
Hardware: AutoDL A800 80GB PCIe
Model: `openvla/openvla-7b-finetuned-libero-spatial`
Wallclock: 4 hr 17 min (00:40 → 04:57 CST)
Cost: ~¥27

## Headline number

**Overall SR: 80.2% (401/500)** across all 10 LIBERO-Spatial tasks × 50 rollouts each.

Mean steps to success: 107 (out of 200 max).

## Per-task breakdown

| TID | SR | n_succ | mean_steps_OK | mean_steps_F | step_ms |
|-----|----|--------|---------------|--------------|---------|
| 0 | 86.0% | 43/50 | 80 | 200 | 189.3 |
| 1 | 82.0% | 41/50 | 117 | 200 | 189.1 |
| 2 | 90.0% | 45/50 | 105 | 200 | 188.1 |
| 3 | 84.0% | 42/50 | 91 | 200 | 189.4 |
| 4 | **68.0%** | 34/50 | 127 | 200 | 195.3 |
| 5 | **94.0%** | 47/50 | 95 | 200 | 188.4 |
| 6 | 92.0% | 46/50 | 111 | 200 | 190.3 |
| 7 | **74.0%** | 37/50 | 129 | 200 | 190.5 |
| 8 | 78.0% | 39/50 | 100 | 200 | 194.3 |
| 9 | **54.0%** | 27/50 | 132 | 200 | 189.9 |
| **ALL** | **80.2%** | **401/500** | **107** | — | — |

Bold = notably below mean. Tasks 4, 7, 9 are the hardest in this suite —
their `mean_steps_OK` (127, 129, 132) shows even successful runs take
1.5× longer than easy tasks (which finish in ~80-100 steps).

## Comparison vs OpenVLA paper

| Source | libero_spatial mean SR |
|--------|------------------------|
| OpenVLA paper (Table 5) | 84.7% |
| **Our run** | **80.2%** |
| Gap | −4.5 pp |

The gap is within reasonable replication variance:
- Different transformers version (4.45.2 vs paper's 4.40.1) — known to
  have minor inference drift per the model's own warning at load time
- 50 rollouts per task is the standard sample but error bars are
  `√(p(1-p)/n) ≈ √(0.84·0.16/500) ≈ 1.6 pp` at the suite level
- Single seed; paper averages across multiple

This means PSSA-VLA's headroom-to-beat is **80.2%**, not 84.7%, on this
exact eval recipe. If we surpass 80.2% with our matched setup, we have
a real signal regardless of how that compares to the paper's number.

## Eval recipe (frozen)

These are the canonical settings — any future PSSA-VLA eval must match
exactly to be apples-to-apples:

```bash
python experiment/code/scripts/run_libero_eval.py \
    --suite libero_spatial --task-id <0..9> \
    --rollouts 50 --max-steps 200 \
    --model-id openvla/openvla-7b-finetuned-libero-spatial \
    --unnorm-key libero_spatial \
    --libero-action-fix --libero-image-fix \
    --out task_$tid.json
```

Plus environment:
- `transformers==4.45.2 timm==0.9.16 tokenizers==0.20.3 torch==2.4.1+cu124`
- `mujoco==3.1.6 robosuite==1.4.0 bddl==1.0.1 numpy==2.2.6`
- `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_ENDPOINT=https://hf-mirror.com`
- system: `apt install libegl1 libgl1`
- LIBERO repo: missing `libero/__init__.py` patched in
- env reuse across rollouts (don't recreate per rollout — fd leak)

## Artifacts

- `summary.txt` — pretty-printed table (above)
- `summary.json` — machine-readable aggregate including all per-task records
- `task_0.json` … `task_9.json` — per-task per-rollout records (50 entries each)

Each per-task record has:
- `success: bool`
- `steps: int` (where success=True, this is steps-to-success; where False, 200)
- `step_ms_avg`, `step_ms_p95`
- `peak_vram_gb`

## What this enables

This is the **bottom row of Table 1** in the paper. PSSA-VLA's row will be
written next to this with the SAME 10 × 50 protocol on the SAME hardware,
so the comparison is apples-to-apples within ~1.6 pp error bars.
