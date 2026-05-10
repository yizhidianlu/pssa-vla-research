# LIBERO-Goal Baseline Sweep — 2026-05-11

Session: 20260509000456c3a8
Hardware: AutoDL dual-A800 80GB (port 16967), GPU 1 isolated via `CUDA_VISIBLE_DEVICES=1`
Model: `openvla/openvla-7b-finetuned-libero-goal`
Suite: libero_goal, 10 tasks × 50 rollouts = 500 rollouts

## Result — Suite SR **72.8%** (364/500)

| TID | SR | n_succ | mean_steps_succ | step_ms |
|----:|---:|-------:|----------------:|--------:|
|  0 | 62.0% | 31/50 | 125 | 190.3 |
|  1 | 86.0% | 43/50 |  88 | 189.9 |
|  2 | 90.0% | 45/50 |  95 | 191.6 |
|  3 | **26.0%** | 13/50 | 183 | 189.3 |
|  4 | 80.0% | 40/50 |  92 | 195.0 |
|  5 | 80.0% | 40/50 | 142 | 196.6 |
|  6 | 64.0% | 32/50 | 110 | 195.4 |
|  7 | **96.0%** | 48/50 |  82 | 195.2 |
|  8 | 88.0% | 44/50 |  77 | 191.6 |
|  9 | 56.0% | 28/50 | 154 | 197.0 |
| **mean** | **72.8%** | 364/500 | 107 | 193.2 |

## vs OpenVLA paper

Paper Table 5 reports **79.2%** suite-level SR on libero_goal. Our 72.8%
is **6.4 pp under**, consistent with the cross-suite -4 to -9 pp drift
pattern from running the released checkpoint under `transformers==4.45.2`
instead of `4.40.1`.

## Largest per-task spread

| Easiest | Hardest |
|---------|---------|
| Task 7 — **96.0%** | Task 3 — **26.0%** |

70-pp spread — PSSA-VLA's headroom on this suite is heavily concentrated
on Task 3 ("specific goal-conditioned" task, hard because OpenVLA's
per-frame grounding loses track of the goal across many steps).

## Cross-suite Table 1 update (baseline ceiling)

| Suite | Ours | Paper | Δ |
|-------|------|-------|---|
| LIBERO-Spatial | 80.2% | 84.7% | -4.5 |
| LIBERO-Object | 80.0% | 88.4% | -8.4 |
| LIBERO-Goal   | **72.8%** | 79.2% | -6.4 |
| LIBERO-LONG   | ~48% (partial) | 53.7% | -5.7 |

## Run-time

| Metric | Value |
|--------|-------|
| Wallclock total | ~30 min (paralleling Object on GPU 0) |
| Per-task wallclock | ~3 min |
| Stable step latency | 193 ms/step |
| Peak VRAM | ~15.3 GB |
