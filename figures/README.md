# Figures

All sources are TikZ-only standalone `.tex` files. Compile any of them via
`pdflatex <name>.tex`. No external data files required (numbers are
inlined from the experimental results in `manifest.json` / paper draft).

| File | What it shows | Referenced in `main.tex` |
|------|----------------|---------------------------|
| `framework.tex` | PSSA-VLA v2c architecture as actually implemented (CNN PSE encoder + frozen OpenVLA backbone + LoRA on q/v + autoregressive action-token decode) | §3 Method, Figure 1 |
| `sequence_layout.tex` | Three sequence layouts (OpenVLA canonical / variant\_A after\_image / variant\_B before\_action) with position-slot ruler and Gate-1 SRs | §5.2, Figure 2 |
| `bar_pertask_sr.tex` | Per-task SR bar chart: OpenVLA-FT baseline (n=50 rollouts/task) vs untrained PSE prefix (n=10/task) vs best trained run (n=10/task) | §5.2 / §5.3, Figure 3 |
| `loss_trajectory.tex` | Action-token CE loss curves for 4 training configurations + random baseline; annotates that even all-bugs-fixed v2c plateaus at CE~2-3 rather than reaching ~0 | §5.3, Figure 4 |

## Compile commands

```bash
cd figures/
pdflatex framework.tex
pdflatex sequence_layout.tex
pdflatex bar_pertask_sr.tex   # needs pgfplots
pdflatex loss_trajectory.tex  # needs pgfplots
```

## Data provenance

* `framework.tex` — fixed architecture; no data dependency.
* `sequence_layout.tex` — Gate-1 R3 results in `experiment/runs/gate1_v2c_all-20260512-000927/*/task_0.json` (control 30%, variant_A 80%, variant_B 40%).
* `bar_pertask_sr.tex`:
    * OpenVLA-FT row: Phase 1 manifest entry `libero_spatial.per_task_sr` (500 rollouts).
    * Untrained PSE row: `untrained_variantA_full-20260512-151912/task_*.json` (T0-T8 confirmed; T9 captured on persistent disk pending machine restart).
    * Trained row: `autonomy_v2c-20260512-071037/gate3_eval/summary.json` (22%, 100 rollouts).
* `loss_trajectory.tex` — `experiment/runs/*/gate3_train/train.log` step lines.

## Deprecated / not used in this draft

The original blueprint listed `qualitative_rollout.tex`,
`bar_libero_long.tex`, `cred_residual_trace.tex`, and `ablation_grid.tex`
(the RQ1/RQ3/RQ4 figures). The current paper draft is an empirical
pipeline study on LIBERO-Spatial only; those four figures are deferred
to the follow-up paper that recovers the OpenVLA-FT baseline ceiling
after the 3 follow-ups listed in §6 Limitations.
