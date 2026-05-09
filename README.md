# PSSA-VLA — Persistent Temporal Scene-Spatial Alignment for VLA Models

NanoResearch session `20260509000456c3a8`. Research engine: Claude Code +
NanoResearch pipeline. Compute: AutoDL via `experiment/code/scripts/autodl/`.

## Status (2026-05-09)

- ✅ ideation / planning / setup / coding / writing / review (doc-only)
- 🚧 execution / analysis (deferred — smoke-tier compute via AutoDL pending)

See `output/checkpoint_summary.md` for the resume command list.

## Layout

```
manifest.json               session state — current_stage, checkpoints, artifacts
papers/                     literature seed (annotated bibliography)
plans/                      ideation summary + experiment blueprint
experiment/
  setup.md                  hardware / software / dataset / baseline plan
  env/                      conda + pip pinning
  code/                     scaffold: pssa/ baselines/ data/ configs/ scripts/ tests/
  runs/                     produced on AutoDL — smoke-* dirs are tracked
drafts/                     LaTeX manuscript + bib (Overleaf-synced)
figures/                    framework.tex (TikZ standalone)
output/                     review.md + checkpoint_summary.md
```

## Run smoke on AutoDL (mode C)

See `experiment/code/scripts/autodl/README.md` — 4 lines after SSH.
