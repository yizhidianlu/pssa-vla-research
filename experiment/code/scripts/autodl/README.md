# AutoDL × PSSA-VLA — Smoke Run Runbook

This runbook expects mode **C (git two-way sync)** with a **smoke** budget
of 1–2 GPU-hours on a single A100 80GB AutoDL instance.

## Cost ceiling

| Item | Estimate |
|---|---|
| AutoDL A100 80GB on-demand | ~¥7–9 / hr |
| Smoke wallclock (boot → push) | ~30–50 min |
| OpenVLA-7B HF download | 14 GB once, free |
| **Total expected spend** | **¥4–7** |

Set a **¥30 hard cap** in AutoDL → 控制台 → 余额预警 before starting.

## One-time setup (your machine)

1. Get a HuggingFace token: <https://huggingface.co/settings/tokens> → New
   token → `Read` scope. Save it as `HF_TOKEN`.
2. Create a private GitHub repo (e.g. `pssa-vla-research`).
3. Tell Claude Code the repo URL — Claude will push the workspace and
   give you the AutoDL-side commands.

## On AutoDL — the entire smoke run is 4 lines

After SSH'ing into the rental:

```bash
# 0. Clone into the persistent data disk
cd /root/autodl-tmp
git clone https://github.com/<you>/pssa-vla-research.git pssa-vla
cd pssa-vla

# 1. Bootstrap (env + OpenVLA download, ~10 min on A100 box)
export HF_TOKEN=hf_xxx_yourkey
bash experiment/code/scripts/autodl/bootstrap.sh

# 2. Smoke (~5 min on A100)
bash experiment/code/scripts/autodl/smoke_run.sh

# 3. Push results back as a new branch
bash experiment/code/scripts/autodl/package_results.sh
# prints:  results-smoke-YYYYMMDD-HHMMSS  ← tell Claude this branch name
```

## Expected output

The smoke run produces one directory under `experiment/runs/smoke-<ts>/`
containing:

| File | What it tells us |
|------|------------------|
| `01_stub_smoke.log` | tests/smoke.py (PSE-Tok forward / training_step / CRED) passed |
| `02_e2e_metrics.json` | OpenVLA-7B + PSSA: rollouts × {steps, step_ms_avg, cred_triggers, peak_vram_gb} |
| `02_e2e.log` | console output of the e2e run |
| `03_env.txt` | GPU model + CUDA + pip freeze fingerprint |

The `package_results.sh` step then commits this dir to a `results-smoke-<ts>`
branch and pushes to origin. **Tell Claude that branch name** — Claude
will pull, parse the metrics, and decide whether to recommend scaling up
(中预算) or fix something first.

## Costs to watch

- **Don't leave the instance running idle.** Stop it (`关机`, not `释放`) the
  moment results push. Re-rent later costs ~¥0.10/hr in stopped state.
- **Don't 释放 (release) the instance** unless the data disk is empty —
  that wipes `/root/autodl-tmp` and you lose the HF cache (re-downloading
  costs another ~10 min next time).

## If things break

- `HF_TOKEN unset` → bootstrap will print a warning and fail at OpenVLA
  download. Re-run bootstrap with the token exported.
- `CUDA OOM` on smoke — should not happen at smoke scale; if it does, the
  AutoDL instance is probably 24 GB not 80 GB. `nvidia-smi -L` to verify.
- `git push` rejects — likely you didn't add the AutoDL host's SSH key to
  the GitHub repo (Settings → Deploy keys), or the repo URL uses HTTPS
  without a personal access token. Set `git remote set-url origin` to a
  PAT-embedded HTTPS URL.

## Next step after smoke succeeds

Tell Claude `pull branch <name>`. Claude will:
1. `git fetch && git checkout <branch>` in the source workspace.
2. Parse `02_e2e_metrics.json`.
3. Update `manifest.json` to record smoke success + measurements.
4. Recommend whether to scale up to mid budget or fix something.
