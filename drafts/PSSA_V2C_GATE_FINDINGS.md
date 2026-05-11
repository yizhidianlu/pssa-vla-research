# PSSA v2c — Multi-Agent Diagnostic Findings (2026-05-12 03:00-04:00)

## Headline numbers

| Run | Setup | LIBERO-Spatial SR |
|---|---|---|
| OpenVLA-FT (Phase 1 baseline) | — | **80.2%** (500 rollouts) |
| v1.1 (vision-bypass) | — | 0/100 |
| v2 (MLP head, vision integrated) | — | 0/100 |
| v2c-r1 (manual concat [text\|image\|PSE]) | — | 0/100 |
| **v2c-r2 Gate-1 control (no-PSE, untrained)** | `[BOS\|image\|text]` | 3/10 = 30% |
| **v2c-r2 Gate-1 variant_A (PSE after_image, untrained)** | `[BOS\|image\|PSE\|text]` | **8/10 = 80%** ✨ |
| **v2c-r2 Gate-1 variant_B (PSE before_action, untrained)** | `[BOS\|image\|text\|PSE]` | 4/10 = 40% |
| **v2c-r2 Gate-3 (variant_A trained 3000 steps)** | — | **0/10 on step_001000, _002000, _003000** ✗ |

## Bugs identified + fixed in this round

1. **Sequence layout backwards** (`model_v2.py`): v1.1/v2/v2c-r1 built `[text|image|PSE|action]` but OpenVLA's finetuned layout is `[BOS|image|text]`. Position embeddings were anchored to the wrong positions; cross-attention completely mis-grounded. **Fix:** swap to OpenVLA's canonical order; add `pse_position` flag for A/B testing.

2. **Action vocab boundary off-by-pad** (`model_v2.py`): OpenVLA uses `action_vocab_boundary = config.text_config.vocab_size - config.pad_to_multiple_of = 32064 - 64 = 32000`, not the inner Llama `vocab_size`. The 64-token offset shifted all action token IDs out of OpenVLA's trained slot. **Fix:** new constants `_lm_vocab_size=32064` for logits dim, `_action_vocab_boundary=32000` for token IDs.

3. **Bin scheme off-by-one** (`model_v2.py`): we used 256 bin centers; OpenVLA's `ActionTokenizer` produces 255 (from 256 bin edges). Round-trip discretization was misaligned by half-a-bin. **Fix:** match OpenVLA exactly — `bins = np.linspace(-1, 1, 256)`, `bin_centers = mean of adjacent edges` (shape 255).

4. **LR schedule doubled under DDP** (`train.py`): `accelerator.prepare(sched)` advances per-rank, so 2-GPU cosine schedule consumed its full 3000-step budget by step 1500. **Fix:** `T_max = max_steps * accelerator.num_processes`.

5. **Eval missing gripper convention fix** (`run_pssa_eval.py`): Phase 1's `run_libero_eval.py` applied `action[-1] = -sign(2*x - 1)` (OpenVLA [0,1] → LIBERO {-1,+1}). v2c eval was missing it. **Fix:** apply same transform, default `--libero-action-fix=True`.

## ROOT CAUSE of Gate-3 0/100 collapse (new this round) ✨

**LIBERO HDF5 demos store gripper in `{-1, +1}` (1 = open).** **OpenVLA was finetuned with gripper in `[0, 1]` (1 = close).** When we CE-train against raw LIBERO demos using OpenVLA's `q01`/`q99` (which describe the OpenVLA convention), the gripper signal is **inverted**:

- Demo open (LIBERO +1) → normalize → token close-most (31744)
- Demo close (LIBERO -1) → normalize → token open-most (31999)

Training pulls LoRA to predict the **wrong** action-token slot for gripper, then Phase-1's eval-time gripper fix applies on top of an already-inverted prediction → garbage. This explains:

- Why **untrained** variant_A reaches 80% (PSE prefix is benign noise; OpenVLA's native gripper output is correct)
- Why **trained** v2c collapses to 0% at the very first checkpoint (1000 steps of inverted-gripper supervision is enough to break grasping)
- Why all 3 checkpoints (1k/2k/3k) behave identically (gripper inversion is the dominant failure mode regardless of training duration)

**Fix applied** in `experiment/code/pssa/dataset.py::LIBEROEpisodeDataset._load_demo`:
```python
# LIBERO {-1, +1} -> OpenVLA [0, 1]
actions[:, 6] = (1.0 - actions[:, 6]) / 2.0
```

## Status

- Machine offline (autonomous shutdown fired after diagnostic scan completed)
- Persistent disk retains all Gate-1/Gate-3 artifacts on `/root/autodl-tmp/pssa-vla/experiment/runs/`
- Code fix committed locally + pushed
- **Next session**: re-rent new A800 dual-GPU, pull latest code (includes gripper fix), launch fresh `autonomy_v2c_full_pipeline.sh`. Expected: Gate-3 SR ≥ 50% on Spatial-100-rollout (recovering at least Gate-1 untrained 80% baseline, plus PSE-trained improvement).

## Compute spent this round

- Gate 1 R1+R2+R3: ~1 hr × dual-GPU ≈ ¥6
- Gate 3 (training broken): ~2.5 hr × dual-GPU ≈ ¥20
- Diagnostic ckpt scan: ~25 min × dual-GPU ≈ ¥3
- **Total this round: ~¥30**
