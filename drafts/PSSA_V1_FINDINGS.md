# PSSA-VLA v1.1 — Baseline + Ablation findings

Date: 2026-05-11
Suite: LIBERO-Spatial, 10 tasks × 10 rollouts = 100 rollouts per config.

## Setup

Two parallel training runs on different A800 instances, identical config except `lambda_xtc`:

| Run | Machine | λ_xtc | Steps | Final loss | LIBERO-Spatial SR |
|---|---|---|---|---|---|
| Baseline | new dual-A800 (16967) | 0.1 | 3000 | 0.288 | **0/100 (0.0%)** |
| Ablation (no-XTC) | old A800 (50535) | 0.0 | 3000 | ~0.24 | **0/100 (0.0%)** |

OpenVLA-7B-finetuned-libero-spatial baseline (Phase 1): **80.2%** (401/500).

## Architecture observation

v1.1 PSSA-VLA bypasses OpenVLA's DinoSigLIP vision backbone. The mismatch:
- DinoSigLIP expects `(B, 6, 224, 224)` after its own preprocessing pipeline
- LIBERO demos come as raw `(B, 3, 128, 128)`

In v1.1 we feed `[text_embeds | PSE_tokens]` only — vision is replaced by entity tokens
encoded from the first 4 frames after `env.reset()`. The model is then trained to predict
continuous 7-DoF actions via a small MLP head on the last LLM hidden state.

## Why XTC is numerically dead in v1.1

XTC = `mean((f_t - f_{t-1})^2)` over per-step entity features.

`model_v2.py:179` re-encodes entity features for each step `t` using:
```python
ent_t = self.pse_encoder(rgb_seq[:, t:t+1], masks_init[:, :1].expand(...))
```

The mask is the **same** init mask across all timesteps. With LIBERO 128×128 frames that
change slowly (robot motion is local), the CNN output for adjacent frames is nearly
identical → `L_xtc < 5e-5` throughout training (rounds to 0.0000 with `:.4f` format).

Effective contribution to total loss:
```
λ_xtc=0.1 × L_xtc(~5e-5) = 5e-6  vs  L_act ≈ 0.2-0.3
```

XTC contributes < 0.003% of the gradient signal. So the "λ_xtc=0.1 vs 0" comparison
**collapses into a 2-seed sanity check**, which we confirmed: both runs hit 0% SR with
training loss trajectories within RNG noise.

## What 0/100 SR means

Without per-step visual feedback, the policy outputs a sequence that's conditioned on:
1. Init entity tokens (8 vectors derived from frame 0-3)
2. Language instruction
3. **Same hidden context for all 200 steps within a rollout**

This is open-loop control — the LLM never sees what the gripper just did. Continuous-MLP
prediction also lacks the discretization-by-token structure OpenVLA's native head relies
on. Result: policy fails to reach the manipulation target in all 100 rollouts.

## Implications for the paper

1. **v1.1 cannot be the headline PSSA result.** It validates the training pipeline
   (LoRA + PSE-Tok + action head all train successfully) but the policy is blind.
2. **XTC ablation needs v2.** Until per-step visual entity encoding fires the L_xtc term,
   the ablation cannot inform whether XTC helps.
3. **v2 path**: integrate OpenVLA's DinoSigLIP preprocessor + use its native discrete
   action token decoder, treat PSE-Tok as an additional prefix to the existing
   [vision | text | action_query] sequence.

## Footnote draft for §5 Ablation table

> v1.1's vision-bypass means the cross-time consistency loss collapses to <5×10⁻⁵
> throughout training (numerical floor), so λ_xtc=0.1 vs 0.0 produce statistically
> indistinguishable training trajectories and identical 0/100 LIBERO-Spatial SR. We
> defer the XTC ablation to v2, which restores per-step visual entity features.
