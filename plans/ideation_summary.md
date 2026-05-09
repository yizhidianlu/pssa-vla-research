# Ideation Summary — Persistent Temporal Scene-Spatial Alignment for VLA

Session: 20260509000456c3a8
Stage: ideation → planning
Date: 2026-05-09

---

## 1. Problem statement (in one paragraph)

Existing VLA models (RT-2, OpenVLA, π0, and recent long-horizon successors Long-VLA / SeqVLA / Seer) ground language onto vision **per frame**: at every decision step the VLM re-binds the instruction to whatever pixels it currently sees, with no enforced identity continuity for the same scene entity across time. This produces three failure modes that are now well-documented in the literature: (a) cascading errors over long-horizon and multi-stage tasks (LIBERO-LONG, CALVIN ABC-D), (b) brittle behavior under occlusion, distractors, and viewpoint perturbations (LIBERO-PRO, LIBERO-Plus), and (c) reactive-only correction — current self-correction methods (VLA-in-the-Loop, ReCAPA, KitchenVLA) detect failure either after it occurs or via an extra generative world model / VLM evaluator at inference cost.

## 2. Identified research gap

Three nearby threads exist, but none combines:
- **Long-horizon VLA** (Long-VLA, SeqVLA, Seer) — temporal at the action / phase level, not at the entity level.
- **Persistent / 4D scene representation** (POGS, 4D-GS, DovSG, Motion-Blender GS) — produces identity-stable entity tracks, but used for tracking, rendering, or planning prompts, NOT as the grounding signal of the action head.
- **Execution-time correction** (VLA-in-the-Loop, ReCAPA) — uses an external world model to correct, paying generative-model inference cost.

The unfilled niche: **a persistent scene-entity representation that (i) replaces per-frame VLM grounding inside the VLA action head, and (ii) produces an intrinsic cross-time consistency residual usable for execution-time error detection and correction without an external world model.**

## 3. Proposed contribution (high-level)

We propose **PSSA-VLA** (Persistent Scene-Spatial Alignment VLA), with three contributions:

1. **Persistent Scene-Entity Tokenizer (PSE-Tok).** A lightweight 4D scene representation (initialized from a POGS-style persistent Gaussian splat) maintained across the full episode, producing per-step identity-stable entity tokens. The action head conditions on these entity tokens *plus* the current frame, instead of on the current frame alone.

2. **Cross-Time Consistency Loss (XTC-Loss).** A self-supervised loss enforcing that entity tokens for the same scene entity remain on a smooth trajectory in entity-feature space, with bounded geometric residual under predicted action effects. Adds no new ground-truth annotation.

3. **Consistency-Residual Error Detector (CRED).** At inference, deviations from the entity-trajectory prior produce a per-step residual that (a) flags execution drift earlier than action-confidence baselines, and (b) drives a cheap, action-space corrective replan — no extra world model, no extra VLM evaluator.

## 4. Research questions

- **RQ1 (capability):** Does PSSA-VLA improve long-horizon success rate on LIBERO-LONG and CALVIN ABC-D over a matched OpenVLA / π0 / Long-VLA baseline at the same parameter budget?
- **RQ2 (robustness):** Does persistent scene-entity grounding improve robustness on LIBERO-Plus / LIBERO-PRO perturbations (occluder, distractor, viewpoint shift) over per-frame-grounded baselines?
- **RQ3 (correction):** Does the consistency residual (CRED) detect impending failures earlier and at lower compute cost than world-model-based VLA-in-the-Loop and confidence-based baselines, and does on-the-fly correction translate the detection into success-rate gain?
- **RQ4 (ablation):** Which of {PSE-Tok, XTC-Loss, CRED} contributes the most to each of RQ1–RQ3?

## 5. Hypotheses (target effect sizes for hypothesis testing)

| ID | Hypothesis | Target metric | Target effect | Tied to |
|----|-----------|---------------|---------------|---------|
| H1 | PSE-Tok grounding > per-frame grounding on long-horizon | LIBERO-LONG success rate | +5 to +10 pp absolute over matched baseline | RQ1 |
| H2 | PSE-Tok improves robustness under perturbation | LIBERO-Plus avg success rate | +6 pp absolute | RQ2 |
| H3a | CRED detects failure earlier than action-confidence | AUROC of failure-prediction over residual horizon | AUROC ≥ 0.80 vs ≤ 0.65 baseline | RQ3 |
| H3b | Closed-loop correction with CRED improves success | LIBERO-LONG success with CRED on/off | +3 pp on top of H1 gain | RQ3 |
| H4 | PSE-Tok is the dominant contribution | Ablation drop when PSE-Tok off | ≥ 60% of total gain attributable to PSE-Tok | RQ4 |

## 6. Risks and mitigations

- **R1 — Persistent splat init is unreliable on novel scenes.** Mitigation: warm-start from off-the-shelf SAM-2 + tracker, use POGS-style self-supervision; fall back to per-frame tokens if entity confidence < threshold.
- **R2 — XTC-Loss collapses (entity tokens become near-constant).** Mitigation: contrastive negative against augmented frames; impose action-conditioned predicted-residual constraint.
- **R3 — CRED produces false alarms during legitimate object pose changes.** Mitigation: condition residual on predicted-action-effect rather than raw motion; quantify on a held-out validation slice.
- **R4 — Compute budget for 4D scene rep on-robot.** Mitigation: use Hybrid 3D-4D GS or sparse Motion-Blender variant; profile inference latency as a first-class metric.

## 7. Decision for next stage

Proceed to PLANNING with original_research mode. Primary benchmarks: LIBERO-LONG, LIBERO-Plus, CALVIN ABC-D. Secondary: VLABench. Baselines: OpenVLA, π0 (reproduction), Long-VLA, Seer. Strongest baseline target: Seer.
