# Experiment Blueprint — PSSA-VLA

Session: 20260509000456c3a8
Stage: planning
Date: 2026-05-09
Tied-to: plans/ideation_summary.md (RQ1–RQ4, H1–H4)

---

## 1. Problem framing for experiments

We test whether replacing per-frame VLM grounding inside a VLA action head with a persistent, identity-stable scene-entity representation, plus a consistency residual at inference, yields:

- (E1) higher success on long-horizon tasks,
- (E2) higher robustness on perturbed evaluation,
- (E3) earlier and cheaper failure detection / correction than world-model correctors.

## 2. Method under test — PSSA-VLA

Three components:

- **PSE-Tok (Persistent Scene-Entity Tokenizer).** Per episode: initialize a persistent Gaussian splat of the scene from the first 2–4 frames using SAM-2 + Gaussian Splatting; maintain it across the rollout via per-step warp + ID-preserving update (POGS-style). Read out N entity tokens (positions in 3D + appearance feature + identity ID) at every action step.
- **VLA action head conditioning.** Replace the per-frame VLM grounding tokens of an OpenVLA-style head with concatenation of {language tokens, current-frame ViT tokens, PSE entity tokens}. Action head architecture and parameter count held constant; only the grounding token source changes.
- **XTC-Loss (cross-time consistency, training-time).** L_xtc = λ1·||f_t − f_{t-1} − Δf_pred(a_{t-1})|| + λ2·contrastive(f_t, augmented frame).
- **CRED (consistency-residual error detector, inference-time).** r_t = ||f_t − (f_{t-1} + Δf_pred(a_{t-1}))||; if r_t > τ for K consecutive steps, freeze action and trigger a corrective replan from the persistent entity prior.

## 3. Baselines (matched parameter budget where feasible)

| Baseline | Reason | Source |
|----------|--------|--------|
| OpenVLA-7B (frozen weights, fine-tuned head) | canonical per-frame VLA | public ckpt |
| π0 (reproduction or open weights) | strongest open generalist VLA | public ckpt |
| Long-VLA | direct long-horizon SOTA-comparison | arXiv 2508.19958 |
| SeqVLA | direct subtask-completion comparison | arXiv 2509.14138 |
| Seer | strongest long-horizon baseline | ICLR 2025 |
| VLA-in-the-Loop | external-world-model corrector — RQ3 head-to-head | OpenReview |
| (Ablation A) PSSA without XTC-Loss | tests H4 | ours |
| (Ablation B) PSSA without CRED | tests H3b vs H1 | ours |
| (Ablation C) PSSA with PSE-Tok replaced by per-frame ViT tokens | isolates persistence | ours |

## 4. Datasets / benchmarks

| Benchmark | Purpose | Splits used |
|-----------|---------|-------------|
| LIBERO-LONG | RQ1 — long-horizon success | full |
| LIBERO-Spatial / Object / Goal | sanity at short-horizon | full |
| LIBERO-Plus / LIBERO-PRO | RQ2 — robustness under perturbation | OOD splits |
| CALVIN ABC-D | RQ1, RQ3 — chained 5-step instructions, persistent scene | ABC→D zero-shot |
| VLABench (subset) | secondary RQ1 | long-horizon track |
| Open X-Embodiment (small subset) | pretraining warmup for PSE-Tok | bridge / RT-1 mix |

## 5. Metrics

- **Primary:** task success rate (SR), average completed instructions per rollout (ACR), long-horizon success @ horizon ≥4.
- **Robustness:** Δ-SR between LIBERO-base and LIBERO-Plus splits.
- **Detection:** AUROC of per-step failure prediction with horizon h ∈ {5, 10, 20} steps.
- **Correction:** SR with CRED on vs off; corrections-per-rollout count; latency overhead.
- **Compute:** params, peak GPU memory, inference latency (ms / step), end-to-end episode wall-clock.
- **Reporting:** mean ± stderr over 3 seeds; LIBERO uses 50 rollouts per task; CALVIN follows ABC-D protocol.

## 6. Statistical plan

- 3 seeds; report mean ± 1 stderr.
- Wilcoxon signed-rank test on per-task success counts for PSSA vs each baseline; Bonferroni correction across the 5 baselines.
- For H3a (AUROC): bootstrap 95% CI over 5000 resamples of failure-prediction labels.

## 7. Threats to validity (named, since reviewers will ask)

1. **Memorization / overfitting on LIBERO** (LIBERO-PRO showed VLAs may memorize). Mitigation: report LIBERO-Plus and CALVIN ABC-D zero-shot as primary, not LIBERO-base.
2. **Param-count confound.** Mitigation: held-constant action-head budget; report total params and GPU mem in every comparison.
3. **PSE-Tok hand-tuned per benchmark.** Mitigation: one shared init recipe across all benchmarks; report variance across seeds.
4. **Reproduction quality of baselines (π0, Long-VLA, Seer).** Mitigation: cite original numbers + run our reproduction; flag any gap > 3 pp.
5. **CRED's correction adds latency.** Mitigation: ablation reporting latency-vs-gain tradeoff.

## 8. Compute budget (planning estimate)

- Pretraining warmup of PSE-Tok on Open X mini-subset: ~1× A100 day.
- Fine-tune action head on LIBERO + CALVIN: ~2× A100-day per config; 6 configs (PSSA, 3 ablations, OpenVLA reproduction, π0 reproduction) → ~12 A100-days.
- Eval rollouts (3 seeds × benchmarks): ~2 A100-days.
- **Total target: 16–20 A100-days across all experiments.**
- Hardware needed: ≥2× A100 80GB or ≥4× RTX 4090 with offload; SLURM-friendly.

## 9. Decision gates

- After SETUP: confirm SAM-2 + GS init runs on a single CALVIN scene at < 200 ms per frame; if not, fall back to point-track-only PSE-Tok variant.
- After CODING: smoke-test pass on 5-rollout LIBERO-LONG subset with random init.
- After EXECUTION (LIBERO-LONG only): if PSSA SR < OpenVLA reproduction SR + 2 pp, halt and diagnose before scaling to CALVIN / VLABench.

## 10. Deliverables expected from each later stage

| Stage | Artifact | Path |
|-------|----------|------|
| setup | env + dataset list | experiment/setup.md |
| coding | runnable code with smoke test | experiment/code/, experiment/smoke_test.log |
| execution | metrics CSV per config × seed × benchmark | experiment/runs/{config}_{seed}_{benchmark}.csv |
| analysis | results.md + tables.csv | analysis/ |
| figure_gen | framework + qualitative + bar plots | figures/ |
| writing | full LaTeX draft | drafts/main.tex |
| review | revised draft + response notes | output/ |
