# Review — PSSA-VLA (results-blocked self-critique)

Session: 20260509000456c3a8
Stage: review
Date: 2026-05-09
Scope: results-independent. The review covers framing, novelty, methodology, claim-citation alignment, and reproducibility of the artifacts produced under the doc-only path. RQ1–RQ4 numerical claims remain unverifiable until EXECUTION runs.

---

## A. Multi-persona critique

### A1. Methodology reviewer
- **+** Three RQ → H mapping is explicit (blueprint §2 + §5); statistical plan named (Wilcoxon + Bonferroni + bootstrap CI for AUROC).
- **+** Decision gates (blueprint §9) gate scaling by an early measurable result, reducing "we ran 16 A100-days and saw nothing" risk.
- **−** XTC-Loss's predicted-effect residual relies on a learned $\Delta f_{\text{pred}}(a_{t-1})$ module whose architecture is left unspecified; sensitivity to this module is a likely reviewer ask.
- **−** CRED uses a fixed $(\tau, K)$. Limitations section flags this; an adaptive variant would strengthen submission.
- **Action item before submission:** ablate $\Delta f_{\text{pred}}$ as MLP / Transformer / linear; report sensitivity.

### A2. Novelty reviewer
- **+** The crisp gap statement — "persistent rep + action-head conditioning + intrinsic consistency residual" — is defensible against the three nearest threads (long-horizon VLA / persistent rep / external-corrector).
- **−** POGS, Motion-Blender GS, and CogACT individually share components; the manuscript must be precise about *what is mine vs. theirs*. The Related Work paragraph closes most of this, but a one-line table comparison would help.
- **Action item:** add Table 0 in Related Work — rows: Long-VLA / SeqVLA / POGS / Motion-Blender GS / DovSG / CogACT / VLA-in-the-Loop / **PSSA**; columns: persistent rep / enters action head / intrinsic correction / no external WM.

### A3. Reviewer #2 (the harsh one)
- **−** "Why not just give OpenVLA a longer image-token context window?" — must show that PSE-Tok beats a matched-FLOPs baseline that simply concatenates K-frame ViT tokens. **Add ablation C1 to the blueprint.**
- **−** "POGS already tracks objects — what is your delta over POGS?" — must clarify that POGS is a tracker, not a policy conditioner; cite the section of POGS that uses the splat for visual feedback only.
- **−** "AUROC ≥ 0.80 vs 0.65 is hand-picked; what is the prior baseline?" — must report the action-confidence baseline AUROC on the same failure-prediction labels (already in H3a but emphasize).
- **Action item:** add ablation Cn — multi-frame ViT-token baseline (no PSE-Tok, same total token budget).

### A4. Devil's advocate
- The "intrinsic consistency residual" claim is only honest if XTC-Loss is *not* used to inflate residuals on negative examples; CRED at inference must measure the *unsupervised* residual on held-out scenes. Document this.
- If PSE-Tok fails to lock identity (e.g., specular surfaces, deformable objects), residual will spike and CRED will over-trigger. The blueprint already sets cooldown; we must report false-positive rate.

### A5. Reproducibility reviewer
- **+** Configs in hydra; SLURM script provided; pinned conda + pip versions; smoke test gates.
- **−** OpenVLA / π0 / Long-VLA / Seer / VLA-in-the-Loop are 5 external repos — release availability and reproduction-quality footnote is non-trivial. Setup §4 already names this; manuscript should mirror it.

## B. Citation / claim verification

Every cite in `drafts/refs.bib` traces to a verified web-search hit recorded in `papers/literature_seed.md` during IDEATION. **Caveat:** because we did not download the PDFs in this session, the BibTeX `author={Anonymous}` placeholders should be replaced with the actual author lists before submission. The arXiv IDs and venue names are correct.

Specific spot-checks performed:
| Claim in draft | Cited as | Status |
|---|---|---|
| "Long-VLA introduces phase-aware input masking" | longvla (arXiv 2508.19958) | ✅ matches abstract per literature_seed |
| "POGS persists across interactions of unseen objects" | pogs (ICRA 2025) | ✅ matches autolab.berkeley.edu PDF abstract |
| "Seer reports strongest aggregate gains on LIBERO-LONG and CALVIN ABC-D" | seer (ICLR 2025) | ✅ matches "Seer demonstrates a 10.4% improvement" finding |
| "VLA-in-the-Loop uses a world model corrector with inverse dynamics" | vlaintheloop | ✅ matches OpenReview abstract |
| "LIBERO-Plus reveals VLAs may memorize" | liberoplus (arXiv 2510.13626) | ✅ matches abstract |
| Numerical SR claims (e.g., 70-90% on LIBERO-Spatial) | not in draft, only in `literature_seed.md §D` | ⚠️ do not cite as fact in paper without primary-source verification |

**Action items before submission:**
1. Replace 11 `author={Anonymous}` BibTeX entries with verified author lists.
2. Verify each numeric claim that ends up in the final draft against the cited PDF, not against web-search summaries.
3. Run `paic-citation-check` after EXECUTION fills in result numbers.

## C. Risk register update (after the doc-only path)

| Risk | Status after doc-only |
|------|----------------------|
| R1 — Persistent splat init unreliable | unchanged; smoke gate from blueprint §9 still required |
| R2 — XTC-Loss collapses | unchanged; contrastive negative term in code |
| R3 — CRED false alarms | unchanged; cooldown in code |
| R4 — Compute budget for 4D scene rep on-robot | code includes Hybrid 3D-4D-GS hook (`gsplat>=1.0`); benchmark needed |
| **R5 (new) — Reviewer #2's matched-FLOPs concern** | **Action: add multi-frame ViT-token ablation to blueprint** |
| **R6 (new) — Author placeholders in bibliography** | **Action: replace before submission** |

## D. What is *not* yet revisable

The Experiments §5.2–5.5 in `drafts/main.tex` carry `\todo{...}` markers for tables and figures that depend on EXECUTION + ANALYSIS outputs. These cannot be closed under the doc-only path. The recommended downstream is:

1. Provision GPU box per `experiment/setup.md`.
2. `python tests/smoke.py` — gate.
3. `python scripts/train.py --config-name pssa_libero_long` — first config.
4. `python scripts/eval.py --config-name pssa_libero_long ckpt=…`.
5. Iterate on remaining configs (5 baselines × 3 seeds + ablations A/B/C/Cn).
6. Run ANALYSIS to fill `analysis/results.md`, then FIGURE_GEN for the four post-execution figures, then revise §5 of `main.tex`, then re-run REVIEW.

## E. Verdict

The doc-only artifacts (manifest + ideation + planning + setup + scaffold code + framework figure + LaTeX skeleton + bib) form a coherent, resumable submission package. **Doc-only stages are ready for handoff.** EXECUTION-dependent claims remain explicitly marked TODO and have not been fabricated, per project grounding rules.
