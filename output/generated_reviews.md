# Generated 5-persona adversarial review — PSSA-VLA draft (commit `1b2efdf`)

This is a synthetic review used as input to the revision-coach pipeline.
Five personas review the paper as if it were submitted to a top robot
learning venue (CoRL / ICRA / RSS-class). Each reviewer is asked to score
on a 1–5 scale and list specific weaknesses + revision asks.

---

## Reviewer 1 (Methodology + statistics)

**Score: 2/5 (Borderline reject).**

The paper's most fundamental problem is the mismatch between the
**Method** section's claim ("PSE-Tok built on a POGS-style Gaussian
splat", "SAM-2 masks", "3D depth lift", §3.2) and **what the experiments
actually implement** (§5 / Figure 1 caption: "CNN PSE-Tok encoder", no
SAM-2, no Gaussian splat, no depth). I cannot tell from the paper whether
PSSA-VLA-as-described would behave the same as PSSA-VLA-as-implemented.
The four contributions in §1 do not match the four findings in §5.

Specific weaknesses:

W1.1 The Method (§3.2, §3.3) describes a system involving Gaussian
splats, depth, SAM-2 masks, exponential moving averages on per-entity
features, identity InfoNCE contrastive loss, and predicted-effect
residuals via $\Delta f_\text{pred}(a_{t-1})$. None of these components
appear in the experimental implementation. The implementation
(Figure 1, §5.5) is a tiny CNN with a learned id-embedding and a 2-layer
MLP projector. **The paper as written reports experiments on a
strictly weaker system than the one it claims to evaluate.** This is
a serious truth-in-advertising problem.

W1.2 The "5-config sweep" (Table 2 / §5.3) reports one seed per
configuration on a 10-task / 100-rollout evaluation. With $p \approx 0.5$
and $n=100$, the 95\% confidence interval on a single SR is $\pm 9.8$ pp;
on per-task $n=10$ values it's $\pm 30$ pp. The conclusion that
"training degrades the policy" needs at least 3 seeds per config, or a
Wilcoxon signed-rank test against the untrained-baseline distribution.

W1.3 The "negative training result" is presented as a sharpening of the
design space (§7), but no actual mechanism is isolated. Was it LoRA, was
it the PSE encoder, was it the per-step re-encoding code path, was it
the action-token discretization round-trip? The paper waves at "5
compounding mechanisms" without ablating any of them. Plan C
(frozen-LoRA, §6) is listed as future work but is the most obvious
next experiment a reviewer would want already in the paper.

Revision asks:
- R1.1 Either (a) implement the system described in §3 (POGS / SAM-2 /
predicted-effect residual) and report on THAT, or (b) rewrite §3 to
describe the simpler system actually evaluated, and state the gap
explicitly. Don't have it both ways.
- R1.2 Three seeds, confidence intervals, Wilcoxon vs untrained.
- R1.3 Frozen-LoRA ablation included in the main paper, not future work.

---

## Reviewer 2 (Domain expert, robot learning)

**Score: 3/5 (Weak accept).**

The honest "we ran into a wall" framing is unusual and valuable. The
6-bug checklist in §5.5 is the strongest contribution of the paper and
will save the next implementor weeks. The 4-suite OpenVLA reproduction
in Table 1 is a useful community baseline. However:

W2.1 The headline empirical finding ("PSE prefix preserves backbone,
trained PSE degrades policy") is established on **LIBERO-Spatial only**
(10 tasks, single-suite). The original promise (§1.1, abstract) was
long-horizon manipulation. None of LIBERO-LONG, LIBERO-Plus, or
CALVIN ABC-D were touched. The paper does not deliver on its own
research questions.

W2.2 The 51.0% untrained-prefix SR (Table 2 row 1) is **27 pp below**
the 80.2% no-prefix Phase-1 baseline (Table 1). The paper frames this
as "the backbone retains useful behavior under modest position-embedding
perturbation". A robotics reviewer reads this as "a 27 pp regression
from baseline just by inserting 8 zero-valued tokens." That is not
"modest"; that is a serious cost. The paper must explain why this
regression is acceptable, OR explore prefix-tuning positions that don't
incur it (e.g., adapter-tuning instead of prefix injection).

W2.3 §5.4 admits XTC loss is "below numerical resolution" in the
current implementation. The whole RQ3 (failure detection via CRED) is
deferred because CRED depends on XTC firing. The paper therefore tests
neither RQ1 (long-horizon) nor RQ3 (failure detection). Only RQ2-ish
(prefix injection on LIBERO-Spatial) is reported, and the result is
negative.

Revision asks:
- R2.1 At minimum, report LIBERO-LONG untrained-prefix results (one
  control run; the Phase-1 LONG baseline is already 45.8\% so a
  $-27$ pp ceiling is 18\% — still meaningful as a control)
- R2.2 Position the 27 pp regression as a real cost; consider an
  adapter-based variant where PSE features mix into existing
  token positions rather than occupying new ones
- R2.3 Acknowledge in the abstract that CRED is not tested

---

## Reviewer 3 (Reviewer-2 archetype, hostile)

**Score: 1/5 (Strong reject).**

This paper does not contain a method that improves on OpenVLA. It
contains an architecture diagram, a list of bugs the authors hit while
implementing it, and a table showing every training run does worse than
the un-trained baseline. The contributions list (§1.4) overpromises
massively.

Specific complaints:

W3.1 "Apples-to-apples 4-suite OpenVLA baseline" as a contribution is a
**reproduction**, not a contribution. OpenVLA itself already reports
these numbers; the $-6.8$ pp gap is attributed to a transformers version
mismatch the authors knew about and didn't fix.

W3.2 "Empirical study of PSE-prefix injection" delivers a negative
result on one suite with one seed. This is a workshop paper at best.

W3.3 The "Implementation gotchas" §5.5 is a debug log. It does not
belong in a conference paper. Move it to a blog post or technical
report.

W3.4 The paper repeatedly uses passive-voice softening to hide negative
results: "naive supervised training degrades the policy" instead of
"our training procedure makes the model worse"; "operationally
non-zero" instead of "we have not tested CRED"; "below numerical
resolution" instead of "XTC loss is essentially zero". This is
unprofessional.

W3.5 No author identification, no compute statement, no broader
impact statement.

W3.6 Citation `kitchenvla` is not in the reference list (I checked).
Other citations (e.g. `recapa`, `cogact`) need verification.

Revision asks:
- R3.1 Cut the contributions list to one: the bug checklist. Reframe
the rest as background context.
- R3.2 Run at least one configuration that beats the un-trained
baseline. Otherwise the paper has no positive finding.
- R3.3 Verify every citation. Several look hallucinated.

---

## Reviewer 4 (Area chair / generalist)

**Score: 2/5.**

The paper is in an awkward position: too negative for the headline
claims in §1, too thin on alternative empirical contributions to stand
on its own as a debugging report. The two natural paths forward are:

Path A (positive paper): implement the per-frame SAM-2 mask path
(§5.6 follow-up 1) and frozen-LoRA ablation (§5.6 follow-up 2), recover
the OpenVLA-FT ceiling, then claim a smaller-but-real improvement.
This requires another month of work.

Path B (negative paper / experience report): drop the architectural
contribution framing entirely, reframe as "lessons learned implementing
prefix-tuned VLAs on top of OpenVLA-7B", lead with the 6-bug checklist,
demote PSE/XTC/CRED to discussion. This is a tech-report or workshop
contribution.

The current draft tries to be Path A in the abstract and Path B in §5;
that incoherence will sink it at any top-tier venue.

W4.1 Coherence — pick A or B.
W4.2 The four contributions in §1 do not align with the four findings in §5.
W4.3 No ablation of the entity count $N$, the prefix position
(after\_image vs before\_action — Figure 2 only reports task-0 results),
or the LoRA rank.
W4.4 "Reproducible bug inventory" (§5.5) is genuinely useful but is
buried; promote to its own section.

Revision asks:
- R4.1 Pick path A or path B; rewrite accordingly.
- R4.2 Add ablations on $N$ and prefix position with full $n=100$ rollouts.
- R4.3 Promote §5.5 to §4 (Methods → §3, Bug-inventory → §4).

---

## Reviewer 5 (Devil's advocate / statistics-focused)

**Score: 2/5.**

I want to focus on one issue: the "PSE prefix is benign" claim
(abstract finding b) versus the data.

Phase-1 baseline OpenVLA-FT, $n=500$ rollouts per task, gives 80.2\%
mean SR with per-task SRs $[86, 82, 90, 84, 68, 94, 92, 74, 78, 54]$.
PSSA-untrained-variant_A, $n=10$ rollouts per task, gives 51.0\% mean
SR with per-task SRs $[70, 80, 30, 70, 40, 10, 80, 100, 10, 20]$.

These are TWO DIFFERENT EVALUATIONS with TWO DIFFERENT SAMPLE SIZES.
The 27 pp gap could be:
(a) genuine effect of the PSE prefix (the paper's claim)
(b) sampling noise from $n=10$ per task — variance on $p=0.8$, $n=10$
is $\sigma = 0.13$, so tasks 5 and 8 dropping from 94\%/78\% to 10\%
are within ~3-4σ but only barely
(c) some uncontrolled difference (HF cache, transformers version,
process-level RNG between Phase 1 and Phase 2 runs)

The paper does not even attempt to control for (b) and (c). Without a
matched-evaluation control (i.e., OpenVLA-FT WITH the same machine,
same HF cache, same eval harness, $n=10$ per task), the 27 pp number
is uninterpretable.

W5.1 Need a matched OpenVLA-FT control evaluation at $n=10$ rollouts/task
on the same hardware/cache as PSSA. This is one experiment.

W5.2 Confidence intervals on Table 2.

W5.3 §5.3 reports loss "plateau at 3.3" as if it were a fixed point.
With single-seed runs the loss is one realization; the plateau could
be a local minimum specific to that initialization.

Revision asks:
- R5.1 Matched OpenVLA-FT $n=10$/task baseline as a single new
  experiment. This is the **#1 most important revision** to interpret
  the negative result.

---

## End of generated reviews.
