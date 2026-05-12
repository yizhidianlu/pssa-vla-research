# Response Letter Skeleton — PSSA-VLA Revision

*Use this skeleton as a starting point for the response letter once the
revisions in `revision_roadmap.md` are implemented. Per the
`/ars-revision-coach` skill: this letter is a SKELETON, not the
revised paper itself; the author writes the actual responses after
the revisions are executed.*

---

## Top-of-letter summary (1 paragraph)

> We thank all five reviewers for their detailed and constructive
> feedback. The reviewers correctly identified two coherence problems
> in the original submission: (i) a mismatch between the Method
> section's description and the implementation actually evaluated, and
> (ii) a confusion between Path A (a positive-result paper) and Path B
> (an experience report). We have rewritten the paper to commit to
> **Path \_\_\_** [author fills in]. The major changes are:
> [bulleted list of 3–5 most important changes, e.g., "(a) Method §3
> now describes only the CNN-based PSE encoder actually evaluated; the
> POGS / SAM-2 / depth narrative is moved to Future Work §6", "(b)
> Added a matched OpenVLA-FT control evaluation at $n=10$/task on the
> same hardware/cache as PSSA, closing a previously uncontrolled
> source of the 27 pp regression we report", "(c) Re-ran the
> untrained variant_A evaluation with 3 seeds and added 95\%
> bootstrap CIs to Table 2"...].

---

## Per-reviewer response stubs

### To Reviewer 1 (Methodology + statistics)

**On W1.1 (Method↔Implementation mismatch):**
> We have rewritten §3 to describe only the implemented CNN-based PSE
> encoder. The POGS / SAM-2 / depth-lift / predicted-effect-residual
> narrative is now confined to §6 Future Work, where we explicitly
> mark each component as "proposed but not yet implemented." This
> resolves the truth-in-advertising concern. [If Path A: alternatively,
> we have implemented the per-frame SAM-2 mask path; see §5.X for
> the new experimental results.]

**On W1.2 (single seed, no CIs):**
> We have re-run [untrained variant_A | best trained config] with
> three independent seeds and added 95\% bootstrap confidence
> intervals to Table 2. Specifically:
> - Untrained variant_A: 51.0\% [SR_seed1, SR_seed2, SR_seed3] →
>   bootstrap CI [LO, HI]
> - Trained best (autonomy_v2c-071037): 22\% [SR_seed1, SR_seed2,
>   SR_seed3] → bootstrap CI [LO, HI]
> The conclusion that training degrades the policy holds at the
> p < 0.0X level under Wilcoxon signed-rank.

**On W1.3 (frozen-LoRA isolation):**
> We have added a frozen-LoRA ablation in §5.X (Plan C). The result
> is that training only the PSE encoder with LoRA frozen yields
> SR = XX% on 100 LIBERO-Spatial rollouts, which is [better than /
> equivalent to / worse than] the untrained variant_A baseline. This
> isolates [LoRA / PSE / both] as the destructive ingredient.

---

### To Reviewer 2 (Domain expert)

**On W2.1 (LIBERO-Spatial only, no long-horizon):**
> We agree the original abstract overstated the empirical scope.
> [Path B: We have rewritten the paper as a focused empirical study
> on LIBERO-Spatial and explicitly defer long-horizon evaluation to
> the follow-up. The abstract now reflects this scope precisely.]
> [Path A: We have added LIBERO-LONG untrained results in §5.X
> (single-suite control). The untrained-prefix SR on LIBERO-LONG is
> XX%, $-$YY pp from the Phase-1 LONG baseline of 45.8\%.]

**On W2.2 (27 pp regression is not "modest"):**
> The reviewer is correct that calling the 27 pp regression "modest"
> understated the cost. We have rewritten §5.2 to frame this as a
> substantial regression that the architecture must overcome via
> training. We additionally explored an adapter-tuning variant where
> PSE features are fused with existing image tokens rather than
> occupying new positions [if implemented; otherwise note as future
> work]. Result: [SR number].

**On W2.3 (CRED not tested):**
> We have moved CRED from a claimed contribution to a clearly-marked
> proposed-but-untested mechanism in §3.4 and §6. The abstract no
> longer claims CRED is evaluated. We have not implemented the
> SAM-2-mask path that would make CRED operationally testable; this
> is the most important follow-up.

---

### To Reviewer 3 (Hostile / Reviewer-2 archetype)

**On W3.1–W3.3 (contribution overclaiming):**
> We have substantially narrowed the contributions list. The current
> version is:
> 1. [Path B: A reproducible 6-bug implementation checklist for
>    prefix-tuned VLAs, with cited code references.]
> 2. [A 4-suite OpenVLA-FT reproduction, including matched-$n=10$
>    control.]
> 3. [Empirical study of PSE-prefix injection on LIBERO-Spatial with
>    3 seeds and Wilcoxon test.]
> We do not claim a positive method that improves on OpenVLA;
> instead, we present a negative finding under a specific prefix-tuning
> recipe and document what would be required for prefix-tuned VLAs to
> recover the baseline.

**On W3.4 (passive-voice softening):**
> The reviewer is correct. We have replaced softened phrasings:
> - "naive supervised training degrades the policy" → "our supervised
>   training procedure makes the model worse than the un-trained
>   baseline across five learning-rate / initialization configurations"
> - "operationally non-zero" → "we did not test CRED end-to-end"
> - "below numerical resolution" → "the XTC loss is essentially zero
>   throughout training in the implementation reported"

**On W3.5 (author / compute / impact statements):**
> Added in §6.X. Total compute: XXX GPU-hours across YY days on 1–2
> NVIDIA A800 80 GB GPUs; cost ≈ ¥XXX.

**On W3.6 (citation verification):**
> We have audited `refs.bib`. [Specifically: kitchenvla was [missing /
> present] in the original bibliography; we have added / corrected the
> entry. We verified all 14 citations against Semantic Scholar /
> arXiv as of [date].]

---

### To Reviewer 4 (Area chair, coherence)

**On Path A vs Path B:**
> We have committed to Path \_\_\_ [author fills]. The abstract,
> introduction, and §5 now consistently frame the paper as
> [a positive-result method / an empirical experience report].

**On contribution↔finding alignment:**
> The four contributions in §1 now map 1-to-1 to the four findings in
> §5.

**On ablation of $N$ and prefix position:**
> [If executed: added Figure 5 showing $N \in \{4, 8, 16, 32\}$
> untrained SR and Table 3 extending Figure 2's Gate-1 control to
> $n=10$ × 10 tasks for both variant_A (after_image) and variant_B
> (before_action).]
> [If not executed: deferred to follow-up; we acknowledge this is a
> limitation.]

**On §5.5 (bug checklist) promotion:**
> We have promoted the bug inventory to its own §4. Each of the six
> bugs is now described in ~150 words with a code excerpt.

---

### To Reviewer 5 (Statistics)

**On W5.1 (matched control evaluation):**
> We have run an additional matched OpenVLA-FT control at $n=10$
> rollouts/task on the same machine and same HF cache as the PSSA
> evaluations. Result: OpenVLA-FT (no PSE) at matched $n=10$/task is
> XX%. The PSE-prefix regression vs this matched baseline is YY pp,
> [smaller / equal / larger] than the YY pp reported against the
> Phase-1 $n=50$/task baseline. This [strengthens / weakens] our
> claim that the 27 pp regression is attributable to the PSE prefix
> rather than sample-size or cache effects.

**On W5.2 (Table 2 confidence intervals):**
> All Table 2 SRs now carry 95\% bootstrap CIs (5,000 resamples).

**On W5.3 (single-seed loss plateau):**
> We have re-run [the best trained config] with 3 seeds. The loss
> plateaus at [µ ± σ] across seeds, confirming the plateau is not
> an artifact of a particular initialization.

---

## Remaining open items (acknowledged limitations)

Items the author chose NOT to address in this revision, with
justification:

- [e.g., "Full LIBERO-LONG / Plus / CALVIN ABC-D head-to-heads are
  deferred to a follow-up because they would only be informative
  conditional on recovering the OpenVLA-FT ceiling, which is itself
  the subject of this revision."]
- [e.g., "Per-frame SAM-2 masks code (§5.6 follow-up 1) is committed
  to the repository (`experiment/code/pssa/sam2_masker.py`) but is
  not yet trained end-to-end."]

---

## Sign-off (author fills)

> We thank the reviewers and ACs again for the careful reading. We
> believe the revised manuscript addresses the major concerns and
> we look forward to the next round of review.
