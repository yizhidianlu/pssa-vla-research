# Literature Seed — Persistent Temporal Scene-Spatial Alignment for VLA

Session: 20260509000456c3a8
Stage: ideation
Compiled: 2026-05-09

Verified web-search hits only. Each entry annotated with what it does and how it relates to our proposed gap.

---

## A. Long-horizon VLA architectures (the "rivals")

### A1. Long-VLA — Unleashing Long-Horizon Capability of Vision Language Action Model for Robot Manipulation
- arXiv: 2508.19958
- Method: phase-aware input masking; segments each subtask into MOVE / INTERACT phases; adaptive sensory focus.
- Relation to us: addresses long-horizon, but **grounding is still per-frame inside each phase**; no identity-persistent entity track across the whole rollout.

### A2. SeqVLA — Sequential Task Execution for Long-Horizon Manipulation
- arXiv: 2509.14138
- Method: completion-aware extension of π0 with a lightweight detection head; auto-triggers subtask transitions.
- Relation to us: solves "when did subtask k finish?" — orthogonal to our "what is the same object across all frames" question.

### A3. FUTURE-VLA — Forecasting Unified Trajectories Under Real-time Execution
- arXiv: 2602.15882
- Method: spatiotemporal compression + autoregressive latent prediction of long-horizon action chunks.
- Relation to us: temporal at the *action* level, not the *scene-entity* level. We supply identity-stable scene tracks; FUTURE-VLA could consume them as input.

### A4. ThinkAct — Vision-Language-Action Reasoning via Reinforced Visual Latent Planning (NeurIPS 2025)
- jasper0314-huang.github.io/thinkact-vla
- Method: dual-system; multimodal LLM generates embodied plans guided by RL with action-aligned visual rewards including trajectory consistency.
- Relation to us: closest in spirit on "consistency reward", but consistency is an RL signal over latent plans, not over identity-persistent scene entities.

### A5. DynamicVLA — A VLA Model for Dynamic Object Manipulation
- arXiv: 2601.22153
- Method: 0.4B VLA, latent-aware action streaming, continuous inference for inference-delay-tolerant control.
- Relation to us: targets dynamic scenes through *streaming inference*, not through *persistent scene representation*.

### A6. DAM-VLA — Dynamic Action Model-Based VLA Framework
- Samsung Research / arXiv 2603.00926
- Relation to us: another dynamic-VLA point of comparison; baseline candidate.

### A7. Green-VLA — Staged VLA for Generalist Robots
- HuggingFace papers / arXiv 2602.00919
- Method: 5-stage curriculum + temporal alignment + embodiment-aware action interface.
- Relation to us: temporal alignment is curriculum-side, not scene-rep-side.

### A8. MMaDA-VLA — Large Diffusion VLA
- arXiv: 2603.25406
- Method: discrete-diffusion masked-token denoising; jointly generates future goal observations and action chunks.
- Relation to us: future-frame generation as the consistency signal; complementary, but generative not entity-tracking.

---

## B. Persistent / 4D scene representation (the "ingredient")

### B1. POGS — Persistent Object Gaussian Splat for Tracking and Manipulation (ICRA 2025)
- Berkeley AUTOLab
- Method: integrates language, grouping, self-supervised visual features into an explicit 3D Gaussian rep that persists across interactions of unseen, irregular objects.
- Relation to us: **closest persistent-object building block**; we cite as the candidate scene-entity backbone but extend by *coupling the persistent splat into the VLA action head as grounding tokens*, not just as a tracker.

### B2. 4D Gaussian Splatting — 4D-GS (CVPR 2024) and 4DGS-1K (NeurIPS 2025)
- arXiv: 2310.08528
- Method: 4D primitives for real-time dynamic scene rendering at 1000+ FPS.
- Relation to us: rendering primitives we can repurpose as a temporally consistent latent scene state.

### B3. Hybrid 3D-4D Gaussian Splatting (OpenReview)
- Method: static regions → 3D, dynamic → 4D, adaptive split.
- Relation to us: budget-aware variant — interesting for real-time on-robot use.

### B4. Motion-Blender Gaussian Splatting
- arXiv: 2503.09040
- Method: explicit, sparse motion graphs over GS for dynamic-scene reconstruction; supports robot manipulation planning, demo synthesis, and visual action prediction.
- Relation to us: explicit motion graph is one realization of "scene-entity track"; we differ by *learning the entity grouping end-to-end with the VLA head*.

### B5. POGS / SLAM lineage — 4D Gaussian Splatting SLAM (ICCV 2025)
- Method: SLAM-side temporal-consistent GS.
- Relation to us: localization layer, complementary.

### B6. DovSG — Dynamic Open-Vocabulary 3D Scene Graphs for Long-term Language-Guided Mobile Manipulation
- arXiv: 2410.11989
- Method: open-vocab 3D scene graph + language-guided planner; locally updates graph during interactions.
- Relation to us: graph used for *high-level planning prompts*, not as low-level grounding tokens for the action head — that's our extension.

### B7. CogACT — semantic scene graph + diffusion action head
- Method: scene graph as conditioning for diffusion policy.
- Relation to us: confirms scene-graph conditioning works; we differ by maintaining graph identity across all frames and surfacing consistency residuals.

---

## C. Execution-time error detection & self-correction (the "second mechanism")

### C1. VLA-in-the-Loop — Online Policy Correction with World Models for Robust Robotic Grasping (OpenReview)
- Method: lightweight composite world model as event-triggered "corrector"; if proposed action is unviable, generative model synthesizes successful future video; inverse-dynamics decoder produces correction actions.
- Relation to us: our consistency residual replaces the world-model corrector with a cheaper, identity-anchored signal — no generative video step.

### C2. ReCAPA — Hierarchical Predictive Correction to Mitigate Cascading Failures
- arXiv: 2604.21232
- Method: hierarchical prediction over executions to catch cascading errors.
- Relation to us: confirms the cascading-failure problem is open; we attack it with a different mechanism (entity-track consistency, not predictive hierarchy).

### C3. KitchenVLA — Zero-shot Action Planning and Correction
- Method: VLM-evaluator analyzes human video vs robot observation to detect domain mismatches.
- Relation to us: external VLM evaluator vs our self-supervised entity-track residual — we are cheaper at inference time.

---

## D. Benchmarks

### D1. LIBERO suite (Spatial / Object / Goal / 100 / LONG / Plus)
- LIBERO-PRO (arXiv 2510.03827) and LIBERO-Plus (arXiv 2510.13626): show VLAs are brittle to perturbations and may memorize.
- Relation to us: primary evaluation; LIBERO-LONG is where our long-horizon claim must land. LIBERO-Plus is where our consistency-correction claim must show robustness gain.

### D2. CALVIN (ABC-D)
- 5 chained language instructions per rollout, shared tabletop scene.
- Relation to us: **the cleanest test for persistent scene rep** — same scene, evolving objects across instructions.

### D3. VLABench (ICCV 2025)
- Large-scale language-conditioned long-horizon manipulation benchmark.
- Relation to us: secondary benchmark; complementary to LIBERO/CALVIN.

### D4. Seer (ICLR 2025) — strongest reported long-horizon baseline
- +10.4 pp success rate, +0.75 average task length on LIBERO-LONG and CALVIN ABC-D over prior SOTA.
- Relation to us: strongest baseline to beat or match-with-fewer-params on long-horizon tracks.

---

## E. Surveys / context (cite for related work)

- Vision-Language-Action Models: Concepts, Progress, Applications and Challenges — arXiv 2505.04769
- VLA Models in Robotic Manipulation: A Systematic Review — arXiv 2507.10672
- Pure VLA Models: A Comprehensive Survey — arXiv 2509.19012
- Large VLM-based VLA Models for Robotic Manipulation: A Survey — arXiv 2508.13073
- A Survey on Efficient VLA — arXiv 2510.24795
- A Survey on VLA: An Action Tokenization Perspective — arXiv 2507.01925
