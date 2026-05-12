# 行动指南 — PSSA-VLA（防跑偏 / 防遗忘）

**用途**：跨 session、跨上下文压缩持久存在的"北极星文档"。任何继续工作的 Claude 实例**必须**先读这一份，再读 manifest 与代码。每次接续工作前 grep 一次本文件，确认目标没有漂移。

---

## 0. 一句话定位

PSSA-VLA 是一篇**正向论文**：论证持久场景-实体表示（PSE-Tok）作为 VLA 动作头条件，能在长时程操控上**超过** OpenVLA-FT 基线。

> ❌ 不是负面报告。不是 "训练让模型变差" 的经验教训。不是责任划分练习。
> 当前 v2c 训后 22 % 是 **bug 累积**，不是 finding —— 必须修，不能写。

---

## 1. 原始预期指标（hypotheses H1–H4，源自 `plans/ideation_summary.md` §5）

| ID | 假设 | 指标 | OpenVLA-FT 基线 | PSSA 目标 | 状态 |
|----|------|------|------------------|-----------|------|
| H1 | PSE-Tok 在长时程超过 per-frame grounding | LIBERO-LONG SR | **45.8 %** | **+5～+10 pp ⇒ 50.8 %～55.8 %** | ⏳ 未验证 |
| H2 | PSE-Tok 在扰动下更鲁棒 | LIBERO-Plus 均值 SR | 待测 | 基线 + 6 pp | ⏳ 未验证 |
| H3a | CRED 比 action-confidence 更早预测失败 | failure-prediction AUROC | ≤ 0.65 | **≥ 0.80** | ⏳ 未验证 |
| H3b | CRED 闭环纠错增益 | LIBERO-LONG SR (CRED on vs off) | — | **+3 pp** on top of H1 ⇒ 53.8 %～58.8 % | ⏳ 未验证 |
| H4 | PSE-Tok 是主导贡献 | 消融下降 | — | ≥ 60 % 总增益归 PSE-Tok | ⏳ 未验证 |

**短时程套件不是卖点**。LIBERO-Spatial / Object / Goal 的目标只是 "不显著掉点"（−≤ 3 pp）。当前 Spatial 70 % → 22 % 不是预期下界。

**OpenVLA-FT Phase-1 ceiling**（自家复现，单 seed × 500 rollouts/suite）：

| Suite | SR | 备注 |
|-------|----|----|
| Spatial | 80.2 % | sanity |
| Object | 80.0 % | sanity |
| Goal | 72.8 % | sanity |
| **LONG** | **45.8 %** | **RQ1 唯一战场** |

---

## 2. 原始实验方案（三件套 + benchmark；源自 `plans/experiment_blueprint.md`）

### 2.1 三个组件 —— 论文 §3 描述的版本

| 组件 | 原始设计 | 当前实现差距 |
|------|---------|--------------|
| **PSE-Tok** | 首 2–4 帧 SAM-2 → 深度提升到 3D → POGS 风格 persistent Gaussian splat → 每步 warp+ID update → 读出 N=8 实体 token（3D pos + 外观 feat + ID） | tiny CNN + MLP；无 SAM-2、无深度、无 splat、无 warp |
| **XTC-Loss** | L_xtc = λ1·‖f_t − f_{t-1} − Δf_pred(a_{t-1})‖ + λ2·contrastive(f_t, augmented) | 数值塌缩到 < 5e-5，无 Δf_pred，无对比负样本 |
| **CRED** | r_t = ‖f_t − (f_{t-1} + Δf_pred(a_{t-1}))‖；连续 K 步 > τ ⇒ freeze + replan | 完全未实测 |

> **绝对禁令**：论文 §3 写 POGS / SAM-2，§5 跑 tiny CNN —— 这是 R1/R4 评审"strictly weaker system"的根因。**要么补实现，要么改 §3**。两条路只能选一条，不能并存。

### 2.2 基线（原始）

OpenVLA-7B、π0、Long-VLA、SeqVLA、Seer、VLA-in-the-Loop + 3 个内部消融（-PSE / -XTC / -CRED）。

### 2.3 数据集（原始）

| Benchmark | 用途 | 状态 |
|-----------|------|------|
| **LIBERO-LONG** | **RQ1 主战场** | 基线已跑（45.8 %），PSSA 未跑 |
| LIBERO-Spatial/Object/Goal | sanity | 基线已跑 |
| LIBERO-Plus / LIBERO-PRO | RQ2 鲁棒性 | 未触及 |
| CALVIN ABC-D | RQ1/RQ3 零样本 | 未触及 |
| VLABench (subset) | secondary | 未触及 |
| Open X-Embodiment mini | PSE-Tok 预训练 warm-up | 未触及 |

### 2.4 统计协议（不可妥协）

- **3 seeds**，mean ± 1 stderr
- LIBERO **50 rollouts / task**（当前未训用 10，远低于规格）
- Wilcoxon signed-rank + Bonferroni（5 基线）
- H3a AUROC 5000-resample bootstrap 95 % CI

---

## 3. 决策门（原始 `experiment_blueprint.md` §9 —— 保留不放松）

1. **SETUP 门**：SAM-2 + GS 在 CALVIN 单场景必须 < 200 ms/帧，否则退回 point-track-only PSE-Tok。
2. **CODING 门**：5-rollout LIBERO-LONG 随机初始化 smoke 通过。
3. **EXECUTION 门（关键）**：LIBERO-LONG 上 PSSA SR < OpenVLA 复现 SR + 2 pp（即 < 47.8 %）⇒ **停止扩展到 CALVIN，回头诊断架构**。

> 当前位置：v2c 训后 Spatial 22 %，连 sanity 都没过。**不允许跳过 Exp-0/1/2 直接上 CALVIN/LONG**——但**也不允许把 22 % 当结论收尾**。

---

## 4. 反跑偏铁律（必须每次接续工作时复读）

1. **目标是原始 H1–H4 正向假设**，不是负面叙事 / 责任划分 / 失败经验报告。
2. **22 % 是 bug，不是 finding**。修，不要写。
3. **§3 与 §5 必须一致**：要么实装 POGS+SAM-2，要么改写 §3 为 CNN prototype。绝不允许"宣传重武器，实现轻武器"。
4. **LIBERO-LONG 是 H1 的唯一战场**。Spatial 数据不能代替 LONG 结论。
5. **3 seeds × 50 rollouts/task** 是 LIBERO 协议下限，单 seed × 10 rollouts 的对比不进表 2。
6. **OpenVLA-native control 73 % @ n=10**（已捕获）是"我们 wrapper + n=10 协议"的真实上限——超过它才算 PSE 真正"良性"。
7. 每次开新实验前必须在 `manifest.json` 里登记 hypothesis ID + 目标值 + 决策门动作，不允许"先跑跑看"。

---

## 5. 当前位置（截至 2026-05-12）

**已捕获的事实**：

- OpenVLA-FT 4-suite Phase-1 baseline：80.2 / 80.0 / 72.8 / 45.8 %（500 roll/task）
- v2c 未训 variant_A：51.0 % on Spatial（100 roll，n=10/task）
- v2c 训后最佳：22.0 % on Spatial（同协议）
- **OpenVLA-native control（无 wrapper）：73.0 % on Spatial @ n=10**（matched protocol，已写入 §5.3）
- 评审 5 人均分 2.0/5（borderline reject），主诉：M1 §3↔§5 不一致；M2 匹配对照；M3 3 seeds + CI
- Plan C frozen-LoRA：**已取消**（用户决定不沿用"责任划分"路径）

**已写好但未训的代码**：

- `experiment/code/pssa/sam2_masker.py`（SAM-2 per-frame mask）
- `model_v2.py` / `dataset.py` SAM-2 接线
- `configs/train.yaml` 的 `use_sam2_masks: false` 默认值 —— **下一步把它翻成 true**

---

## 6. 下一步路线图（按 H1 目标倒推，不是按"修哪个 bug"组织）

### 阶段 A —— 让 PSE 真的成为 PSE（接通 SAM-2，对齐 §3 描述）

| 步骤 | 内容 | 目标 | 决策门 |
|------|------|------|--------|
| A0 | SAM-2 mask 缓存 smoke：单 episode 跑通，验证 mask 在帧间 ID 稳定 | < 200 ms/帧 | 若 ≥ 200 ms → 退回 point-track-only |
| A1 | use_sam2_masks=true 在 Spatial 训 3000 步 × 3 seeds | Spatial SR ≥ 73 %（OpenVLA-native ceiling）| 若 < 60 % → 诊断（不是宣布"训练让模型变差"）|
| A2 | 同 ckpt 上 LIBERO-LONG（n=50/task × 3 seeds）| **LONG SR ≥ 50.8 %（H1 下限）** | **若 < 47.8 % → 停下重新设计，不上 CALVIN** |

### 阶段 B —— 验证 XTC 真的在工作（接通 Δf_pred + 对比损失）

| 步骤 | 内容 | 目标 |
|------|------|------|
| B1 | 实装 Δf_pred(a_{t-1})：动作条件下的实体特征预测残差 | XTC loss 数量级 ≥ 1e-2，不是 1e-5 |
| B2 | 在 A2 同协议上对比 with/without XTC | H4 消融准备 |

### 阶段 C —— 验证 CRED（H3a/H3b）

| 步骤 | 内容 | 目标 |
|------|------|------|
| C1 | 在 LONG rollout 上记录 r_t，标注失败时刻，算 AUROC | **AUROC ≥ 0.80** |
| C2 | 闭环：r_t > τ 触发 freeze + replan，对比 SR | **+3 pp on top of H1** |

### 阶段 D —— 扩展（只有 A2 通过决策门才进）

LIBERO-Plus（H2）→ CALVIN ABC-D 零样本 → Open X warm-up。

---

## 7. 跨 session 接续协议

任何继续这个 session 的 Claude 实例，开局必做：

1. `Read C:\Users\jielu\.nanoresearch\workspace\research\20260509000456c3a8\plans\ACTION_GUIDE.md`（本文件）
2. `Read manifest.json` → 找 `current_stage` 与最新 phase 状态
3. 把当前任务对齐到本文件 §6 的某个步骤 ID（A0/A1/A2/B1/B2/C1/C2/D）；不在表中的任务必须先回答"它服务于哪个 H1–H4 假设"
4. 任何 "把 22 % 写进论文" 或 "证明 XX 不行" 的提议 **拒绝**，回到 §4 铁律第 1、2 条

最后修改：2026-05-12
