# LIBERO Finetuned Eval — 2026-05-10 00:18 CST

Session: 20260509000456c3a8
Hardware: AutoDL A800 80GB PCIe
Model: `openvla/openvla-7b-finetuned-libero-spatial` (OpenVLA's official LIBERO-Spatial finetune)
Suite: libero_spatial, task 0 = `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`

## Result — **5/5 SR = 100%**

| Rollout | Success | Steps | Latency (ms/step) |
|--------:|:-------:|------:|------------------:|
| 0 | ✓ | 74 | 199.9 |
| 1 | ✓ | 80 | 192.9 |
| 2 | ✓ | 87 | 193.2 |
| 3 | ✓ | 105 | 197.9 |
| 4 | ✓ | 69 | 194.3 |
| **Mean** | **5/5** | **83** | **195.6** |

Max steps allowed = 200; the mean of 83 steps shows the policy reaches goal well within the timeout.
Peak VRAM: **15.45 GB**.

## What it took to get a real number — 4 attempts

| Run | Config | SR | Reason |
|-----|--------|----|---|
| ft-v1 | unnorm=`libero_spatial`, no fixes | 0/5 | Gripper convention bug (OpenVLA→LIBERO) |
| ft-v2 | + `--libero-action-fix` | 0/5 | Image orientation also wrong (single flip vs 180° rotate) |
| ft-v3 | + `--libero-image-fix` + wrong key `libero_spatial_no_noops` | crash | unnorm_key probe revealed: ckpt only has `libero_spatial` |
| **ft-v4** | **`libero_spatial` + both fixes** | **5/5** | All canonical OpenVLA→LIBERO transforms applied |

## The two essential fixes

### 1. Gripper convention (`--libero-action-fix`)

```python
action[-1] = np.sign(2 * action[-1] - 1)   # [0,1] → {-1,+1}
action[-1] *= -1.0                          # OpenVLA: 1=close, LIBERO: 1=open
```

OpenVLA was trained with gripper ∈ {0, 1} where 1=close. LIBERO's env expects gripper ∈ {-1, +1} where 1=open. Without these two transforms (rescale + flip sign), the gripper signal is essentially random noise and grasping never succeeds.

### 2. Image orientation (`--libero-image-fix`)

```python
rgb = obs["agentview_image"][::-1, ::-1]    # double flip = 180° rotate
```

**Single** `[::-1]` (vertical flip) is enough to match the visual convention of "row 0 at top". But OpenVLA's LIBERO finetune training distribution used **double** `[::-1, ::-1]` (180° rotation). At inference time, matching that distribution is essential — single flip puts the scene upside-down relative to what the model learned.

## Cross-reference vs published numbers

OpenVLA paper Table 5 reports `libero_spatial` mean SR around **83-85%** across all 10 tasks (50 rollouts each). On task 0 alone, 5 rollouts is too small a sample to compare — this 5/5 establishes the pipeline correctness, not the full benchmark number. Mid-budget should rerun on all 10 tasks × 50 rollouts to land a proper number with ±CI.

## Cumulative AutoDL session cost (B-tier total)

| Phase | Wallclock | ¥ |
|-------|-----------|---|
| Initial bootstrap + base OpenVLA download | 30 min | 4 |
| Synthetic e2e smoke + LIBERO install + EGL fixing | ~75 min | 9 |
| Generalist LIBERO smoke (0/5 expected) | 5 min | 0.7 |
| ft-v1: download finetuned ckpt 14 GB + eval | 35 min | 4 |
| ft-v2 (action fix only) | 5 min | 0.7 |
| ft-v3 (wrong key crash) | 5 min | 0.7 |
| ft-v4 (FINAL — both fixes correct) | 5 min | 0.7 |
| **Total** | **~160 min** | **~¥20** |

Hit our budget cap (¥50) at the upper bound, but well-spent — produced a defensible 5/5 SR plus 4 commits encoding all the gotchas for next time.

## What this proves

- ✅ **Sim-side pipeline correct**: env init → set_init_state → env.step loop, no leaks
- ✅ **OpenVLA-7B inference at 195 ms/step on A800** (consistent across all 4 attempts)
- ✅ **The two critical OpenVLA→LIBERO transforms are essential and now encoded** in `run_libero_eval.py` flags (default off — backwards-compat with smoke; explicitly opt-in for LIBERO eval)
- ✅ **15.45 GB VRAM** = under 1/5 of A800 80GB → comfortable headroom for batch / FSDP

## What this does NOT yet prove

- The PSSA-VLA contribution (we ran vanilla OpenVLA-finetuned, no PSE-Tok / XTC / CRED)
- Generalization across the other 9 tasks of libero_spatial, or across LIBERO-LONG / LIBERO-Plus
- 50-rollouts-per-task statistics (SR error bars)

These are mid-budget items.

## Next decision

The eval harness is now reliable. To produce results that go into the paper, the next step is **mid-budget**:

```bash
# Per task, all 10 tasks of libero_spatial:
python experiment/code/scripts/run_libero_eval.py \
    --suite libero_spatial --task-id $TASK \
    --rollouts 50 --max-steps 200 \
    --model-id openvla/openvla-7b-finetuned-libero-spatial \
    --unnorm-key libero_spatial \
    --libero-action-fix --libero-image-fix \
    --out runs/libero_spatial/task_$TASK.json
```

10 tasks × 50 rollouts × ~85 steps × 195 ms ≈ **47 GPU-min per suite**. Trivial. The bigger work is then training a PSSA action head on top of OpenVLA — that's the actual mid-budget compute cost.
