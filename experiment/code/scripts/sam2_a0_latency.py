"""A0 决策门：SAM-2 propagation 延迟探针。

action_guide §6 阶段 A0 的硬门：单 episode 跑通 + 帧间 mask ID 稳定 + < 200 ms/帧。
本脚本在远端 GPU 上跑一次，给出 ms/frame 与 across-time variance，对照阈值通过/退回。

阈值（来自 experiment_blueprint.md §9 SETUP gate）：
    ms/frame > 200      ⇒ FAIL — 退回 point-track-only PSE-Tok 变体
    across_time_var ≤ 1e-5 ⇒ FAIL — mask 没在帧间变化，XTC 不会 fire

用法（远端 GPU）：

    cd /root/autodl-tmp/pssa-vla/experiment/code
    NR_DATA=/root/autodl-tmp/datasets python -m scripts.sam2_a0_latency

预期输出：

    [warmup] 1 clip done
    [measure] clip 1: 1.85 s for 20 frames -> 92.5 ms/frame
    [measure] clip 2: 1.42 s for 20 frames -> 71.0 ms/frame
    [measure] clip 3: 1.39 s for 20 frames -> 69.5 ms/frame
    median ms/frame: 71.0  (gate: < 200)
    across-time variance: 0.0271  (gate: > 1e-5)
    A0 GATE PASS
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pssa.dataset import LIBEROEpisodeDataset

# ---- 阈值 -----------------------------------------------------------------
GATE_MS_PER_FRAME = 200.0
GATE_VARIANCE = 1e-5
N_WARMUP = 1
N_MEASURE = 3
WINDOW_LEN = 16
N_INIT = 4


def main() -> int:
    ds = LIBEROEpisodeDataset(
        suite="libero_spatial",
        window_len=WINDOW_LEN,
        n_init_frames=N_INIT,
        n_entities=8,
        use_sam2_masks=True,
        sam2_cache_dir=None,   # 关 cache，否则第二次直接 0 ms
        sam2_device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
    )
    it = iter(ds)

    # warmup（模型加载 + JIT 编译 + HF 下载第一次都很慢）
    for _ in range(N_WARMUP):
        _ = next(it)
    print(f"[warmup] {N_WARMUP} clip done", flush=True)

    timings_ms_per_frame: list[float] = []
    variances: list[float] = []

    for k in range(N_MEASURE):
        t0 = time.perf_counter()
        sample = next(it)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        masks_init = sample["masks_init"]   # (T0+T, N, H, W)
        n_frames = masks_init.shape[0]
        ms_per_frame = dt * 1000.0 / n_frames
        timings_ms_per_frame.append(ms_per_frame)

        per_pixel_var = masks_init.var(dim=0)
        variances.append(per_pixel_var.mean().item())

        print(
            f"[measure] clip {k+1}: {dt:.2f} s for {n_frames} frames "
            f"-> {ms_per_frame:.1f} ms/frame",
            flush=True,
        )

    med_ms = median(timings_ms_per_frame)
    med_var = median(variances)

    print()
    print(f"median ms/frame: {med_ms:.1f}  (gate: < {GATE_MS_PER_FRAME})")
    print(f"across-time variance: {med_var:.4f}  (gate: > {GATE_VARIANCE})")

    pass_latency = med_ms < GATE_MS_PER_FRAME
    pass_variance = med_var > GATE_VARIANCE

    if pass_latency and pass_variance:
        print("A0 GATE PASS")
        return 0

    print("A0 GATE FAIL:")
    if not pass_latency:
        print(f"  - latency {med_ms:.1f} ms/frame >= {GATE_MS_PER_FRAME}")
        print("    → 退回 point-track-only PSE-Tok 变体（blueprint §9）")
    if not pass_variance:
        print(f"  - variance {med_var:.2e} <= {GATE_VARIANCE}")
        print("    → SAM-2 propagation 没产生时变 mask, XTC 仍会塌缩")
    return 1


if __name__ == "__main__":
    sys.exit(main())
