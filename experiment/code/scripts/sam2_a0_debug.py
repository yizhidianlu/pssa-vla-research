"""A0 诊断：variance=0 是 mask 全 0 还是 mask 不变？

跑单 episode SAM-2 propagation, 打印:
- 每个 entity 的 fill rate
- 第一帧 vs 末帧的 IoU
- 整段视频 propagation 是否真正调用 (frame_idx 步进?)
- mask logits 原始范围 (是否阈值太严)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pssa.dataset import LIBEROEpisodeDataset


def main() -> int:
    ds = LIBEROEpisodeDataset(
        suite="libero_spatial",
        window_len=16,
        n_init_frames=4,
        n_entities=8,
        use_sam2_masks=True,
        sam2_cache_dir=None,
        sam2_device="cuda",
        seed=0,
    )
    sample = next(iter(ds))
    masks = sample["masks_init"]  # (T0+T, N, H, W) torch.float32
    T, N, H, W = masks.shape
    print(f"masks shape: ({T}, {N}, {H}, {W})  dtype={masks.dtype}")
    print(f"global min/max/mean: {masks.min().item():.4f} / {masks.max().item():.4f} / {masks.mean().item():.6f}")

    print("\n--- per-entity fill rate (mean across all frames) ---")
    for n in range(N):
        fill = masks[:, n].mean().item()
        nonzero_frames = (masks[:, n].sum(dim=(-1, -2)) > 0).sum().item()
        print(f"  entity {n}: fill={fill:.4f}  nonzero_frames={nonzero_frames}/{T}")

    print("\n--- per-frame fill rate (mean across all entities) ---")
    for t in range(T):
        fill = masks[t].mean().item()
        print(f"  frame {t}: fill={fill:.4f}")

    print("\n--- temporal change (IoU between frame 0 and frame t) ---")
    f0 = masks[0]
    for t in [1, 5, 10, T-1]:
        ft = masks[t]
        inter = (f0 * ft).sum().item()
        union = ((f0 + ft) > 0).float().sum().item()
        iou = inter / max(union, 1)
        print(f"  frame 0 vs frame {t}: IoU={iou:.4f}")

    print("\n--- raw RGB sanity check ---")
    rgb_init = sample["rgb_init"]
    rgb_seq = sample["rgb_seq"]
    print(f"rgb_init shape: {tuple(rgb_init.shape)}  min/max: {rgb_init.min().item():.4f} / {rgb_init.max().item():.4f}")
    print(f"rgb_seq  shape: {tuple(rgb_seq.shape)}   min/max: {rgb_seq.min().item():.4f} / {rgb_seq.max().item():.4f}")
    # Across-frame RGB variation
    rgb_all = torch.cat([rgb_init, rgb_seq], dim=0)
    per_pixel_var = rgb_all.var(dim=0).mean().item()
    print(f"RGB across-time variance: {per_pixel_var:.6f}  (should be > 0 — scene moves)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
