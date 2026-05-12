"""Smoke test for SAM-2 per-frame mask integration.

Verifies that `LIBEROEpisodeDataset(use_sam2_masks=True)` produces
real per-frame entity masks (not the all-ones placeholder) and that
the masks DO vary across time — which is the necessary condition for
the XTC loss in §5.6 follow-up 1 to fire operationally.

This script does NOT exercise training: it only pulls a single demo
through the dataset pipeline + SAM-2 once and inspects the resulting
tensors.

Prerequisites (install separately on the GPU box):

    pip install git+https://github.com/facebookresearch/sam2.git
    # huggingface_hub must be able to fetch `facebook/sam2-hiera-tiny`

Run command (on a GPU box with LIBERO HDF5 demos at $NR_DATA):

    cd experiment/code
    NR_DATA=/root/autodl-tmp/datasets python -m scripts.test_sam2_dataset

Expected output:

    masks_seq shape: (16, 8, 256, 256)
    masks_seq mean fill rate: 0.1842
    masks_seq across-time variance: 0.0271   # >> 0 — XTC will fire
    OK
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pssa.dataset import LIBEROEpisodeDataset


def main() -> None:
    ds = LIBEROEpisodeDataset(
        suite="libero_spatial",
        window_len=16,
        n_init_frames=4,
        n_entities=8,
        use_sam2_masks=True,
        sam2_cache_dir="/tmp/sam2_smoke_cache",
        sam2_device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
    )
    loader = DataLoader(
        ds, batch_size=1, num_workers=0, collate_fn=LIBEROEpisodeDataset.collate
    )

    batch = next(iter(loader))

    assert "masks_seq" in batch, (
        "masks_seq missing from batch — dataset did not run SAM-2 path"
    )

    masks_seq: torch.Tensor = batch["masks_seq"]  # (B, T, N, H, W)
    print(f"masks_seq shape: {tuple(masks_seq.shape)}")

    B, T, N, H, W = masks_seq.shape
    assert B == 1
    assert T == 16, f"expected T=16, got {T}"
    assert N == 8, f"expected N=8, got {N}"
    assert masks_seq.dtype == torch.float32

    fill = masks_seq.mean().item()
    print(f"masks_seq mean fill rate: {fill:.4f}")
    assert 0.0 < fill < 1.0, (
        f"degenerate fill rate {fill}; SAM-2 likely failed or returned constant"
    )

    # The critical check: variance ACROSS the time axis must be > 0.
    # If SAM-2 propagation worked, the mask of each entity will shift
    # frame-to-frame as the scene moves.
    per_pixel_var = masks_seq[0].var(dim=0)   # (N, H, W)
    across_time_var = per_pixel_var.mean().item()
    print(f"masks_seq across-time variance: {across_time_var:.4f}")
    assert across_time_var > 1e-5, (
        f"masks do NOT vary across time (var={across_time_var}); "
        f"XTC will collapse to ~0 like the v2c baseline"
    )

    # Also verify masks_init has the extended shape (T0 + T, N, H, W).
    masks_init: torch.Tensor = batch["masks_init"]
    expected_total = ds.n_init_frames + ds.window_len
    assert masks_init.shape[1] == expected_total, (
        f"masks_init has {masks_init.shape[1]} frames; "
        f"expected {expected_total} (n_init_frames + window_len)"
    )

    print("OK")


if __name__ == "__main__":
    main()
