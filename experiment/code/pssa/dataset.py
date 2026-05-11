"""LIBERO demo dataset for PSSA training.

LIBERO demos are HDF5 files at $NR_DATA/libero_<suite>/<task>.hdf5 with
structure:

    data/
        demo_0/
            actions       (T, 7)         delta xyz/rpy + gripper
            obs/
                agentview_rgb  (T, H, W, 3)  uint8
                ee_pos         (T, 3)
                ee_quat        (T, 4)
                gripper_states (T, 2)
                ...
            states        (T, state_dim)  full sim state for replay

Each task file has ~50 demos. We sample (episode, window) pairs and yield
batches of {rgb_seq, actions, prompt, masks_init} matching the format
expected by `pssa.model_v2.PSSAVLAv2.training_step`.

Usage:
    ds = LIBEROEpisodeDataset(suite="libero_spatial", root="$NR_DATA")
    loader = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=ds.collate)
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset


SUITES = {
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
    "libero_10": "libero_10",
}


class LIBEROEpisodeDataset(IterableDataset):
    """Yields (rgb_seq, actions, language, masks_init) tuples per episode.

    IterableDataset so we can lazily open HDF5 handles per worker without
    holding all demos in memory. Each iteration samples a uniformly-random
    task from the suite, then a random demo from that task, then a random
    contiguous window of `window_len` frames.

    masks_init is currently a placeholder (all-ones, will be replaced by
    SAM-2 per-frame masks in a later iteration). For phase-2 training we
    only use the first frame's RGB to seed PSE-Tok via a learned
    embedding, not actual segmentation masks.
    """

    def __init__(
        self,
        suite: str = "libero_spatial",
        root: str | None = None,
        window_len: int = 16,
        n_init_frames: int = 4,
        n_entities: int = 8,
        image_hw: tuple[int, int] = (224, 224),
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if suite not in SUITES:
            raise ValueError(f"unknown suite {suite}; choose from {list(SUITES)}")
        self.suite = suite
        self.root = Path(root or os.environ.get("NR_DATA", "/root/autodl-tmp/datasets"))
        self.window_len = window_len
        self.n_init_frames = n_init_frames
        self.n_entities = n_entities
        self.image_hw = image_hw
        self._task_files = sorted((self.root / suite).glob("*.hdf5"))
        if not self._task_files:
            raise FileNotFoundError(
                f"no .hdf5 demos under {self.root / suite}. "
                f"Run download_libero_demos.sh first."
            )
        self._seed = seed

    # ----- helpers --------------------------------------------------------
    def _load_demo(self, h5path: Path, demo_key: str) -> dict:
        """Read one demo from an HDF5 file."""
        with h5py.File(h5path, "r") as f:
            grp = f["data"][demo_key]
            actions = grp["actions"][...].astype(np.float32)         # (T, 7)
            rgb = grp["obs/agentview_rgb"][...].astype(np.uint8)     # (T, H, W, 3)
            language = grp.attrs.get("language", h5path.stem)
        return {"actions": actions, "rgb": rgb, "language": str(language)}

    def _list_demos(self, h5path: Path) -> list[str]:
        with h5py.File(h5path, "r") as f:
            return list(f["data"].keys())

    # ----- iterator -------------------------------------------------------
    def __iter__(self) -> Iterator[dict[str, torch.Tensor | str]]:
        worker = torch.utils.data.get_worker_info()
        # Combine DDP rank + worker id so each (rank, worker) sees a unique
        # random stream — otherwise dual-GPU sees identical batches and the
        # second GPU does redundant work.
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        except Exception:
            rank, world = 0, 1
        worker_id = worker.id if worker is not None else 0
        seed = (self._seed or 0) + rank * 1_000 + worker_id
        rng = random.Random(seed)
        while True:
            task_h5 = rng.choice(self._task_files)
            try:
                demo_keys = self._list_demos(task_h5)
                if not demo_keys:
                    continue
                demo_key = rng.choice(demo_keys)
                demo = self._load_demo(task_h5, demo_key)
            except Exception:
                continue  # skip unreadable demo

            actions = demo["actions"]
            rgb = demo["rgb"]                  # (T, H, W, 3)
            T = rgb.shape[0]
            if T < self.window_len + self.n_init_frames:
                continue

            # Take init frames from the start, then a random window after that
            init_end = self.n_init_frames
            win_start = rng.randint(init_end, T - self.window_len)
            win_end = win_start + self.window_len

            # Use init frames + window (no overlap)
            rgb_init = rgb[:init_end]                              # (T0, H, W, 3)
            rgb_win = rgb[win_start:win_end]                       # (T, H, W, 3)
            actions_win = actions[win_start:win_end]               # (T, 7)

            # Convert to torch (B=1 in iterator; collate stacks)
            rgb_init_t = torch.from_numpy(rgb_init).permute(0, 3, 1, 2).float() / 255.0
            rgb_win_t = torch.from_numpy(rgb_win).permute(0, 3, 1, 2).float() / 255.0
            actions_t = torch.from_numpy(actions_win)

            # Placeholder masks: all-ones over each entity slot.
            # In a later iteration this should be SAM-2 segmentation masks.
            H, W = rgb_init_t.shape[-2:]
            masks_init = torch.ones(self.n_init_frames, self.n_entities, H, W,
                                    dtype=torch.float32)

            yield {
                "rgb_init": rgb_init_t,        # (T0, 3, H, W)
                "rgb_seq": rgb_win_t,          # (T, 3, H, W)
                "actions": actions_t,          # (T, 7)
                "masks_init": masks_init,      # (T0, N, H, W)
                "language": demo["language"],  # str
            }

    # ----- collate --------------------------------------------------------
    @staticmethod
    def collate(batch: list[dict]) -> dict:
        """Stack a list of episode samples into batched tensors."""
        return {
            "rgb_init": torch.stack([b["rgb_init"] for b in batch]),     # (B, T0, 3, H, W)
            "rgb_seq": torch.stack([b["rgb_seq"] for b in batch]),       # (B, T, 3, H, W)
            "actions": torch.stack([b["actions"] for b in batch]),       # (B, T, 7)
            "masks_init": torch.stack([b["masks_init"] for b in batch]), # (B, T0, N, H, W)
            "language": [b["language"] for b in batch],                  # list[str]
        }
