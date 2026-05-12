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

import hashlib
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
    """Yields (rgb_seq, actions, language, masks_init[, masks_seq]) tuples.

    IterableDataset so we can lazily open HDF5 handles per worker without
    holding all demos in memory. Each iteration samples a uniformly-random
    task from the suite, then a random demo from that task, then a random
    contiguous window of `window_len` frames.

    Mask sources
    ------------
    With `use_sam2_masks=False` (default), `masks_init` is the legacy
    all-ones placeholder of shape `(T0, N, H, W)` and no `masks_seq`
    field is emitted — preserving prior behavior.

    With `use_sam2_masks=True`, a per-worker `SAM2Masker` produces real
    per-frame entity masks. We then yield:
        masks_init : (T0 + T, N, H, W)   covers both init frames AND the
                                          training window so the prefix
                                          encoder can use init-frame
                                          masks and the per-step XTC
                                          encoder can use window masks.
        masks_seq  : (T, N, H, W)        the window slice, broken out
                                          for convenience.

    Masks are cached at `{sam2_cache_dir}/{task_hash}/{demo_id}.npy`
    keyed by (task_path, demo_key, n_init_frames + window length, H, W,
    n_entities) so a re-run with the same hyperparameters skips SAM-2.
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
        use_sam2_masks: bool = False,
        sam2_cache_dir: str | None = None,
        sam2_model_id: str = "facebook/sam2-hiera-tiny",
        sam2_device: str = "cuda",
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
        # ----- SAM-2 mask config -------------------------------------
        self.use_sam2_masks = bool(use_sam2_masks)
        self.sam2_cache_dir = (
            Path(sam2_cache_dir) if sam2_cache_dir is not None else None
        )
        self.sam2_model_id = sam2_model_id
        self.sam2_device = sam2_device
        # Per-worker lazy slot for the SAM2Masker; populated on first
        # __iter__ call inside the worker process.
        self._sam2: object | None = None

    # ----- helpers --------------------------------------------------------
    def _load_demo(self, h5path: Path, demo_key: str) -> dict:
        """Read one demo from an HDF5 file.

        Action conventions (LIBERO-spatial OpenVLA-FT):
          - dims 0..5 (delta xyz/rpy): masked-True. q01/q99 stretch applied
            at INFERENCE only (predict_action). Training uses raw clipped.
          - dim 6 (gripper): masked-False. Demos {-1, +1} bimodal already
            in OpenVLA's normalized range — pass through.

        Image orientation fix: LIBERO env's `obs["agentview_image"]` is
        upside-down (mujoco convention); Phase-1's `--libero-image-fix`
        applies `rgb[::-1, ::-1]` to make it match OpenVLA's training
        distribution. Demos in HDF5 are stored in the same raw orientation
        as env obs, so we must apply the same rotation here for training.
        Without this, training distribution differs from eval distribution.
        """
        with h5py.File(h5path, "r") as f:
            grp = f["data"][demo_key]
            actions = grp["actions"][...].astype(np.float32)         # (T, 7)
            rgb = grp["obs/agentview_rgb"][...].astype(np.uint8)     # (T, H, W, 3)
            language = grp.attrs.get("language", h5path.stem)
        # 180° rotate (vertical + horizontal flip) to match Phase-1 eval distribution
        rgb = rgb[:, ::-1, ::-1, :].copy()
        return {"actions": actions, "rgb": rgb, "language": str(language)}

    def _list_demos(self, h5path: Path) -> list[str]:
        with h5py.File(h5path, "r") as f:
            return list(f["data"].keys())

    # ----- SAM-2 lazy init ------------------------------------------------
    def _ensure_sam2(self) -> object:
        """Construct a per-worker SAM2Masker on first use.

        Building SAM-2 in `__init__` would pin GPU memory for every
        worker process and import torch eagerly in the parent — both
        undesirable.  This lazy path runs once per worker.
        """
        if self._sam2 is not None:
            return self._sam2
        # Import here so that pure CPU smoke tests can import this file
        # without sam2 installed.
        from pssa.sam2_masker import SAM2Masker
        self._sam2 = SAM2Masker(
            model_id=self.sam2_model_id,
            device=self.sam2_device,
        )
        return self._sam2

    def _sam2_cache_key(self, task_h5: Path, demo_key: str, total_frames: int) -> tuple[str, str]:
        """Return (task_hash, demo_id) used to locate the cache file.

        We include image_hw, n_entities, and frame count in the demo_id
        so different mask configurations don't collide.
        """
        task_hash = hashlib.sha1(
            f"{task_h5.resolve()}|{self.suite}".encode()
        ).hexdigest()[:16]
        H, W = self.image_hw
        demo_id = (
            f"{demo_key}_T{total_frames}_N{self.n_entities}_H{H}_W{W}_"
            f"{self.sam2_model_id.replace('/', '_')}"
        )
        return task_hash, demo_id

    def _get_sam2_masks(
        self,
        task_h5: Path,
        demo_key: str,
        rgb_clip: np.ndarray,
    ) -> np.ndarray:
        """Return (T_total, N, H, W) float32 masks, hitting cache when
        possible.  `rgb_clip` is the concatenation of init frames +
        training window in HxWx3 uint8 form.
        """
        from pssa.sam2_masker import cache_path, load_cached_masks, save_cached_masks
        T_total = rgb_clip.shape[0]
        if self.sam2_cache_dir is not None:
            task_hash, demo_id = self._sam2_cache_key(task_h5, demo_key, T_total)
            cpath = cache_path(self.sam2_cache_dir, task_hash, demo_id)
            cached = load_cached_masks(cpath)
            if cached is not None and cached.shape == (
                T_total,
                self.n_entities,
                rgb_clip.shape[1],
                rgb_clip.shape[2],
            ):
                return cached
        else:
            cpath = None

        masker = self._ensure_sam2()
        masks = masker.mask_frames(rgb_clip, n_entities=self.n_entities)
        if cpath is not None:
            try:
                save_cached_masks(cpath, masks)
            except OSError:
                # Cache write failed (full disk, perm error) — non-fatal.
                pass
        return masks

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
            H, W = rgb_init_t.shape[-2:]

            sample: dict[str, torch.Tensor | str] = {
                "rgb_init": rgb_init_t,        # (T0, 3, H, W)
                "rgb_seq": rgb_win_t,          # (T, 3, H, W)
                "actions": actions_t,          # (T, 7)
                "language": demo["language"],  # str
            }

            if self.use_sam2_masks:
                # Stack init + window in temporal order — SAM-2 needs a
                # contiguous video for memory-bank propagation. Note that
                # init_end < win_start in general, so frames between them
                # are NOT included in the propagation. For our 4-init /
                # 16-window default with init_end=4 and win_start ≥ 4,
                # if win_start == init_end the stitched clip is also
                # temporally contiguous; otherwise there's a small jump
                # at the boundary which SAM-2's memory bank tolerates in
                # practice. (LIBERO demos are typically 100-200 frames so
                # the jump is bounded.)
                rgb_clip = np.concatenate([rgb_init, rgb_win], axis=0)
                masks_full = self._get_sam2_masks(task_h5, demo_key, rgb_clip)
                # masks_full: (T0 + T, N, H, W)
                masks_init_t = torch.from_numpy(masks_full).float()
                masks_seq_t = torch.from_numpy(
                    masks_full[init_end : init_end + self.window_len]
                ).float()
                sample["masks_init"] = masks_init_t            # (T0+T, N, H, W)
                sample["masks_seq"] = masks_seq_t              # (T, N, H, W)
            else:
                # Placeholder masks: all-ones over each entity slot.
                masks_init = torch.ones(
                    self.n_init_frames, self.n_entities, H, W, dtype=torch.float32
                )
                sample["masks_init"] = masks_init              # (T0, N, H, W)

            yield sample

    # ----- collate --------------------------------------------------------
    @staticmethod
    def collate(batch: list[dict]) -> dict:
        """Stack a list of episode samples into batched tensors.

        Handles both legacy (no `masks_seq`) and SAM-2 (with `masks_seq`)
        sample dicts.  We branch on the presence of the `masks_seq` key
        in the first sample — within a single dataset run every sample
        has the same schema.
        """
        out = {
            "rgb_init": torch.stack([b["rgb_init"] for b in batch]),
            "rgb_seq": torch.stack([b["rgb_seq"] for b in batch]),
            "actions": torch.stack([b["actions"] for b in batch]),
            "masks_init": torch.stack([b["masks_init"] for b in batch]),
            "language": [b["language"] for b in batch],
        }
        if "masks_seq" in batch[0]:
            out["masks_seq"] = torch.stack([b["masks_seq"] for b in batch])
        return out
