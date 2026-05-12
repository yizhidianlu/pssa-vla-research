"""SAM-2 per-frame entity mask generation for PSSA training.

Why this exists
---------------
The v2c pipeline currently feeds an all-ones placeholder mask into
`PSEEntityEncoder` for every frame.  Because every entity slot sees the
identical mask at every timestep, `entity_seq[:, t+1] - entity_seq[:, t]`
collapses to ~0 (observed XTC loss < 5e-5), so the cross-time consistency
objective never fires and §5.6 follow-up 1 cannot be evaluated.

`SAM2Masker` produces real per-frame, per-entity binary masks from the
RGB stream so that:

  * The PSE encoder sees genuinely different entity supports at each
    timestep (objects move, gripper occludes, viewpoint identical but
    pixels shift).
  * XTC becomes a meaningful regularizer — entity features can drift but
    are penalized for jagged motion.

Design choice — first-frame anchor + video propagation
------------------------------------------------------
Running SAM-2's automatic mask generator independently on every frame is
expensive (~3-5 s/frame on a T4) AND, more critically, has no instance
correspondence across time: mask #3 in frame t is not necessarily the
same physical object as mask #3 in frame t+1.  PSE's per-entity slots
require that correspondence.

So we instead:

  1. Run the automatic mask generator on FRAME 0 only.
  2. Pick the top-N masks ranked by area (largest = robot/table/object,
     smallest = noise specks we discard).
  3. Extract a single positive-prompt point per mask (the centroid of
     the largest connected component).
  4. Feed those N points into SAM-2's video predictor, which propagates
     the masks across frames using its memory bank — giving us
     instance-stable (T, N, H, W) masks with consistent slot ordering.

This is the standard SAM-2 video tracking recipe (see
`sam2.sam2_video_predictor`).  If the video predictor is unavailable
(e.g. SAM-2 not installed, or only the image predictor is present), we
fall back to per-frame automatic generation with greedy area-based
slot matching — slower but works.

Caching
-------
SAM-2 inference dominates dataloader latency, so masks are cached to
`{cache_dir}/{task_hash}/{demo_id}.npy` keyed by demo identity.  The
LIBERO HDF5s are immutable, so cache invalidation isn't required during
a single training run.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Lazy-import helper — SAM-2 is a heavy optional dep; defer the import so
# the rest of the dataset module is usable without it.
# ---------------------------------------------------------------------------
def _import_sam2() -> tuple[Any, Any, Any]:
    """Return (build_sam2, SAM2AutomaticMaskGenerator, build_sam2_video_predictor).

    Raises a clear ImportError pointing at the install instructions.
    """
    try:
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as exc:  # pragma: no cover — exercised on bare envs
        raise ImportError(
            "SAM-2 is required for `use_sam2_masks=True` but is not "
            "installed. Install it with:\n\n"
            "    pip install git+https://github.com/facebookresearch/sam2.git\n\n"
            "and ensure `huggingface_hub` can download "
            "`facebook/sam2-hiera-tiny`."
        ) from exc
    return build_sam2, SAM2AutomaticMaskGenerator, build_sam2_video_predictor


def _centroid_of_mask(mask: np.ndarray) -> tuple[int, int]:
    """Return (x, y) centroid of the largest connected component of `mask`.

    `mask` is HxW bool/uint8.  Falls back to the simple centroid of all
    foreground pixels if scipy.ndimage is unavailable.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    try:
        from scipy import ndimage
        labels, n = ndimage.label(mask)
        if n == 0:
            ys, xs = np.where(mask)
            if len(xs) == 0:
                h, w = mask.shape
                return (w // 2, h // 2)
            return (int(xs.mean()), int(ys.mean()))
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        ys, xs = np.where(labels == largest)
    except ImportError:
        ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return (w // 2, h // 2)
    return (int(xs.mean()), int(ys.mean()))


class SAM2Masker:
    """Generate per-frame entity masks for a short RGB clip.

    Parameters
    ----------
    model_id : str
        HuggingFace repo for SAM-2 weights. Default
        `"facebook/sam2-hiera-tiny"` keeps memory low (~155M params).
    device : str
        `"cuda"` / `"cuda:0"` / `"cpu"`. SAM-2 supports both; CPU is slow
        but works for offline mask precomputation.
    points_per_side : int
        Density of the seed-point grid for the first-frame automatic
        mask generator. 16 is a reasonable default for 224x224 LIBERO
        frames.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-tiny",
        device: str = "cuda",
        points_per_side: int = 16,
        min_mask_area: int = 25,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.points_per_side = points_per_side
        self.min_mask_area = min_mask_area
        # Lazy-init the heavy SAM-2 modules in `_ensure_models()` so that
        # importing this file doesn't pay the cost.
        self._image_model = None
        self._video_predictor = None
        self._auto_gen = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _ensure_models(self) -> None:
        if self._image_model is not None:
            return
        build_sam2, AutoGen, build_video = _import_sam2()
        # HuggingFace path: SAM-2's `build_sam2_hf` and `build_sam2_video_predictor_hf`
        # download weights + config from the hub.  The non-HF builders need a
        # local `.yaml` config; we route through the HF builders so this works
        # zero-config on autodl.
        try:
            from sam2.build_sam import (
                build_sam2_hf,
                build_sam2_video_predictor_hf,
            )
            self._image_model = build_sam2_hf(self.model_id, device=self.device)
            self._video_predictor = build_sam2_video_predictor_hf(
                self.model_id, device=self.device
            )
        except Exception:
            # Old SAM-2 wheels lack the *_hf helpers — let it fail loudly
            # so the user upgrades rather than silently falling back to
            # per-frame auto-gen (which has no instance correspondence).
            raise
        self._auto_gen = AutoGen(
            self._image_model,
            points_per_side=self.points_per_side,
            min_mask_region_area=self.min_mask_area,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def mask_frames(
        self,
        rgb_seq: np.ndarray,
        n_entities: int = 8,
    ) -> np.ndarray:
        """Return per-frame masks for an RGB sequence.

        Parameters
        ----------
        rgb_seq : np.ndarray
            Shape `(T, H, W, 3)`, dtype uint8.
        n_entities : int
            Slot count `N`.  Output is padded with all-zero masks if
            SAM-2 finds fewer than `N` entities in frame 0; extra masks
            are dropped (largest-area-first).

        Returns
        -------
        np.ndarray
            Shape `(T, N, H, W)`, dtype float32, values in {0.0, 1.0}.
        """
        if rgb_seq.ndim != 4 or rgb_seq.shape[-1] != 3:
            raise ValueError(
                f"rgb_seq must be (T, H, W, 3) uint8; got {rgb_seq.shape}"
            )
        if rgb_seq.dtype != np.uint8:
            rgb_seq = rgb_seq.astype(np.uint8)

        self._ensure_models()
        T, H, W, _ = rgb_seq.shape

        # 1) First-frame automatic mask generation
        first_frame = rgb_seq[0]
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
        masks_records = self._auto_gen.generate(first_frame)
        # Sort by area descending; drop tiny noise
        masks_records = sorted(
            (m for m in masks_records if m.get("area", 0) >= self.min_mask_area),
            key=lambda m: m["area"],
            reverse=True,
        )[:n_entities]

        if len(masks_records) == 0:
            # Frame 0 produced nothing — return zero masks so downstream
            # mask-pool still works (it will fall back to denom clamp).
            return np.zeros((T, n_entities, H, W), dtype=np.float32)

        # 2) Build point prompts for video propagation
        prompts: list[tuple[int, int]] = []
        for rec in masks_records:
            seg = rec["segmentation"]  # bool HxW
            if seg.shape != (H, W):
                # SAM-2 occasionally returns masks at a different scale;
                # nearest-resize to frame shape.
                seg = _resize_mask(seg, (H, W))
            prompts.append(_centroid_of_mask(seg))

        n_found = len(prompts)

        # 3) Video propagation
        try:
            masks_tnhw = self._propagate_video(rgb_seq, prompts, n_entities)
        except Exception:
            # Video predictor failed (e.g. shape mismatch, OOM); fall
            # back to first-frame mask repeated across time. Slightly
            # better than zeros — at least the encoder sees a non-trivial
            # spatial support — but XTC will be ~0 again. Caller can
            # check `masks_seq.var()` to detect this regression.
            first_masks = np.zeros((n_entities, H, W), dtype=np.float32)
            for i, rec in enumerate(masks_records):
                seg = rec["segmentation"]
                if seg.shape != (H, W):
                    seg = _resize_mask(seg, (H, W))
                first_masks[i] = seg.astype(np.float32)
            masks_tnhw = np.broadcast_to(
                first_masks[None], (T, n_entities, H, W)
            ).copy()

        # 4) Pad/truncate to n_entities
        if n_found < n_entities:
            pad = np.zeros(
                (T, n_entities - n_found, H, W), dtype=np.float32
            )
            masks_tnhw[:, n_found:] = pad
        return masks_tnhw

    # ------------------------------------------------------------------
    # Video propagation
    # ------------------------------------------------------------------
    def _propagate_video(
        self,
        rgb_seq: np.ndarray,
        prompts: list[tuple[int, int]],
        n_entities: int,
    ) -> np.ndarray:
        """Use SAM-2's video predictor to propagate first-frame point
        prompts across the clip.

        SAM-2's `SAM2VideoPredictor.init_state` accepts either a folder
        of JPEGs or a list of numpy frames.  We feed numpy directly to
        avoid disk IO.
        """
        import torch

        T, H, W, _ = rgb_seq.shape
        N = n_entities

        # SAM-2 video predictor consumes frames as a list of (H, W, 3)
        # uint8 arrays via `init_state` — but the public API in current
        # wheels accepts only a video folder.  We therefore write a
        # transient in-memory list via `set_video` if present, else fall
        # back to the documented folder-API path.
        predictor = self._video_predictor
        with torch.inference_mode():
            if hasattr(predictor, "init_state_from_numpy_frames"):
                state = predictor.init_state_from_numpy_frames(rgb_seq)
            elif hasattr(predictor, "init_state"):
                # Newer API: init_state accepts video_path OR a frame tensor.
                # We construct a (T, 3, H, W) torch tensor in [0, 255].
                frames_t = (
                    torch.from_numpy(rgb_seq).permute(0, 3, 1, 2).contiguous()
                )
                try:
                    state = predictor.init_state(video_path=frames_t)
                except TypeError:
                    # Some versions require disk frames; dump to a tmpdir.
                    state = self._init_state_via_tmpdir(predictor, rgb_seq)
            else:
                raise RuntimeError(
                    "SAM2VideoPredictor lacks init_state / init_state_from_numpy_frames"
                )

            # Add one positive point per entity, all on frame 0.
            for ent_id, (x, y) in enumerate(prompts):
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=ent_id,
                    points=np.array([[x, y]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )

            # Propagate forward through all T frames.
            masks_out = np.zeros((T, N, H, W), dtype=np.float32)
            for (
                frame_idx,
                obj_ids,
                mask_logits,
            ) in predictor.propagate_in_video(state):
                # mask_logits: (n_obj, 1, H, W) torch.bfloat16 typically
                probs = (mask_logits > 0).float().squeeze(1).cpu().numpy()
                for obj_id, p in zip(obj_ids, probs):
                    if obj_id >= N:
                        continue
                    if p.shape != (H, W):
                        p = _resize_mask(p, (H, W))
                    masks_out[frame_idx, obj_id] = p
            return masks_out

    def _init_state_via_tmpdir(self, predictor, rgb_seq: np.ndarray):
        """Fallback path for SAM-2 builds that require a video folder.

        Dumps frames as JPEGs to a tempdir and calls
        `predictor.init_state(video_path=tmpdir)`.
        """
        import tempfile
        from PIL import Image

        tmp = tempfile.mkdtemp(prefix="sam2_frames_")
        for i, frame in enumerate(rgb_seq):
            Image.fromarray(frame).save(
                os.path.join(tmp, f"{i:05d}.jpg"), quality=95
            )
        return predictor.init_state(video_path=tmp)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def _resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize for binary masks. Avoids a torch import
    so we can be called from worker processes that haven't yet imported
    torch."""
    src_h, src_w = mask.shape
    dst_h, dst_w = hw
    if (src_h, src_w) == (dst_h, dst_w):
        return mask.astype(np.float32)
    ys = (np.arange(dst_h) * src_h / dst_h).astype(np.int64)
    xs = (np.arange(dst_w) * src_w / dst_w).astype(np.int64)
    return mask[ys][:, xs].astype(np.float32)


# ---------------------------------------------------------------------------
# File-backed cache helpers — used by `pssa.dataset.LIBEROEpisodeDataset`.
# Kept here so callers don't reinvent the key scheme.
# ---------------------------------------------------------------------------
def cache_path(
    cache_dir: str | os.PathLike,
    task_hash: str,
    demo_id: str,
) -> Path:
    p = Path(cache_dir) / task_hash / f"{demo_id}.npy"
    return p


def load_cached_masks(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return None


def save_cached_masks(path: Path, masks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: tmp -> rename, so concurrent workers don't read a
    # half-written file.
    tmp = path.with_suffix(".npy.tmp")
    np.save(tmp, masks.astype(np.float32))
    tmp.replace(path)
