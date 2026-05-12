"""One-shot SAM-2 mask precompute for a LIBERO suite.

Why this exists
---------------
A0 measured SAM-2 propagation cold latency at ~377 ms/frame on the
A800. Training step needs (T0 + window_len) = 20 frames → ~7.5 s per
batch, GPU-idle. With 3000 steps × bs=2 that's ~12 hours of data prep
versus ~3 hr forward/backward — data loader becomes the bottleneck.

Fix: precompute SAM-2 propagation ONCE per demo (full demo length),
cache (T_demo, N, H, W) float32 mask tensor to disk, training reads
the slice. Per-demo cost: T_demo * 377 ms ≈ 60 s; 500 demos / 2 GPUs ≈
2.5 hr one-time cost. Subsequent training runs (any window position)
read from cache near-zero.

Output schema
-------------
For each demo: {cache_dir}/{task_hash}/{demo_id}_full.npy where:
  task_hash = sha1(task_h5_path | suite)[:16]
  demo_id   = "{demo_key}_N{n_entities}_H{H}_W{W}_full"
  contents  = (T_demo, n_entities, H, W) float32 in {0.0, 1.0}

Run command (dual-GPU parallel, split by demo modulo):

  # GPU 0
  CUDA_VISIBLE_DEVICES=0 \
      NR_DATA=/root/autodl-tmp/datasets \
      HF_ENDPOINT=https://hf-mirror.com \
      python -m scripts.precompute_sam2_masks \
          --suite libero_spatial \
          --cache-dir /root/autodl-tmp/sam2_cache \
          --rank 0 --world 2

  # GPU 1  (different rank)
  CUDA_VISIBLE_DEVICES=1 ... --rank 1 --world 2
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _task_hash(task_h5: Path, suite: str) -> str:
    return hashlib.sha1(f"{task_h5.resolve()}|{suite}".encode()).hexdigest()[:16]


def _cache_path(cache_dir: Path, suite: str, task_h5: Path, demo_key: str,
                n_entities: int, H: int, W: int) -> Path:
    th = _task_hash(task_h5, suite)
    fname = f"{demo_key}_N{n_entities}_H{H}_W{W}_full.npy"
    return cache_dir / th / fname


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--data-root", default=None,
                        help="defaults to $NR_DATA")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--n-entities", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0,
                        help="this worker's rank (for dual-GPU split)")
    parser.add_argument("--world", type=int, default=1,
                        help="total number of workers")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="facebook/sam2-hiera-tiny")
    parser.add_argument("--limit", type=int, default=None,
                        help="optional cap on (task, demo) pairs for smoke runs")
    args = parser.parse_args()

    import os
    data_root = Path(args.data_root or os.environ.get(
        "NR_DATA", "/root/autodl-tmp/datasets"))
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    suite_dir = data_root / args.suite
    task_files = sorted(suite_dir.glob("*.hdf5"))
    if not task_files:
        print(f"ERROR: no .hdf5 under {suite_dir}", file=sys.stderr)
        return 1
    print(f"[rank {args.rank}/{args.world}] suite={args.suite} "
          f"tasks={len(task_files)} cache={cache_root}")

    # Lazy-init SAM2Masker to avoid import-time CUDA setup if not needed.
    from pssa.sam2_masker import SAM2Masker
    masker = SAM2Masker(model_id=args.model_id, device=args.device)

    done = 0
    skipped = 0
    t0 = time.perf_counter()

    for ti, task_h5 in enumerate(task_files):
        with h5py.File(task_h5, "r") as f:
            demo_keys = list(f["data"].keys())
        # Round-robin demos across workers: (task_idx, demo_idx) → rank
        # via flat index mod world.
        for di, demo_key in enumerate(demo_keys):
            flat = ti * 1000 + di  # rough flat index
            if flat % args.world != args.rank:
                continue

            with h5py.File(task_h5, "r") as f:
                rgb = f["data"][demo_key]["obs/agentview_rgb"][...].astype(np.uint8)
            # Match dataset's 180° rotation
            rgb = rgb[:, ::-1, ::-1, :].copy()
            T, H, W, _ = rgb.shape

            cpath = _cache_path(cache_root, args.suite, task_h5,
                                demo_key, args.n_entities, H, W)
            if cpath.is_file():
                skipped += 1
                continue
            cpath.parent.mkdir(parents=True, exist_ok=True)

            ts = time.perf_counter()
            try:
                masks = masker.mask_frames(rgb, n_entities=args.n_entities)
            except Exception as exc:
                print(f"[rank {args.rank}] FAILED {task_h5.stem}/{demo_key}: {exc}",
                      file=sys.stderr)
                continue
            elapsed = time.perf_counter() - ts

            # Atomic write
            tmp = cpath.with_suffix(".npy.tmp")
            np.save(tmp, masks.astype(np.float32))
            tmp.replace(cpath)
            done += 1
            total_elapsed = time.perf_counter() - t0
            print(f"[rank {args.rank}] {ti+1}/{len(task_files)} "
                  f"{task_h5.stem}/{demo_key} T={T} -> "
                  f"{elapsed:.1f}s ({elapsed*1000/T:.0f} ms/frame) "
                  f"| done={done} skipped={skipped} elapsed={total_elapsed/60:.1f}m",
                  flush=True)

            if args.limit is not None and done >= args.limit:
                print(f"[rank {args.rank}] hit --limit {args.limit}, stopping")
                break
        if args.limit is not None and done >= args.limit:
            break

    print(f"[rank {args.rank}] DONE: written={done} skipped={skipped} "
          f"total_time={ (time.perf_counter()-t0)/60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
