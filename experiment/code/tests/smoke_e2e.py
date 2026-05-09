"""End-to-end smoke test on real GPU.

Loads OpenVLA-7B, wraps it with PSSAVLA, and runs N rollouts in a
synthetic stepping environment that supplies random RGB/depth/masks per
step. Captures: per-step latency, peak GPU memory, CRED trigger counts,
and proves the entire forward path works against the real backbone
without needing LIBERO/CALVIN installed.

Usage:
    python smoke_e2e.py --rollouts 5 --episode-len 50 --out metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.openvla_wrapper import OpenVLABackbone
from pssa import CRED, PSSAVLA, PersistentSceneEntityTokenizer


class _SyntheticEnv:
    """Stepping env that returns random observations of fixed shape."""

    def __init__(self, episode_len: int = 50, n_entities: int = 8,
                 image_hw: tuple[int, int] = (224, 224), device: str = "cuda") -> None:
        self.episode_len = episode_len
        self.n_entities = n_entities
        self.image_hw = image_hw
        self.device = device
        self._t = 0

    def reset(self) -> dict:
        self._t = 0
        H, W = self.image_hw
        return {
            "rgb": torch.rand(3, H, W, device=self.device),
            "masks_init": (torch.rand(2, self.n_entities, H, W,
                                      device=self.device) > 0.5).float(),
        }

    def step(self, action) -> tuple[dict, float, bool, dict]:
        self._t += 1
        H, W = self.image_hw
        obs = {"rgb": torch.rand(3, H, W, device=self.device)}
        done = self._t >= self.episode_len
        return obs, 0.0, done, {"success": False}

    def task_text_ids(self) -> torch.Tensor:
        return torch.randint(0, 1024, (1, 8), device=self.device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=5)
    ap.add_argument("--episode-len", type=int, default=50)
    ap.add_argument("--model-id", default="openvla/openvla-7b")
    ap.add_argument("--out", default="metrics.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARN: no CUDA, smoke will run on CPU and be much slower.")

    print(f"==> Loading {args.model_id}")
    t0 = time.time()
    backbone = OpenVLABackbone(args.model_id).to(device).eval()
    load_s = time.time() - t0
    print(f"    load took {load_s:.1f}s")

    pse = PersistentSceneEntityTokenizer(
        n_entities=8, feature_dim=384, token_dim=backbone.embed_dim,
    ).to(device)
    cred = CRED(tau=0.5, k_consecutive=3, cooldown_steps=8)
    model = PSSAVLA(
        backbone=backbone, pse_tok=pse, cred=cred,
        use_pse=True, use_xtc=False, use_cred=True,
    ).to(device).eval()

    env = _SyntheticEnv(episode_len=args.episode_len, device=device)
    metrics = {
        "model_id": args.model_id, "device": device,
        "load_s": load_s, "rollouts": []
    }
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for r in range(args.rollouts):
        t0 = time.time()
        result = model.rollout(env, env.task_text_ids(), max_steps=args.episode_len)
        dt = time.time() - t0
        rec = {
            "rollout": r,
            "wall_s": dt,
            "step_ms_avg": 1000 * dt / max(result["steps"], 1),
            "cred_triggers": result["cred_triggers"],
            "steps": result["steps"],
        }
        if device == "cuda":
            rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        metrics["rollouts"].append(rec)
        print(f"    rollout {r}: {rec['steps']} steps, "
              f"{rec['step_ms_avg']:.1f} ms/step, "
              f"{rec['cred_triggers']} CRED triggers")

    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"==> wrote {args.out}")
    print(f"==> SMOKE E2E OK  (avg {sum(r['step_ms_avg'] for r in metrics['rollouts'])/len(metrics['rollouts']):.1f} ms/step)")


if __name__ == "__main__":
    main()
