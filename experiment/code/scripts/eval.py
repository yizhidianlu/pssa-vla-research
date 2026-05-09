"""Eval entry point — runs rollouts over the configured benchmarks.

Saves per-rollout JSON to runs/<run_name>/rollouts/<benchmark>_<seed>.jsonl
and an aggregate metrics CSV to runs/<run_name>/metrics.csv.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.openvla_wrapper import OpenVLABackbone
from pssa import PSSAVLA


def _load_env(name: str):
    if name.startswith("libero"):
        from libero.libero import LIBEROEnv  # noqa
        return LIBEROEnv(name)
    if name.startswith("calvin"):
        from calvin_env.envs.tasks import Tasks  # noqa
        return Tasks(name)
    raise ValueError(f"unknown benchmark {name}")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    backbone = OpenVLABackbone(cfg.backbone.model_id).to(device)
    model = PSSAVLA(backbone=backbone).to(device).eval()

    out_root = Path(f"runs/{cfg.run_name}")
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for bench in cfg.eval.benchmarks:
        for seed in cfg.eval.seeds:
            torch.manual_seed(seed)
            env = _load_env(bench)
            successes = 0
            triggers_total = 0
            for r in range(cfg.eval.rollouts_per_task):
                text_ids = env.task_text_ids().to(device)
                result = model.rollout(env, text_ids)
                successes += int(result["success"])
                triggers_total += int(result["cred_triggers"])
                with open(out_root / "rollouts" / f"{bench}_{seed}.jsonl", "a") as f:
                    f.write(json.dumps({**result, "task": getattr(env, "task_name", "?")}) + "\n")
            rows.append({
                "benchmark": bench,
                "seed": seed,
                "success_rate": successes / cfg.eval.rollouts_per_task,
                "cred_triggers_per_rollout": triggers_total / cfg.eval.rollouts_per_task,
            })
    with open(out_root / "metrics.csv", "w") as f:
        f.write("benchmark,seed,success_rate,cred_triggers_per_rollout\n")
        for r in rows:
            f.write(f"{r['benchmark']},{r['seed']},{r['success_rate']:.4f},{r['cred_triggers_per_rollout']:.4f}\n")


if __name__ == "__main__":
    main()
