"""PSSA-VLA v1 eval — load trained ckpt + run rollouts in LIBERO sim.

v1 caveats:
  - Model bypasses OpenVLA's vision_backbone; PSE-Tok carries all visual
    info, encoded from the first 4 frames after env.reset().
  - Action head outputs continuous (7,) deltas directly (no discretization).
  - Per-frame visual feedback during rollout is intentionally absent in v1;
    rollouts that need closed-loop visual correction will fail.

Usage:
    python run_pssa_eval.py \\
        --ckpt experiment/runs/pssa_train_v1-XXX/checkpoints/step_003000 \\
        --suite libero_spatial --task-id 0 --rollouts 10 \\
        --out runs/pssa_eval_v1-XXX/task_0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="path to checkpoints/step_XXXXXX/ dir")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--resolution", type=int, default=128,
                    help="must match training (LIBERO demos are 128x128)")
    ap.add_argument("--n-init-frames", type=int, default=4)
    ap.add_argument("--n-entities", type=int, default=8)
    ap.add_argument("--backbone-id", default="openvla/openvla-7b-finetuned-libero-spatial")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel
    from pssa.model_v2 import PSSAVLAv2, PSEEntityEncoder
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    ckpt_dir = Path(args.ckpt)
    device = "cuda:0"

    print(f"==> loading backbone {args.backbone_id}")
    proc = AutoProcessor.from_pretrained(args.backbone_id, trust_remote_code=True)
    backbone = AutoModelForVision2Seq.from_pretrained(
        args.backbone_id, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
        device_map=device,
    ).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    lora_path = ckpt_dir / "lora"
    if lora_path.exists():
        print(f"==> loading LoRA from {lora_path}")
        backbone = PeftModel.from_pretrained(backbone, str(lora_path))
        backbone.eval()

    cfg = getattr(backbone, "config", None) or backbone.base_model.config
    hidden_dim = getattr(cfg, "hidden_size", 4096)

    pse = PSEEntityEncoder(
        n_entities=args.n_entities, hidden_dim=hidden_dim, cnn_dim=256,
    )
    model = PSSAVLAv2(backbone=backbone, pse_encoder=pse, processor=proc)
    # Load PSE + action_head from saved ckpt
    modules_pt = ckpt_dir / "pssa_modules.pt"
    if not modules_pt.exists():
        raise FileNotFoundError(f"missing {modules_pt} — train.py must have saved both pse+action")
    state = torch.load(modules_pt, map_location="cpu")
    model.pse_encoder.load_state_dict(state["pse_encoder"])
    model.action_head.load_state_dict(state["action_head"])
    model = model.to(device).eval()

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    bddl_path = os.path.join(get_libero_path("bddl_files"),
                             task.problem_folder, task.bddl_file)
    print(f"==> task {args.task_id}: {task.name}")
    print(f"    language: {task.language}")

    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": args.resolution,
        "camera_widths": args.resolution,
        "camera_names": ["agentview"],
    }
    env = OffScreenRenderEnv(**env_args)

    rollouts_data = []
    torch.cuda.reset_peak_memory_stats()
    for r in range(args.rollouts):
        env.reset()
        env.set_init_state(init_states[r % len(init_states)])

        # Settle + collect init frames
        dummy = np.array([0., 0., 0., 0., 0., 0., -1.])
        init_rgbs = []
        obs = None
        for _ in range(args.n_init_frames + 10):  # 10 settle + 4 collected
            obs, _, _, _ = env.step(dummy)
            init_rgbs.append(obs["agentview_image"][::-1, ::-1].copy())
        # Use the LAST 4 settled frames as init
        init_rgbs = init_rgbs[-args.n_init_frames:]
        init_t = torch.from_numpy(np.stack(init_rgbs)).permute(0, 3, 1, 2).float() / 255.0
        init_t = init_t.unsqueeze(0).to(device)        # (1, T0, 3, H, W)
        masks_init = torch.ones(
            1, args.n_init_frames, args.n_entities,
            args.resolution, args.resolution, device=device,
        )

        # Compute persistent entity tokens once per rollout
        with torch.no_grad():
            pse_tokens = model.pse_encoder(init_t, masks_init)   # (1, N, D)

        per_step_ms = []
        success = False
        for t in range(args.max_steps):
            rgb_t = obs["agentview_image"][::-1, ::-1].copy()
            rgb_tensor = (torch.from_numpy(rgb_t).permute(2, 0, 1)
                          .float().unsqueeze(0) / 255.0).to(device)
            s_t0 = time.time()
            with torch.no_grad():
                # ent_t is unused by v1 _step_action_logits (vision bypassed)
                action_pred = model._step_action_logits(
                    pse_tokens, pse_tokens.unsqueeze(1),  # ent_t placeholder
                    rgb_tensor, [task.language],
                )
            torch.cuda.synchronize()
            per_step_ms.append((time.time() - s_t0) * 1000)
            action_np = action_pred[0].detach().cpu().float().numpy()
            obs, reward, done, info = env.step(action_np.tolist())
            if done:
                success = bool(info.get("success", reward > 0))
                break

        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        rec = {
            "rollout": r, "success": success, "steps": t + 1,
            "step_ms_avg": float(np.mean(per_step_ms)),
            "step_ms_p95": float(np.percentile(per_step_ms, 95)),
            "peak_vram_gb": float(peak_vram),
        }
        rollouts_data.append(rec)
        print(f"    rollout {r}: success={success} steps={t+1} "
              f"step_ms={rec['step_ms_avg']:.1f}")

    env.close()
    n_succ = sum(1 for r in rollouts_data if r["success"])
    out_data = {
        "suite": args.suite, "task_id": args.task_id, "task_name": task.name,
        "model": "PSSA-VLA v1.1 (text+PSE, vision bypassed)",
        "ckpt": str(ckpt_dir), "backbone_id": args.backbone_id,
        "n_rollouts": args.rollouts, "n_success": n_succ,
        "success_rate": n_succ / args.rollouts if args.rollouts else 0.0,
        "rollouts": rollouts_data,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"==> SR: {n_succ}/{args.rollouts} ({100*n_succ/args.rollouts:.0f}%)")
    print(f"==> wrote {args.out}")


if __name__ == "__main__":
    main()
