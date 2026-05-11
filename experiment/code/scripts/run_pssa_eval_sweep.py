"""PSSA-VLA v1 sweep eval — load ckpt once, eval across multiple LIBERO tasks.

Loads backbone + LoRA + PSE + action_head ONCE then loops over --task-ids.
Writes one JSON per task plus a summary.json with mean SR.

Usage:
    python run_pssa_eval_sweep.py \\
        --ckpt experiment/runs/pssa_train_v1-XXX/checkpoints/step_003000 \\
        --suite libero_spatial --task-ids 0,1,2,3,4,5,6,7,8,9 \\
        --rollouts 10 --max-steps 200 \\
        --out runs/pssa_eval_v1_sweep-XXX
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

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rollout_one_task(model, env_args_template, task, init_states, args, device):
    from libero.libero.envs import OffScreenRenderEnv

    env_args = dict(env_args_template)
    env_args["bddl_file_name"] = args["bddl_path"]
    env = OffScreenRenderEnv(**env_args)

    rollouts_data = []
    n_init_frames = args["n_init_frames"]
    n_entities = args["n_entities"]
    resolution = args["resolution"]
    max_steps = args["max_steps"]
    n_rollouts = args["rollouts"]

    for r in range(n_rollouts):
        env.reset()
        env.set_init_state(init_states[r % len(init_states)])

        dummy = np.array([0., 0., 0., 0., 0., 0., -1.])
        init_rgbs = []
        obs = None
        for _ in range(n_init_frames + 10):
            obs, _, _, _ = env.step(dummy)
            init_rgbs.append(obs["agentview_image"][::-1, ::-1].copy())
        init_rgbs = init_rgbs[-n_init_frames:]
        init_t = torch.from_numpy(np.stack(init_rgbs)).permute(0, 3, 1, 2).float() / 255.0
        init_t = init_t.unsqueeze(0).to(device)
        masks_init = torch.ones(
            1, n_init_frames, n_entities, resolution, resolution, device=device,
        )

        with torch.no_grad():
            pse_tokens = model.pse_encoder(init_t, masks_init)

        per_step_ms = []
        success = False
        t = 0
        for t in range(max_steps):
            rgb_t = obs["agentview_image"][::-1, ::-1].copy()
            rgb_tensor = (torch.from_numpy(rgb_t).permute(2, 0, 1)
                          .float().unsqueeze(0) / 255.0).to(device)
            s_t0 = time.time()
            with torch.no_grad():
                action_pred = model._step_action_logits(
                    pse_tokens, pse_tokens.unsqueeze(1),
                    rgb_tensor, [task.language],
                )
            torch.cuda.synchronize()
            per_step_ms.append((time.time() - s_t0) * 1000)
            action_np = action_pred[0].detach().cpu().float().numpy()
            obs, reward, done, info = env.step(action_np.tolist())
            if done:
                success = bool(info.get("success", reward > 0))
                break

        rollouts_data.append({
            "rollout": r,
            "success": success,
            "steps": t + 1,
            "step_ms_avg": float(np.mean(per_step_ms)) if per_step_ms else 0.0,
        })
    env.close()
    return rollouts_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9",
                    help="comma-separated task ids")
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--n-init-frames", type=int, default=4)
    ap.add_argument("--n-entities", type=int, default=8)
    ap.add_argument("--backbone-id", default="openvla/openvla-7b-finetuned-libero-spatial")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel
    from pssa.model_v2 import PSSAVLAv2, PSEEntityEncoder
    from libero.libero import benchmark, get_libero_path

    ckpt_dir = Path(args.ckpt)
    device = args.device

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
    modules_pt = ckpt_dir / "pssa_modules.pt"
    if not modules_pt.exists():
        raise FileNotFoundError(f"missing {modules_pt}")
    state = torch.load(modules_pt, map_location="cpu")
    model.pse_encoder.load_state_dict(state["pse_encoder"])
    model.action_head.load_state_dict(state["action_head"])
    model = model.to(device).eval()

    suite = benchmark.get_benchmark_dict()[args.suite]()
    bddl_root = get_libero_path("bddl_files")

    env_args_template = {
        "camera_heights": args.resolution,
        "camera_widths": args.resolution,
        "camera_names": ["agentview"],
    }
    args_dict = {
        "n_init_frames": args.n_init_frames,
        "n_entities": args.n_entities,
        "resolution": args.resolution,
        "max_steps": args.max_steps,
        "rollouts": args.rollouts,
    }

    all_results = {}
    t_start_sweep = time.time()
    for tid in task_ids:
        task = suite.get_task(tid)
        init_states = suite.get_task_init_states(tid)
        bddl_path = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        args_dict["bddl_path"] = bddl_path
        print(f"==> task {tid}: {task.name}")
        print(f"    language: {task.language}")

        t_start = time.time()
        rollouts = rollout_one_task(model, env_args_template, task,
                                    init_states, args_dict, device)
        elapsed = time.time() - t_start
        n_succ = sum(1 for r in rollouts if r["success"])
        task_data = {
            "suite": args.suite, "task_id": tid, "task_name": task.name,
            "ckpt": str(ckpt_dir), "backbone_id": args.backbone_id,
            "n_rollouts": args.rollouts, "n_success": n_succ,
            "success_rate": n_succ / args.rollouts,
            "elapsed_s": elapsed,
            "rollouts": rollouts,
        }
        out_path = out_dir / f"task_{tid:02d}.json"
        with open(out_path, "w") as f:
            json.dump(task_data, f, indent=2)
        print(f"    task {tid}: SR {n_succ}/{args.rollouts} "
              f"({100*n_succ/args.rollouts:.0f}%)  elapsed {elapsed:.0f}s")
        all_results[tid] = task_data

    sweep_elapsed = time.time() - t_start_sweep
    total_succ = sum(d["n_success"] for d in all_results.values())
    total_rol = sum(d["n_rollouts"] for d in all_results.values())
    summary = {
        "suite": args.suite,
        "ckpt": str(ckpt_dir),
        "backbone_id": args.backbone_id,
        "task_ids": task_ids,
        "n_tasks": len(task_ids),
        "n_rollouts_per_task": args.rollouts,
        "n_success": total_succ,
        "n_rollouts": total_rol,
        "mean_success_rate": total_succ / total_rol if total_rol else 0.0,
        "per_task_sr": {
            str(tid): all_results[tid]["success_rate"] for tid in task_ids
        },
        "sweep_elapsed_s": sweep_elapsed,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"==> SWEEP SR: {total_succ}/{total_rol} "
          f"({100*total_succ/total_rol:.1f}%)  elapsed {sweep_elapsed/60:.1f} min")
    print(f"==> wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
