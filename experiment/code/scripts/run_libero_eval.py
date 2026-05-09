"""Smoke-tier LIBERO eval — OpenVLA-7B generalist on a single task.

Generalist OpenVLA on LIBERO will have low SR (no fine-tune); the goal here
is to validate the sim+model+rollout pipeline end-to-end on a real LIBERO
env, not to claim a number. Future runs will swap in the LIBERO-finetuned
checkpoints (openvla/openvla-7b-finetuned-libero-*).

Usage:
    python run_libero_eval.py \\
        --suite libero_spatial --task-id 0 \\
        --rollouts 5 --max-steps 200 \\
        --out experiment/runs/libero-smoke-XXX/02_libero_metrics.json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--unnorm-key", default="bridge_orig",
                    help="action de-normalization key — bridge_orig works for "
                         "smoke; LIBERO-finetuned ckpts use libero_spatial etc.")
    ap.add_argument("--model-id", default="openvla/openvla-7b")
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--libero-action-fix", action="store_true",
                    help="apply OpenVLA→LIBERO gripper convention fix: "
                         "rescale [0,1]→[-1,+1] via sign() and flip sign "
                         "(OpenVLA's 1=close, LIBERO's 1=open)")
    ap.add_argument("--libero-image-fix", action="store_true",
                    help="apply rgb[::-1, ::-1] (double flip / 180° rotate) "
                         "matching OpenVLA's official LIBERO image preprocessing")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"==> bench {args.suite}, task {args.task_id}")
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    print(f"    name: {task.name}")
    print(f"    language: {task.language}")
    init_states = suite.get_task_init_states(args.task_id)
    print(f"    init_states: {len(init_states)}")

    print(f"==> loading {args.model_id}")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
    ).eval()
    print(f"    loaded in {time.time()-t0:.1f}s")
    # Probe available unnorm keys so we can verify args.unnorm_key matches
    norm_stats = getattr(model, "norm_stats", None) or getattr(model.config, "norm_stats", None)
    if norm_stats:
        keys = list(norm_stats.keys())
        print(f"    available unnorm keys: {keys}")
        if args.unnorm_key not in keys:
            print(f"    WARN: --unnorm-key={args.unnorm_key} NOT in available keys")
    torch.cuda.reset_peak_memory_stats()

    bddl_path = os.path.join(get_libero_path("bddl_files"),
                             task.problem_folder, task.bddl_file)
    print(f"    bddl: {bddl_path}")
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": args.resolution,
        "camera_widths": args.resolution,
        "camera_names": ["agentview"],
    }

    # Single env across rollouts — recreating per rollout leaks fds in robosuite
    # MjRenderContext / EGLGLContext, hitting ulimit -n after ~10 rollouts.
    env = OffScreenRenderEnv(**env_args)
    rollouts = []
    for r in range(args.rollouts):
        print(f"==> rollout {r}")
        env.reset()
        env.set_init_state(init_states[r % len(init_states)])

        # LIBERO convention: 10 dummy steps to settle dynamics; keep last obs
        dummy = np.array([0., 0., 0., 0., 0., 0., -1.])
        obs = None
        for _ in range(10):
            obs, _, _, _ = env.step(dummy)

        success = False
        per_step_ms = []
        for t in range(args.max_steps):
            rgb = obs["agentview_image"]
            if args.libero_image_fix:
                # OpenVLA's official LIBERO eval applies double flip (180° rotate)
                rgb = rgb[::-1, ::-1].copy()
            else:
                # Default: single vertical flip (mujoco→standard image convention)
                rgb = rgb[::-1].copy()
            pil = Image.fromarray(rgb.astype(np.uint8))
            prompt = (f"In: What action should the robot take to "
                      f"{task.language.strip().lower()}?\nOut:")
            inputs = proc(prompt, pil).to("cuda:0", dtype=torch.bfloat16)
            s_t0 = time.time()
            with torch.no_grad():
                action = model.predict_action(**inputs,
                                              unnorm_key=args.unnorm_key,
                                              do_sample=False)
            torch.cuda.synchronize()
            per_step_ms.append((time.time() - s_t0) * 1000)
            # action: numpy (7,) — 6 DoF deltas + gripper
            if args.libero_action_fix:
                # OpenVLA gripper is in [0,1]; LIBERO expects sign(-1/+1).
                # Plus OpenVLA defines 1=close while LIBERO defines 1=open,
                # so we flip the sign as the last step.
                action = np.asarray(action, dtype=np.float32).copy()
                action[-1] = np.sign(2 * action[-1] - 1)
                action[-1] *= -1.0
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = bool(info.get("success", reward > 0))
                break

        rec = {
            "rollout": r,
            "success": success,
            "steps": t + 1,
            "step_ms_avg": float(np.mean(per_step_ms)),
            "step_ms_p95": float(np.percentile(per_step_ms, 95)),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        rollouts.append(rec)
        print(f"    success={success} steps={t+1} "
              f"step_ms={rec['step_ms_avg']:.1f} p95={rec['step_ms_p95']:.1f}")

    env.close()
    n_succ = sum(1 for r in rollouts if r["success"])
    out = {
        "suite": args.suite, "task_id": args.task_id,
        "task_name": task.name, "task_language": task.language,
        "model_id": args.model_id,
        "unnorm_key": args.unnorm_key,
        "n_rollouts": args.rollouts,
        "n_success": n_succ,
        "success_rate": n_succ / args.rollouts if args.rollouts else 0.0,
        "rollouts": rollouts,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"==> SR: {n_succ}/{args.rollouts} "
          f"({100*n_succ/args.rollouts if args.rollouts else 0:.0f}%)")
    print(f"==> wrote {args.out}")


if __name__ == "__main__":
    main()
