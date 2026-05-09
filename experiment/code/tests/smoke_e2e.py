"""End-to-end smoke test on real GPU — split into two independent probes.

Probe A: OpenVLA-7B inference on synthetic observations using the model's
  own predict_action() pipeline. Validates the model loads, the processor
  works, and CUDA inference completes. Captures latency + VRAM.

Probe B: PSSA components (PSE-Tok, XTC-Loss, CRED) on a stub backbone —
  identical to tests/smoke.py logic, repeated here for self-containment.

Probe A and B are intentionally decoupled: integrating PSE-Tok prefix
tokens into OpenVLA's discretized-action token stream is a fine-tune-time
problem, not a smoke-time problem.

Usage:
    python smoke_e2e.py --rollouts 5 --episode-len 50 --out metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def probe_pssa_on_gpu() -> dict:
    """Probe B — PSSA components on stub backbone, on the actual GPU device."""
    from pssa import CRED, PSSAVLA, PersistentSceneEntityTokenizer, XTCLoss
    import torch.nn as nn

    class _Stub(nn.Module):
        embed_dim = 64
        def __init__(self):
            super().__init__()
            self.lang = nn.Embedding(1024, self.embed_dim)
            self.vision = nn.Conv2d(3, self.embed_dim, 32, 32)
            self.head = nn.Linear(self.embed_dim, 7)
        def encode_language(self, ids): return self.lang(ids)
        def encode_image(self, x): return self.vision(x).flatten(2).transpose(1, 2)
        def action_head(self, lang, img, prefix=None):
            seq = [lang, img]
            if prefix is not None: seq.insert(1, prefix)
            return self.head(torch.cat(seq, dim=1).mean(dim=1))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = _Stub().to(device)
    pse = PersistentSceneEntityTokenizer(n_entities=8, feature_dim=64, token_dim=64).to(device)
    model = PSSAVLA(backbone=backbone, pse_tok=pse, xtc_loss=XTCLoss(), cred=CRED()).to(device)
    B, T = 1, 3
    batch = {
        "rgb_seq": torch.randn(B, T, 3, 64, 64, device=device),
        "text_ids": torch.randint(0, 1024, (B, 5), device=device),
        "actions": torch.randn(B, T, 7, device=device),
        "masks_init": (torch.rand(B, 2, 8, 64, 64, device=device) > 0.5).float(),
    }
    out = model.training_step(batch)
    out["loss"].backward()
    return {
        "device": device,
        "loss": float(out["loss"]),
        "loss_action": float(out["loss_action"]),
        "loss_xtc": float(out["loss_xtc"]),
    }


def probe_openvla(model_id: str, rollouts: int, episode_len: int) -> dict:
    """Probe A — OpenVLA-7B predict_action() on synthetic RGB observations."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==> loading {model_id} on {device}")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True, device_map=device,
    )
    model.eval()
    load_s = time.time() - t0
    print(f"    loaded in {load_s:.1f}s, params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    instruction = "pick up the red block and place it on the blue plate"
    rollouts_data = []
    for r in range(rollouts):
        ep_t0 = time.time()
        per_step_ms = []
        for t in range(episode_len):
            # synthetic 224x224 RGB observation
            rgb = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
            img = Image.fromarray(rgb)
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            inputs = proc(prompt, img).to(device, dtype=torch.bfloat16)
            s_t0 = time.time()
            with torch.no_grad():
                action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            if device == "cuda":
                torch.cuda.synchronize()
            per_step_ms.append((time.time() - s_t0) * 1000)
        ep_dt = time.time() - ep_t0
        rec = {
            "rollout": r,
            "wall_s": ep_dt,
            "step_ms_avg": float(np.mean(per_step_ms)),
            "step_ms_p50": float(np.percentile(per_step_ms, 50)),
            "step_ms_p95": float(np.percentile(per_step_ms, 95)),
            "steps": len(per_step_ms),
            "last_action": action.tolist() if hasattr(action, "tolist") else list(action),
        }
        if device == "cuda":
            rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        rollouts_data.append(rec)
        print(f"    rollout {r}: {rec['steps']} steps, {rec['step_ms_avg']:.1f}±{rec['step_ms_p95']-rec['step_ms_p50']:.1f} ms/step")

    return {
        "model_id": model_id, "device": device, "load_s": load_s,
        "rollouts": rollouts_data,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=5)
    ap.add_argument("--episode-len", type=int, default=50)
    ap.add_argument("--model-id", default="openvla/openvla-7b")
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--skip-openvla", action="store_true",
                    help="run only probe B (PSSA on stub) — useful for offline testing")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")

    out: dict = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch": torch.__version__,
        "cuda_avail": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

    print("==> Probe B: PSSA components on GPU")
    out["probe_b_pssa"] = probe_pssa_on_gpu()
    print(f"    OK: loss={out['probe_b_pssa']['loss']:.4f}")

    if not args.skip_openvla:
        print("==> Probe A: OpenVLA-7B predict_action()")
        out["probe_a_openvla"] = probe_openvla(args.model_id, args.rollouts, args.episode_len)
        ar = out["probe_a_openvla"]["rollouts"]
        avg = sum(r["step_ms_avg"] for r in ar) / len(ar)
        print(f"    overall avg: {avg:.1f} ms/step")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"==> wrote {args.out}")
    print("==> SMOKE E2E OK")


if __name__ == "__main__":
    main()
