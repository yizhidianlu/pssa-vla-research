"""Phase-2 PSSA training entrypoint.

Single-GPU usage:
    python -m pssa.train --config-path ../configs --config-name train

Dual-GPU FSDP via accelerate:
    accelerate launch --num_processes 2 \
        --use_fsdp --fsdp_sharding_strategy FULL_SHARD \
        -m pssa.train --config-path ../configs --config-name train

Outputs:
    experiment/runs/pssa_train-<TS>/
        checkpoints/{step,latest}.pt
        config.yaml
        train.log
        wandb/   (if enabled)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pssa.dataset import LIBEROEpisodeDataset
from pssa.model_v2 import PSSAVLAv2, PSEEntityEncoder


def _load_backbone(model_id: str, freeze: bool = True):
    """Load OpenVLA-7B-finetuned-libero-X + matching processor; freeze optionally."""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    return model, proc


def _apply_lora(model, r: int = 32, alpha: int = 64, dropout: float = 0.05):
    """Apply LoRA on q_proj/v_proj of the LLM stack."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError("Install peft>=0.10 for LoRA training") from e
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # For DDP with our IterableDataset + list[str] language field, disable
    # accelerate's automatic batch dispatch — each rank reads its own batches
    # via rank-aware seeding inside LIBEROEpisodeDataset.__iter__.
    from accelerate import DataLoaderConfiguration
    dl_cfg = DataLoaderConfiguration(dispatch_batches=False, split_batches=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        dataloader_config=dl_cfg,
        log_with="wandb" if cfg.wandb.enable else None,
    )
    accelerator.print(OmegaConf.to_yaml(cfg))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, out_dir / "config.yaml")

    # ----- model -----
    accelerator.print(f"==> loading backbone {cfg.model.id}")
    backbone, processor = _load_backbone(cfg.model.id, freeze=cfg.model.freeze_backbone)
    if cfg.model.lora.enable:
        backbone = _apply_lora(
            backbone,
            r=cfg.model.lora.r,
            alpha=cfg.model.lora.alpha,
            dropout=cfg.model.lora.dropout,
        )
        backbone.print_trainable_parameters()

    pse_encoder = PSEEntityEncoder(
        n_entities=cfg.model.pse.n_entities,
        hidden_dim=cfg.model.pse.hidden_dim,
        cnn_dim=cfg.model.pse.cnn_dim,
        zero_init_output=cfg.model.pse.get("zero_init_output", False),
    )
    model = PSSAVLAv2(
        backbone=backbone,
        pse_encoder=pse_encoder,
        processor=processor,
        lambda_xtc=cfg.model.lambda_xtc,
        pse_position=cfg.model.get("pse_position", "after_image"),
        n_pse_tokens=cfg.model.pse.n_entities,
    )

    # ----- data -----
    train_ds = LIBEROEpisodeDataset(
        suite=cfg.data.suite,
        root=cfg.data.root,
        window_len=cfg.data.window_len,
        n_init_frames=cfg.data.n_init_frames,
        n_entities=cfg.model.pse.n_entities,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=LIBEROEpisodeDataset.collate,
    )

    # ----- optimizer -----
    trainable = [p for p in model.parameters() if p.requires_grad]
    accelerator.print(f"==> {sum(p.numel() for p in trainable)/1e6:.2f}M trainable params")
    opt = torch.optim.AdamW(trainable, lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    # Compensate for accelerate's per-rank scheduler stepping under DDP:
    # with N processes, the wrapped scheduler is stepped N times per
    # data iteration, so we scale T_max to keep the cosine schedule
    # spanning the full intended number of OPTIMIZER steps.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cfg.train.max_steps * accelerator.num_processes,
        eta_min=cfg.train.lr * 0.1,
    )

    model, opt, train_loader, sched = accelerator.prepare(model, opt, train_loader, sched)

    # ----- training loop -----
    step = 0
    t_start = time.time()
    log_lines = []
    for batch in train_loader:
        with accelerator.accumulate(model):
            # Call via forward() so DDP/FSDP wrappers can hook gradient sync
            out = model(batch)
            accelerator.backward(out.loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable, cfg.train.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad()
        step += 1

        if step % cfg.train.log_every == 0 and accelerator.is_main_process:
            dt = time.time() - t_start
            line = (f"step {step:>6d} | loss {float(out.loss):.4f} "
                    f"| L_act {float(out.loss_action):.4f} "
                    f"| L_xtc {float(out.loss_xtc):.4f} "
                    f"| lr {sched.get_last_lr()[0]:.2e} "
                    f"| {step/dt:.2f} step/s")
            accelerator.print(line)
            log_lines.append(line)

        if step % cfg.train.save_every == 0 and accelerator.is_main_process:
            ckpt_dir = out_dir / "checkpoints" / f"step_{step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            # C route: only pse_encoder is trainable outside LoRA (no action_head).
            torch.save({
                "pse_encoder": unwrapped.pse_encoder.state_dict(),
            }, ckpt_dir / "pssa_modules.pt")
            if cfg.model.lora.enable:
                unwrapped.backbone.save_pretrained(ckpt_dir / "lora")
            (ckpt_dir / "step.txt").write_text(str(step))
            accelerator.print(f"saved {ckpt_dir}")

        if step >= cfg.train.max_steps:
            break

    if accelerator.is_main_process:
        (out_dir / "train.log").write_text("\n".join(log_lines) + "\n")
        accelerator.print(f"==> training done in {(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
