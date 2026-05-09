"""Training entry point. Usage:

    python scripts/train.py --config-name pssa_libero_long
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.openvla_wrapper import OpenVLABackbone
from data.libero_dataset import LIBEROEpisodeDataset
from pssa import CRED, PSSAVLA, PersistentSceneEntityTokenizer, XTCLoss


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    backbone = OpenVLABackbone(cfg.backbone.model_id).to(device)

    pse = PersistentSceneEntityTokenizer(
        n_entities=cfg.pssa.pse.n_entities,
        feature_dim=cfg.pssa.pse.feature_dim,
        token_dim=cfg.pssa.pse.token_dim,
        confidence_threshold=cfg.pssa.pse.confidence_threshold,
    ).to(device)
    xtc = XTCLoss(
        lambda_pred=cfg.pssa.xtc.lambda_pred,
        lambda_contrast=cfg.pssa.xtc.lambda_contrast,
        contrast_temperature=cfg.pssa.xtc.contrast_temperature,
    ).to(device)
    cred = CRED(
        tau=cfg.pssa.cred.tau,
        k_consecutive=cfg.pssa.cred.k_consecutive,
        cooldown_steps=cfg.pssa.cred.cooldown_steps,
    )

    model = PSSAVLA(
        backbone=backbone,
        pse_tok=pse,
        xtc_loss=xtc,
        cred=cred,
        use_pse=cfg.pssa.use_pse,
        use_xtc=cfg.pssa.use_xtc,
        use_cred=cfg.pssa.use_cred,
    ).to(device)

    ds = LIBEROEpisodeDataset(split="long", root=str(cfg.data_root))
    if len(ds) == 0:
        raise FileNotFoundError(
            f"No episodes under {ds.root}. Populate $NR_DATA per experiment/setup.md."
        )
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    step = 0
    while step < cfg.train.max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.training_step(batch)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            step += 1
            if step % cfg.train.log_every == 0:
                print(f"step={step} loss={float(out['loss']):.4f} "
                      f"loss_action={float(out['loss_action']):.4f} "
                      f"loss_xtc={float(out['loss_xtc']):.4f}")
            if step >= cfg.train.max_steps:
                break


if __name__ == "__main__":
    main()
