"""Cross-Time Consistency Loss (XTC-Loss).

L_xtc = λ1 · ||f_t − f_{t-1} − Δf_pred(a_{t-1})||  (predicted-effect residual)
       + λ2 · contrastive(f_t, augmented(f_t))      (identity stability)

f_t is the per-entity feature emitted by PSE-Tok at step t.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class XTCLoss(nn.Module):
    def __init__(
        self,
        lambda_pred: float = 1.0,
        lambda_contrast: float = 0.1,
        contrast_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_contrast = lambda_contrast
        self.tau = contrast_temperature

    def forward(
        self,
        feat_t: torch.Tensor,        # (B, N, D)
        feat_t_prev: torch.Tensor,   # (B, N, D)
        feat_t_pred: torch.Tensor,   # (B, N, D), predicted from a_{t-1}
        feat_t_aug: torch.Tensor,    # (B, N, D), same step under augmentation
        confidence: torch.Tensor,    # (B, N) entity confidence at step t
    ) -> dict[str, torch.Tensor]:
        # 1. predicted-effect residual, masked by confidence
        residual = feat_t - feat_t_pred
        per_entity_loss = residual.pow(2).mean(dim=-1)  # (B, N)
        weights = confidence.clamp_min(0.0)
        denom = weights.sum().clamp_min(1.0)
        loss_pred = (weights * per_entity_loss).sum() / denom

        # 2. NT-Xent contrastive over (entity, aug_entity) pairs in a batch
        loss_contrast = self._info_nce(feat_t, feat_t_aug)

        loss = self.lambda_pred * loss_pred + self.lambda_contrast * loss_contrast
        return {
            "loss_xtc": loss,
            "loss_xtc_pred": loss_pred.detach(),
            "loss_xtc_contrast": loss_contrast.detach(),
        }

    def _info_nce(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # flatten batch and entity dims into a single sequence
        B, N, D = a.shape
        a = F.normalize(a.reshape(B * N, D), dim=-1)
        b = F.normalize(b.reshape(B * N, D), dim=-1)
        logits = a @ b.t() / self.tau
        labels = torch.arange(B * N, device=a.device)
        return F.cross_entropy(logits, labels)
