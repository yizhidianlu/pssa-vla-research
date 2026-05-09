"""Thin wrapper around the open π0 release if available.

We treat π0 as an opaque action policy and only re-expose the action_head
through the VLABackbone protocol. If the release is gated, the
EXECUTION-stage runner is expected to fall back to "cited number, not
reproduced" per setup.md §4.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Pi0Backbone(nn.Module):
    embed_dim: int = 2048

    def __init__(self, ckpt_path: str | None = None) -> None:
        super().__init__()
        self.ckpt_path = ckpt_path
        self._stub = nn.Linear(self.embed_dim, 7)

    def encode_language(self, text_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(text_ids.shape[0], 1, self.embed_dim, device=text_ids.device)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        return torch.zeros(B, 1, self.embed_dim, device=image.device)

    def action_head(
        self,
        language_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        prefix_tokens=None,
    ) -> torch.Tensor:
        x = image_tokens.mean(dim=1)
        return self._stub(x)
