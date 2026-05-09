"""Thin wrapper to load OpenVLA-7B and expose the VLABackbone protocol.

We do NOT redistribute the OpenVLA weights here; the wrapper expects the
HF checkpoint to be available locally or downloadable.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OpenVLABackbone(nn.Module):
    embed_dim: int = 4096

    def __init__(self, model_id: str = "openvla/openvla-7b") -> None:
        super().__init__()
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:
            raise ImportError("`transformers>=4.45` required for OpenVLA") from e
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        self._action_proj = nn.Linear(self.embed_dim, 7)  # 6-DoF + gripper

    def encode_language(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(text_ids)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        # OpenVLA's VLM expects raw RGB tensors at the processor stage; for
        # this wrapper we assume images are already preprocessed by the caller
        feats = self.model.vision_backbone(image)
        return feats

    def action_head(
        self,
        language_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        prefix_tokens=None,
    ) -> torch.Tensor:
        seq = [language_tokens, image_tokens]
        if prefix_tokens is not None:
            seq.insert(1, prefix_tokens)
        x = torch.cat(seq, dim=1)
        h = self.model.language_model(inputs_embeds=x).last_hidden_state[:, -1]
        return self._action_proj(h)
