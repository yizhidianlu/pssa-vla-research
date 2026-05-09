"""PSSAVLA — composes a VLA backbone with PSE-Tok prefix tokens.

Backbone is OpenVLA-style: a vision encoder + LLM with an action head. We
inject PSE entity tokens into the sequence right after the language tokens
and before the per-frame ViT tokens.

This file keeps the actual OpenVLA load behind a thin protocol so unit
tests can run with a stub backbone.
"""
from __future__ import annotations

from typing import Optional, Protocol

import torch
import torch.nn as nn

from .pse_tok import EntityState, PersistentSceneEntityTokenizer
from .xtc_loss import XTCLoss
from .cred import CRED, CREDState


class VLABackbone(Protocol):
    """Minimal interface a VLA backbone must implement."""
    embed_dim: int

    def encode_language(self, text_ids: torch.Tensor) -> torch.Tensor: ...
    def encode_image(self, image: torch.Tensor) -> torch.Tensor: ...
    def action_head(
        self,
        language_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        prefix_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor: ...


class PSSAVLA(nn.Module):
    """Wraps a VLA backbone with PSE-Tok grounding."""

    def __init__(
        self,
        backbone: VLABackbone,
        pse_tok: Optional[PersistentSceneEntityTokenizer] = None,
        xtc_loss: Optional[XTCLoss] = None,
        cred: Optional[CRED] = None,
        use_pse: bool = True,
        use_xtc: bool = True,
        use_cred: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_pse = use_pse
        self.use_xtc = use_xtc
        self.use_cred = use_cred
        self.pse_tok = pse_tok or PersistentSceneEntityTokenizer(token_dim=backbone.embed_dim)
        self.xtc_loss = xtc_loss or XTCLoss()
        self.cred = cred or CRED()

    # ---- training ---------------------------------------------------------
    def training_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        rgb_seq = batch["rgb_seq"]            # (B, T, 3, H, W)
        text_ids = batch["text_ids"]          # (B, L)
        actions = batch["actions"]            # (B, T, A)
        masks_init = batch["masks_init"]      # (B, T0, N, H, W)
        depth_init = batch.get("depth_init")  # (B, T0, 1, H, W)

        B, T = rgb_seq.shape[:2]
        loss_action = rgb_seq.new_zeros(())
        loss_xtc = rgb_seq.new_zeros(())
        log = {}

        states: list[EntityState] = []
        for b in range(B):
            states.append(self.pse_tok.init_episode(
                rgb_seq[b, :masks_init.shape[1]],
                masks_init[b],
                depth_init[b] if depth_init is not None else None,
            ))

        lang = self.backbone.encode_language(text_ids)  # (B, L', D)

        for t in range(T):
            img_tok = self.backbone.encode_image(rgb_seq[:, t])
            entity_tokens = []
            entity_features = []
            entity_features_pred = []
            entity_conf = []
            new_states = []
            for b in range(B):
                a_prev = actions[b, t - 1].unsqueeze(0) if t > 0 else None
                new_state, tok = self.pse_tok.step(
                    states[b],
                    rgb_seq[b, t].unsqueeze(0),
                    None,
                    a_prev,
                )
                entity_tokens.append(tok)
                entity_features.append(new_state.features)
                entity_features_pred.append(states[b].features)  # before update
                entity_conf.append(new_state.confidence)
                new_states.append(new_state)
            states = new_states
            prefix = torch.cat(entity_tokens, dim=0) if self.use_pse else None
            pred_action = self.backbone.action_head(lang, img_tok, prefix)
            loss_action = loss_action + (pred_action - actions[:, t]).pow(2).mean()

            if self.use_xtc and t > 0:
                f_t = torch.stack(entity_features, dim=0)
                f_pred = torch.stack(entity_features_pred, dim=0)
                f_aug = f_t + 0.01 * torch.randn_like(f_t)  # cheap stand-in aug
                conf = torch.stack(entity_conf, dim=0)
                xtc_out = self.xtc_loss(f_t, f_pred, f_pred, f_aug, conf)
                loss_xtc = loss_xtc + xtc_out["loss_xtc"]
                log.setdefault("loss_xtc_pred", []).append(xtc_out["loss_xtc_pred"])

        loss_action = loss_action / T
        loss_xtc = loss_xtc / max(T - 1, 1) if self.use_xtc else loss_xtc
        loss = loss_action + loss_xtc
        return {"loss": loss, "loss_action": loss_action.detach(), "loss_xtc": loss_xtc.detach()}

    # ---- inference --------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        env,
        text_ids: torch.Tensor,
        max_steps: int = 600,
    ) -> dict:
        """Single-episode rollout, returning success + CRED telemetry."""
        obs = env.reset()
        rgb0 = obs["rgb"].unsqueeze(0)
        masks0 = obs["masks_init"].unsqueeze(0)
        depth0 = obs.get("depth_init")
        state = self.pse_tok.init_episode(rgb0[0], masks0[0], depth0[0] if depth0 is not None else None)
        cred_state = self.cred.reset()
        lang = self.backbone.encode_language(text_ids)
        action_prev: Optional[torch.Tensor] = None
        residuals: list[float] = []
        triggers: int = 0
        success: bool = False
        for t in range(max_steps):
            img_tok = self.backbone.encode_image(obs["rgb"].unsqueeze(0))
            new_state, tok = self.pse_tok.step(
                state, obs["rgb"].unsqueeze(0), None,
                action_prev.unsqueeze(0) if action_prev is not None else None,
            )
            prefix = tok if self.use_pse else None
            action = self.backbone.action_head(lang, img_tok, prefix)[0]

            if self.use_cred:
                cred_state, fired, info = self.cred.step(
                    new_state.features, state.features, new_state.confidence, cred_state,
                )
                residuals.append(info["residual"])
                if fired:
                    triggers += 1
                    action = action.zero_()  # freeze; replan hook can be inserted here
            obs, reward, done, info = env.step(action.cpu().numpy())
            action_prev = action.detach()
            state = new_state
            if done:
                success = bool(info.get("success", reward > 0))
                break
        return {
            "success": success,
            "steps": t + 1,
            "cred_triggers": triggers,
            "cred_residual_trace": residuals,
        }
