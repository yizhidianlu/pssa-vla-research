"""Persistent Scene-Entity Tokenizer (PSE-Tok).

Episode-level state: a Persistent Gaussian Splat of scene entities with
identity-stable IDs across all frames. Per step it emits up to N entity
tokens (id, 3D position, appearance feature) for the action head.

This file is a reference scaffold. The Gaussian-splat update path here is
deliberately lightweight: full POGS reconstruction is delegated to gsplat
at episode init and per-step warps are linearized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EntityState:
    """Per-entity state maintained across an episode."""
    ids: torch.Tensor          # (N,) long, stable IDs
    centers: torch.Tensor      # (N, 3) world-frame centers
    features: torch.Tensor     # (N, D) appearance features
    confidence: torch.Tensor   # (N,) [0, 1]
    last_seen: torch.Tensor    # (N,) step index

    def detach(self) -> "EntityState":
        return EntityState(
            self.ids.detach(),
            self.centers.detach(),
            self.features.detach(),
            self.confidence.detach(),
            self.last_seen.detach(),
        )


class PersistentSceneEntityTokenizer(nn.Module):
    """Maintains a persistent set of N entity tokens across an episode.

    Args:
        n_entities: max entities tracked per episode
        feature_dim: appearance feature dimensionality
        token_dim: output token width consumed by the action head
        feature_extractor: any nn.Module mapping (B, 3, H, W) → (B, D, H', W')
        confidence_threshold: below this confidence the entity is considered lost
    """

    def __init__(
        self,
        n_entities: int = 32,
        feature_dim: int = 384,
        token_dim: int = 768,
        feature_extractor: Optional[nn.Module] = None,
        confidence_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_entities = n_entities
        self.feature_dim = feature_dim
        self.token_dim = token_dim
        self.confidence_threshold = confidence_threshold
        self.feature_extractor = feature_extractor or _DefaultViTLite(feature_dim)
        self.token_proj = nn.Linear(feature_dim + 3, token_dim)
        self.id_embed = nn.Embedding(n_entities, token_dim)
        self.register_buffer("_id_arange", torch.arange(n_entities))

    @torch.no_grad()
    def init_episode(
        self,
        rgb_seq: torch.Tensor,         # (T0, 3, H, W) — first ~2-4 frames
        masks: torch.Tensor,           # (T0, n_entities, H, W) from SAM-2
        depth: Optional[torch.Tensor], # (T0, 1, H, W), unprojected to xyz outside
    ) -> EntityState:
        """Build initial EntityState from the first few frames of an episode."""
        T0 = rgb_seq.shape[0]
        device = rgb_seq.device
        feats = self.feature_extractor(rgb_seq)  # (T0, D, H', W')
        feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)  # (T0, D)
        # mask-pooled features per entity, averaged across the init window
        m = masks.float().mean(dim=0)  # (n_entities, H, W)
        f_per_entity = feats.mean(dim=0).unsqueeze(0).expand(self.n_entities, -1).clone()
        if depth is not None:
            xyz = _mask_centroid_xyz(masks, depth)  # (T0, n_entities, 3)
            centers = xyz.mean(dim=0)
        else:
            centers = torch.zeros(self.n_entities, 3, device=device)
        confidence = m.mean(dim=(-1, -2)).clamp(0, 1)
        return EntityState(
            ids=self._id_arange.clone(),
            centers=centers,
            features=f_per_entity,
            confidence=confidence,
            last_seen=torch.zeros(self.n_entities, device=device, dtype=torch.long),
        )

    def step(
        self,
        state: EntityState,
        rgb_t: torch.Tensor,      # (B, 3, H, W)
        depth_t: Optional[torch.Tensor],
        action_prev: Optional[torch.Tensor],  # (B, A) for predicted-effect warp
    ) -> tuple[EntityState, torch.Tensor]:
        """Advance state by one frame and emit (B, n_entities, token_dim) tokens.

        Returns (new_state, tokens). Tokens are zero-masked where confidence
        is below `self.confidence_threshold`; the action head should interpret
        the per-token confidence channel rather than gating tokens itself.
        """
        feat_t = F.adaptive_avg_pool2d(self.feature_extractor(rgb_t), 1)
        feat_t = feat_t.squeeze(-1).squeeze(-1)  # (B, D)
        # linearized predicted-effect warp on centers (placeholder — replace
        # with learned dynamics or predicted SE(3))
        if action_prev is not None:
            delta = self._predict_center_delta(action_prev, state)  # (B, N, 3)
            centers_pred = state.centers.unsqueeze(0) + delta
        else:
            centers_pred = state.centers.unsqueeze(0).expand(rgb_t.shape[0], -1, -1)
        # exponential update of features; full POGS would warp Gaussians here
        alpha = 0.7
        features_new = alpha * state.features.unsqueeze(0) + (1 - alpha) * feat_t.unsqueeze(1)
        # confidence decays slowly; would be re-asserted by SAM-2 mask matches
        conf_new = state.confidence.clamp(0, 1) * 0.99
        # tokens: concat (feature, center) and project, then add ID embedding
        tok_in = torch.cat([features_new, centers_pred], dim=-1)
        tokens = self.token_proj(tok_in) + self.id_embed(state.ids).unsqueeze(0)
        # zero-mask low-confidence — but pass confidence as a side channel
        tokens = tokens * (conf_new.unsqueeze(0).unsqueeze(-1) > self.confidence_threshold).float()
        new_state = EntityState(
            ids=state.ids,
            centers=centers_pred.mean(dim=0).detach() if action_prev is None
                    else centers_pred[0].detach(),
            features=features_new.mean(dim=0).detach(),
            confidence=conf_new.detach(),
            last_seen=state.last_seen + 1,
        )
        return new_state, tokens

    def _predict_center_delta(self, action_prev: torch.Tensor, state: EntityState) -> torch.Tensor:
        """Map the previous action to a per-entity center delta. Placeholder
        implementation: only the active gripper-target entity moves.

        Replace with a learned action-conditioned dynamics module.
        """
        B = action_prev.shape[0]
        N = state.ids.shape[0]
        delta = torch.zeros(B, N, 3, device=action_prev.device)
        if action_prev.shape[-1] >= 3:
            delta[:, 0, :3] = action_prev[..., :3]
        return delta


class _DefaultViTLite(nn.Module):
    """Tiny stand-in feature extractor for unit tests; replace with timm ViT."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(128, feature_dim, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mask_centroid_xyz(masks: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Compute mask-weighted centroids in (x, y, z) image space.

    Args:
        masks: (T, N, H, W)
        depth: (T, 1, H, W)
    Returns:
        (T, N, 3)
    """
    T, N, H, W = masks.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=masks.device, dtype=torch.float32),
        torch.arange(W, device=masks.device, dtype=torch.float32),
        indexing="ij",
    )
    out = []
    for t in range(T):
        d = depth[t, 0]  # (H, W)
        per_t = []
        for n in range(N):
            m = masks[t, n].float()
            w = m.sum().clamp_min(1.0)
            cx = (m * xs).sum() / w
            cy = (m * ys).sum() / w
            cz = (m * d).sum() / w
            per_t.append(torch.stack([cx, cy, cz]))
        out.append(torch.stack(per_t))
    return torch.stack(out)
