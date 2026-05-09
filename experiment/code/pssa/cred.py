"""Consistency-Residual Error Detector (CRED).

At inference, CRED tracks per-step feature residuals against the persistent
entity prior. If residual exceeds threshold τ for K consecutive steps the
controller is asked to freeze and replan from the persistent prior.

This is intentionally NOT a learned module — keeping it parameter-free
makes the head-to-head with VLA-in-the-Loop honest on inference cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CREDState:
    consecutive_violations: int = 0
    last_residual: float = 0.0
    triggered: bool = False


class CRED:
    def __init__(
        self,
        tau: float = 0.5,
        k_consecutive: int = 3,
        cooldown_steps: int = 8,
    ) -> None:
        self.tau = tau
        self.k = k_consecutive
        self.cooldown = cooldown_steps
        self._cooldown_left = 0

    def reset(self) -> CREDState:
        self._cooldown_left = 0
        return CREDState()

    def step(
        self,
        feat_t: torch.Tensor,        # (N, D)
        feat_t_pred: torch.Tensor,   # (N, D)
        confidence: torch.Tensor,    # (N,)
        state: CREDState,
    ) -> tuple[CREDState, bool, dict[str, float]]:
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return state, False, {"residual": 0.0, "tau": self.tau}
        residual = (feat_t - feat_t_pred).pow(2).mean(dim=-1)  # (N,)
        # confidence-weighted scalar residual
        w = confidence.clamp_min(0.0)
        denom = w.sum().clamp_min(1.0)
        scalar_residual = float((w * residual).sum() / denom)
        if scalar_residual > self.tau:
            state.consecutive_violations += 1
        else:
            state.consecutive_violations = 0
        state.last_residual = scalar_residual
        triggered = state.consecutive_violations >= self.k
        if triggered:
            state.triggered = True
            self._cooldown_left = self.cooldown
            state.consecutive_violations = 0
        return state, triggered, {"residual": scalar_residual, "tau": self.tau}
