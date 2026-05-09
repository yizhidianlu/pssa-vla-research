"""Smoke test — gate before EXECUTION (blueprint §9).

Runs in three layers:
  1. PSE-Tok forward on dummy RGB+masks → produces N tokens.
  2. PSSAVLA training_step on dummy batch with a stub backbone → backward
     pass closes without NaN.
  3. CRED.step over a synthetic residual trace → triggers exactly once.

Layer 1+2 use a tiny stub backbone so the smoke test does not require
OpenVLA weights or a GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pssa import CRED, PSSAVLA, PersistentSceneEntityTokenizer, XTCLoss


class _StubBackbone(nn.Module):
    embed_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.lang = nn.Embedding(1024, self.embed_dim)
        self.vision = nn.Conv2d(3, self.embed_dim, 32, 32)
        self.head = nn.Linear(self.embed_dim, 7)

    def encode_language(self, text_ids):
        return self.lang(text_ids)

    def encode_image(self, image):
        return self.vision(image).flatten(2).transpose(1, 2)

    def action_head(self, language_tokens, image_tokens, prefix_tokens=None):
        seq = [language_tokens, image_tokens]
        if prefix_tokens is not None:
            seq.insert(1, prefix_tokens)
        x = torch.cat(seq, dim=1).mean(dim=1)
        return self.head(x)


def test_pse_tok() -> None:
    pse = PersistentSceneEntityTokenizer(n_entities=8, feature_dim=64, token_dim=64)
    rgb = torch.randn(4, 3, 64, 64)
    masks = (torch.rand(4, 8, 64, 64) > 0.7).float()
    state = pse.init_episode(rgb, masks, depth=None)
    new_state, tokens = pse.step(state, rgb[0:1], None, action_prev=torch.randn(1, 7))
    assert tokens.shape == (1, 8, 64), tokens.shape
    assert new_state.confidence.shape == (8,)
    print(f"[1/3] PSE-Tok OK — tokens={tuple(tokens.shape)}")


def test_training_step() -> None:
    backbone = _StubBackbone()
    pse = PersistentSceneEntityTokenizer(n_entities=4, feature_dim=64, token_dim=64)
    model = PSSAVLA(backbone=backbone, pse_tok=pse, xtc_loss=XTCLoss(), cred=CRED())
    B, T = 1, 3
    batch = {
        "rgb_seq": torch.randn(B, T, 3, 64, 64),
        "text_ids": torch.randint(0, 1024, (B, 5)),
        "actions": torch.randn(B, T, 7),
        "masks_init": (torch.rand(B, 2, 4, 64, 64) > 0.5).float(),
    }
    out = model.training_step(batch)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]), "training loss is non-finite"
    print(f"[2/3] PSSAVLA training_step OK — loss={float(out['loss']):.4f}")


def test_cred_trigger() -> None:
    cred = CRED(tau=0.1, k_consecutive=2, cooldown_steps=4)
    state = cred.reset()
    fired_count = 0
    feat = torch.zeros(4, 16)
    feat_pred = torch.zeros(4, 16)
    conf = torch.ones(4)
    # 5 normal steps
    for _ in range(5):
        state, fired, _ = cred.step(feat, feat_pred, conf, state)
        fired_count += int(fired)
    # 3 violation steps in a row
    for _ in range(3):
        feat_high = feat + 1.0
        state, fired, _ = cred.step(feat_high, feat_pred, conf, state)
        fired_count += int(fired)
    assert fired_count == 1, f"expected one fire, got {fired_count}"
    print("[3/3] CRED OK — exactly one trigger")


if __name__ == "__main__":
    test_pse_tok()
    test_training_step()
    test_cred_trigger()
    print("\nSMOKE OK")
