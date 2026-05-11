"""PSSAVLAv2 (Route C) — OpenVLA's native discrete-action-token decoder + PSE prefix.

Architecture:
    [text | image_embeds (256) | PSE_prefix (N)] -> Llama -> action tokens (7)

Loss:
    L_action : CE on the 7 action token positions (OpenVLA's native objective).
    L_xtc    : cross-time consistency on per-step PSE entity features.
    L_total  = L_action + lambda_xtc * L_xtc

Training scope:
- Frozen   : vision_backbone, projector, LM head
- Trainable: PSE-Tok encoder, LoRA adapters on Llama q/v projections.

The crucial difference from v1.1: we predict action tokens via the same LM head
OpenVLA was finetuned with (so we inherit the 80.2% LIBERO-Spatial behavior as
the starting policy), and only train the PSE prefix + LoRA delta on top.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PSSATrainingOutput:
    loss: torch.Tensor
    loss_action: torch.Tensor
    loss_xtc: torch.Tensor
    n_entity_tokens: int


class PSEEntityEncoder(nn.Module):
    """Encodes (B, T0, 3, H, W) initial frames + (B, T0, N, H, W) masks into
    (B, N, hidden_dim) persistent entity tokens.
    """

    def __init__(self, n_entities: int = 8, hidden_dim: int = 4096,
                 cnn_dim: int = 256) -> None:
        super().__init__()
        self.n_entities = n_entities
        self.hidden_dim = hidden_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, cnn_dim, 3, stride=2, padding=1),
        )
        self.id_embed = nn.Embedding(n_entities, cnn_dim)
        self.proj = nn.Sequential(
            nn.Linear(cnn_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, rgb_init: torch.Tensor, masks_init: torch.Tensor) -> torch.Tensor:
        B, T0, _, H, W = rgb_init.shape
        N = self.n_entities
        feats = self.cnn(rgb_init.flatten(0, 1))
        _, C, h, w = feats.shape
        masks_small = F.adaptive_avg_pool2d(masks_init.flatten(0, 1), (h, w))
        feats_per_entity = torch.einsum("bchw,bnhw->bnc", feats, masks_small)
        denom = masks_small.flatten(2).sum(-1, keepdim=True).clamp_min(1.0)
        feats_per_entity = feats_per_entity / denom
        feats_per_entity = feats_per_entity.view(B, T0, N, C).mean(dim=1)
        ids = torch.arange(N, device=feats_per_entity.device)
        id_e = self.id_embed(ids).unsqueeze(0).expand(B, -1, -1)
        out = self.proj(torch.cat([feats_per_entity, id_e], dim=-1))
        return out


class XTCLoss(nn.Module):
    """Cross-time consistency: smoothness penalty on entity features."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, entity_seq: torch.Tensor) -> torch.Tensor:
        if entity_seq.size(1) < 2:
            return entity_seq.new_zeros(())
        diff = entity_seq[:, 1:] - entity_seq[:, :-1]
        return diff.pow(2).mean()


class PSSAVLAv2(nn.Module):
    """Route C: PSE prefix + OpenVLA native action-token CE loss."""

    def __init__(
        self,
        backbone,
        pse_encoder: PSEEntityEncoder | None = None,
        processor=None,
        lambda_xtc: float = 0.1,
        unnorm_key: str = "libero_spatial",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        cfg = getattr(backbone, "config", None) or backbone.base_model.config
        hidden_dim = getattr(cfg, "hidden_size", 4096)
        self.pse_encoder = pse_encoder or PSEEntityEncoder(hidden_dim=hidden_dim)
        self.xtc_loss = XTCLoss()
        self.lambda_xtc = lambda_xtc
        self.processor = processor
        self.unnorm_key = unnorm_key
        self._cache_action_stats()

    # ---- action discretization helpers ------------------------------------

    def _cache_action_stats(self) -> None:
        """Pull action norm stats + vocab size from the OpenVLA backbone once."""
        inner = self._unwrap()
        cfg = inner.config
        # vocab_size lives on the inner Llama config, not the OpenVLA wrapper
        self._vocab_size = inner.language_model.config.vocab_size  # 32000
        self._n_action_bins = cfg.n_action_bins  # 256
        stats = cfg.norm_stats[self.unnorm_key]["action"]
        # q01/q99 are the unnormalization bounds; mask indicates which dims to unnormalize
        self._action_q01 = np.array(stats["q01"], dtype=np.float32)  # (7,)
        self._action_q99 = np.array(stats["q99"], dtype=np.float32)
        if "mask" in stats:
            self._action_mask = np.array(stats["mask"], dtype=bool)
        else:
            self._action_mask = np.ones_like(self._action_q01, dtype=bool)
        # Bin edges in [-1, 1]
        self._bin_edges = np.linspace(-1.0, 1.0, self._n_action_bins + 1)
        self._bin_centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2.0
        self._action_dim = len(self._action_q01)  # 7

    def _discretize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Continuous action (B, 7) -> token IDs (B, 7) in [vocab-256, vocab-1].

        Follows OpenVLA's convention: token_id = vocab_size - bin_index - 1
        where bin_index in [0, n_action_bins-1].
        """
        device = action.device
        a_np = action.detach().cpu().float().numpy()  # (B, 7)

        # Unnormalize -> normalize. The model was finetuned on action-token-ed
        # continuous actions. Map back: normalized = 2 * (a - q01) / (q99 - q01) - 1
        q01 = self._action_q01[None, :]
        q99 = self._action_q99[None, :]
        normalized = 2.0 * (a_np - q01) / np.clip(q99 - q01, 1e-8, None) - 1.0
        # For masked dims (typically all True for libero), keep normalized; for
        # unmasked dims (none here, but defensive), use raw action.
        mask = self._action_mask[None, :]
        normalized = np.where(mask, normalized, a_np)
        normalized = np.clip(normalized, -1.0, 1.0)
        # digitize -> bin_index in [0, 255]
        bin_idx = np.digitize(normalized, self._bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, self._n_action_bins - 1).astype(np.int64)
        # token_id = vocab_size - bin_idx - 1
        token_ids = self._vocab_size - bin_idx - 1
        return torch.from_numpy(token_ids).to(device)

    # ---- training step ----------------------------------------------------

    def training_step(self, batch: dict) -> PSSATrainingOutput:
        rgb_init = batch["rgb_init"]
        rgb_seq = batch["rgb_seq"]
        actions = batch["actions"]              # (B, T, 7) continuous
        masks_init = batch["masks_init"]
        language = batch["language"]

        B, T = rgb_seq.shape[:2]
        device = rgb_seq.device

        # Persistent entity tokens from init frames
        pse_tokens = self.pse_encoder(rgb_init, masks_init)            # (B, N, D)

        loss_action = rgb_seq.new_zeros(())
        entity_seq = []
        for t in range(T):
            ent_t = self.pse_encoder(
                rgb_seq[:, t:t+1], masks_init[:, :1].expand(-1, 1, -1, -1, -1)
            )
            entity_seq.append(ent_t)

            loss_t = self._step_action_ce_loss(
                pse_tokens, rgb_seq[:, t], language, actions[:, t]
            )
            loss_action = loss_action + loss_t

        loss_action = loss_action / max(T, 1)

        # XTC on per-step entity sequence
        if entity_seq:
            entity_stack = torch.stack(entity_seq, dim=1)              # (B, T, 1, N, D)
            entity_stack = entity_stack.squeeze(2)                     # (B, T, N, D)
            loss_xtc = self.xtc_loss(entity_stack)
        else:
            loss_xtc = rgb_seq.new_zeros(())

        loss = loss_action + self.lambda_xtc * loss_xtc
        return PSSATrainingOutput(
            loss=loss,
            loss_action=loss_action.detach(),
            loss_xtc=loss_xtc.detach(),
            n_entity_tokens=pse_tokens.shape[1],
        )

    def _step_action_ce_loss(
        self,
        pse_tokens: torch.Tensor,        # (B, N, D)
        rgb_t: torch.Tensor,             # (B, 3, H, W) in [0, 1]
        language: list[str],
        action_target: torch.Tensor,     # (B, 7) continuous
    ) -> torch.Tensor:
        """Build [text | image | PSE | action_target_embeds] and compute CE
        on the 7 action token positions."""
        device = rgb_t.device
        B = rgb_t.shape[0]

        from PIL import Image
        rgb_np = (rgb_t.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu()
                  .float().numpy() * 255).astype(np.uint8)
        pil_imgs = [Image.fromarray(rgb_np[i]) for i in range(B)]

        prompts = [f"In: what action to {ln.strip().lower()}? Out:" for ln in language]

        inputs = self.processor(
            text=prompts, images=pil_imgs, return_tensors="pt", padding=True,
        )
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"].to(device)
        attn_text = inputs["attention_mask"].to(device)

        # Append the special-empty token (29871) that predict_action uses, then
        # the discretized action target — this matches OpenVLA's training layout
        special_empty = input_ids.new_full((B, 1), 29871)
        action_token_ids = self._discretize_action(action_target)     # (B, 7) long
        target_ids = torch.cat([special_empty, action_token_ids], dim=1)  # (B, 8)

        # Embed everything via the model's input embeddings (frozen unless LoRA wraps)
        embed_layer = self._get_input_embeddings()
        text_embeds = embed_layer(input_ids)                          # (B, L_text, D)
        target_embeds = embed_layer(target_ids)                       # (B, 8, D)

        inner = self._unwrap()
        with torch.no_grad():
            img_feats = inner.vision_backbone(pixel_values)           # (B, 256, 2176)
            image_embeds = inner.projector(img_feats).detach()        # (B, 256, D)

        pse_embeds = pse_tokens.to(text_embeds.dtype)

        # Final sequence: [text | image | PSE | special_empty + action_tokens]
        inputs_embeds = torch.cat(
            [text_embeds, image_embeds, pse_embeds, target_embeds], dim=1,
        )
        attn_img = torch.ones(image_embeds.shape[:2], device=device, dtype=attn_text.dtype)
        attn_pse = torch.ones(pse_embeds.shape[:2], device=device, dtype=attn_text.dtype)
        attn_tgt = torch.ones(target_embeds.shape[:2], device=device, dtype=attn_text.dtype)
        attn = torch.cat([attn_text, attn_img, attn_pse, attn_tgt], dim=1)

        # Run inner transformer (no lm_head over all positions).
        # Apply lm_head only at the 7 positions that predict action tokens —
        # saves ~277x activation memory on logits + backward grads vs full
        # vocab logits over the whole 284-token sequence.
        inner_lm = self._get_inner_transformer()       # LlamaModel (no head)
        out = inner_lm(inputs_embeds=inputs_embeds, attention_mask=attn)
        hidden_states = out.last_hidden_state          # (B, L, D)

        # Sequence layout: [text | image | PSE | special_empty | a0 a1 .. a6]
        # We want logits at positions [pse_end .. pse_end+6] to predict
        # [a0 .. a6] (the model uses position k's hidden state to predict
        # the token AT position k+1; here pse_end's hidden predicts a0).
        text_len = text_embeds.shape[1]
        img_len = image_embeds.shape[1]
        pse_len = pse_embeds.shape[1]
        pse_end = text_len + img_len + pse_len         # index of special_empty
        relevant = hidden_states[:, pse_end : pse_end + 7]  # (B, 7, D)

        lm_head = self._get_lm_head()
        logits = lm_head(relevant)                     # (B, 7, V) in lm_head dtype
        loss = F.cross_entropy(
            logits.float().reshape(-1, self._vocab_size),
            action_token_ids.reshape(-1),
        )
        return loss

    # ---- inference --------------------------------------------------------

    @torch.no_grad()
    def compute_pse(
        self,
        rgb_init: torch.Tensor,         # (1, T0, 3, H, W)
        masks_init: torch.Tensor,       # (1, T0, N, H, W)
    ) -> torch.Tensor:
        """One-shot PSE encoding from initial frames. Call once per rollout."""
        return self.pse_encoder(rgb_init, masks_init)                  # (1, N, D)

    @torch.no_grad()
    def predict_action(
        self,
        pse_tokens: torch.Tensor,       # (1, N, D)  pre-computed via compute_pse
        rgb_t: torch.Tensor,            # (1, 3, H, W)  current frame in [0, 1]
        language: list[str],            # length-1
    ) -> np.ndarray:
        """Predict a 7-DoF continuous action for the current observation.

        Build [text | image | PSE | special_empty] and autoregressively decode
        7 action tokens, then de-discretize + unnormalize. Returns shape (7,).
        """
        device = rgb_t.device

        from PIL import Image
        rgb_np = (rgb_t.clamp(0, 1).permute(0, 2, 3, 1).cpu().float()
                  .numpy() * 255).astype(np.uint8)
        pil_imgs = [Image.fromarray(rgb_np[0])]
        prompts = [f"In: what action to {language[0].strip().lower()}? Out:"]

        inputs = self.processor(
            text=prompts, images=pil_imgs, return_tensors="pt", padding=True,
        )
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"].to(device)
        attn_text = inputs["attention_mask"].to(device)

        special_empty = input_ids.new_full((1, 1), 29871)

        embed_layer = self._get_input_embeddings()
        text_embeds = embed_layer(input_ids)
        special_embed = embed_layer(special_empty)

        inner = self._unwrap()
        img_feats = inner.vision_backbone(pixel_values)
        image_embeds = inner.projector(img_feats)

        pse_embeds = pse_tokens.to(text_embeds.dtype)

        prefix_embeds = torch.cat(
            [text_embeds, image_embeds, pse_embeds, special_embed], dim=1,
        )
        attn_img = torch.ones(image_embeds.shape[:2], device=device, dtype=attn_text.dtype)
        attn_pse = torch.ones(pse_embeds.shape[:2], device=device, dtype=attn_text.dtype)
        attn_se = torch.ones((1, 1), device=device, dtype=attn_text.dtype)
        attn_prefix = torch.cat([attn_text, attn_img, attn_pse, attn_se], dim=1)

        inner_lm = self._get_inner_transformer()
        lm_head = self._get_lm_head()
        gen_ids = []
        cur_embeds = prefix_embeds
        cur_attn = attn_prefix
        for _ in range(self._action_dim):
            out = inner_lm(inputs_embeds=cur_embeds, attention_mask=cur_attn)
            last_hidden = out.last_hidden_state[:, -1:, :]              # (1, 1, D)
            next_logits = lm_head(last_hidden).float().squeeze(1)       # (1, V)
            # Restrict to action token range
            mask = torch.full_like(next_logits, -float("inf"))
            mask[:, self._vocab_size - self._n_action_bins : self._vocab_size] = 0.0
            next_logits = next_logits + mask
            next_id = next_logits.argmax(dim=-1, keepdim=True)
            gen_ids.append(next_id.item())
            next_embed = embed_layer(next_id)
            cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
            cur_attn = torch.cat(
                [cur_attn, torch.ones((1, 1), device=device, dtype=attn_text.dtype)],
                dim=1,
            )

        token_ids = np.array(gen_ids, dtype=np.int64)
        bin_idx = self._vocab_size - token_ids - 1
        bin_idx = np.clip(bin_idx, 0, self._n_action_bins - 1)
        normalized = self._bin_centers[bin_idx]
        unnormed = self._action_q01 + (normalized + 1.0) / 2.0 * (
            self._action_q99 - self._action_q01
        )
        action = np.where(self._action_mask, unnormed, normalized)
        return action.astype(np.float32)

    # ---- module-access helpers --------------------------------------------

    def _unwrap(self):
        m = self.backbone
        for attr in ("base_model", "model"):
            if hasattr(m, attr):
                inner = getattr(m, attr)
                if hasattr(inner, "vision_backbone"):
                    return inner
        return m

    def _get_input_embeddings(self):
        return self._unwrap().get_input_embeddings()

    def _get_inner_transformer(self):
        """Return the inner LlamaModel (no lm_head) so we can apply the head
        manually only at the positions we need."""
        lm = getattr(self._unwrap(), "language_model", None)
        if lm is None:
            raise AttributeError("backbone has no .language_model")
        inner = getattr(lm, "model", None)
        return inner if inner is not None else lm

    def _get_lm_head(self):
        """Return the standalone lm_head Linear, which we'll apply to a tiny
        sub-slice of hidden states to compute logits over the action vocab."""
        lm = getattr(self._unwrap(), "language_model", None)
        if lm is None:
            raise AttributeError("backbone has no .language_model")
        return lm.lm_head
