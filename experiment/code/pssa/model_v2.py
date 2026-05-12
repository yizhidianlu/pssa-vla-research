"""PSSAVLAv2 (Route C v2) — OpenVLA-correct sequence layout + PSE prefix.

OpenVLA's modeling_prismatic.py forward concatenates as:
    [input_ids[:, :1] (BOS) | projected_patch_embeddings (256) | input_ids[:, 1:] (text)]

Anything that deviates from this layout corrupts the position embeddings the
LLM was finetuned with. The first v2c iteration put text first and got 0/100 SR
on LIBERO-Spatial. This rewrite restores the canonical layout and adds an
explicit `pse_position` flag so we can A/B-test where to insert the PSE prefix:

    pse_position="after_image":  [BOS | image(256) | PSE(N) | text(L-1) | empty | actions(7)]
    pse_position="before_action":[BOS | image(256) | text(L-1) | PSE(N) | empty | actions(7)]

The 7 action-token positions for CE loss are always the LAST 7 positions
(action_target_embeds at training time, or autoregressively generated at eval).

Loss:
    L_action : CE on the 7 action token positions (OpenVLA's native objective).
    L_xtc    : cross-time consistency on per-step PSE entity features.
    L_total  = L_action + lambda_xtc * L_xtc

Training scope:
- Frozen   : vision_backbone, projector, LM head, input embeddings
- Trainable: PSE-Tok encoder, LoRA adapters on Llama q/v projections.
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
                 cnn_dim: int = 256, zero_init_output: bool = False) -> None:
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
        # Prefix-tuning trick: zero-init the FINAL linear so the encoder
        # starts producing exactly-zero outputs. The model then trivially
        # reproduces OpenVLA's behavior at step 0 and learns to inject
        # nonzero PSE signal only when it improves the action-token CE.
        if zero_init_output:
            nn.init.zeros_(self.proj[2].weight)
            nn.init.zeros_(self.proj[2].bias)

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
    """Route C v2: OpenVLA-correct [BOS|image|text] layout + PSE prefix
    injected at a configurable position.
    """

    def __init__(
        self,
        backbone,
        pse_encoder: PSEEntityEncoder | None = None,
        processor=None,
        lambda_xtc: float = 0.1,
        unnorm_key: str = "libero_spatial",
        pse_position: str = "after_image",  # or "before_action"
        n_pse_tokens: int = 8,              # 0 = no-PSE control
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
        if pse_position not in ("after_image", "before_action"):
            raise ValueError(f"pse_position must be 'after_image' or 'before_action', got {pse_position}")
        self.pse_position = pse_position
        self.n_pse_tokens = n_pse_tokens
        self._cache_action_stats()

    # ---- action discretization helpers ------------------------------------

    def _cache_action_stats(self) -> None:
        """Pull action norm stats + vocab sizes from the OpenVLA backbone once.

        Key distinction (root cause of Gate-1 R1 failure): OpenVLA uses
            action_vocab_boundary = text_config.vocab_size - pad_to_multiple_of
        as the action-token anchor, NOT the inner Llama vocab_size. For OpenVLA-
        7B-libero this gives 32064 - 64 = 32000. Action token IDs sit in
            [action_vocab_boundary - 256, action_vocab_boundary - 1]
        = [31744, 31999]. The lm_head outputs the full padded vocab 32064.

        Bin centers also follow OpenVLA's exact scheme: 256 bin edges in [-1, 1]
        produce 255 bin centers (NOT 256) for de-discretization.
        """
        inner = self._unwrap()
        cfg = inner.config
        # Full lm_head output vocab (32064 for OpenVLA-7B)
        self._lm_vocab_size = inner.language_model.config.vocab_size
        # Action-token anchor (32000 = 32064 - 64)
        self._action_vocab_boundary = (
            cfg.text_config.vocab_size - cfg.pad_to_multiple_of
        )
        self._n_action_bins = cfg.n_action_bins  # 256
        stats = cfg.norm_stats[self.unnorm_key]["action"]
        self._action_q01 = np.array(stats["q01"], dtype=np.float32)
        self._action_q99 = np.array(stats["q99"], dtype=np.float32)
        if "mask" in stats:
            self._action_mask = np.array(stats["mask"], dtype=bool)
        else:
            self._action_mask = np.ones_like(self._action_q01, dtype=bool)
        # OpenVLA's exact discretization scheme: 256 edges -> 255 centers
        self._bins = np.linspace(-1.0, 1.0, self._n_action_bins)
        self._bin_centers = (self._bins[:-1] + self._bins[1:]) / 2.0
        self._action_dim = len(self._action_q01)

    def _discretize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Continuous action (B, 7) -> token IDs (B, 7) in [31744, 31999].

        Matches OpenVLA's ActionTokenizer (prismatic/vla/action_tokenizer.py):
            action = np.clip(action, -1, +1)
            discretized = np.digitize(action, bins=linspace(-1, 1, 256))
            token_id = vocab_boundary - discretized

        NOTE: OpenVLA does NOT apply q01/q99 normalization at training time.
        The q01/q99 stretch is the INVERSE map at inference (predict_action)
        — it expands the model's normalized [-1, 1] output to the env's
        action range. Training simply clips raw demo actions to [-1, 1].
        """
        device = action.device
        a_np = action.detach().cpu().float().numpy()  # (B, 7)
        a_np = np.clip(a_np, -1.0, 1.0)
        discretized = np.digitize(a_np, self._bins)
        token_ids = self._action_vocab_boundary - discretized
        token_ids = np.clip(
            token_ids,
            self._action_vocab_boundary - self._n_action_bins,
            self._action_vocab_boundary - 1,
        ).astype(np.int64)
        return torch.from_numpy(token_ids).to(device)

    # ---- training step ----------------------------------------------------

    def forward(self, batch: dict) -> PSSATrainingOutput:
        """Training forward — exposed as `forward` so DDP can wrap it."""
        return self.training_step(batch)

    def training_step(self, batch: dict) -> PSSATrainingOutput:
        rgb_init = batch["rgb_init"]
        rgb_seq = batch["rgb_seq"]
        actions = batch["actions"]              # (B, T, 7) continuous
        masks_init = batch["masks_init"]
        language = batch["language"]

        B, T = rgb_seq.shape[:2]
        device = rgb_seq.device

        # Persistent entity tokens from init frames; empty (B, 0, D) if no-PSE
        if self.n_pse_tokens == 0:
            pse_tokens = rgb_seq.new_zeros((B, 0, self.pse_encoder.hidden_dim))
        else:
            pse_tokens = self.pse_encoder(rgb_init, masks_init)        # (B, N, D)

        loss_action = rgb_seq.new_zeros(())
        entity_seq = []
        for t in range(T):
            if self.n_pse_tokens > 0:
                ent_t = self.pse_encoder(
                    rgb_seq[:, t:t+1], masks_init[:, :1].expand(-1, 1, -1, -1, -1)
                )
                entity_seq.append(ent_t)

            loss_t = self._step_action_ce_loss(
                pse_tokens, rgb_seq[:, t], language, actions[:, t]
            )
            loss_action = loss_action + loss_t

        loss_action = loss_action / max(T, 1)

        # XTC on per-step entity sequence (skip if no-PSE)
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

    def _build_prefix_embeds(
        self,
        pse_tokens: torch.Tensor,       # (B, N, D); empty (B, 0, D) if no_PSE
        rgb_t: torch.Tensor,            # (B, 3, H, W) in [0, 1]
        language: list[str],
    ):
        """Assemble OpenVLA-correct prefix [BOS | image | (PSE if after_image)
        | text | (PSE if before_action)] and the matching attention mask.

        Returns (prefix_embeds, prefix_attn, lengths_dict). prefix_embeds has
        shape (B, total_prefix_len, D); the caller appends special_empty +
        action targets (training) or generates autoregressively (eval).
        """
        device = rgb_t.device
        B = rgb_t.shape[0]

        from PIL import Image
        rgb_np = (rgb_t.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu()
                  .float().numpy() * 255).astype(np.uint8)
        pil_imgs = [Image.fromarray(rgb_np[i]) for i in range(B)]
        # Match OpenVLA's exact training-time prompt format (per Phase-1
        # run_libero_eval.py which achieves 80.2% SR). Wording AND newline
        # before "Out:" matter for the Llama tokenizer.
        prompts = [
            f"In: What action should the robot take to {ln.strip().lower()}?\nOut:"
            for ln in language
        ]

        inputs = self.processor(
            text=prompts, images=pil_imgs, return_tensors="pt", padding=True,
        )
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"].to(device)                    # (B, L_text)
        attn_text_full = inputs["attention_mask"].to(device)          # (B, L_text)

        embed_layer = self._get_input_embeddings()
        # OpenVLA splits input_ids at index 0: BOS goes pre-image, rest goes post-image
        bos_ids = input_ids[:, :1]
        rest_ids = input_ids[:, 1:]
        bos_embed = embed_layer(bos_ids)                              # (B, 1, D)
        text_embed = embed_layer(rest_ids)                            # (B, L_text-1, D)
        attn_bos = attn_text_full[:, :1]
        attn_text = attn_text_full[:, 1:]

        # Image features (frozen)
        inner = self._unwrap()
        with torch.no_grad():
            img_feats = inner.vision_backbone(pixel_values)           # (B, 256, 2176)
            image_embeds = inner.projector(img_feats).detach()        # (B, 256, D)
        attn_img = torch.ones(image_embeds.shape[:2], device=device, dtype=attn_text.dtype)

        # PSE prefix (may be 0-length for no-PSE control)
        pse_embeds = pse_tokens.to(bos_embed.dtype)
        attn_pse = torch.ones(pse_embeds.shape[:2], device=device, dtype=attn_text.dtype)

        # Assemble per pse_position. OpenVLA's mandatory order is [BOS | image | text].
        # We insert PSE either right after image (Variant A) or right before action
        # (Variant B, i.e. after text).
        if self.pse_position == "after_image":
            prefix_embeds = torch.cat([bos_embed, image_embeds, pse_embeds, text_embed], dim=1)
            prefix_attn = torch.cat([attn_bos, attn_img, attn_pse, attn_text], dim=1)
        else:  # before_action
            prefix_embeds = torch.cat([bos_embed, image_embeds, text_embed, pse_embeds], dim=1)
            prefix_attn = torch.cat([attn_bos, attn_img, attn_text, attn_pse], dim=1)

        lengths = {
            "bos": 1, "image": image_embeds.shape[1],
            "pse": pse_embeds.shape[1], "text": text_embed.shape[1],
        }
        return prefix_embeds, prefix_attn, lengths, embed_layer

    def _step_action_ce_loss(
        self,
        pse_tokens: torch.Tensor,        # (B, N, D)
        rgb_t: torch.Tensor,             # (B, 3, H, W) in [0, 1]
        language: list[str],
        action_target: torch.Tensor,     # (B, 7) continuous
    ) -> torch.Tensor:
        """Build OpenVLA-correct sequence + append special_empty + action
        targets, then CE on the 7 action token positions."""
        device = rgb_t.device
        B = rgb_t.shape[0]

        prefix_embeds, prefix_attn, _, embed_layer = self._build_prefix_embeds(
            pse_tokens, rgb_t, language,
        )

        # Append special_empty(29871) + discretized action targets for teacher forcing
        special_empty = torch.full((B, 1), 29871, dtype=torch.long, device=device)
        action_token_ids = self._discretize_action(action_target)     # (B, 7) long
        target_ids = torch.cat([special_empty, action_token_ids], dim=1)  # (B, 8)
        target_embeds = embed_layer(target_ids)                       # (B, 8, D)

        inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)
        attn_target = torch.ones(target_embeds.shape[:2], device=device, dtype=prefix_attn.dtype)
        attn = torch.cat([prefix_attn, attn_target], dim=1)

        # Forward inner LlamaModel (no lm_head over all positions).
        inner_lm = self._get_inner_transformer()
        out = inner_lm(inputs_embeds=inputs_embeds, attention_mask=attn)
        hidden_states = out.last_hidden_state          # (B, L, D)

        # CE on last 7 positions: hidden at [-8, -7, ..., -2] predicts tokens at [-7, -6, ..., -1]
        # which are exactly action_token_ids[0..6].
        # i.e. hidden_states[:, -8 : -1] -> predicts targets, which are at positions [-7..-1] = action_token_ids.
        relevant = hidden_states[:, -8:-1]             # (B, 7, D)  positions of [empty, a0..a5]
        lm_head = self._get_lm_head()
        logits = lm_head(relevant)                     # (B, 7, V_lm) in lm_head dtype
        loss = F.cross_entropy(
            logits.float().reshape(-1, self._lm_vocab_size),
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
        """One-shot PSE encoding from initial frames. Returns (1, n_pse_tokens, D).

        n_pse_tokens=0 path returns an empty tensor for no-PSE control.
        """
        if self.n_pse_tokens == 0:
            B = rgb_init.shape[0]
            hidden = self.pse_encoder.hidden_dim
            return rgb_init.new_zeros((B, 0, hidden))
        return self.pse_encoder(rgb_init, masks_init)                  # (1, N, D)

    @torch.no_grad()
    def predict_action(
        self,
        pse_tokens: torch.Tensor,       # (1, N, D)  pre-computed via compute_pse
        rgb_t: torch.Tensor,            # (1, 3, H, W)  current frame in [0, 1]
        language: list[str],            # length-1
    ) -> np.ndarray:
        """Predict a 7-DoF continuous action via OpenVLA-correct layout +
        autoregressive 7-token decode + de-discretize. Returns shape (7,)."""
        device = rgb_t.device

        prefix_embeds, prefix_attn, _, embed_layer = self._build_prefix_embeds(
            pse_tokens, rgb_t, language,
        )

        # Append special_empty (29871) — OpenVLA's predict_action mandates this
        # appears immediately before action generation.
        special_empty = torch.full((1, 1), 29871, dtype=torch.long, device=device)
        special_embed = embed_layer(special_empty)
        cur_embeds = torch.cat([prefix_embeds, special_embed], dim=1)
        attn_se = torch.ones((1, 1), device=device, dtype=prefix_attn.dtype)
        cur_attn = torch.cat([prefix_attn, attn_se], dim=1)

        inner_lm = self._get_inner_transformer()
        lm_head = self._get_lm_head()
        gen_ids = []
        action_lo = self._action_vocab_boundary - self._n_action_bins  # 31744
        action_hi = self._action_vocab_boundary                         # 32000 (exclusive)
        for _ in range(self._action_dim):
            out = inner_lm(inputs_embeds=cur_embeds, attention_mask=cur_attn)
            last_hidden = out.last_hidden_state[:, -1:, :]              # (1, 1, D)
            next_logits = lm_head(last_hidden).float().squeeze(1)       # (1, V_lm)
            # Restrict to the OpenVLA action-token slot [31744, 31999]
            mask = torch.full_like(next_logits, -float("inf"))
            mask[:, action_lo:action_hi] = 0.0
            next_logits = next_logits + mask
            next_id = next_logits.argmax(dim=-1, keepdim=True)
            gen_ids.append(next_id.item())
            next_embed = embed_layer(next_id)
            cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
            cur_attn = torch.cat(
                [cur_attn, torch.ones((1, 1), device=device, dtype=prefix_attn.dtype)],
                dim=1,
            )

        # OpenVLA's exact reverse map: bin_idx = clip(boundary - token_id - 1, 0, 254)
        token_ids = np.array(gen_ids, dtype=np.int64)
        discretized = self._action_vocab_boundary - token_ids
        bin_idx = np.clip(discretized - 1, 0, self._bin_centers.shape[0] - 1)
        normalized = self._bin_centers[bin_idx]
        unnormed = 0.5 * (normalized + 1.0) * (
            self._action_q99 - self._action_q01
        ) + self._action_q01
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
