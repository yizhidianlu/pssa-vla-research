"""PSSAVLAv2 — production training wrapper around OpenVLA-7B.

Architecture (Path A from plan):
    [BOS, lang_embeds, PSE_entity_embeds, image_embeds, action_query] → LLM → action tokens

Training scope:
- Frozen:    vision_backbone, projector, original action token decoder
- Trainable: PSE-Tok module, LoRA adapters on LLM (q/v projections), input
             projection that maps PSE entity features into LLM hidden dim.

Loss:
    L_action  : standard CE on discretized action tokens (OpenVLA's native loss)
    L_xtc     : cross-time consistency on entity features (predicted-effect residual)
    L_total   = L_action + lambda_xtc * L_xtc
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Forward-declare types via TYPE_CHECKING for clean imports
@dataclass
class PSSATrainingOutput:
    loss: torch.Tensor
    loss_action: torch.Tensor
    loss_xtc: torch.Tensor
    n_entity_tokens: int


class PSEEntityEncoder(nn.Module):
    """Encodes (B, T0, 3, H, W) initial frames + (B, T0, N, H, W) masks into
    (B, N, hidden_dim) persistent entity tokens.

    Uses a small CNN over masked regions of each entity. Each entity slot's
    feature is the temporal-average of its T0 mask-pooled CNN features.
    """

    def __init__(self, n_entities: int = 8, hidden_dim: int = 4096,
                 cnn_dim: int = 256) -> None:
        super().__init__()
        self.n_entities = n_entities
        self.hidden_dim = hidden_dim
        # Tiny CNN — frozen vision_backbone is the heavy lifter elsewhere
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
        """
        rgb_init:   (B, T0, 3, H, W)
        masks_init: (B, T0, N, H, W)
        returns:    (B, N, hidden_dim) PSE entity tokens
        """
        B, T0, _, H, W = rgb_init.shape
        N = self.n_entities
        # CNN over each frame: (B*T0, cnn_dim, h, w)
        feats = self.cnn(rgb_init.flatten(0, 1))
        _, C, h, w = feats.shape
        # Down-sample masks to feature spatial resolution
        masks_small = F.adaptive_avg_pool2d(masks_init.flatten(0, 1), (h, w))
        # masks_small: (B*T0, N, h, w)
        # Mask-pool features per entity: (B*T0, N, C)
        feats_per_entity = torch.einsum("bchw,bnhw->bnc", feats, masks_small)
        denom = masks_small.flatten(2).sum(-1, keepdim=True).clamp_min(1.0)
        feats_per_entity = feats_per_entity / denom
        # Average over T0 frames: (B, N, C)
        feats_per_entity = feats_per_entity.view(B, T0, N, C).mean(dim=1)
        # Add entity id embedding
        ids = torch.arange(N, device=feats_per_entity.device)
        id_e = self.id_embed(ids).unsqueeze(0).expand(B, -1, -1)
        # Project to LLM hidden dim
        out = self.proj(torch.cat([feats_per_entity, id_e], dim=-1))
        return out  # (B, N, hidden_dim)


class XTCLoss(nn.Module):
    """Cross-time consistency loss on PSE entity features.

    For phase-2 v0, we use a simple smoothness penalty:
        L_xtc = mean((f_t - f_{t-1})^2)

    This will be upgraded to predicted-effect residual once the
    Δf_pred(a_{t-1}) module is wired in.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, entity_seq: torch.Tensor) -> torch.Tensor:
        # entity_seq: (B, T, N, D)
        if entity_seq.size(1) < 2:
            return entity_seq.new_zeros(())
        diff = entity_seq[:, 1:] - entity_seq[:, :-1]
        return diff.pow(2).mean()


class PSSAVLAv2(nn.Module):
    """Wraps a frozen OpenVLA-7B + adds PSE entity prefix + LoRA on LLM.

    The OpenVLA model is loaded externally and passed in. We add:
    - pse_encoder: PSEEntityEncoder
    - lora adapters via peft (added externally; this class accepts a PEFT-
      wrapped model)
    - xtc_loss: XTCLoss
    """

    def __init__(
        self,
        backbone,                                # OpenVLAForActionPrediction (PEFT-wrapped)
        pse_encoder: PSEEntityEncoder | None = None,
        processor=None,                          # AutoProcessor for tokenizer
        lambda_xtc: float = 0.1,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # Find LLM hidden dim — may be wrapped by PEFT
        cfg = getattr(backbone, "config", None) or backbone.base_model.config
        hidden_dim = getattr(cfg, "hidden_size", 4096)
        self.pse_encoder = pse_encoder or PSEEntityEncoder(hidden_dim=hidden_dim)
        self.xtc_loss = XTCLoss()
        self.lambda_xtc = lambda_xtc
        self.processor = processor
        # Continuous action regression head (small MLP on LLM final hidden state)
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, action_dim),
        )

    def training_step(self, batch: dict) -> PSSATrainingOutput:
        """One training step.

        batch:
            rgb_init   (B, T0, 3, H, W)
            rgb_seq    (B, T, 3, H, W)
            actions    (B, T, 7)
            masks_init (B, T0, N, H, W)
            language   list[str] of length B

        returns: PSSATrainingOutput
        """
        rgb_init = batch["rgb_init"]
        rgb_seq = batch["rgb_seq"]
        actions = batch["actions"]
        masks_init = batch["masks_init"]
        language = batch["language"]

        B, T = rgb_seq.shape[:2]

        # 1) Encode persistent entities from initial frames
        pse_tokens = self.pse_encoder(rgb_init, masks_init)            # (B, N, D)

        # 2) For each step in the window, build OpenVLA-style input + PSE prefix
        #    and accumulate action loss. We loop because OpenVLA's input
        #    construction is per-frame; this is a phase-2 v0 trade-off.
        #    A future v1 will batch the temporal axis as a single LLM call.
        device = rgb_seq.device
        loss_action = rgb_seq.new_zeros(())
        entity_seq = []
        for t in range(T):
            # Re-encode entity features from current frame for XTC.
            # Keep gradient flowing through pse_encoder so XTC loss trains it
            # (in v0, XTC is the only training signal since _step_action_logits
            # raises NotImplementedError).
            ent_t = self.pse_encoder(
                rgb_seq[:, t:t+1], masks_init[:, :1].expand(-1, 1, -1, -1, -1)
            )
            entity_seq.append(ent_t)
            # NB: real OpenVLA forward with PSE prefix injection requires a
            # PEFT/transformers integration that's beyond this v0 scaffold.
            # For now we run the action head on a coarse readout to get a
            # gradient flowing through pse_encoder + the LLM.
            try:
                # Best-effort: use backbone's predict_action via the LLM
                # input_embeds path. Subclasses can override this.
                pred_action = self._step_action_logits(
                    pse_tokens, ent_t, rgb_seq[:, t], language
                )
                loss_action = loss_action + F.mse_loss(pred_action, actions[:, t])
            except NotImplementedError:
                # In the v0 scaffold the backbone hook may not be implemented;
                # we still accumulate XTC loss which trains pse_encoder.
                pass

        loss_action = loss_action / max(T, 1)

        # 3) XTC loss on per-step entity features
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

    def _step_action_logits(
        self,
        pse_tokens: torch.Tensor,        # (B, N, D)
        ent_t: torch.Tensor,             # (B, 1, N, D) — unused
        rgb_t: torch.Tensor,             # (B, 3, H, W), values in [0, 1]
        language: list[str],             # B-list
    ) -> torch.Tensor:
        """v2: full DinoSigLIP vision + text + PSE prefix → LLM → action_head.

        Sequence: [text_embeds | image_embeds (256) | pse_tokens (N)] → LLM →
        last hidden state → action_head → (B, action_dim).
        """
        if self.processor is None:
            raise RuntimeError("PSSAVLAv2 needs a processor for v2")
        device = rgb_t.device
        B = rgb_t.shape[0]

        # 1) RGB tensor → PIL list (processor expects PIL, handles 128→224 resize)
        from PIL import Image
        import numpy as np
        rgb_np = (rgb_t.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu()
                  .float().numpy() * 255).astype(np.uint8)
        pil_imgs = [Image.fromarray(rgb_np[i]) for i in range(B)]

        # 2) Prompts
        prompts = [f"In: what action to {ln.strip().lower()}? Out:" for ln in language]

        # 3) Processor produces input_ids + attention_mask + pixel_values (B,6,224,224)
        inputs = self.processor(
            text=prompts, images=pil_imgs, return_tensors="pt", padding=True,
        )
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"].to(device)
        attn_text = inputs["attention_mask"].to(device)

        # 4) Vision: backbone frozen — but we keep gradients flowing for LLM
        #    (vision_backbone has requires_grad_(False) so storage is light)
        inner = self._unwrap()
        with torch.no_grad():  # vision is frozen; saves activation memory
            img_feats = inner.vision_backbone(pixel_values)         # (B, 256, 2176)
            image_embeds = inner.projector(img_feats)               # (B, 256, D)
        # Detach to be explicit (projector is also frozen)
        image_embeds = image_embeds.detach()

        # 5) Text embeds (input_embeddings is frozen unless LoRA wraps it)
        text_embeds = self._get_input_embeddings()(input_ids)       # (B, L, D)

        # 6) PSE prefix (cast to LLM dtype)
        pse_embeds = pse_tokens.to(text_embeds.dtype)

        # 7) Concat: [text | image | PSE]
        inputs_embeds = torch.cat([text_embeds, image_embeds, pse_embeds], dim=1)
        attn_img = torch.ones(image_embeds.shape[:2], device=device, dtype=attn_text.dtype)
        attn_pse = torch.ones(pse_embeds.shape[:2], device=device, dtype=attn_text.dtype)
        attn = torch.cat([attn_text, attn_img, attn_pse], dim=1)

        # 8) LLM forward
        llm = self._get_llm()
        out = llm(inputs_embeds=inputs_embeds, attention_mask=attn)
        last_hidden = out.last_hidden_state[:, -1]                  # (B, D)
        return self.action_head(last_hidden.float())                # (B, 7)

    # ---- module-access helpers ---------------------------------------------
    def _unwrap(self):
        """Return inner OpenVLAForActionPrediction (PEFT may wrap)."""
        m = self.backbone
        for attr in ("base_model", "model"):
            if hasattr(m, attr):
                inner = getattr(m, attr)
                if hasattr(inner, "vision_backbone"):
                    return inner
        return m

    def _get_input_embeddings(self):
        return self._unwrap().get_input_embeddings()

    def _get_llm(self):
        """Return the inner LLM transformer for inputs_embeds forward."""
        lm = getattr(self._unwrap(), "language_model", None)
        if lm is None:
            raise AttributeError("backbone has no .language_model")
        # Prefer the inner transformer (.model) over the LM-with-head wrapper
        inner = getattr(lm, "model", None)
        if inner is not None and not isinstance(inner, type):
            return inner
        return lm
