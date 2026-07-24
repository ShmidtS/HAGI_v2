"""MultimodalFusion — separable source coding + shared subspace.

Separate source coding theorem -> one factorized source encoder per modality
(text reuses the shared ``ConvEmbedding``; image = ViT patch Linear; audio =
mel Linear). An explicit cross-modal alignment via a shared subspace, with the
residual kept modality-specific and gated by a learned inverse-variance head
(uncertain modality contributes less of its specific residual; the shared
subspace is always kept).

All parameters here are FP32 source-codebook / 1D heads: they are routed to
AdamW (the names ``image_embed`` / ``audio_embed`` / ``shared_down`` /
``shared_up`` / ``log_var`` / ``embed`` are in ``_MUON_EXCLUDE``). Nothing in
this module is ternary — the source codebook is rate-critical.

Forward: ``forward(input_ids, images, spectrograms) -> (h, modality_ids, lengths)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hagi_v4.config import Config


def _inv_var_gate(h: torch.Tensor, log_var_head: nn.Module) -> torch.Tensor:
    """Learned inverse-variance gate in (0, 1).

    ``sigmoid(-logit)`` keeps the modality-specific residual when the modality
    is confident (low variance -> gate -> 1) and suppresses it when uncertain.
    Bounded, differentiable, and never rescales the shared subspace.
    """
    logit = log_var_head(h).squeeze(-1)
    return torch.sigmoid(-logit.float()).to(h.dtype)


class _InvVarHead(nn.Module):
    """Per-position scalar log-variance estimator.

    Zero-bias init keeps the gate at ``sigmoid(0)=0.5`` at start (neutral).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.log_var = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.log_var.weight, std=1.0 / max(hidden_size, 1))
        nn.init.zeros_(self.log_var.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.log_var(h)


class MultimodalFusion(nn.Module):
    """Per-modality source encoders + shared/specific subspace + inv-var gating.

    Args:
        cfg: top-level config.
        text_encoder: the model's shared ``ConvEmbedding`` when provided (one
            source codebook per model). Falls back to a standalone
            ``ConvEmbedding`` sized by ``embeddings.factor_rank``.
    """

    NUM_MODALITIES = 3

    def __init__(self, cfg: Config, text_encoder: nn.Module | None = None) -> None:
        super().__init__()
        m = cfg.model
        H = m.hidden_size
        mm = m.multimodal
        self.H = H
        self.image_patch_size = mm.image_patch_size
        self.audio_n_mels = mm.audio_mel_bins

        if text_encoder is not None:
            self.text_embed = text_encoder
            self._text_shared = True
        else:
            from hagi_v4.model.conv_embedding import ConvEmbedding

            self.text_embed = ConvEmbedding(
                vocab_size=m.vocab_size,
                hidden_size=H,
                factor_rank=m.embeddings.factor_rank,
                kernel_size=m.embeddings.kernel_size,
                norm_eps=m.norm_eps,
            )
            self._text_shared = False

        # ONE shared cross-modal projection pair across all modalities. The
        # ``shared_up`` is ZERO-init so z_shared starts at 0 (the specific
        # residual dominates at init, shared subspace learned during training).
        r_shared = max(8, H // 4)
        self.r_shared = r_shared
        self.shared_down = nn.Linear(H, r_shared, bias=False)
        self.shared_up = nn.Linear(r_shared, H, bias=False)
        nn.init.normal_(self.shared_down.weight, std=1.0 / max(H, 1))
        nn.init.zeros_(self.shared_up.weight)

        self.image_embed = nn.Linear(mm.image_channels * self.image_patch_size**2, H, bias=False)
        self.audio_embed = nn.Linear(self.audio_n_mels, H, bias=False)

        self.text_unc = _InvVarHead(H)
        self.image_unc = _InvVarHead(H)
        self.audio_unc = _InvVarHead(H)

        self.modality_embeds = nn.Parameter(torch.zeros(self.NUM_MODALITIES, H))
        nn.init.normal_(self.modality_embeds, std=mm.modality_embed_std)

        pos_std = 1.0 / (H**0.5)
        self.image_pos_embed = nn.Parameter(torch.zeros(mm.max_image_patches, H))
        nn.init.normal_(self.image_pos_embed, std=pos_std)
        self.audio_pos_embed = nn.Parameter(torch.zeros(mm.max_audio_frames, H))
        nn.init.normal_(self.audio_pos_embed, std=pos_std)

    def _fuse(self, h: torch.Tensor, unc: nn.Module) -> torch.Tensor:
        """Shared/specific split + inverse-variance gating of the specific residual."""
        z_shared = self.shared_up(self.shared_down(h))
        z_specific = h - z_shared
        gate = _inv_var_gate(h, unc).unsqueeze(-1)
        return z_shared + gate * z_specific

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.text_embed(input_ids)
        h = h + self.modality_embeds[0]
        return self._fuse(h, self.text_unc)

    def encode_image(self, images: torch.Tensor) -> tuple[torch.Tensor, int]:
        B, C, H_img, W_img = images.shape
        p = self.image_patch_size
        n_h, n_w = H_img // p, W_img // p
        T_i = n_h * n_w
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, T_i, C * p * p)
        h = self.image_embed(patches)
        h = h + self.image_pos_embed[:T_i].unsqueeze(0)
        h = h + self.modality_embeds[1]
        return self._fuse(h, self.image_unc), T_i

    def encode_audio(self, spectrograms: torch.Tensor) -> tuple[torch.Tensor, int]:
        B, _n_mels, T_frames = spectrograms.shape
        frames = spectrograms.transpose(1, 2)
        h = self.audio_embed(frames)
        h = h + self.audio_pos_embed[:T_frames].unsqueeze(0)
        h = h + self.modality_embeds[2]
        return self._fuse(h, self.audio_unc), T_frames

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        spectrograms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Early fusion: encode each modality, then concatenate along the sequence axis.

        Returns:
            ``(h, modality_ids, lengths)`` where ``h`` is ``[B, T_total, H]`` and
            ``modality_ids`` is ``[B, T_total]`` long (0=text, 1=image, 2=audio).
        """
        parts: list[torch.Tensor] = []
        mod_ids: list[torch.Tensor] = []
        T_i = 0
        T_a = 0

        if input_ids is not None:
            h_t = self.encode_text(input_ids)
            parts.append(h_t)
            mod_ids.append(torch.zeros(h_t.shape[0], h_t.shape[1], dtype=torch.long, device=h_t.device))
        if images is not None:
            h_i, T_i = self.encode_image(images)
            parts.append(h_i)
            mod_ids.append(torch.ones(h_i.shape[0], h_i.shape[1], dtype=torch.long, device=h_i.device))
        if spectrograms is not None:
            h_a, T_a = self.encode_audio(spectrograms)
            parts.append(h_a)
            mod_ids.append(torch.full((h_a.shape[0], h_a.shape[1]), 2, dtype=torch.long, device=h_a.device))

        if not parts:
            raise ValueError("MultimodalFusion.forward requires at least one modality")

        h = torch.cat(parts, dim=1)
        modality_ids = torch.cat(mod_ids, dim=1)
        lengths = {
            "text": input_ids.shape[1] if input_ids is not None else 0,
            "image": T_i,
            "audio": T_a,
        }
        return h, modality_ids, lengths
