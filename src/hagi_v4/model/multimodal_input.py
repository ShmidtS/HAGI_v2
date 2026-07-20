"""Multimodal source encoders + modality type embedding (V7).

Each modality has its own source encoder projecting to H:
  - Text: existing nn.Embedding (token IDs -> H)
  - Image: ViT-style patch embedding (RGB patches -> H)
  - Audio: spectrogram frame embedding (mel frames -> H)

Modality type embedding (CDMA spreading code) separates modalities
in the shared latent space. Max-entropy mask embeds per modality
serve as explicit erasure indicators (BEC analog).

Information theory:
  - Separate source encoders = optimal source coding per modality
  - Modality type embedding = CDMA orthogonal codes in shared latent
  - Sequence concatenation = OFDM symbol (multiple subcarriers)
  - Max-entropy mask_embed = explicit erasure indicator
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hagi_v4.config import HAGIv4Config


class MultimodalInput(nn.Module):
    """Multimodal source encoders + modality type embedding.

    Each modality has its own source encoder that projects to H.
    Modality type embedding (OFDM subcarrier ID) distinguishes modalities.
    """

    def __init__(self, cfg: HAGIv4Config, text_encoder: nn.Module | None = None) -> None:
        super().__init__()
        m = cfg.model
        H = m.hidden_size
        self.H = H
        mm = m.multimodal

        # V12: share the text source encoder with the main model when one
        # is provided. This removes the previous standalone ``nn.Embedding(V, H)``
        # which dominated the parameter budget at large vocabularies (2.34B
        # params for V=262146, H=8920). The text path now uses the same
        # factorized ``V*r + r*H`` source encoder as the rest of the model —
        # one source codebook per model, not per modality. When no encoder
        # is passed (legacy path), fall back to a factorized ConvEmbedding
        # sized by ``embeddings.factor_rank`` instead of dense ``V*H``.
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
                init=m.embeddings.init,
            )
            self._text_shared = False

        self.image_patch_size = m.image.patch_size
        self.image_embed = nn.Linear(m.image.input_channels * self.image_patch_size**2, H, bias=False)

        self.audio_n_mels = m.audio.n_mels
        self.audio_embed = nn.Linear(self.audio_n_mels, H, bias=False)

        self.num_modalities = mm.num_modalities
        self.modality_embeds = nn.Parameter(torch.zeros(mm.num_modalities, H))
        nn.init.normal_(self.modality_embeds, std=mm.modality_embed_std)

        self.mask_embeds = nn.Parameter(torch.zeros(mm.num_modalities, H))
        for i in range(mm.num_modalities):
            self.mask_embeds.data[i] = torch.ones(H) / (H**0.5)

        pos_std = 1.0 / (H**0.5)
        self.image_pos_embed = nn.Parameter(torch.zeros(m.image.max_image_patches, H))
        nn.init.normal_(self.image_pos_embed, std=pos_std)

        self.audio_pos_embed = nn.Parameter(torch.zeros(m.audio.max_audio_frames, H))
        nn.init.normal_(self.audio_pos_embed, std=pos_std)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.text_embed(input_ids)
        h = h + self.modality_embeds[0]
        return h

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
        return h, T_i

    def encode_audio(self, spectrograms: torch.Tensor) -> tuple[torch.Tensor, int]:
        B, n_mels, T_frames = spectrograms.shape
        frames = spectrograms.transpose(1, 2)
        h = self.audio_embed(frames)
        h = h + self.audio_pos_embed[:T_frames].unsqueeze(0)
        h = h + self.modality_embeds[2]
        return h, T_frames

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        spectrograms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
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

        h = torch.cat(parts, dim=1)
        modality_ids = torch.cat(mod_ids, dim=1)

        lengths = {
            "text": input_ids.shape[1] if input_ids is not None else 0,
            "image": T_i,
            "audio": T_a,
        }
        return h, modality_ids, lengths

    def get_mask_embed(self, modality_id: int) -> torch.Tensor:
        return self.mask_embeds[modality_id]
