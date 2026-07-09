"""Contrastive modality alignment — InfoNCE / Slepian-Wolf (V7).

Aligns modality embedding spaces via InfoNCE loss. This is distributed
source coding: learn the correlation between modalities so the decoder
can use side information efficiently.

InfoNCE: L = -log(exp(sim(z_i, z_j+)/tau) / sum exp(sim(z_i, z_k)/tau))
where z_i = text embedding, z_j+ = matching image embedding,
z_k = non-matching image embeddings.

Information theory:
  - Minimizing InfoNCE maximizes lower bound on I(text; image)
  - Higher I(text; image) -> more rate reduction from Slepian-Wolf
  - Contrastive alignment = learning the correlation structure
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAlignment(nn.Module):
    """Contrastive learning for modality alignment (InfoNCE)."""

    def __init__(self, hidden_size: int, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.text_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.image_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.audio_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss across modality pairs.

        Args:
            h: [B, T, H] -- hidden states (concatenated modalities)
            modality_ids: [B, T] -- 0=text, 1=image, 2=audio

        Returns:
            InfoNCE loss (scalar)
        """
        B = h.shape[0]
        loss = h.new_zeros(())
        n_pairs = 0

        pairs = [(0, 1), (0, 2), (1, 2)]

        for mod_a, mod_b in pairs:
            mask_a = modality_ids == mod_a
            mask_b = modality_ids == mod_b

            if not (mask_a.any() and mask_b.any()):
                continue

            for b in range(B):
                ma = mask_a[b]
                mb = mask_b[b]
                if not (ma.any() and mb.any()):
                    continue

                h_a = h[b][ma].mean(dim=0, keepdim=True)
                h_b = h[b][mb].mean(dim=0, keepdim=True)

                proj_a = self._proj(mod_a)(h_a)
                proj_b = self._proj(mod_b)(h_b)

                sim = F.cosine_similarity(proj_a, proj_b, dim=-1) / self.temperature
                sim_exp = sim.exp()
                loss = loss - sim_exp.log() / (sim_exp + 1e-8)
                n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs

        return loss

    def _proj(self, modality: int) -> nn.Module:
        if modality == 0:
            return self.text_proj
        elif modality == 1:
            return self.image_proj
        else:
            return self.audio_proj
