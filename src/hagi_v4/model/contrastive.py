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

                h_a = h[b][ma].mean(dim=0)
                h_b = h[b][mb].mean(dim=0)

                z_a = self._proj(mod_a)(h_a)
                z_b = self._proj(mod_b)(h_b)

                negatives_a = []
                negatives_b = []
                for other_b in range(B):
                    if other_b == b:
                        continue
                    other_ma = mask_a[other_b]
                    other_mb = mask_b[other_b]
                    if other_ma.any():
                        negatives_a.append(self._proj(mod_a)(h[other_b][other_ma].mean(dim=0)))
                    if other_mb.any():
                        negatives_b.append(self._proj(mod_b)(h[other_b][other_mb].mean(dim=0)))

                if negatives_b:
                    all_b = torch.stack([z_b] + negatives_b)
                    logits_a = F.cosine_similarity(z_a.unsqueeze(0), all_b, dim=-1) / self.temperature
                    loss = loss - F.log_softmax(logits_a, dim=0)[0]
                    n_pairs += 1

                if negatives_a:
                    all_a = torch.stack([z_a] + negatives_a)
                    logits_b = F.cosine_similarity(z_b.unsqueeze(0), all_a, dim=-1) / self.temperature
                    loss = loss - F.log_softmax(logits_b, dim=0)[0]
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
