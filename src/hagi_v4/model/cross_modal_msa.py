"""Cross-modal MSA — Wyner-Ziv side information (V7).

Each modality has its own slot registry. During decoding:
  - Text decoder reads from image slots (Wyner-Ziv side info)
  - Image decoder reads from text slots (cross-modal refinement)
  - Audio decoder reads from both text and image slots

This implements distributed source coding: each modality's decoder
uses other modalities as side information, reducing required rate.

Information theory:
  - Per-modality slot registries = separate codebooks per source
  - Cross-modal read = Wyner-Ziv coding (side info at decoder only)
  - Self-modal read = Slepian-Wolf coding (side info at encoder+decoder)
  - Rate reduction: R(X|Y) = H(X) - I(X;Y) for cross-modal decoding
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hagi_v4.model.codec_contracts import MSADecodeConfig
from hagi_v4.model.msa import MSAModule


class CrossModalMSA(nn.Module):
    """MSA with modality-specific slot registries for Wyner-Ziv coding."""

    def __init__(self, cfg: MSADecodeConfig, hidden_size: int, num_modalities: int = 3) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.msa_modules = nn.ModuleList([MSAModule(cfg, hidden_size) for _ in range(num_modalities)])

    def write(self, h: torch.Tensor, modality_ids: torch.Tensor) -> None:
        for mod_id in range(self.num_modalities):
            mask = modality_ids == mod_id
            if mask.any():
                for b in range(h.shape[0]):
                    mask_b = modality_ids[b] == mod_id
                    h_mod = h[b][mask_b].unsqueeze(0)
                    self.msa_modules[mod_id].write(h_mod)

    def read_cross(
        self,
        h: torch.Tensor,
        modality_ids: torch.Tensor,
        target_modality: int,
        top_k: int = 6,
    ) -> torch.Tensor:
        side_info = h.new_zeros_like(h)

        for b in range(h.shape[0]):
            target_mask = modality_ids[b] == target_modality
            if not target_mask.any():
                continue

            h_target = h[b][target_mask].unsqueeze(0)
            idx = torch.where(target_mask)[0]

            for other_mod in range(self.num_modalities):
                if other_mod == target_modality:
                    continue
                retrieved, _lb = self.msa_modules[other_mod].read(h_target, top_k)
                n = min(idx.shape[0], retrieved.shape[1])
                side_info[b, idx[:n]] += retrieved.squeeze(0)[:n]

        return side_info

    def read_self(
        self,
        h: torch.Tensor,
        modality_ids: torch.Tensor,
        modality: int,
        top_k: int = 6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = modality_ids == modality
        if not mask.any():
            return h.new_zeros_like(h), h.new_zeros(())

        h_mod = h[mask].unsqueeze(0)
        return self.msa_modules[modality].read(h_mod, top_k)

    def clear(self) -> None:
        for msa in self.msa_modules:
            msa.clear()
