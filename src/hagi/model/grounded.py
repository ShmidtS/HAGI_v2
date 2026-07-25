"""Grounded infomax — VICReg + InfoNCE auxiliary multimodal grounding.

Replaces the V25 weak lag-1 perception-autocorrelation. Two off-path
auxiliaries stabilize and align the multimodal joint embedding:

  * VICReg — variance (Hinge keeping per-dim std >= gamma, preventing collapse),
    invariance (MSE stability across the batch's pooled embedding), covariance
    (off-diagonal -> 0, Barlow redundancy reduction). Stable without posterior
    sampling, unlike the IB.
  * InfoNCE — maximizes a lower bound on cross-modal mutual information
    I(text; image). This is Slepian-Wolf distributed source coding: learn the
    correlation between modalities so the decoder can exploit side information.

Both are off-path. Each modality's pooled embedding (mean over its tokens) is
projected into a shared d-dimensional space; VICReg regularizes each
projection, InfoNCE aligns the cross-modal pairs across the batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hagi.config import GroundedInfomaxConfig


class GroundedInfomax(nn.Module):
    """Per-modality projectors + VICReg/InfoNCE losses.

    Args:
        hidden_size: H.
        cfg: grounded-infomax config.
        num_modalities: number of modalities with projectors (text/image/audio).
    """

    def __init__(self, hidden_size: int, cfg: GroundedInfomaxConfig, num_modalities: int = 3) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.projectors = nn.ModuleList(nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_modalities))

    @staticmethod
    def _pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool ``h`` over the masked positions per batch row.

        Args:
            h: ``[B, T, H]``.
            mask: ``[B, T]`` boolean.

        Returns:
            ``[B, H]``.
        """
        m = mask.unsqueeze(-1).to(h.dtype)
        summed = (h * m).sum(dim=1)
        count = m.sum(dim=1).clamp_min(1.0)
        return summed / count

    def _vicreg(self, z: torch.Tensor, mask_valid: torch.Tensor) -> torch.Tensor:
        """VICReg on one modality's projected pooled embedding.

        Args:
            z: ``[B, H]`` projected pooled embeddings.
            mask_valid: ``[B]`` rows that actually contain this modality.

        Returns:
            scalar VICReg loss (0 if no valid rows).
        """
        if not mask_valid.any():
            return z.new_zeros(())
        cfg = self.cfg
        z = z[mask_valid]
        z = z - z.mean(dim=0, keepdim=True)
        # Variance hinge: keep per-dim std >= gamma.
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-8)
        var_loss = F.relu(cfg.vicreg_gamma - std).mean()
        # Covariance decorrelation (off-diagonal -> 0).
        n = z.shape[0]
        if n > 1:
            cov = (z.t() @ z) / (n - 1)
            cov_loss = (cov.fill_diagonal_(0.0) ** 2).sum() / z.shape[1]
        else:
            cov_loss = z.new_zeros(())
        # Invariance: zero here (cross-modal alignment is InfoNCE's job; the
        # pooled embedding has no augmentation to be invariant to in the
        # forward path). Retained in the config for future augmentation work.
        inv_loss = z.new_zeros(())
        return cfg.vicreg_var_weight * var_loss + cfg.vicreg_inv_weight * inv_loss + cfg.vicreg_cov_weight * cov_loss

    def _infonce_pair(self, z_a: torch.Tensor, z_b: torch.Tensor, valid_a: torch.Tensor, valid_b: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE between two modalities over the shared valid batch.

        Args:
            z_a, z_b: ``[B, H]`` projected pooled embeddings.
            valid_a, valid_b: ``[B]`` validity masks; a row is a positive pair
                only where BOTH are valid.

        Returns:
            scalar InfoNCE loss (0 if < 2 valid pairs).
        """
        both = valid_a & valid_b
        if int(both.sum()) < 2:
            return z_a.new_zeros(())
        za = F.normalize(z_a[both], dim=-1)
        zb = F.normalize(z_b[both], dim=-1)
        logits = za @ zb.t() / self.cfg.infonce_temperature
        labels = torch.arange(za.shape[0], device=za.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def forward(
        self,
        h: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute VICReg + InfoNCE over the pooled per-modality embeddings.

        Args:
            h: ``[B, T, H]`` hidden states (multimodal prefix + text).
            modality_ids: ``[B, T]`` long (0=text, 1=image, 2=audio).

        Returns:
            ``(vicreg_loss, infonce_loss)``; each None if no modality present.
        """
        b = h.shape[0]
        pooled = []
        valid = []
        for m_idx in range(self.num_modalities):
            mask = modality_ids == m_idx  # [B, T]
            row_valid = mask.any(dim=1)  # [B]
            pooled_m = self._pool(h, mask)
            pooled.append(self.projectors[m_idx](pooled_m))
            valid.append(row_valid)

        vicreg = h.new_zeros(())
        any_vicreg = False
        for m_idx in range(self.num_modalities):
            if valid[m_idx].any():
                vicreg = vicreg + self._vicreg(pooled[m_idx], valid[m_idx])
                any_vicreg = True
        vicreg = vicreg if any_vicreg else None

        infonce = h.new_zeros(())
        any_infonce = False
        for a in range(self.num_modalities):
            for bb in range(a + 1, self.num_modalities):
                if valid[a].any() and valid[bb].any():
                    pair = self._infonce_pair(pooled[a], pooled[bb], valid[a], valid[bb])
                    infonce = infonce + pair
                    any_infonce = any_infonce or (pair.item() > 0)
        infonce = infonce if any_infonce else None
        del b
        return vicreg, infonce
