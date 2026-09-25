"""Arm S: a scratch model with the MERGED arm's architecture, not the baseline's.

The M2 audit established that the merged arm and the baseline are not two
parameterisations of one model: the merged checkpoint carries
``BlockRMSNorm`` gains shaped (3, 384) and **cannot be loaded into ``HAGI`` at
all** (it has to be reconstructed through ``MergedHAGI``), while the baseline
carries plain ``RMSNorm`` (1152,). Comparing them therefore compares
architecture, initialisation, and the 2x token budget at once.

To test merging specifically, the scratch arm has to share the merged arm's
architecture exactly. This module provides that: a scratch H=1152 HAGI whose
norms are per-block (3, 384), matching what ``MergedHAGI`` builds, so the
only difference from the merged model is where the weights came from.

Deliberately narrow: it does not merge, does not alter the merge code path,
and does not touch the production constructor. If the merged model later
needs this, it should live in ``hagi.model.merge``; for the experiment it
belongs with the experiment.
"""

from __future__ import annotations

import torch.nn as nn

from hagi.config import Config
from hagi.model.model import HAGI
from hagi.model.norms import BlockRMSNorm


class ScratchBlockNormHAGI(HAGI):
    """HAGI with per-expert block normalisation, trained from random init.

    Mirrors ``MergedHAGI``'s norm structure without any of its merge logic, so
    the scratch and merged arms differ only in initialisation and in whether
    the body weights came from pretrained experts.
    """

    def __init__(self, cfg: Config, *, n_blocks: int) -> None:
        super().__init__(cfg)
        if cfg.model.hidden_size % n_blocks:
            raise ValueError(
                f"hidden_size {cfg.model.hidden_size} must be divisible by n_blocks {n_blocks}"
            )
        block_dim = cfg.model.hidden_size // n_blocks
        for block in self.blocks:
            block.attn.attn_norm = BlockRMSNorm(
                n_blocks, block_dim, cfg.model.norm_eps
            )
            block.mixer.norm = BlockRMSNorm(n_blocks, block_dim, cfg.model.norm_eps)
        self.out_norm: nn.Module = BlockRMSNorm(
            n_blocks, block_dim, cfg.model.norm_eps
        )
