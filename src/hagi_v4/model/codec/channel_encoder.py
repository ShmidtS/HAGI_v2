"""HAGI V21 Channel Encoder — FEC + interleaving stage.

Owns: SparseParityEncoder (LDPC parity generation), BlockInterleaver
(burst-error spreading). Applies AWGN + erasure on the systematic part.

V21 refactor: extracted verbatim from the monolithic HAGIv4 class. No
behavioural changes; only the ownership boundary moved.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.codec_contracts import (
    ChannelEncodeResult,
    CodecShapeConfig,
    SourceEncodeResult,
)
from hagi_v4.model.interleaver import BlockInterleaver
from hagi_v4.model.sparse_parity import SparseParityEncoder


class ChannelEncoder(nn.Module):
    """Channel encoder: systematic -> parity + interleaved codeword.

    Pipeline (verbatim from HAGIv4 V18):
      systematic z_sys
        -> SparseParityEncoder (LDPC: z_sys -> parity)
        -> concat [systematic, parity]
        -> BlockInterleaver (QPP burst spreading)
        -> codeword
    """

    def __init__(
        self,
        cfg: HAGIv4Config,
        codec_shape: CodecShapeConfig,
        core_mask_embed: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        m = cfg.model
        self.codec_shape = codec_shape
        C = m.core_hidden_size
        self._C = C

        # core_mask_embed is owned by SourceEncoder but needed here for
        # erasure fill-in. Passed by reference to preserve weight sharing.
        if core_mask_embed is not None:
            self.core_mask_embed = core_mask_embed

        self.parity_encoder = SparseParityEncoder(
            n_vars=C,
            n_checks=codec_shape.n_parity_checks,
            edges_per_check=codec_shape.edges_per_check,
            seed=42,
            norm_eps=m.norm_eps,
        )

        self.interleaver = BlockInterleaver(
            block_len=m.attention.max_seq_len,
            mode=codec_shape.interleaver_mode,
        )

    def forward(self, encoded: SourceEncodeResult) -> ChannelEncodeResult:
        systematic = encoded.systematic
        parity = self.parity_encoder(systematic)
        codeword = torch.cat([systematic, parity], dim=-1)
        codeword = self.interleaver.interleave(codeword)
        return ChannelEncodeResult(codeword=codeword, systematic=systematic, parity=parity)

    def apply_erasure(
        self,
        encoded: ChannelEncodeResult,
        erasure_mask: torch.Tensor | None,
        awgn_sigma: float = 0.0,
    ) -> ChannelEncodeResult:
        if erasure_mask is None and (not self.training or awgn_sigma <= 0.0):
            return encoded
        codeword = self.interleaver.deinterleave(encoded.codeword).clone()
        systematic = codeword[..., : self._C]
        if self.training and awgn_sigma > 0.0:
            systematic.add_(awgn_sigma * torch.randn_like(systematic))
        if erasure_mask is not None:
            systematic[erasure_mask] = self.core_mask_embed.to(systematic.dtype)
        return ChannelEncodeResult(
            codeword=self.interleaver.interleave(codeword),
            systematic=encoded.systematic,
            parity=encoded.parity,
            interleaver_perm=encoded.interleaver_perm,
            erasure_mask=erasure_mask,
        )
