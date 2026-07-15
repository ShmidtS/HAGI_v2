"""Local contracts for the HAGI V8 codec pipeline.

V8 implements true Source-Channel Separation:
  - Source encoder produces systematic bits (compressed data)
  - Channel encoder adds parity bits (FEC, before channel)
  - Channel decoder recovers systematic via iterative BP (extrinsic-only)
  - Source decoder reconstructs from recovered systematic

All stage boundaries are explicit dataclasses for type safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from hagi_v4.config import HAGIv4Config


@dataclass(frozen=True)
class CodecShapeConfig:
    """Static shape parameters for the V8 codec pipeline."""

    hidden_size: int
    core_hidden_size: int
    vocab_size: int
    norm_eps: float
    pilot_spacing: int
    code_rate: float
    n_parity_checks: int
    edges_per_check: int
    interleaver_mode: str
    exit_threshold: float
    msa_top_k: int
    use_whiteness_loss: bool

    @classmethod
    def from_hagi_config(cls, cfg: "HAGIv4Config") -> CodecShapeConfig:
        m = cfg.model
        codec = m.codec
        C = m.core_hidden_size
        n_checks = codec.n_checks if codec.n_checks > 0 else max(1, int(C * (1.0 / codec.code_rate - 1.0)))
        return cls(
            hidden_size=m.hidden_size,
            core_hidden_size=C,
            vocab_size=m.vocab_size,
            norm_eps=m.norm_eps,
            pilot_spacing=m.pilot_spacing,
            code_rate=codec.code_rate,
            n_parity_checks=n_checks,
            edges_per_check=codec.edges_per_check,
            interleaver_mode=codec.interleaver_mode,
            exit_threshold=codec.exit_threshold,
            msa_top_k=m.msa.top_k,
            use_whiteness_loss=m.gp2d.use_whiteness_loss,
        )


@dataclass(frozen=True)
class TurboDecodeConfig:
    """Configuration for the iterative LDPC-style decoder."""

    num_iterations: int
    min_iterations: int
    convergence_threshold: float
    use_convergence_halt: bool
    tanh_scale: float
    reasoning_layers: int
    norm_eps: float
    attention_head_dim: int
    attention_max_seq_len: int
    freq_n_modes_t: int
    freq_n_modes_h: int
    freq_complex_rank: int
    msa: "MSADecodeConfig"

    @classmethod
    def from_hagi_config(cls, cfg: "HAGIv4Config") -> TurboDecodeConfig:
        m = cfg.model
        r = m.refinement
        return cls(
            num_iterations=r.num_iterations,
            min_iterations=r.min_iterations,
            convergence_threshold=r.convergence_threshold,
            use_convergence_halt=r.use_convergence_halt,
            tanh_scale=r.tanh_scale,
            reasoning_layers=m.reasoning_layers,
            norm_eps=m.norm_eps,
            attention_head_dim=m.attention.head_dim,
            attention_max_seq_len=m.attention.max_seq_len,
            freq_n_modes_t=m.freq_coding.n_modes_t,
            freq_n_modes_h=m.freq_coding.n_modes_h,
            freq_complex_rank=m.freq_coding.complex_rank,
            msa=MSADecodeConfig(
                max_slots=m.msa.max_slots,
                slot_chunk_size=m.msa.slot_chunk_size,
                top_k=m.msa.top_k,
                routing_key_dim=m.msa.routing_key_dim,
                n_kv_heads=m.msa.n_kv_heads,
                head_dim=m.msa.head_dim,
                mla_compress_dim=m.msa.mla_compress_dim,
                mla_up_dim=m.msa.mla_up_dim,
            ),
        )


@dataclass(frozen=True)
class TrainLossConfig:
    """Loss weights for the 3-level V8 loss hierarchy."""

    whiteness_weight: float
    parity_weight: float
    extrinsic_info_weight: float
    rate_distortion_weight: float
    contrastive_weight: float

    @classmethod
    def from_hagi_config(cls, cfg: "HAGIv4Config") -> TrainLossConfig:
        t = cfg.train
        return cls(
            whiteness_weight=t.w_whiteness,
            parity_weight=t.w_parity,
            extrinsic_info_weight=t.w_extrinsic_info,
            rate_distortion_weight=t.w_rate_distortion,
            contrastive_weight=t.w_contrastive,
        )


@dataclass(frozen=True)
class InferenceShapeConfig:
    vocab_size: int

    @classmethod
    def from_hagi_config(cls, cfg: "HAGIv4Config") -> InferenceShapeConfig:
        return cls(vocab_size=cfg.model.vocab_size)


InferenceConfig = InferenceShapeConfig


@dataclass(frozen=True)
class MSADecodeConfig:
    """HARQ buffer configuration (extrinsic-only storage)."""

    max_slots: int
    slot_chunk_size: int
    top_k: int
    routing_key_dim: int
    n_kv_heads: int
    head_dim: int
    mla_compress_dim: int
    mla_up_dim: int


@dataclass
class DecodeState:
    """Mutable state carried through the decoder and across cached blocks.

    V8: kalman_p tracks per-dimension uncertainty.
    harq_feedback stores serialized extrinsic deltas (not full states).
    """

    kalman_p: torch.Tensor | None = None
    harq_feedback: torch.Tensor | None = None
    iteration: int = 0
    cache_active: bool = False


@dataclass
class SourceEncodeResult:
    """Output of the source encoder stage."""

    systematic: torch.Tensor
    mask: torch.Tensor | None
    cqi: torch.Tensor
    pre_bottleneck: torch.Tensor


@dataclass
class ChannelEncodeResult:
    """Output of the channel encoder stage (FEC + rate matching)."""

    codeword: torch.Tensor
    systematic: torch.Tensor
    parity: torch.Tensor
    interleaver_perm: torch.Tensor | None = None


@dataclass
class DecodeResult:
    """Output of the channel decoder stage."""

    latent: torch.Tensor
    state: DecodeState
    side_info: dict


@dataclass
class RateMatchResult:
    """Intermediate result after rate matching."""

    latent: torch.Tensor
    source: SourceEncodeResult
