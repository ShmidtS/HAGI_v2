"""Local contracts for the HAGI codec pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from hagi_v4.config import HAGIv4Config


@dataclass(frozen=True)
class CodecShapeConfig:
    hidden_size: int
    core_hidden_size: int
    vocab_size: int
    norm_eps: float
    pilot_spacing: int
    num_modalities: int
    msa_top_k: int
    use_whiteness_loss: bool

    @classmethod
    def from_hagi_config(cls, cfg: HAGIv4Config) -> CodecShapeConfig:
        m = cfg.model
        return cls(
            hidden_size=m.hidden_size,
            core_hidden_size=m.core_hidden_size,
            vocab_size=m.vocab_size,
            norm_eps=m.norm_eps,
            pilot_spacing=m.pilot_spacing,
            num_modalities=m.multimodal.num_modalities,
            msa_top_k=m.msa.top_k,
            use_whiteness_loss=m.gp2d.use_whiteness_loss,
        )


@dataclass(frozen=True)
class TurboDecodeConfig:
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
    gp2d: GP2DDecodeConfig
    msa: MSADecodeConfig

    @classmethod
    def from_hagi_config(cls, cfg: HAGIv4Config) -> TurboDecodeConfig:
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
            gp2d=GP2DDecodeConfig(
                window=m.gp2d.window,
                gate_init=m.gp2d.gate_init,
                use_multiscale=m.gp2d.use_multiscale,
                multiscale_windows=m.gp2d.multiscale_windows,
                multiscale_gate_inits=m.gp2d.multiscale_gate_inits,
                use_interleave=m.gp2d.use_interleave,
            ),
            msa=MSADecodeConfig(
                max_slots=m.msa.max_slots,
                slot_chunk_size=m.msa.slot_chunk_size,
                top_k=m.msa.top_k,
                routing_key_dim=m.msa.routing_key_dim,
                n_kv_heads=m.msa.n_kv_heads,
                head_dim=m.msa.head_dim,
                mla_compress_dim=m.msa.mla_compress_dim,
                mla_up_dim=m.msa.mla_up_dim,
                load_balance_weight=m.msa.load_balance_weight,
            ),
        )


@dataclass(frozen=True)
class TrainLossConfig:
    whiteness_weight: float
    parity_weight: float
    extrinsic_info_weight: float
    efficiency_weight: float
    msa_lb_weight: float
    rate_distortion_weight: float
    contrastive_weight: float

    @classmethod
    def from_hagi_config(cls, cfg: HAGIv4Config) -> TrainLossConfig:
        t = cfg.train
        return cls(
            whiteness_weight=t.w_whiteness,
            parity_weight=t.w_parity,
            extrinsic_info_weight=t.w_extrinsic_info,
            efficiency_weight=t.w_efficiency,
            msa_lb_weight=getattr(t, "w_msa_lb", 0.01),
            rate_distortion_weight=t.w_rate_distortion,
            contrastive_weight=t.w_contrastive if hasattr(t, "w_contrastive") else 0.0,
        )


@dataclass(frozen=True)
class InferenceShapeConfig:
    vocab_size: int

    @classmethod
    def from_hagi_config(cls, cfg: HAGIv4Config) -> InferenceShapeConfig:
        return cls(vocab_size=cfg.model.vocab_size)


InferenceConfig = InferenceShapeConfig


@dataclass(frozen=True)
class GP2DDecodeConfig:
    window: int
    gate_init: float
    use_multiscale: bool
    multiscale_windows: tuple[int, ...]
    multiscale_gate_inits: tuple[float, ...]
    use_interleave: bool


@dataclass(frozen=True)
class MSADecodeConfig:
    max_slots: int
    slot_chunk_size: int
    top_k: int
    routing_key_dim: int
    n_kv_heads: int
    head_dim: int
    mla_compress_dim: int
    mla_up_dim: int
    load_balance_weight: float


@dataclass
class DecodeState:
    kalman_p: torch.Tensor | None = None
    msa_feedback: torch.Tensor | None = None
    iteration: int = 0
    cache_active: bool = False


@dataclass
class DecodeResult:
    latent: torch.Tensor
    state: DecodeState
    side_info: dict


@dataclass
class SourceEncodeResult:
    source: torch.Tensor
    mask: torch.Tensor | None
    modality_ids: torch.Tensor | None
    cqi: torch.Tensor
    pre_bottleneck: torch.Tensor


@dataclass
class RateMatchResult:
    latent: torch.Tensor
    source: SourceEncodeResult
