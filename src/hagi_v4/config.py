"""HAGI V4 configuration dataclasses.

V4 is a PLANE PREDICTION model — predicts entire text simultaneously through
iterative refinement. Key differences from V1/V3:
- Bidirectional attention (no causal mask)
- Masked CE training (not next-token)
- 2D geometric product (temporal convolution)
- Iterative refinement (4 iterations)
- 74M params target
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AlgebraConfig:
    """Cl(3,0,0) algebra layout within hidden state."""

    blade_count: int = 8
    grade_dims: tuple = (64, 96, 96, 64, 256)
    hidden_size: int = 576


@dataclass
class AttentionConfig:
    """Grouped-query attention with bidirectional support."""

    num_query_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 72
    rope_theta: float = 10000.0
    max_seq_len: int = 4096
    bidirectional: bool = True
    fp16_attention: bool = True
    fp32_rmsnorm: bool = True


@dataclass
class GP2DConfig:
    """2D Geometric Product — systematic parity channel code.

    V5: GP2D acts as a channel encoder — geometric product between
    adjacent positions generates parity bits for error correction.
    The decoder can check consistency via inverse GP.
    """

    window: int = 1
    gate_init: float = -2.0
    use_whiteness_loss: bool = True
    whiteness_weight: float = 0.01
    use_systematic_parity: bool = True
    parity_weight: float = 0.1


@dataclass
class RefinementConfig:
    """Iterative refinement loop — channel decoder (belief propagation).

    V5: extrinsic information separation prevents information recycling.
    Each iteration computes extrinsic = h_out - h_prior and passes only
    extrinsic forward. Convergence halt based on extrinsic norm.
    """

    num_iterations: int = 4
    min_iterations: int = 1
    use_adaptive_halt: bool = True
    halt_threshold: float = 0.9
    halt_threshold_start: float = 0.05
    halt_threshold_end: float = 0.001
    use_deep_supervision: bool = True
    deep_supervision_decay: float = 0.1
    deep_supervision_weight: float = 0.1
    use_adaptive_ds_weight: bool = True
    ds_ema_decay: float = 0.99
    use_entropy_adaptive_refinement: bool = True
    entropy_low_threshold: float = 0.01
    entropy_high_threshold: float = 0.1
    entropy_low_iterations: int = 2
    entropy_high_iterations: int = 6
    extrinsic_alpha: float = 1.0
    convergence_threshold: float = 0.01
    use_convergence_halt: bool = True


@dataclass
class MaskingConfig:
    """Adaptive erasure channel for V5 codec training.

    V5: mask ratio adapts to model confidence (capacity matching).
    mask_embed initialized as max-entropy vector (not zero) so the
    model receives a clear "erasure here" signal.
    """

    mask_ratio: float = 0.3
    mask_token_id: int = 49153
    use_span_masking: bool = True
    span_length: int = 3
    use_progressive: bool = True
    use_adaptive_erasure: bool = True
    mask_embed_init: str = "max_entropy"
    adaptation_rate: float = 0.01


@dataclass
class HRMConfig:
    """Hierarchical Recurrent Memory — spatial z_H + z_L planes."""

    h_state_dim: int = 256
    l_state_dim: int = 256
    h_stride: int = 4


@dataclass
class GDRConfig:
    """Grade-Decomposed Recurrence — per-grade momentum update."""

    scalar_momentum: float = 0.8
    vector_momentum: float = 0.5
    bivector_momentum: float = 0.0
    trivector_momentum: float = 0.0
    use_grade_router: bool = True
    grade_router_alpha: float = 0.01


@dataclass
class MSAConfig:
    """Memory-Augmented Attention — slot registry + local 2D routing."""

    max_slots: int = 4096
    slot_chunk_size: int = 4
    use_adaptive_chunk_size: bool = True
    chunk_size_low_entropy: int = 8
    chunk_size_high_entropy: int = 2
    top_k: int = 6
    routing_key_dim: int = 64
    n_kv_heads: int = 4
    head_dim: int = 72
    grade_dims: tuple = (64, 96, 96, 64, 256)
    mla_compress_dim: int = 128
    mla_up_dim: int = 288


@dataclass
class MoEConfig:
    """Mixture of Experts with Mixture-of-Depths skip."""

    num_experts: int = 4
    top_k: int = 1
    intermediate_size: int = 384
    use_mod_skip: bool = True
    alpha: float = 0.01
    use_grade_specialization: bool = True
    grade_specialization_weight: float = 0.01


@dataclass
class CASTConfig:
    """Coherence-Aware Spatial Temporal — geometric coherence regularizer."""

    use_coherence: bool = True
    coherence_gate_init: float = -5.0
    use_per_grade_coherence: bool = True
    scalar_dim: int = 64
    vector_dim: int = 96


@dataclass
class ModelConfig:
    """Full model architecture configuration."""

    vocab_size: int = 49154
    hidden_size: int = 576
    perception_layers: int = 2
    reasoning_layers: int = 7
    expression_layers: int = 2
    norm_eps: float = 1e-6
    bottleneck_ratio: float = 0.5
    core_hidden_size: int = 288
    algebra: AlgebraConfig = field(default_factory=AlgebraConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    gp2d: GP2DConfig = field(default_factory=GP2DConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    hrm: HRMConfig = field(default_factory=HRMConfig)
    gdr: GDRConfig = field(default_factory=GDRConfig)
    msa: MSAConfig = field(default_factory=MSAConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    cast: CASTConfig = field(default_factory=CASTConfig)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    max_steps: int = 150000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.5
    weight_decay: float = 0.1
    precision: str = "bf16"
    grad_accum_steps: int = 2
    batch_size: int = 10
    seq_len: int = 1024
    w_ce: float = 1.0
    w_moe_aux: float = 0.01
    w_gdr_router: float = 0.01
    w_coherence: float = 0.001
    w_ib: float = 0.01
    ib_beta: float = 1.0
    w_whiteness: float = 0.01
    w_grade_specialization: float = 0.01
    w_parity: float = 0.1
    w_extrinsic_info: float = 0.01
    w_efficiency: float = 0.001
    use_two_phase_schedule: bool = True
    two_phase_split: float = 0.5
    phase1_mask_ratio: float = 0.15
    phase2_mask_ratio: float = 0.35
    phase1_w_coherence: float = 0.0001
    phase2_w_coherence: float = 0.001
    phase1_gp2d_gate_init: float = -1.0
    phase2_gp2d_gate_init: float = -2.0
    log_grade_variance: bool = True
    grade_log_interval: int = 100
    # Distillation
    distill_enabled: bool = True
    distill_teacher: str = "HuggingFaceTB/SmolLM2-360M"
    distill_embed_teacher: str = "HuggingFaceTB/SmolLM2-135M"
    distill_alpha_start: float = 0.5
    distill_alpha_end: float = 0.3
    distill_temperature: float = 2.0
    distill_temp_start: float = 4.0
    distill_temp_end: float = 1.0
    distill_use_temp_anneal: bool = True
    distill_end_frac: float = 0.6
    distill_every: int = 2
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5000
    checkpoint_keep_last: int = 3
    # Sequential cycling (v1-style curriculum)
    sequential_cycles: int = 3
    curriculum_enabled: bool = True
    curriculum_stage2_start: int = 100000
    curriculum_order: list[str] = field(
        default_factory=lambda: [
            "tinystories",
            "python_instruct",
            "smoltalk",
            "wikipedia_en",
            "wikipedia_ru",
            "openwebmath",
            "oscar_ru",
            "slimpajama",
            "edu",
        ]
    )
    stage2_datasets: list[str] = field(default_factory=lambda: ["openwebmath", "edu", "slimpajama"])
    # Data format
    data_dtype: str = "auto"  # auto/uint16/uint32
    data_dir: str = "data"


@dataclass
class HAGIv4Config:
    """Top-level HAGI V4 configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _apply_dict(obj: object, data: dict) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply_dict(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | None = None, **overrides: object) -> HAGIv4Config:
    """Load config from YAML file, then apply keyword overrides."""
    import yaml

    cfg = HAGIv4Config()
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _apply_dict(cfg, data)
    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            obj: object = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
