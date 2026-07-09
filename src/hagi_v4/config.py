"""HAGI V4 configuration dataclasses.

V4 is a PLANE PREDICTION model — predicts entire text simultaneously through
iterative refinement. Key differences from V1/V3:
- Bidirectional attention (no causal mask)
- Masked CE training (not next-token)
- 2D geometric product (temporal convolution)
- Iterative refinement (4 iterations)

Auto-configure: set `target_params` in YAML, all sizes computed automatically.
Optimal ratios derived from evolutionary parameter search (ratio_search.py):
  - C/H = 0.50 (bottleneck compression)
  - layers: balanced 2/2/2 (scales with depth_ratio for larger models)
  - grade split: [0.125, 0.1875, 0.1875, 0.125, 0.375]
  - moe_int/C = 1.5
  - q_heads * head_dim = H, kv_heads = q_heads / 2
  - n_shared_bases = 2
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

    V6: Multi-scale GP2D with interleaving (LDPC-like). Multiple window
    sizes provide parity at different frequency bands.
    """

    window: int = 1
    gate_init: float = -2.0
    use_whiteness_loss: bool = True
    whiteness_weight: float = 0.01
    use_systematic_parity: bool = True
    parity_weight: float = 0.1
    use_multiscale: bool = True
    multiscale_windows: tuple = (1, 4, 16)
    multiscale_gate_inits: tuple = (-2.0, -3.0, -4.0)
    use_interleave: bool = True


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
    deep_supervision_decay: float = 0.5
    deep_supervision_weight: float = 0.1
    use_adaptive_ds_weight: bool = True
    ds_ema_decay: float = 0.99
    use_entropy_adaptive_refinement: bool = True
    entropy_low_threshold: float = 0.01
    entropy_high_threshold: float = 0.1
    entropy_low_iterations: int = 2
    entropy_high_iterations: int = 6
    extrinsic_alpha: float = 0.5
    convergence_threshold: float = 0.01
    use_convergence_halt: bool = True
    use_gradient_checkpointing: bool = True


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
    router_noise: float = 0.01


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
    load_balance_weight: float = 0.01


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
    n_shared_bases: int = 0
    router_noise: float = 0.01
    router_init_std: float = 0.01


@dataclass
class CASTConfig:
    """Coherence-Aware Spatial Temporal — geometric coherence regularizer."""

    use_coherence: bool = True
    coherence_gate_init: float = -5.0
    use_per_grade_coherence: bool = True
    scalar_dim: int = 64
    vector_dim: int = 96


@dataclass
class VariationalBottleneckConfig:
    """Variational Information Bottleneck — optimal source coding (V6).

    Replaces deterministic linear bottleneck with stochastic encoding:
    z = mu + eps * sigma, where KL(q(z|x)||N(0,I)) controls compression.
    """

    enabled: bool = True
    kl_weight: float = 0.01
    prior: str = "standard_normal"


@dataclass
class TurboDecoderConfig:
    """Turbo decoder — dual-component iterative BP (V6).

    Component A: attention-based (local parity).
    Component B: MSA-based (long-range parity).
    Extrinsic exchange between components with adaptive alpha.
    """

    enabled: bool = True
    alpha_a_init: float = 0.8
    alpha_b_init: float = 0.8
    convergence_threshold: float = 0.01
    min_iterations: int = 1
    max_iterations: int = 6


@dataclass
class WaterFillingConfig:
    """Water-filling capacity allocation across grades (V6).

    Dynamic dimension allocation based on per-grade variance.
    High-variance grades get more dims (more capacity).
    """

    enabled: bool = True
    adaptation_rate: float = 0.001
    min_dims: int = 8
    temperature: float = 1.0
    reg_weight: float = 0.001


@dataclass
class MultimodalImageConfig:
    """Image encoder config (ViT-style patches)."""

    enabled: bool = True
    patch_size: int = 16
    input_channels: int = 3
    max_image_patches: int = 1024


@dataclass
class MultimodalAudioConfig:
    """Audio encoder config (spectrogram frames)."""

    enabled: bool = True
    n_mels: int = 128
    max_audio_frames: int = 512


@dataclass
class CrossModalAttentionConfig:
    """Cross-modal attention (MIMO space-time coding)."""

    enabled: bool = True
    gate_init: float = 0.0


@dataclass
class CrossModalGP2DConfig:
    """Cross-modal GP2D parity (Multiple Description Coding)."""

    enabled: bool = True
    gate_init: float = -3.0


@dataclass
class CrossModalMSAConfig:
    """Cross-modal MSA (Wyner-Ziv side information)."""

    enabled: bool = True
    slots_per_modality: int = 4096
    cross_read_top_k: int = 4


@dataclass
class ContrastiveConfig:
    """Contrastive modality alignment (InfoNCE / Slepian-Wolf)."""

    enabled: bool = True
    temperature: float = 0.07
    weight: float = 0.1


@dataclass
class MultimodalConfig:
    """Multimodal architecture config (V7).

    Each modality has its own source encoder projecting to H.
    Modality type embedding (CDMA spreading code) separates modalities.
    Cross-modal components implement MIMO, MDC, Wyner-Ziv analogies.
    """

    enabled: bool = False
    num_modalities: int = 3
    image: MultimodalImageConfig = field(default_factory=MultimodalImageConfig)
    audio: MultimodalAudioConfig = field(default_factory=MultimodalAudioConfig)
    modality_embed_std: float = 0.02
    modality_dropout_prob: float = 0.10
    modality_mask_ratios: tuple = (0.15, 0.30, 0.25)
    cross_modal_attention: CrossModalAttentionConfig = field(default_factory=CrossModalAttentionConfig)
    cross_modal_gp2d: CrossModalGP2DConfig = field(default_factory=CrossModalGP2DConfig)
    cross_modal_msa: CrossModalMSAConfig = field(default_factory=CrossModalMSAConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    grade_modality_weights: tuple = (
        (1.0, 1.0, 0.5, 0.3),
        (0.3, 0.5, 1.0, 1.0),
        (0.5, 1.0, 0.8, 0.3),
    )
    kl_weights: tuple = (0.01, 0.05, 0.02)


@dataclass
class WaveRoutingConfig:
    """Wave routing architecture — non-transformer alternative (V7).

    Replaces attention with frequency-domain resonance (sound physics).
    All-to-all layer connections. Selective route activation.
    90% linear, 10% nonlinear.

    Shannon mapping:
      - Hz frequencies = OFDM subcarriers
      - Sympathy = matched filter (max SNR receiver)
      - All-to-all = full parity check matrix (LDPC)
      - Route selection = IRA selective decoding
      - Linear (90%) = linear block codes
    """

    enabled: bool = False
    n_frequencies: int = 32
    top_k_ratio: float = 0.25
    ffn_intermediate_ratio: float = 1.333
    route_top_k: int | None = None
    route_threshold: float = 0.5
    use_all_to_all: bool = True


@dataclass
class FreqCodingConfig:
    """Phase-frequency coding — replaces attention with FFT (V7).

    FFT → learnable complex filter → phase modulation → IFFT.
    O(T log T) vs O(T^2 * H). Fewer params than attention.
    No RoPE, no softmax, no causal mask needed.

    Communication theory:
      FFT = OFDM demodulation, IFFT = OFDM modulation
      Complex weight = channel equalizer (frequency-selective)
      Phase = PSK (phase shift keying)
      K modes = sparse frequency allocation (low-pass)
    """

    enabled: bool = True
    n_modes: int = 16
    n_modes_t: int = 16
    n_modes_h: int = 12


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
    target_params: int = 0  # >0 = auto-compute all sizes from this (non-embed+core_lm)
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
    variational_bottleneck: VariationalBottleneckConfig = field(default_factory=VariationalBottleneckConfig)
    turbo_decoder: TurboDecoderConfig = field(default_factory=TurboDecoderConfig)
    water_filling: WaterFillingConfig = field(default_factory=WaterFillingConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    wave_routing: WaveRoutingConfig = field(default_factory=WaveRoutingConfig)
    freq_coding: FreqCodingConfig = field(default_factory=FreqCodingConfig)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    max_steps: int = 150000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.5
    muon_ns_steps: int = 5
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
    w_kl_variational: float = 0.01
    w_water_filling_reg: float = 0.001
    w_contrastive: float = 0.1
    w_cross_modal_coherence: float = 0.001
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
    tokenizer: str = "HuggingFaceTB/SmolLM2-135M"
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
class InferenceConfig:
    """Inference/generation parameters (previously hardcoded in generate.py)."""

    temperature: float = 0.8
    top_k: int = 50
    min_tokens: int = 2
    block_size: int = 16
    refine_passes: int = 2
    repetition_penalty: float = 0.8
    repetition_window: int = 32
    max_iterations: int = 4
    max_new_tokens: int = 128


@dataclass
class HAGIv4Config:
    """Top-level HAGI V4 configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# Original ratios from d147dae (HAGI V4 production config)
# These define how total params split across all components.
_GRADE_RATIOS = (0.1111, 0.1667, 0.1667, 0.1111, 0.4444)
_BOTTLENECK_RATIO = 0.5  # C / H
_MOE_INT_RATIO = 1.3333  # moe_intermediate / C
_SHARED_BASES = 2  # PAW-style shared basis MoE
_KV_HEAD_RATIO = 0.5  # kv_heads / q_heads
_HEAD_DIM_MIN = 16  # minimum head_dim


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    """Round to nearest multiple of 8 for hardware efficiency."""
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def auto_configure(target_params: int, vocab_size: int = 49154) -> ModelConfig:
    """Auto-compute all model sizes from target non-embedding params.

    Uses optimal ratios from evolutionary search:
      - C/H = 0.50 (bottleneck)
      - grade split: [12.5%, 18.75%, 18.75%, 12.5%, 37.5%]
      - moe_int/C = 1.5
      - q_heads * head_dim = H, kv = q/2
      - layers: depth scales as params^0.33, balanced perc/reason/expr
      - n_shared_bases = 2

    Args:
        target_params: desired non-embedding + non-core_lm_head params.
        vocab_size: for core_lm_head sizing (not counted in target).

    Returns:
        ModelConfig with all sizes filled in.
    """
    import math

    # Empirical formula: non_embed_params ≈ k * H^2
    # Measured: H=128 → 0.74M (balanced 2/2/2), H=576 → ~3.5M (7 reasoning)
    # k ≈ 18 for balanced 2/2/2 at C/H=0.5, moe_int=1.5*C
    # For deeper models, k grows linearly with total layers.
    # depth ≈ max(6, round(2 * (target / 0.74e6) ** 0.33))
    # But reasoning layers dominate (shared across iterations).

    # Solve for H: target = k * H^2, where k depends on layer count
    # Iterative: start with estimate, compute layers, refine H.
    target = float(target_params)

    perc, reason, expr = 2, 7, 2

    # Initial estimate: H = sqrt(target / 42) for production layer ratios
    H = int(math.sqrt(target / 42.0))
    H = _round_to_multiple(H, 8)

    for _ in range(5):
        # Reasoning layers scale with model size, but capped
        # Small models (<5M): 3-4 reasoning, large: up to 7
        if target < 2_000_000:
            perc, expr = 1, 1
            reason = max(3, min(7, round(4 * (target / 1e6) ** 0.3)))
        elif target < 10_000_000:
            perc, expr = 2, 2
            reason = max(5, min(7, round(5 * (target / 5e6) ** 0.3)))
        else:
            perc, expr = 2, 2
            reason = 7
        total_layers = perc + reason + expr

        C = _round_to_multiple(int(H * _BOTTLENECK_RATIO), 8)

        # Recompute k: per-layer cost
        # perception/expr at H: ~4*H^2 (attn) + ~3*H*moe_int (MoE) + 2*H (norms)
        # reasoning at C: ~4*C^2 + ~3*C*moe_int_c + 2*C
        moe_int_h = _round_to_multiple(int(H * _MOE_INT_RATIO * _BOTTLENECK_RATIO), 8)
        moe_int_c = _round_to_multiple(int(C * _MOE_INT_RATIO), 8)
        cost_h = 4 * H * H + 3 * H * moe_int_h + 2 * H
        cost_c = 4 * C * C + 3 * C * moe_int_c + 2 * C
        extra = 50 * C + 20 * H  # GDR, GP2D, MSA, HRM, CAST approx
        k = (cost_h * (perc + expr) + cost_c * reason + extra) / (H * H)
        H_new = int(math.sqrt(target / k))
        H_new = _round_to_multiple(H_new, 8)
        if abs(H_new - H) <= 8:
            H = H_new
            break
        H = H_new

    # Correction: formula underestimates by ~1.4x due to uncounted params
    # (HRM transitions, MSA projections, GDR gates, CAST, bottleneck, etc.)
    H = _round_to_multiple(int(H / 1.18), 8)

    C = _round_to_multiple(int(H * _BOTTLENECK_RATIO), 8)

    # Grade dims from ratios
    grade_dims = tuple(_round_to_multiple(int(H * r), 4) for r in _GRADE_RATIOS)
    # Adjust to sum to H exactly
    diff = H - sum(grade_dims)
    if diff != 0:
        grade_dims = list(grade_dims)
        grade_dims[4] += diff  # residual absorbs remainder
        grade_dims = tuple(max(0, g) for g in grade_dims)

    # Attention: q_heads * head_dim ≈ H
    # Original: H=576 → q=8, kv=4, hd=72 (q*hd=576)
    # Scale: keep head_dim ≈ H/8 for small, increase for large
    if H <= 64:
        n_q = 2
    elif H <= 128:
        n_q = 4
    elif H <= 256:
        n_q = 4
    elif H <= 768:
        n_q = 8
    elif H <= 1536:
        n_q = 16
    else:
        n_q = 32
    head_dim = _round_to_multiple(H // n_q, 8)
    # Ensure n_q * head_dim >= H (pad if needed)
    if n_q * head_dim < H:
        head_dim = _round_to_multiple(H // n_q + 8, 8)
    n_kv = max(1, n_q // 2)
    # Ensure n_q divisible by n_kv
    while n_q % n_kv != 0:
        n_kv -= 1

    moe_int = _round_to_multiple(int(C * _MOE_INT_RATIO), 8)

    # MSA dims
    mla_up = n_kv * head_dim
    mla_compress = max(16, mla_up // 2)

    # HRM state dims — original: h_state=256 for C=288 (ratio ~0.89)
    h_state = _round_to_multiple(int(C * 0.89), 8)
    l_state = h_state

    m = ModelConfig()
    m.vocab_size = vocab_size
    m.hidden_size = H
    m.core_hidden_size = C
    m.perception_layers = perc
    m.reasoning_layers = reason
    m.expression_layers = expr
    m.bottleneck_ratio = _BOTTLENECK_RATIO
    m.target_params = target_params

    m.algebra.grade_dims = grade_dims
    m.algebra.hidden_size = H

    m.attention.num_query_heads = n_q
    m.attention.num_kv_heads = n_kv
    m.attention.head_dim = head_dim

    m.hrm.h_state_dim = h_state
    m.hrm.l_state_dim = l_state

    m.msa.n_kv_heads = n_kv
    m.msa.head_dim = head_dim
    m.msa.grade_dims = grade_dims
    m.msa.mla_compress_dim = mla_compress
    m.msa.mla_up_dim = mla_up

    m.moe.intermediate_size = moe_int
    m.moe.n_shared_bases = _SHARED_BASES

    m.cast.scalar_dim = grade_dims[0]
    m.cast.vector_dim = grade_dims[1]

    return m


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
    """Load config from YAML file, then apply keyword overrides.

    If model.target_params is set (>0), all model sizes are auto-computed
    from that single number using optimal ratios from evolutionary search.
    Explicit size overrides in YAML take precedence over auto-computed values.
    """
    import yaml

    cfg = HAGIv4Config()
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _apply_dict(cfg, data)

        # Auto-configure if target_params is set
        tp = cfg.model.target_params
        if tp and tp > 0:
            # Save which fields were explicitly set in YAML
            model_data = data.get("model", {})
            auto = auto_configure(tp, cfg.model.vocab_size)
            # Apply auto-computed values, but don't override explicitly set fields
            _apply_auto(cfg.model, auto, model_data)

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


def _apply_auto(model: ModelConfig, auto: ModelConfig, yaml_data: dict) -> None:
    """Apply auto-computed values, skipping fields explicitly set in YAML."""
    # Top-level model fields
    size_fields = [
        "hidden_size",
        "core_hidden_size",
        "perception_layers",
        "reasoning_layers",
        "expression_layers",
        "bottleneck_ratio",
    ]
    for f in size_fields:
        if f not in yaml_data:
            setattr(model, f, getattr(auto, f))

    # Nested configs
    nested = {
        "algebra": ["grade_dims", "hidden_size"],
        "attention": ["num_query_heads", "num_kv_heads", "head_dim"],
        "hrm": ["h_state_dim", "l_state_dim"],
        "msa": ["n_kv_heads", "head_dim", "grade_dims", "mla_compress_dim", "mla_up_dim"],
        "moe": ["intermediate_size", "n_shared_bases"],
        "cast": ["scalar_dim", "vector_dim"],
    }
    for section, fields in nested.items():
        yaml_section = yaml_data.get(section, {})
        auto_section = getattr(auto, section)
        model_section = getattr(model, section)
        for f in fields:
            if f not in yaml_section:
                setattr(model_section, f, getattr(auto_section, f))
