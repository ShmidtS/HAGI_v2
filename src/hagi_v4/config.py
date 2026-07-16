"""HAGI V7.1 configuration dataclasses.

5G NR-style codec language model with Cl(3,0,0) geometric algebra.
Pipeline: Source Encoder → Rate Matching → LDPC Turbo Decoder → Rate Dematching → Source Decoder.

Auto-configure: set `target_params` in YAML, all sizes computed automatically.
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
    """Grouped-query attention with bidirectional support (fallback when freq_coding disabled)."""

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
    """Parity channel configuration (V8: used by SparseParity, kept for compat).

    V8 replaces dense GP2D with sparse LDPC-style parity. These fields
    are retained for backward compatibility with existing configs.
    """

    window: int = 1
    gate_init: float = -2.0
    use_whiteness_loss: bool = True
    whiteness_weight: float = 0.01
    use_systematic_parity: bool = True
    parity_weight: float = 0.1


@dataclass
class CodecConfig:
    """V8 channel codec configuration — LDPC-style FEC parameters.

    Implements true Source-Channel Separation: parity is generated
    BEFORE the channel, not inside the decoder.

    5G NR analog: LDPC base graph + rate matching + interleaving.
    """

    code_rate: float = 0.5
    n_checks: int = 0
    edges_per_check: int = 4
    interleaver_mode: str = "qpp"
    exit_threshold: float = 0.01


@dataclass
class RefinementConfig:
    """Turbo decoding loop — LDPC iterative belief propagation.

    EXIT chart stopping: per-token convergence based on innovation norm.
    Kalman filter: optimal Bayesian blend of prediction and measurement.
    """

    num_iterations: int = 4
    min_iterations: int = 1
    convergence_threshold: float = 0.01
    use_convergence_halt: bool = True
    tanh_scale: float = 10.0


@dataclass
class MaskingConfig:
    """Adaptive erasure channel for codec training.

    Mask ratio adapts to model confidence (capacity matching).
    mask_embed initialized as max-entropy vector (not zero).
    """

    mask_ratio: float = 0.3
    use_span_masking: bool = True
    span_length: int = 3
    use_progressive: bool = True
    use_adaptive_erasure: bool = True
    mask_embed_init: str = "max_entropy"
    adaptation_rate: float = 0.01


@dataclass
class MSAConfig:
    """Memory Sparse Attention — DFE (read) + HARQ buffer (write).

    Ring buffer slot registry + MLA (Multi-head Latent Attention).
    """

    max_slots: int = 2048
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
class FreqCodingConfig:
    """Phase-frequency coding — replaces attention with 2D FFT (V7).

    FFT = OFDM demodulation, IFFT = OFDM modulation.
    Complex weight = MIMO channel equalizer (frequency-selective).
    Phase = PSK (phase shift keying).
    Soft frequency gating = adaptive modulation (5G AMC).
    """

    enabled: bool = True
    n_modes_t: int = 16
    n_modes_h: int = 12
    complex_rank: int = 16


@dataclass
class EmbeddingsConfig:
    """Embedding configuration (V8: trainable by default).

    V7 froze embeddings (copied from SmolLM2). V8 trains embeddings
    from scratch with proper regularization.
    """

    trainable: bool = True
    init: str = "normal"
    weight_decay: float = 0.01


@dataclass
class MultimodalImageConfig:
    """Image encoder config (ViT-style patches)."""

    patch_size: int = 16
    input_channels: int = 3
    max_image_patches: int = 1024


@dataclass
class MultimodalAudioConfig:
    """Audio encoder config (spectrogram frames)."""

    n_mels: int = 128
    max_audio_frames: int = 512


@dataclass
class CrossModalConfig:
    """Cross-modal mixing — MIMO space-time coding in frequency domain.

    Cross-spectrum mixing = MIMO channel estimation across modalities.
    Cross-modal GP2D = Multiple Description Coding (MDC) parity.
    Cross-modal MSA = Wyner-Ziv side information at decoder.
    """

    enabled: bool = False
    num_modalities: int = 3
    modality_embed_std: float = 0.02
    modality_dropout_prob: float = 0.10
    modality_mask_ratios: tuple = (0.15, 0.30, 0.25)
    cross_freq_gate_init: float = 0.0
    cross_gp2d_gate_init: float = -3.0
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 0.1


@dataclass
class ModelConfig:
    """Full model architecture configuration."""

    vocab_size: int = 49154
    hidden_size: int = 576
    perception_layers: int = 2
    reasoning_layers: int = 7
    norm_eps: float = 1e-6
    bottleneck_ratio: float = 0.5
    core_hidden_size: int = 288
    pilot_spacing: int = 8
    target_params: int = 0
    algebra: AlgebraConfig = field(default_factory=AlgebraConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    gp2d: GP2DConfig = field(default_factory=GP2DConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    msa: MSAConfig = field(default_factory=MSAConfig)
    freq_coding: FreqCodingConfig = field(default_factory=FreqCodingConfig)
    image: MultimodalImageConfig = field(default_factory=MultimodalImageConfig)
    audio: MultimodalAudioConfig = field(default_factory=MultimodalAudioConfig)
    multimodal: CrossModalConfig = field(default_factory=CrossModalConfig)


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
    max_grad_norm: float | None = None
    batch_size: int = 10
    seq_len: int = 1024
    w_ce: float = 1.0
    w_whiteness: float = 0.01
    w_parity: float = 0.1
    w_correction_alignment: float = 0.01
    w_rate_distortion: float = 0.01
    w_contrastive: float = 0.1
    use_two_phase_schedule: bool = True
    two_phase_split: float = 0.5
    phase1_mask_ratio: float = 0.15
    phase2_mask_ratio: float = 0.35
    phase3_mask_ratio: float = 0.50
    use_continuous_anneal: bool = True
    distill_enabled: bool = True
    distill_kl_enabled: bool = False
    distill_teacher: str = "HuggingFaceTB/SmolLM2-360M"
    distill_teacher_hidden_size: int = 576
    distill_embed_teacher: str = "HuggingFaceTB/SmolLM2-135M"
    distill_alpha_start: float = 0.5
    distill_alpha_end: float = 0.3
    distill_temperature: float = 2.0
    distill_temp_start: float = 4.0
    distill_temp_end: float = 1.0
    distill_use_temp_anneal: bool = True
    distill_end_frac: float = 0.6
    awgn_enabled: bool = True
    awgn_sigma_start: float = 0.005
    awgn_sigma_end: float = 0.0
    awgn_end_frac: float = 0.5
    freeze_embeddings: bool = False
    tokenizer: str = "HuggingFaceTB/SmolLM2-135M"
    eos_token_id: int = 0
    pad_token_id: int = 49152
    checkpoint_dir: str = "checkpoints"
    checkpoint_format_version: int = 3
    checkpoint_interval: int = 5000
    checkpoint_keep_last: int = 3
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
    data_dtype: str = "auto"
    data_dir: str = "data"


@dataclass
class InferenceConfig:
    """Inference/generation parameters."""

    temperature: float = 0.8
    top_k: int = 50
    min_new_tokens: int = 2
    repetition_penalty: float = 1.1
    repetition_window: int = 64
    no_repeat_ngram_size: int = 3
    max_iterations: int = 4
    max_new_tokens: int = 128


@dataclass
class HAGIv4Config:
    """Top-level HAGI V7.1 configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


_BOTTLENECK_RATIO = 0.5
_HEAD_DIM_TARGET = 64
_GRADE_RATIOS = (1, 1.5, 1.5, 1, 4)  # Cl(3,0,0): scalar, 3 vectors, 3 bivectors, pseudoscalar


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n, minimum 2."""
    p = 2
    while p < n:
        p *= 2
    return p


def auto_configure(target_params: int, vocab_size: int = 49154) -> ModelConfig:
    """Auto-compute all model sizes from target non-embedding params.

    All dimensions derived from target_params via continuous formulas —
    no hardcoded thresholds, no magic constants. Convergence is natural
    via iterative H-size refinement.
    """
    import math

    target = float(target_params)

    ratio = _BOTTLENECK_RATIO
    int_ratio = 4.0 / 3.0

    layers = max(2, round(math.log10(target / 1e5) * 3))
    perc = max(1, layers // 4)
    expr = perc
    reason = max(3, layers - 2 * perc)

    cost_per_h_sq = (
        4 * (perc + expr)
        + 4 * ratio * ratio * reason
        + 3 * int_ratio * ratio * (perc + expr)
        + 3 * int_ratio * ratio * ratio * reason
    )
    H = _round_to_multiple(int(math.sqrt(target / cost_per_h_sq)), 8)

    for _ in range(8):
        C = _round_to_multiple(int(H * ratio), 8)
        ffn_int_h = _round_to_multiple(int(H * int_ratio * ratio), 8)
        ffn_int_c = _round_to_multiple(int(C * int_ratio), 8)
        cost_h = 4 * H * H + 3 * H * ffn_int_h + 2 * H
        cost_c = 4 * C * C + 3 * C * ffn_int_c + 2 * C
        extra = 50 * C + 20 * H
        k = (cost_h * (perc + expr) + cost_c * reason + extra) / (H * H)
        H_new = _round_to_multiple(int(math.sqrt(target / k)), 8)
        if abs(H_new - H) <= 8:
            H = H_new
            break
        H = H_new

    C = _round_to_multiple(int(H * ratio), 8)

    head_dim = _round_to_multiple(_HEAD_DIM_TARGET, 8)
    n_q = max(2, _next_pow2(H // head_dim))
    head_dim = _round_to_multiple(H // n_q, 8)
    while n_q * head_dim < H:
        head_dim += 8
    n_kv = max(1, n_q // 2)
    while n_q % n_kv != 0:
        n_kv -= 1

    mla_up = n_kv * head_dim
    mla_compress = max(16, mla_up // 2)

    total_grade = sum(_GRADE_RATIOS)
    grade_dims = tuple(_round_to_multiple(int(H * r / total_grade), 4) for r in _GRADE_RATIOS)
    diff = H - sum(grade_dims)
    if diff != 0:
        grade_dims = list(grade_dims)
        grade_dims[-1] += diff
        grade_dims = tuple(max(0, g) for g in grade_dims)

    n_modes_t = _round_to_multiple(max(4, H // 32), 4)
    n_modes_h = _round_to_multiple(max(4, head_dim // 4), 4)
    complex_rank = max(8, head_dim // 4)

    n_checks = C
    edges_per_check = max(3, min(8, C // 32))

    m = ModelConfig()
    m.vocab_size = vocab_size
    m.hidden_size = H
    m.core_hidden_size = C
    m.perception_layers = perc
    m.reasoning_layers = reason
    m.bottleneck_ratio = ratio
    m.target_params = target_params

    m.algebra.grade_dims = grade_dims
    m.algebra.hidden_size = H

    m.attention.num_query_heads = n_q
    m.attention.num_kv_heads = n_kv
    m.attention.head_dim = head_dim

    m.msa.n_kv_heads = n_kv
    m.msa.head_dim = head_dim
    m.msa.grade_dims = grade_dims
    m.msa.mla_compress_dim = mla_compress
    m.msa.mla_up_dim = mla_up

    m.freq_coding.n_modes_t = n_modes_t
    m.freq_coding.n_modes_h = n_modes_h
    m.freq_coding.complex_rank = complex_rank

    m.codec.n_checks = n_checks
    m.codec.edges_per_check = edges_per_check

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
    """Load config from YAML file, then apply keyword overrides."""
    import yaml

    cfg = HAGIv4Config()
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _apply_dict(cfg, data)

        tp = cfg.model.target_params
        if tp and tp > 0:
            model_data = data.get("model", {})
            auto = auto_configure(tp, cfg.model.vocab_size)
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
    validate_config(cfg)
    return cfg


def validate_config(cfg: HAGIv4Config) -> None:
    def bounded_int(name: str, value: int, upper: int) -> None:
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{name} must be an integer in [1, {upper}]")

    bounded_int("model.vocab_size", cfg.model.vocab_size, 1_000_000)
    bounded_int("model.hidden_size", cfg.model.hidden_size, 16_384)
    bounded_int("model.core_hidden_size", cfg.model.core_hidden_size, 16_384)
    bounded_int("model.attention.max_seq_len", cfg.model.attention.max_seq_len, 65_536)
    bounded_int("train.seq_len", cfg.train.seq_len, 65_536)
    bounded_int("train.batch_size", cfg.train.batch_size, 4_096)
    bounded_int("model.refinement.num_iterations", cfg.model.refinement.num_iterations, 64)
    bounded_int("model.refinement.min_iterations", cfg.model.refinement.min_iterations, 64)
    bounded_int("inference.max_new_tokens", cfg.inference.max_new_tokens, 8_192)
    bounded_int("inference.max_iterations", cfg.inference.max_iterations, 64)
    if cfg.model.core_hidden_size > cfg.model.hidden_size:
        raise ValueError("model.core_hidden_size must not exceed model.hidden_size")
    if cfg.model.refinement.min_iterations > cfg.model.refinement.num_iterations:
        raise ValueError("model.refinement.min_iterations must not exceed num_iterations")
    if not 0 <= cfg.inference.min_new_tokens <= cfg.inference.max_new_tokens:
        raise ValueError("inference.min_new_tokens must be within max_new_tokens")
    if cfg.train.checkpoint_format_version != 3:
        raise ValueError("checkpoint_format_version must be 3 for channel-correct fresh training")
    if type(cfg.train.checkpoint_keep_last) is not int or cfg.train.checkpoint_keep_last < 1:
        raise ValueError("checkpoint_keep_last must be an integer of at least 1")
    if type(cfg.train.distill_enabled) is not bool:
        raise ValueError("distill_enabled must be a boolean")
    if cfg.train.distill_enabled is True and cfg.train.distill_teacher_hidden_size <= 0:
        raise ValueError("distill_teacher_hidden_size must be positive when distillation is enabled")
    if cfg.train.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    if cfg.train.eos_token_id == cfg.train.pad_token_id:
        raise ValueError("eos_token_id and pad_token_id must be distinct")
    if not 0 <= cfg.train.eos_token_id < cfg.model.vocab_size:
        raise ValueError("eos_token_id must be within the model vocabulary")
    if not 0 <= cfg.train.pad_token_id < cfg.model.vocab_size:
        raise ValueError("pad_token_id must be within the model vocabulary")


def _apply_auto(model: ModelConfig, auto: ModelConfig, yaml_data: dict) -> None:
    """Apply auto-computed values, skipping fields explicitly set in YAML."""
    size_fields = [
        "hidden_size",
        "core_hidden_size",
        "perception_layers",
        "reasoning_layers",
        "bottleneck_ratio",
    ]
    for f in size_fields:
        if f not in yaml_data:
            setattr(model, f, getattr(auto, f))

    nested = {
        "algebra": ["grade_dims", "hidden_size"],
        "attention": ["num_query_heads", "num_kv_heads", "head_dim"],
        "msa": ["n_kv_heads", "head_dim", "grade_dims", "mla_compress_dim", "mla_up_dim"],
        "freq_coding": ["n_modes_t", "n_modes_h", "complex_rank"],
        "codec": ["n_checks", "edges_per_check"],
    }
    for section, fields in nested.items():
        yaml_section = yaml_data.get(section, {})
        auto_section = getattr(auto, section)
        model_section = getattr(model, section)
        for f in fields:
            if f not in yaml_section:
                setattr(model_section, f, getattr(auto_section, f))
