"""HAGI V21 configuration dataclasses — recovered modular architecture.

5G NR-style codec language model with Cl(3,0,0) geometric algebra.
Pipeline: Source Encoder → Rate Matching → Turbo BP Decoder → Rate Dematching → Source Decoder (SCS preserved).

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
    # V21: geometric product normalization
    normalize_geometric_product: bool = True
    # V21: bivector blade activation (grade-2)
    bivector_activation: str = "tanh"


@dataclass
class AttentionConfig:
    """Grouped-query attention with bidirectional support (fallback when freq_coding disabled)."""

    num_query_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 64
    rope_theta: float = 10000.0
    max_seq_len: int = 4096
    bidirectional: bool = True
    fp16_attention: bool = True
    fp32_rmsnorm: bool = True


@dataclass
class GP2DConfig:
    """Parity channel configuration — SparseParity FEC encoder/checker.

    V21: sparse LDPC-style parity generated BEFORE the channel (true SCS).
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

    V21: turbo decoder with extrinsic information exchange between spatial
    (z_H) and per-token (z_L) components. Entropy-adaptive iteration count
    and deep supervision support recovered from V6 HRM.

    V9: default iterations raised to 4 (min 2). Real LDPC BP needs several
    iterations to converge; the V8 default of 2 left the decoder unable to
    propagate extrinsic information, and the fifth reasoning layer never
    received a gradient (dead weights in the step-500 checkpoint).
    EXIT chart stopping: per-token convergence based on innovation norm.
    """

    num_iterations: int = 4
    min_iterations: int = 2
    convergence_threshold: float = 0.01
    use_convergence_halt: bool = True
    tanh_scale: float = 10.0
    # V21: Turbo decoder extrinsic exchange (from V6 HRM)
    extrinsic_alpha: float = 1.0
    use_adaptive_halt: bool = False
    halt_threshold: float = 0.01
    halt_threshold_start: float = 0.05
    halt_threshold_end: float = 0.001
    # V21: Entropy-adaptive iteration count
    use_entropy_adaptive_refinement: bool = False
    entropy_low_threshold: float = 0.1
    entropy_high_threshold: float = 1.0
    entropy_low_iterations: int = 2
    entropy_high_iterations: int = 6
    # V21: Deep supervision
    use_deep_supervision: bool = False
    deep_supervision_decay: float = 0.5
    deep_supervision_weight: float = 0.5
    use_adaptive_ds_weight: bool = False
    ds_ema_decay: float = 0.9


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
    """Memory Sparse Attention — HARQ buffer + Memory Sparse Attention with MLA.

    V21: DFE (read) + HARQ buffer (write). Ring buffer slot registry
    combined with Multi-head Latent Attention for compressed KV.
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
    """Phase-frequency coding — 2D FFT OFDM with MIMO equalization.

    V21: FFT = OFDM demodulation, IFFT = OFDM modulation.
    Complex weight = MIMO channel equalizer (frequency-selective).
    Phase = PSK (phase shift keying).
    Soft frequency gating = adaptive modulation (5G AMC).
    """

    enabled: bool = True
    n_modes_t: int = 16
    n_modes_h: int = 12
    complex_rank: int = 16
    use_derivative: bool = True
    share_branch_weights: bool = False


@dataclass
class MoEConfig:
    """Mixture of Experts — Switch Transformer style with entropy-aware routing.

    V21: variable-rate capacity allocation. Low-entropy positions route to
    simple experts, high-entropy to complex experts.
    """

    num_experts: int = 4
    top_k: int = 1
    intermediate_size: int = 1024
    use_mod_skip: bool = True
    mod_skip_rate: float = 0.1
    load_balance_weight: float = 0.01
    router_jitter: float = 0.0
    entropy_aware_routing: bool = True


@dataclass
class HRMConfig:
    """Hierarchical Refinement Model — spatial plane (z_H) + per-token state (z_L).

    V21: dual-component Turbo decoder state. z_H is spatial (coarse, strided),
    z_L is per-token (fine, full resolution). Extrinsic-only exchange between
    components avoids information recycling (LDPC BP principle).
    """

    h_state_dim: int = 128
    l_state_dim: int = 256
    h_stride: int = 4


@dataclass
class CQIConfig:
    """Channel Quality Indicator — adaptive sigma/CQI estimation.

    V21: learns SNR estimate from hidden state to drive water-filling
    and adaptive coding rate. 5G analog: CQI feedback from UE to BS.
    """

    hidden_size: int = 384
    cqi_dim: int = 16
    sigma_min: float = 0.02
    sigma_max: float = 0.3


@dataclass
class WaterFillingConfig:
    """Water-Filling capacity allocator — optimal dim allocation across grades.

    V21: Shannon water-filling theorem. High-variance grades get more dims.
    """

    total_dims: int = 288
    num_grades: int = 4
    min_dims: int = 8
    temperature: float = 1.0
    variance_ema_decay: float = 0.99


@dataclass
class KalmanConfig:
    """Kalman filter for iterative decoding — optimal state estimation.

    V21: diagonal covariance O(C). Q/R learnable, zero-init.
    """

    init_process_noise: float = 0.0
    init_measurement_noise: float = 0.0
    covariance_floor: float = 1e-6


@dataclass
class UncertaintyConfig:
    """Learned per-position uncertainty for iterative decoding.

    V21: scalar variance per position (V9 simplification — per-dim
    variance carried no information for Kalman-form update).
    """

    hidden_size: int = 384


@dataclass
class EXITChartConfig:
    """EXIT chart convergence estimator for iterative decoding.

    V21: norm-ratio convergence proxy (avoids cosine-sim MI fiction).
    """

    threshold: float = 0.01
    min_iterations: int = 1


@dataclass
class MultimodalConfig:
    """Multimodal source encoders — text/image/audio to shared latent.

    V21: separate source encoders per modality (optimal source coding),
    modality type embedding (CDMA spreading code).
    """

    enabled: bool = False
    image_patch_size: int = 16
    image_channels: int = 3
    audio_mel_bins: int = 80
    audio_frame_size: int = 10
    modality_embed_dim: int = 32


@dataclass
class FOXP2Config:
    """FOXP2 plasticity controller — per-parameter-group LR modulation.

    V21: adaptive coding rate analog. Applied AFTER backward(), BEFORE step().
    """

    num_groups: int = 8
    hidden: int = 32
    ema_decay: float = 0.99


@dataclass
class KVCacheConfig:
    """KV Cache for O(T) block-parallel generation.

    V21: convolutional code decoder state analog.
    """

    enabled: bool = True
    block_size: int = 16


@dataclass
class SpeculativeConfig:
    """Speculative decoding — draft-then-verify paradigm.

    V21: rate-distortion analog. Draft model = low-rate code, verify = high-rate.
    """

    enabled: bool = False
    draft_length: int = 4
    acceptance_threshold: float = 0.5


@dataclass
class EmbeddingsConfig:
    """Embedding configuration (V9: ConvEmbedding by default).

    V7 froze embeddings (copied from SmolLM2). V8 trained embeddings
    from scratch with proper regularization. V9 replaces the monolithic
    V×H table with a factorized (V×r + r×H) + depthwise Conv1d
    pulse-shaping filter — a memoryless source encoder composed with a
    local temporal mixer (FIR filter analog in communication theory).

    factor_rank: inner rank r of the low-rank embedding approximation.
    kernel_size: causal depthwise Conv1d kernel (pulse-shaping filter).
    """

    trainable: bool = True
    init: str = "normal"
    weight_decay: float = 0.01
    factor_rank: int = 128
    kernel_size: int = 5
    use_conv_embedding: bool = True


@dataclass
class ModelConfig:
    """Full model architecture configuration."""

    vocab_size: int = 49154
    hidden_size: int = 384
    perception_layers: int = 4
    expression_layers: int = 4
    reasoning_layers: int = 3
    norm_eps: float = 1e-6
    bottleneck_ratio: float = 0.5
    core_hidden_size: int = 192
    pilot_spacing: int = 8
    target_params: int = 0
    target_nonembed_params: int = 0
    algebra: AlgebraConfig = field(default_factory=AlgebraConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    gp2d: GP2DConfig = field(default_factory=GP2DConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    msa: MSAConfig = field(default_factory=MSAConfig)
    freq_coding: FreqCodingConfig = field(default_factory=FreqCodingConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    hrm: HRMConfig = field(default_factory=HRMConfig)
    cqi: CQIConfig | None = None
    water_filling: WaterFillingConfig = field(default_factory=WaterFillingConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    uncertainty: UncertaintyConfig | None = None
    exit_chart: EXITChartConfig | None = None
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    max_steps: int = 150000
    warmup_steps: int = 1000
    learning_rate: float = 3e-4
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.5
    muon_ns_steps: int = 5
    weight_decay: float = 0.1
    precision: str = "bf16"
    grad_accum_steps: int = 2
    max_grad_norm: float | None = 1.0
    batch_size: int = 10
    seq_len: int = 1024
    w_ce: float = 1.0
    w_whiteness: float = 0.01
    w_parity: float = 0.05
    w_correction_alignment: float = 0.0
    w_rate_distortion: float = 0.005
    w_contrastive: float = 0.0
    # V12: parity-code diversity regularizer. V18 keeps the default 0.05
    # (V17 yaml bug set it to 0.0 — collapsed the LDPC graph).
    w_parity_diversity: float = 0.05
    # V18: train BP iterations (infer uses model.refinement.num_iterations).
    bp_iterations: int = 3
    use_two_phase_schedule: bool = True
    two_phase_split: float = 0.5
    phase1_mask_ratio: float = 0.15
    phase2_mask_ratio: float = 0.35
    phase3_mask_ratio: float = 0.50
    # V12: suffix-vs-random mask probability for ``create_semantic_corruption``.
    # Inference uses suffix-only masking (contiguous block at the end), so
    # the model must be exposed to that distribution during training. The
    # previous default (0.5) spent half the batch on random masks, causing
    # ``suffix_ce - masked_ce`` gap to grow from 0 → 0.65 by step 1000 —
    # the model specialised on random-mask recovery and underperformed on
    # the suffix regime used for generation. 0.7 biases 70% of batches
    # toward suffix masking, matching the inference distribution.
    suffix_probability: float = 0.3
    use_continuous_anneal: bool = True
    distill_enabled: bool = True
    distill_kl_enabled: bool = False
    distill_teacher: str = "HuggingFaceTB/SmolLM2-360M"
    distill_teacher_hidden_size: int = 576
    distill_embed_teacher: str = "HuggingFaceTB/SmolLM2-135M"
    distill_alpha_start: float = 0.0
    distill_alpha_end: float = 0.0
    distill_temperature: float = 2.0
    distill_temp_start: float = 4.0
    distill_temp_end: float = 1.0
    distill_use_temp_anneal: bool = True
    distill_end_frac: float = 0.6
    awgn_enabled: bool = True
    # V19: raised sigma. V18 had channel 9.6x below capacity (SNR=29dB,
    # capacity=4.82 bits/use vs rate 0.5). BP fully cancelled AWGN even at
    # 3 iterations. Raise to sigma=0.15 so SNR ~ 17dB, capacity ~ 3.0 bits/use,
    # still above rate 0.5 but forces BP to do real work.
    awgn_sigma_start: float = 0.15
    awgn_sigma_end: float = 0.05
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
    foxp2: FOXP2Config = field(default_factory=FOXP2Config)


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
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)


@dataclass
class HAGIv4Config:
    """Top-level HAGI V21 configuration — recovered modular architecture."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


_BOTTLENECK_RATIO = 0.75
_HEAD_DIM_TARGET = 64
_GRADE_RATIOS = (
    1,
    1.5,
    1.5,
    1,
    4,
)  # Cl(3,0,0): scalar, 3 vectors, 3 bivectors, pseudoscalar


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n, minimum 2."""
    p = 2
    while p < n:
        p *= 2
    return p


def auto_configure(target_params: int, vocab_size: int = 49154) -> ModelConfig:
    """Auto-compute model sizes from a parameter budget.

    The budget is split into two independent parts following the
    Source-Channel Separation Theorem:

      * ``target_nonembed_params`` (or ``target_params`` when the former is
        unset) governs the channel codec + source codec body. ``H`` is solved
        from this budget alone, so the body capacity is invariant to the
        vocabulary size.
      * The source embedding is a factorized ``V x r + r x H`` table plus a
        depthwise Conv1d pulse-shaping filter. Its cost is reported
        separately and does not inflate the channel budget.

    V9 cost model matches the real architecture:
      * ``FactoredSwiGLU`` FFN uses rank ``r_ffn = H / 4`` (SVD-style), not
        a dense ``H x intermediate`` matrix. Each FFN contributes
        ``H*r_ffn + r_ffn*2*intermediate + intermediate*r_ffn + r_ffn*H``.
      * ``FreqCoding2D`` uses shared low-rank complex weights
        ``[H, head_dim, rank] + [H, rank, head_dim]`` (4 tensors per branch),
        not the dense ``4*H*H`` of the V8 estimate.
      * ``SparseParity`` stores ``M*C`` masked weights (already small).

    This fixes the V8 regression where the cost model overestimated the body
    and produced an ``H`` much larger than the budget warranted (a 17.5M
    target produced a 97M checkpoint because embedding dominated at 91% and
    the body solver ignored the FFN rank compression).
    """
    import math

    target = float(target_params)
    ratio = _BOTTLENECK_RATIO
    int_ratio = 4.0 / 3.0
    # Factored FFN inner rank relative to H (matches FactoredSwiGLU default).
    ffn_rank_rel = 0.25

    layers = max(2, round(math.log10(target / 1e5) * 3))
    perc = max(1, layers // 4)
    expr = perc
    reason = max(3, layers - 2 * perc)

    # Per-H^2 cost of one FreqBlock:
    #   FreqCoding2D: 4 * head_dim * rank (low-rank A,B) per main branch, and
    #     the same again for dT/dH derivative branches, but those are shared
    #     across layers so we charge only the per-layer residual (phase + gate
    #     + layer scales). We approximate the per-layer freq cost as 2*H.
    #   FactoredSwiGLU: H*r_ffn + r_ffn*2*intermediate + intermediate*r_ffn + r_ffn*H,
    #     with r_ffn = H*ffn_rank_rel and intermediate = H*int_ratio*ratio.
    # Per-H^2 cost of one reasoning block (channel side):
    #   FreqCoding2D (shared): ~0 (charged once globally, small).
    #   FactoredSwiGLU on C: 2*(C*r_ffn_c) + 2*(r_ffn_c * 2*intermediate_c),
    #     with r_ffn_c = C*ffn_rank_rel and intermediate_c = C*int_ratio.
    # We fold these into a single cost-per-H^2 coefficient via the ratios.
    cost_per_h_sq = (
        2.0 * (perc + expr)
        + 4.0 * ffn_rank_rel * (1.0 + int_ratio * ratio) * (perc + expr)
        + 4.0 * ratio * ratio * ffn_rank_rel * (1.0 + int_ratio) * reason
        + 2.0 * ratio * reason
    )
    H = _round_to_multiple(int(math.sqrt(target / cost_per_h_sq)), 8)

    for _ in range(8):
        C = _round_to_multiple(int(H * ratio), 8)
        ffn_int_h = _round_to_multiple(int(H * int_ratio * ratio), 8)
        ffn_int_c = _round_to_multiple(int(C * int_ratio), 8)
        r_ffn_h = max(32, H // 4)
        r_ffn_c = max(32, C // 4)
        # FreqBlock cost: layer scales + freq gates + phase (small per layer)
        # + derivative branch params (shared across layers, charge once).
        cost_h_freq = 2 * H
        cost_h_ffn = H * r_ffn_h + r_ffn_h * 2 * ffn_int_h + ffn_int_h * r_ffn_h + r_ffn_h * H
        cost_h = cost_h_freq + cost_h_ffn + 2 * H
        cost_c_freq = 2 * C
        cost_c_ffn = C * r_ffn_c + r_ffn_c * 2 * ffn_int_c + ffn_int_c * r_ffn_c + r_ffn_c * C
        cost_c = cost_c_freq + cost_c_ffn + 2 * C
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
    m.target_nonembed_params = int(target)

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
            # Prefer the explicit non-embedding budget when provided so that
            # the channel codec body capacity is independent of vocabulary size.
            nonembed_budget = model_data.get("target_nonembed_params")
            budget = nonembed_budget if nonembed_budget else tp
            auto = auto_configure(int(budget), cfg.model.vocab_size)
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

    # V19 SCS invariants — protect against regressions discovered in V17/V18.
    # 1. Train/infer BP asymmetry: infer must use MORE iterations than train
    #    (the V13 win). If train >= infer, we are not doing asymmetric codec.
    train_bp = getattr(cfg.train, "bp_iterations", None)
    if train_bp is not None and type(train_bp) is int and train_bp >= cfg.model.refinement.num_iterations:
        raise ValueError(
            f"train.bp_iterations ({train_bp}) must be < refinement.num_iterations "
            f"({cfg.model.refinement.num_iterations}). Train/infer BP asymmetry "
            f"is a V13-vintage invariant — inference must use MORE BP iterations "
            f"than training for recovery without gradient overhead."
        )
    # 2. AWGN sigma_end must be > 0 — channel never closes completely.
    #    V17 bug: channel never opened because the schedule bottomed at 0.0.
    if cfg.train.awgn_enabled and cfg.train.awgn_sigma_end <= 0.0:
        raise ValueError(
            "train.awgn_sigma_end must be > 0 when awgn_enabled. V17 regression: "
            "channel_closed at sigma=0 broke the LDPC training signal entirely."
        )
    # 3. parity_diversity weight must be > 0 — prevents LDPC code collapse.
    #    V17 YAML bug: w_parity_diversity=0.0 caused parity code to collapse
    #    to a single edge per check.
    if cfg.train.w_parity_diversity <= 0.0:
        raise ValueError(
            "train.w_parity_diversity must be > 0. V17 regression: a YAML bug "
            "set this to 0.0 and the LDPC code collapsed (parity_metric=0.0096)."
        )
    # 4. AWGN sigma_start must be >= sigma_end (schedule goes from noisy to clean).
    if cfg.train.awgn_enabled and cfg.train.awgn_sigma_start < cfg.train.awgn_sigma_end:
        raise ValueError(
            f"awgn_sigma_start ({cfg.train.awgn_sigma_start}) must be >= "
            f"awgn_sigma_end ({cfg.train.awgn_sigma_end}). Schedule must go "
            f"from noisy (high sigma) to clean (low sigma), not the reverse."
        )


def _apply_auto(model: ModelConfig, auto: ModelConfig, yaml_data: dict) -> None:
    """Apply auto-computed values, skipping fields explicitly set in YAML."""
    size_fields = [
        "hidden_size",
        "core_hidden_size",
        "perception_layers",
        "expression_layers",
        "reasoning_layers",
        "bottleneck_ratio",
    ]
    for f in size_fields:
        if f not in yaml_data:
            setattr(model, f, getattr(auto, f))

    nested = {
        "algebra": ["grade_dims", "hidden_size"],
        "attention": ["num_query_heads", "num_kv_heads", "head_dim"],
        "msa": [
            "n_kv_heads",
            "head_dim",
            "grade_dims",
            "mla_compress_dim",
            "mla_up_dim",
        ],
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

    # V11: recompute codec dimensions from the FINAL core_hidden_size, not from
    # the auto_configure estimate. When the YAML overrides hidden_size/
    # core_hidden_size, the auto-computed n_checks (= auto.C = 1056) no longer
    # matches the actual C (= 320), producing a 3.3x oversized parity matrix.
    # code_rate governs the ratio: n_checks = C * (1/rate - 1).
    if "n_checks" not in yaml_data.get("codec", {}):
        rate = model.codec.code_rate
        model.codec.n_checks = max(1, int(model.core_hidden_size * (1.0 / rate - 1.0)))
    if "edges_per_check" not in yaml_data.get("codec", {}):
        model.codec.edges_per_check = max(3, min(8, model.core_hidden_size // 32))
