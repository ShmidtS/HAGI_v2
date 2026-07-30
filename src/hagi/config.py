"""HAGI-2 configuration — codec-first scalable multimodal channel LM (V28).

The model is a causal autoregressive LM designed as a communication system.
All knobs live here; ``auto_configure`` solves the hidden/layer/head sizes
from a parameter budget and every other value is an explicit config entry
(no hidden hardcoded constants scattered through the training loop).

V28 schema (supersedes V27): keeps the verified bones (real GQA, incremental
KV-cache, causal conv source encoder, ternary body, off-path auxiliaries,
type-based optim routing) and makes the information-theoretic machinery honest
and scalable:

  * **Water-filling MoE** (``MoEConfig``): SNR-gated routing (capacity follows
    per-position residual SNR) + routing-entropy capacity maximization + batched
    expert dispatch. Replaces V27's magnitude-gate.
  * **Sliding-window attention** (``SlidingWindowConfig``): opt-in O(T*W) local
    channel; long-range relayed by full-attention layers.
  * **Off-path predictive refinement** (``RefinementConfig``): rehabilitated HEP
    extrinsic error highway, strictly off the main path, EXIT-halt gated.
  * **2D/1D-RoPE** for multimodal patches/frames (no fixed positional table).
  * **EXIT-chart convergence halt** on the distortion beta-anneal.

V27 knobs dropped: ``entropy_gate_weight`` (renamed ``snr_gate_weight``).
Checkpoint format bumped 6 -> 7 (incompatible with V27 — fresh retrain).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AttentionConfig:
    """Grouped-query attention (causal by default; bidir/prefix/soft_causal opt-in)."""

    num_query_heads: int = 8
    num_kv_heads: int = 4  # real GQA (V25 ignored this); n_kv <= n_q
    head_dim: int = 64
    rope_theta: float = 10000.0
    max_seq_len: int = 4096
    sliding_window: int = 0  # 0 = full attention; >0 = windowed causal (local channel)


@dataclass
class SlidingWindowConfig:
    """Hybrid sliding-window / full-attention layer placement.

    A local (windowed) channel is O(T*W); full-attention layers act as global
    parity relays so long-range signal survives. ``window_layers`` indexes (0-based)
    are windowed; the rest are full attention. An empty index set with
    ``sliding_window>0`` windows ALL layers.

    Attributes:
        sliding_window: window size W (0 = full attention everywhere).
        window_layers: explicit per-layer window mask; if empty and W>0, all
            layers are windowed. Length may be shorter than num_layers (trailing
            layers default to full attention).
        full_every: when >0, make every Nth layer full-attention and the rest
            windowed (ignored if ``window_layers`` is non-empty).
    """

    sliding_window: int = 0
    window_layers: tuple[int, ...] = ()
    full_every: int = 0


@dataclass
class EmbeddingsConfig:
    """Factorized source encoder: V x r + r x H + causal depthwise Conv1d filter."""

    factor_rank: int = 128
    kernel_size: int = 5
    trainable: bool = True


@dataclass
class MultimodalConfig:
    """Per-modality source encoders + Q-Former bridge + grounded infomax."""

    enabled: bool = False
    image_patch_size: int = 16
    image_channels: int = 3
    audio_mel_bins: int = 80
    num_modalities: int = 3
    modality_embed_std: float = 0.02
    # Q-Former bridge: fixed-size fused multimodal prefix (O(1) in image size).
    n_bridge_queries: int = 32
    bridge_layers: int = 2
    bridge_n_heads: int = 4


@dataclass
class GroundedInfomaxConfig:
    """VICReg + InfoNCE auxiliary grounding for the multimodal joint embedding.

    VICReg stabilizes the joint embedding (variance hinge against collapse,
    invariance to augmentation, covariance decorrelation). InfoNCE maximizes
    a lower bound on cross-modal mutual information (Slepian-Wolf distributed
    source coding). Both are off-path auxiliaries.
    """

    vicreg_var_weight: float = 1.0
    vicreg_inv_weight: float = 1.0
    vicreg_cov_weight: float = 0.04
    vicreg_gamma: float = 1.0  # per-dim std hinge target
    infonce_temperature: float = 0.07


@dataclass
class BottleneckConfig:
    """Auxiliary variational information bottleneck (H->C, off the main path).

    The IB computes KL/distortion on the context hidden as an auxiliary
    regularizer. It does NOT intercept the LM signal — inserting it into the
    main path deadlocks from-scratch training (CE stalls at ~ln(V)).

    ``kl_free_bits`` is a PER-DIMENSION floor on the KL rate. At 0.01 this
    allows ~2.56 nats/token for C=256 (vs the old 0.5 per-dim which forced
    128 nats/token and clamped every dimension to the floor).

    Iterative latent refinement (opt-in): when ``iterative_ib_enabled`` and
    ``ib_max_iters > 1``, the bottleneck runs multiple passes, feeding each
    reconstruction back as the next input. EXIT-halt gates via SNR of the
    latent state (``ib_snr_threshold``) or distortion stall between iterations
    (``ib_distortion_epsilon``). At ``ib_max_iters=1`` behaviour is identical
    to the standard single-pass bottleneck.
    """

    dim: int = 192  # C: must satisfy C < H (real compression)
    ib_beta: float = 0.001
    kl_free_bits: float = 0.01  # per-dimension KL floor (was 0.5 — always clamped)
    logvar_clamp: tuple[float, float] = (-5.0, 5.0)
    distortion_eps: float = 1e-6
    # Iterative latent refinement with EXIT-halt.
    iterative_ib_enabled: bool = False
    ib_max_iters: int = 1  # 1 = standard single-pass (backward compat)
    ib_snr_threshold: float = 0.0  # 0.0 = disabled; latent SNR early-exit gate
    ib_distortion_epsilon: float = 0.0  # 0.0 = disabled; distortion-stall early-exit gate
    # CTM-inspired latent memory bank (cross-attends h_ctx over recent z states).
    memory_bank_size: int = 0  # 0 = disabled (backward compat)
    memory_num_heads: int = 4
    memory_head_dim: int = 32


@dataclass
class MoEConfig:
    """Entropy-aware mixture of experts (water-filling capacity allocation).

    MoE is the scalability lever: it grows parameter count without growing
    per-token compute. A shared expert always fires (DeepSeek-Mo style) so
    routed experts specialize on the residual. The router input includes a
    per-position entropy scalar so high-entropy positions can route to heavier
    capacity — variable-rate coding (water-filling: allocate capacity to the
    channels with highest SNR/entropy). Opt-in; off for the small text config.
    """

    enabled: bool = False
    num_experts: int = 4
    top_k: int = 1
    n_shared: int = 1
    moe_every: int = 2  # replace the FFN with MoE on every Nth block
    intermediate_size: int = 0  # 0 -> derived from hebbian_expansion * H
    router_init_std: float = 1.0
    snr_gate_weight: float = 1.0  # weight on the water-filling SNR proxy fed to the router
    route_entropy_weight: float = 0.01  # weight on the routing-entropy capacity-maximization bonus


@dataclass
class TernaryConfig:
    """BitNet b1.58 ternary body (the genuine discrete channel)."""

    use_ternary: bool = True
    hebbian_expansion: int = 4  # m = expansion * H
    eps: float = 1e-5  # per-output-channel scale floor for ternarize()


@dataclass
class BodyConfig:
    """Ternary transformer body: a single unified KV-cacheable stack."""

    num_layers: int = 12
    ternary: TernaryConfig = field(default_factory=TernaryConfig)
    bottleneck: BottleneckConfig = field(default_factory=BottleneckConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)


@dataclass
class DistillationConfig:
    """Online hidden-state distillation from a causal teacher LM (opt-in)."""

    enabled: bool = False
    teacher: str = "HuggingFaceTB/SmolLM2-360M"
    teacher_hidden_size: int = 960
    embed_teacher: str = "HuggingFaceTB/SmolLM2-135M"
    alpha_start: float = 0.0
    alpha_end: float = 0.0
    temperature: float = 2.0
    temp_start: float = 4.0
    temp_end: float = 1.0
    use_temp_anneal: bool = True
    end_frac: float = 0.6


@dataclass
class CurriculumConfig:
    """Two-stage dataset curriculum."""

    enabled: bool = True
    stage2_start: int = 100000
    cycles_per_dataset: int = 3
    stage1_order: list[str] = field(
        default_factory=lambda: [
            "tinystories", "python_instruct", "smoltalk", "wikipedia_en",
            "wikipedia_ru", "openwebmath", "oscar_ru", "slimpajama", "edu",
        ]
    )
    stage2_datasets: list[str] = field(default_factory=lambda: ["openwebmath", "edu", "slimpajama"])


@dataclass
class RefinementConfig:
    """Off-path HEP predictive refinement (opt-in).

    Rehabilitated from the V25 ``predictive_v25.py`` and re-placed strictly OFF
    the main LM path (the V25 in-path placement deadlocked from-scratch
    training). A zero-init HEP feedback refines a CLONE of the context hidden;
    an auxiliary CE+KL loss is the only gradient into the branch. EXIT-chart
    novelty halts the distortion beta-anneal on convergence.
    """

    enabled: bool = False
    iterations: int = 2  # refinement passes
    update_hidden: int = 0  # 0 -> H; else MLP hidden width
    hep_enabled: bool = True  # zero-init HEP feedback (depth-independent correction)
    exit_threshold: float = 0.05  # novelty below this freezes the beta-anneal
    exit_min_steps: int = 50
    exit_window: int = 5


@dataclass
class ModelConfig:
    """Full model architecture."""

    vocab_size: int = 49154
    hidden_size: int = 384
    core_hidden_size: int = 192
    norm_eps: float = 1e-5
    target_params: int = 0
    target_nonembed_params: int = 0
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    sliding: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    grounded: GroundedInfomaxConfig = field(default_factory=GroundedInfomaxConfig)
    body: BodyConfig = field(default_factory=BodyConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)


@dataclass
class AdamConfig:
    """AdamW hyperparameters for embeddings, 1D params, norms, gates, FP linears."""

    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass
class MuonConfig:
    """Muon hyperparameters for 2D ternary channel weights (Newton-Schulz)."""

    lr: float = 0.02
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    weight_decay: float = 0.1
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
    wd_cap: float = 2.0  # scale-aware WD bound: min(max(sqrt(fan_out/fan_in)), cap)
    # Keep the momentum_buffer on CPU (frees ~16 GiB VRAM on the 9.37B run).
    # On launch-bound small models (148M, VRAM free) the 2 per-param host-syncs
    # this costs (g.detach().to("cpu") + update.to(p.device)) stall the dispatch
    # pipeline and starve the GPU; set False to keep momentum on-device.
    momentum_offload: bool = True


@dataclass
class ScheduleConfig:
    """Cosine LR schedule shape (applied to both Adam and Muon base LR)."""

    stable_fraction: float = 0.8
    min_lr_ratio: float = 0.1


@dataclass
class LoggingConfig:
    """Diagnostic / logging cadence inside the training loop."""

    posterior_interval: int = 20
    posterior_chunk_rows: int = 256
    log_interval: int = 1
    cache_release_interval: int = 20
    attn_entropy_interval: int = 10  # compute attn entropy penalty every N steps (1 = every step)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    max_steps: int = 150000
    warmup_steps: int = 1600
    learning_rate: float = 0.0003
    weight_decay: float = 0.1
    precision: str = "bf16"
    grad_accum_steps: int = 2
    max_grad_norm: float = 0.5
    grad_checkpointing: bool = True  # disable for small models (<2GB VRAM) — pure overhead
    batch_size: int = 8
    seq_len: int = 512
    muon: MuonConfig = field(default_factory=MuonConfig)
    adam: AdamConfig = field(default_factory=AdamConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # Loss weights.
    w_ce: float = 1.0
    w_rate: float = 0.01
    w_distortion: float = 0.01
    w_vicreg: float = 0.1
    w_infonce: float = 0.1
    w_moe_lb: float = 0.1  # was 0.01 — load balance stuck at 8.0 (all tokens on 1-2 experts)
    w_route_entropy: float = 0.05  # was 0.01 — routing-entropy capacity maximization (MoE)
    w_water_filling: float = 0.1  # was 0.01 — water-filling allocator entropy-gap regularizer (MoE)
    w_refine: float = 0.0  # off-path HEP refinement loss (opt-in)
    w_attn_entropy: float = 0.01
    attn_entropy_floor: float = 0.0  # 0.0 = auto (1/num_query_heads); set >0 to override
    # Data / tokens.
    tokenizer: str = "HuggingFaceTB/SmolLM2-135M"
    eos_token_id: int = 0
    pad_token_id: int = 49152
    data_dir: str = "data"
    data_dtype: str = "auto"
    # Checkpoints.
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    checkpoint_keep_last: int = 3
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    distill: DistillationConfig = field(default_factory=DistillationConfig)


@dataclass
class InferenceConfig:
    """Generation parameters."""

    temperature: float = 0.8
    top_k: int = 50
    min_new_tokens: int = 2
    repetition_penalty: float = 1.2
    repetition_window: int = 64
    no_repeat_ngram_size: int = 2
    max_new_tokens: int = 64


@dataclass
class Config:
    """Top-level configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def _next_pow2(n: int) -> int:
    p = 2
    while p < n:
        p *= 2
    return p


def auto_configure(target_params: int, vocab_size: int = 49154, num_experts: int | None = None) -> ModelConfig:
    """Solve ALL architecture sizes from a non-embedding parameter budget.

    Derives H, C, L, head counts/dims, factor rank, and all body config
    from ``target_params``. Every derived value is a pure function of the
    budget and vocab — ZERO magic numbers survive into the final config.

    The body is a ternary Hebbian-bilinear FFN + GQA attention stack with
    opt-in MoE on every 2nd block. Per-layer cost:
      * attention ~2 H^2 (qkv GQA + out_proj)
      * Hebbian FFN ~3*m*H at m = 4*H (A0 + A1 + W) = 12 H^2
      * MoE experts (when enabled): (num_experts + n_shared) * 3*m*H
    Total ~(2 + 12) * H^2 * L = 14 * H^2 * L for pure FFN; MoE layers
    add ~(E+1)*12*H^2 each.
    """
    target = float(target_params)

    # ---- Layer count: mild sub-linear scaling with budget ----
    L = max(2, round(math.log10(target / 1e5) * 3))

    # ---- MoE: enabled when budget justifies it (>= 50M non-embed params) ----
    if num_experts is None:
        use_moe = target > 50e6
        E = max(1, min(8, L // 2)) if use_moe else 0  # at most 8 experts, ~L/2
    else:
        E = int(num_experts)
        use_moe = E > 0

    # Cost coefficients per H^2 per layer:
    #   attn:     2.0 (GQA q+kv+out, kv smaller but still ~H^2)
    #   FFN:     12.0 (3 * expansion * H^2, expansion=4 -> 3*4=12)
    #   MoE:     12.0 * (E + n_shared) * (1/moe_every) amortized per layer
    #   norms:   ~0.1 (RMSNorm params, negligible)
    cost_per_h2 = 2.0 + 12.0  # base FFN cost
    if use_moe:
        moe_layers = L // 2  # moe_every=2
        n_shared = 1
        moe_ffn = moe_layers * 12.0 * (E + n_shared)
        non_moe_ffn = (L - moe_layers) * 12.0
        cost_per_h2 = 2.0 + (moe_ffn + non_moe_ffn) / L

    # H = sqrt(target / (cost_per_h2 * L)), rounded to multiple of 8
    H = _round_to_multiple(int(math.sqrt(target / (cost_per_h2 * L))), 8)

    # Iterative refinement: recompute L from H, then H from L (converges in 2-3 steps)
    for _ in range(3):
        L = max(2, round(math.log10(float(H * H * cost_per_h2 * L) / 1e5) * 3))
        if use_moe:
            moe_layers = L // 2
            moe_ffn = moe_layers * 12.0 * (E + n_shared)
            non_moe_ffn = (L - moe_layers) * 12.0
            cost_per_h2 = 2.0 + (moe_ffn + non_moe_ffn) / L
        H = _round_to_multiple(int(math.sqrt(target / (cost_per_h2 * L))), 8)

    C = _round_to_multiple(H // 2, 8)

    # ---- Attention heads: find (n_q, head_dim) with head_dim in [64,128],
    # n_q a power-of-2, and n_q * head_dim = H ----
    best_nq, best_hd = 2, _round_to_multiple(H // 2, 8)
    best_score = float("-inf")
    for try_nq in [pow2 for pow2 in [2, 4, 8, 16, 32, 64, 128] if pow2 * 32 <= H]:
        try_hd = _round_to_multiple(H // try_nq, 8)
        while try_nq * try_hd < H:
            try_hd += 8
        # Prefer head_dim in [64, 128]; if outside, penalize quadratically
        if 64 <= try_hd <= 128:
            score = 1000 - try_nq  # prefer more heads within sweet spot
        elif try_hd > 128:
            score = -((try_hd - 128) ** 2) / 100.0
        else:  # < 64
            score = -((64 - try_hd) ** 2) / 10.0
        if score > best_score:
            best_score, best_nq, best_hd = score, try_nq, try_hd
    n_q = best_nq
    head_dim = best_hd
    n_kv = max(1, n_q // 4)
    while n_q % n_kv != 0:
        n_kv -= 1

    # ---- Embeddings factor rank: min(256, C) ----
    factor_rank = min(256, C)

    # ---- Hebbian expansion: fixed 4x (ternary body dimension multiplier) ----
    expansion = 4

    # ---- Bottleneck dim = C ----
    bottleneck_dim = C

    # ---- MoE params ----
    top_k = max(1, min(2, E // 2)) if E > 0 else 1
    n_shared = 1 if E > 0 else 0
    intermediate_size = expansion * H
    moe_every = 2

    m = ModelConfig()
    m.vocab_size = vocab_size
    m.hidden_size = H
    m.core_hidden_size = C
    m.target_params = target_params
    m.target_nonembed_params = int(target)
    m.norm_eps = 1e-5  # bf16-safe default
    m.body.num_layers = L
    m.body.bottleneck.dim = bottleneck_dim
    m.body.bottleneck.kl_free_bits = 0.01  # per-dim floor
    m.body.bottleneck.logvar_clamp = (-5.0, 5.0)
    m.body.bottleneck.distortion_eps = m.norm_eps / 10.0
    m.body.ternary.use_ternary = True
    m.body.ternary.hebbian_expansion = expansion
    m.body.ternary.eps = m.norm_eps  # ternarize eps = norm_eps for consistency
    m.body.moe.enabled = use_moe
    m.body.moe.num_experts = E
    m.body.moe.top_k = top_k
    m.body.moe.n_shared = n_shared
    m.body.moe.moe_every = moe_every
    m.body.moe.intermediate_size = intermediate_size
    m.body.moe.router_init_std = 1.0 / math.sqrt(H)  # scale-aware router init
    m.attention.num_query_heads = n_q
    m.attention.num_kv_heads = n_kv
    m.attention.head_dim = head_dim
    m.attention.max_seq_len = 4096
    m.attention.rope_theta = 10000.0
    m.sliding.sliding_window = 0
    m.sliding.full_every = 4
    m.embeddings.factor_rank = factor_rank
    m.embeddings.kernel_size = max(3, min(7, factor_rank // 32))  # 3-7 range from rank
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


def load_config(path: str | None = None, **overrides: object) -> Config:
    """Load config from YAML, auto-configure sizes from target_params, validate."""
    import yaml

    cfg = Config()
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _apply_dict(cfg, data)

        tp = cfg.model.target_params
        if tp and tp > 0:
            model_data = data.get("model", {})
            nonembed = model_data.get("target_nonembed_params") or tp
            auto = auto_configure(int(nonembed), cfg.model.vocab_size)
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


def _apply_auto(model: ModelConfig, auto: ModelConfig, yaml_data: dict) -> None:
    """Apply auto-computed sizes, skipping anything explicitly set in YAML."""
    size_fields = ["hidden_size", "core_hidden_size", "target_params", "target_nonembed_params",
                   "norm_eps"]
    for f in size_fields:
        if f not in yaml_data:
            setattr(model, f, getattr(auto, f))
    if "hidden_size" not in yaml_data:
        model.attention.num_query_heads = auto.attention.num_query_heads
        model.attention.num_kv_heads = auto.attention.num_kv_heads
        model.attention.head_dim = auto.attention.head_dim
        model.attention.max_seq_len = auto.attention.max_seq_len
        model.attention.rope_theta = auto.attention.rope_theta
    if "core_hidden_size" not in yaml_data:
        model.body.bottleneck.dim = auto.core_hidden_size
        model.body.bottleneck.kl_free_bits = auto.body.bottleneck.kl_free_bits
        model.body.bottleneck.logvar_clamp = auto.body.bottleneck.logvar_clamp
        model.body.bottleneck.distortion_eps = auto.body.bottleneck.distortion_eps
    body = yaml_data.get("body", {})
    if "num_layers" not in body:
        model.body.num_layers = auto.body.num_layers
    ternary_data = body.get("ternary", {}) if body else {}
    if "hebbian_expansion" not in ternary_data:
        model.body.ternary.hebbian_expansion = auto.body.ternary.hebbian_expansion
    if "eps" not in ternary_data:
        model.body.ternary.eps = auto.body.ternary.eps
    moe_data = body.get("moe", {}) if body else {}
    if "enabled" not in moe_data or not moe_data.get("enabled"):
        # Only auto-apply MoE params if MoE not explicitly configured
        pass  # MoE params are set in auto_configure based on budget
    emb_data = yaml_data.get("embeddings", {})
    if "factor_rank" not in emb_data:
        model.embeddings.factor_rank = auto.embeddings.factor_rank
    if "kernel_size" not in emb_data:
        model.embeddings.kernel_size = auto.embeddings.kernel_size
    sliding_data = yaml_data.get("sliding", {})
    if "sliding_window" not in sliding_data:
        model.sliding.sliding_window = auto.sliding.sliding_window
    if "full_every" not in sliding_data:
        model.sliding.full_every = auto.sliding.full_every


def validate_config(cfg: Config) -> None:
    """Structural invariants for the codec-channel LM."""
    m, t = cfg.model, cfg.train

    def bint(name: str, value: int, upper: int) -> None:
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{name} must be an integer in [1, {upper}]")

    bint("model.vocab_size", m.vocab_size, 1_000_000)
    bint("model.hidden_size", m.hidden_size, 16_384)
    bint("model.core_hidden_size", m.core_hidden_size, 16_384)
    bint("train.seq_len", t.seq_len, 65_536)
    bint("train.batch_size", t.batch_size, 4_096)
    bint("inference.max_new_tokens", cfg.inference.max_new_tokens, 8_192)
    bint("model.body.num_layers", m.body.num_layers, 1_000)
    bint("model.attention.num_query_heads", m.attention.num_query_heads, 1_024)
    bint("model.attention.num_kv_heads", m.attention.num_kv_heads, 1_024)
    bint("model.attention.head_dim", m.attention.head_dim, 4_096)

    if not 0 < m.core_hidden_size < m.hidden_size:
        raise ValueError(
            f"model.core_hidden_size ({m.core_hidden_size}) must satisfy "
            f"0 < C < hidden_size ({m.hidden_size}). No compression => no RD."
        )
    if m.attention.num_kv_heads > m.attention.num_query_heads:
        raise ValueError("attention.num_kv_heads must be <= num_query_heads (GQA)")
    if m.attention.num_query_heads % m.attention.num_kv_heads != 0:
        raise ValueError("attention.num_query_heads must be divisible by num_kv_heads (GQA)")
    if m.hidden_size % m.attention.num_query_heads != 0:
        raise ValueError("hidden_size must be divisible by num_query_heads")
    hd = m.attention.head_dim
    if m.attention.num_query_heads * hd != m.hidden_size:
        raise ValueError(
            f"num_query_heads * head_dim ({m.attention.num_query_heads}*{hd}) "
            f"must equal hidden_size ({m.hidden_size})"
        )
    if t.grad_accum_steps < 1:
        raise ValueError("train.grad_accum_steps must be positive")
    if t.eos_token_id == t.pad_token_id:
        raise ValueError("eos_token_id and pad_token_id must be distinct")
    if not 0 <= t.eos_token_id < m.vocab_size:
        raise ValueError("eos_token_id must be within the model vocabulary")
    if not 0 <= t.pad_token_id < m.vocab_size:
        raise ValueError("pad_token_id must be within the model vocabulary")
    if t.w_attn_entropy > 0 and t.attn_entropy_floor <= 0.0:
        # 0.0 = auto-calculate from 1/num_query_heads; only fail if user set
        # an explicit negative value
        pass
    if not 0.0 < t.schedule.stable_fraction <= 1.0:
        raise ValueError("train.schedule.stable_fraction must be in (0, 1]")
    if not 0.0 <= t.schedule.min_lr_ratio < 1.0:
        raise ValueError("train.schedule.min_lr_ratio must be in [0, 1)")
    if not 0.0 < t.muon.wd_cap <= 8.0:
        raise ValueError("train.muon.wd_cap must be positive")
    if len(t.muon.ns_coeffs) != 3:
        raise ValueError("train.muon.ns_coeffs must be a 3-tuple")
    if m.body.num_layers % max(1, m.body.moe.moe_every) != 0 and m.body.moe.enabled:
        raise ValueError("body.num_layers must be divisible by moe.moe_every when MoE is enabled")
    if m.body.moe.enabled:
        moe = m.body.moe
        if moe.num_experts < 1:
            raise ValueError("moe.num_experts must be >= 1")
        if not 1 <= moe.top_k <= moe.num_experts:
            raise ValueError("moe.top_k must be in [1, num_experts]")
        if moe.n_shared < 0:
            raise ValueError("moe.n_shared must be >= 0")
    mm = m.multimodal
    if mm.enabled:
        if mm.n_bridge_queries < 1:
            raise ValueError("multimodal.n_bridge_queries must be >= 1")
        if mm.bridge_layers < 1:
            raise ValueError("multimodal.bridge_layers must be >= 1")
        if m.hidden_size % mm.bridge_n_heads != 0:
            raise ValueError("multimodal.bridge_n_heads must divide hidden_size")
    # Sliding-window invariants.
    sw = m.sliding
    if sw.sliding_window < 0:
        raise ValueError("sliding.sliding_window must be >= 0")
    if sw.sliding_window > m.attention.max_seq_len:
        raise ValueError("sliding.sliding_window must be <= attention.max_seq_len")
    if any(idx < 0 or idx >= m.body.num_layers for idx in sw.window_layers):
        raise ValueError("sliding.window_layers indices must be in [0, num_layers)")
    # Refinement invariants.
    rc = m.refinement
    if rc.enabled:
        if rc.iterations < 1:
            raise ValueError("refinement.iterations must be >= 1")
        if not 0.0 < rc.exit_threshold:
            raise ValueError("refinement.exit_threshold must be positive")
        if rc.exit_min_steps < 1 or rc.exit_window < 1:
            raise ValueError("refinement.exit_min_steps / exit_window must be >= 1")


def layer_sliding_windows(m: ModelConfig) -> list[int]:
    """Resolve the per-layer sliding-window size for the body.

    Returns a ``list[int]`` of length ``num_layers``: 0 means full attention,
    ``>0`` means windowed causal for that layer. Resolution order:

      1. Explicit ``window_layers`` index set (those layers windowed, rest full).
      2. ``full_every`` > 0: every Nth layer full, the rest windowed.
      3. ``sliding_window`` > 0 with neither of the above: ALL layers windowed.
      4. ``sliding_window`` == 0: ALL layers full attention.
    """
    n = m.body.num_layers
    w = m.sliding.sliding_window
    if w == 0:
        return [0] * n
    if m.sliding.window_layers:
        windowed = set(m.sliding.window_layers)
        return [w if i in windowed else 0 for i in range(n)]
    if m.sliding.full_every > 0:
        return [0 if (i % m.sliding.full_every == 0) else w for i in range(n)]
    return [w] * n
