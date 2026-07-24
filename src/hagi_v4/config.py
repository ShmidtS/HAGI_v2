"""HAGI configuration — ternary RD-channel language model.

The model is a causal autoregressive LM framed as a communication channel:
  * factorized source encoder (ConvEmbedding, CAUSAL conv — no future leak)
  * ternary BitNet b1.58 transformer body (the genuine discrete channel;
    quantization noise is the only impairment — there is no self-inflicted
    AWGN/LDPC physical channel)
  * auxiliary variational information bottleneck (KL rate regularizer, kept
    OUT of the main LM path — inserting it deadlocked from-scratch training)
  * optional predictive decoder (extrinsic error highway, off the main path)
  * optional multimodal source coding (per-modality encoder + early fusion)

All knobs live here. ``auto_configure`` solves the hidden/layer sizes from a
parameter budget; everything else is an explicit config value (no hidden
hardcoded constants scattered through the training loop).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AttentionConfig:
    """Grouped-query attention (bidir / causal / prefix / soft_causal at train)."""

    num_query_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 64
    rope_theta: float = 10000.0
    max_seq_len: int = 4096


@dataclass
class EmbeddingsConfig:
    """Factorized source encoder V x r + r x H + causal depthwise Conv1d filter."""

    factor_rank: int = 128
    kernel_size: int = 5
    use_conv_embedding: bool = True
    trainable: bool = True


@dataclass
class MultimodalConfig:
    """Per-modality source encoders + early sequence concatenation."""

    enabled: bool = False
    image_patch_size: int = 16
    image_channels: int = 3
    audio_mel_bins: int = 80
    modality_embed_dim: int = 32
    num_modalities: int = 3
    modality_embed_std: float = 0.02
    max_image_patches: int = 1024
    max_audio_frames: int = 1024


@dataclass
class BottleneckConfig:
    """Auxiliary variational information bottleneck (H->C, off the main LM path).

    The IB computes KL/distortion/perception on the context hidden as an
    auxiliary regularizer. It does NOT intercept the LM signal — inserting it
    in the main path deadlocks from-scratch training.
    """

    bottleneck_in_path: bool = False  # ablation: restore the failed in-path design


@dataclass
class PredictiveConfig:
    """Optional extrinsic error highway (off the main path by default)."""

    enabled: bool = False  # off the main path; opt-in for ablation/research
    train_iterations: int = 2
    infer_iterations: int = 4
    convergence_threshold: float = 0.01
    update_hidden: int = 256
    use_kalman_blend: bool = True
    hep_enabled: bool = True


@dataclass
class TernaryConfig:
    """BitNet b1.58 ternary body (the genuine discrete channel)."""

    use_ternary: bool = True
    hebbian_expansion: int = 4  # m = expansion * H
    hebbian_dropout: float = 0.0


@dataclass
class BodyConfig:
    """Ternary transformer body: context (perception) + expression stacks."""

    context_layers: int = 8
    expression_layers: int = 8
    ternary: TernaryConfig = field(default_factory=TernaryConfig)
    bottleneck: BottleneckConfig = field(default_factory=BottleneckConfig)
    predictive: PredictiveConfig = field(default_factory=PredictiveConfig)
    # Rate-distortion loss weights (the only genuine "rate" is the IB KL).
    ib_beta: float = 0.001
    distortion_weight: float = 1.0
    perception_weight: float = 0.01
    kl_free_bits: float = 0.5
    logvar_clamp: tuple[float, float] = (-10.0, 10.0)
    distortion_eps: float = 1e-6
    moe_enabled: bool = False  # dropped (YAGNI); flag retained for opt-in


@dataclass
class DistillationConfig:
    """Online hidden-state distillation from a causal teacher LM."""

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
    """Two-stage dataset curriculum + attention-mode mixing.

    The model is a CAUSAL generative LM; attention_mode stays causal-dominant
    from step 0 (bidir/soft_causal slices add a denser representation gradient).
    """

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
class ModelConfig:
    """Full model architecture."""

    vocab_size: int = 49154
    hidden_size: int = 384
    core_hidden_size: int = 192
    norm_eps: float = 1e-6
    target_params: int = 0
    target_nonembed_params: int = 0
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    body: BodyConfig = field(default_factory=BodyConfig)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    max_steps: int = 150000
    warmup_steps: int = 1600
    learning_rate: float = 0.0003
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.5
    muon_ns_steps: int = 5
    weight_decay: float = 0.1
    precision: str = "bf16"
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    batch_size: int = 8
    seq_len: int = 512
    # Loss weights.
    w_ce: float = 1.0
    w_rate: float = 0.001
    w_distortion: float = 1.0
    w_perception: float = 0.01
    w_ternary_bias: float = 0.0
    w_moe_load_balance: float = 0.01
    w_attn_entropy: float = 0.01
    attn_entropy_floor: float = 0.5
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


# Backwards-compatible alias for callers that import HAGIv4Config.
HAGIv4Config = Config


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def _next_pow2(n: int) -> int:
    p = 2
    while p < n:
        p *= 2
    return p


def auto_configure(target_params: int, vocab_size: int = 49154) -> ModelConfig:
    """Solve hidden/layer sizes from a non-embedding parameter budget.

    The body is a ternary Hebbian-bilinear FFN + attention stack. Cost per
    layer is dominated by the FFN (A0, A1: H x 4H; W: 4H x H) and attention
    qkv+out (4 H^2). Two stacks (context + expression) each of depth L.
    """
    target = float(target_params)
    layers = max(2, round(math.log10(target / 1e5) * 3))
    perc = max(1, layers // 2)
    expr = perc
    ffn_cost_per_layer = 2 * (_round_to_multiple(int(target ** 0.5)) * 4)  # rough
    # Per-H^2 coefficient: attn ~4, FFN ~8 (A0+A1+W at m=4H => 3*4H^2 ~ 12, /2 master)
    cost_per_h_sq = 4.0 + 6.0
    H = _round_to_multiple(int(math.sqrt(target / (cost_per_h_sq * (perc + expr)))), 8)
    C = _round_to_multiple(H // 2, 8)
    head_dim = _round_to_multiple(64, 8)
    n_q = max(2, _next_pow2(H // head_dim))
    head_dim = _round_to_multiple(H // n_q, 8)
    while n_q * head_dim < H:
        head_dim += 8
    del ffn_cost_per_layer

    m = ModelConfig()
    m.vocab_size = vocab_size
    m.hidden_size = H
    m.core_hidden_size = C
    m.target_params = target_params
    m.target_nonembed_params = int(target)
    m.body.context_layers = perc
    m.body.expression_layers = expr
    m.attention.num_query_heads = n_q
    m.attention.num_kv_heads = max(1, n_q // 2)
    m.attention.head_dim = head_dim
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
    yaml_keys = {"train": set(), "inference": set()}
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        yaml_keys["train"] = set(data.get("train", {}).keys())
        yaml_keys["inference"] = set(data.get("inference", {}).keys())
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
    size_fields = ["hidden_size", "core_hidden_size", "target_params", "target_nonembed_params"]
    for f in size_fields:
        if f not in yaml_data:
            setattr(model, f, getattr(auto, f))
    if "hidden_size" not in yaml_data:
        model.attention.num_query_heads = auto.attention.num_query_heads
        model.attention.num_kv_heads = auto.attention.num_kv_heads
        model.attention.head_dim = auto.attention.head_dim
    body = yaml_data.get("body", {})
    if "context_layers" not in body:
        model.body.context_layers = auto.body.context_layers
    if "expression_layers" not in body:
        model.body.expression_layers = auto.body.expression_layers


def validate_config(cfg: Config) -> None:
    """Structural invariants for the ternary RD-channel LM."""
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
    if not 0 < m.core_hidden_size < m.hidden_size:
        raise ValueError(
            f"model.core_hidden_size ({m.core_hidden_size}) must satisfy "
            f"0 < C < hidden_size ({m.hidden_size}). No compression => no RD."
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
        raise ValueError("train.attn_entropy_floor must be > 0 when w_attn_entropy > 0")
    if t.w_perception < 0.0:
        raise ValueError("train.w_perception must be >= 0 (RDP perception axis)")
