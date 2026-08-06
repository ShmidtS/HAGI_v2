"""HAGI V40 configuration -- source-matched conditional channel.

The current SSOT is source coder -> ternary body -> conditional NCE receiver,
with periodic exact full-alphabet CE as the coding-cost calibration channel.
Every architectural knob lives here; the history below records why removed
constraints are not silently reintroduced.

    source symbols          tokens
    source code             tied codebook + unigram logit prior
    transmit filter         causal depthwise conv (pulse shaping)
    channel                 ternary (b1.58) transformer stack
    correlator              QK-normalized GQA
    variable-rate coding    MoE with bias-corrected (auxiliary-loss-free) routing
    finite-state channel    sliding-window layers, full-attention relays
    receiver                tied LM head + chunked cross-entropy

What V31 removes from V28/V30, and why (all of it was measured, not guessed):

  * **Variational information bottleneck** (``rate`` / ``distortion`` /
    ``memory_bank``). With a ternary channel the rate constraint already lives
    in the weights at log2(3) bits each. A second KL on the hidden double-counts
    the same constraint; in two runs it posterior-collapsed (rate -> 0.02,
    h_hat -> 0) and both shipped configs had ``ib_beta=0``.
  * **Off-path HEP refinement + EXIT halt.** Never enabled (``w_refine=0`` in
    every config). Its only live effect was gating a beta-anneal for a loss term
    that was itself disabled.
  * **SNR-gated water-filling MoE.** The gate multiplied routing logits by
    ``1 + 1/||residual||``, which is a *temperature reduction*, and the
    allocator added a log-bias toward experts whose residual was already small.
    Together: rich-get-richer with no opposing force. Measured ``moe_lb = 8.2``
    against a ceiling of 8.0 for E=4/top_k=2 — total collapse.
  * **Rank-r factored LM head.** A rank-128 head cannot represent the
    conditional distribution: fitting a known full-rank target left 1.42 nats of
    residual KL at r=128 versus 0.92 at r>=512. That is a permanent ~0.5
    nats/token floor. V31 ties one full-rank codebook to the head instead.
  * **Per-document padding.** Measured sequence utilization 0.41-0.80 at T=512
    and 0.10-0.43 at T=2048: most of the compute was spent on PAD. V31 packs
    documents contiguously and separates them with a block-diagonal causal mask.

What V31 adds:

  * Unigram logit prior (zero-order source code; the network learns only the
    conditional correction — worth ~4.4 nats/token at initialization).
  * QK-norm (keeps the softmax out of saturation under matrix-sign updates).
  * Auxiliary-loss-free load balancing (routing entropy maximized by a bias
    controller, so no gradient competes with the LM signal).
  * Optional vocabulary compaction (85,632 of 262,144 symbols are unreachable
    in this corpus; emitting logits for them wastes head FLOPs and forces the
    model to spend capacity suppressing them).
  * z-loss on the log-partition function (bounds normalization drift).
  * Chunked fused cross-entropy (never materializes an ``[N, V]`` logit tensor).
  * WSD learning-rate schedule (horizon-free: the stable phase can be extended
    and only the cooldown depends on the total step budget).

Checkpoint format 12. V41 changes receiver sampling semantics and trains from scratch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

CHECKPOINT_FORMAT_VERSION = 12


@dataclass
class AttentionConfig:
    """Grouped-query attention geometry.

    ``num_query_heads * head_dim`` must equal ``hidden_size``; ``num_kv_heads``
    must divide ``num_query_heads``. GQA shrinks the KV-cache by
    ``num_query_heads / num_kv_heads`` at nearly no quality cost, which is what
    makes long-context decoding affordable.
    """

    num_query_heads: int = 24
    num_kv_heads: int = 6
    head_dim: int = 64
    rope_theta: float = 10000.0
    max_seq_len: int = 4096
    qk_norm: bool = True


@dataclass
class SlidingWindowConfig:
    """Hybrid local/global attention placement.

    A windowed layer is a finite-state channel: cost O(T*W), memory bounded by
    W. Full-attention layers are the global parity relays that carry long-range
    dependence. ``full_every=4`` means layers 0, 4, 8, ... are full and the rest
    are windowed — the standard 1:3 ratio.

    Attributes:
        window: window size W; 0 disables windowing everywhere.
        full_every: every Nth layer is full attention (N<=1 means all full).
    """

    window: int = 0
    full_every: int = 4
    history_stride: int = 0


@dataclass
class EmbeddingConfig:
    """Source codebook and transmit filter.

    Attributes:
        tie_lm_head: share one ``[V, H]`` codebook between the encoder and the
            LM head. At V=262144 this is the difference between paying for the
            table once or twice, and it removes the rank bottleneck that a
            factored head imposes.
        conv_kernel: causal depthwise Conv1d width (pulse shaping over adjacent
            symbols). 1 disables it. Left-padded only: a symmetric pad leaks
            future tokens into position t and silently destroys the causal
            objective.
        init_std: codebook init standard deviation.
    """

    tie_lm_head: bool = True
    conv_kernel: int = 4
    init_std: float = 0.02


@dataclass
class FFNConfig:
    """SwiGLU channel mixer width.

    ``intermediate_size = 0`` derives the width from ``expansion * H`` rounded
    up to ``multiple_of`` (matmul alignment). The 8/3 default keeps SwiGLU's
    three matrices at the same parameter count as a 4x two-matrix MLP.
    """

    intermediate_size: int = 0
    expansion: float = 8.0 / 3.0
    multiple_of: int = 128


@dataclass
class MoEConfig:
    """Mixture of experts with auxiliary-loss-free load balancing.

    MoE is the only lever that grows capacity without growing per-token
    compute, which makes it the scaling mechanism. Variable-rate coding: each
    token is assigned to the experts that code it best.

    Balancing uses a per-expert *bias* added to the routing logits before
    top-k, updated by a controller rather than by gradient descent:

        b_e <- b_e + gamma * sign(load_target - load_e)

    The bias affects selection only, never the combining weights, so it steers
    routing entropy toward the maximum without adding a gradient that competes
    with the language-modelling signal. This is why V28's auxiliary CV^2 loss
    could sit at its ceiling for 50k steps without correcting anything: it was
    fighting the LM gradient and losing.

    Attributes:
        enabled: replace the dense FFN with MoE on selected layers.
        num_experts: number of routed experts E.
        top_k: experts activated per token.
        n_shared: always-on experts carrying the common code (their output is
            added unconditionally, so routed experts specialize on what is left).
        moe_every: MoE on every Nth layer; layer 0 and the last layer stay dense
            (routing on an un-mixed or final representation is unstable).
        bias_update_rate: gamma of the balancing controller. 0 disables it.
        router_init_std: 0 means ``1/sqrt(H)``.
        z_loss_weight: penalty on the router's log-partition function; keeps
            routing logits from drifting to a scale where top-k is numerically
            arbitrary.
    """

    enabled: bool = False
    num_experts: int = 8
    top_k: int = 2
    n_shared: int = 1
    moe_every: int = 2
    bias_update_rate: float = 0.001
    router_init_std: float = 0.0
    z_loss_weight: float = 0.001


@dataclass
class TernaryConfig:
    """BitNet b1.58 quantization of the 2D channel weights."""

    enabled: bool = True
    eps: float = 1e-5


@dataclass
class SpectralConfig:
    """Recurrent Fourier mixing — parallel causal complex filter.

    A damped-oscillator recurrence ``S_t = A * S_{t-1} + x_t`` with
    ``A = r * exp(-i*omega)`` is a first-order IIR whose impulse response is a
    complex sinusoid times a geometric envelope. Read as a channel, it is a
    sub-quadratic correlator: O(T*K) work and a K-dimensional state instead of
    the O(T^2) of full attention. Because it is a recurrence it is
    KV-cacheable at exactly ``2*K`` floats per layer (the complex state),
    independent of context length.

    The recurrence is evaluated in closed form with a *parallel scan* — no
    sequential loop over T. Using the identity::

        S(t) = A^tau * S_block_start + A^tau * sum_{j<=tau} x_j A^{-j}

    and the block recursion ``S_end = A^L * S_prev + local_end``, the full
    sequence is computed with ``T/L`` sequential block steps, each ``O(B*L*K)``,
    which is exact (no approximation) and ~2 orders of magnitude faster than
    ``torch.fft`` on this ROCm build.

    Why "Fourier": the frequency response of the filter at frequency ``phi`` is
    ``r / (1 - r*e^{-i(omega-phi)})`` — a Lorentzian peak centered at ``omega``
    with width set by ``r``. K oscillators are K band-pass filters; the layer
    learns *where* in the spectrum the sequence's energy lives. This is the
    spectral content that attention, which is shift-invariant per pair, cannot
    directly represent.

    **Grokking acceleration (spectral ramp).** Grokking (memorization then
    sudden generalization) is accelerated by starting the network with only the
    low-frequency modes active and releasing high frequencies as training
    progresses — the "spectral shift hypothesis". ``ramp_steps`` scales the
    oscillator bandwidths from narrow (low frequencies dominate) to wide over
    the first ``ramp_steps`` optimizer steps. It is implemented as a running
    scalar updated by the trainer (see ``commit_controller_updates``), so it
    needs no auxiliary loss.

    Attributes:
        enabled: add the spectral branch to selected layers.
        num_modes: K complex oscillators (band-pass filters).
        every: add the branch to every Nth layer (0 means none).
        block_len: parallel-scan block size (T/L sequential steps).
        freq_base: base frequency of the geometric oscillator spacing.
        freq_max: highest oscillator frequency (radians per step).
        damp_min, damp_max: retention r range — the filter's bandwidth.
        ramp_steps: optimizer steps over which oscillator bandwidths widen
            from ``damp_min`` (low-freq only) to ``damp_max`` (full spectrum).
        out_channels: readout width per mode (defaults to H/K); larger is a
            richer mixing of modes before the residual add.
        use_2d: treat the head dimension as a second (non-causal) frequency
            axis, giving a true 2D Fourier structure: row = temporal
            oscillator, column = harmonic channel mode.
    """

    enabled: bool = False
    num_modes: int = 32
    every: int = 2
    block_len: int = 64
    freq_base: float = 0.02
    freq_max: float = 3.0
    damp_min: float = 0.90
    damp_max: float = 0.999
    ramp_steps: int = 20000
    out_channels: int = 0
    use_2d: bool = True


@dataclass
class HeadConfig:
    """Receiver: LM head, source prior, and normalization control.

    Attributes:
        unigram_prior: add a fixed ``log p_unigram`` bias to the logits. The
            zero-order source code is free knowledge; without it the network
            spends its first thousands of steps rediscovering token frequencies.
            Requires ``unigram_path``.
        unigram_path: ``.npy`` of per-id counts (see scripts/count_unigram.py).
        unigram_smoothing: additive smoothing on the counts, so unseen ids get a
            finite (very negative) prior rather than ``-inf``.
        z_loss_weight: weight on ``log(sum exp(logits))^2``. The log-partition
            function has no effect on the distribution but every effect on
            numerical conditioning; pinning it near 0 keeps bf16 softmax exact.
        ce_chunk_rows: row block size for the chunked cross-entropy. Logits for
            a block are computed, consumed, and recomputed in backward, so peak
            memory is ``chunk * V`` instead of ``N * V``.
        ce_save_logits: store the chunked logits from forward and reuse them in
            backward instead of recomputing the projection. Saves ~40% of head
            backward time at a cost of ``N * V * 2`` bytes VRAM (2 GB at
            15k x 32768). Default off; turn on when VRAM is ample.
        logit_scale_init: initial receiver gain on the correlation
            ``<hidden, codebook_row>``. 0 means ``1/sqrt(H)``. A tied codebook
            makes the raw correlation start at ``H * embedding.init_std`` for the
            input token itself (the residual stream is still nearly collinear
            with its own code word at initialization), so at H=2048 the head
            emits a +41 logit before it has learned anything and the free
            unigram prior is destroyed. The gain is one learnable scalar; it
            starts the receiver at the prior and lets training raise the gain.
        sampled_softmax_k: number of *negative* classes drawn per scored row for
            sampled softmax (Jean et al.). 0 = full vocabulary CE (chunked).
            K>0 replaces the O(N·V) receiver with target dots plus a shared
            O(N·K) negative-bank GEMM. Biased but consistent as K→V.
        sampled_proposal: negative sampling distribution: ``uniform`` or
            ``prior``. The latter turns the sampled receiver into conditional
            NCE: it learns the likelihood ratio against the source prior, which
            is added back exactly once by the full-vocabulary receiver.
        logit_scale_max: post-step upper bound for the receiver gain. 0 disables
            the bound. This prevents sampled partitions from driving a gain that
            is unstable when generation returns to the full vocabulary.
    """

    unigram_prior: bool = False
    unigram_path: str = ""
    unigram_smoothing: float = 1.0
    z_loss_weight: float = 0.0001
    ce_chunk_rows: int = 4096
    ce_save_logits: bool = False
    logit_scale_init: float = 0.0
    sampled_softmax_k: int = 0
    sampled_proposal: str = "uniform"
    sampled_in_batch_fraction: float = 0.0
    logit_scale_max: float = 0.0


@dataclass
class MultimodalConfig:
    """Per-modality source coders plus a fixed-rate bridge.

    Separate source coding per modality (each has its own statistics), then a
    Q-Former-style bottleneck that compresses any modality to exactly
    ``n_bridge_queries`` tokens. Fixed rate is what makes this scalable: the
    text sequence grows by a constant regardless of image resolution or audio
    length.

    Attributes:
        enabled: turn the multimodal path on.
        image_patch_size, image_channels: ViT-style patchification.
        audio_mel_bins: mel filterbank width.
        n_bridge_queries: bridge output length per modality (the fixed rate).
        bridge_layers, bridge_heads: bridge depth/width.
        infonce_temperature: temperature of the cross-modal MI bound.
        variance_gamma: per-dimension standard-deviation floor of the anti-
            collapse hinge.
        modality_dropout: train-time probability of dropping each non-text
            modality independently. Forces the text channel to remain
            self-sufficient (no free-ride on always-present side information)
            and is the multimodal analogue of a Rayleigh fade.
        waterfill_enabled: reallocate bridge tokens across modalities by a
            soft water-filling rule on per-modality pooled energy (more tokens
            to the higher-SNR modality). When False every present modality
            gets exactly ``n_bridge_queries`` tokens.
        waterfill_min_rate: floor fraction of ``n_bridge_queries`` any live
            modality keeps under water-filling (prevents total starvation).
        grounding_weight: multiplier on the off-path InfoNCE+VICReg term.
    """

    enabled: bool = False
    image_patch_size: int = 16
    image_channels: int = 3
    audio_mel_bins: int = 80
    n_bridge_queries: int = 32
    bridge_layers: int = 2
    bridge_heads: int = 8
    infonce_temperature: float = 0.07
    variance_gamma: float = 1.0
    modality_dropout: float = 0.0
    waterfill_enabled: bool = False
    waterfill_min_rate: float = 0.25
    grounding_weight: float = 1.0


@dataclass
class XMConfig:
    """Explorative Modeling — best-of-K list decoding on high-entropy positions.

    Forward XM (Gladstone, Ji, Du — arxiv 2607.27372) factors the *training
    loop* instead of the generation procedure: fix a data target, draw K
    candidates from the model's own generation, train on the closest. This
    scales *generative expressivity* — how many modes a prediction can commit
    to — which ordinary next-token factorization fixes at design time.

    Channel reading: the LM head is a *list decoder*. Standard CE forces one
    hypothesis (the blur: the mean of all valid continuations, which matches no
    real continuation). Best-of-K emits K hypotheses (one per mode-code) and
    keeps the one nearest the received target — the maximum-likelihood decision
    over K codewords. Gradient flows only through the winner (Algorithm 1).

    Theory (Proposition 1, Appendix F): at large K this is maximum likelihood
    of a K-component mixture; K=1 is ordinary MLE. Gains grow with scale
    (4.1x FLOP, 6.2x sample, 47% parameter efficiency measured on image/video/
    language).

    On the bandwidth-bound Radeon 8060S the head *forward* costs 141ms vs
    551ms for head backward (both at N=30720, V=32768). Best-of-K adds only
    K-1 forward passes (cheap) while backward stays through the single winner —
    so exploration is nearly free relative to its quality gain.

    Attributes:
        enabled: turn best-of-K exploration on.
        num_modes: K learnable mode-codes (hypotheses per position).
        entropy_gate: explore only positions whose conditional entropy exceeds
            this fraction of the *achievable* maximum — the unigram entropy when
            a source prior is loaded (8.06 nats on this corpus), else ln(V).
            Where the next token is unambiguous there is nothing to explore;
            spending the budget there is wasted forward passes. 0.0 explores
            everywhere.
        explore_fraction: additionally keep only the ``explore_fraction``
            highest-entropy positions *within* the gated set. At initialization
            the model is near-uniform, so entropy_gate alone fires on ~100% of
            positions and best-of-K costs K full head passes; capping the
            fraction bounds the exploration budget while still targeting the
            genuinely ambiguous tail.
        mode_std: init std of the mode-code embeddings (small, so a mode is a
            perturbation of the base prediction, not a new prediction).
        min_modes: positions below ``entropy_gate`` still get this many
            candidates (1 = the base prediction, i.e. ordinary CE).
    """

    enabled: bool = False
    num_modes: int = 4
    entropy_gate: float = 0.7
    explore_fraction: float = 0.2
    mode_std: float = 0.05
    min_modes: int = 1


@dataclass
class ModelConfig:
    """Full architecture."""

    vocab_size: int = 262144
    hidden_size: int = 1536
    num_layers: int = 24
    # Weight-tied loop depth. Each of the ``num_layers`` unique blocks is applied
    # ``loop_depth`` times in sequence (Mobius / DeepLoop). Effective depth is
    # ``num_layers * loop_depth`` while parameter count and optimizer state stay
    # at ``num_layers``. Residual scale is computed from the *effective* depth
    # so stream variance stays O(1). 1 disables looping (standard stack).
    loop_depth: int = 1
    norm_eps: float = 1e-5
    target_params: int = 0
    # CDMA/OFDM precoding init: give every 2D channel weight a full-rank
    # orthogonal start (all singular values = 1) instead of i.i.d. Gaussian,
    # whose minimum singular value is ~0 at large width — a spectral hole in the
    # transmit matrix. Free (one-time QR), keeps every input direction
    # transmitted from step 0. Codebook (a source codec, not a precoder) and 1D
    # gains are excluded.
    init_orthogonal: bool = False
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    sliding: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    ternary: TernaryConfig = field(default_factory=TernaryConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    xm: XMConfig = field(default_factory=XMConfig)


@dataclass
class MuonConfig:
    """Muon: momentum SGD whose update is orthogonalized by Newton-Schulz.

    The orthogonalized update has all singular values equal to 1, so every
    direction of the weight matrix receives the same step size. On a ternary
    master that is exactly right: the quantizer only reads the *pattern* of
    signs relative to the row absmean, and an isotropic update explores that
    pattern efficiently.

    Muon removes plain SGD's implicit ``1/||W||`` brake, so the spectral norm
    drifts outward. For ternary weights the per-row absmean scale absorbs that
    drift (verified: effective RMS 0.0293 -> 0.0308 over 20k steps, i.e. flat),
    which is why no spectral cap is needed here.

    Attributes:
        lr: base learning rate (the schedule scales it).
        momentum, nesterov: heavy-ball parameters.
        ns_steps, ns_coeffs: quintic Newton-Schulz iteration.
        weight_decay: decoupled, scaled by ``min(sqrt(fan_out/fan_in), wd_cap)``
            so wide-output matrices are not under-regularized.
        wd_cap: upper bound of that scaling.
        momentum_offload: keep the momentum buffer in host memory. Saves VRAM
            proportional to the Muon parameter count, at the cost of two
            host syncs per parameter per step — a loss on small models where
            kernel launch latency already dominates.
    """

    lr: float = 0.02
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
    weight_decay: float = 0.1
    wd_cap: float = 2.0
    momentum_offload: bool = False


@dataclass
class AdamConfig:
    """AdamW for everything that is not a 2D ternary channel weight.

    Codebooks (sparse, extremely non-stationary gradients), 1D gains, biases,
    routers and floating-point projections. Per-coordinate adaptivity is the
    right prior there; orthogonalization is not defined for 1D and is wrong for
    a codebook whose rows are updated at wildly different frequencies.

    ``body_lr_scale``: the ternary channel weights (``is_channel_weight``) get
    ``learning_rate * body_lr_scale``. Measured on real data the codebook
    gradient norm is 20-30x the body's (``gr ~0.23`` vs ``gb ~0.008``), so on a
    shared AdamW the body barely moves. A separate, higher body LR is the
    cheap substitute for Muon's orthogonalization — it rebalances the two
    gradient scales without the Newton-Schulz cost. 1.0 disables it.
    """

    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    body_lr_scale: float = 8.0


@dataclass
class ScheduleConfig:
    """Warmup - stable - decay learning-rate schedule.

    WSD rather than cosine: the peak learning rate is tuned once, the stable
    phase can be extended after the fact, and only the cooldown depends on the
    total horizon. A cosine schedule bakes the horizon into every step, so
    extending a run means either re-tuning or accepting a mismatched curve.

    Attributes:
        warmup_steps: linear ramp from 0. Non-negotiable with a matrix-sign
            optimizer: the first orthogonalized steps are full-magnitude in
            every direction.
        decay_fraction: fraction of the horizon spent in the final linear
            cooldown.
        min_lr_ratio: floor as a fraction of peak.
        inverse_sqrt_stable: shape the stable phase as ``1/sqrt(1 + t/tau)``
            instead of a constant. Horizon-free and minimax-optimal in the
            stochastic convex setting; in practice it removes the late-training
            instability that a long constant phase at peak LR produces.
        inverse_sqrt_tau: 0 means ``max(warmup_steps, 1)``.
    """

    warmup_steps: int = 2000
    decay_fraction: float = 0.2
    min_lr_ratio: float = 0.02
    inverse_sqrt_stable: bool = True
    inverse_sqrt_tau: int = 0


@dataclass
class DataConfig:
    """Packed-corpus data pipeline.

    Documents are concatenated into a flat token stream and cut into fixed
    ``seq_len`` windows with no padding. Each window carries ``doc_ids`` so the
    attention mask is block-diagonal per document: full utilization of every
    position, and no cross-document leakage.

    The previous per-document scheme padded every short document to ``seq_len``.
    Measured utilization: 0.41 (tinystories) to 0.80 (edu) at T=512, and
    0.10-0.43 at T=2048. Packing recovers all of it.

    Attributes:
        data_dir: directory of ``<name>.bin`` (flat uint32) plus ``mix.json``.
        seq_len: training window length.
        eos_token_id: document delimiter present in the .bin stream.
        pad_token_id: retained for generation only; packing emits no padding.
        weights: per-source sampling weights; empty means read ``mix.json``.
        num_workers: dataloader workers.
        seed: sampling seed.
        cross_doc_attention: allow attention across document boundaries inside a
            packed window. False is correct (a document is an independent
            source); True is cheaper because it needs no mask.
    """

    data_dir: str = "data"
    seq_len: int = 1024
    eos_token_id: int = 1
    pad_token_id: int = 0
    weights: dict[str, float] = field(default_factory=dict)
    num_workers: int = 2
    seed: int = 1234
    cross_doc_attention: bool = False


@dataclass
class LoggingConfig:
    """Diagnostic cadence.

    ``diag_interval`` gates model-state diagnostics. ``exact_ce_interval``
    gates an exact full-alphabet receiver measurement on ``exact_ce_rows``
    randomly selected positions. The latter is the coding-cost SSOT when the
    training receiver uses a biased local partition; zero disables it.
    """

    log_interval: int = 10
    diag_interval: int = 100
    diag_chunk_rows: int = 512
    exact_ce_interval: int = 0
    exact_ce_rows: int = 512
    exact_ce_seed: int = 1729


@dataclass
class TrainConfig:
    """Training hyperparameters.

    The objective is a single term plus two conditioning penalties:

        loss = CE + z_loss_weight * z_loss + moe_z_loss_weight * router_z_loss

    There is no rate term, no distortion term, no refinement term and no
    load-balance term. Load balance is handled by the bias controller inside the
    router; the rate constraint is the ternary quantizer. Every objective that
    competes with CE for gradient must justify itself, and none of the removed
    ones could.
    """

    max_steps: int = 200000
    batch_size: int = 8
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    grad_checkpointing: bool = True
    compile_model: bool = False
    # Variance in bf16 (fused kernel) instead of fp32. Measured 5x faster on
    # ROCm and numerically identical (max diff 0.0) at the value ranges seen
    # in training, so the safe default is the fast path. Set false to restore
    # the fp32 accumulator.
    fp32_norm: bool = False
    # Muon's Newton-Schulz orthogonalization costs ~29% of step time on this
    # ROCm build (measured 0.70s of 2.41s). BitNet b1.58 weights are ternary:
    # the quantizer reads only the sign pattern relative to row absmean, so the
    # per-direction isotropy Muon provides buys nothing that AdamW does not.
    # Default off; Muon remains selectable for fp16/dense bodies where it helps.
    use_muon: bool = False
    # Freeze BitLinear ternary maps once per optimizer step across grad_accum
    # microbatches (OFDM coherence-interval analogy). Masters only change on
    # optimizer.step, so re-quantizing every microbatch is pure waste. STE via
    # W+(Q-W).detach() keeps the gradient on the master. No-op when accum=1.
    ternary_step_cache: bool = True
    # Punctured CE (erasure channel on the supervision stream). Fraction of
    # positions that contribute to the head CE each step. Body still sees the
    # full sequence; only the receiver cost is thinned. Unbiased Monte-Carlo
    # estimate of full CE under Bernoulli sampling (keep_mode=bernoulli);
    # stride is a regular lattice puncture. Measured on 8060S: keep=0.5 →
    # +27% body tok/s at fixed L=6 (head is ~1/3 of step and linear in N).
    # 1.0 = score every token (legacy).
    ce_keep_rate: float = 1.0
    # "bernoulli" | "stride". Stride keeps every round(1/rate)-th position with
    # a step-dependent phase so the lattice covers the sequence over time.
    ce_keep_mode: str = "bernoulli"
    muon: MuonConfig = field(default_factory=MuonConfig)
    adam: AdamConfig = field(default_factory=AdamConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    z_loss_weight: float = 0.0001
    moe_z_loss_weight: float = 0.001
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    checkpoint_keep_last: int = 3
    tokenizer: str = "google/gemma-4-E2B-it"


@dataclass
class InferenceConfig:
    """Generation parameters."""

    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    min_new_tokens: int = 1
    max_new_tokens: int = 512
    repetition_penalty: float = 1.05
    repetition_window: int = 128


@dataclass
class Config:
    """Top-level configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _round_up(value: int, multiple: int) -> int:
    return max(multiple, -(-int(value) // multiple) * multiple)


def ffn_width(cfg: ModelConfig) -> int:
    """Resolve the SwiGLU intermediate width (explicit, or derived from H)."""
    if cfg.ffn.intermediate_size > 0:
        return cfg.ffn.intermediate_size
    return _round_up(int(cfg.ffn.expansion * cfg.hidden_size), cfg.ffn.multiple_of)


def layer_windows(cfg: ModelConfig) -> list[int]:
    """Per-layer attention window: 0 = full attention, >0 = windowed causal.

    With ``full_every = N`` layers ``0, N, 2N, ...`` are full-attention relays
    and the rest are windowed. Layer 0 is deliberately always a relay: the first
    layer is where global structure must be available, and windowing it costs
    accuracy for almost no compute.
    """
    n, w = cfg.num_layers, cfg.sliding.window
    if w <= 0:
        return [0] * n
    every = cfg.sliding.full_every
    if every <= 1:
        return [0] * n
    return [0 if i % every == 0 else w for i in range(n)]


def moe_layers(cfg: ModelConfig) -> list[bool]:
    """Which layers use MoE.

    Layer 0 and the final layer stay dense. Routing on the first layer's output
    means routing on something barely different from the raw embedding; routing
    on the last means the router's decision is never refined downstream. Both
    are known-unstable placements.
    """
    n = cfg.num_layers
    if not cfg.moe.enabled:
        return [False] * n
    every = max(1, cfg.moe.moe_every)
    return [(i % every == every - 1) and 0 < i < n - 1 for i in range(n)]


def spectral_layers(cfg: ModelConfig) -> list[bool]:
    """Which layers add the recurrent spectral branch.

    Like MoE, layer 0 and the last layer stay pure: the spectral branch is a
    local-frequency refinement on an already-mixed signal, and routing it onto
    the raw embedding or the final pre-head state adds noise without a
    downstream layer to use the correction.
    """
    n = cfg.num_layers
    if not cfg.spectral.enabled:
        return [False] * n
    every = max(1, cfg.spectral.every)
    return [(i % every == every - 1) and 0 < i < n - 1 for i in range(n)]


def count_params(cfg: ModelConfig) -> dict[str, int]:
    """Analytic parameter count by group (no model instantiation).

    Separating embedding from body is the number that actually matters: a body
    too small relative to the codebook cannot learn language no matter how long
    it trains. The V30 155.6M configuration spent 43% of its parameters on the
    262k-entry table and 88M on the body — it plateaued at ce ~5 and then
    diverged.
    """
    h, n = cfg.hidden_size, cfg.num_layers
    a = cfg.attention
    inter = ffn_width(cfg)

    embed = cfg.vocab_size * h
    head = 0 if cfg.embedding.tie_lm_head else cfg.vocab_size * h
    # Filter: depthwise kernel + bias + its output norm.
    conv = h * cfg.embedding.conv_kernel + 2 * h if cfg.embedding.conv_kernel > 1 else 0

    per_attn = h * (a.num_query_heads * a.head_dim)  # q
    per_attn += h * (2 * a.num_kv_heads * a.head_dim)  # kv
    per_attn += (a.num_query_heads * a.head_dim) * h  # out
    if a.qk_norm:
        per_attn += 2 * a.head_dim

    per_dense_ffn = 3 * h * inter
    is_moe = moe_layers(cfg)
    n_moe = sum(is_moe)
    experts_per_moe = cfg.moe.num_experts + cfg.moe.n_shared
    ffn_total = (n - n_moe) * per_dense_ffn + n_moe * (experts_per_moe * per_dense_ffn + h * cfg.moe.num_experts)
    norms = n * 2 * h + h
    # One BranchScale per residual branch: every layer has an attention branch;
    # a dense layer has one mixer branch, a MoE layer has one per expert SwiGLU
    # (num_experts + n_shared).
    branch_scales = n + (n - n_moe) + n_moe * (cfg.moe.num_experts + cfg.moe.n_shared)
    # Spectral branch: per selected layer, input proj (h*h_comp), K complex
    # oscillators (damping logits K + freq buffer non-trainable), mode readout
    # (2*K*K_out for w_re/w_im), out proj (K_out*h).
    n_spec = int(sum(spectral_layers(cfg)))
    if cfg.spectral.enabled and n_spec:
        sc = cfg.spectral
        k = sc.num_modes
        k_out = sc.out_channels or max(1, h // k)
        per_spec = h * max(8, h // 4) + 2 * k * k_out + k_out * h
        spectral = n_spec * per_spec
        branch_scales += n_spec
    else:
        spectral = 0

    body = n * per_attn + ffn_total + norms + conv + branch_scales + spectral
    # The receiver gain (LMHead.logit_scale) is one scalar, always present.
    body += 1
    return {
        "embedding": embed,
        "lm_head": head,
        "body": body,
        "total": embed + head + body,
        "active_body": n * per_attn
        + (n - n_moe) * per_dense_ffn
        + n_moe * ((cfg.moe.top_k + cfg.moe.n_shared) * per_dense_ffn)
        + norms
        + conv
        + n
        + (n - n_moe)
        + n_moe * (cfg.moe.top_k + cfg.moe.n_shared)
        + spectral
        + 1,
    }


def auto_configure(
    target_body_params: int,
    vocab_size: int = 262144,
    *,
    num_experts: int = 0,
    aspect_ratio: float = 80.0,
) -> ModelConfig:
    """Solve H, L and head geometry from a *body* parameter budget.

    The budget is on the body only — embeddings are a separate, corpus-driven
    cost and including them in the solve is what produced V30's 43%-table
    configuration.

    Shape follows the standard aspect ratio ``H/L ~ 80``, which is where the
    depth-versus-width tradeoff sits for decoder-only transformers across three
    orders of magnitude of scale. Per layer the cost is
    ``2*H^2`` (attention, GQA making KV cheaper than that bound) plus
    ``3 * expansion * H^2`` (SwiGLU), so with ``expansion = 8/3`` the body is
    ``~10 * H^2 * L``.

    Args:
        target_body_params: non-embedding parameter budget.
        vocab_size: source alphabet size.
        num_experts: E>0 enables MoE; the budget then buys *total* parameters,
            not active ones.
        aspect_ratio: target ``H / L``.

    Returns:
        A fully populated :class:`ModelConfig`.
    """
    if target_body_params <= 0:
        raise ValueError("target_body_params must be positive")

    per_layer_h2 = 2.0 + 3.0 * (8.0 / 3.0)
    if num_experts > 0:
        # Half the layers are MoE, each holding (E + 1) expert copies of the FFN.
        per_layer_h2 = 2.0 + 3.0 * (8.0 / 3.0) * (0.5 + 0.5 * (num_experts + 1))

    # H^2 * L = target / c  and  L = H / ratio  ->  H = (target * ratio / c)^(1/3)
    hidden = (target_body_params * aspect_ratio / per_layer_h2) ** (1.0 / 3.0)
    hidden = _round_up(int(hidden), 128)
    layers = max(2, round(hidden / aspect_ratio))

    def geometry(h: int, n: int) -> ModelConfig:
        head_dim = 128 if h >= 2048 else 64
        while h % head_dim != 0 and head_dim > 32:
            head_dim -= 32
        n_q = h // head_dim
        n_kv = max(1, n_q // 4)
        while n_q % n_kv != 0:
            n_kv -= 1
        probe = ModelConfig()
        probe.vocab_size = vocab_size
        probe.hidden_size = h
        probe.num_layers = n
        probe.attention.num_query_heads = n_q
        probe.attention.num_kv_heads = n_kv
        probe.attention.head_dim = head_dim
        probe.moe.enabled = num_experts > 0
        if num_experts > 0:
            probe.moe.num_experts = num_experts
            probe.moe.top_k = max(1, min(2, num_experts // 4))
        return probe

    # The closed form ignores FFN width rounding, QK-norm gains and MoE router
    # weights (7-50% error). H is coarse (multiple of 128), so correcting through
    # H alone oscillates; instead fix H from the cube root and solve L exactly.
    #
    # The exact marginal cost of a layer is recovered as a finite difference of
    # the analytic count. It must be measured over a span containing the
    # dense/MoE layer pattern at its steady ratio — a 2-vs-3 difference lands
    # entirely on one MoE layer and overestimates the marginal cost by ~E times.
    span = 2 * ModelConfig().moe.moe_every if num_experts > 0 else 1
    lo, hi = 4 * span, 12 * span
    layers = max(2, layers)
    for _ in range(8):
        c_lo = count_params(geometry(hidden, lo))["body"]
        c_hi = count_params(geometry(hidden, hi))["body"]
        per_layer = (c_hi - c_lo) / (hi - lo)
        fixed = c_lo - per_layer * lo
        layers = max(2, round((target_body_params - fixed) / per_layer))
        ratio = hidden / layers
        # Keep the aspect ratio within ~1.5x of target; H is what moves.
        if 0.67 * aspect_ratio <= ratio <= 1.5 * aspect_ratio:
            break
        hidden = _round_up(int(hidden * (aspect_ratio / ratio) ** 0.5), 128)

    cfg = geometry(hidden, layers)
    cfg.target_params = target_body_params
    return cfg


def _apply_dict(obj: object, data: dict) -> None:
    """Recursively overlay a plain dict onto a dataclass instance.

    Unknown keys are rejected rather than ignored: a silently dropped key in a
    training config is a multi-hour failure that looks like a modelling result.
    """
    for key, value in data.items():
        if not hasattr(obj, key):
            raise ValueError(f"unknown config key {key!r} for {type(obj).__name__}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply_dict(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)


def load_config(path: str | None = None, **overrides: object) -> Config:
    """Load YAML, optionally auto-size from a body budget, then validate.

    Dotted override keys address nested fields, e.g.
    ``load_config(p, **{"train.max_steps": 1000})``.
    """
    import yaml

    cfg = Config()
    if path:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        model_data = data.get("model", {}) or {}
        budget = int(model_data.get("target_params", 0) or 0)
        if budget > 0:
            auto = auto_configure(
                budget,
                int(model_data.get("vocab_size", cfg.model.vocab_size)),
                num_experts=int((model_data.get("moe", {}) or {}).get("num_experts", 0))
                if (model_data.get("moe", {}) or {}).get("enabled")
                else 0,
            )
            cfg.model = auto
        _apply_dict(cfg, data)

    for key, value in overrides.items():
        target: object = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise ValueError(f"unknown override {key!r}")
        setattr(target, parts[-1], value)

    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """Enforce every structural invariant the model relies on.

    Each check corresponds to a failure that is either silent or expensive to
    diagnose at runtime, so all of them fail fast at load time.
    """
    m, t = cfg.model, cfg.train
    a = m.attention

    def positive_int(name: str, value: object, upper: int) -> None:
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{name} must be an integer in [1, {upper}], got {value!r}")

    positive_int("model.vocab_size", m.vocab_size, 1 << 21)
    positive_int("model.hidden_size", m.hidden_size, 1 << 15)
    positive_int("model.num_layers", m.num_layers, 512)
    positive_int("model.loop_depth", m.loop_depth, 64)
    positive_int("model.attention.num_query_heads", a.num_query_heads, 1024)
    positive_int("model.attention.num_kv_heads", a.num_kv_heads, 1024)
    positive_int("model.attention.head_dim", a.head_dim, 512)
    positive_int("train.batch_size", t.batch_size, 4096)
    positive_int("train.grad_accum_steps", t.grad_accum_steps, 1024)
    positive_int("train.data.seq_len", t.data.seq_len, 1 << 20)

    if a.num_query_heads * a.head_dim != m.hidden_size:
        raise ValueError(
            f"num_query_heads*head_dim ({a.num_query_heads}*{a.head_dim}) must equal "
            f"hidden_size ({m.hidden_size})"
        )
    if a.num_kv_heads > a.num_query_heads or a.num_query_heads % a.num_kv_heads:
        raise ValueError("num_kv_heads must divide num_query_heads (GQA)")
    if a.head_dim % 2:
        raise ValueError("head_dim must be even (RoPE rotates dimension pairs)")
    if t.data.seq_len > a.max_seq_len:
        raise ValueError(
            f"train.data.seq_len ({t.data.seq_len}) exceeds attention.max_seq_len ({a.max_seq_len})"
        )

    if m.sliding.window < 0 or m.sliding.window > a.max_seq_len:
        raise ValueError("sliding.window must be in [0, attention.max_seq_len]")
    if m.sliding.history_stride < 0:
        raise ValueError("sliding.history_stride must be >= 0")
    if m.sliding.window > 0 and m.sliding.full_every < 2:
        raise ValueError(
            f"sliding.window={m.sliding.window} with full_every={m.sliding.full_every} makes every "
            "layer a full-attention relay, silently ignoring the window; set window=0 to disable it"
        )
    if m.sliding.window > 0 and m.sliding.full_every > 1 and all(w > 0 for w in layer_windows(m)):
        raise ValueError("sliding-window config leaves no full-attention relay layer")

    if m.embedding.conv_kernel < 1:
        raise ValueError("embedding.conv_kernel must be >= 1 (1 disables the filter)")
    if m.ffn.intermediate_size < 0 or m.ffn.multiple_of < 1:
        raise ValueError("ffn.intermediate_size must be >= 0 and multiple_of >= 1")

    if m.moe.enabled:
        moe = m.moe
        if moe.num_experts < 2:
            raise ValueError("moe.num_experts must be >= 2 (1 expert is a dense FFN)")
        if not 1 <= moe.top_k <= moe.num_experts:
            raise ValueError("moe.top_k must be in [1, num_experts]")
        if moe.n_shared < 0:
            raise ValueError("moe.n_shared must be >= 0")
        if moe.bias_update_rate < 0:
            raise ValueError("moe.bias_update_rate must be >= 0")
        if not any(moe_layers(m)):
            raise ValueError(
                f"moe.enabled with moe_every={moe.moe_every} and num_layers={m.num_layers} "
                "selects no layer (layer 0 and the last layer stay dense)"
            )

    if m.head.unigram_prior and not m.head.unigram_path:
        raise ValueError("head.unigram_prior requires head.unigram_path")
    if m.head.ce_chunk_rows < 1:
        raise ValueError("head.ce_chunk_rows must be >= 1")
    if m.head.sampled_softmax_k < 0:
        raise ValueError("head.sampled_softmax_k must be >= 0 (0 = full CE)")
    if m.head.sampled_softmax_k >= m.vocab_size:
        raise ValueError("head.sampled_softmax_k must be < vocab_size")
    if m.head.sampled_proposal not in {"uniform", "prior"}:
        raise ValueError("head.sampled_proposal must be 'uniform' or 'prior'")
    if m.head.sampled_proposal == "prior" and not m.head.unigram_prior:
        raise ValueError("head.sampled_proposal='prior' requires unigram_prior")
    if not 0.0 <= m.head.sampled_in_batch_fraction <= 1.0:
        raise ValueError("head.sampled_in_batch_fraction must be in [0, 1]")
    if m.head.logit_scale_max < 0:
        raise ValueError("head.logit_scale_max must be >= 0 (0 = disabled)")
    if m.head.logit_scale_init < 0:
        raise ValueError("head.logit_scale_init must be >= 0 (0 means 1/sqrt(H))")
    if m.head.z_loss_weight < 0 or t.z_loss_weight < 0 or t.moe_z_loss_weight < 0:
        raise ValueError("z-loss weights must be non-negative")

    if t.data.eos_token_id == t.data.pad_token_id:
        raise ValueError("data.eos_token_id and data.pad_token_id must differ")
    for name, tok in (("eos_token_id", t.data.eos_token_id), ("pad_token_id", t.data.pad_token_id)):
        if not 0 <= tok < m.vocab_size:
            raise ValueError(f"data.{name} ({tok}) is outside the vocabulary")

    s = t.schedule
    if s.warmup_steps < 0:
        raise ValueError("schedule.warmup_steps must be non-negative")
    # warmup may equal or exceed max_steps only for short smoke runs (--steps N):
    # the schedule then never leaves the ramp, which is the intended behaviour of
    # a tiny validation run, not a misconfiguration. Real runs keep warmup << total.
    if s.warmup_steps >= t.max_steps and t.max_steps >= 1000:
        raise ValueError(
            f"schedule.warmup_steps ({s.warmup_steps}) must be < max_steps ({t.max_steps}) "
            "for a real training run"
        )
    if not 0.0 < s.decay_fraction < 1.0:
        raise ValueError("schedule.decay_fraction must be in (0, 1)")
    if not 0.0 <= s.min_lr_ratio < 1.0:
        raise ValueError("schedule.min_lr_ratio must be in [0, 1)")
    if s.inverse_sqrt_tau < 0:
        raise ValueError("schedule.inverse_sqrt_tau must be >= 0")

    if t.precision not in {"bf16", "fp32"}:
        raise ValueError("train.precision must be 'bf16' or 'fp32'")
    if t.max_grad_norm <= 0:
        raise ValueError("train.max_grad_norm must be positive")
    if t.logging.exact_ce_interval < 0:
        raise ValueError("train.logging.exact_ce_interval must be >= 0")
    positive_int("train.logging.exact_ce_rows", t.logging.exact_ce_rows, 1 << 20)
    if type(t.logging.exact_ce_seed) is not int or t.logging.exact_ce_seed < 0:
        raise ValueError("train.logging.exact_ce_seed must be a non-negative integer")
    if not 0.0 < t.ce_keep_rate <= 1.0:
        raise ValueError("train.ce_keep_rate must be in (0, 1]")
    if t.ce_keep_mode not in {"bernoulli", "stride"}:
        raise ValueError("train.ce_keep_mode must be 'bernoulli' or 'stride'")
    if not 0.0 < t.muon.wd_cap <= 8.0:
        raise ValueError("muon.wd_cap must be in (0, 8]")
    if len(t.muon.ns_coeffs) != 3:
        raise ValueError("muon.ns_coeffs must hold exactly 3 coefficients")

    mm = m.multimodal
    if mm.enabled:
        if mm.n_bridge_queries < 1 or mm.bridge_layers < 1:
            raise ValueError("multimodal.n_bridge_queries and bridge_layers must be >= 1")
        if m.hidden_size % mm.bridge_heads:
            raise ValueError("multimodal.bridge_heads must divide hidden_size")
        if (m.hidden_size // mm.bridge_heads) % 4:
            raise ValueError(
                "multimodal bridge head_dim (hidden_size // bridge_heads) must be "
                "divisible by 4 for 2D-RoPE (row and column bands each need even width)"
            )
    if not 0.0 <= mm.modality_dropout <= 1.0:
        raise ValueError("multimodal.modality_dropout must be in [0, 1]")
    if not 0.0 < mm.waterfill_min_rate <= 1.0:
        raise ValueError("multimodal.waterfill_min_rate must be in (0, 1]")
    if mm.grounding_weight < 0:
        raise ValueError("multimodal.grounding_weight must be non-negative")

    xm = m.xm
    if xm.enabled:
        if xm.num_modes < 2:
            raise ValueError("xm.num_modes must be >= 2 (1 mode is ordinary CE)")
        if not 0.0 <= xm.entropy_gate <= 1.0:
            raise ValueError("xm.entropy_gate must be in [0, 1]")
        if not 0.0 < xm.explore_fraction <= 1.0:
            raise ValueError("xm.explore_fraction must be in (0, 1]")
        if xm.mode_std <= 0:
            raise ValueError("xm.mode_std must be positive")
        if xm.min_modes < 1:
            raise ValueError("xm.min_modes must be >= 1")

    if cfg.inference.temperature < 0 or cfg.inference.top_k < 0:
        raise ValueError("inference.temperature and top_k must be non-negative")
    if not 0.0 < cfg.inference.top_p <= 1.0:
        raise ValueError("inference.top_p must be in (0, 1]")


def describe(cfg: Config) -> str:
    """One-block human summary: shape, parameter split, and channel rate."""
    m = cfg.model
    counts = count_params(m)
    windows = layer_windows(m)
    n_full = sum(1 for w in windows if w == 0)
    bits = math.log2(3) if m.ternary.enabled else 16.0
    loop = max(1, int(m.loop_depth))
    eff_l = m.num_layers * loop
    lines = [
        f"H={m.hidden_size} L={m.num_layers}"
        + (f"x{loop}={eff_l}" if loop > 1 else "")
        + f" heads={m.attention.num_query_heads}q/"
        f"{m.attention.num_kv_heads}kv x {m.attention.head_dim} ffn={ffn_width(m)}",
        f"vocab={m.vocab_size} tied_head={m.embedding.tie_lm_head} conv_k={m.embedding.conv_kernel}",
        f"params total={counts['total'] / 1e6:.1f}M body={counts['body'] / 1e6:.1f}M "
        f"embed={counts['embedding'] / 1e6:.1f}M active_body={counts['active_body'] / 1e6:.1f}M",
        f"body share of total: {counts['body'] / max(counts['total'], 1):.1%}",
        f"attention: {n_full} full / {m.num_layers - n_full} windowed(W={m.sliding.window})",
        f"moe: {'on ' + str(m.moe.num_experts) + 'e top' + str(m.moe.top_k) if m.moe.enabled else 'off'}"
        f" on layers {[i for i, f in enumerate(moe_layers(m)) if f]}",
        f"spectral: {'on every ' + str(m.spectral.every) if m.spectral.enabled else 'off'}"
        f" | mm: {'on' if m.multimodal.enabled else 'off'}"
        f" | ternary_cache: {cfg.train.ternary_step_cache}"
        f" | ce_keep: {cfg.train.ce_keep_rate:g}/{cfg.train.ce_keep_mode}"
        f" | sampled_k: {m.head.sampled_softmax_k or 'full'}",
        f"weight rate: {bits:.3f} bits/weight -> body {counts['body'] * bits / 8 / 1e9:.3f} GB packed",
    ]
    return "\n".join(lines)

