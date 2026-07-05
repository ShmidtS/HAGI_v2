# HAGI V4 — Rate-Distortion Analysis

> Adaptation of Shannon's rate-distortion theory and Information Bottleneck (Tishby)
> to HAGI V4 architecture. Each component analyzed as a compression primitive,
> with concrete optimization recommendations.

---

## 0. Theoretical Foundation

### Core Identity

```
Prediction = Compression (Shannon 1948)
Hallucination = Lossy Compression Artifact (Guo & Li 2026, arXiv:2602.00906)
Training = Information Bottleneck Optimization (Tishby, arXiv:1503.02406)
    minimize: F_beta = I(X;Z) - beta * I(Y;Z)
```

### Rate-Distortion for LLMs (Guo & Li 2026)

```
Minimum memory per fact = n * KL(mu_K || mu_N) + o(n) bits
```

Even with optimal training, perfect data, closed-world assumption — the
information-theoretically optimal strategy under limited capacity is NOT to
abstain or forget, but to assign high confidence to some non-facts.
**Hallucination is the optimal error mode under log-loss.**

### HAGI V4 vs Standard LLM

| Property | Standard LLM (GPT-style) | HAGI V4 |
|----------|-------------------------|---------|
| Prediction mode | Autoregressive (sequential) | Plane prediction (parallel) |
| Compression analogy | Arithmetic coding (sequential) | Transform coding (JPEG DCT) |
| Attention | Causal mask | Bidirectional (no mask) |
| Training objective | Next-token CE | Masked CE (predict masked positions) |
| Iterative structure | Single forward pass | 4 refinement iterations |
| Hidden structure | Flat vector | Cl(3,0,0) multivector (8 blades, 4 grades) |
| Memory | KV cache (within sequence) | MSA ring buffer (cross-iteration) |
| Generation | Left-to-right unrolling | Progressive unmasking (confidence-ordered) |

---

## 1. Architecture Audit — What HAGI V4 Already Does Right

### 1.1 Cl(3,0,0) Grade Structure = Structured Rate Allocation

**Current**: Hidden state [576] split into grades:
- Scalar (64): confidence/resolution — slow momentum 0.8
- Vector (96): entities/concepts — medium momentum 0.5
- Bivector (96): relations — full update 0.0
- Trivector (64): higher-order structure — full update 0.0
- Residual (256): unconstrained pass-through

**Compression interpretation**: Different grades carry different information
types with different entropy rates. This is **unequal error protection (UEP)**
from coding theory — high-value information (scalar=confidence) gets more
protection (slow update = more bits effectively), while volatile information
(bivector=relations) gets less protection (fast update = fewer bits).

**Verdict**: Already near-optimal. Grade structure implements semantic-aware
rate allocation that flat Transformers cannot express.

### 1.2 Iterative Refinement = Successive Refinement Coding

**Current**: 4 refinement iterations with:
- Deep supervision: CE loss at each iteration, weight = 0.1^iteration
- Adaptive halting: relative_delta < 0.01 -> stop
- Gradient checkpointing (no h.detach — full gradient flow)

**Compression interpretation**: Successive refinement is a fundamental
rate-distortion concept — encode at progressively higher fidelity, where
each refinement layer reduces distortion incrementally. The decoder can
stop at any layer and get quality proportional to bits spent.

**Deep supervision decay 0.1^iter** = exponential weighting. This is
actually close to optimal for successive refinement: early iterations
get strong gradient (big distortion reduction), late iterations get
weak gradient (small distortion reduction — diminishing returns).

**Verdict**: Near-optimal. The exponential decay matches theoretical
predictions for successive refinement under log-loss.

### 1.3 GP2D = Predictive Coding

**Current**: Geometric product between tokens at t-1, t, t+1.
Residual blend: `h = h + sigmoid(gate) * proj(gp_product)`.
Gate init = -2.0 -> sigmoid(-2) = 0.12 (low initial contribution).

**Compression interpretation**: Predictive coding — predict current token
from neighbors, encode only the residual. This is the basis of DPCM
(Differential Pulse Code Modulation) and modern video codecs (P-frames).

The geometric product captures cross-token structure that simple linear
prediction cannot — it models relational geometry (bivector area between
tokens), not just scalar similarity.

**Low gate init (0.12)**: Correct — start with minimal prediction
contribution, let model learn when to trust cross-token prediction vs
intra-token representation. Analogous to starting with I-frames only in
video, then gradually enabling P-frames.

**Verdict**: Sound. Geometric product is a richer predictor than dot-product.

### 1.4 GDR Cross-Grade Mixing = Nonlinear Source Coding

**Current**: `vector x vector -> scalar + bivector` via geometric self-product.
Gated contributions to scalar_new and bivector_new.

**Compression interpretation**: Cross-grade mixing creates nonlinear
dependencies between information streams. In source coding, this is
analogous to transform coding where coefficients interact (DCT, wavelet).
The geometric product transforms vector-grade information into scalar
(confidence update) and bivector (relation update) — a structured
information transform.

**Verdict**: Novel and theoretically motivated. No standard LLM equivalent.

### 1.5 MSA = Side Information / Slepian-Wolf Coding

**Current**: Ring buffer with 4096 slots, slot_chunk_size=4 (4:1 compression),
top-k=6 retrieval, Clifford scalar routing key (64d), MLA compress (128d).

**Compression interpretation**: MSA is **external memory with compression**.
In rate-distortion theory, side information available to decoder reduces
required rate (Slepian-Wolf theorem). MSA slots = side information from
previous iterations.

- `slot_chunk_size=4`: 4 tokens averaged into 1 slot = 4:1 lossy compression
- `top_k=6`: Selective retrieval — only decode relevant side information
- `MLA compress (128d)`: Latent compression of KV — another lossy layer

**Verdict**: Implements Slepian-Wolf-style side information exploitation.
Two-stage compression (chunk + MLA) is aggressive but principled.

### 1.6 MoE = Conditional Computation = Variable-Rate Coding

**Current**: 4 experts, top-1 routing, SwiGLU, MoD skip for trivial tokens.

**Compression interpretation**: Each token gets exactly one expert = one
"codec". This is variable-rate coding — different tokens get different
processing depending on their complexity. MoD skip = zero-rate coding
for trivial tokens (identity passthrough).

**Load-balance aux loss**: Prevents collapse to one expert = prevents
all tokens using same codec (which would waste capacity).

**Verdict**: Correct implementation of variable-rate conditional computation.

### 1.7 Bidirectional Attention = Parallel Decompression

**Current**: No causal mask. All positions attend to all positions.

**Compression interpretation**: Autoregressive LLM = sequential decompression
(arithmetic coding — decode one token at a time). HAGI V4 = parallel
decompression (transform coding — decode all positions simultaneously
from frequency-domain representation).

This is fundamentally more efficient for the compression analogy:
- Sequential: O(T) decode steps, each depends on previous
- Parallel: O(1) decode steps (single forward pass), all positions independent
- Iterative refinement: O(log T) effective steps (4 iterations refine globally)

**Verdict**: Architecturally superior for compression analogy. Trade-off:
harder to train (no teacher forcing from causal structure).

### 1.8 Distillation = Distributed Source Coding

**Current**: KL divergence from SmolLM2-360M (teacher) to HAGI V4 (student).
Alpha schedule: 0.5 -> 0.3 (more teacher signal over time).
Temperature = 2.0. Ends at 60% of training.

**Compression interpretation**: Teacher = high-rate encoder (360M params).
Student = low-rate encoder (~74M params). KL divergence = distortion
measure between high-rate and low-rate representations.

Optimal distillation = minimize KL(student || teacher) subject to
student capacity constraint. This is exactly the rate-distortion problem
for knowledge transfer.

**Alpha schedule**: Starting with 50% CE + 50% KL, ending with 30% CE +
70% KL. This is correct — early training needs CE signal (find structure),
late training needs KL signal (refine compression to match teacher).

**Verdict**: Well-designed. Alpha schedule matches compression theory
(early=fitting phase, late=compression phase).

### 1.9 Coherence Loss = Smoothness Prior = Low-Pass Filter

**Current**: `mean(||geometric_product(h[t], h[t+1])||^2)`.
Gate init = -5.0 -> sigmoid(-5) = 0.007 (near-off at init).

**Compression interpretation**: Penalizes high-frequency variation between
adjacent positions. Analogous to quantization matrix in JPEG that penalizes
high-frequency DCT coefficients more — smooth signals compress better.

**Near-off init (0.007)**: Correct — don't enforce smoothness early
(let model find structure), gradually enable as training progresses.

**Verdict**: Correct. Acts as entropy prior — smooth signals have lower
entropy, compress better.

---

## 2. Optimization Opportunities — Per Component

### 2.1 Grade Dimensions = Capacity Allocation (HIGH IMPACT)

**Problem**: `grade_dims = [64, 96, 96, 64, 256]` is fixed. Different
datasets/tasks have different entropy per grade.

**Rate-distortion principle**: Optimal capacity allocation equalizes
marginal distortion across components:
```
dD/dR_grade0 = dD/dR_grade1 = dD/dR_grade2 = dD/dR_grade3
```

**Recommendation**: Make grade dims soft-learnable via a meta-router.
At each forward pass, compute per-grade entropy. Reallocate capacity
proportional to entropy. Implementation: learnable temperature on
grade projections.

**Quick win**: Monitor per-grade activation variance during training.
If scalar grade has low variance (confidence saturated) -> reduce d_scalar,
increase d_bivector (relations need more capacity).

### 2.2 Deep Supervision Decay = Optimal Bit Allocation (MEDIUM IMPACT)

**Problem**: Fixed decay `0.1^iteration` = [1.0, 0.1, 0.01, 0.001].

**Rate-distortion principle**: Optimal weighting for successive refinement
is proportional to expected distortion reduction at each layer:
```
w_i = E[||h_i - h_{i-1}||^2]  # actual delta, not fixed
```

**Recommendation**: Replace fixed decay with adaptive weighting.
In RefinementCore.forward, iteration loop, track actual delta_norm
between iterations. Use exponential moving average of delta_norm as
deep supervision weight instead of fixed 0.1^iter.

This automatically allocates more gradient to iterations where the model
is actually changing its representation (high information gain) and less
where it has converged (low information gain).

### 2.3 MSA Chunk Size = Adaptive Compression Rate (HIGH IMPACT)

**Problem**: `slot_chunk_size=4` is fixed. All content gets 4:1 compression
regardless of entropy.

**Rate-distortion principle**: Optimal compression rate depends on source
entropy. High-entropy content needs smaller chunks (less compression,
more slots). Low-entropy content can use larger chunks (more compression,
fewer slots).

**Recommendation**: Make chunk size adaptive based on grade router output.
- Scalar-dominant tokens (high confidence, low entropy) -> chunk_size=8
- Bivector-dominant tokens (complex relations, high entropy) -> chunk_size=2
- Implementation: grade router already computes per-token gate. Use the
  gate to select chunk size (discrete: 1, 2, 4, 8).

### 2.4 MoE Expert as Specialized Compressor (MEDIUM IMPACT)

**Problem**: 4 generic experts with no specialization guidance beyond
load-balance.

**Rate-distortion principle**: In transform coding, different basis
functions specialize in different frequency bands. Experts should
specialize in different "information bands".

**Recommendation**: Add an expert specialization signal. During training,
correlate expert routing with grade dominance:
- Expert 0: scalar-dominant tokens (confidence resolution)
- Expert 1: vector-dominant tokens (entity processing)
- Expert 2: bivector-dominant tokens (relation processing)
- Expert 3: trivector-dominant tokens (higher-order structure)
- MoD skip: residual-dominant tokens (pass-through, no processing needed)

Implementation: add small aux loss that encourages correlation between
grade_router gate and MoE router. This creates structured specialization
without forcing it.

### 2.5 GP2D Residual Whitening (MEDIUM IMPACT)

**Problem**: GP2D output is blended into residual stream without
checking if the residual is white (uncorrelated).

**Rate-distortion principle**: Optimal predictive coding produces white
(uncorrelated) residuals. If residual has autocorrelation, the model is
not fully exploiting prediction — there is remaining structure to compress.

**Recommendation**: Add a residual whiteness loss. After GP2D, compute
autocorrelation of the residual (gate * proj(out)) along the temporal
dimension. Penalize nonzero lag-1 autocorrelation. This pushes the model
to fully exploit cross-token prediction, maximizing compression efficiency.

### 2.6 Mask Ratio = Rate-Distortion Operating Point (MEDIUM IMPACT)

**Problem**: Progressive masking 15% -> 30% is heuristic.

**Rate-distortion principle**: Mask ratio = fraction of information removed
= distortion. Training at a given mask ratio places the model on a specific
point of the R(D) curve. The optimal operating point depends on the
desired rate-distortion tradeoff at inference.

**Recommendation**: Match training mask ratio to inference confidence
schedule. Inference uses confidence_schedule = (0.9, 0.7, 0.5, 0.1),
which means ~10%, ~30%, ~50%, ~90% of tokens unmasked per round. Training
mask ratio should cover this range: cycle through mask ratios [0.1, 0.3,
0.5, 0.7] instead of fixed progressive ramp. This trains the model at
multiple operating points on the R(D) curve.

### 2.7 Distillation Temperature = Rate Matching (LOW IMPACT)

**Problem**: Fixed temperature = 2.0.

**Rate-distortion principle**: Temperature controls the "softness" of the
target distribution. High temperature = more uniform = lower rate (less
information). Low temperature = sharper = higher rate (more information).
Optimal temperature matches the rate of the student.

**Recommendation**: Anneal temperature from 4.0 (early) to 1.0 (late).
Early training: high temperature (student needs coarse structure, low rate).
Late training: low temperature (student can handle fine structure, high rate).
This is rate-matching in source coding — adapt the encoder rate to the
decoder capacity.

### 2.8 Adaptive Halting Threshold = Rate-Distortion Stopping (MEDIUM IMPACT)

**Problem**: Fixed threshold `relative_delta < 0.01`.

**Rate-distortion principle**: Optimal stopping in successive refinement:
stop when marginal distortion reduction < marginal rate cost. The threshold
should depend on the remaining capacity budget.

**Recommendation**: Make threshold training-aware:
- Early training: threshold = 0.05 (coarse, stop early — save compute)
- Late training: threshold = 0.001 (fine, iterate longer — maximize quality)
- Implementation: linear ramp from 0.05 to 0.001 over training steps

This allocates more compute (iterations) to later training when the model
is refining fine details, and saves compute early when coarse structure
is being learned.

### 2.9 Per-Grade Coherence Weighting (LOW IMPACT)

**Problem**: Coherence loss treats all grades equally.

**Rate-distortion principle**: Different grades have different expected
smoothness. Scalar (confidence) should be smooth (high coherence expected).
Bivector (relations) can change rapidly (low coherence expected).

**Recommendation**: Apply coherence loss only to scalar and vector grades.
Skip bivector and trivector — these represent relational structure that
can legitimately vary between adjacent positions.

### 2.10 Curriculum = Entropy-Ordered Source Coding (ALREADY PRESENT)

**Current**: `curriculum_enabled: true`, `curriculum_stage2_start: 100000`.

**Compression interpretation**: Training on low-entropy data first (easy
patterns), then high-entropy data (hard reasoning) is entropy-ordered
coding. This is optimal for successive refinement — build coarse structure
first, then refine with high-entropy details.

**Verdict**: Already implemented. No changes needed.

---

## 3. New Components Inspired by Rate-Distortion Theory

### 3.1 Information Bottleneck Regularizer (NEW, HIGH IMPACT)

**Idea**: Add explicit IB-regularizer to training loss.

```python
# In losses.py, add:
def information_bottleneck_loss(h, targets, beta=1.0):
    """I(X;Z) - beta * I(Y;Z) estimated via variational bounds."""
    # I(X;Z): estimate via MINE (Mutual Information Neural Estimator)
    # or simpler: hidden state variance (proxy for complexity)
    complexity = h.var(dim=(0,1)).sum()  # proxy for I(X;Z)

    # I(Y;Z): estimate via predictive information
    # hidden state should predict targets
    logits = F.linear(h, lm_head_weight)
    predictive_info = F.cross_entropy(logits, targets)  # negative I(Y;Z)

    return complexity + beta * predictive_info
```

**Effect**: Explicitly drives model toward Information Bottleneck bound
(theoretically optimal compression). Currently the model approaches this
bound implicitly through gradient descent — the regularizer makes it
explicit and faster.

### 3.2 Rate-Distortion Aware Quantization (NEW, MEDIUM IMPACT)

**Idea**: Apply Radio-style rate-distortion optimization to the bf16
quantization of HAGI V4.

**Current**: `precision: "bf16"` — uniform 16-bit for all weights.

**Rate-distortion principle**: Not all weights need 16 bits. The equal
slope condition says optimal bit allocation equalizes dD/dR across all
parameters.

**Recommendation**: Implement mixed-precision based on weight importance:
- GDR momentum parameters (scalar_mom_logit, vector_mom_logit): fp32
  (critical for grade dynamics, small parameter count)
- Grade trunk/head: fp32 (core recurrent computation)
- Attention QKV/O proj: bf16 (standard, high parameter count)
- MoE expert weights: int8 with calibration (large parameter count,
  less sensitive to quantization)
- Embedding: bf16 (already transferred from teacher)

This reduces effective model size by ~30% with minimal quality loss,
following the Radio framework's equal-slope optimization.

### 3.3 Hallucination Floor Estimation (NEW, DIAGNOSTIC)

**Idea**: Estimate the theoretical hallucination floor for HAGI V4 using
the Guo & Li rate-distortion theorem.

```python
def estimate_hallucination_floor(model, dataset):
    """Estimate minimum achievable hallucination rate.

    Based on: R_min = n * KL(mu_K || mu_N)
    where mu_K = confidence distribution on facts
          mu_N = confidence distribution on non-facts
    """
    # Collect confidence scores on known facts vs unknown
    fact_confidences = collect_confidences(model, dataset.known_facts)
    nonfact_confidences = collect_confidences(model, dataset.unknown_facts)

    # Estimate distributions
    mu_K = estimate_distribution(fact_confidences)
    mu_N = estimate_distribution(nonfact_confidences)

    # Theoretical minimum memory per fact
    kl_div = F.kl_div(mu_K, mu_N, reduction='sum')
    n_facts = len(dataset.known_facts)

    # Model capacity (bits)
    model_capacity = count_parameters(model) * 16  # bf16

    # If n_facts * kl_div > model_capacity -> hallucination is inevitable
    hallucination_floor = max(0, 1 - model_capacity / (n_facts * kl_div))
    return hallucination_floor
```

**Effect**: Provides a theoretical lower bound on hallucination rate for
the current model capacity. If actual hallucination rate is close to this
floor, further training improvements have diminishing returns — need more
capacity (bigger model) or external memory (RAG).

### 3.4 Entropy-Adaptive Refinement (NEW, HIGH IMPACT)

**Idea**: Number of refinement iterations should depend on input entropy.

**Rate-distortion principle**: High-entropy inputs need more refinement
(more bits to allocate). Low-entropy inputs converge quickly (few bits needed).

**Current**: 4 fixed iterations with adaptive halting (threshold 0.01).

**Recommendation**: Make initial iteration count adaptive based on input
complexity:
- Compute input entropy proxy: `H_proxy = h.var(dim=1).mean()` after
  perception blocks
- High H_proxy -> start with 6 iterations (allocate more compute)
- Low H_proxy -> start with 2 iterations (save compute)
- Keep adaptive halting for per-token early stopping

This is rate-adaptive coding — spend more bits on complex inputs.

---

## 4. Training Pipeline Optimization

### 4.1 Two-Phase Training Schedule (IB-Aligned)

**Phase 1: Fitting (0-50% of training)**
- Low mask ratio (15%): high signal density, learn structure
- High distillation alpha (0.5): rely on teacher for guidance
- High LR: fast convergence to rough representation
- High GP2D gate (0.3): encourage cross-token prediction learning
- Low coherence weight (0.0001): let model find structure first
- Objective: maximize I(Y;Z) — fit the target

**Phase 2: Compression (50-100% of training)**
- High mask ratio (30-40%): force model to compress, predict from less
- Low distillation alpha (0.3): rely on own CE signal
- Low LR (cosine decay): fine-tune compression
- Low GP2D gate (learned, likely ~0.1): minimal prediction needed
- High coherence weight (0.001): enforce smoothness
- Add IB regularizer (if implemented): explicit compression pressure
- Objective: minimize I(X;Z) while maintaining I(Y;Z) — compress

This mirrors the two-phase Information Bottleneck trajectory confirmed
by Conklin et al. (2026) for standard LLMs, adapted to HAGI V4's
unique components.

### 4.2 Data Curation = Source Entropy Reduction

**Rate-distortion principle**: Cleaner data = lower source entropy H(X).
For fixed model capacity R, lower H(X) means lower distortion D.
This is why curated datasets (Phi, FineWeb) outperform raw web crawl
at smaller model sizes.

**For HAGI V4**:
- Remove duplicate/near-duplicate sequences (reduce H)
- Quality filter: remove noisy, contradictory, or low-quality text
- Grade-aware data selection: balance scalar-heavy (factual) and
  bivector-heavy (relational) content
- The curriculum stage1 -> stage2 already implements a version of this

### 4.3 Progressive Capacity Growth (Optional)

**Idea**: Start training with fewer grades active, progressively enable more.

**Rate-distortion principle**: Successive refinement at the architecture
level — start with a low-rate encoder (few grades), then grow to full
capacity. This is analogous to growing a codec from baseline to enhanced.

**Schedule**:
- Steps 0-30k: Only scalar + vector grades active (160 dims)
- Steps 30k-60k: Add bivector (256 dims)
- Steps 60k-90k: Add trivector (320 dims)
- Steps 90k+: Full model (576 dims)

Implementation: zero out inactive grade dimensions in GDR forward.
This forces the model to first learn low-grade structure (confidence,
entities) before adding higher grades (relations, higher-order).

---

## 5. Summary — Priority Matrix

| Optimization | Impact | Effort | Component |
|-------------|--------|--------|-----------|
| IB regularizer in loss | HIGH | MEDIUM | losses.py, loop.py |
| Entropy-adaptive refinement | HIGH | MEDIUM | hrm.py |
| Adaptive MSA chunk size | HIGH | MEDIUM | msa.py |
| Per-grade capacity monitoring | HIGH | LOW | logging in loop.py |
| Two-phase training schedule | MEDIUM | LOW | config yaml |
| Adaptive deep supervision weight | MEDIUM | LOW | hrm.py |
| Expert specialization signal | MEDIUM | MEDIUM | moe.py |
| GP2D residual whitening | MEDIUM | LOW | gp2d.py, losses.py |
| Adaptive halting threshold ramp | MEDIUM | LOW | hrm.py |
| Mixed-precision (Radio-style) | MEDIUM | HIGH | optim.py |
| Temperature annealing | LOW | LOW | distillation.py |
| Per-grade coherence | LOW | LOW | cast.py |
| Hallucination floor estimation | DIAGNOSTIC | MEDIUM | new script |
| Progressive capacity growth | LOW | MEDIUM | gdr.py, loop.py |

**Quick wins** (low effort, high impact):
1. Per-grade activation variance logging — diagnostic, no code changes
2. Two-phase training schedule — config-only change
3. Adaptive halting threshold ramp — 3 lines in hrm.py
4. Temperature annealing — 3 lines in distillation.py

**Strategic investments** (high impact, medium effort):
1. Information Bottleneck regularizer — new loss term
2. Entropy-adaptive refinement — modify RefinementCore
3. Adaptive MSA chunk size — modify MSAModule
