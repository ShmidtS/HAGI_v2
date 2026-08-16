# HAGI

HAGI is a **ternary RD-channel causal language model**:
source coder → (optional multimodal bridge) → ternary transformer channel →
tied receiver.

The current architecture is **V42** (`hagi-channel-v42`) — full causal
attention + T=512 + punctured receiver. See
[docs/V42_ARCHITECTURE.md](docs/V42_ARCHITECTURE.md).

## Growth cycle algorithm

HAGI grows a large model from small domain experts instead of training from
scratch. The idea and results live in
[GROWING_HYPOTHESIS.md](GROWING_HYPOTHESIS.md).

The current pipeline is a **recursive ternary growth cycle** — merge happens
in **groups of three** via the complex DFT-3 (F₃) mixer, never by powers of
two. Full driver: `scripts/run_growth_cycle.sh`.

The cycle, repeated for each level:

1. **Sort corpora into weakly-correlated domains.** `scripts/analyze_corpora.py`
   builds a unigram profile per corpus, computes the pairwise correlation
   matrix and greedily partitions corpora into decorrelated groups. Weakly
   correlated domains (e.g. RU ↔ EN have negative correlation) make each
   small expert specialize on a genuinely different subspace.
2. **Train small experts (H=128) to maximum saturation.** Experts are trained
   one at a time with early stopping on saturation (`saturation_patience`,
   `saturation_tol`, `saturation_min_steps`) — not a fixed step count.
3. **Lossless compaction.** `scripts/compact_checkpoint.py` packs each expert
   checkpoint with zstd (level 19) with zero information loss; restore is
   bit-identical (verified).
4. **Merge three experts via the ternary DFT-3 mixer.** `scripts/merge_experts.py`
   block-diagonally stacks the experts and recombines them with a unitary
   F₃ ⊗ I mixer (`mixer_hadamard_groups: [3]`). F₃ is unitary, every entry has
   modulus 1/√3 — uniform mixing with no blind channels. Groups of size 3ᵏ
   (3, 9, 27, …) are supported via Kronecker recursion F₃ᵏ.
5. **Joint-train the merged model to saturation.** Teaches the recombined
   blocks to interact; also early-stopped by saturation, not a fixed count.

The merged H=384 model can itself be treated as a level-1 expert, so the
cycle recurses (3 → 9 → 27 experts equivalent up to channel permutation).

### Scripts

- `scripts/analyze_corpora.py` — corpus correlation analysis / domain sorting.
- `scripts/train.py` — train / resume / init-from a model (saturation-aware).
- `scripts/compact_checkpoint.py` — lossless zstd compaction of checkpoints.
- `scripts/merge_experts.py` — low-level block-diagonal merge CLI (DFT-3).
- `scripts/run_growth_cycle.sh` — end-to-end growth cycle driver.
- `scripts/compact_model.py` — geometry experiments (rotation / SVD / sorting).

### Configs

- `configs/level0_experts/expert_*.yaml` — small (H=128) domain experts
  (ru_general / en_general / math_code / instruct).
- `configs/level0_merged_3.yaml` — merged H=384 model (ternary DFT-3 mixer
  `mixer_hadamard_groups: [3]`, saturation-based training).

## Quick start

```bash
pip install -e .
python scripts/train.py --config configs/level0_merged_3.yaml --resume checkpoints_l0_merged/step-0000000.pt
```

## DeepSeek-V4 MoE compression: copy minimal orthogonal blocks

This is the loss-minimising compression scheme discovered while shrinking
DeepSeek-V4-Flash (256 routed experts/layer × 43 layers = 11008 experts).

### The idea in one sentence

Each of the 11008 routed experts is a **separate communication channel**
(`x → y`), an independent unitary unit. You cannot merge or factor them
(weights are mutually orthogonal white noise), but you can **measure each
channel's transfer function** by driving it with a universal test signal
(unifold) and then **copy that function** with a compact block — per expert,
individually, not per layer.

### Why the blocks cannot be merged

Measured on the real checkpoint, the routed experts are pairwise orthogonal
(cos ≈ 1/√N, flat singular spectrum): each weight matrix is white noise.
Consequently any linear re-mixing — Hadamard, DFT-3, Procrustes — is
information-neutral: a rotation preserves the Gram matrix, so the "sum"
channel is just another random direction. There is **no shared component** to
recover. The only way to combine orthogonal experts is distillation (copy
their outputs), never their weights.

### The trick: the weights are noise, but the activations are low-rank

SVD on the expert weights is dead (rank ≈ full, flat spectrum). But the
**activations** flowing through the FFN live in a tiny subspace — the top
K=512 input directions explain ≈ 99.93% of the variance, the top Kp=384
output directions ≈ 99.9%. So the signal is compressible even though the
weights are not: project the I/O onto those subspaces and only the *residual
mapping between them* needs to be stored.

### One expert, copied minimally

Each 4096→4096 expert FFN is replaced by:

```
y ≈ Q · tern(z) · P
z  = x @ P          # input POD:  4096 → 512
h  = silu(z·w1ᵀ)·(z·w3ᵀ)   # ternary SwiGLU core, inter = 4096
yc = h @ w2ᵀ        # 4096 → 384
Q  = int8 [4096×384], per-column scale
```

- **P** [4096×512] fp32 — top-512 right singular vectors of the real FFN
  inputs (POD, computed once per layer).
- **w1, w3** [4096×512], **w2** [384×4096] — ternary {-1, 0, +1}, packed
  5 trits/byte. Trained 300 steps (lr 2e-3 cosine, batch 2048, G=32) to
  minimise `MSE(y_pred, y) / MSE(y, 0)` — the *residual %*.
- **Q** [4096×384] int8 with per-column float32 scale: this costs only
  0.012% extra error versus fp8's 0.165% at the same 1.57 MB.

### How the channel is measured (unifold + router split)

We drive each expert with a **universal test signal (unifold)**: bootstrap of
the layer's real POD manifold + 0.1σ jitter. This probes the honest working
band — the manifold where the router actually sends tokens plus a small
neighbourhood. We deliberately do NOT probe the full ±5σ volume: there the
ternary kernel cannot express the expert's response (25–36% residual), so
full-volume probing is waste.

- **Covered experts (87%)** — the router activated them on the 774K-sample
  collection; they refit on their **real routed activations** with an early
  stop at **≤0.01%** honest residual.
- **Uncovered experts (13%)** — no real samples. They refit on a **proxy
  manifold from their nearest router-weight neighbours** (cosine similarity),
  residual ~0.5–1.8% on the proxy (they are rare, so their weighted error is
  small). Stall-based stop once the residual plateaus.

### Results

| Metric | Value |
|---|---|
| Size per expert | **~1.3–2.4 MB** (adaptive inter/kp; vs 12.6 MB FP4) |
| Covered residual (real routed activations) | **≤ 0.01%** (early stop) |
| Uncovered residual (proxy manifold) | ~0.5–1.8% (13% of experts) |
| Q format | int4 QAT, per-column scale max/28 |
| Adaptive size | inter/kp by n_k: <200→(1024,512), 200–400→(2048,512), ≥400→(4096,768) |

0.1% residual requires inter=6144 (3.28 MB — *larger*), so 4096 is the
rate-distortion sweet spot: the smallest core that keeps the loss below the
round-trip budget.

### What actually reduces the error

The residual is dominated by a handful of decisions; everything else is noise.

1. **POD on activations, never on weights.** SVD of the expert weights is
dead — full rank, flat spectrum (white noise): no low-rank weight factor
exists. But POD on the *real FFN inputs/outputs* is sharp: top-512 input
directions explain ≈ 99.93% of variance (K=512 → 0.002% reconstruction
error), top-384 output ≈ 99.9% (Kp=384 → 0.004%). The signal is
low-rank even though the weights are not — this is the single biggest
error lever.

2. **int8 Q with per-column scale, not fp8.** The output basis Q is
quantised to int8 with one float32 scale *per column* (not per-tensor).
This costs **0.012%** extra error versus fp8's **0.165%** at the same
1.57 MB — a ~13× error cut for free.

3. **Accurate per-layer activations (no contamination).** Collecting x_L
through the *already-reduced* forward compounds error across layers — the
residual climbs 0.19% → 0.54% → 0.83% as depth grows, because each layer's
error leaks into the next layer's "input signal". Collecting x_L from the
*original* model (or a clean x₀) holds the residual at 0.157–0.26% across
all 43 layers.

4. **Train the ternary core, don't round it.** w1/w3/w2 are fitted by
300 steps of gradient descent (lr 2e-3 cosine, batch 2048, G=32) directly
against `MSE(y_pred, y)/MSE(y, 0)` — not by rounding a dense matrix. The
swiglu clamp (±10) matches the original model's own gate/up clamp, so the
teacher target is exact.

### Tried and rejected (they do NOT lower the error)

- **SVD / low-rank on weights** — flat spectrum, nothing to keep.
- **Linear re-mixing (Hadamard / DFT-3 / Procrustes)** — orthogonal experts
  stay orthogonal; the rotation is information-neutral.
- **bf16 Q** — output basis in bf16 loses too much precision.
- **warm-init** (start the ternary from a rounded dense fit) — 0.301% vs
  0.161% for random init; worse.
- **fp8 Q** — 0.165% vs int8's 0.012% at identical size.

### Error reduction levers (how to push it lower)

The residual is not at a hard wall yet — these knobs still move it.

1. **Measure more activations for the POD basis (N↑).** P and Q are
esimated from N=3000 tokens; a larger activation corpus (10k–100k tokens,
sampled across real prompts) makes the principal-subspace estimate converge
to the true one. Both the POD reconstruction error and the ternary fit
improve, because the fit sees a wider slice of the input distribution.

2. **Run more activations through the compressed layer (out-of-sample
residual).** The reported 0.157–0.26% is *in-sample* (the same 3000 tokens
used to fit). The honest metric is the residual on a held-out activation
set the ternary never saw — this reveals overfitting to x₀ and is the real
number the 43-layer round-trip will pay.

3. **Per-layer accurate activations (x_L, not the x₀ fallback).** Using the
true layer input x_L instead of x₀ holds the residual at ~0.157%; the x₀
fallback costs 0.22–0.26% per layer. Across 43 stacked layers this gap
compounds, so accurate per-layer x_L is the cheapest ~0.1%/layer win (the
cost is only the collection pass).

4. **More training steps (300 → 600+).** The ternary core is still
descending at step 300; more steps push it toward the inter=4096 floor.
Watch the out-of-sample residual — once it stops falling while in-sample
keeps falling, the fit is overfitting and more steps stop helping.

5. **Wider ternary core (inter 4096 → 6144).** The rate-distortion floor
moves ~0.156% → ~0.1%, at +0.5 MB/expert. Only worth it if the round-trip
budget requires it.

6. **Wider POD (K 512 → 768, Kp 384 → 512).** Lower POD reconstruction
error, at the cost of larger P/Q and a bigger ternary input/output space.

7. **Better Q basis (svd_lowrank niter 2 → 10+).** More iterations give a
tighter output basis Q and directly lower the residual for the same Kp.

8. **Measure the full 43-layer round-trip.** The per-expert residual is a
proxy; the true error is `reduced_model(x) vs original_model(x)` over the
whole stack (or generation quality). Only that number decides whether the
remaining levers are worth their size.

### Pipeline

```
lossless_layers/{layer}_ffn.safetensors   (FP4 experts, per layer)
        │  dequant_fp4 (dsv4_experts.py)
        ▼
dsv4_reduce_layer.py <L>  ── teacher y=ffn(x₀,w) → POD P,Q → ternary fit
        ▼
dsv4_reduced/layer_<L>/expert_<k>.pt   (2.77 MB each, int8 Q)
        ▼
dsv4_generate_reduced.py   ── skeleton + hooks → full reduced model → decode
```

Key scripts: `scripts/dsv4_reduce_layer.py` (per-expert fit),
`scripts/dsv4_reduce_all.py` / `scripts/dsv4_reduce_parallel.py` (loop),
`scripts/dsv4_collect_x0.py` (activation basis), `scripts/dsv4_generate_reduced.py`
(assembly + generation).
