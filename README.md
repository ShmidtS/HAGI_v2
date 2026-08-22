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

## DeepSeek-V4 MoE compression (status: in progress)

This is an ongoing experiment to shrink DeepSeek-V4-Flash
(256 routed experts/layer × 43 layers = 11008 experts) with minimal quality
loss. Findings below are current as of the latest measurements.

### What holds: the weights are white noise

Measured on the real checkpoint, the routed experts are pairwise orthogonal
(cos ≈ 1/√N, flat singular spectrum): each weight matrix is white noise. Any
linear re-mixing — Hadamard, DFT-3, Procrustes — is information-neutral: a
rotation preserves the Gram matrix, so the "sum" channel is just another
random direction. There is **no shared component** to recover; the only way to
combine experts is distillation (copy their outputs), never weight factoring.
SVD on the weights is dead: full rank, flat spectrum.

### What was wrong (corrected): the activations are *not* low-rank

An earlier claim — "top-512 input directions explain 99.93% of variance" —
was an overfit artifact: it was measured on only N=3000 tokens. The honest
spectrum (full 259072-token sample, layer 10) is almost full-rank:

| K (kept directions) | energy, in-sample | out-of-sample |
| --- | --- | --- |
| 512 | 59.1% | ~41% |
| 1024 | 71.8% | — |
| 2048 | 86.2% | — |
| 3072 | 94.9% | — |
| 4096 | 100% | — |

So compressing the residual stream to K=512 loses ~40% of the energy, and
low-rank POD compression of activations was **rejected**. The refit now works
at full rank (K=4096).

### Current compression (per expert)

Each 4096→4096 routed expert is replaced by a full-rank, two-stage ternary
SwiGLU block — no POD bottleneck, no output basis:

```
z = (x - mu) @ P            # per-layer mean-centring + orthogonal rotation
g = silu(z @ W1) * (z @ W3) # two-stage ternary SwiGLU, inter = 2048
y = g @ W2                  # identity output (4096 → 4096, no Q)
```

- **P** [4096×4096] fp32 — per-layer orthogonal rotation (whitening only, no
  dimensionality reduction); **mu** [1×4096] per-layer input mean.
- **W1, W3, W2** are ternary {−1, 0, +1}, each stored as a **sum of two
  ternary matrices** (`W` + `W_q2·scale2`) for finer precision, packed
  5 trits/byte, `inter = 2048`.
- The output is **identity** (kp = 4096 = D): the int4-QAT output basis Q was
  abandoned once the model ran at full rank — it added a bottleneck with no
  benefit there.

**Honest residual (full 4096-dim, weighted MSE(y_pred,y)/MSE(y,0))**: ~0.6–0.9%
per expert (measured on the current refit). The earlier "≤0.01%" figure was a
reduced 384-dim output-space loss and is not comparable.

### The actual VRAM lever: int8 quantization

Since the activations are full-rank, the memory win comes from quantization,
not low-rank:

- **int8 activations (residual stream)** — per-channel, ~2× with ~0.01%
  reconstruction error (per-tensor ~0.08%).
- **int8 KV-cache** — per-channel static scale, RoPE-safe (K is stored
  post-RoPE), ~2× with ~0.005% error (worst 0.0125% across 43 layers). This
  replaces the earlier low-rank KV-POD (512→256), which was rejected.

A distance-dependent **pyramid sliding window** (ported from the old KV-POD
idea) sits on top of the int8 cache: nearby tokens keep full 512 channels,
older tokens are channel-truncated per
`r(d) = clamp(512 >> ⌊log2(d/1024 + 1)⌋, 32, 512)` (base 512, window 1024,
min 32).

### Tried and rejected

- **SVD / low-rank on weights** — flat spectrum, nothing to keep.
- **Low-rank POD on activations (K=512)** — ~40% energy loss out-of-sample.
- **int4-QAT output basis Q** — bottleneck with no benefit at full rank.
- **KV-cache low-rank POD (512→256)** — replaced by int8 (both 2×, int8 loses less).
- **Linear re-mixing (Hadamard / DFT-3 / Procrustes)** — orthogonal experts stay orthogonal.
- **bf16 output basis** — loses too much precision vs int8.
- **warm-init ternary** — worse than random init.
- **fp8 output basis** — 0.165% vs int8's 0.012% at identical size.

### Status

The refit is **in progress**: 239 / 11008 experts across 3 layers are done so
far. Reduced-model generation is not yet wired up (the old
`dsv4_generate_reduced.py` was removed); generation currently runs the exact
model from `lossless_layers`.

### Pipeline

```
lossless_layers/{layer}_ffn.safetensors   (FP4 routed experts)
        │  dsv4_experts.py (dequant_fp4, load_selected_experts)
        ▼
dsv4_refit_experts.py <L>   ── exact x/y → two-stage ternary fit (full 4096-dim)
        ▼
dsv4_reduced/layer_<L>/P.pt, mu.pt, expert_<k>.pt
```

Key scripts:

- `scripts/dsv4_experts.py` — shared decode / load / ternary pack-unpack utilities.
- `scripts/dsv4_refit_experts.py` — per-expert full-rank two-stage ternary refit.
- `scripts/dsv4_collect_all_tokens.py` / `dsv4_collect_attention.py` — activation / KV collection.
- `scripts/dsv4_kvcache_int8.py` — int8 KV-cache (+ pyramid sliding window).
- `scripts/dsv4_generate_fast.py` / `dsv4_generate_real.py` — exact-model generation.
