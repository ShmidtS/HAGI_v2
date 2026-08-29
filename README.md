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

Each 4096→4096 routed expert is replaced by a mixed-precision full-rank
block — no POD bottleneck, no output basis:

```
z = (x - mu) @ P                        # per-layer mean-centring + orthogonal rotation
g = soft_lim(z @ W1ᵀ + b1)              # W1: BINARY ±1, per-row scale (mean|w|)
u = soft_lim(z @ W3ᵀ + b3)              # W3: BINARY ±1, per-row scale
y = (silu(g) * u) @ W2ᵀ                 # W2: int4 grid ±7, per-(row,g128) scales, GPTQ
```

- **P** [4096×4096] fp32 — per-layer orthogonal rotation (whitening only);
  **mu** [1×4096] folded into the biases b1/b3.
- **W1, W3** (gate/up): **binary ±1**, 1 bit/weight, per-row fp32 scale
  (mean|w| — the LS-optimal scale). Signs are taken from exact FP32 rotated
  weights with a tie-break (FP4 weights contain exact 0s; `sign(0)=0` is
  unstorable and previously corrupted saved files silently).
- **W2** (down): **int4 grid ±7**, 4 bits/weight, per-(row, group-of-128)
  LS-refined scales [4096×16] fp32, quantized with **GPTQ/LDLQ error
  feedback** over the activation Hessian `H = hᵀh / n` (h computed with the
  already-quantized W13). GPTQ minimizes the *functional* output error, not
  the weight error — this removed W2 as the dominant error source.
- Adaptive Cholesky jitter for thin experts (n < 2048 rows → rank-deficient
  Hessian); mode marker `i1i4` stored in each file.

**Size**: 6.6 MB/expert vs 12.6 MB FP4 = **1.91× compression** of the routed
expert mass (97%+ of model parameters).

**Honest quality** (norm error ‖y_pred−y‖/‖y‖ per expert, measured on real
activations): median ~13–16%; thin/dead experts better. Note the refit logs
report *squared* MSE-ratio (≈0.1–2.7%), take sqrt for norm error. The floor
of this scheme is the binary W13 contribution (12.4%); the W2 int4
contribution was 13.6% with naive CD quantization and is now ~0 thanks to
GPTQ. An eps-injection benchmark (noise added to FP4 expert outputs) showed
5–10% → clean text, 15% → still coherent, 20%+ → degradation.

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

### Tried and rejected (quantization search)

The scheme above is the measured Pareto optimum at ~2× on scalar grids;
alternatives were prototyped and rejected by measurement:

- **SVD / low-rank on weights** — flat spectrum, nothing to keep.
- **Low-rank POD on activations (K=512)** — ~40% energy loss out-of-sample.
- **int4-QAT output basis Q** — bottleneck with no benefit at full rank.
- **KV-cache low-rank POD (512→256)** — replaced by int8 (both 2×, int8 loses less).
- **Linear re-mixing (Hadamard / DFT-3 / Procrustes)** — orthogonal experts stay orthogonal.
- **E8-lattice quantization (GLQ/QuIP#-style)** — 36.8% norm error, worse than int4.
- **Spherical VQ @ 2.25 bpw** — 39.2% (rate-distortion limit: 2.25 bits ≠ 4 bits).
- **2-stage VQ @ ~4.25 bpw** — 15.3% but needs per-group fp16 scales → int8-size
  footprint, an accounting illusion.
- **2/3-bit W13 grids** — no gain over binary (12–13% vs 12.4% floor).
- **Per-column / finer W13 scales** — helped in an unstorable format only;
  honest group scales on W13 ≈ +0.1%.
- **Low-rank correction of the W2 quantization delta** — diffuse spectrum, dead end.
- **Naive (CD/RTN) int4 W2** — 13.6% contribution from W2 alone; replaced by GPTQ.
- **Gradient-trained signs (2-stage ternary era)** — better W13 (7.4% vs 12.4%)
  at the same size; kept as the next quality step on top of GPTQ-W2.
- **bf16 output basis**, **warm-init ternary**, **fp8 output basis** — as before.

Cross-validated against industry recipes: the Tencent/AngelSlim Hy4 GGUF
(ternary 3:4-sparse gate/up ~1.3–2 bpw, down-projection deliberately ~2 levels
higher "because it writes straight into the residual stream", LS scales
instead of amax = 90% of their PTQ win, per-layer sensitivity split) matches
our mixed-precision layout and our LS-scale choices.

### Status

Full refit of all 43 layers in the `i1i4` format (bin W13 + int4 GPTQ W2) is
**in progress** (v19b; ~20/43 layers done, ~270 s/layer). Generation runs
from the compressed expert files via `scripts/dsv4_generate_ttt.py`
(`INT4X_OFF=1` switches to the FP4 baseline for A/B checks). The e2e text
quality gate on the compressed model is the next milestone; the eps-injection
benchmark predicts coherence at the measured per-expert error levels. On top
of the persisted files, the same pipeline supports **TTT (anchored RLS
"eternal thinking")** updates and `--evolve` self-talk sessions.

### Pipeline

```
lossless_layers/{layer}_ffn.safetensors   (FP4 routed experts)
        │  dsv4_experts.py (dequant_fp4, load_selected_experts, pack/unpack intN/binary)
        ▼
dsv4_refit_experts.py <L>   ── exact x/y → bin W13 + GPTQ int4 W2 (mode i1i4, 6.6 MB/expert)
        ▼
dsv4_reduced/layer_<L>/P.pt, mu.pt, expert_<k>.pt
        ▼
dsv4_generate_ttt.py        ── decode from packed files on the fly (+ TTT/evolve)
```

Key scripts:

- `scripts/dsv4_experts.py` — shared decode / load / bit-pack utilities
  (binary, int4, n-bit, int6).
- `scripts/dsv4_refit_experts.py` — per-expert PTQ refit: binary W13,
  GPTQ W2 over the activation Hessian (`W13_BITS`/`W2_BITS`/`W2_GPTQ` env).
- `scripts/dsv4_generate_ttt.py` — generation from compressed files
  (`INT4X_OFF=1` → FP4 baseline; `EXPERT_NOISE=ε` → noise benchmark).
