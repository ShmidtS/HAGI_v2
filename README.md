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
  post-RoPE; scale bound uses the rotation-safe max over each RoPE pair,
  position-independent, verified flat to 500K positions), ~2× with ~0.005%
  error (worst 0.0125% across 43 layers). This replaces the earlier
  low-rank KV-POD (512→256), which was rejected: measured KV spectra are
  near full-rank, so the low-rank assumption was wrong from the start —
  same lesson as whitening/rotations in the expert track: measure the
  spectrum first, then choose the method.

**Target KV design — precision pyramid** (the old pyramid idea survives,
on the right carrier). The original KV-POD pyramid (2531698) halved the
read *rank* per distance doubling; the rank carrier died with POD. What
survives is the principle: nearby tokens matter more, distant tokens get
small softmax weights, so their error enters the output damped. Budget
error by contribution, not uniformly — the same telescopic principle as
the expert refit:

- sliding window (~4K) — full precision (bf16 or int8);
- mid distances — int8 (0.003–0.005% per channel, negligible next to the
  9.5% per-expert compression error);
- far history (≥64K) — int4 (halving effective bits per distance
  doubling, the same law as the old `r(d)`, applied to bit depth);
- a few sink tokens — full precision.

At 2M context this is ~3–4× smaller than bf16 cache with no perceptible
degradation (dominant share of tokens is far history). Wiring plan:
enable plain int8 KV in the TTT generator after the ternary e2e gate
(one-line install); the two-tier int8/int4 pyramid is a small patch to
`Int8KVStore` (tokens evicted from the window re-pack to int4), needed
only for >128K contexts.

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

The winning recipe is **`terni4`**: ternary W13 ({−1,0,1} grid, GPTQ,
group scales g128, 5 trits/byte) + int4 GPTQ W2 (g32) — **1.51× smaller**
than the FP4 original, e2e coherence gate on layers 0–3 passed.
Measured per-expert RMS error ~9.5 % (hot expert, honest ‖Δy‖/‖y‖);
binary W13 everywhere is ruled out (~14.3 %/expert → e_N ≈ 39 % with
compounding). Next round: adaptive tern-hot / bin-cold experts (~1.6–1.9×).

Full 43-layer pass runs **sequentially with telescopic correction**
(BRECQ-style): each layer is fitted on activations collected through the
already-compressed prefix, so per-layer errors add up (Σ) instead of
multiplying (Π). A/B measured on layer 1: drift-fit **0.73 %** vs
clean-fit **13.46 %** — 18× in favour; error propagation α = 0.87–0.99
per layer. `seq_full_pass.sh` orchestrates the pass in blocks of 3
(collect 3 layers in one model pass → refit → free acts, ~14 h total).

Note on metrics: refit logs print `resid` as an **MSE fraction** (squared
relative error); the honest RMS error is `sqrt(resid)` — e.g. logged
0.0082 ≈ 9.1 % RMS. Do not read logged residuals as percentages.

Generation runs from the compressed expert files via
`scripts/dsv4_generate_ttt.py` (`INT4X_OFF=1` → FP4 baseline for A/B).
On top of the persisted files the pipeline supports **TTT (anchored RLS
"eternal thinking")** updates and `--evolve` self-talk sessions.

Rejected by measurement (do not retry): whitening, frozen scales,
h/y rotations (QuIP#), sign branches, k-means codebooks, channel
rescaling of W3↔W2 (provably invariant), KL-Root-Kron, act_order.
Local group scales + ternary grid beat all of them.

### Pipeline

```
lossless_layers/{layer}_ffn.safetensors   (FP4 routed experts)
        │  dsv4_experts.py (dequant_fp4, pack/unpack intN/binary/ternary)
        ▼
dsv4_collect_seq.py    ── collect drifted x/y for layers L..L+2 through
        │                 the compressed prefix (SEQ_LAYERS, I4X_LAYERS,
        │                 SEQ_CH chunk; 8 K random tokens + unifold seeds)
        ▼
dsv4_refit_experts.py <L>   ── PTQ: tern W13 + GPTQ int4 W2 on drifted
        │                     acts (W13_MODE=tern W13_BITS=2 W13_GS=128
        │                     W2_GPTQ=1 PTQ_ONLY=1), real+unifold rows
        ▼
dsv4_reduced/layer_<L>/P.pt, mu.pt, expert_<k>.pt   (mode "terni4")
        ▼
dsv4_generate_ttt.py        ── decode from packed files on the fly (+ TTT/evolve)
```

Key scripts:

- `scripts/dsv4_experts.py` — shared decode / load / bit-pack utilities
  (binary, ternary, int4, n-bit, int6).
- `scripts/dsv4_refit_experts.py` — per-expert PTQ refit (tern/binary W13,
  GPTQ W2 over the activation Hessian; `W13_MODE`/`W2_GPTQ` env).
- `scripts/dsv4_collect_seq.py` — sequential drift collector (multi-layer
  capture in one model pass, chunked to keep HIP stable).
- `scripts/probe_alpha.py` — measures per-layer error propagation α.
- `scripts/probe_binary_check.py` — honest per-format error on a real
  expert (binary vs ternary vs two-level).
- `scripts/eval_ab_layer1.py` — A/B eval of a refit variant on held-out
  rows (`drift` vs `clean` fits).
- `seq_full_pass.sh` — full 43-layer sequential ternary pass (block-3,
  disk hygiene, single-instance lock).
- `scripts/dsv4_generate_ttt.py` — generation from compressed files
  (`INT4X_OFF=1` → FP4 baseline; `EXPERT_NOISE=ε` → noise benchmark).
