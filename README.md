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

## DeepSeek-V4 MoE compression (status: pass complete, e2e validated; gate training)

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

### Current compression (per expert) — the terni4 recipe

Each 4096→4096 routed expert is replaced by a mixed-precision full-rank
block — no POD bottleneck, no output basis:

```
z = (x − mu) @ P                       # per-layer mean-centring + orthogonal rotation
g = soft_lim(z @ W1ᵀ + b1)             # W1: TERNARY {−1,0,+1}, 5 trits/byte, g128 scales
u = soft_lim(z @ W3ᵀ + b3)             # W3: ternary, same grid/scales
y = (silu(g) · u) @ W2ᵀ                # W2: int4 ±7, ridge re-solved, then GPTQ
```

- **P** [4096×4096] fp32 — per-layer orthogonal basis (SVD of centered
  activations; verified orthonormal to machine zero). **mu** [4096] is
  folded into the biases b1/b3. P is a compression *transform*, not
  compression: 64 MiB/layer ≈ 2.75 GiB over 43 layers — it belongs on
  the Pareto ledger (candidates: fp16/int8 storage, Hadamard fold).
- **W1, W3** (gate/up): **ternary {−1,0,+1}**, 5 trits packed per byte
  (3⁵ = 243 ≤ 256), per-(row, group-of-128) LS scales. Group size is a
  measurement: per-row costs 3-4 pp, 64 adds ~nothing. Scales are
  importance-weighted (wLS — per-column z-energy, an imatrix analog),
  then **GPTQ/LDLQ error feedback** over the per-expert z-Hessian
  (H = zᵀz/n). The zero level is essential — grids without it lose
  ~8 pp on the FP4 source.
- **W2** (down): **int4 grid ±7**, per-(row, g128) scales. Before
  quantization W2 is **re-solved in closed form** (guarded ridge) to
  best explain the expert's target outputs *at the already-quantized
  W13* — a functional fit, not a weight fit — then GPTQ over the
  h-Hessian, then a final re-LS of the W13 scales + biases at the
  quantized W2. This is what removed W2 from the dominant-error spot.
- **Honest metric (v2)**: each expert's rows are split **80/20
  train/val (VAL_FRAC=0.2)**; the checkpoint residual is the
  **out-of-sample** error. An `n_val` marker distinguishes v2-era
  files from old in-sample ones (resume purges v1 partials — they
  would falsely pass the threshold). Threshold policy: strict 1e-3 on
  the first try, 1.0 on retries (accept whatever passed val).

**Size**: ~8.2 MB/expert vs 12.6 MB FP4 = **1.51×** on the routed mass
(97%+ of parameters). Stored density 1.6 bpw on W13 (not the 1.585
theoretical log₂3); whole-expert average ~2.4 bpw of weight codes
((2×1.6 + 4)/3 with W13:W2 = 2:1).

**Honest quality** (per-expert ‖Δy‖/‖y‖ on held-out rows): hot experts
~9% RMS; SVD-fixed dead experts 11-14% RMS. Refit logs print `resid`
as an **MSE fraction** — the honest RMS error is `sqrt(resid)`
(logged 0.0082 ≈ 9.1% RMS; never read logged residuals as
percentages). eps-injection benchmark: 5-10% → clean text, 15% → still
coherent, 20%+ → degradation.

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

### Status (2026-09-10)

- **The v2 pass is COMPLETE: all 43 layers (0..42) compressed —
  11,008 terni4 experts** (256/256 per layer, honest val residual in
  every checkpoint; ~89 GB total vs 169 GB FP4 original). Two
  incidents survived on the way (L23 OOM → expandable_segments +
  4096-token collect chunks; a Windows-Update reboot at L37 → stale
  lock removal, resume from 37).
- **Dead-expert fix (history, validated 2026-09-05)**: dead experts
  (zero routed rows) used to be pool-distilled on foreign inputs — a
  silent function replacement; now they are SVD-self-synthesized from
  their own W1 top singular subspaces. Layers 14..26 were redone with
  the fix; the 27..42 pass hit only 1–2 dead experts per layer
  (resid 0.2–1.5%).
- **E2E validation of the full stack (the milestone)**: generation on
  all 43 compressed layers produces coherent, factually correct text.
  "The capital of France is Paris..." → Rome, Madrid, Amsterdam,
  Brussels, Warsaw, Stockholm, Bern, Oslo, Bratislava, Vienna, Sofia,
  Budapest, Ankara, Nicosia — the whole chain correct; the Russian
  variant (Москва, Пекин, Токио, Вашингтон, Лондон...) equally clean.
- **Honest val-resid grows with depth** (medians: L18 0.78% → L26
  12.4% → L32 16.9% → L36 23.7%) yet e2e holds — the residual stream
  and routing absorb per-expert error. The trend motivates the gate
  (below) and the round-2 depth-waterfilling.
- **Bit-identical speedups (CPU-proven, measured)**: stacked W1+W3
  into ONE GPTQ call (shared per-expert Hzz; refit 44 → 22–28
  min/layer, −42% over 5 layers); sort-scatter collect routing (2
  device syncs/layer instead of ~10/expert).
- **Expert banks**: `dsv4_bank/layer{L}.safetensors`, 2.0 GB/layer
  (87 GB total); one file loads a layer in 4 s, the whole 92 GB stack
  goes resident in 137 s; bitwise-verified against the source
  expert_*.pt files.
- **int8 KV-cache validated**: 2× KV memory with text quality
  identical to bf16 (divergence only in continuation choice after
  ~15 tokens); scales collected ON the compressed stack
  (`KV_COLLECT=2048`, then `KV_MODE=int8`).
- **In progress**: latent-gate training on the full stack (781
  chunks × 512 tokens, ~45 h; 60 GB bank budget + LRU eviction +
  serialize; chunk 0: loss 13.29, p1=200 s, p2+bwd=5.7 s, reuse 43/43).
- **Next**: gate A/B generation → round 2 (per-layer bit budget from
  Σw_L·ρ_L², escape codes, Hadamard-vs-SVD A/B) → speed track
  (triton bank kernels) → GLM-5.3 through the same pipeline.

Generation runs from the compressed expert files via
`scripts/dsv4_generate_ttt.py` (`INT4X_OFF=1` → FP4 baseline for A/B).
On top of the persisted files the pipeline supports **TTT (anchored RLS
"eternal thinking")** updates and `--evolve` self-talk sessions.

Rejected by measurement (do not retry): whitening, frozen scales,
h/y rotations (QuIP#), sign branches, k-means codebooks, channel
rescaling of W3↔W2 (provably invariant), KL-Root-Kron, act_order.
Local group scales + ternary grid beat all of them.

### Latent feedback gate (training in progress)

Residual damage after the dead-fix is repaired by an adaptation of the
full-bandwidth transformer idea: the model's top state feeds back to the
input.

```
fused = e + RMSNorm(W_U · h_prev) ⊙ σ(W_G · e + b_G)
```

- h_prev = final-norm'd top hidden state of the previous position; e =
  next-token embedding. **W_U zero-init** → the starting behavior is
  bit-identical to the base; only the gate's 33M parameters train, the
  stack is frozen.
- Training data: **structured synthetic streams** (cycles, copy-blocks,
  zipf-markov, random mix). A random-only stream carries no NTP signal
  (loss ≈ ln V — compression damage is invisible); structure makes the
  damage measurable and repairable. No real data.
- Two passes per 256-token chunk: pass 1 (no grad) collects states and
  caches every MoE layer's output; pass 2 (grad) **reuses the cached MoE
  outputs** — a second-order approximation while the gate is small
  (measured: pass2+backward = 1.4 s/chunk vs 93 s for the full pass-1
  forward). MoE blocks are detached wrt gradient: the signal flows through
  the residual stream + attention (also sidesteps a ROCm
  autograd-through-unpack crash).
- Fast training path (`scripts/gate_moe_fast.py`): terni4 banks loaded
  from `dsv4_bank` as ONE safetensors read per layer (per-expert
  entries are zero-copy views), LUT ternary unpack (3× faster than
  divmod), z hoisted once per layer, sync-free sort-scatter (2 device
  syncs/layer vs ~50), block-grouped W13 batching (GATE_W13_BLOCK=32,
  W1/W3 built and freed interleaved — ~3 GB unpack peak instead of
  8.4), and an LRU bank budget (`GATE_BANK_BUDGET`; 60 GB → ~72 GB
  total, stable where 95 GB triggered HIP launch failures). The batch
  path is verified against a per-expert reference (rel 5e-03, dense
  and sparse routing — the sparse test caught a real position-vs-id
  slicing bug).
- Decode integration: `GATE_FILE=<ckpt>` in `dsv4_generate_ttt.py` — each
  step feeds `inputs_embeds=fused` (hash-router patched for embed passes).

### Memory & speed tooling (2026-09-10)

- **terni4 banks** (`scripts/build_bank.py` → `dsv4_bank/`, 2.0
  GB/layer, atomic writes): w13t u8 [256,4096,820] (ternary W1|W3,
  5 trits/byte), s13 f32 g128, b13, w2a int4, s2. Consumer:
  `gate_moe_fast.load_bank` (fast path) with fallback to expert_*.pt.
- **int8 KV** (`dsv4_generate_ttt.py`: `KV_COLLECT=<N>` collects
  per-channel scales on the compressed stack via kv_norm hooks;
  `KV_MODE=int8` patches the DynamicCache after prefill AND
  retrofits the already-written bf16 prefill — without the retrofit,
  cat(bf16, int8) promotes raw int8 into bf16 and the text collapses).
- **Blockwise eager attention** (`gate_moe_fast.install_blockwise_attention`,
  active for S>1024): S² → S×256; row-wise math identical (rel
  4.2e-05 at S=2048); kills the 10 GiB combined_logits spike on long
  prefills (the L23 incident). Auto-installed in train_gate and ttt.
- **VM-pressure rule**: at ~95 GB occupied (of the 107.87 GB APU)
  attention kernels start dying with "unspecified launch failure" —
  a pressure symptom, not an attention bug; fix = lower bank budget
  (60 GB → ~72 total) + serialize.

### Machine stability (this box)

- `AMD_SERIALIZE_KERNEL=3` is the only empirically stable mode under
  sustained GPU load (async HIP failures escalate to wedges /
  Kernel-Power 41 reboots). All pass scripts pin it.
- Never more than one CUDA process with the model on GPU.
- All supervisors are reboot-resilient: per-expert checkpoints land as
  they complete, resume skips them (n_val marker), stale v1 files are
  purged, wedges cool down 120 s and retry with threshold 1.0.

### Pipeline

```
lossless_layers/{layer}_ffn.safetensors   (FP4 routed experts)
        │  dsv4_experts.py (dequant_fp4, pack/unpack intN/binary/ternary)
        ▼
dsv4_collect_seq.py    ── collect drifted x/y for layer L through
        │                 the compressed prefix (SEQ_LAYERS, I4X_LAYERS;
        │                 vocab sweep + random tokens, 262K total,
        │                 cap 8192 rows/expert, layers above L identity)
        ▼
dsv4_refit_experts.py <L>   ── PTQ: tern W13 + ridge+GPTQ int4 W2 on drifted
        │                     acts (W13_MODE=tern W13_GS=128 W2_GPTQ=1
        │                     W13_GPTQ=1 PTQ_ONLY=1 VAL_FRAC=0.2);
        │                     dead experts -> SVD self-synth fit
        ▼
dsv4_reduced/layer_<L>/P.pt, mu.pt, expert_<k>.pt   (mode "terni4")
        ▼
dsv4_generate_ttt.py        ── decode from packed files on the fly (+ TTT/evolve
        │                     + GATE_FILE latent feedback gate)
        ▼
dsv4_train_gate.py + gate_moe_fast.py   ── latent gate training on
                              structured synthetic streams
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
- `seq_v2_resume.sh` — reboot-resilient v2 pass supervisor (n_val purge
  of v1 checkpoints, threshold 1.0 on retries, AMD_SERIALIZE_KERNEL=3).
- `redo_1426.sh` — dead-fix redo of the damage-zone layers 14..26
  (telescopic order, same retry policy).
- `scripts/dsv4_train_gate.py` — latent feedback gate training (structured
  synthetic streams, two-pass scheme, detached MoE).
- `scripts/gate_moe_fast.py` — fast MoE path for gate training (resident
  terni4 banks, FP4 mmap streaming, LUT unpack, sync-free scatter,
  pass-2 reuse).
- `scripts/dsv4_generate_ttt.py` — generation from compressed files
  (`INT4X_OFF=1` → FP4 baseline; `EXPERT_NOISE=ε` → noise benchmark;
  `GATE_FILE=<ckpt>` → latent gate decode).
