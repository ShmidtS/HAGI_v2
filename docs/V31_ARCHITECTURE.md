# HAGI V31 Architecture

HAGI V31 is a causal language model stated as a communication system. Every
design decision below is either a measurement from the V30 run or a property of
the channel model; nothing is inherited from V28 on authority.

```
tokens
  -> source coder        (codebook + causal pulse-shaping filter)
  -> [multimodal prefix] (per-modality coders + fixed-rate bridge)
  -> channel             (L ternary blocks: QK-normed GQA + SwiGLU/MoE)
  -> output norm
  -> receiver            (tied head + unigram prior + receiver gain + chunked CE)
```

One path. Every tensor that leaves the source coder reaches the receiver; there
are no auxiliary branches reading detached copies of the hidden state and no
loss terms competing with the coding objective.

The objective:

```
loss = CE + w_z * z_loss + w_router_z * router_z_loss [+ w_ground * grounding]
```

CE is the channel's coding cost. The two z-losses bound log-partition drift,
which is numerical conditioning rather than a modelling preference. Grounding
appears only when a second modality is present. Load balance is not a loss: it
is a bias controller inside the router, so nothing competes with CE for
gradient.

## The channel analogy, with the role each module plays

| Concept | Module |
|---|---|
| source code | codebook + unigram prior |
| pulse shaping | causal depthwise Conv1d (left-pad only) |
| channel rate constraint | ternary quantization, 1.585 bits/weight |
| correlator | attention softmax |
| finite-state channel | sliding-window attention |
| variable-rate coding | MoE routing |
| decoder state register | KV-cache |
| receiver | tied LM head |

## What was removed from V28, and the measured reason

Each of these existed in V28 as a real code path and was either disabled in
both shipped configs or measured to be dead weight:

- **Variational bottleneck.** An off-path KL term competing with CE. Deleted.
- **Latent memory bank.** An auxiliary read/write branch on detached hidden
  states. Deleted.
- **HEP refinement.** An off-path second stack refining a copy of the stream.
  Deleted.
- **EXIT-chart halt.** A convergence-halt heuristic whose stop condition was
  never reached in a real run. Deleted.
- **Water-filling MoE gate.** Multiplied routing logits by `1 + 1/||residual||`
  (a temperature reduction) while a companion allocator added a log-bias toward
  already-low-residual experts — positive feedback with nothing opposing it.
  Deleted; the bias controller below replaces it.
- **Auxiliary load-balance loss.** A CV² penalty was a *gradient* the LM
  gradient simply overruled (measured `moe_lb = 8.2` at its ceiling for 50k
  steps). Deleted.
- **Rank-r factored head.** Fitting a known full-rank distribution left 1.42
  nats of irreducible KL at r=128 (measured). Replaced by the full-rank tied
  codebook.
- **Per-document padding.** At T=2048 the loader spent 57–90% of every forward
  on PAD tokens (measured). Replaced by sequence packing with block-diagonal
  doc mask.
- **Sequential dataset cycling.** 11k consecutive steps on one source produce a
  step change in ce at every dataset boundary. Replaced by proportional
  interleaving.
- **Distillation, attention-mode curriculum, nine auxiliary losses.** Each
  disabled in both shipped configs; deleted.

## Design decisions with their evidence

### QK-norm (the V30 divergence fix)

The V30 run's ce went 2.32 → 6.6 between step 19k and 53k with no
configuration change. Root cause: Muon removes the `1/||W||` brake that plain
SGD applies, so projection norms drift outward monotonically; unnormalized
q·k rides on `||q|| ||k||`, and past a point the softmax saturates — it stops
transporting information and stops passing gradient. RMS-normalizing each
head's q and k makes the logit scale a function of `head_dim` alone. Asserted
in `tests/test_attention.py`.

### The receiver gain (V31 initialization fix)

A tied codebook does not start at the prior. At initialization the residual
stream is still largely collinear with the input token's own code word, so the
correlation `<h, w_token>` reaches `H * init_std` — a +41 logit at H=2048
against a logit standard deviation of 0.9 over the rest of the alphabet. That
single outlier moved the starting cross-entropy to 31.5 nats against the 8.05
the unigram prior should have delivered. One learnable scalar gain on the
correlation, initialized to `1/sqrt(H)`, puts the receiver exactly at the prior
at step 0; the dry-run now reads `ce=7.75` against the 8.05 baseline. The gain
is kept in fp32 under bf16.

### Source prior (free bits)

The unigram distribution is counted from the corpus before training starts
(`scripts/count_unigram.py`) and added as a fixed logit bias. On this corpus:
unigram entropy 8.05 nats against `ln V = 12.48` — ~4.4 nats/token of avoidable
early work.

### Auxiliary-loss-free MoE balancing

The router computes `logits_e = router(x)_e`; selection is
`top_k(logits + b)`; combining weights are `softmax` over the *selected*
logits only, without the bias. The per-expert bias is updated by a controller,
not by gradient:

```
b_e <- b_e + gamma * sign(target_load - load_e)
b   <- b - mean(b)                     # anchor the gauge
```

Two properties follow from construction. The bias affects selection only, never
the weights, so the experts receive the pure LM gradient. And it is a fixed-step
feedback controller, so it corrects an imbalance at a guaranteed rate regardless
of how strong the opposing LM gradient is — the property the CV² loss lacked.
The controller state (`expert_bias`, `load_ema`) is a buffer, survives
checkpointing, and is committed once per optimizer step after backward so the
forward stays pure under activation checkpointing.

### Ternary quantization as the rate constraint

`{-s, 0, +s}` per output channel, `s = absmean(W, dim=1)`, identity
straight-through estimator. `log2(3) = 1.585` bits/weight. Because `s` is the
per-row absmean of the master, any uniform outward drift of `||W||` cancels in
`W/s` — the effective weight RMS is self-stabilizing, which is why no spectral
cap is needed under Muon (simulated over 20k Muon steps; asserted in
`tests/test_ternary.py`).

### Muon / AdamW split

- **Muon** — 2D hidden-mixing matrices (`is_channel_weight` marker). Newton-
  Schulz orthogonalization sets every singular value of the update to 1, so all
  directions advance equally; on a ternary master this explores the reachable
  ternary patterns efficiently.
- **AdamW** — codebooks, 1D gains, biases, routers. A codebook's rows are
  updated at wildly different frequencies (five orders of magnitude on this
  corpus); per-coordinate adaptivity is exactly what that needs.

The partition reads an explicit marker set at construction, so disabling
ternary quantization for an ablation does not silently move the whole body from
Muon to AdamW.

### fp32 preservation under bf16

Three parameter kinds are kept in fp32: normalization gains (a gain at 1.0
receives gradients ~1e-4; the smallest bf16 step above 1.0 is ~0.0078, so they
would freeze at initialization), the receiver gain (one scalar near 0.022
controlling the whole output sharpness), and the MoE router (top-k over E
logits compares nearby numbers; a 7-bit mantissa makes the ordering arbitrary).

## Observables

- `ce` in nats/token, against the measured unigram entropy of 8.05. A model
  above that is worse than counting token frequencies.
- `qk_gain` — mean QK-norm gain. Rising means the correlator is heading for
  saturation, the V30 failure mode.
- `logit_scale` — receiver gain. Should rise as the conditional part of the
  code becomes informative; falling toward 0 means the head is giving up and
  emitting the prior.
- `moe/entropy_ratio` — usable fraction of the expert channels. Falling toward
  `1/E` means routing collapse.

## Vocabulary

The Gemma-4 tokenizer alphabet is 262144; only 225833 ids ever occur in this
corpus and the top 131072 carry 99.93% of all token mass. `scripts/compact_vocab.py`
rebuilds the `.bin` streams and the unigram counts against a dense id map
(dropped ids → UNK, never deleted, so document lengths do not shift), halving
the codebook: body share of total goes 64.5% → ~78%.

## Data

`scripts/preprocess_gemma.py` writes flat uint32 `.bin` streams with EOS(1) as
the document delimiter. `PackedMixDataset` reads them sequentially from a
randomized start offset, cuts windows of `seq_len+1` (the extra token is the
last position's target), and annotates each window with `doc_ids =
cumsum(is_eos) - is_eos` so the attention mask is block-diagonal per document.
`mix.json` weights are proportional and interleaved per window — every
optimizer step sees the full corpus distribution.

## Config

`configs/v31_1b.yaml` is the only shipped configuration: H=2048, L=22,
16q/4kv×128, ffn=5504, window=1024 with full relays every 4 layers, tied
codebook, conv_kernel=4, ternary on, unigram prior on, ce_chunk_rows=4096,
batch 4 × accum 8 × seq 1024 = 32768 tokens/step, 200k steps = 6.55B tokens.
Muon lr 0.02, AdamW lr 3e-4. `src/hagi/config.py`'s `auto_configure` solves
H, L from a body budget; `count_params` is exact and asserted against real
instantiations.
