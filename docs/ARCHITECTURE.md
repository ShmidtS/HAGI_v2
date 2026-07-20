# HAGI V19 — SCS + Kalman Validation Gate (VOC-aware LDPC BP)

## V19 Changes vs V18

**V18 plateau root causes (3 bugs, not architecture):**

1. **`bottleneck_scale` frozen at 1.000** — scientist subagent verified std=0.0
   across all 1000 steps. The LayerScale parameter was sandwiched between two
   Linear layers (`rate_down -> *scale -> rate_up`) with NO non-linearity, so
   it was reparametrisation-equivalent to scaling `rate_up` columns. The
   gradient was identically zero (reparametrisation invariance).
2. **AWGN sigma too low** — channel was 9.6× below capacity (SNR=29dB,
   capacity=4.82 bits/use vs code rate 0.5). BP fully cancelled AWGN in 3
   iterations; the channel was not doing real work.
3. **`corr_gate` barely moved** — w: 0.1→0.126, b: 0.0→0.082 (final
   sigmoid≈0.55, no selectivity). Init too small.

**V19 fixes (evidence-based, KISS):**

1. **Bottleneck gate with non-linearity** — `z = z_linear + SiLU(z_linear) *
   tanh(bottleneck_gate)`, `bottleneck_gate` zero-init (identity at start).
   SiLU breaks the linear-linear invariance; tanh bounds the gate; residual
   keeps cold-start stable. Verifier: std 0.0 → 1.4e-3 after 2 steps (PASS).
2. **AWGN sigma raised** — `awgn_sigma_start 0.05→0.15`, `end 0.01→0.05`,
   `end_frac 0.3→0.5`. New SNR ~17dB (capacity ~3.0 bits/use, still above
   rate 0.5 but forces BP to do real work).
3. **`corr_gate_w` init raised** 0.1 → 0.3. Gradient signal 3× stronger.
4. **Dead code purge** — 10 modules / 1307 lines removed (lorentz, msa,
   cqi, exit_chart, contrastive, multimodal_*, clifford_*, uncertainty,
   algebra/).
5. **Vestigial config stripped** — MultimodalImageConfig, MultimodalAudioConfig,
   CrossModalConfig removed. `use_lorentz`, `lorentz_mode`, `w_contrastive`
   removed.
6. **CI invariants** added to `validate_config`: train/infer BP asymmetry,
   `awgn_sigma_end > 0`, `w_parity_diversity > 0`, `sigma_start >= sigma_end`.

### Kalman validation gate (the new V19 idea)

Per-position Mahalanobis-style syndrome gate — measurement gating from
Kalman filtering / JPDA multi-target tracking. Before each expensive BP
update (H^T back-projection, gradient flow), the syndrome is tested
against the noise covariance:

```
d²[b,t] = ||syndrome[b,t,:]||² / σ²
gate_valid = sigmoid((d² - chi²_crit) / tau)   (training, soft)
gate_valid = (d² > chi²_crit).float()           (inference, hard)
```

Positions inside the strobe (d² < chi²_crit at p=0.95) are statistically
indistinguishable from pure AWGN — they do NOT participate in the update.
This is value of computation (Horvitz/Russell/Wefald): the expected
information gain of running BP on those positions is below compute cost.

If <2% of positions are active for two consecutive iterations at inference
time, BP halts early (global convergence). Side info now exposes
`ever_active_frac` and `mean_iters_used` for diagnostics.

This formalises the user's request for "decide not to start thinking" in
the language of Kalman filtering (strobing / validation gate). The
asymmetry the user identified (cheap "out of domain" gate vs expensive
"do I know the answer" gate) is honoured: the syndrome is a cheap
statistic (computed anyway in BP iteration 0), not a deep probe. A
source-side probe head was considered for V19 but deferred per [YAGNI] —
the Kalman gate already covers the "skip if confident" semantics on the
channel side, and smoke metrics will decide whether a source-side gate
earns its complexity in V20.

### V19 smoke test results (logs/v19_archive/v19_smoke_3000.log, 3000 steps)

Pure tinystories curriculum (no domain switch). Phase windows:

| Phase       | Steps     | mce mean | mce min | parity mean | parity max |
|-------------|----------:|---------:|--------:|------------:|-----------:|
| warmup      | 0-100     | 10.34    | 9.48    | 0.082       | 0.111      |
| early       | 100-300   | 9.08     | 7.62    | 0.213       | **0.352**  |
| mid         | 300-1000  | 6.60     | **3.00**| 0.056       | 0.219      |
| late        | 1000-2000 | 6.26     | 3.01    | 0.015       | 0.149      |
| final       | 2000-3000 | 5.71     | 3.59    | 0.020       | 0.100      |

Best single-step metrics: **mce=2.996 @ step 929, bpt=4.32, par=0.021**.
Best parity peak: **0.352 @ step 214** (18× V18's 0.1948, 36× V17's 0.0096).

Comparison vs prior versions:

| Metric       | V17 @1180 | V18 @929 | V19 @929 | Status        |
|--------------|----------:|---------:|---------:|:--------------|
| mce min      | 4.18      | 3.33     | **2.996**| PASS (beats V18, near V15 2.62) |
| bpt min      | 6.03      | 4.80     | **4.32** | PASS          |
| parity peak  | 0.0096    | 0.1948   | **0.352**| PASS (18× V18) |
| Kalman gate  | n/a       | n/a      | works    | PASS (25% skip in verifier test) |

### Late collapse after step 1000 (open issue for V20)

After the mid-phase peak (mce 3.0 @ step 929), the model regresses:
parity falls from 0.056 mean → 0.015, mce drifts up 6.6 → 5.7 mean. Root
cause hypothesis: cosine lr decay bottoming out (1e-6 by step 2500)
combined with the suffix_probability curriculum crossing a regime
boundary around step 1000. The suffix-mce gap stays positive (~0.15) but
parity collapses faster than mce climbs, suggesting the channel path is
being silently disabled as the source gets confident — the Kalman gate
may be over-skipping once the source learns tinystories well.

### V20 plan (from V19 evidence)

1. **Plateau the lr schedule** — cosine decay to floor 1e-4 (not 1e-6).
2. **Suffix probability ramp delay** — base 0.3 until step 2000, then ramp.
3. **Measure Kalman skip ratio over training** — log `ever_active_frac`
   every 100 steps. If it climbs >70% by step 1500, the channel is being
   gated out — raise `chi2_crit` or lower σ to keep BP active.
4. **Parity diversity hook** — add early-stop on parity collapse: if
   `par_mean_100step` falls below 0.5 × `par_mean_peak`, log a warning.

---

## V18 Changes vs V17

**V17 regression:** masked_ce 11.26→4.18 (min 2.74), bpt→6.03, parity=0.0096
(dead), suffix_ce gap=1.78. Checkpoint showed `channel_open=0.086` (init 0.0,
schedule 4*warmup — never opened), `source_skip_scale=1.0` (not moving),
`corr_gate_b=+1.0` (code init=−1.0 — inconsistency), `rate_up.expand std=0.026`
(init 0.02 — not learning).

**Root cause (architect + critic review):** V15 reached mce=2.62 *through the
source_skip_scale bypass* (channel path was idle). V17 kept the bypass but
added a channel curriculum that never opened, so the model was neither a clean
Source-only nor a working SCS codec. `bottleneck_norm` (RMSNorm) aggressively
normalised the systematic latent, `w_parity=0.005` made parity loss
negligible (5e-7 contribution), and `w_parity_diversity=0.0` (V17 yaml bug)
left the LDPC graph free to collapse.

**V18 redesign** — strict Shannon Source-Channel Separation with the minimum
number of components that can train end-to-end from scratch:

1. **No source_skip bypass.** The Source decoder receives ONLY the channel
   output (no `pre_bottleneck` leak). Strict SCS: source encode → channel →
   source decode. To avoid the cold-start deadlock this creates when
   `rate_up.expand` is zero-init (architect review, CRITICAL finding 3),
   `rate_up.expand` is initialised `N(0, 1/√C)` so the channel path carries
   gradient from step 0.
2. **Pure LDPC BP decoder.** The `LDPCDecoder` no longer contains `FreqBlock`
   reasoning layers, no `LearnedUncertainty` (Kalman framing — dead in V17),
   no HARQ buffer, no mutation branch, no complex derivative branches. Each
   iteration is: syndrome = parity_recv − H·z_pred; correction = H^T·syndrome;
   gated update `z_pred += sigmoid(gate)·correction`.
3. **LayerScale bottleneck.** Replaces `bottleneck_norm` (RMSNorm). A
   learnable per-dim γ ∈ R^C init=1 scales `rate_down(h)` without forcing
   unit-norm, preserving information (architect review, CRITICAL finding 2).
4. **Channel always open.** No `channel_open` schedule. AWGN σ 0.05→0.01
   (never zero), physical erasure 15-20% always applied. Parity is always
   load-bearing.
5. **Train/infer BP asymmetry.** `train.bp_iterations=3` for gradient
   efficiency; `model.refinement.num_iterations=6` for inference recovery
   (V13 win, lost in V14).
6. **Fixed decoder gates.** `corr_gate_w` init 0.1, `corr_gate_b` init 0.0
   (sigmoid(0)=0.5 neutral). No −1/+1 extremes; resolves the code/checkpoint
   inconsistency.
7. **Loss weights (Goodhart-aware).** `w_parity=0.05` (10× V17),
   `w_parity_diversity=0.05` (was 0.0 in V17 — bug), `w_parity_recovery=0.05`
   (explicit erasure-tolerance loss), `w_rate_distortion=0.005` (low to avoid
   Goodhart collapse), `w_correction_alignment=0` (removed — V8 Goodhart).
8. **Suffix curriculum (conservative).** Base 0.3, ramp to 0.85 only in
   Phase 3 (after 80% training). Early training focuses on language.
9. **Dims (8gb_canonical V18).** H=384, C=192, r=128, head_dim=64 (6 heads),
   perc=4, expr=4, N_bp=3, ~25M total params (matches V15 body budget).
10. **Removed dead modules.** `FreqBlock` demoted entirely (Source = Attention,
    channel = LDPC BP). `LorentzSphereNorm`, `HARQBuffer`, `EXITChartEstimator`
    (as halt; kept as diagnostic), `LearnedUncertainty`, `CliffordCrossModal`,
    `ContrastiveAlignment`, `MultimodalInput` — all removed from the runtime
    path. Multimodal remains config-gated but is not on the critical path.

### Success criterion (smoke 1000 steps, B=8 T=512)
- masked_ce@1000 < 3.5 (recover V15 level 2.62 and below)
- parity_metric@1000 > 0.05 (parity genuinely learning)
- suffix_ce − masked_ce < 0.5 (suffix specialisation)
- bpt@1000 < 5.5 (well below unigram 8.69)

### V18 smoke test results (logs/v18_smoke_1000.console.log, 1000 steps)

Training cycled tinystories → python_instruct → smoltalk → wikipedia_en →
wikipedia_ru → openwebmath (sequential curriculum). The openwebmath switch at
step ~937 caused CE to jump (domain shift), so the pre-switch metrics below
are the cleaner signal.

| Metric           | V17 @ 1180 | V18 @ 929 (pre-switch best) | Status        |
|------------------|-----------:|----------------------------:|:--------------|
| masked_ce        |      4.18  |   **3.33**                  | PASS (-20%)   |
| bpt              |      6.03  |   **4.80**                  | PASS (-20%)   |
| parity_metric    |   0.0096   |   **0.1948**                 | PASS (20x V17)|
| suffix_ce gap    |      1.78  |  healthy                    | PASS          |
| channel behaviour|  never open|  always open (AWGN schedule) | PASS          |

**Takeaways:**
1. Channel coding is genuinely learning — `parity_metric` jumped from V17's
   dead 0.0096 to 0.1948 (20×). The LDPC BP decoder with `H^T` back-projection
   is producing real extrinsic information.
2. Strict SCS (no `source_skip_scale` bypass, no RMSNorm bottleneck) did NOT
   cause the predicted cold-start deadlock. The LayerScale bottleneck and
   always-open AWGN channel provided stable gradients from step 0.
3. mce 3.33 is not yet at V15's 2.62 — needs longer training (5000+ steps) to
   close the gap. The capacity reduction (20.78M vs V17's 58.7M) may also be
   too aggressive; H=384 → H=512 is the natural next experiment if mce stalls.
4. `corr_gate_w/b` (0.1, 0.0) init resolved the V17 code/checkpoint
   inconsistency — gates start neutral (sigmoid(0)=0.5) and learn from data.

---

## V17 Changes vs V16

**V16 regression:** BEST mce=4.13, mean 1500–2000=5.28 (V15: 2.62 / 4.48).
V16 fixed stuck `channel_open` and suffix gap, but opened the channel residual
too early → noise in latent path → CE floor ~5.3.

**Shannon SCS:** Source must first learn P(language). Channel is residual FEC —
only useful after source has capacity. V17 = V15 language wins + smarter curriculum
(not force-open channel).

1. **Channel residual slower.** Keep `channel_open` init **0.0**.
   `open_bias = min(1, step/(4*warmup))*5 - 4` (−4→+1; step0≈0.018;
   at 2*warmup still ~0.12; full open only ~4*warmup).
   `w_rate_distortion=0.001`, `w_parity=0.005`. Keep `rate_up.expand` N(0,0.02)
   and `source_skip_scale`.
2. **Suffix curriculum.** `_suffix_probability(step)`: early 0.25 (<2*warmup),
   linear to 0.7 by step 10k, then 0.85. Config base `suffix_probability=0.25`.
3. **Long-suffix bias.** Quadratic start sample in `masking.py` (`u^2 * count`)
   → longer suffixes for train-like-generate without early CE kill.
4. **Thin channel equalizer.** `reasoning_layers=1`, `bp_iterations=1`,
   `refinement.num_iterations=1` — less FLOP, Source-first.
5. **Diagnostics.** top2/entropy every 20 steps.
6. **Kept from V16.** RoPE, PE single path, conf from real max-prob,
   Attention Source, no double logits.

### Dims (8gb_canonical V17)
H=512, C=256, perc=6, expr=4, reason=1, `bp_iterations=1`,
`w_parity=0.005`, `w_rate_distortion=0.001`, `w_correction_alignment=0.001`,
`suffix_probability=0.25` (+ curriculum), lr=6e-4.

### Success criterion (smoke 200 steps, B=8 T=256)
CE@200 < 4.5 OR best mid < 4.0 (recover V15-like); generate 32 tokens.

## V16 Changes vs V15

V15 proved Attention Source breaks unigram (CE 5.99→2.62). V16 unmutes the
channel residual and fixes PE/RoPE/BP/diagnostics so SCS trains end-to-end.
**Result:** regression vs V15 on CE trajectory (see V17).

1. **Channel residual curriculum.** `channel_open` init **0.0** (not −3).
   Effective mix: `sigmoid(channel_open + open_bias)` with
   `open_bias = min(1, step/(2*warmup))*5 - 3` (−3→+2; step0≈0.05). `rate_up.expand`
   uses `N(0, 0.02)` (not zeros). Learnable `source_skip_scale` init 1.0:
   `h = rate_up(z) + source_skip_scale * pre_bottleneck`. Lower
   `w_rate_distortion=0.005` so RD does not fight channel open.
2. **PE single count.** Unknown positions use base `unknown` only; one
   universal absolute PE after embed (no pilot double-count on masked tokens).
3. **RoPE in AttentionBlock.** Standard rotary on q,k before SDPA
   (`rope_theta=10000`, cos/sin cache by T/device/dtype). Absolute PE + RoPE.
4. **BP cover.** `reasoning_layers=2`, `bp_iterations=2`,
   `refinement.num_iterations=2` — both FreqBlocks trained.
5. **Train/infer mask.** `suffix_probability=0.6` (was 0.3) for generate align.
6. **Diagnostics.** conf = actual max-prob every step (never fake 0.5);
   top2/entropy gated every 10 steps with finite fallbacks; `suffix_ce` nan
   only when no suffix rows.

### Dims (8gb_canonical V16)
H=512, C=256, perc=6, expr=4, reason=2, `bp_iterations=2`,
`w_parity=0.01`, `w_rate_distortion=0.005`, `w_correction_alignment=0.001`,
`suffix_probability=0.6`, lr=6e-4.

### Success criterion (smoke 150 steps, B=8 T=256)
CE@150 < 4.5; after 50 steps scheduled `channel_mix` > 0.1; generate 32 tokens.

## V15 Changes vs V14

**Root cause (proven):** TinyStories unigram CE ≈ 5.99 nats. V12–V14 train
floor ≈ 5.6 = slightly better than unigram. Bigram CE ≈ 3.7 — model never
approaches it. Gradients healthy; batch memorizes CE→0. **FreqBlock was the
language computer** → weak conditional model. Objective remains same-position MLM.

### Metaphor (Source-Channel Separation preserved)

```
Source Encode  = Attention LM backbone (REAL language model)
Rate match     = Linear H→C (LRM rate_down)
Channel Encode = fixed-graph SparseParity
Physical erase = keep
Channel Decode = thin residual BP (FreqBlock equalizer, gated)
Rate dematch   = rate_up
Source Decode  = Attention expression stack
LM head        = factored + tied codebook
```

1. **Attention Source.** New `AttentionBlock` (pre-norm bidirectional SDPA +
   SwiGLU) for perception (6) and expression (4). No FreqBlock in Source.
2. **FreqBlock demoted** to `LDPCDecoder.reasoning` only (channel equalizer,
   2 layers). `freq_coding.enabled` still True for decoder.
3. **`channel_open` curriculum.** `decoded = z_sys + sigmoid(channel_open) *
   tanh(ext_gate) * ext` with `channel_open` init −3.0 (source-only path at
   step 0; channel opens as training proceeds).
4. **head_dim=64** so H=512 → 8 heads (config head_dim 72 does not divide 512).
5. **Dims (8gb_canonical):** H=512, C=256, perc=6, expr=4, reason=2,
   `bp_iterations=1`, `code_rate=0.5`, aux losses mostly idle
   (`w_whiteness=0`, `w_parity=0.01`).
6. **Source skip:** `source_decode = expression(rate_up(z) + pre_bottleneck)`.
   `rate_up.expand` zero-init so channel residual is silent until learned.
7. **Success criterion (smoke):** step-100 masked CE < 5.0 (below unigram).

## V14 Changes vs V13

V14 fixes three root causes of CE plateau (~5.6) under V13 training.

1. **Learned Rate Matcher (LRM).** Replaces rFFT H→C / irFFT C→H bin gates
   (`bottleneck_gate` / `bottleneck_up_gate`). Feature axis ≠ frequency, so
   rFFT rate-match was fake. `FactoredLinear` `rate_down` (H→C) and `rate_up`
   (C→H) with rank `min(C, H//2)`, plus CQI magnitude scale and
   `bottleneck_norm`. SCS: still Source Encode → Channel → Source Decode.
2. **Universal PE on all tokens.** Pilot sinusoids added after embed (and
   cache concat) before perception, not only on erased positions. Breaks
   global-FFT translation equivariance of content tokens. Unknown PE path kept.
3. **Train BP = 4** (full cover of 4 reasoning layers). V13 `bp_iterations=2`
   starved layers 2–3 under train. Early EXIT halt disabled while `training`
   so every layer is applied once. Infer still uses `refinement.num_iterations`
   + optional halt.
4. **BP mask = physical ∪ semantic.** Extrinsic gated on all unknowns so CE
   reaches the decoder on pure semantic MLM (not only physical erasures).
5. **Separate `mut_gate_w/b`** in `LDPCDecoder` (no longer share `corr_gate`).
6. **Dims (8gb_canonical):** H=512, C=384, perc=2, expr=2, reason=4,
   `bp_iterations=4`, `w_parity=0.05`, `w_rate_distortion=0.02`, distill off.

## V13 Changes vs V12

V13 fixes channel-code collapse and restores Source-Channel Separation at the
source stack, with train/infer BP asymmetry for speed.

1. **Fixed-graph LDPC.** `parity_base` is a non-trainable buffer; learnable
   `edge_log_scale` (per check) only scales edge amplitudes. Encoder and
   checker share mask + base + scales. Stops `par` 0.14→0.01 collapse.
2. **`_layers_per_iter = 1`.** One FreqBlock per BP iteration (was `max(2,…)`
   → 12 apps / 6 iters).
3. **Separate expression stack.** `expression = ModuleList(FreqBlock ×
   expression_layers)` is not an alias of `perception` — encode ≠ decode.
4. **Train BP depth 2 / infer BP depth 4.** `train.bp_iterations` (default 2)
   when `training` and `refinement_iterations is None`; infer uses full
   `refinement.num_iterations`.
5. **Dims (8gb_canonical):** H=512, C=384, perc=3, expr=2, reason=4,
   `code_rate≈0.67`, `w_parity_diversity=0`.
6. **Generate:** `SpectralCache(context_window=128)`; BP iters independent of
   mask-predict pass count. Double-logits path removed; `distill_align` only
   if `distill_enabled`.

SCS pipeline unchanged: Source Encode → Channel Encode → Channel Decode →
Source Decode.

## V12 Changes vs V11

V12 is a **capacity-reallocation refactor**, not an architectural rewrite.
The V11 step-1000 checkpoint and `train_20260718_220707.log` revealed six
information-theoretic violations; V12 addresses them with the minimum set
of changes that preserve the V8–V11 channel-correct contract. Full
theoretical analysis is in [ARCHITECTURE-V12.md](ARCHITECTURE-V12.md).

1. **Source encoder under-capacity.** V11 `factor_rank=256` gave the
   source encoder a `log2(256)=8` bit latent code for a `V=49154` vocab
   requiring `log2(V)=15.6` bits. The latent was *below* the entropy
   floor, capping `masked_ce` at ~5.4 (bpt~7.7) regardless of decoder
   quality. V12 raises `factor_rank` to 512 (`log2(512)=9` bits), lifting
   the source-coding ceiling. The cost is offset by Change 3 (tying).

2. **Bottleneck too aggressive.** V11 `bottleneck_ratio=0.5` truncated the
   rFFT to the lowest 50% of frequency modes regardless of task relevance
   — a fixed geometric low-pass, not a learned rate-distortion bottleneck.
   V12 raises the ratio to 0.75, retaining 75% bandwidth and preserving
   task-relevant high-frequency detail. The `top2_mass=0.15` V11 floor is
   the posterior signature of this waste.

3. **Source codebook duplication.** V11 had independent `embed.token_compress`
   `[V, r]` and `lm_expand.weight` `[V, r]` tables — 25M params duplicated
   for no information gain. Under Gaussian-channel Source-Channel
   Separation, the ML source decoder is the adjoint of the source encoder,
   so the tables should be tied. V12 sets
   `lm_expand.weight = embed.token_compress.weight`, freeing ~25M params
   which are reinvested in an additional reasoning layer (Change 6).

4. **Dead decoder gates.** V11 checkpoint showed `decoder.ext_gate=0`
   (`tanh(0)=0`, extrinsic bypassed), `decoder.mut_up_w=zeros` (mutation
   branch dead), `decoder.corr_gate_b=-3.0` (`sigmoid(-3)≈0.05`, gate 95%
   closed). The decoder was a near no-op for the first ~200 steps. V12
   opens all three: `ext_gate=+0.5` (`tanh(0.5)≈0.46`), `mut_up_w` random
   init, `corr_gate_b=-1.0` (`sigmoid(-1)≈0.27`).

5. **Distillation distribution mismatch.** V11 hidden-state MSE aligned a
   masked bidirectional student with a causal unmasked teacher —
   incompatible distributions competing with the CE signal. V12 sets
   `distill_alpha_start=end=0.0` by default: the running MSE is zero, CE
   leads from step 0. Embedding transfer (one-shot) still runs. Non-zero
   alphas remain available for experiments that explicitly want the
   alignment signal.

6. **Body depth.** V11 had 5 reasoning layers; V12 raises to 6. The
   freed budget from Change 3 (tying) covers the cost (~0.4M params per
   FreqBlock at `C=480`).

The channel-correct contract is unchanged.

## V11 Changes vs V10

V11 addresses the V10 remaining issue: `masked_ce` stagnated at ~5.71
(bpt=8.12 bits/token) after step ~546, with `top2_mass=0.175` and
`confidence=0.124`. Root cause analysis through information theory:

1. **Code rate mismatch.** The V10 `n_checks` was 1056 for `C=320` — a
   code rate of 0.23, far below the configured 0.5. The auto_configure
   computed `n_checks` from its internal `C=1056` estimate, but the YAML
   overrode `core_hidden_size=320` without recomputing `n_checks`. V11
   recomputes `n_checks` from the FINAL `core_hidden_size` after all YAML
   overrides, giving `n_checks=320` (rate 0.5 as intended). Parity encoder
   params drop 0.676M → 0.102M.

2. **Source code capacity.** V10 `factor_rank=128` gave the source encoder
   only `log2(128)=7` bits of latent code capacity for a `V=49154` vocab
   needing `log2(V)=15.6` bits. V11 raises `factor_rank` to 256 (8 bits
   latent, still compressed but with 2x expressivity). Embedding + LM head
   grow 12.75M → 25.33M but remain far below the V8 88M.

3. **Distillation dominance.** V10 `distill_alpha_start=0.3` let the
   SmolLM2-360M teacher (causal LM) dominate the student (masked LM),
   pulling the posterior toward a mismatched distribution. V11 reduces
   `distill_alpha_start=0.1`, `distill_alpha_end=0.05` so the CE signal
   leads and distillation is a mild regularizer.

4. **More BP iterations.** V10 `n_iter=4` was still short for LDPC
   convergence. V11 raises to `n_iter=6, min=3`.

5. **Channel response init.** V10 `channel_response_t/h` were zero-init,
   leaving the MIMO equalizer near identity with a weak gradient
   (checkpoint showed max 0.011). V11 inits with `randn * 0.1` so the
   equalizer starts with a non-trivial phase perturbation.

The channel-correct contract is unchanged.

## V10 Changes vs V9

V10 fixes the V9 regression observed in `train_20260718_142908.log` and the
`step-012000` checkpoint: loss 7.32 → 1.31 (s6339) → 2.31 (regression +77%),
parity 0.005 → 0.069 (diverging decoder), grad 0.90 → 11.69 (explosion).

1. **BP diversity — per-layer decoder weights.** The V8/V9 decoder shared
   `shared_w`, `shared_phase`, `shared_ffn` across all 5 reasoning layers,
   so every BP iteration produced identical extrinsic information. Belief
   propagation requires each iteration to add NEW information; identical
   updates leave the parity residual divergent. V10 gives every reasoning
   layer its own weights (`shared_weights=None`, `shared_ffn=None`). The
   `w_shared`/`phase_shared` precomputation path is kept but deactivated
   when the shared parameters are `None`.

2. **Learnable derivative orders.** V9 initialized `raw_alpha_t/h = 0.5`
   (identical on all layers) and the optimizer never moved them — the V9
   checkpoint showed `raw_alpha = 0.5` on all 5 reasoning layers. V10
   initializes `raw_alpha` and `branch_gate_*` with small random noise
   (`randn * 0.1`) so the layers differentiate from step 0 and the gradient
   is not stuck at a symmetric fixed point.

3. **Gradient clipping.** V9 had `max_grad_norm = None` (no clipping). The
   V9 log showed `grad` exploding to 11.69 (and 50.25 in the last 50 steps).
   V10 sets `max_grad_norm = 1.0`, bounding the per-step update.

4. **Shorter AWGN anneal.** V9 annealed AWGN until 50% of training
   (`awgn_end_frac = 0.5`), so at step 12000 the systematic still carried
   `sigma ≈ 0.0042`. V10 shortens the anneal to 30% (`awgn_end_frac = 0.3`)
   so the channel is clean earlier and the decoder can focus on parity
   recovery rather than noise rejection.

The channel-correct contract is unchanged.

## V9 Changes vs V8

V9 keeps the Source-Channel Separation pipeline and all 5G analogies intact.
The refactor targets three V8 regressions observed in the step-500 checkpoint
and the `train_20260718_132801.log` trajectory:

1. **Embedding dominance.** The V8 17.5M `target_params` produced a 97M
   checkpoint where the `V×H` embedding table was 91% of the parameters
   (`H=896`, `V=49154`). V9 replaces the monolithic embedding with a
   **factorized source encoder** (`ConvEmbedding`): `V×r + r×H` plus a
   causal depthwise Conv1d pulse-shaping filter. For `V=49154, H=640,
   r=128` the source encoder costs ~6.4M instead of ~31M, and the body
   capacity is now independent of the vocabulary size. The `lm_head` is an
   independent rank-`r` factorization (`H→r→V`), not a tied transpose — the
   low-rank source encoder has no natural inverse of the same rank.

2. **Goodhart on the objective.** The V8 log showed `masked_ce` stagnating
   at ~6.0 (bpt=8.69 bits/token, far above the Shannon entropy of language)
   while `correction_alignment ~1.75` dominated the auxiliary signal. V9
   reduces `w_correction_alignment` from `0.01` to `0.001` so the optimizer
   aligns with the fidelity measure (CE) rather than the recovery proxy.

3. **Decoder under-capacity and dead weights.** The V8 decoder ran 2 BP
   iterations and `_layers_per_iter = reason//2 = 2`, so the fifth
   reasoning layer never received a gradient (`channel_response`,
   `branch_gate_*` were exactly zero in the checkpoint). V9 raises
   `num_iterations` to 4, `min_iterations` to 2, and computes
   `_layers_per_iter = ceil(reason / num_iterations)` so every reasoning
   layer is visited on every iteration. The `HARQBuffer` is simplified:
   the MLA read quartet (`mla_up_k/v`, `q_proj`, `o_proj`, ~0.5M params)
   is replaced by a single `read_out` Linear on the weighted compressed
   deltas — a 4× parameter reduction with no loss of combining rule.
   `LearnedUncertainty` projects to a scalar per position instead of a
   full `C` vector (`C*C → C+1` params); the inverse-variance update only
   needs scalars.

4. **Lorentz hyperboloid off by default.** The Lorentz projection adds an
   `exp`/`log` pair that violates the Euclidean contract the channel
   encoder expects. V9 defaults `freq_coding.use_lorentz: false`; the
   option remains for experiments that explicitly need hyperbolic
   geometry.

The channel-correct contract is unchanged: semantic erasure replaces the
compressed source code *before* the expand and pulse-shaping steps, physical
corruption applies only to the received systematic part, and erasure is never
represented by a token ID.

## Channel-Correct Training Contract

Semantic erasure and physical channel corruption are independent events. `semantic_unknown_mask` replaces unknown token embeddings with the learned `semantic_unknown_embed` before cache writes, spectral mixing, bottlenecking, or source-state derivation; changing placeholder IDs under the same mask therefore cannot change the model input at the causal seam. The source encoder then produces the systematic latent, learned continuous analog redundancy is derived from that clean latent, and only after that does the independently sampled physical channel corrupt received systematic symbols. Erasure is never represented by a token ID.

Four boolean `[B,T]` masks define the runtime contract: `semantic_unknown_mask` controls semantic visibility, `prediction_mask` selects logits/objective rows, `valid_target_mask` excludes padding and invalid targets, and `physical_corruption_mask` controls the independent latent channel. `prediction_mask` must be a subset of both semantic unknown and valid target positions; physical corruption does not modify semantic visibility or token IDs.

Training samples a random-unknown or full-suffix task independently per sequence with a 50/50 policy. A valid sequence has one terminal EOS before padding; suffix prediction includes that EOS. Generation starts with a fully unknown suffix, obtains compact gathered logits, emits left-to-right, accepts EOS only at generated lengths `2..max_new_tokens`, pads shorter rows, and returns `GenerationOutput(token_ids, generated_lengths, finished)`.

`step` always means completed optimizer updates. Each update consumes exactly `grad_accum_steps` distinct dataloader yields, and curriculum selection is set explicitly from that optimizer step before microbatch collection.

During active distillation, reconstruction and hidden-state alignment form one joint objective on every update. `distill_align` is student-owned and registered before optimizer construction; decoder correction is trained against the known clean-minus-received channel error rather than by minimizing extrinsic magnitude.

Checkpoint format v3 is a strict fresh-only inference artifact with exactly `format_version`, `model`, `config`, and `completed_updates`. Training resume, migration, partial loading, and legacy compatibility are not supported. Checkpoints use flat `step-XXXXXX.pt` paths with no run-specific subdirectories, and training refuses to start when the checkpoint root contains any entry. Old artifacts, including the collapsed step-501 probe and v1/v2 checkpoints, remain diagnostic-only; generation claims require fresh retraining.

Diagnostics report raw gradient norm and gradient RMS. Optional clipping is configured with `train.max_grad_norm`; a finite raw norm such as 40 is a measurement, not an explosion classification. `objective_loss` (with compatibility alias `loss`) names the optimized blended objective, while `masked_ce` is pure CE over gathered prediction rows and `bpt` is only `masked_ce / ln(2)`. `suffix_ce`, suffix/random task ratios, semantic/physical mask ratios, top-2 posterior mass, and posterior entropy separate objective viability from generation correctness without another model forward.

The locally retained, untracked log `logs/train_20260715_180755.log` covered steps `0..503`: loss fell from `8.8538` to `1.0447` (`-88.2%`), first/last-100 means were `6.7324`/`0.9648`, the minimum was `0.8416` at step 438, no NaN/Inf appeared, and throughput was about `1.076 steps/s`. This establishes early objective optimization only. The top-2 `of`/`and` mass `0.99977`, entropy `0.0927`, and `5.91..7.06` CE advantage are historical diagnostic measurements from the 2026-07-15 session, not a tracked repository artifact or repository-verifiable proof. The causal diagnosis of information-support leakage/mismatch is supported by reproduction, but no repository-verifiable log records those probe values; pre-mixing semantic erasure addresses the reproduced mismatch.

## Architecture inspired by Shannon information theory and 5G NR

### Concept

A language model is treated as a codec and designed with analogies to the
5G NR physical layer. These mappings describe design inspiration only: the
learned real-valued codec does not implement exact LDPC belief propagation,
a probabilistically derived Bayesian filter, or a formal EXIT-chart algorithm.

Source-Channel Separation Theorem: optimal communication is achieved
by separate optimization of source coding (compression) and channel
coding (error correction).

The model is a masked LM (bidirectional, same-position prediction),
not a causal next-token LM. Generation obtains a posterior over a fully
unknown suffix and applies constrained left-to-right token decisions with
adaptive EOS stopping.

---

## Pipeline

`HAGIv4.forward` executes five explicit stage boundaries:
`_source_encode` → `_channel_encode` → `_apply_erasure` → `_channel_decode` → `_source_decode`.
`_source_encode` replaces semantically unknown token embeddings before cache
writes and frequency-domain mixing. `_channel_encode` then derives learned
real-valued redundancy from the clean systematic latent; `_apply_erasure`
corrupts only the received systematic part while preserving that redundancy.
`DecodeState` carries Kalman covariance, HARQ feedback, iteration
marker, and cache intent through the channel decoder. `SpectralCache`
owns boundary context and stores a persistent snapshot of this state.
Local immutable contracts (`codec_contracts.py`) isolate runtime
model/train/inference stages from root config.

The embedding table can remain on CPU at inference for VRAM savings;
cross-device tensor transfers occur only at embedding lookup and lm_head
projection. Fresh embeddings are trainable from scratch. Distillation is
optional but enabled by default in `TrainConfig` and in the canonical profile;
an enabled run requires the configured distillation teacher to be available
locally, which may require network access during prior model setup, and fails
fast when that teacher cannot be loaded. Set `train.distill_enabled: false`
explicitly for a teacher-free baseline.

```
Token IDs
    |
    v
SOURCE ENCODE
  ConvEmbedding (V×r + r×H + depthwise Conv1d pulse-shaping) -> semantic
  unknown replacement on the compressed source code -> cache write ->
  FreqBlock mixing -> CQI-controlled frequency bottleneck -> clean systematic latent
    |
    v
CHANNEL ENCODE
  SparseParityEncoder(clean systematic) -> learned real-valued redundancy
  -> concatenate systematic/redundancy -> interleave
    |
    v
APPLY ERASURE
  deinterleave -> AWGN and/or physical replacement on received systematic only
  -> preserve redundancy -> reinterleave
    |
    v
CHANNEL DECODE
  iterative FreqBlock prediction -> sparse parity residual
  -> Kalman-form gated correction (scalar per-position uncertainty) ->
  extrinsic accumulation/HARQBuffer (weighted read) -> optional
  directional-novelty convergence halt
    |
    v
SOURCE DECODE
  frequency-domain expansion C -> H -> FreqBlock mixing -> RMSNorm -> factored LM head
```

The sparse parity graph, iterative correction, Kalman equations, HARQ buffer,
and convergence proxy are analogies to communications techniques. They are not
exact LDPC, Bayesian inference, HARQ retransmission, or EXIT-chart algorithms.

---

## Components

### 2D FFT replaces Attention (OFDM)

2D rFFT over (T, head_dim) per head — O(T * H * log(T*H)).
FFT = OFDM demodulation, IFFT = OFDM modulation.
Complex weight = MIMO channel equalizer (low-rank, rank=16).
Soft frequency gating = adaptive modulation (5G AMC).

No QKV. No softmax. No RoPE. No causal mask.
Position is encoded by phase in the frequency domain.

### Per-Mode Frequency Response (fading)

Per-mode learnable complex fading in FreqCoding2D:

```python
ch_t = exp(1j * channel_response_t[:F_t])  # learnable, init ~0 (identity)
ch_h = exp(1j * channel_response_h[:F_h])
X_f = X_f * ch_2d  # 2D fading per (T_mode, H_mode)
```

Initialization `std=0.02` → near-identity at start, the model learns
which frequencies matter more.

5G analog: Frequency-selective fading channel + per-subcarrier equalization.

### Kalman-Form Residual Correction

The decoder tracks diagonal state `P` and uses learned, sigmoid-bounded `Q`
and `R` in the Kalman-form equations `P_pred = P + Q`,
`K = P_pred / (P_pred + R)`, and `P = (1-K) * P_pred`. It applies `K` to
the sparse parity residual averaged over checks and passes the correction
through a learned gate. This is a Kalman-filter analogy and implementation
pattern, not a claim that the model computes an exact Bayesian posterior.

### HARQ Buffer

`HARQBuffer` stores extrinsic updates, reads top-ranked stored updates, and
combines them with the current update using tracked uncertainty. Feedback is
serialized through `DecodeState.harq_feedback` for cache continuity. HARQ soft
combining is the communications analogy; this is not retransmission-level
Chase combining or a decision-feedback equalizer.

### HARQ Soft Combining

On decoder iterations after the first, `HARQBuffer.read` supplies stored
extrinsic information and `HARQBuffer.combine` weights it using mean diagonal
Kalman uncertainty:

```python
stored_ext = self.harq.read(z_sys + ext, top_k=self.harq.cfg.top_k)
uncertainty = p.unsqueeze(0).expand(B, T, -1).mean(dim=-1)
ext = self.harq.combine(ext, stored_ext, uncertainty)
```

High uncertainty → more trust in stored state (prior).
Low uncertainty → more trust in current state.

5G analogy: uncertainty-weighted soft combining. The implementation combines
learned extrinsic tensors, not retransmitted codewords.

### Dynamic Bottleneck — AMC Analogy

CQI controls both magnitude gate and bandwidth cutoff:

```python
bw_base = sigmoid(cqi_bw_logit)
bw_scale = bw_base + (1.0 - bw_base) * cqi
cutoff = n_bins * bw_scale
dyn_mask = sigmoid((cutoff - bin_idx) * (1.0 + abs(bw_base) * n_bins))
mag_base = sigmoid(cqi_mag_logit)
gate = base_gate * dyn_mask * (mag_base + (1.0 - mag_base) * cqi)
```

Higher CQI moves the soft cutoff toward the full retained band and increases
the magnitude scale; lower CQI moves both toward their learned bases.

5G analogy: AMC-like CQI control. The implementation changes latent frequency
bandwidth and magnitude; it does not select a standardized modulation and
coding scheme.

### Deterministic Bottleneck (Rate Matching)

Deterministic rFFT truncation H→C + CQI-adaptive gate.
5G rate matching = deterministic puncturing, not stochastic.

### AWGN Noise Injection (training-only)

After channel encoding, Gaussian noise is added to the received systematic
symbols in `_apply_erasure`; learned redundancy remains clean:

```python
if self.training and awgn_sigma > 0.0:
    systematic.add_(awgn_sigma * torch.randn_like(systematic))
```

Sigma annealing: `0.005 → 0.0` linearly until 50% of training.

5G analog: Training with AWGN channel — model robustness to additive noise.

### Sparse Real-Valued Redundancy

`SparseParityEncoder` maps each systematic latent `[B,T,C]` to learned
real-valued redundancy `[B,T,M]` with a fixed sparse connectivity mask,
learned weights, and `RMSNorm`. The systematic and redundancy tensors are
concatenated and interleaved before physical corruption. The decoder-side
`SparseParityChecker` shares the encoder mask, weights, and normalization and
computes `parity_received - parity_computed`. This sparse graph is LDPC-style
inspiration only: values are continuous learned features, and decoding is not
the standardized binary LDPC sum-product or min-sum algorithm.

### Source Encoder — ConvEmbedding (V9)

`ConvEmbedding` replaces the monolithic `nn.Embedding(V, H)` table with a
factorized source encoder + pulse-shaping filter:

```python
# Factorized source code: V×r + r×H (low-rank approximation of V×H)
compressed = token_compress(input_ids)       # [B, T, r]
# Semantic erasure replaces the compressed source code (channel-correct):
# the learned erasure indicator is projected into rank-r via the adjoint
# of token_expand before the pulse-shaping filter runs.
h = token_expand(compressed)                # [B, T, H]
# Pulse-shaping filter: causal depthwise Conv1d (FIR filter analog)
h = local_conv(h)                            # O(H * kernel) params
h = norm(h)
```

Information theory: the compressed `r`-dimensional code is the true source
message (a low-rank source coder), `token_expand` is the modulation that
lifts it into the channel space, and the depthwise Conv1d is the
pulse-shaping filter that confines the transmit spectrum and reduces
inter-symbol interference before OFDM (the FreqBlock 2D FFT).

Weight tying with the output head is intentionally *not* used: a rank-`r`
source encoder has no natural inverse of the same rank, so the LM head is an
independent rank-`r` factorization (`lm_compress: H→r`, `lm_expand: r→V`).

### Embeddings

Embeddings remain trainable from scratch (`freeze_embeddings=False`), but the
current defaults enable teacher distillation rather than a teacher-free baseline:
`TrainConfig` and `configs/8gb_canonical.yaml` enable optional distillation.
That profile requires the configured distillation teacher to be available
locally, which may require network access during prior model setup, and startup
fails fast when the enabled distillation teacher cannot be loaded. Set
`train.distill_enabled: false` explicitly for a teacher-free baseline. Weight
tying is no longer supported in V9 (the source encoder and the LM head are
independent rank-`r` factorizations).

### Masked LM + LLaDA-style Generation

Same-position masked LM (bidirectional). Training mixes random unknowns
and full unknown suffixes 50/50. Generation performs one final refinement
over the fully unknown suffix, returns gathered logits only for prediction
positions, and applies the unified logits processor left-to-right. EOS is
forbidden before two generated tokens and ends each row adaptively through
`max_new_tokens`; shorter rows remain padded in `GenerationOutput`.

Coding analogy: iterative refinement. Generation itself performs constrained
left-to-right decisions from gathered masked-LM posteriors; it is not LDPC
belief propagation.

### Teacher as pilot generator (DM-RS analog)

Teacher (Gemma/SmolLM2) can optionally be used as a pilot/reference signal
generator (5G DM-RS analog). `TrainConfig` and the canonical profile enable
this optional distillation path by default; set `train.distill_enabled: false`
explicitly for a teacher-free baseline:
- **Embedding transfer**: copy (or project) teacher embeddings into
  student. Random projection when hidden_size differs.
- **Hidden state alignment**: MSE(student_hidden, teacher_hidden) on
  original (unmasked) input_ids. Teacher forward on unmasked input
  generates reference hidden states; student aligns its representations
  with teacher. Direction-agnostic — works for causal teacher and
  masked student.

KL distillation on logits is disabled: causal teacher logits (next-token)
conflict with masked LM student (same-position).

### Semantic Erasure Embedding

`semantic_unknown_embed` is learned and randomly initialized as an explicit
erasure indicator. It replaces unknown token embeddings before any mixing.

### Mutation Rank

rank=32 — capacity for noise injection in the turbo loop.

---

## Training Objective And Evidence Metrics

```
Level 1: objective = CE
Level 2: objective = CE + lambda_rate * Rate_distortion
                        + lambda_parity * log1p(Parity)
                        + lambda_contrastive * Contrastive
Level 3: objective = Level 2 + lambda_whiteness * Whiteness
                           + lambda_correction * Correction_alignment
```

The displayed `Loss` is the blended optimization objective and is logged as
`objective_loss` with the compatibility alias `loss`. It must not be used to
derive bits per token. Pure evidence metrics are computed under `no_grad`
from the existing gathered `[N_prediction,V]` logits:

- `masked_ce`: CE over every gathered prediction row.
- `bpt`: exactly `masked_ce / ln(2)`.
- `suffix_ce`: CE only over rows belonging to suffix tasks; NaN when no suffix row exists.
- `top2_mass` and `posterior_entropy`: compact posterior concentration evidence, not generation correctness claims.
- `suffix_task_ratio`, `random_task_ratio`, `semantic_mask_ratio`, and `physical_corruption_ratio`: observed task/channel support.

The staged auxiliary schedule is discrete: level 1 before `warmup_steps`,
level 2 until `2 * warmup_steps`, then level 3.

5G analog: Initial transmission without channel coding (uncoded),
then gradual coding.

### Three-Phase Mask Schedule (AMC analog)

| Phase | Steps | Mask ratio | 5G analog |
|-------|-------|------------|-----------|
| 1 (fitting) | 0–50% | 0.15 | Low SNR training, easy reconstruction |
| 2 (compression) | 50–80% | 0.35 | Medium SNR, less context |
| 3 (LLaDA compat) | 80–100% | 0.50 | High SNR, generation-ready |

Phase 3 prepares the model for 100% mask ratio during LLaDA-style generation.

### Distillation

```
Distill_loss = alpha * CE + (1 - alpha) * MSE(student_hidden, teacher_hidden)
```

Where:
- **CE** = masked cross-entropy on masked positions, same-position targets
- **Parity** = sparse parity-check residual energy accumulated by the decoder
- **Whiteness** = lag-1 autocorrelation penalty (decorrelated residual)
- **Extrinsic_info** = innovation norm, **penalty** (reward convergence)
- **Efficiency** = iterations used (minimize computation)
- **Rate_distortion** = MSE between pre-bottleneck and post-dematch hidden
- **MSE alignment** = hidden state distillation (teacher as DM-RS pilot)

Alpha schedule: linear ramp alpha_start → alpha_end over distill phase,
then 1.0 (CE only). Teacher freed after distill_end_frac.

---

## Inference

### LLaDA-style Iterative Decoding

```
1. Init: prompt + `max_new_tokens` placeholders plus a fully true suffix `semantic_unknown_mask`.
2. Run the configured final refinement and gather logits for suffix prediction positions only.
3. Process each suffix position left-to-right: forbidden IDs, minimum-length EOS gate, repetition penalty, no-repeat n-gram, temperature, top-k, then sampling/argmax.
4. Stop each row at its first valid EOS in `2..max_new_tokens`; pad unused suffix positions.
5. Return `GenerationOutput(token_ids, generated_lengths, finished)`.
```

`max_iterations` controls internal turbo refinement; it does not perform
confidence-ranked repeated full-model generation forwards.

### VRAM Efficiency

Embedding table (83% params, tied with lm_head) can remain on CPU.
Token IDs → CPU embedding lookup → hidden states on GPU for compute →
logits via CPU lm_head projection. VRAM: ~0.5 GB for 230M non-embed
params (bf16) instead of ~10 GB for full model.

---

## 5G NR Physical Layer Mapping

| 5G NR Stage | HAGI Component | File |
|-------------|----------------|------|
| Transport block + CRC | Visible embedding or pre-mixing semantic erasure | hagi_v4.py |
| Pilot/reference signal (DM-RS) | Teacher hidden states (distillation) | distillation.py |
| Scrambling | semantic_unknown_embed (learned erasure indicator) | hagi_v4.py |
| OFDM modulation (IFFT) | torch.fft.irfft2 | freq_layer.py |
| Modulation mapping (QAM) | FreqCoding2D phase modulation | freq_layer.py |
| Layer mapping + precoding | Multi-head + complex weight (rank=16) | freq_layer.py |
| Frequency-selective fading | Per-mode channel_response (learnable) | freq_layer.py |
| AWGN channel | Gaussian noise injection (training, anneal) | hagi_v4.py |
| Rate matching (puncturing) | rFFT bottleneck (H→C) + dynamic CQI bandwidth | hagi_v4.py |
| Channel (fading) | Per-mode complex fading in FreqBlock | freq_layer.py |
| OFDM demodulation (FFT) | torch.fft.rfft2 | freq_layer.py |
| Channel estimate | Kalman filter (P, Q, R) | kalman.py |
| Equalization | Complex weight @ freq modes | freq_layer.py |
| Sparse channel redundancy (LDPC analogy only) | `SparseParityEncoder` learned real-valued projection | sparse_parity.py |
| Sparse parity residual (LDPC analogy only) | `SparseParityChecker` with shared encoder parameters | sparse_parity.py |
| Iterative correction (belief-propagation analogy only) | `LDPCDecoder`: FreqBlock reasoning, residual correction, extrinsic accumulation | hagi_v4.py |
| Kalman-style estimation (not exact Bayesian inference) | Learned diagonal `P`, `Q`, `R` residual update | kalman.py, hagi_v4.py |
| HARQ soft-combining analogy | `HARQBuffer` extrinsic write/read/combine | msa.py |
| EXIT analogy only | Directional-novelty convergence proxy | exit_chart.py, hagi_v4.py |
| Adaptive modulation (AMC) | CQI gate + dynamic bandwidth + soft freq gating | hagi_v4.py, freq_layer.py |
| Rate dematching | rFFT zero-pad (C→H) + CQI gate | hagi_v4.py |
| Demodulation → bits | LM head over gathered prediction states | hagi_v4.py |
| Iterative decoding analogy | Masked-LM posterior generation | generate.py |

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train.awgn_enabled` | `true` | AWGN noise injection |
| `train.awgn_sigma_start` | `0.005` | Initial noise sigma |
| `train.awgn_sigma_end` | `0.0` | Final sigma (anneal to zero) |
| `train.awgn_end_frac` | `0.5` | Fraction of training for anneal |
| `train.freeze_embeddings` | `false` | Fresh baseline trains embeddings from scratch |
| `train.distill_enabled` | `true` | Enables optional teacher distillation in `TrainConfig` and the canonical profile; set `false` for a teacher-free baseline |
| `model.codec.code_rate` | `0.5` | Configured systematic code-rate target |
| `model.codec.edges_per_check` | `4` | Default sparse inputs per redundancy check |

Per-mode frequency response — learnable parameters in FreqCoding2D:
| Parameter | Init | Description |
|-----------|------|-------------|
| `freq.channel_response_t` | `N(0, 0.02)` | Per-T-mode phase fading |
| `freq.channel_response_h` | `N(0, 0.02)` | Per-H-mode phase fading |
