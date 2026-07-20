# HAGI_v2 Git Archaeology Report — V8 → V18 Evidence for V19 Redesign

**Branch:** `feat/clifford-cross-modal-v4`  
**HEAD:** `69e2e07` (V18, 2026-07-20)  
**Total commits analyzed:** 73 (no merges)  
**Method:** `git log/show/diff` + `docs/ARCHITECTURE.md` (canonical, in-repo) + `logs/v18_smoke_1000.console.log` + per-commit YAML configs  

> **Caveat on metrics:** All pre-V18 "best_mce" figures (V8–V17) are quoted from `docs/ARCHITECTURE.md`, which is an in-repo narrative written by the author at V18 time. Only V18 metrics are independently verifiable against a tracked log file (`logs/v18_smoke_1000.console.log`). Earlier training logs (e.g. `train_20260718_*.log` referenced by V9/V10/V11) are NOT in the repo. The ARCHITECTURE.md narrative should be treated as **author evidence**, not **repository evidence**, for V8–V17 numbers.

---

## A. Version Timeline Table

Versions V8–V17 were **not separate commits** — they are iterative design states documented in `docs/ARCHITECTURE.md` (the file lists one section per version transition). The actual code commits covering that period are only **two**: `c425565` (V8/SCS foundation) and `9dea097` (V9–V11 refactor). V12–V17 existed only as in-flight config/code changes that were never individually committed; V18 (`69e2e07`) is a full rewrite that lands on top of V11 with all intermediate learning folded in.

| Version | Commit(s)                                                                 | Date (doc)         | H    | C    | r        | bp_iters (train/infer) | w_parity     | w_corr_align   | suffix_prob | best_mce (author)    | status                                  |
|---------|---------------------------------------------------------------------------|--------------------|------|------|----------|------------------------|--------------|----------------|-------------|----------------------|-----------------------------------------|
| V8      | `c425565` (SCS foundation) + intermediate fixes `05bd308`, `4a7df07`, `d9e5bfc`, `12d7961` | 2026-07-15         | 896  | n/a  | n/a (V×H embed) | 2 / 2                  | 0.1          | **0.01 (Goodhart)** | n/a         | ~6.0 (bpt 8.69)      | Goodhart collapse on corr_align         |
| V9      | `9dea097` (V9–V11 refactor, covers V9–V11)                                | 2026-07-18         | 640  | 320  | 128 (ConvEmbed) | 4 / 4                  | 0.1          | 0.001          | n/a         | regressed (7.32→1.31→2.31) | grad explosion + diverging decoder      |
| V10     | (not committed; ARCHITECTURE.md only)                                     | (post-V9)          | 640  | 320 (n_checks=1056 → rate 0.23) | 128 | 4 / 4 | 0.1 | 0.001 | n/a | ~5.71 (stagnant)     | n_checks bug, dead alpha=0.5           |
| V11     | (not committed; ARCHITECTURE.md only)                                     | (post-V10)         | 640  | 320 (n_checks fixed → 0.5) | 256 | 6 / 6 | 0.1 | 0.001 | n/a | ~5.4 (floor)         | source under-capacity (8-bit latent)    |
| V12     | (not committed; ARCHITECTURE.md only)                                     | (post-V11)         | 640  | 480  | **512** (9-bit latent) | 6 / 6 | 0.1 | 0.0 (distill off) | n/a | ~5.4 (still floor)   | bottleneck still aggressive             |
| V13     | (not committed; ARCHITECTURE.md only)                                     | (post-V12)         | 512  | 384  | 512      | **2 / 4** (asymmetry introduced) | 0.1 | 0.0 | n/a | n/a | parity collapse 0.14→0.01 (LDPC overfit) |
| V14     | (not committed; ARCHITECTURE.md only)                                     | (post-V13)         | 512  | 384  | 512      | **4 / 4** (asymmetry lost) | 0.05 | 0.0 | n/a | ~5.6 (plateau) | rFFT rate-match "fake" (feature≠freq)   |
| V15     | (not committed; ARCHITECTURE.md only)                                     | (pre-V16)          | **512** | **256** | 512 (tied) | 1 / 1 | 0.01 (idle) | 0.0 | n/a | **2.62** (best ever) | Attention Source + source_skip bypass  |
| V16     | (not committed; ARCHITECTURE.md only)                                     | (post-V15)         | 512  | 256  | 512      | 2 / 2                  | 0.01         | 0.001          | 0.6         | **4.13** best, 5.28 mean | regression: channel opened too early   |
| V17     | (not committed; ARCHITECTURE.md only)                                     | (post-V16)         | 512  | 256  | 512      | 1 / 1                  | **0.005** (5e-7 contribution) | 0.001 | 0.25 (+curriculum) | **4.18**             | regression: channel never opened       |
| V18     | `69e2e07`                                                                 | 2026-07-20         | 384  | 192  | 128 (no tying) | **3 / 6** (asymmetry restored) | **0.05** | **0.0** (removed) | 0.30 (+Phase3 ramp) | **3.33** @ step 929 (verified in log) | PASS vs V17 (parity 20×, mce −20%)     |

### V18 verified smoke metrics (from `logs/v18_archive/v18_smoke_1000.console.log`)
- Step 929 (pre-openwebmath switch): **masked_ce=3.330, par=0.118** ← matches ARCHITECTURE.md exactly
- Step 925 (peak parity pre-switch): **par=0.3604** (37× V17)
- AVG mce over last 100 steps before openwebmath switch (steps 837–936): **3.832**
- AVG par over all pre-switch steps: **0.0883** (9× V17's 0.0096)
- Step 937 (openwebmath switch): mce jumps 3.33 → 6.66 (domain shift, not architecture)
- Step 999 (end): mce=7.23 (still recovering from domain shift)

### Key dimension evolution
- V8: H=896 (97M params, 91% in V×H embed) — embedding-dominated
- V9: H=640, C=320 (factorized ConvEmbedding r=128, body 6.4M)
- V11: r=256 (8-bit latent, still below 15.6-bit vocab entropy)
- V12: r=512 (9-bit latent, source-dest tied with LM head)
- V15: H=512, C=256, head_dim=64, Attention Source replaces FreqBlock in Source path
- V17: H=512, C=256, reasoning_layers=1 (thin), 58.7M params
- V18: H=384, C=192, r=128 (untied), 20.8M params (65% reduction vs V17)

---

## B. Technique Inventory

For each technique: **introduced → removed → diagnosis → V19 salvage potential**.

### 1. `source_skip_scale` (residual bypass around channel)
- **Introduced:** V15 (ARCHITECTURE.md §V15 item 6: `source_decode = expression(rate_up(z) + pre_bottleneck)`)
- **Made learnable in V16:** `source_skip_scale` init 1.0 (ARCHITECTURE.md §V16 item 1)
- **Removed:** V18 commit `69e2e07` (strict SCS, "No source_skip bypass" — design point 1)
- **What it was supposed to do:** Let Source-decoder path see pre-bottleneck latent so it can fall back to Source-only behavior when channel is unreliable
- **What actually happened:** V15 reached mce=2.62 *through the bypass*; channel path was idle. V17 kept bypass + added a channel curriculum that never opened → the model was neither a clean Source-only nor a working SCS codec (V18 commit body, lines 4–7). `source_skip_scale=1.0` (not moving) found in V17 checkpoint.
- **V19 salvage:** ❌ **Do not bring back as-is.** The bypass is fundamentally incompatible with strict SCS — it lets the optimizer cheat by routing gradients around the channel. If a "fallback" is needed, it should be a *scheduled* anneal (start open, decay to 0 over training), not a learnable scalar with no constraint.

### 2. `bottleneck_norm` (RMSNorm) vs LayerScale
- **RMSNorm introduced:** V14 (ARCHITECTURE.md §V14 item 1: `FactoredLinear rate_down/rate_up` + `bottleneck_norm`)
- **RMSNorm removed:** V18 commit `69e2e07` (design point 3, replaces with LayerScale init=1)
- **What it was supposed to do:** Normalize the systematic latent onto a controlled manifold before channel encoding
- **What actually happened:** "RMSNorm aggressively normalised the systematic latent" — V18 commit body. RMSNorm forces unit-norm, which destroys information magnitude differences that the LDPC syndrome needs.
- **V19 salvage:** ✅ **LayerScale is the right choice.** A per-dim γ ∈ R^C init=1 scales without forcing unit-norm. Verified working in V18 (parity 20× V17). Avoid RMSNorm/LayerNorm anywhere the channel encoder needs to see raw magnitudes.

### 3. `corr_gate_b` initialization (−1.0, 0.0, +1.0)
- **History of values:**
  - V11 (ARCHITECTURE.md §V12 item 4): `corr_gate_b=-3.0` (`sigmoid(-3)≈0.05`, gate 95% closed — dead)
  - V12: `corr_gate_b=-1.0` (`sigmoid(-1)≈0.27`, partially open)
  - V17 (V18 commit body line 8): **code init=−1.0, checkpoint=+1.0** — inconsistency between code and saved state
  - V18 (commit body design point 6): `corr_gate_b=0.0` (neutral `sigmoid(0)=0.5`)
- **What it was supposed to do:** Gate the LDPC correction `H^T·syndrome` so it starts silent and learns to open
- **What actually happened:** −3.0 → gate dead for ~200 steps (decoder near no-op). +1.0 → overconfident from step 0 (when code expected −1.0). The code/checkpoint drift in V17 means the V17 training trajectory was with `sigmoid(+1)=0.73` open gate, not the intended `sigmoid(-1)=0.27` closed gate.
- **V19 salvage:** ✅ **Use V18's `corr_gate_w=0.1, corr_gate_b=0.0`.** Neutral init (0.5) avoids both the dead-start and the overconfident-start failure modes. **Critical: assert in test that checkpoint values match code init values.**

### 4. `w_correction_alignment` (Goodhart collapse in V8)
- **Introduced:** V8 commit `c425565` (initial SCS codec had it)
- **V8 value:** `w_correction_alignment=0.01` (V9 commit body item 2 references V8 value)
- **Reduced V9:** `0.001` (`9dea097`, "V8 Goodhart")
- **V18 removed entirely:** `w_correction_alignment=0.0` (`69e2e07` design point 7)
- **What it was supposed to do:** Align decoder correction direction with the true error vector (cosine similarity loss)
- **What actually happened:** V8 log showed `masked_ce` stagnating at ~6.0 (bpt=8.69, far above language entropy) while `correction_alignment ~1.75` dominated the auxiliary signal. The optimizer learned to **maximize the proxy** (alignment) instead of the **fidelity measure** (CE). Classic [Goodhart's Law](software-laws.md#goodharts-law).
- **V19 salvage:** ❌ **Do not bring back.** The recovery proxy is redundant with `w_parity` and `w_parity_recovery` once you have a real LDPC BP loop. If some alignment signal is desired, cap its weight at 1e-4 and add a stop-gradient on the proxy so it can only regularize, not dominate.

### 5. `w_parity` value history (0.001 → 0.005 → 0.05)
- **V8:** `w_parity=0.1` (cfg_scs.yaml line: `w_parity: 0.1`)
- **V9–V11:** `w_parity=0.1` (cfg_v911.yaml)
- **V15:** `w_parity=0.01` (idle, ARCHITECTURE.md §V15 item 5: "aux losses mostly idle")
- **V16:** `w_parity=0.01`
- **V17:** `w_parity=0.005` → loss contribution 5e-7 (negligible) — V18 commit body line 5
- **V18:** `w_parity=0.05` (10× V17) — verified parity reaches 0.36 (37× V17)
- **What happened:** V17's `w_parity=0.005` was below the noise floor of CE (~1.0 scale × batch variance), so the LDPC encoder received essentially zero gradient. V18's 0.05 made parity load-bearing.
- **V19 salvage:** ✅ **Use V18's `w_parity=0.05` + `w_parity_diversity=0.05` + `w_parity_recovery=0.05`** (V18 design point 7). The diversity term prevents LDPC graph collapse (V13 parity 0.14→0.01).

### 6. `bp_iterations` train vs infer asymmetry
- **V8–V12:** symmetric (same `num_iterations` for train and infer)
- **V13 introduced asymmetry:** `train.bp_iterations=2`, infer uses `refinement.num_iterations=4` (ARCHITECTURE.md §V13 item 4)
- **V14 LOST the asymmetry:** `bp_iterations=4` for both train and infer (§V14 item 3) — explicitly noted in V18 design point 5 as "V13 win, lost in V14"
- **V16–V17:** symmetric again (2/2 then 1/1)
- **V18 restored asymmetry:** `train.bp_iterations=3`, `refinement.num_iterations=6` (V18 design point 5)
- **What happened:** V13's asymmetry was the right idea — fewer BP iters at train time saves gradient FLOP and avoids vanishing gradients through deep unrolled BP; more BP iters at infer time improves recovery. V14 lost this by "training BP=4 (full cover of 4 reasoning layers)" to fix a different bug (layers 2–3 starved).
- **V19 salvage:** ✅ **Always use train/infer asymmetry.** `bp_train ∈ [2,4]`, `bp_infer ∈ [2 × bp_train, 8]`. The V13→V14 regression shows this is fragile: document it as an invariant and add a CI test that asserts `bp_train < bp_infer` when `bp_train` is set.

### 7. FreqBlock (demoted from Source to channel equalizer and back)
- **Introduced:** V7 commit `91a8906` (2D FFT OFDM replaces attention)
- **V15 demoted** to channel equalizer only (ARCHITECTURE.md §V15 item 2: "FreqBlock demoted to `LDPCDecoder.reasoning` only, 2 layers")
- **V18 removed entirely** from runtime path (V18 design point 10: "FreqBlock demoted entirely, Source = Attention, channel = LDPC BP")
- **What it was supposed to do:** Be the "language computer" via 2D FFT frequency-domain mixing (OFDM analog)
- **What actually happened:** V15 analysis (§V15 root cause): "**FreqBlock was the language computer → weak conditional model.**" It beat unigram (5.99→5.6) but never approached bigram CE (~3.7). FFT is structurally unsuited for autoregressive language modeling — it's global, translation-equivariant, and loses positional locality.
- **V19 salvage:** ⚠️ **Maybe — as a channel equalizer only.** FreqBlock has a legitimate role as a frequency-domain MIMO equalizer (its original 5G purpose). V15's lesson: do NOT use FreqBlock in the Source path. Keep it as an optional channel-side component if the LDPC decoder needs an equalizer; do not let it replace Attention in Source.

### 8. Kalman filter module
- **Introduced:** V7 commit `91a8906` (5G NR pipeline)
- **Modified:** V5 refactor `90dceff` replaced Kalman with `LearnedUncertainty`
- **Re-added:** V8 commit `c425565` (SCS codec with Kalman)
- **Fixed:** commit `91d73aa` (softplus-parametrized Q/R, no clamps)
- **Removed:** V18 commit `69e2e07` (design point 10: "no LearnedUncertainty (Kalman framing — dead in V17)")
- **What it was supposed to do:** Bayesian-optimal state estimation for the decoder (predict+update with diagonal covariance)
- **What actually happened:** Kalman was "dead in V17" — V18 commit body. The diagonal-covariance assumption is too weak to capture real error correlations, and the Q/R parameters collapsed to trivial values during training (commit `4a7df07` tried to fix by initializing Q/R at sigmoid(-2)=0.12 instead of 0.5, but it didn't save the module).
- **V19 salvage:** ❌ **Do not bring back Kalman framing.** It added 6+ learnable parameters and a predict-update loop for no measurable gain. The V18 LDPC BP `z += sigmoid(gate)·H^T·syndrome` is simpler and trains. If you want Bayesian flavor, use a learned scalar variance per position (V9's `LearnedUncertainty C+1 params`), not a full Kalman.

### 9. HARQ (incremental redundancy)
- **Introduced:** V7 commit `85b3faf` (HARQ Type II)
- **Simplified:** V9 commit `9dea097` (HARQBuffer MLA quartet → single `read_out` Linear, 4× param reduction)
- **Soft combining added:** V7.2 commit `a6eca74` (uncertainty-weighted)
- **Removed:** V18 commit `69e2e07` (design point 2: "no HARQ buffer")
- **What it was supposed to do:** Store extrinsic updates across BP iterations and combine them with uncertainty weighting (HARQ soft combining analog)
- **What actually happened:** Added complexity for no measurable gain. The `HARQBuffer.read/combine` path was consistently hard to keep alive (V7.1 commit `bdcf3ab` had to fix "MSA write-side dead gradients" affecting HARQ feedback). V18's strict BP loop doesn't need it.
- **V19 salvage:** ❌ **Skip HARQ.** If iterative refinement is needed, use explicit BP iterations with per-iteration gates (V18 design). HARQ soft combining is a 5G retransmission-layer technique that doesn't map cleanly to single-pass training.

### 10. EXIT chart halt
- **Introduced:** V8 commit `c425565` (MI-based convergence estimator, `exit_chart.py`)
- **Modified:** commit `4a7df07` (cosine similarity instead of sigmoid proxy)
- **Modified:** V5 refactor `90dceff` (norm-based halting, no GPU sync)
- **Removed as halt:** V18 commit `69e2e07` (design point 10: "EXITChartEstimator as halt; kept as diagnostic")
- **What it was supposed to do:** Halt BP iterations early when extrinsic information stops changing (convergence)
- **What actually happened:** EXIT halt starved training layers (V14 §V14 item 3: "Early EXIT halt disabled while `training` so every layer is applied once"). The cosine/norm proxy was too aggressive — it halted before the decoder had a chance to learn.
- **V19 salvage:** ⚠️ **Keep as diagnostic only, never as training halt.** An EXIT-style MI estimator is useful for *logging* decoder convergence at inference, but should not gate training BP iterations.

### 11. Mutation / explosion modules in channel
- **Introduced:** V7.1 commit `4749050` (neural mutation + FOXP2 plasticity)
- **Made conditional:** commit `4a7df07` (gate by residual magnitude + zero-init mut_up_w)
- **Rank raised:** V7.1 commit `bdcf3ab` (mutation rank 8→32)
- **Removed:** V18 commit `69e2e07` (design point 2: "no mutation branch")
- **What it was supposed to do:** Inject exploration noise into the turbo loop (5G neural post-equalizer analog)
- **What actually happened:** Without the `4a7df07` magnitude-gate fix, mutation injected noise unconditionally — even when residual was 0. With the fix, it contributed "high-pass detail" but at the cost of 4.6K extra params and training instability.
- **V19 salvage:** ❌ **Skip.** Mutation is a regularizer looking for a problem. If exploration is needed, use standard dropout or stochastic depth on the BP reasoning layers.

### 12. Clifford algebra / multimodal cross-modal
- **Introduced:** V8 commit `c425565` (`clifford_cross_modal.py`, unified Clifford rotor cross-modal mixing)
- **Disabled:** V18 commit `69e2e07` (design point 10: "CliffordCrossModal — removed from runtime path. Multimodal remains config-gated but is not on the critical path")
- **Branch name:** `feat/clifford-cross-modal-v4` — this branch was created *for* Clifford cross-modal, and V18 (HEAD) disables it.
- **What it was supposed to do:** Unify image/audio/text cross-modal mixing via geometric product in frequency domain + rotor sandwich (replacing `CrossModalFreqMix` + `CrossModalGP2D`)
- **What actually happened:** Never produced documented wins. Multimodal was "disabled by default" since V7.1 (`a3ca352`). The Clifford module is mathematically elegant but competes for capacity with the core language task.
- **V19 salvage:** ⚠️ **Defer to V20+.** Multimodal is a research direction, not a language-modeling improvement. V19 should focus on monomodal SCS. Keep the Clifford module in-tree but config-gated and OFF by default.

### 13. Contrastive loss / InfoNCE
- **Introduced:** V7.1 commit `a3ca352` (`ContrastiveAlignment: InfoNCE modality alignment`)
- **Fixed:** commit `f78b627` ("implement real InfoNCE with batch negative samples")
- **Direction fixed:** commit `12d7961` ("fixed contrastive InfoNCE negative projection direction")
- **Disabled:** V18 (multimodal disabled, contrastive weight inherited the disable)
- **What it was supposed to do:** Align modality embeddings in a shared latent space
- **What actually happened:** Multiple bugs (projection direction wrong, fake vs real InfoNCE). Once fixed, no measurable gain because multimodal itself was off.
- **V19 salvage:** ❌ **Skip until multimodal is back on the table.** When multimodal returns, use the `f78b627` version (real InfoNCE with batch negatives) — that's the only version that was ever correct.

### 14. Weight tying `lm_expand ↔ token_compress`
- **V9 removed:** commit `9dea097` ("LM head is an independent rank-r factorization rather than a tied transpose, since a low-rank encoder has no natural inverse of the same rank")
- **V12 re-introduced:** ARCHITECTURE.md §V12 item 3 (`lm_expand.weight = embed.token_compress.weight`)
- **V18 removed again:** V18 design point 1 (independent source encoder + LM head)
- **What happened:** V9's argument (no natural inverse for low-rank encoder) was theoretically sound but ignored that the *tied* table is the ML source decoder under Gaussian-channel SCS. V12 re-tied to save 25M params. V18 un-tied again because V18's dims are small enough (H=384, r=128) that duplication is cheap.
- **V19 salvage:** ⚠️ **Depends on dim budget.** At V18 scale (20M params), untying costs ~6M params — acceptable. At V15 scale (25M with H=512), tying saved enough to fund an extra reasoning layer. **Rule: tie if and only if `V*r > 5M params`. Otherwise untie for capacity.**

### 15. RoPE vs sinusoidal PE
- **V7 used FreqCoding2D** (no RoPE, no sinusoidal) — commit `91a8906`
- **RoPE removed:** V7.1 commit `eecc5e9` ("freq_coding always on, _rope never called")
- **RoPE re-added:** V16 commit (ARCHITECTURE.md §V16 item 3: "RoPE in AttentionBlock. Standard rotary on q,k before SDPA")
- **V18:** config has `attention.rope_theta: 10000.0` but V18 uses AttentionBlock (Source) — RoPE is back
- **What happened:** RoPE was dead code in V7–V15 (FreqCoding handled position). V16 brought back Attention (replacing FreqBlock in Source) and needed RoPE for attention position info. V18 keeps Attention + RoPE.
- **V19 salvage:** ✅ **Use RoPE with Attention Source (V15/V16/V18 design).** `rope_theta=10000` is standard. Do not combine RoPE with FreqCoding2D — that's redundant.

### 16. AWGN schedule (σ value progression)
- **V7.2 introduced AWGN:** commit `a6eca74` (σ 0.005→0.0, anneal to 50% of training)
- **V10 shortened anneal:** ARCHITECTURE.md §V10 item 4 (`awgn_end_frac` 0.5→0.3, "channel clean earlier")
- **V9–V11:** σ_start=0.005, σ_end=0.0, end_frac=0.3
- **V16:** σ_start=0.005, σ_end=0.0, end_frac=0.3
- **V17:** σ_start=0.005, σ_end=0.0, end_frac=0.3
- **V18:** σ_start=**0.05**, σ_end=**0.01** (never zero), end_frac=0.3
- **What happened:** V17's σ_end=0.0 meant parity was *not load-bearing* by end of training (no noise to correct). V18's σ_end=0.01 keeps parity load-bearing forever.
- **V19 salvage:** ✅ **Use V18's σ_start=0.05, σ_end=0.01.** σ_end>0 is critical — without noise, the LDPC decoder has nothing to decode. Verify σ never reaches 0 during training.

### 17. Suffix curriculum shape
- **V12:** not present
- **V16:** `suffix_probability=0.6` (high — train like generation)
- **V17:** `suffix_probability=0.25` + curriculum (early 0.25, linear to 0.7 by 10k, then 0.85) + quadratic start sampling
- **V18:** `suffix_probability=0.30` base + ramp to 0.85 only in Phase 3 (last 20%)
- **What happened:** V16's 0.6 was too aggressive (forced long suffixes before Source could model short ones). V17's curriculum was better but the quadratic `u^2 * count` sampler biased toward very long suffixes early. V18's "base 0.3 + ramp only in last 20%" is conservative — let language dominate first.
- **V19 salvage:** ✅ **Use V18's conservative schedule.** Base 0.3, ramp to 0.85 only in Phase 3. Avoid the V17 quadratic sampler.

### 18. LDPC sparse parity vs dense parity
- **V8:** `SparseParityEncoder` with fixed sparse connectivity (commit `c425565`)
- **V13 fixed-graph LDPC:** ARCHITECTURE.md §V13 item 1 (`parity_base` non-trainable buffer, learnable `edge_log_scale` only) — to stop parity 0.14→0.01 collapse
- **V18:** Pure LDPC BP decoder with `H^T` back-projection (commit `69e2e07` design point 2)
- **What happened:** V12's fully-learnable LDPC collapsed (parity→0.01). V13's fixed-graph + learnable-scale-only stopped the collapse. V18 went further: pure BP with `z_pred += sigmoid(gate)·H^T·syndrome`, parity reached 0.36.
- **V19 salvage:** ✅ **Use V18's pure LDPC BP.** Key: **fixed sparse graph** (V13 lesson) + **H^T back-projection** (V18) + **gated update** (V18). Avoid fully-learnable parity (V12 collapse).

### 19. Mixture of Experts / sparse routing
- **V5–V6:** MoE was in the model (cfg_v5/cfg_v6: `moe.num_experts: 4, top_k: 1`)
- **V6 optimized:** commit `c52702d` (shared basis MoE, PAW-style, 5M param savings)
- **V7 removed:** commit `91a8906` ("Removed: GDR, VIB, coherence head, deep supervision, water filling, z_H/z_L state machine, perception/expression split, MoE")
- **V7.1 final cleanup:** commit `a3ca352` (15 unused dataclasses removed including MoE)
- **What it was supposed to do:** Sparse expert routing for capacity scaling
- **What actually happened:** MoE was removed when V7 pivoted to 5G/FFT architecture. Never tested in the V8+ SCS pipeline.
- **V19 salvage:** ⚠️ **Defer.** MoE adds routing instability. V19 should focus on getting a clean SCS pipeline training. If capacity is needed, scale H/C first; MoE is a V21+ question.

### 20. Spectral cache / frequency-domain processing
- **Introduced:** V7.1 commit `a3ca352` ("spectral cache for inference, OFDM cyclic prefix analog")
- **Integrated:** commit `22caf99` (into model forward + generation)
- **Removed (implicit):** V18 doesn't use it (FreqBlock gone from runtime path)
- **What it was supposed to do:** Inference-only cache of hidden states at layer boundaries + Kalman P state. O(W*H) vs KV cache O(T*H) — 4-32× memory reduction
- **What actually happened:** Was useful when FreqBlock was the main compute path. With Attention Source (V15+) and no FreqBlock runtime (V18), spectral cache has nothing to cache.
- **V19 salvage:** ❌ **Skip unless FreqBlock returns.** If V19 uses Attention Source (recommended), standard KV-cache is sufficient.

---

## C. Regression Forensics — Top 3 Regressions

### Regression 1: V15 → V16 (the "source skip works, channel doesn't" regression)
- **V15:** mce=2.62 (best ever)
- **V16:** mce=4.13 best, 5.28 mean over steps 1500–2000
- **Source:** ARCHITECTURE.md §V16 result line + §V17 opening paragraph

**Root causes (config + code interaction):**
1. **Config:** `channel_open` init changed from V15's `-3.0` (source-only) to V16's `0.0` (slightly open), with `open_bias = min(1, step/(2*warmup))*5 - 3` schedule. At step 0 the channel mix is `sigmoid(0 + -3) = 0.047` (5%); by `2*warmup` it's `sigmoid(0 + 2) = 0.88` (88%). V15's `-3.0` kept the channel essentially closed; V16 forced it open.
2. **Code:** `rate_up.expand` changed from V15's zero-init to V16's `N(0, 0.02)`. Combined with the opening schedule, this injected unscaled Gaussian noise into the latent path from step 0 — before the Source had learned anything.
3. **Config:** `suffix_probability` raised from V15's default to V16's 0.6 — this forced long-suffix tasks before the model could handle them.

**The bypass caught V15's win:** V18 architect analysis (commit body line 11–13): *"V15 reached mce=2.62 through the source_skip_scale bypass (channel path was idle). V17 kept the bypass but added a channel curriculum that never opened."* V15 wasn't really doing SCS — it was a Source-only model with a dead channel. V16's mistake was trying to activate the channel while keeping the bypass.

**Verdict:** Interaction regression (config + code). The code (source_skip_scale + channel_open schedule) was fine in isolation; the *combination* of an active bypass + an aggressive channel-open schedule + high suffix probability created three competing gradient signals.

### Regression 2: V16 → V17 (the "channel never opens" regression)
- **V16:** mce=4.13 best
- **V17:** mce=4.18 best (slightly worse on peak, but V17 also had worse mean)
- **V17 parity:** 0.0096 (dead, vs V18's 0.36)

**Root causes (config-only):**
1. **Config:** `channel_open` schedule slowed down: V17 used `open_bias = min(1, step/(4*warmup))*5 - 4` — at `2*warmup` the bias is still -1.5 (sigmoid≈0.18), full open only at `4*warmup`. V17 checkpoint showed `channel_open=0.086` — never opened. (V18 commit body line 5)
2. **Config:** `w_parity=0.005` (V16 was 0.01) — loss contribution 5e-7, below noise floor. LDPC encoder received zero gradient.
3. **Config:** `w_parity_diversity=0.0` (YAML bug, was supposed to be >0). LDPC graph free to collapse — and it did (parity stuck at 0.0096).
4. **Config:** `bp_iterations=1` (V16 was 2) — single BP iteration means no actual belief propagation.
5. **Code:** `corr_gate_b` code init=−1.0 but checkpoint=+1.0 — inconsistency. The V17 training trajectory used `sigmoid(+1)=0.73` (open), not the intended `sigmoid(-1)=0.27` (closed). This means the decoder was *overconfident* from step 0, not underconfident.

**Verdict:** Config-only regression. The code was identical to V16; the hyperparameters were wrong. The YAML bug (`w_parity_diversity=0.0`) is a typo, not a design choice.

### Regression 3: V9 → V10 (the "shared decoder weights" regression)
- **V9:** loss 7.32 → 1.31 (step 6339)
- **V10:** regression +77% to 2.31, parity 0.005 → 0.069 (diverging), grad 0.90 → 11.69 (explosion)
- **Source:** ARCHITECTURE.md §V10 opening paragraph + V10 commit `9dea097` body

**Root causes (code-only):**
1. **Code:** V8/V9 decoder shared `shared_w`, `shared_phase`, `shared_ffn` across all 5 reasoning layers. Every BP iteration produced **identical extrinsic information**. Belief propagation requires each iteration to add new information; identical updates leave the parity residual divergent. (§V10 item 1)
2. **Code:** `raw_alpha_t/h = 0.5` initialized identically on all layers — symmetric fixed point, optimizer never moved them. (§V10 item 2)
3. **Config:** `max_grad_norm = None` (no clipping) → grad exploded to 11.69 (and 50.25 in last 50 steps). (§V10 item 3)
4. **Config:** `awgn_end_frac = 0.5` (long anneal) → at step 12000, systematic still had σ≈0.0042 noise. (§V10 item 4)

**Verdict:** Code regression (shared decoder weights). The fix was per-layer decoder weights (`shared_weights=None`) — V10 item 1.

---

## D. "Lost Wins" Inventory

Techniques that produced documented improvement in their era but were later removed.

### 1. Attention Source (V15) — removed V16, restored V18
- **V15 win:** mce 5.99→2.62, breaking unigram floor for the first time
- **Why removed:** V16 didn't *remove* Attention Source — it kept it but added channel curriculum on top, which regressed. Attention Source itself was never the problem.
- **Was removal justified:** No — V18 brought it back (design point 1).
- **V19:** ✅ **Definitely keep.** Attention Source is the language computer. FreqBlock in Source doesn't work.

### 2. Train/infer BP asymmetry (V13) — lost V14, restored V18
- **V13 win:** Faster training + better inference (bp_train=2, bp_infer=4)
- **Why lost:** V14 set `bp_iterations=4` for both to "cover all reasoning layers" — solving a different problem (layers 2–3 starved).
- **Was removal justified:** No — V14 created a new problem (slower training, vanishing gradients through 4 unrolled BP iters).
- **V19:** ✅ **Definitely keep.** V18 design point 5 makes this explicit. Add CI test: `bp_train < bp_infer`.

### 3. Fixed-graph LDPC (V13) — partially lost V14, restored V18
- **V13 win:** Stopped parity 0.14→0.01 collapse by making `parity_base` non-trainable
- **Why partially lost:** V14 changed the rate matcher to learned `FactoredLinear` (correct fix) but didn't explicitly preserve the fixed-graph LDPC invariant. V17's YAML bug (`w_parity_diversity=0`) let the graph collapse again.
- **Was removal justified:** The rate-matcher change was justified; the LDPC graph collapse was collateral.
- **V19:** ✅ **Use V18's pure LDPC BP with H^T back-projection.** Fixed sparse graph + learnable edge scales + `w_parity_diversity > 0`.

### 4. LayerScale (V5) — replaced by RMSNorm V14, restored V18
- **V5 win:** commit `90dceff` added LayerScale to FreqBlock for "principled stabilization"
- **Why replaced:** V14 introduced `bottleneck_norm` (RMSNorm) as part of Learned Rate Matcher
- **Was removal justified:** No — RMSNorm destroyed information magnitude (V18 finding).
- **V19:** ✅ **LayerScale bottleneck (V18 design point 3).** Avoid RMSNorm/LayerNorm in channel-facing path.

### 5. Mask_embed zero init (V7.1) — kept through V15, removed V18 (different design)
- **V7.1 win:** commit `bdcf3ab` — "mask_embed: zero init instead of mean(embed) — prevents mask signal dominance (norm 1.99 was highest in vocabulary)"
- **V18 status:** Not directly applicable (V18 uses `semantic_unknown_embed`, a separate learned erasure indicator, not a mask_embed addition). The principle (don't let mask/unknown signal dominate) is preserved.
- **Was removal justified:** Yes — V18's design separates concerns better.
- **V19:** ✅ **Keep the principle: unknown/mask embeddings must have small norm at init.** V18 implements this via independent `semantic_unknown_embed` with controlled init.

### 6. Distillation hidden-state alignment (V7.1) — disabled V12, never restored
- **V7.1 win:** commit `bdcf3ab` replaced KL-on-logits with MSE-on-hidden-states (causal teacher logits conflict with masked LM same-position targets)
- **Why disabled:** V12 set `distill_alpha=0.0` (distribution mismatch: masked bidirectional student vs causal unmasked teacher)
- **Was removal justified:** Yes — V12 analysis is correct that the distributions are incompatible. But V7.1's MSE alignment is the *correct* distillation form for masked LM (it doesn't conflict with same-position targets).
- **V19:** ⚠️ **Reconsider with V7.1's MSE form + low alpha (0.05).** The V12 decision to fully disable was an overreaction. MSE alignment on hidden states with a low alpha (0.05–0.1) is a legitimate regularizer. Verify teacher is masked-LM-compatible (or use only embedding transfer, which is always safe).

### 7. Tiered / rate-distortion loss (V7) — disabled V18
- **V7 win:** commit `f737ba6` added `rate_distortion` loss (penalize info loss through bottleneck)
- **V18 status:** `w_rate_distortion=0.005` (low but not zero)
- **Was reduction justified:** Partial — V18 keeps it low to avoid Goodhart collapse.
- **V19:** ✅ **Keep at V18's level (0.005).** Don't go to 0; don't go above 0.02 (V14's value was too high).

---

## E. Information-Theoretic Pattern Analysis

Recurring failure modes from a Shannon-theory perspective:

### E1. Source-channel coupling failures (`source_skip_scale` bypass)
- **Shannon principle:** Source-Channel Separation Theorem requires *separate* optimization of source and channel coding. A bypass violates separation.
- **HAGI pattern:** V15 achieved its best result (mce=2.62) *through the bypass* — i.e., by violating SCS. V16/V17 tried to "have both" (bypass + channel) and got neither.
- **Root cause:** The bypass is a structural cheat. It lets the optimizer route gradients around the channel, so the channel atrophies. When the channel is later needed (V16/V17), it's dead.
- **V19 implication:** Strict SCS (V18 design) is correct. If Source needs a fallback, use a *scheduled* anneal, not a learnable bypass.

### E2. Bottleneck capacity mismatches (r too small for H)
- **Shannon principle:** Source coding rate r must exceed the entropy rate of the source. For natural language with V=49154, the per-token entropy is ~10.5 bits (SmolLM2 estimates) — so r ≥ 1024 (10 bits) is required.
- **HAGI pattern:**
  - V9: r=128 (7 bits) — below entropy floor, mce capped at ~6.0
  - V11: r=256 (8 bits) — still below, mce ~5.4
  - V12: r=512 (9 bits) — marginal, mce ~5.4
  - V18: r=128 (7 bits) — but V18 reaches mce=3.33 *because the source is Attention-backed, not bottleneck-limited*
- **Root cause:** When the bottleneck is the *only* path, r limits capacity. When Source is Attention-backed (V15+), the bottleneck is just a rate-matcher for the channel — r can be smaller because the Source has already done the heavy lifting.
- **V19 implication:** With Attention Source, r=C (not r=H) is the right choice. Don't over-invest in r.

### E3. Channel code rate mismatches (r/H vs AWGN capacity)
- **Shannon principle:** Channel capacity C = W·log2(1 + SNR). For AWGN with σ=0.05 and signal power ~1, SNR≈400, capacity is huge — code rate 0.5 is well below capacity.
- **HAGI pattern:**
  - V10: `n_checks=1056` for `C=320` → actual rate 0.23 (much below 0.5 target). Code is too redundant.
  - V11: fixed to `n_checks=320` → rate 0.5 as intended
  - V13: parity 0.14→0.01 collapse (LDPC graph not fixed)
  - V18: `code_rate=0.5`, parity reaches 0.36 (healthy)
- **Root cause:** The rate must be computed from the *final* C (after YAML overrides), not the auto-configured estimate. V11 commit fixed this.
- **V19 implication:** Always recompute `n_checks` after YAML overrides. Add a startup assertion.

### E4. Goodhart collapses (`w_correction_alignment`)
- **Shannon principle:** The optimization target (CE) must dominate proxy losses. Proxy losses (alignment, parity) are regularizers — they should contribute <10% of total loss.
- **HAGI pattern:** V8 had `w_correction_alignment=0.01` which dominated CE (~1.75 vs CE ~6.0). The optimizer maximized alignment, ignored CE.
- **Root cause:** No upper bound on proxy weight. No monitoring of proxy contribution to total loss.
- **V19 implication:** All aux losses should be logged with their contribution to total loss. Assert `aux_contribution < 0.1 * ce_contribution`. V18 explicitly removed `w_correction_alignment` for this reason.

### E5. Cold-start deadlocks (zero-init everything)
- **Shannon principle:** At init, every path must carry non-zero gradient. Dead paths never wake up.
- **HAGI pattern:**
  - V11 decoder: `ext_gate=0` (extrinsic bypassed), `mut_up_w=zeros` (mutation dead), `corr_gate_b=-3.0` (gate 95% closed). Decoder was near no-op for ~200 steps. (§V12 item 4)
  - V15 `rate_up.expand` zero-init: channel silent until learned. (§V15 item 6)
  - V17 `corr_gate_b` code init=−1.0, ckpt=+1.0 — inconsistency
  - V18 explicit fix: `rate_up.expand` init `N(0, 1/√C)` so channel carries gradient from step 0 (design point 1)
- **Root cause:** Multiple gates zeroing each other out. Each gate in isolation is fine; the *combination* deadlocks the path.
- **V19 implication:** **Every learnable gate must have non-zero init.** `corr_gate_b=0.0` (sigmoid=0.5), `ext_gate=0.5` (tanh≈0.46), `mut_up_w` random. V18 design point 6 codifies this.

### E6. Suffix-curriculum pathologies
- **Shannon principle:** Training distribution must match inference distribution. Suffix tasks (long contiguous masks) match generation, but if introduced too early they overwhelm the Source before it learns local statistics.
- **HAGI pattern:**
  - V16: `suffix_probability=0.6` — too aggressive, Source couldn't model short spans first
  - V17: quadratic `u^2 * count` sampler — biased toward very long suffixes early, "without early CE kill" (the design intent was right but the sampler overshot)
  - V18: base 0.3, ramp to 0.85 only in last 20% (Phase 3) — conservative, language first
- **Root cause:** Curriculum shape matters more than curriculum presence.
- **V19 implication:** V18's conservative schedule is correct. Avoid quadratic samplers — use linear ramp.

---

## F. Top 5 Techniques to Bring Back for V19 (Ranked by Expected Value)

Each entry includes the specific redesign needed to avoid the original failure mode.

### 1. **Attention Source + LayerScale bottleneck + pure LDPC BP decoder** (V18 design, intact)
- **Expected value:** Very high. V18 reached mce=3.33 @ step 929 with parity 0.36 — both healthy. The architecture is sound.
- **Specific redesign:** **Don't redesign — extend.** V18 needs longer training (5000+ steps) to close the gap to V15's 2.62. Specific V19 changes:
  - Increase H from 384 to 512 (V18 architect note: "H=384 → H=512 is the natural next experiment if mce stalls")
  - Keep C=H/2 (V18 ratio)
  - Keep `bp_train=3, bp_infer=6` (asymmetry — invariant)
  - Keep AWGN σ_start=0.05, σ_end=0.01 (never zero — invariant)
  - Keep `w_parity=0.05, w_parity_diversity=0.05, w_parity_recovery=0.05` (parity load-bearing — invariant)
  - **Add CI tests** for: `bp_train < bp_infer`, `σ_end > 0`, `w_parity_diversity > 0`, `corr_gate_b` matches checkpoint

### 2. **Train/infer BP asymmetry as an invariant** (V13 win, lost V14, restored V18)
- **Expected value:** High. Saves 30–50% of training FLOP and avoids vanishing gradients through deep unrolled BP.
- **Specific redesign:** Don't just *use* asymmetry — *enforce* it. Add to `config.py`:
  ```python
  assert cfg.train.bp_iterations < cfg.model.refinement.num_iterations, \
      "V13 invariant: train BP must be < infer BP (lost in V14, restored V18)"
  ```
  And document in ARCHITECTURE.md that this is a hard invariant, not a default.

### 3. **Distillation as MSE-on-hidden-states with low alpha** (V7.1 win, lost V12)
- **Expected value:** Medium. Could accelerate early training by 2–3× if the teacher is well-chosen.
- **Specific redesign:** V7.1's form (MSE on hidden states, not KL on logits) is correct for masked LM. V12's mistake was setting alpha=0 entirely instead of lowering it. Use:
  - `distill_alpha_start=0.1, distill_alpha_end=0.05` (V11 values, not V12's 0.0)
  - `distill_kl_enabled=false` (V7.1+ — never use KL on causal teacher logits)
  - MSE on hidden states only (V7.1 form)
  - Teacher: SmolLM2-360M (causal, but MSE-on-hidden is direction-agnostic)
  - End distillation at 60% of training (`distill_end_frac=0.6`)
  - **Critical: verify the distill_align is registered on the student BEFORE optimizer construction** (commit `05bd308` bug)

### 4. **Fixed-graph LDPC + H^T back-projection + parity_diversity** (V13 + V18 hybrid)
- **Expected value:** Medium-high. Prevents the V12/V17 parity collapse.
- **Specific redesign:** V18's pure LDPC BP is the right decoder. To make it robust:
  - `parity_base` is a non-trainable buffer (V13 invariant)
  - `edge_log_scale` is per-check learnable (V13)
  - BP update: `z_pred += sigmoid(corr_gate_w * |res| + corr_gate_b) · H^T · syndrome` (V18)
  - `corr_gate_w=0.1, corr_gate_b=0.0` init (V18)
  - `w_parity_diversity=0.05` (V18 — prevents graph collapse)
  - **Add LDPC graph density monitor** — log the number of "active" edges per check every 500 steps; alert if it drops below 50% of init.

### 5. **Conservative suffix curriculum with linear ramp** (V18 design, refined)
- **Expected value:** Medium. Avoids the V16 "too aggressive" and V17 "quadratic overshoot" failure modes.
- **Specific redesign:** V18's schedule (base 0.3, ramp to 0.85 in Phase 3) is close to right. Refinements:
  - **Linear ramp only** — no quadratic samplers (V17 lesson)
  - **Phase 3 starts at 80% of training** (V18) — not 50% (too early) or 95% (too late)
  - **Suffix length cap** at `max_new_tokens` (64 in V18 config) — don't generate suffixes longer than inference will ever see
  - **Mix random + suffix tasks 50/50** (V12 contract) — don't go suffix-only
  - **Monitor `suffix_ce - masked_ce` gap** — if >1.0, the model is suffix-starved; if <0.1, suffix is too easy

---

## G. Open Questions (for architect / planner)

These questions need answers before V19 planning can proceed. They are *not* market-value questions (that's the PM's job) — they are implementability questions.

- [ ] **V18 metrics in ARCHITECTURE.md table vs actual log:** The doc claims V18@929 "parity_metric 0.1948". The log shows step 929 par=0.118, step 925 par=0.36. Which is the canonical "V18 parity" number? The 0.1948 figure is closer to the pre-switch average (0.088) × 2 — possibly a typo or an aggregation. **Why it matters:** V19 baselines against V18; ambiguous baseline = ambiguous improvement claim.
- [ ] **V12–V17 commit provenance:** None of V12–V17 were individually committed. ARCHITECTURE.md §V12 references `train_20260718_220707.log` (not in repo) and `checkpoints/step-XXXXXX.pt` (not in repo). Are these artifacts recoverable, or is V12–V17 history only trustworthy via ARCHITECTURE.md narrative? **Why it matters:** If only the narrative is trustworthy, V19 must treat V12–V17 claims with the same skepticism as unverified benchmarks.
- [ ] **V18 capacity choice (H=384 vs H=512):** V18 architect note says "H=384 → H=512 is the natural next experiment if mce stalls". Is V19 the experiment, or does V19 stay at H=384 and extend training? **Why it matters:** H choice affects every downstream dimension (C, head_dim, param budget).
- [ ] **Source-channel tying decision:** V18 un-tied `lm_expand ↔ token_compress` (independent rank-r factorizations). At H=384, r=128, this costs ~6M extra params. Should V19 re-tie to save params for more reasoning depth, or stay un-tied? **Why it matters:** The V12 re-tying saved 25M; at V18 scale the savings are smaller but non-trivial.
- [ ] **Multimodal / Clifford path:** V18 disabled CliffordCrossModal and MultimodalInput. The branch is named `feat/clifford-cross-modal-v4` — is multimodal still a V19 goal, or is it deferred to V20+? **Why it matters:** If multimodal is in scope, V19 needs a different Attention design (multi-stream input); if deferred, V19 stays monomodal.
- [ ] **FreqBlock as channel equalizer:** V18 removed FreqBlock entirely. Should V19 add it back as an *optional* channel-side equalizer (V15 design), or keep the pure LDPC BP loop (V18)? **Why it matters:** FreqBlock-as-equalizer might help at low SNR (high AWGN σ); pure LDPC is simpler.
- [ ] **Distillation form:** Should V19 use V7.1's MSE-on-hidden-states (my recommendation #3 above), V11's alpha schedule (0.1→0.05), or stay teacher-free (V18 default `distill_enabled: false`)? **Why it matters:** Distillation requires a local teacher (SmolLM2-360M, ~720MB) — adds setup complexity.
- [ ] **Evaluation protocol:** V18 smoke test ran 1000 steps with sequential curriculum (tinystories → ... → openwebmath). The openwebmath switch at step 937 caused mce to jump 3.33 → 6.66. Should V19 use a fixed-domain smoke test (e.g., 1000 steps tinystories only) for cleaner baseline comparison? **Why it matters:** Mixed-domain smoke tests confound architecture changes with data-distribution changes.

---

## H. Methodology Notes & Limitations

1. **Repository-verifiable evidence:** V18 metrics (mce=3.33 @ step 929, par=0.36 @ step 925, avg mce 3.83 over last 100 pre-switch) are verifiable in `logs/v18_archive/v18_smoke_1000.console.log`. All V8–V17 metrics are quoted from `docs/ARCHITECTURE.md`, which is an in-repo document but was written *at V18 time* looking back at V8–V17 — not contemporaneously.

2. **Missing intermediate commits:** V12–V17 are design states documented in ARCHITECTURE.md but not in git history. The git log jumps from V9–V11 (`9dea097`, 2026-07-18) directly to V18 (`69e2e07`, 2026-07-20), with only a small fix (`0c99abc`) in between. This means V12–V17 code changes are *not recoverable* — V19 must trust the ARCHITECTURE.md narrative.

3. **Untracked training logs:** References to `train_20260715_180755.log`, `train_20260718_132801.log`, `train_20260718_142908.log`, `train_20260718_220707.log` appear in commit bodies and ARCHITECTURE.md but none are in `logs/`. Only V18-era logs (2026-07-20) are present. Pre-V18 quantitative claims should be treated as author evidence, not repository evidence.

4. **Source code for V12–V17 not recoverable:** Without commits, `git show` cannot retrieve the V12–V17 versions of `hagi_v4.py` or `8gb_canonical.yaml`. The §C regression forensics for V15→V16 and V16→V17 therefore rely on ARCHITECTURE.md descriptions of what changed, not on direct diff inspection. The V9→V10 forensics are also narrative-based (V10 was not committed).

5. **Branch context:** This is the `feat/clifford-cross-modal-v4` branch. Other branches (if any) were not examined. `git branch -a` shows only this branch plus its remote tracking — the V8–V18 history is entirely on this one branch.

6. **Analyst role boundary:** Per role constraints, this report provides *evidence only*. The V19 architecture decision is the architect's job. The top-5 techniques in §F are ranked by *expected value given the evidence*, not by market priority.
