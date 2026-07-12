# HAGI V7.1 — Architecture Walkthrough & Refactoring Plan

## Architecture: 5G NR Codec Pipeline

```
Transport Block (tokens)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  SOURCE ENCODER                                         │
│  Embedding + Masking (BEC erasure)                      │
│  mask_token = erasure symbol, mask_ratio = erasure p    │
│  Adaptive masking = AMC (CQI-driven erasure rate)       │
└────────────────────┬────────────────────────────────────┘
                     │ [B, T, H]
                     ▼
┌─────────────────────────────────────────────────────────┐
│  OFDM PRE-EQUALIZATION (perception layers x N)          │
│  FreqBlock = 2D rFFT (OFDM demod) + complex weight      │
│  (MIMO equalizer) + FactoredSwiGLU (SVD FFN)            │
│  Soft freq gate = adaptive modulation (QPSK→256QAM)     │
└────────────────────┬────────────────────────────────────┘
                     │ [B, T, H]
                     ▼
┌─────────────────────────────────────────────────────────┐
│  RATE MATCHING (bottleneck)                             │
│  FFT-based H→C compression (raised-cosine roll-off)     │
│  CQI-adaptive gate = coding rate adaptation             │
│  H=576, C=288 → rate 1/2 code                           │
└────────────────────┬────────────────────────────────────┘
                     │ [B, T, C]
                     ▼
┌─────────────────────────────────────────────────────────┐
│  LDPC ITERATIVE DECODER (TurboLoop, N iterations)       │
│  Each iteration:                                        │
│    1. MSA.read  → DFE (decision feedback equalizer)     │
│    2. FreqBlock → OFDM re-equalization                  │
│    3. Kalman.predict → P += Q (uncertainty growth)      │
│    4. GP2D → parity check (LDPC check node)             │
│       MultiScaleGP2D = multi-degree parity (1,4,16)     │
│       Interleaving = burst error protection             │
│    5. Kalman.update → K = P/(P+R), z += K*innovation    │
│    6. Mutation + tanh saturator → soft clipping          │
│    7. MSA.write → HARQ buffer (soft info storage)       │
│  Convergence halt = EXIT chart stopping criterion       │
└────────────────────┬────────────────────────────────────┘
                     │ [B, T, C]
                     ▼
┌─────────────────────────────────────────────────────────┐
│  RATE DEMATCHING (up-sampling)                          │
│  Zero-pad C→H + raised-cosine window                    │
│  Spectral cache = OFDM cyclic prefix (ISI suppression)  │
└────────────────────┬────────────────────────────────────┘
                     │ [B, T, H]
                     ▼
┌─────────────────────────────────────────────────────────┐
│  SOURCE DECODER (expression layers x N)                 │
│  FreqBlock (shared weights with perception)             │
└────────────────────┬────────────────────────────────────┘
                     │ [B, T, H]
                     ▼
┌─────────────────────────────────────────────────────────┐
│  DEMODULATION → BITS                                    │
│  RMSNorm + LM Head (weight-tied with embedding)         │
│  logits → softmax → token probabilities                 │
└─────────────────────────────────────────────────────────┘
```

## Component → 5G NR Mapping

| HAGI Component | 5G NR Analog | File |
|---|---|---|
| Embedding + Masking | Transport block + BEC erasure | `hagi_v4.py`, `masking.py` |
| FreqBlock (2D rFFT) | OFDM demodulation (Hermitian) | `freq_layer.py` |
| FactoredLinear | CDMA spreading/despreading | `freq_layer.py` |
| FactoredSwiGLU | SVD-compressed FFN | `freq_layer.py` |
| Complex weight (low-rank) | MIMO channel equalizer | `freq_layer.py` |
| Soft frequency gate | Adaptive modulation (AMC) | `freq_layer.py` |
| FFT Bottleneck (H→C) | Rate matching (puncturing) | `hagi_v4.py` |
| CQI Estimator | Channel Quality Indicator | `cqi.py` |
| TurboLoop | LDPC iterative decoder | `hagi_v4.py` |
| GP2D | LDPC parity-check node | `gp2d.py` |
| MultiScaleGP2D | Multi-degree LDPC + interleaver | `multiscale_gp2d.py` |
| KalmanFilter | Optimal channel estimation | `kalman.py` |
| MSA.read | DFE (decision feedback equalizer) | `msa.py` |
| MSA.write | HARQ buffer (soft combining) | `msa.py` |
| TensorSlotRegistry | HARQ ring buffer | `msa.py` |
| SpectralCache | OFDM cyclic prefix | `spectral_cache.py` |
| Cl(3,0,0) geometric product | Space-time block code | `clifford.py` |
| ContrastiveAlignment | Slepian-Wolf distributed coding | `contrastive.py` |
| CrossModalFreqMix | MIMO cross-spectrum estimation | `cross_modal_attention.py` |
| Echo cancellation | G.168 echo canceller | `generate.py` |
| N-gram ban | LDPC forbidden codeword | `generate.py` |
| Distillation (KL) | Soft-decision combining | `distillation.py` |
| Embedding transfer | Pilot signal injection | `distillation.py` |
| FOXP2Controller | Adaptive coding rate (AMC) | `foxp2.py` |
| Muon optimizer | MIMO precoding (orthogonalization) | `optim.py` |
| Loss: CE | Fidelity (BER) | `losses.py` |
| Loss: Parity | Redundancy reward | `losses.py` |
| Loss: Whiteness | Decorrelation (white noise test) | `losses.py` |
| Loss: ExtrinsicInfo | Innovation (EXIT chart) | `losses.py` |
| Loss: Efficiency | Convergence cost | `losses.py` |
| Loss: RateDistortion | Information loss | `losses.py` |

## Refactoring Plan

### Phase 1: Critical Bug Fixes (8 fixes)

| # | File | Bug | Fix |
|---|---|---|---|
| B1 | `distillation.py` | `dtype=` instead of `torch_dtype=` | Replace param name |
| B2 | `outputs.py` | `torch.tensor(0.0)` without device | Add `device=residual.device` |
| B3 | `foxp2.py` | `torch.tensor([0.0, 0.0])` without device | Add device param |
| B4 | `train_v4.py` | `torch.device("auto")` invalid fallback | `else "cpu"` |
| B5 | `generate.py` | EOS suppression only position 0 | Apply to all positions < min_tokens |
| B6 | `msa.py` | `read_topk` includes empty slots | Mask slots beyond `num_written` |
| B7 | `msa.py` | `write` silently drops incomplete chunks | Pad or include remainder |
| B8 | `losses.py` | `w_msa_lb` hardcoded 0.01 | Use `cfg.train.w_msa_lb` |

### Phase 2: Dead Code Removal (12 items)

| # | File | Dead code | Action |
|---|---|---|---|
| D1 | `norms.py` | `build_rope_cache`, `apply_rope` | Remove (RoPE not used) |
| D2 | `freq_layer.py` | `cos`, `sin` params in `forward` | Remove from signature |
| D3 | `freq_layer.py` | `modality_ids`, `all_outputs` params | Remove from signature |
| D4 | `freq_layer.py` | `FreqBlock.forward` returns `(x, None, None)` | Return just `x` |
| D5 | `msa.py` | `_select_chunk_size`, `_chunk_low/high`, `_adaptive_chunk` | Remove |
| D6 | `msa.py` | `_scalar_slice`, `_bivector_slice` | Remove |
| D7 | `msa.py` | `attn_norm` (initialized, never used) | Remove |
| D8 | `generate.py` | `noise_ratio` param | Remove from signature |
| D9 | `clifford.py` | `METRIC = [1, 1, 1]` | Remove |
| D10 | `loop.py` | `sample_mask_pattern()` always returns "random" | Inline, remove span masking path |
| D11 | `inference/__init__.py` | Docstring "length prediction" | Fix docstring |
| D12 | `optim.py` | `_MOON_EXCLUDE` contains `"w_time"` | Remove if no such param |

### Phase 3: Performance Optimizations (7 items)

| # | File | Optimization | Technique |
|---|---|---|---|
| P1 | `freq_layer.py` | Cache complex weights for perception layers | Pre-compute `w_re_a @ w_re_b` once, reuse |
| P2 | `msa.py` | Eliminate GPU-CPU sync in `write` | Use `torch.roll` + `index_copy_` without `.item()` |
| P3 | `foxp2.py` | Use `torch._foreach_norm` for grad stats | Replace Python loop with vectorized call |
| P4 | `loop.py` | `soft_grad_scale` — use `torch.stack` | Replace `sum()` of tensors with `torch.stack([...]).sum()` |
| P5 | `optim.py` | Compute `scale_wd` once | Remove duplicate computation |
| P6 | `gp2d.py` | Replace `torch.roll` with padded shift | Avoid circular wrap-around artifacts |
| P7 | `generate.py` | Vectorize `ban_repeated_ngrams` for B>1 | Batch processing instead of Python loop |

### Phase 4: Architecture Improvements (6 items)

| # | File | Issue | Fix |
|---|---|---|---|
| A1 | `contrastive.py` | Not real InfoNCE (no negative samples) | Implement proper InfoNCE with batch negatives |
| A2 | `multiscale_gp2d.py` | `softmax` scale weights = competition | Use `sigmoid` gates for additive multi-scale |
| A3 | `loop.py` | FOXP2 `num_groups=1` = global gate | Split into per-layer groups for real plasticity |
| A4 | `config.py` | `auto_configure` counts removed MoE layers | Remove MoE terms from cost model |
| A5 | `checkpoint.py` | `get_latest_checkpoint` lex sort | Parse step number from filename, sort numerically |
| A6 | `multiscale_gp2d.py` | Triple gated blending (per-scale + fusion) | Simplify to per-scale gates + single fusion |

### Execution Order

```
Phase 1 (B1-B8) → Phase 2 (D1-D12) → Phase 3 (P1-P7) → Phase 4 (A1-A6) → Verify
```

### Verification

1. `python -c "from hagi_v4.model.hagi_v4 import HAGIv4"` — import smoke test
2. `python -c "from hagi_v4.train.loop import train"` — training import test
3. `python scripts/infer_v4.py --checkpoint checkpoints/step-007500.pt --prompt "hello"` — inference test
4. Check param count unchanged (dead code removal should not alter architecture)
