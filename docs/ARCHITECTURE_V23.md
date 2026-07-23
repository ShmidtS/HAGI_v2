# HAGI V23 — Full SCS Codec with Integrated Information-Theoretic Modules

## Overview

V23 completes the integration of ALL 16 information-theoretic modules into the
runtime path. No module is dead code — each has a defined position in the
Shannon Source-Channel Separation (SCS) pipeline with information-theoretic
justification.

V22 had 3000+ LOC of dead code (modules existed but were not wired). V23 wires
every module into its correct position, config-gated where appropriate.

## Architecture

```
                    HAGI V23 SCS Pipeline
                    =====================

SOURCE ENCODE (compress)
  ConvEmbedding (factorized V×r + r×H, pulse-shaping Conv1d)
    → semantic_unknown_embed (max-entropy erasure indicator)
    → sinusoidal pilot PE
    → AttentionBlock × N_perc (perception stack)
    → [MoE] variable-rate FFN (entropy-aware routing)        ← V23
    → [CQI] per-position channel quality estimation           ← V23
    → rate_down (FactoredLinear H→C)
    → LayerScale bottleneck (SiLU + tanh gate)
    → z_sys in R^C

CHANNEL ENCODE (protect)
  SparseParity (fixed LDPC graph + learnable edge_log_scale)
    → [WaterFilling] per-dim capacity allocation              ← V23
    → BlockInterleaver (QPP)
    → codeword [z_sys | parity]

PHYSICAL CHANNEL (corrupt)
  AWGN (adaptive sigma: 0.50 → 0.30)
    → Erasure (adaptive mask ratio via CQI)                   ← V23

CHANNEL DECODE (recover)
  De-interleave
    → [FreqCoding2D] MIMO equalizer (NOT source path!)        ← V23
    → split [z_sys, parity]
    → LDPC BP Loop:
        ├─ HARQ read (iterations > 0): extrinsic soft combine ← V23
        ├─ Kalman predict: P_pred = P + Q                     ← V23
        ├─ Syndrome: s = parity_recv - H @ z_pred
        ├─ Mahalanobis gate (d² vs χ²_crit)
        ├─ [LearnedUncertainty] per-position σ²               ← V23
        ├─ [CQI] modulates gate magnitude                     ← V23
        ├─ BP correction: H^T @ s (extrinsic-only)
        ├─ Kalman update: K = P/(P+R), z += K·innovation      ← V23
        ├─ HRM: z_L ← transition(z_L, z_pred)                 ← V23
        │        z_H ← transition(z_H, z_L_coarse)
        │        z_pred += z_h_to_hidden(z_H) + z_l_to_hidden(z_L)
        ├─ HARQ write: store extrinsic delta                  ← V23
        └─ EXIT chart convergence halt (‖extrinsic‖ < ε)      ← V23

SOURCE DECODE (reconstruct)
  rate_up (FactoredLinear C→H)
    → [LorentzSphereNorm] geometric regularization            ← V23
    → AttentionBlock × N_expr (expression stack, causal)
    → [ContrastiveAlignment] latent consistency (InfoNCE)     ← V23
    → lm_head

INFERENCE
  [SpectralCache] VRAM-efficient KV caching                   ← V23
  [SpeculativeDecoding] draft-then-verify                     ← V23
  [KVCache] block-parallel generation

MULTIMODAL (config-gated OFF by default)                      ← V23
  [MultimodalInput] text/image/audio source encoders
  [Clifford] Cl(3,0,0) geometric product cross-modal mixing
  [multimodal_masking] per-modality erasure indicators
```

## Module Integration Summary

| Module | Position | Role | Config Gate |
|--------|----------|------|-------------|
| CQI | Source encoder | Per-position channel quality → adaptive sigma, gate, mask | `model.cqi` |
| EXIT chart | Channel decoder | ‖extrinsic‖ < ε convergence halt | `model.exit_chart` |
| LearnedUncertainty | Channel decoder | Bayes-optimal Mahalanobis σ² | `model.uncertainty` |
| KalmanFilter | Channel decoder BP loop | K = P/(P+R) optimal blend | `model.kalman` |
| HARQBuffer (MSA) | Channel decoder BP loop | Extrinsic-only soft combining | `model.msa` |
| HRM (z_H + z_L) | Channel decoder BP loop | Dual-component turbo state | `model.hrm` |
| FreqCoding2D | Channel decoder pre-BP | MIMO frequency equalizer | `freq_coding.enabled` |
| MoE | Source encoder FFN | Variable-rate capacity allocation | `moe.enabled` |
| WaterFilling | Channel encoder | Shannon optimal dim allocation | `water_filling.enabled` |
| Lorentz | Source decoder | Geometric hyperboloid regularization | `lorentz_enabled` |
| Contrastive | Post source-encode | InfoNCE latent consistency | `w_contrastive > 0` |
| SpectralCache | Inference | VRAM-efficient caching | `kv_cache.enabled` |
| Speculative | Inference | Draft-then-verify | `speculative.enabled` |
| MultimodalInput | Source encoder | Multi-modality source coding | `multimodal.enabled` |
| Clifford | Cross-modal | Cl(3,0,0) geometric product | `multimodal.enabled` |
| multimodal_masking | Source encoder | Per-modality erasure | `multimodal.enabled` |

## Gigatoken Integration

V23 uses [gigatoken](https://github.com/marcelroed/gigatoken) for ~1000x faster
tokenization. The wrapper (`src/hagi_v4/data/tokenizer.py`) provides:
- `load_tokenizer(name)` — gigatoken compatibility mode with HF fallback
- `encode_files(name, paths)` — native Rust file API (fastest path)

Information theory: tokenization = source coding (BPE = variable-length code).
Gigatoken accelerates the encoder without changing the code, preserving
information-theoretic properties.

## CI Invariants (V23)

| # | Invariant | Rationale |
|---|-----------|-----------|
| 1 | `bp_train < bp_infer` | V13: train/infer BP asymmetry |
| 2 | `awgn_sigma_end > 0` | V17: channel never fully closes |
| 3 | `w_parity_diversity > 0` | V17: prevents LDPC graph collapse |
| 4 | `sigma_start >= sigma_end` | Schedule direction: noisy → clean |
| 5 | `attn_entropy_floor > 0` | V22: prevents attention collapse |
| 6 | `w_attn_entropy > 0` | V22: entropy regularization active |
| 7 | `code_rate ∈ (0, 1)` | Shannon SCS: neither pure parity nor pure systematic |
| 8 | `edges_per_check >= 1` | LDPC: each check needs ≥1 edge |
| 9 | `capacity(sigma_end) >= code_rate` | Shannon limit: C ≥ R for BP convergence |
| 10 | `hrm_stride >= 1` | HRM: spatial coarsening stride positive |

## AWGN Schedule (V23)

| Parameter | V22 | V23 | Rationale |
|-----------|-----|-----|-----------|
| sigma_start | 0.40 | 0.50 | SNR=4.0, capacity=1.17, 2.3× overcapacity |
| sigma_end | 0.15 | 0.30 | SNR=11.1, capacity=1.80, 3.6× overcapacity |

V22 had 2.8–5.5× overcapacity — BP did minimal work. V23 schedule (2.3–3.6×)
forces LDPC to do real error correction while staying above Shannon limit.

## Config (8gb_canonical.yaml)

Active modules by default: CQI, EXIT chart, LearnedUncertainty, Kalman,
HARQ, HRM. Disabled by default (config-gated): FreqCoding, MoE, WaterFilling,
Lorentz, Contrastive, Multimodal, SpectralCache, Speculative.

## Codec Framing (omc-codec-framing)

| Mechanism | V23 Implementation |
|-----------|-------------------|
| Extrinsic-only handoff | HARQ stores deltas, HRM uses gated residuals, BP correction from syndrome only |
| Convergence halt | EXIT chart ‖extrinsic‖ < ε (enabled by default) |
| Capacity matching | CQI → adaptive AWGN sigma, adaptive mask ratio, gate magnitude |
| Uncertainty signaling | LearnedUncertainty per-position σ², max-entropy mask_embed |
