# HAGI V34 Architecture

V34 is a **diagnostic release**: it establishes, with measurements, that the
V33 speed-tuned core already sits at the hardware ceiling of the Radeon 8060S
iGPU, adds one structural improvement (fused QKV), and adds a complete but
opt-in implementation of **Explorative Modeling** (XM) as the scaling lever
that matters when the bottleneck is memory bandwidth rather than FLOPs.

## The measured hardware ceiling

Everything below was measured on the Radeon 8060S (115 GB unified, ROCm 7.13):

| Metric | Value |
|---|---|
| Memory bandwidth | **107 GB/s** (DDR5) |
| Peak compute (bf16) | 33.7 TFLOPS |
| V33 baseline step | 2.47-2.55 s, ~12.1-12.4k tok/s (batch 30×1024) |
| Forward | 0.72 s (29%) |
| **Backward** | **1.76 s (71%)** — the bottleneck |

The model is **bandwidth-bound**: backward re-reads activations and weights at
107 GB/s, and that traffic dominates wall-clock. Every micro-optimization was
measured and none moved the needle more than 0-6%:

- **torch.compile** — 0.56x *slower* than eager on this ROCm build
- **batch 30→96** — tok/s flat (12k→11.5k): bandwidth, not launch, is the cap
- **seq 1024→512** — +5%: attention is not the bottleneck
- **H 512→1152, L 40→13** — the current H=1152 L=13 is *fastest*: small matrices
  underfill the iGPU
- **gradient checkpointing** — −17%: recompute burns bandwidth
- **ternary on/off** — 0%
- **fused QKV** — 0-2% (kept: structurally cleaner, halves the linear count)
- **fp32 variance on/off** — 0%
- **head chunk 8192→30720** — −3%

**Head loss is 692 ms = 27% of the step** (forward 141 ms, backward 551 ms at
N=30720, V=32768). The chunked CE with `save_logits=True` is already optimal;
single-shot materialization measured *slower* with z-loss (1020 vs 693 ms).

**Conclusion:** V33's core cannot be made meaningfully faster on this part. The
path to "faster training" on a bandwidth-bound GPU is **more quality per step**
(converge in fewer steps), not more tokens per second.

## Explorative Modeling (XM) — opt-in

XM (Gladstone, Ji, Du — arxiv 2607.27372) factors the *training loop* instead
of the generation procedure: fix a data target, draw K candidates from the
model's own generation, train on the closest. This scales *generative
expressivity* — how many modes a prediction can commit to — which ordinary
next-token factorization fixes at design time.

**Channel reading:** the LM head becomes a *list decoder*. Standard CE forces
one hypothesis (the blur: the mean of all valid continuations). Best-of-K emits
K hypotheses (one per learnable mode-code) and keeps the one nearest the target
— the maximum-likelihood decision over K codewords. Gradient flows only through
the winner (paper Algorithm 1, memory-saving variant: K candidates forwarded
without gradients, only the winner re-forwarded with gradients).

**Implementation** (`config.XMConfig`, `head.py:_loss_xm`):
- `mode_codes [K, H]` — learnable hypotheses (small init std).
- Entropy gate — explore only positions whose *posterior entropy* (always
  `≤ ln V`) exceeds a fraction of `ln V`. CE was rejected as the gate: on an
  ambiguous target CE far exceeds `ln V`, firing the gate everywhere.
- One `[N,V]` forward computes log-partition, target logit and entropy together
  (no extra gate pass).
- Batched candidate scoring on explored rows; winner re-forwarded with
  gradients; rows where exploration found nothing better keep the base.

**Measured on 50 real-data steps:** XM ce=7.2016 vs baseline ce=7.2018 —
identical loss at +11% step time. The mode-codes (`init_std=0.05`) produce
candidates too similar to the base prediction to diverge into distinct modes.
XM is therefore **off by default** (`xm.enabled: false` in v34_1b.yaml). It is
the right scaling lever on a compute-bound GPU where extra forwards are cheap;
on this bandwidth-bound part the extra forwards cost more than the exploration
buys.

## Changes vs V33

1. **Fused QKV** (`attention.py`) — one `qkv_proj` instead of separate
   `q_proj`/`kv_proj`. Structurally cleaner (halves the linear count per layer),
   speed-neutral (measured 0-2%). Tests updated.
2. **XM module** — `XMConfig` in config.py (validated), `_loss_xm` +
   `_head_stats_blocks` in head.py, `xm_cfg` wiring in model.py. Opt-in.
3. **v34_1b.yaml** — same 147M core as V33, `xm.enabled: false`.

Checkpoint format unchanged (10). V34 trains from scratch; resume not
compatible with V33 checkpoints for XM-enabled runs (mode_codes are new
parameters).
