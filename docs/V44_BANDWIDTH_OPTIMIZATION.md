# HAGI V44 — bandwidth-optimal micro-geometry (measured)

## TL;DR

On the Radeon 8060S iGPU (DDR5 ~107 GB/s) the model is **bandwidth-bound**,
not compute-bound. Shrinking `hidden_size` cuts GEMM bytes moved through
DDR5 and raises tok/s monotonically. The V44 config (`H=128, L=4, ffn=1.0,
T=128, B=192`) runs **6.3x faster** than V43 (`H=768`) at **5x less memory**:

| Config | ms/step | body tok/s | peak mem | nce @ ~490s |
|--------|---------|-----------|----------|-------------|
| V43 (H=768) | 237 | 103.6k | 3.2 GiB | 1.81 |
| V44 (H=128) | 37.5 | 654.8k | 0.63 GiB | 2.08 |

## The bandwidth-bound regime inverts the depth/width tradeoff

A small wide model that streams tokens 6x faster reaches a *worse* loss in
the same wall-time than a large model that is bandwidth-starved — **on a long
horizon, capacity wins**. V43 (H=768) is the quality-per-wall-time optimum;
V44 (H=128) is the throughput optimum. Both are shipped.

## Measured sweep (all on the 8060S, T=128, B=192 unless noted)

### Hidden size (the decisive lever)
```
H=768  L=4  ffn=1.0  237 ms · 103.6k tok/s   (V43)
H=512  L=4  ffn=1.0  160 ms · 153.6k
H=384  L=4  ffn=1.0  109 ms · 225.6k
H=256  L=4  ffn=1.0   75 ms · 327.4k
H=192  L=4  ffn=1.0   55 ms · 448.3k
H=128  L=4  ffn=1.0   39 ms · 629.9k        (V44)
```

### Quality per wall-time (~490 s budget, nce after training)
```
H=768  L=4  x2000  steps  491 s -> nce 1.81   (best quality)
H=512  L=8  x2000  steps  601 s -> nce 2.01
H=256  L=4  x6500  steps  505 s -> nce 1.93
H=192  L=4  x8900  steps  506 s -> nce 1.97
H=128  L=4  x11000 steps  444 s -> nce 2.08   (best throughput)
```

## Rejected hypotheses (all measured slower per wall-time)

- **ffn=0.5 / L=3 at H=768** → slower (launch-bound, not FLOPs-bound)
- **torch.compile** → 262 ms vs 234 ms (ROCm codegen not optimal)
- **grad_checkpointing** → 297 ms (recompute costs more than saved BW)
- **B=384 / B=96** → B=192 is the tok/s sweet spot
- **T=256 / T=512** → flash-attn backward ∝ T² dominates
- **head_dim=64** → no change (not flash-attn-bound at T=128)
- **tie_lm_head=True** → 3.13 vs 3.06 nce (tied head gather is slower)
- **loop_depth=2** → 3.06 vs 2.66 nce (fewer unique steps)
- **L=6 / L=8 at H=128** → fewer steps per wall-time, worse nce
- **grad_accum=2/3** → no gain (B=192 accum=1 optimal)

## Punctured receiver disabled

`ce_keep_rate=0.5` (V43) is **disabled** in both configs. On this host-bound
iGPU the boolean-gather (`nonzero`+`index`) costs ~188 ms CPU/step — more than
the ~2% head compute it saves. `rate=1.0` scores every token, is faster
(234 vs 237 ms), and doubles the scored-token count.

## Files changed

- `configs/v43_1b.yaml` — `ce_keep_rate: 0.5 → 1.0` (puncturing disabled)
- `configs/v44_1b.yaml` — new bandwidth-optimal micro-geometry (H=128)
- `scripts/bench_train_step.py` — added `--compile`, `--grad-checkpointing`,
  `--seq-len` flags
- `scripts/breakdown.py` — new: precise forward/backward/optimizer wall-time
- `scripts/prof_step.py` — new: torch.profiler breakdown

## Tests

337 passed, 1 pre-existing failure (`test_v42_config_loads` expects
`seq_len=512` but `v42_1b.yaml` ships `seq_len=1024`; unrelated to this work).
