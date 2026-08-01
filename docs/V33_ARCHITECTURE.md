# HAGI V33 Architecture

V33 adds two mechanisms to V32, both grounded in measurement:

1. **Recurrent spectral (Fourier) branch** — a sub-quadratic correlator that
   complements attention instead of replacing it.
2. **Body-scaled learning rate** — fixes the measured 30x gradient imbalance
   between the ternary body and the codebook.

## The spectral branch

```
                 ┌────────────  residual stream ────────────┐
                 │                                           │
   x ──► norm ──► attn ──► + ──► spectral ──► + ──► mixer ──► +
                 │      (content retrieval)  │        (nonlinear map)
                 │                    (frequency-local)
```

Attention is the correlator that content-addresses the past. It is
shift-invariant per query-key pair: to represent "the signal has energy at
frequency omega", a transformer must stack many layers whose softmax
weights compose into that band-pass. The spectral branch does it in one step.

### The recurrence

Each of `K` modes is a damped oscillator:

```
S_t = A * S_{t-1} + x_t,      A_k = r_k * exp(-i * omega_k)
```

`omega_k` are geometrically spaced frequencies (equal octaves), `r_k` the
retention (bandwidth). The impulse response of mode k is a complex sinusoid
decaying at rate `r_k`; the frequency response is a Lorentzian peak centered
at `omega_k`. The layer learns *where* the sequence's spectral energy lives.

### Parallel scan — why it is fast

The recurrence is solved in closed form. Splitting into blocks of `L`:

```
local(tau) = A^tau * sum_{j<=tau} x_{bL+j} A^{-j}     (within block)
S_end(b)   = A^L * S_end(b-1) + local(L-1)            (block recursion)
S(bL+tau)  = local(tau) + A^{tau+1} * S_end(b-1)
```

The first and third lines are `cumsum` plus elementwise multiplies — fully
parallel. Only the `T/L` block-end recursion is sequential. This is *exact*
(verified to ~1e-4 against a sequential reference) and runs in ~0.7ms on ROCm
vs ~360ms for `torch.fft` at the same shape. The cost is O(T*K) per layer with
a K-dimensional state — the KV-cache equivalent for a recurrence is 2*K
floats per layer, independent of context.

### Grokking ramp (spectral shift)

Grokking is the delayed memorization->generalization transition. The
spectral-shift hypothesis says networks first fit the low-frequency
(global, generalizable) structure, then high-frequency (memorizable) detail.
The ramp interpolates retention `r` from `damp_min` (narrow passband: only
low frequencies persist; high frequencies are filtered as noise) to `damp_max`
(full spectrum) over the first `ramp_steps` optimizer steps. It is a buffer
updated by the trainer — a *schedule*, like the MoE bias controller, not a
loss. No auxiliary objective competes with CE.

Measured: at initialization the branch gives `ce = 7.48` against the unigram
entropy baseline `7.63` — a free -0.15 nats/token before training starts.

### Why complement, not replace

model-forge (Rizzist's architecture playground) ran 20+ spectral/relational/
recurrent alternatives against a transformer on identical data. Every attempt
to *replace* attention lost. The winners kept a "protected dense-attention
spine" and added a bounded spectral branch beside it. The recurrent CTM — a
recurrent state beside dense attention — was the most competitive. The HAGI
branch follows that: attention stays intact, the spectral branch adds a
frequency-local increment, the mixer the nonlinear map.

## Body-scaled learning rate

Measured on real data (V32 and early V33):

```
step 0: gb=0.007   gr=0.226    (body / codebook+rest gradient norms)
```

The codebook's gradient is 20-30x the body's. On a shared AdamW the body barely
moves — most of every step's capacity is spent on the embedding table. Muon
would fix the geometry but costs 0.65s/step (measured). The cheap fix is a
separate learning rate for the 2D channel weights:

```
lr_body = learning_rate * body_lr_scale    (default 8.0)
```

`body_lr_scale=8.0` puts the body's effective LR at 2.4e-3 vs the codebook's
3e-4, closing most of the 30x gradient gap without any optimizer overhead.
The body group is identified by the same `is_channel_weight` marker the
optimizer partition already uses, so it survives ternary-on/off ablations.

## Parameters (analytic)

V33 config: H=1152, L=13, 18q/3kv x 64, ffn=3072, MoE 8e top1, spectral
32 modes every 4th layer. `count_params`:

```
total 663.2M | body 625.4M | embed 37.7M | active_body 179.4M (body 94.3%)
```

The spectral branch adds ~1.2M per selected layer (3 layers) — about 3.6M,
<1% of the body.

## Throughput

With `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`:

```
V32: 2.01s/step, 7635 tok/s
V33: 2.05s/step, 7495 tok/s   (+2% for the spectral branch)
```

The branch on every 4th layer costs ~2% of step time for the -0.15 nats init
gain and the sub-quadratic long-range capacity.
