# HAGI V39 Architecture — Sampled Softmax + Thinner Channel

V39 stacks three independent rate reductions. None of them change the channel
noise model (ternary weights remain the only channel noise).

## Stack

| Lever | Theory | What changes | What does **not** |
|---|---|---|---|
| **L=4** | shorter cascade / less activation traffic | body depth, residual scale | weight rate |
| **p=0.5 puncture** | erasure on *supervision* | scored rows N | body path (still full T) |
| **K=64 sampled CE** | local partition / MC MI surrogate | head cost O(N·K·H) | generation (full logits) |

```
source ──► ternary body (L=4, full W every 2) ──► hidden[B,T,H]
                                                    │
                              body grads: every t   │
                                                    ▼
                                          puncture Π_p (p=0.5)
                                                    │
                                                    ▼
                                    sampled softmax over {y}∪K negs
                                    (train only; gen = full logits)
```

## Why each lever

**Depth.** On 8060S body-only f+b: L6≈882 ms, L4≈591 ms, L3≈463 ms. Beyond L=4
returns diminish vs capacity loss. `full_every=2` keeps two global relays.

**Puncture (V38).** Head ∝ N. Bernoulli p keeps E[CE]=CE_full.

**Sampled softmax (Jean et al. 2015).** Full CE is exact decoding metric over
alphabet V. Sampled CE is a Monte-Carlo local decoder over K+1 classes. With
uniform proposal the −log q term is constant and cancels; unigram prior already
in logits reweights candidates. Measured:

| head path | N=15360 | N=30720 |
|---|---:|---:|
| full chunked CE | 351 ms | 684 ms |
| K=64 | 123 ms | 243 ms |
| K=128 | 242 ms | 479 ms |
| K≥256 | slower than full on this part (gather) |

## End-to-end (B=30×3 T=1024)

| Config | ms/step | body tok/s |
|---|---:|---:|
| V35 | ~5770 | ~16k |
| V37 | ~4840 | ~19k |
| V38 L6 p0.5 | ~3800 | ~24k |
| **V39 L4 p0.5 K64** | **~2260** | **~41k** |
| L4 p0.25 K64 | ~1960 | ~47k |
| L3 p0.5 K64 | ~1760 | ~52k |

Ship: capacity-preserving L4 + unbiased p=0.5 + K=64.

## Knobs

```yaml
model:
  num_layers: 4
  sliding: {window: 256, full_every: 2}
  head:
    sampled_softmax_k: 64   # 0 = full CE
train:
  ce_keep_rate: 0.5
  ce_keep_mode: bernoulli
```

Logged `ce` under K>0 is the *local* partition CE (not comparable 1:1 to full
CE). For eval use full logits / `sampled_softmax_k: 0`.

## Train

```bash
python scripts/train.py --config configs/v39_1b.yaml --dry-run
python scripts/train.py --config configs/v39_1b.yaml
```

From scratch. Checkpoint format 10.
