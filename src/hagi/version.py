"""Version and architecture identity — single source of truth.

V39 (hagi-channel-v39) — three stacked rate reductions on the ternary channel:

1. **L=4 body** — less activation traffic; full_every=2 keeps two global relays.
2. **Punctured CE p=0.5** — erasure on supervision (V38); body still full T.
3. **Sampled softmax K=64** — local partition over {target}∪K negatives
   (Jean et al.); train-only; generation uses full logits.

Measured ~41k body tok/s on 8060S (~2.3 s/step) vs V38 ~24k / V35 ~16k.

Channel SSOT: ternary weight rate is the only channel noise. CE (full or
sampled) is the coding cost; puncture only thins its measurement.

Checkpoint format 10. Train from scratch.
"""

__version__ = "3.9.0"
__architecture__ = "hagi-channel-v39"
