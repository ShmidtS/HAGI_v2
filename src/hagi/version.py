"""Version and architecture identity — single source of truth.

V33 adds two measured mechanisms to V32:

1. **Recurrent spectral (Fourier) branch.** A bank of damped oscillators —
   ``S_t = A*S_{t-1} + x_t`` with ``A = r*exp(-i*omega)`` — evaluated with an
   exact parallel scan (O(T*K), 0.7ms vs 360ms for ``torch.fft`` on ROCm). It
   *complements* attention (content-addressed retrieval) with frequency-local
   structure, and carries a grokking ramp that releases high frequencies over
   the first ``ramp_steps`` steps. At init it measures ce=7.48 vs the unigram
   entropy 7.63 — a free -0.15 nats/token.
2. **Body-scaled learning rate** (``adam.body_lr_scale=8``). The codebook
   gradient norm is 20-30x the body's (gb~0.007 vs gr~0.22); a separate,
   higher LR for the 2D channel weights rebalances the two without Muon's
   Newton-Schulz cost.

The channel model is unchanged from V32: full-rank source codebook with a
causal pulse-shaping filter, ternary (BitNet b1.58) transformer channel with
QK-normalized GQA, bias-controlled MoE, recurrent spectral branch, tied
receiver with unigram prior and chunked cross-entropy.
"""

__version__ = "3.3.1"
__architecture__ = "hagi-channel-v33"
