"""Version and architecture identity -- single source of truth.

V41 (hagi-channel-v41) keeps V40's source-matched communication contract and
improves the receiver's finite-sample channel:

1. Source coder: tied codebook, causal pulse shaping, fixed unigram prior.
2. Channel: four dense ternary blocks with two local W=256 correlators and two
   full-attention relays.
3. Receiver: full-rank tied decoder. A shared K=64 bank is interleaved 50/50:
   deterministic in-batch target symbols provide hard, diverse interferers;
   source-prior draws preserve matched conditional NCE. The bank stays one GEMM.
4. Truth channel: independent-RNG exact full-vocabulary CE calibration. Proposal
   RNG can no longer change the measured rows in architecture A/B tests.
5. Full supervision and fixed-rate multimodal input remain unchanged. Text is
   self-sufficient; water filling stays off.

History reuse: early HAGI interleavers tried to transform hidden channels. V41
uses interleaving where probability theory supports it: variance reduction and
coverage in the sampled receiver, without another main-path module.

Controlled 100-step packed-corpus A/B on Radeon 8060S (seed 999, exact 8192-row
CE every 10 steps): 50% in-batch reached exact CE 7.1775 at step 90 versus
7.5412 for prior-only. Median wall-time was 1848 vs 1843 ms (+0.28%).

Checkpoint format 12. Train from scratch.
"""

__version__ = "4.1.0"
__architecture__ = "hagi-channel-v41"
