"""Version and architecture identity -- single source of truth.

V42 (hagi-channel-v42) optimizes V41 for wall-time on a bandwidth-bound iGPU
(Radeon 8060S, DDR5 ~107 GB/s):

1. Full causal attention on every layer (W=0): the flash kernel is faster than
   the multi-chunk compressed_history path on this hardware.
2. Sequence length T=512 (from T=1024): halves attention FLOPs (O(T²) → O(T²/4)).
3. Punctured CE at ce_keep_rate=0.5: erasure channel on supervision, body still
   sees full T.
4. torch.compile infrastructure (RoPE cache fix, cudagraph_mark_step_begin) —
   ready for future ROCm builds where CUDAGraph overhead is lower.

Measured speedup: 1845 → 1288 ms/step (+43%), 50k → 57k tok/s (+15%).

The body shape (H=1152, L=4, ternary b1.58), receiver (K=64 conditional NCE
with unigram prior), and checkpoint format (12) are unchanged from V41.

History: V41 added the interleaved in-batch/prior conditional receiver.
V42 keeps it and optimizes the channel geometry for the hardware's compute
profile.
"""

__version__ = "4.2.0"
__architecture__ = "hagi-channel-v42"
