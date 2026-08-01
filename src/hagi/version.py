"""Version and architecture identity — single source of truth.

V32 scales V31 to a healthy regime: the vocabulary is compacted 262144 -> 131072
(99.9% of token mass retained), moving body share from 29% to 82% of total
parameters — the V30/V31 pathology of a body too small relative to the codebook
is gone. MoE (E=8, top_k=2) is enabled as the scaling lever, the sliding window
grows to the sequence length, and the training budget rises 4x to ~1.77B tokens.

The channel model is unchanged from V31: full-rank source codebook with a causal
pulse-shaping filter, ternary (BitNet b1.58) transformer channel with QK-
normalized GQA, bias-controlled MoE for variable-rate coding, tied receiver with
unigram prior and chunked cross-entropy.
"""

__version__ = "3.2.0"
__architecture__ = "hagi-channel-v32"
