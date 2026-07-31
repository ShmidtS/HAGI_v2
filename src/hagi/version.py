"""Version and architecture identity — single source of truth.

V31 states the model as one communication path: a full-rank source codebook with
a causal pulse-shaping filter, a ternary (BitNet b1.58) transformer channel with
QK-normalized grouped-query attention, bias-controlled mixture-of-experts for
variable-rate coding, and a tied receiver with a unigram source prior and chunked
cross-entropy. Multimodality enters as per-modality source coders behind a
fixed-rate bridge.

The objective is cross-entropy plus two log-partition penalties. Nothing else
competes for gradient: load balance is a feedback controller, and the parameter
rate constraint is the quantizer itself.
"""

__version__ = "3.1.0"
__architecture__ = "hagi-channel-v31"
