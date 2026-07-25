"""HAGI-2 version — single source of truth.

Codec-first scalable multimodal channel LM. The model is a causal
autoregressive language model designed as a communication system: a
factorized per-modality source encoder (causal conv, no future leak), a
single unified ternary BitNet b1.58 transformer body (the genuine discrete
channel — quantization noise is the only impairment; there is no
self-inflicted AWGN/LDPC physical channel), real grouped-query attention
with an incremental KV-cache, an opt-in entropy-aware MoE (water-filling
capacity allocation), a Q-Former multimodal bridge, and grounded infomax
(VICReg + InfoNCE). All information-theoretic machinery is off-path
auxiliary. See docs/V27_DESIGN.md.
"""

__version__ = "2.0.0"
__architecture__ = "hagi2-codec-channel"
