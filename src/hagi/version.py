"""HAGI-2 version — single source of truth.

Codec-first scalable multimodal channel LM (V28). The model is a causal
autoregressive language model designed as a communication system: a factorized
per-modality source encoder (causal conv, no future leak), a single unified
ternary BitNet b1.58 transformer body (the genuine discrete channel —
quantization noise is the only impairment; there is no self-inflicted
AWGN/LDPC physical channel), real grouped-query attention with an incremental
KV-cache and an opt-in sliding-window local channel, a **water-filling** MoE
(SNR-gated routing + routing-entropy capacity maximization), a Q-Former
multimodal bridge with 2D/1D-RoPE, an opt-in off-path HEP predictive refiner
(EXIT-halt gated), and grounded infomax (VICReg + InfoNCE). All
information-theoretic machinery is off-path auxiliary. See docs/V28_DESIGN.md.
"""

__version__ = "2.1.0"
__architecture__ = "hagi2-codec-channel-v28"
