"""HAGI version — single source of truth.

Ternary RD-channel causal language model. Reframed as a communication
channel: a factorized source encoder (causal conv, no future leak), a ternary
BitNet b1.58 transformer body (the genuine discrete channel — quantization
noise is the only impairment; there is no self-inflicted AWGN/LDPC physical
channel), an auxiliary variational information bottleneck (KL rate, kept off
the main LM path), and an optional predictive decoder + multimodal fusion.
"""

__version__ = "25.0.0"
__architecture__ = "ternary-rd-channel"
