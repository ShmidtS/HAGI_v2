"""HAGI version — single source of truth.

V21: Recovery & Proper Integration architecture.
Restores 16 deleted modules (kalman, uncertainty, cqi, exit_chart,
turbo_decoder, water_filling, clifford, lorentz, msa, foxp2,
contrastive, multimodal_input, multimodal_masking, moe, kv_cache,
speculative) and integrates them by their correct theoretical purpose.

Breaking changes vs V20:
  - LDPCDecoder (Richardson iteration) -> TurboDecoder (real BP)
  - Hard gate at inference -> Kalman-filtered soft gate (always)
  - External awgn_sigma -> CQI-estimated adaptive sigma
  - Fixed bottleneck_ratio -> WaterFilling adaptive grade allocation
  - Dead AlgebraConfig/MSAConfig/GP2DConfig -> active content
  - O(T^2) inference -> O(T) KV cache
  - No FOXP2 -> per-layer plasticity gates (training)
"""

__version__ = "21.0.0"
__architecture__ = "V21"
__config_version__ = 4
__checkpoint_compat__ = "none"
