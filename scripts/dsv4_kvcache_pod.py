"""KV-cache POD compression: lossless 512 -> 256 per token/layer.

Goal (context increase, not size reduction): keep the KV cache at the same
memory budget but store each entry in the 256-dim POD subspace (measured
top-256 = 100% on layer 0), so 2x more tokens fit -> 2M context in the same
RAM. Paired with YaRN factor 16 -> 32 for positional validity out to 2M.

P_kv[L] is the orthonormal basis [512, r_kv] (r_kv=256) from
checkpoints_dsv4/pod/. Compression is a matmul:

    z      = kv @ P_kv        # [B,S,512] -> [B,S,256]   (store z in cache)
    kv_hat = z @ P_kv.T       # [B,S,256] -> [B,S,512]   (reconstruct, lossless)

Usage (after reduce + full POD):
    from dsv4_kvcache_pod import KVCompressor, patch_yarn_factor
    kvc = KVCompressor('checkpoints_dsv4/pod')          # loads P_kv_L*.pt
    z   = kvc.compress(kv, layer_idx)                   # pre-cache
    kv  = kvc.decompress(z, layer_idx)                  # post-cache
    patch_yarn_factor(model, factor=32)                 # 1M -> 2M positional
"""
from __future__ import annotations

import glob
import os

import torch


class KVCompressor:
    def __init__(self, pod_dir: str = 'checkpoints_dsv4/pod', rank: int = 256):
        self.rank = rank
        self.bases: dict[int, torch.Tensor] = {}
        for f in sorted(glob.glob(os.path.join(pod_dir, 'P_kv_L*.pt'))):
            li = int(os.path.basename(f).split('_L')[1].split('.')[0])
            p = torch.load(f, map_location='cpu')
            if p.shape[1] != rank:
                p = p[:, :rank]
            self.bases[li] = p
        if not self.bases:
            raise FileNotFoundError(f'no P_kv_L*.pt in {pod_dir}')

    def __len__(self) -> int:
        return len(self.bases)

    def compress(self, kv: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """kv [..., 512] -> z [..., 256] (lossless on the POD subspace)."""
        P = self.bases[layer_idx].to(device=kv.device, dtype=kv.dtype)
        return kv @ P

    def decompress(self, z: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """z [..., 256] -> kv_hat [..., 512]."""
        P = self.bases[layer_idx].to(device=z.device, dtype=z.dtype)
        return z @ P.T


def install_kv_compression(cache, compressor: 'KVCompressor') -> None:
    """Patch a DynamicCache instance so the sliding-window KV is stored at
    rank r (256) instead of 512 dims, reconstructed on read.

    Only the sliding KV path (`DynamicCache.update`) is compressed; the
    CSA/HCA compressor entries (stored via `store_compression_weights`) keep
    their 512-dim format in this first pass.
    """
    import types
    orig_update = cache.update

    def patched_update(self, key_states, value_states, layer_idx, *args, **kwargs):
        zk = compressor.compress(key_states, layer_idx)
        zv = compressor.compress(value_states, layer_idx)
        k, v = orig_update(zk, zv, layer_idx, *args, **kwargs)
        return compressor.decompress(k, layer_idx), compressor.decompress(v, layer_idx)

    cache.update = types.MethodType(patched_update, cache)


def patch_yarn_factor(config, factor: int = 32) -> None:
    """Extend the compressed-attention YaRN factor 16 -> 32 (1M -> 2M tokens).

    Accepts either a model or a config. MUST be called BEFORE
    `from_pretrained` / module construction: the YaRN `inv_freq` buffers are
    baked in at init (DeepseekV4RotaryEmbedding.__init__), so post-init config
    mutation has no effect.

    The rope_parameters['compress'] block drives CSA/HCA positional encoding:
    original_max_position_embeddings=65536 * factor = context length. This is
    an inference-time extrapolation beyond the trained factor 16, so long-tail
    quality may degrade — verify logits on long prompts before shipping.
    """
    if hasattr(config, 'config'):
        config = config.config
    rp = config.rope_parameters
    if isinstance(rp, dict) and 'compress' in rp:
        rp['compress']['factor'] = factor
        rp['compress']['original_max_position_embeddings'] = 65536
        config.max_position_embeddings = 65536 * factor
    else:
        raise ValueError('rope_parameters.compress not found; cannot patch YaRN')
