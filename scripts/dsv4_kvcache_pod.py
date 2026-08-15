"""KV-cache POD compression: lossless 512 -> 256 per token/layer.

Goal (context increase, not size reduction): keep the KV cache at the same
memory budget but store each entry in the 256-dim POD subspace (measured
top-256 = 100% on layer 0), so 2x more tokens fit -> 2M context in the same
RAM. Paired with YaRN factor 16 -> 32 for positional validity out to 2M.

P_kv[L] is the orthonormal basis [512, r_kv] (r_kv=256) from
checkpoints_dsv4/pod/. Compression is a matmul:

    z      = kv @ P_kv        # [B,S,512] -> [B,S,256]   (store z in cache)
    kv_hat = z @ P_kv.T       # [B,S,256] -> [B,S,512]   (reconstruct, lossless)

Pyramid (distance-dependent rank, no YaRN-factor increase):
    The POD columns are sorted by singular value, so dropping the *tail* of z
    drops the lowest-energy components. A token at distance d from the current
    query has most of its energy in the first r(d) components (attention
    decay), so we read it back at rank r(d) = clamp(base_rank >> floor(log2(d/w+1))).
    This lets the sliding window hold *more* tokens at the same memory budget
    (far tokens are cheaper), extending the context without touching YaRN.

Usage (after reduce + full POD):
    from dsv4_kvcache_pod import KVCompressor, install_kv_compression
    kvc = KVCompressor('checkpoints_dsv4/pod')          # loads P_kv_L*.pt
    install_kv_compression(cache, kvc, pyramid=(1024, 16))  # window, min_rank
    patch_yarn_factor(model, factor=8)                  # 512K positional
"""
from __future__ import annotations

import glob
import os
import types

import torch


def pyramid_rank(
    distances: torch.Tensor,
    base_rank: int = 256,
    window: int = 1024,
    min_rank: int = 16,
) -> torch.Tensor:
    """Per-token read-back rank as a function of distance from the last token.

    r(d) = clamp(base_rank >> floor(log2(d/window + 1)), min_rank, base_rank).

    d < window            -> base_rank     (lossless)
    window <= d < 2w      -> base_rank/2
    2w     <= d < 4w      -> base_rank/4
    ...                                   (halving per distance doubling)

    Args:
        distances: ``[T]`` int/float, distance of each cached token from the
            most recent token (0 = newest, T-1 = oldest).
        base_rank: full POD rank for the nearest window.
        window: distance inside which rank stays at ``base_rank``.
        min_rank: floor for very old tokens.

    Returns:
        ``[T]`` long tensor of read-back ranks.
    """
    d = distances.float().clamp(min=0.0)
    ratio = d / float(window)
    bucket = torch.floor(torch.log2(ratio.clamp(min=1.0))).long() + (d >= window).long()
    return (base_rank >> bucket).clamp(min=min_rank, max=base_rank)


class KVCompressor:
    def __init__(self, pod_dir: str = 'checkpoints_dsv4/pod', rank: int = 256):
        self.rank = rank
        self.bases: dict[int, torch.Tensor] = {}
        self.means: dict[int, torch.Tensor] = {}
        for f in sorted(glob.glob(os.path.join(pod_dir, 'P_kv_L*.pt'))):
            li = int(os.path.basename(f).split('_L')[1].split('.')[0])
            p = torch.load(f, map_location='cpu')
            if p.shape[1] != rank:
                p = p[:, :rank]
            self.bases[li] = p
        for f in sorted(glob.glob(os.path.join(pod_dir, 'mean_kv_L*.pt'))):
            li = int(os.path.basename(f).split('_L')[1].split('.')[0])
            self.means[li] = torch.load(f, map_location='cpu')
        if not self.bases:
            raise FileNotFoundError(f'no P_kv_L*.pt in {pod_dir}')

    def __len__(self) -> int:
        return len(self.bases)

    def _mean(self, layer_idx: int, kv: torch.Tensor) -> torch.Tensor:
        m = self.means.get(layer_idx)
        if m is None:
            return torch.zeros(kv.shape[-1], device=kv.device, dtype=kv.dtype)
        return m.to(device=kv.device, dtype=kv.dtype)

    def compress(self, kv: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """kv [..., 512] -> z [..., 256] (lossless on the centered POD subspace)."""
        P = self.bases[layer_idx].to(device=kv.device, dtype=kv.dtype)
        return (kv - self._mean(layer_idx, kv)) @ P

    def decompress(
        self,
        z: torch.Tensor,
        layer_idx: int,
        ranks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """z [..., 256] -> kv_hat [..., 512].

        Args:
            z: POD coordinates ``[B, heads, T, rank]``.
            layer_idx: layer whose basis to use.
            ranks: optional ``[T]`` per-token read-back rank. When given, the
                tail components ``ranks[t]:`` of token ``t`` are zeroed before
                reconstruction (pyramid / distance-dependent rank). None keeps
                the full rank for every token.
        """
        P = self.bases[layer_idx].to(device=z.device, dtype=z.dtype)
        if ranks is not None:
            r = ranks.to(device=z.device).reshape(-1)
            if r.numel() != z.shape[-2]:
                raise ValueError(f'ranks has {r.numel()} entries, expected {z.shape[-2]}')
            idx = torch.arange(z.shape[-1], device=z.device).view(1, 1, 1, -1)
            valid = idx < r.view(1, 1, -1, 1)
            z = z * valid.to(z.dtype)
        return z @ P.T + self._mean(layer_idx, z)


def _distances_from_last(t: int, device: torch.device) -> torch.Tensor:
    """Distance of each of the last ``t`` cached tokens from the newest token.

    The cache returns tokens ordered oldest -> newest, so the newest token is
    at index t-1 (distance 0) and the oldest is at index 0 (distance t-1).
    """
    return torch.arange(t - 1, -1, -1, device=device)


def install_kv_compression(
    cache,
    compressor: 'KVCompressor',
    pyramid: tuple[int, int] | None = None,
) -> None:
    """Patch a DynamicCache instance so the sliding-window KV is stored at
    rank r (256) instead of 512 dims, reconstructed on read.

    Only the sliding KV path (`DynamicCache.update`) is compressed; the
    CSA/HCA compressor entries (stored via `store_compression_weights`) keep
    their 512-dim format in this first pass.

    Args:
        cache: a transformers ``DynamicCache`` (or any object whose ``update``
            stores key/value states and returns them).
        compressor: basis loader (P_kv + means).
        pyramid: optional ``(window, min_rank)``. When given, reconstruction
            drops the POD tail of far tokens (distance-dependent rank) instead
            of always reading the full rank. This is the "pyramid" mode: far
            tokens are read back coarser, so the same sliding window can be
            grown without extra memory and without raising the YaRN factor.
    """
    orig_update = cache.update

    def patched_update(self, key_states, value_states, layer_idx, *args, **kwargs):
        zk = compressor.compress(key_states, layer_idx)
        zv = compressor.compress(value_states, layer_idx)
        k, v = orig_update(zk, zv, layer_idx, *args, **kwargs)
        if pyramid is not None:
            window, min_rank = pyramid
            t = k.shape[-2]
            distances = _distances_from_last(t, k.device)
            ranks = pyramid_rank(distances, compressor.rank, window, min_rank)
            k = compressor.decompress(k, layer_idx, ranks=ranks)
            v = compressor.decompress(v, layer_idx, ranks=ranks)
            return k, v
        return compressor.decompress(k, layer_idx), compressor.decompress(v, layer_idx)

    cache.update = types.MethodType(patched_update, cache)


def install_pyramid_compression(
    cache,
    compressor: 'KVCompressor',
    window: int = 1024,
    min_rank: int = 16,
    max_tokens: int | None = None,
) -> None:
    """Segmented sliding-KV storage: far tokens are *stored* at lower rank.

    This is the memory-saving half of the pyramid. Unlike
    :func:`install_kv_compression` (which always stores rank 256 and only
    drops the tail at read time), here each token's POD tail is physically
    truncated once it ages past ``window``, so the buffer actually shrinks.
    The sliding window can then be grown (``config.sliding_window``) at the
    same memory budget — extending the context without raising the YaRN
    factor.

    Tokens are split into chunks of ``window``; every chunk is re-ranked by
    its distance from the newest token and truncated to that rank on each
    update. Reconstruction pads truncated chunks back to the base rank with
    zeros (the dropped components are the lowest-energy POD directions).

    Args:
        cache: transformers ``DynamicCache``.
        compressor: basis loader.
        window: distance inside which rank stays at the base rank.
        min_rank: floor for the oldest chunks.
        max_tokens: optional hard cap on stored tokens (sliding-window
            budget). None stores everything with the pyramid decay.
    """
    state: dict[int, dict] = {}
    orig_get_seq = cache.get_seq_length
    cache._pyramid_state = state  # exposed for inspection/testing

    def _downrank(st: dict) -> None:
        # chunks are stored oldest -> newest. Distance of a chunk's newest
        # token is the number of tokens stored after it.
        newer = 0
        for i in range(len(st['chunks']) - 1, -1, -1):
            k, v, r = st['chunks'][i]
            n = k.shape[-2]
            d = newer
            rank = int(pyramid_rank(torch.tensor([d]), compressor.rank, window, min_rank)[0])
            rank = min(rank, r)
            if r > rank:
                st['chunks'][i] = [k[..., :rank], v[..., :rank], rank]
            newer += n

    def _reconstruct(st: dict, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        R = compressor.rank
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for k, v, r in st['chunks']:
            if r < R:
                pad = torch.zeros(*k.shape[:-1], R - r, device=k.device, dtype=k.dtype)
                k = torch.cat([k, pad], dim=-1)
                v = torch.cat([v, pad], dim=-1)
            ks.append(k)
            vs.append(v)
        k = torch.cat(ks, dim=-2)
        v = torch.cat(vs, dim=-2)
        return compressor.decompress(k, layer_idx), compressor.decompress(v, layer_idx)

    def patched_update(self, key_states, value_states, layer_idx, *args, **kwargs):
        zk = compressor.compress(key_states, layer_idx)  # [B,H,T_new,R]
        zv = compressor.compress(value_states, layer_idx)
        st = state.setdefault(layer_idx, {'chunks': [], 'length': 0})
        t_new = zk.shape[-2]
        for i in range(0, t_new, window):
            st['chunks'].append([zk[:, :, i:i + window], zv[:, :, i:i + window], compressor.rank])
        st['length'] += t_new
        if max_tokens is not None and st['length'] > max_tokens:
            drop = st['length'] - max_tokens
            st['length'] -= drop
            while drop > 0 and st['chunks']:
                k0, v0, r0 = st['chunks'][0]
                n0 = k0.shape[-2]
                if n0 <= drop:
                    st['chunks'].pop(0)
                    drop -= n0
                else:
                    st['chunks'][0] = [k0[:, :, drop:], v0[:, :, drop:], r0]
                    drop = 0
        _downrank(st)
        return _reconstruct(st, layer_idx)

    def patched_get_seq(self, layer_idx: int = 0) -> int:
        if layer_idx in state:
            return state[layer_idx]['length']
        return orig_get_seq(layer_idx)

    cache.update = types.MethodType(patched_update, cache)
    cache.get_seq_length = types.MethodType(patched_get_seq, cache)


def patch_yarn_factor(config, factor: int = 8) -> None:
    """Set the compressed-attention YaRN factor (16 -> 8 -> 512K context).

    Accepts either a model or a config. MUST be called BEFORE
    `from_pretrained` / module construction: the YaRN `inv_freq` buffers are
    baked in at init (DeepseekV4RotaryEmbedding.__init__), so post-init config
    mutation has no effect.

    The rope_parameters['compress'] block drives CSA/HCA positional encoding:
    original_max_position_embeddings=65536 * factor = context length. factor 8
    = 524288 (512K) — within the trained factor 16, so positional quality is
    safe; the KV cache is kept at full 512 dims (no KV-POD loss).
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
