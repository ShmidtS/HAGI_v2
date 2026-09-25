"""int8 KV-cache compression: store the full 512-dim KV as int8 (2x vs bf16).

Replaces the POD (low-rank 512 -> 256) approach. POD loses energy because
the KV activations are near full rank; int8 quantization of the *full*
512-dim KV loses almost nothing (measured per-channel error 0.003-0.005%
across layers 0/10/21/42).

RoPE note. The cache stores K post-RoPE (rotated) and V pre-RoPE. RoPE
rotates dim pairs (2i, 2i+1), so a per-dim scale computed on the pre-RoPE
KV under-estimates the post-RoPE K. We use the rotation-safe upper bound

    scale[d] = sqrt(max_{2i}^2 + max_{2i+1}^2) / 127,   i = d // 2

which is position-independent and only ~3-4% above the true post-RoPE max
(verified: post-RoPE per-dim max is flat from position 0 to 500K). The same
scale is used for K and V — for V it is ~3% conservative, negligible.

Usage:
    from dsv4_kvcache_int8 import compute_scales, Int8KVStore, install_int8_compression
    scales = compute_scales('checkpoints_dsv4/attention_skeleton')  # -> dict[li] = [512]
    store = Int8KVStore(scales)
    install_int8_compression(cache, store)
"""

from __future__ import annotations

import os
import types

import torch

QMAX = 127  # symmetric int8 range [-127, 127] (avoid the asymmetric -128)


def compute_scales(kv_dir: str, out_path: str | None = None) -> dict[int, torch.Tensor]:
    """Per-channel int8 scale per layer from the collected KV (kv_norm output).

    Reads ``kv_L{li}.pt`` (``[N, 512]`` pre-RoPE, already RMS-normed) and
    returns ``scale[li]`` (``[512]`` fp32) using the RoPE-safe pair bound.
    """
    scales: dict[int, torch.Tensor] = {}
    for li in range(43):
        path = os.path.join(kv_dir, f"kv_L{li}.pt")
        if not os.path.exists(path):
            continue
        kv = torch.load(path, map_location="cpu", weights_only=True).float()
        mx = kv.abs().max(dim=0).values.view(-1, 2)  # [256, 2] = (2i, 2i+1) pairs
        bound = torch.sqrt(mx[:, 0] ** 2 + mx[:, 1] ** 2).repeat_interleave(2)  # [512]
        scales[li] = (bound / QMAX).clamp(min=1e-6)
    if not scales:
        raise FileNotFoundError(f"no kv_L*.pt found in {kv_dir}")
    if out_path is not None:
        torch.save(scales, out_path)
    return scales


class Int8KVStore:
    """Quantize KV to int8 on write, dequantize to bf16 on read."""

    def __init__(self, scales: dict[int, torch.Tensor]):
        self.scales = scales

    def __len__(self) -> int:
        return len(self.scales)

    def compress(self, kv: torch.Tensor, layer_idx: int) -> torch.Tensor:
        s = self.scales[layer_idx].to(device=kv.device)
        return (kv.float() / s).round().clamp(-QMAX, QMAX).to(torch.int8)

    def decompress(self, q: torch.Tensor, layer_idx: int) -> torch.Tensor:
        s = self.scales[layer_idx].to(device=q.device, dtype=torch.float32)
        return (q.float() * s).to(torch.bfloat16)


def install_int8_compression(cache, store: Int8KVStore) -> None:
    """Patch a ``DynamicCache`` so KV is stored as int8 and dequantized on read.

    The int8 tensors stay inside the cache (``DynamicLayer.lazy_initialization``
    takes its dtype from ``key_states``, and ``torch.cat`` preserves it); the
    patched ``update`` returns dequantized bf16 K/V to attention.
    """
    orig_update = cache.update

    def patched_update(self, key_states, value_states, layer_idx, *args, **kwargs):
        qk = store.compress(key_states, layer_idx)
        qv = store.compress(value_states, layer_idx)
        k, v = orig_update(qk, qv, layer_idx, *args, **kwargs)
        return store.decompress(k, layer_idx), store.decompress(v, layer_idx)

    cache.update = types.MethodType(patched_update, cache)


def pyramid_rank(
    distances: torch.Tensor,
    base_rank: int = 512,
    window: int = 1024,
    min_rank: int = 32,
) -> torch.Tensor:
    """Per-token read-back rank vs distance from the last token.

    ``r(d) = clamp(base_rank >> floor(log2(d/window + 1)), min_rank, base_rank)``.
    Halving per distance doubling. Ported from the KV-POD pyramid; here
    ``base_rank`` is the full int8 channel dim (512), so "rank" = channels kept.
    """
    d = distances.float().clamp(min=0.0)
    ratio = d / float(window)
    bucket = torch.floor(torch.log2(ratio.clamp(min=1.0))).long() + (d >= window).long()
    return (base_rank >> bucket).clamp(min=min_rank, max=base_rank)


def install_int8_pyramid_compression(
    cache,
    store: Int8KVStore,
    window: int = 1024,
    min_rank: int = 32,
    max_tokens: int | None = None,
) -> None:
    """Segmented int8 KV storage: far tokens are *stored* with fewer channels.

    Port of the KV-POD ``install_pyramid_compression`` to the int8 cache.
    Tokens are split into chunks of ``window``; each chunk is re-ranked by its
    distance from the newest token and truncated to ``pyramid_rank`` channels
    (halving per distance doubling), so far tokens physically occupy fewer
    bytes -> a larger sliding window at the same memory budget.

    Caveat vs POD: POD truncated in a singular-value-ordered basis (tail =
    lowest energy). The int8 cache stores the raw 512-dim KV, so the channel
    order has no importance ordering (RoPE-frequency for K, arbitrary for V).
    This is a mechanical port: it saves memory but the truncated channels are
    NOT guaranteed to be the least-important. If quality degrades, the
    token-density pyramid (evict far tokens at stride 2^bucket) is the
    principled int8 alternative.

    Args:
        cache: transformers ``DynamicCache``.
        store: int8 scale holder (compress/decompress).
        window: distance inside which rank stays at the full 512 channels.
        min_rank: floor for the oldest chunks.
        max_tokens: optional hard cap on stored tokens (None = pyramid decay
            only, no hard eviction).
    """
    base = 512  # full int8 channel dim
    state: dict[int, dict] = {}
    cache._pyramid_state = state  # exposed for inspection/testing
    orig_get_seq = cache.get_seq_length

    def _downrank(st: dict) -> None:
        # chunks are stored oldest -> newest. Distance of a chunk's newest
        # token is the number of tokens stored after it.
        newer = 0
        for i in range(len(st["chunks"]) - 1, -1, -1):
            k, v, r = st["chunks"][i]
            n = k.shape[-2]
            d = newer
            rank = int(pyramid_rank(torch.tensor([d], device=k.device), base, window, min_rank)[0])
            rank = min(rank, r)
            if r > rank:
                st["chunks"][i] = [k[..., :rank], v[..., :rank], rank]
            newer += n

    def _reconstruct(st: dict, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for k, v, r in st["chunks"]:
            if r < base:
                pad = torch.zeros(*k.shape[:-1], base - r, device=k.device, dtype=k.dtype)
                k = torch.cat([k, pad], dim=-1)
                v = torch.cat([v, pad], dim=-1)
            ks.append(k)
            vs.append(v)
        k = torch.cat(ks, dim=-2)
        v = torch.cat(vs, dim=-2)
        return store.decompress(k, layer_idx), store.decompress(v, layer_idx)

    def patched_update(self, key_states, value_states, layer_idx, *args, **kwargs):
        qk = store.compress(key_states, layer_idx)  # int8 [B,H,T,512]
        qv = store.compress(value_states, layer_idx)
        st = state.setdefault(layer_idx, {"chunks": [], "length": 0})
        t_new = qk.shape[-2]
        for i in range(0, t_new, window):
            st["chunks"].append([qk[:, :, i : i + window], qv[:, :, i : i + window], base])
        st["length"] += t_new
        if max_tokens is not None and st["length"] > max_tokens:
            drop = st["length"] - max_tokens
            st["length"] -= drop
            while drop > 0 and st["chunks"]:
                k0, v0, r0 = st["chunks"][0]
                n0 = k0.shape[-2]
                if n0 <= drop:
                    st["chunks"].pop(0)
                    drop -= n0
                else:
                    st["chunks"][0] = [k0[:, :, drop:], v0[:, :, drop:], r0]
                    drop = 0
        _downrank(st)
        return _reconstruct(st, layer_idx)

    def patched_get_seq(self, layer_idx: int = 0) -> int:
        if layer_idx in state:
            return state[layer_idx]["length"]
        return orig_get_seq(layer_idx)

    cache.update = types.MethodType(patched_update, cache)
    cache.get_seq_length = types.MethodType(patched_get_seq, cache)
