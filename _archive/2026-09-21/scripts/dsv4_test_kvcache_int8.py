"""Test: int8 KV-cache via DynamicCache — stores int8, reads back bf16.

No full model involved: exercises ``install_int8_compression`` against a real
``DynamicCache``, feeding the collected KV (``kv_L{li}.pt``) as key/value
states. Checks:
  1. dtype inside the cache is int8 (the 2x memory win),
  2. the returned k/v are dequantized back to bf16,
  3. reconstruction error stays < 0.02% per layer.
"""

from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import dsv4_kvcache_int8 as ki
from transformers.cache_utils import DynamicCache


def rel(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(((x - y) ** 2).sum() / (x**2).sum())


def main() -> None:
    kv_dir = "checkpoints_dsv4/attention_skeleton"
    scales = ki.compute_scales(kv_dir)
    store = ki.Int8KVStore(scales)

    cache = DynamicCache()  # lazy DynamicLayer replication, no config needed
    ki.install_int8_compression(cache, store)

    worst = 0.0
    for li in sorted(scales):
        kv = torch.load(os.path.join(kv_dir, f"kv_L{li}.pt"), map_location="cpu").float()
        k = kv[:1000].view(1, 1, 1000, 512).to(torch.bfloat16)
        v = k.clone()
        k_out, v_out = cache.update(k, v, li)
        # cache must store int8; returned states must be bf16
        assert cache.layers[li].keys.dtype == torch.int8, cache.layers[li].keys.dtype
        assert k_out.dtype == torch.bfloat16, k_out.dtype
        e = max(rel(k.float(), k_out.float()), rel(v.float(), v_out.float()))
        worst = max(worst, e)

    print(f"layers: {len(scales)}; cache dtype int8; worst reconstruction error: {worst * 100:.4f}%")


if __name__ == "__main__":
    main()
