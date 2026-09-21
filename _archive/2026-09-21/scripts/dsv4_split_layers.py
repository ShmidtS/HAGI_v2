"""Lossless per-layer split of DeepSeek-V4 MoE layers.

Copies every ffn tensor of a layer (256 routed experts in original packed
fp4/int8 + scales, shared expert in fp8, gate) into one per-layer safetensors
file BIT-FOR-BIT (no dequant, no requant, no merge). This is the lossless
intermediate artifact the recursive merge should operate on.

Verification: after saving, the file is loaded back and every tensor is
compared with torch.equal against the original.

Output: lossless_layers/{layer}.safetensors  (~2.3 GB each, 46 files ~106 GB)
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import dsv4_experts as de

from safetensors.torch import load_file, save_file

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lossless_layers"))


def split_layer(snap: str, wm: dict, lp: str, verify: bool = True) -> tuple[int, int]:
    """Copy all ffn tensors of one layer into a lossless safetensors file."""
    keys = [k for k in wm if k.startswith(lp + ".")]
    payload = {}
    for k in keys:
        payload[k] = de.read_tensor(snap, wm, k)
    out_path = os.path.join(OUT, lp.replace(".", "_") + ".safetensors")
    save_file(payload, out_path)
    nbytes = sum(t.numel() * t.element_size() for t in payload.values())

    if verify:
        loaded = load_file(out_path)
        for k in keys:
            if not torch.equal(payload[k], loaded[k]):
                raise SystemExit(f"LOSSY! {k} mismatch after roundtrip")
        del loaded

    del payload
    return len(keys), nbytes


def main() -> None:
    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]
    prefs = de.iter_layer_prefixes(wm)
    os.makedirs(OUT, exist_ok=True)
    print(f"splitting {len(prefs)} layers -> {OUT}", flush=True)
    total = 0
    for lp in prefs:
        t0 = time.time()
        n, b = split_layer(snap, wm, lp)
        total += b
        print(f"{lp}: {n} keys, {b/1e9:.2f} GB, verified lossless ({time.time()-t0:.1f}s)", flush=True)
    print(f"ALL DONE: {total/1e9:.2f} GB total", flush=True)


if __name__ == "__main__":
    main()
