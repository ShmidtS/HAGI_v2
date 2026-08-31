"""Build contiguous expert banks from dsv4_reduced checkpoints.

Output: dsv4_bank/layer{L}.safetensors per layer with
  w13a  u8  [256, 2I, D/8]   (w1a rows [0:I), w3a rows [I:2I))
  s13   f32 [256, 2I]        (w1a_scale | w3a_scale)
  b13   f32 [256, 2I]        (bias1a | bias3a)
  w2a   u8  [256, D, I//2]
  s2    f16 [256, D, 16]     (w2a_scale g128, fp16 to halve size)
  P     f16 [D, D]           (per-layer rotation, for the z GEMM)
  mu    f32 [D]

These banks are what the route-indirect Triton kernels read directly on
GPU (~67 GB total for all 43 layers; DXG unified memory proven to hold it).
"""
import os
import sys
import time

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(__file__))

REDUCED = "dsv4_reduced"
BANK = "dsv4_bank"
D, I, GS = 4096, 2048, 128
NL = 43


def build_layer(L: int):
    t0 = time.time()
    red = os.path.join(REDUCED, f"layer_{L}")
    n = sum(1 for k in range(256) if os.path.exists(os.path.join(red, f"expert_{k}.pt")))
    if n != 256:
        print(f"layer {L}: only {n}/256 experts present - SKIP", flush=True)
        return False
    w13a = torch.empty(256, 2 * I, D // 8, dtype=torch.uint8)
    s13 = torch.empty(256, 2 * I, dtype=torch.float32)
    b13 = torch.empty(256, 2 * I, dtype=torch.float32)
    w2a = torch.empty(256, D, I // 2, dtype=torch.uint8)
    s2 = torch.empty(256, D, I // GS, dtype=torch.float32)
    for k in range(256):
        e = torch.load(os.path.join(red, f"expert_{k}.pt"), map_location="cpu", weights_only=False)
        if e.get("mode") != "i1i4":
            print(f"layer {L} k{k}: mode={e.get('mode')!r} != i1i4 - SKIP layer", flush=True)
            return False
        w13a[k, :I] = e["w1a"]
        w13a[k, I:] = e["w3a"]
        s13[k, :I] = e["w1a_scale"]
        s13[k, I:] = e["w3a_scale"]
        b13[k, :I] = e["bias1a"]
        b13[k, I:] = e["bias3a"]
        w2a[k] = e["w2a"]
        s2[k] = e["w2a_scale"]
    P = torch.load(os.path.join(red, "P.pt"), map_location="cpu").half()
    mu = torch.load(os.path.join(red, "mu.pt"), map_location="cpu").float().reshape(-1)
    out = {
        "w13a": w13a,
        "s13": s13,
        "b13": b13,
        "w2a": w2a,
        "s2": s2.half(),
        "P": P,
        "mu": mu,
    }
    os.makedirs(BANK, exist_ok=True)
    fp = os.path.join(BANK, f"layer{L}.safetensors")
    save_file(out, fp)
    sz = os.path.getsize(fp) / 2**30
    print(f"layer {L}: bank written {sz:.2f} GB in {time.time()-t0:.0f}s", flush=True)
    return True


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else NL
    for L in range(lo, hi):
        if os.path.exists(os.path.join(BANK, f"layer{L}.safetensors")):
            print(f"layer {L}: bank exists - skip", flush=True)
            continue
        build_layer(L)


if __name__ == "__main__":
    main()
