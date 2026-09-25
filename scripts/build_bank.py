"""Build contiguous terni4 expert banks from dsv4_reduced checkpoints.

Output: dsv4_bank/layer{L}.safetensors per layer with
  w13t  u8  [256, 2I, T]     ternary W1|W3, 5 trits/byte (T = ceil(D/5) = 820)
                             w1 rows [0:I), w3 rows [I:2I)
  s13   f32 [256, 2I, D/GS]  per-(row,group) scales g128 (w1 | w3)
  b13   f32 [256, 2I]        bias1 | bias3
  w2a   u8  [256, D, I//2]   int4 packed (+8, lo-nibble = even col)
  s2    f32 [256, D, I/128]  per-(row,group) W2 scales

P/mu stay in dsv4_reduced/layer_L/ (load_pod reads them there).
One 2.1 GB sequential file per layer instead of 256 scattered expert_*.pt
(11008 files total) - bulk loads for gate training and generation.

Usage: python scripts/build_bank.py [lo hi]
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
T = (D + 4) // 5          # 820 bytes per ternary row
NL = 43


def build_layer(L: int):
    t0 = time.time()
    red = os.path.join(REDUCED, f"layer_{L}")
    n = sum(1 for k in range(256) if os.path.exists(os.path.join(red, f"expert_{k}.pt")))
    if n != 256:
        print(f"layer {L}: only {n}/256 experts present - SKIP", flush=True)
        return False
    w13t = torch.empty(256, 2 * I, T, dtype=torch.uint8)
    s13 = torch.empty(256, 2 * I, D // GS, dtype=torch.float32)
    b13 = torch.empty(256, 2 * I, dtype=torch.float32)
    w2a = torch.empty(256, D, I // 2, dtype=torch.uint8)
    s2 = torch.empty(256, D, I // GS, dtype=torch.float32)
    for k in range(256):
        e = torch.load(os.path.join(red, f"expert_{k}.pt"),
                       map_location="cpu", weights_only=False)
        if e.get("mode") != "terni4":
            print(f"layer {L} k{k}: mode={e.get('mode')!r} != terni4 - SKIP layer", flush=True)
            return False
        w13t[k, :I] = e["w1a"]
        w13t[k, I:] = e["w3a"]
        s13[k, :I] = e["w1a_scale"]
        s13[k, I:] = e["w3a_scale"]
        b13[k, :I] = e["bias1a"]
        b13[k, I:] = e["bias3a"]
        w2a[k] = e["w2a"]
        s2[k] = e["w2a_scale"]
        del e
    out = {"w13t": w13t, "s13": s13, "b13": b13, "w2a": w2a, "s2": s2}
    os.makedirs(BANK, exist_ok=True)
    fp = os.path.join(BANK, f"layer{L}.safetensors")
    save_file(out, fp + ".tmp")
    os.replace(fp + ".tmp", fp)  # atomic: no half-written banks on crash
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
