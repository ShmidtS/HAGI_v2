"""Build an expert-major store from the release layer banks:
dsv4_release/experts/L{layer}/E{expert}.safetensors with keys:
  w13 u8 [2I, T3], s13 f16 [2I, NG], b13 f32 [2I], w2a u8 [D, I/2], s2 f16 [D, NG]
One sequential pass over the banks; enables 12MB random reads per expert
instead of 2GB tensor reads. Peaked GPU stays tiny.
"""
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file

REL = "dsv4_release"
D, I, GS = 4096, 2048, 128
NL = 43


def main():
    out_root = os.path.join(REL, "experts")
    t0 = time.time()
    for L in range(NL):
        out_dir = os.path.join(out_root, f"L{L}")
        done = all(os.path.exists(os.path.join(out_dir, f"E{k}.safetensors"))
                   for k in range(0, 256, 64))  # coarse check
        if done and os.path.exists(os.path.join(out_dir, "E255.safetensors")):
            print(f"L{L}: exists, skip", flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)
        t = load_file(os.path.join(REL, "layers", f"layer{L}.safetensors"), device="cpu")
        w13t, s13, b13 = t["w13t"], t["s13"].half(), t["b13"]
        w2a, s2 = t["w2a"], t["s2"].half()
        for k in range(256):
            save_file({
                "w13": w13t[k].contiguous(),
                "s13": s13[k].contiguous(),
                "b13": b13[k].contiguous(),
                "w2a": w2a[k].contiguous(),
                "s2": s2[k].contiguous(),
            }, os.path.join(out_dir, f"E{k}.safetensors"))
        print(f"L{L}: done ({time.time()-t0:.0f}s elapsed)", flush=True)
    print("expert store complete", flush=True)


if __name__ == "__main__":
    main()
