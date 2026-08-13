"""Extract shared experts S_l per layer as bf16 (orthogonal-mixing center).

Fast test: since routed experts are ~orthogonal to S (alpha ~ 0), the
orthogonal-mixing super-expert is (1+alpha)*S_l ~= S_l.  Build a model whose
single FFN expert is the shared expert itself and check coherence.
"""

from __future__ import annotations

import os
import time

import torch

from dsv4_dft3_global import DEVICE, LSCALE, PROJ, load_int

CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints_dsv4", "shared_only"))


def main() -> None:
    lossless_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lossless_layers"))
    os.makedirs(CKPT_DIR, exist_ok=True)
    layers = sorted(f[:-len(".safetensors")].replace("_", ".") for f in os.listdir(lossless_dir) if f.endswith(".safetensors"))
    inv = 2.0 ** -LSCALE
    t0 = time.time()
    for li, prefix in enumerate(layers):
        f = os.path.join(lossless_dir, prefix.replace(".", "_") + ".safetensors")
        S = load_int(f, f"{prefix}.shared_experts", DEVICE)
        sd = {p: (S[p]["v"].float() * inv).to(torch.bfloat16).to("cpu") for p in PROJ}
        torch.save(sd, os.path.join(CKPT_DIR, prefix.replace(".", "_") + ".pt"))
        if (li + 1) % 10 == 0:
            print(f"{li + 1}/{len(layers)} shared extracted ({time.time() - t0:.0f}s)", flush=True)
    print(f"DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
