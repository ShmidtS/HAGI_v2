"""Assemble the self-contained HuggingFace release of the compressed model.

Output layout (dsv4_release/):
  layers/layer{L}.safetensors   43 files, ~2.05 GB each:
      w13t  u8  [256, 2I, T]    ternary W1|W3, 5 trits/byte (T = ceil(D/5))
      s13   f32 [256, 2I, D/GS] per-(row,group) scales g128
      b13   f32 [256, 2I]       bias1 | bias3
      w2a   u8  [256, D, I//2]  int4 packed (+8, lo-nibble = even col)
      s2    f32 [256, D, I/GS]  W2 scales
      P     f16 [D, D]          per-layer POD rotation (z = (x-mu) @ P)
      mu    f32 [D]             per-layer centering
      n_real u8  [256]          routed calibration rows per expert
      n_val  u8  [256]          honest held-out rows per expert (0 = dead)
      resid f32 [256]           honest val residual per expert
      metadata: {"hagi_recipe": "terni4-v2", "layer": L, ...}
  routers.safetensors           ROUTER_W [256,4096] f32 x43 (+bias / tid2eid)
  skeleton/                     attention + embeddings + LM head (as built)
  tokenizer/                    tokenizer.json + config from the original
  hagi_load.py                  self-contained loader (no HAGI paths)
  config.json, README.md        model card

Everything is derived ONLY from already-validated artifacts
(dsv4_bank, dsv4_reduced, dsv4_shared_only, the original snapshot).

Usage: python scripts/build_release.py [--skip-banks]
"""
import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(__file__))

REDUCED = "dsv4_reduced"
BANK = "dsv4_bank"
RELEASE = "dsv4_release"
SNAP = None  # resolved in main()
NL, D, I, GS = 43, 4096, 2048, 128


def build_layer_bank(L: int) -> bool:
    """Merge the packed bank with P/mu + per-expert provenance."""
    out_fp = os.path.join(RELEASE, "layers", f"layer{L}.safetensors")
    if os.path.exists(out_fp):
        print(f"layer {L}: release bank exists - skip", flush=True)
        return True
    t0 = time.time()
    t = load_file(os.path.join(BANK, f"layer{L}.safetensors"), device="cpu")
    P = torch.load(os.path.join(REDUCED, f"layer_{L}", "P.pt"),
                   map_location="cpu").half()
    mu = torch.load(os.path.join(REDUCED, f"layer_{L}", "mu.pt"),
                    map_location="cpu").float().reshape(-1)
    n_real = torch.zeros(256, dtype=torch.uint8)
    n_val = torch.zeros(256, dtype=torch.uint8)
    resid = torch.zeros(256, dtype=torch.float32)
    for k in range(256):
        e = torch.load(os.path.join(REDUCED, f"layer_{L}", f"expert_{k}.pt"),
                       map_location="cpu", weights_only=False)
        n_real[k] = min(int(e.get("n_real", 0)), 255)
        n_val[k] = min(int(e.get("n_val", 0)), 255)
        r = e.get("resid")
        resid[k] = float(r) if isinstance(r, (int, float)) else 0.0
        del e
    out = dict(t)
    out["P"] = P
    out["mu"] = mu
    out["n_real"] = n_real
    out["n_val"] = n_val
    out["resid"] = resid
    save_file(out, out_fp + ".tmp",
              metadata={"hagi_recipe": "terni4-v2", "layer": str(L),
                        "D": str(D), "I": str(I), "GS": str(GS),
                        "ternary": "5 trits/byte", "w2": "int4 g128"})
    os.replace(out_fp + ".tmp", out_fp)
    sz = os.path.getsize(out_fp) / 2**30
    print(f"layer {L}: release bank {sz:.2f} GB in {time.time()-t0:.0f}s", flush=True)
    return True


def build_routers() -> None:
    """Extract the per-layer router tables from the original snapshot."""
    import dsv4_experts as de
    out_fp = os.path.join(RELEASE, "routers.safetensors")
    if os.path.exists(out_fp):
        print("routers: exists - skip", flush=True)
        return
    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]
    out = {}
    for li in range(NL):
        p = f"layers.{li}.ffn.gate"
        out[f"router_w.{li}"] = de.read_tensor(snap, wm, f"{p}.weight", device="cpu").to(torch.float32)
        if li in de.HASH_LAYERS:
            out[f"router_tid.{li}"] = de.read_tensor(snap, wm, f"{p}.tid2eid", device="cpu").to(torch.int64)
        else:
            out[f"router_bias.{li}"] = de.read_tensor(snap, wm, f"{p}.bias", device="cpu").to(torch.float32)
    save_file(out, out_fp + ".tmp",
              metadata={"hash_layers": ",".join(map(str, de.HASH_LAYERS))})
    os.replace(out_fp + ".tmp", out_fp)
    print(f"routers: {len(out)} tensors -> {out_fp}", flush=True)


def copy_static() -> None:
    """Skeleton, tokenizer, original LICENSE, release generation wrapper."""
    sk_dst = os.path.join(RELEASE, "skeleton")
    if not os.path.exists(sk_dst):
        shutil.copytree("dsv4_shared_only", sk_dst)
        print("skeleton copied", flush=True)
    tok_dst = os.path.join(RELEASE, "tokenizer")
    if not os.path.exists(tok_dst):
        os.makedirs(tok_dst, exist_ok=True)
        for f in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
            src = os.path.join(SNAP, f)
            if os.path.exists(src):
                shutil.copy2(src, tok_dst)
        # encoding/ dir if present (vocab assets)
        enc_src = os.path.join(SNAP, "encoding")
        if os.path.isdir(enc_src):
            shutil.copytree(enc_src, os.path.join(tok_dst, "encoding"))
        print("tokenizer copied", flush=True)
    # release generation wrapper (delegates to the validated ttt generator;
    # must be committed alongside, see scripts/release_gen.py)
    wrapper_src = os.path.join(os.path.dirname(__file__), "release_gen.py")
    wrapper_dst = os.path.join(RELEASE, "release_gen.py")
    if os.path.exists(wrapper_src) and not os.path.exists(wrapper_dst):
        shutil.copy2(wrapper_src, wrapper_dst)
        print("release_gen.py copied", flush=True)


def main():
    global SNAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-banks", action="store_true")
    args = ap.parse_args()

    import dsv4_experts as de
    SNAP = de.default_snapshot()
    os.makedirs(os.path.join(RELEASE, "layers"), exist_ok=True)

    copy_static()
    build_routers()
    if not args.skip_banks:
        for L in range(NL):
            build_layer_bank(L)
    print("release assembly done (loader + card are written separately)", flush=True)


if __name__ == "__main__":
    main()
