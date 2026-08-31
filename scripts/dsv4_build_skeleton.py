"""Rebuild the skeleton model dir (dsv4_shared_only): all non-FFN-expert
tensors from the snapshot, fp8 pairs decoded to bf16 (scales dropped),
plus a zero-stub expert 0 per layer (the MoE hook replaces routed experts).

Router (gate) weights are skipped (loaded separately by load_router).
Config: n_routed_experts=1, num_experts_per_tok=1, no quantization_config.
"""
import json
import os
import re
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(__file__))
import dsv4_experts as de  # noqa: E402

snap = de.default_snapshot()
idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
wm = idx["weight_map"]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dsv4_shared_only")
os.makedirs(OUT, exist_ok=True)

keep = {}
for k, f in wm.items():
    if ".ffn.experts." in k or ".ffn.gate." in k:
        continue
    keep.setdefault(f, []).append(k)

# fp8 pairs can span files: preload ALL e8m0 scale tensors first
print("preloading fp8 scales...", flush=True)
scales = {}
for k, f in wm.items():
    if k.endswith(".scale") and ".ffn.experts." not in k and ".ffn.gate." not in k:
        with de.safe_open(os.path.join(snap, f), framework="pt") as fh:
            scales[k] = fh.get_tensor(k).clone()

print("files:", len(keep), flush=True)

out_idx = {}
files = sorted(keep.items())
for i, (f, keys) in enumerate(files):
    tensors = {}
    with de.safe_open(os.path.join(snap, f), framework="pt") as fh:
        for k in keys:
            t = fh.get_tensor(k).clone()
            if t.dtype == torch.float8_e4m3fn:
                sk = k[: -len(".weight")] + ".scale"
                if sk in scales:
                    t = de.dequant_fp8(t, scales[sk]).to(torch.bfloat16)
            tensors[k] = t
    oname = f"skeleton_{i:05d}.safetensors"
    for k in keys:
        if k in tensors:
            out_idx[k] = oname
    save_file(tensors, os.path.join(OUT, oname))
    if (i + 1) % 12 == 0:
        print(f"  {i + 1}/{len(files)}", flush=True)

with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as fh:
    json.dump({"metadata": idx.get("metadata", {}), "weight_map": out_idx}, fh)
cfg = json.load(open(os.path.join(snap, "config.json")))
cfg["n_routed_experts"] = 1
cfg["num_experts_per_tok"] = 1
cfg.pop("quantization_config", None)
with open(os.path.join(OUT, "config.json"), "w") as fh:
    json.dump(cfg, fh, indent=2)
print("DONE:", OUT, flush=True)
