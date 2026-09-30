"""Round 53: DepthBench-style layer utilization on the record model.

DepthBench (review round-8) prescription: layer count L is not effective
depth D_eff; each layer's contribution must be MEASURED. Operationalizes
the formalized dead_layer_stop (V_l = dCE_prune(l)/T_l): skip each block
(identity instead of the block transform) and measure the gate CE delta.

Output per layer l: u_l = dCE_prune(l) / max_j dCE_prune(j), the
utilization that feeds I_eff = I_width + lambda*I_depth and the
architecture controller's argmax dI_eff*eta_transport / dT_wall.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, os.path.join(_REPO, "scripts", "lora"), _REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from pathlib import Path

from lora_c8_deepen import build_stacked_prior  # pyright: ignore[reportMissingImports]
from hagi.train.loop import configure_runtime

CORP = ["edu","python_instruct","wikipedia_en","wikipedia_ru","oscar_ru","openwebmath","tinystories","smoltalk"]
W = [0.3571,0.2232,0.0893,0.0714,0.0625,0.0893,0.0536,0.0536]

class _Identity(torch.nn.Module):
    def forward(self, x, positions=None, mask=None):
        return x

def gate_ce(m, batches) -> float:
    tot = 0.0
    for (x, y), w in zip(batches, W):
        with torch.no_grad():
            o = m(x, y)
            flat = o.hidden.reshape(-1, o.hidden.shape[-1])
            tot += w * float(m.head.exact_loss(flat, y.reshape(-1)))
    return tot

def main() -> None:
    configure_runtime()
    m = build_stacked_prior().eval()
    pl = torch.load("checkpoints/dbridge_gen2_lora_c8_d2/step-0001600.pt", map_location="cpu", weights_only=False)
    m.load_state_dict({k: v.to("cuda") for k, v in pl["model"].items()}, strict=True)

    batches = []
    for c in CORP:
        p = Path(f"data/{c}.compact.bin"); total = p.stat().st_size // 4
        with p.open("rb") as fh:
            fh.seek((total - 2_000_000) * 4)
            T = np.frombuffer(fh.read(300_000 * 4), dtype=np.uint32).astype(np.int64)
        ids = torch.from_numpy(T[:2048]).reshape(2, 1024)
        batches.append((ids[:, :-1].cuda(), ids[:, 1:].cuda()))

    base = gate_ce(m, batches)
    print(f"baseline (all layers): {base:.4f}", flush=True)

    n = len(m.blocks)
    deltas = []
    originals = []
    for l in range(n):
        originals.append(m.blocks[l])
        m.blocks[l] = _Identity().to("cuda").to(torch.bfloat16)
        ce = gate_ce(m, batches)
        m.blocks[l] = originals[l]
        d = ce - base
        deltas.append(d)
        print(f"prune layer {l}: CE {ce:.4f}  dCE {d:+.4f}", flush=True)

    mx = max(deltas)
    print("\nutilization u_l = dCE/max:", flush=True)
    for l, d in enumerate(deltas):
        u = d / mx if mx > 0 else 0.0
        print(f"  layer {l}: dCE {d:+.4f}  u {u:.3f}", flush=True)
    d_eff = sum(max(d, 0.0) / mx for d in deltas) if mx > 0 else 0.0
    print(f"\nD_eff (sum u_l) = {d_eff:.2f} of L={n} layers", flush=True)

if __name__ == "__main__":
    main()
