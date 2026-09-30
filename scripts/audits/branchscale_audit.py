"""BranchScale audit (Lean wave-5 branchscale_minmax transfer).

Theorem: for any scale allocation with sum s_l^2 = B,
  max_l s_l^2 * v_l >= B / sum(1/v_l), equality iff uniform product.
=> a single global scale is provably suboptimal when branch variances
   v_l differ; per-layer scales must compensate measured variance.

This script measures, on a real trained checkpoint:
  v_l  = E[||branch output||^2 / dim] per block (attn + ffn branches)
  s_l  = the learned branch_scale parameters
and reports the spread of s_l^2 * v_l (constant == Lean-optimal shape;
the trainable scales should have converged toward compensation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '../../..'))
for _p in (_HERE, _REPO, os.path.join(_REPO, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hagi.config import load_config  # noqa: E402
from hagi.model.factory import build_model_for_config  # noqa: E402
from hagi.train.checkpoint import load_payload, config_from_dict  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402


def main(ckpt: str) -> int:
    configure_runtime()
    device = "cuda"
    pl = load_payload(ckpt, "cpu")
    cfg = config_from_dict(pl["config"])
    model = build_model_for_config(cfg).to(device).to(torch.bfloat16)
    model.load_state_dict({k: v.to(device) for k, v in pl["model"].items()})
    model.eval()

    # collect branch variances via BranchScale forward hooks: the module
    # sees x (branch output) before scaling; record E[x^2] per dim.
    records: dict[str, dict[str, float]] = {}
    scales: dict[str, float] = {}

    def mk_hook(name: str):
        def hook(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            v = x[0] if isinstance(x, tuple) else x
            v = v.detach().float()
            records.setdefault(name, {})
            records[name]["v"] = records[name].get("v", 0.0) + float(v.pow(2).mean())
            records[name]["n"] = records[name].get("n", 0) + 1
            # the clamp bounds live on the module
            scales[name] = float(mod.scale.detach().float().flatten().mean())
        return hook

    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "BranchScale":
            mod.register_forward_hook(mk_hook(name))

    ids = torch.randint(0, cfg.model.vocab_size, (4, 1024), device=device)
    with torch.no_grad():
        for _ in range(3):
            model(ids[:, :-1].contiguous())

    prods = {}
    print(f"{'branch':48s} {'v_l':>10s} {'s_l':>8s} {'s^2*v':>10s}")
    for name in sorted(records):
        v = records[name]["v"] / records[name]["n"]
        s = scales[name]
        prods[name] = s * s * v
        print(f"{name:48s} {v:10.4f} {s:8.4f} {prods[name]:10.4f}")
    vals = list(prods.values())
    mean = sum(vals) / len(vals)
    spread = (max(vals) - min(vals)) / mean
    print(f"\nspread (max-min)/mean of s_l^2 * v_l: {spread:.2f}")
    print("VERDICT:", "scales compensate variance (Lean-optimal shape)" if spread < 0.5
          else f"s_l^2*v_l NOT constant (spread {spread:.1f}x) -- per-layer "
               "compensation incomplete; branchscale_minmax says headroom exists")
    return 0


if __name__ == "__main__":
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/dbridge_gen2_merged_had/step-0001600.pt"
    raise SystemExit(main(ck))
