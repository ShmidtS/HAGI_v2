"""Alpha probe: how the ORIGINAL model's layer transfers an input perturbation.

For layer L: hook the mlp module. Feed rows with added perturbation
delta * ||x|| * u (random direction, per-row) to a fraction of rows at the
mlp INPUT only, and compare the mlp OUTPUT delta. This measures the mlp
(Lipschitz-ish) alpha for the residual-relevant part. Additionally a whole-
layer probe: perturb the layer's INPUT (pre-attention) and compare layer
output via forward_pre_hook on the decoder layer.

alpha = ||out(x+d) - out(x)|| / ||out(x)||  divided by  (||d||/||x||).

Usage: python scripts/probe_alpha.py [--layers 1,5,20,40] [--deltas 0.05,0.13,0.3]
"""
import argparse
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import stub_import_tf  # noqa: F401
import dsv4_experts as de
from dsv4_experts import load_router
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

MODEL_DIR = "C:/HAGI_v2/dsv4_shared_only"
VOCAB = 129280


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,5,12,20,30,40")
    ap.add_argument("--deltas", default="0.05,0.13,0.30")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    LAYERS = [int(x) for x in args.layers.split(",")]
    DELTAS = [float(x) for x in args.deltas.split(",")]

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]
    load_router(snap, wm)
    torch.set_default_device("cuda")
    model = DeepseekV4ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    torch.set_default_device("cpu")
    model.eval().to(torch.bfloat16)
    model.config._experts_implementation = "eager"

    import torch.nn.functional as F
    from dsv4_experts import (HASH_LAYERS, ROUTER_BIAS, ROUTED_SCALE,
                              ROUTER_TID, ROUTER_W, TOP_K, ffn)

    shared_cache = {}
    CUR: dict[str, torch.Tensor] = {}

    def get_shared(li):
        if li not in shared_cache:
            d = de.load_shared(li)  # dict of fp32 tensors (cpu)
            shared_cache[li] = {k: v.cuda().to(torch.bfloat16) for k, v in d.items()}
        return shared_cache[li]

    REF: dict[int, torch.Tensor] = {}
    ACTIVE: dict = {"L": None, "d": 0.0, "phase": None, "out": None}
    gtor = torch.Generator(device="cuda").manual_seed(args.seed)

    def pre_hook(li):
        def h(module, args, kwargs):
            if ACTIVE["L"] != li or ACTIVE["phase"] is None:
                return args, kwargs
            d_rel = ACTIVE["d"]
            if d_rel <= 0:
                return args, kwargs
            hs = args[0]  # [B, S, H, D]: H hyper-connection streams
            shp = hs.shape
            flat = hs.reshape(-1, shp[-1])
            n = flat.shape[0]
            u = torch.randn(n, shp[-1], generator=gtor, device=flat.device)
            u = u / u.norm(dim=1, keepdim=True)
            xn = flat + d_rel * flat.norm(dim=1, keepdim=True) * u
            return (xn.view(shp).to(hs.dtype),) + tuple(args[1:]), kwargs
        return h

    def post_hook(li):
        def h(module, args, kwargs, output):
            if ACTIVE["L"] != li or ACTIVE["phase"] is None:
                return output
            # output is hidden tuple; take first tensor
            o = output[0] if isinstance(output, tuple) else output
            flat = o.reshape(-1, o.shape[-1]).float()
            ACTIVE["out"] = flat
            return output
        return h

    for li in LAYERS:
        model.model.layers[li].register_forward_pre_hook(pre_hook(li), with_kwargs=True)
        model.model.layers[li].register_forward_hook(post_hook(li), with_kwargs=True)

    torch.manual_seed(1234)
    ids = torch.randint(0, VOCAB, (1, args.tokens))

    # reference pass (no perturbation)
    ACTIVE["phase"] = None
    with torch.no_grad():
        model(ids.cuda())
    # reference outputs were not captured; run again per layer to store REF
    for li in LAYERS:
        ACTIVE["L"] = li
        ACTIVE["phase"] = "ref"
        ACTIVE["d"] = 0.0
        with torch.no_grad():
            model(ids.cuda())
        REF[li] = ACTIVE["out"].clone()
    # perturbed passes
    for li in LAYERS:
        for d in DELTAS:
            ACTIVE["L"] = li
            ACTIVE["phase"] = "pert"
            ACTIVE["d"] = d
            with torch.no_grad():
                model(ids.cuda())
            o = ACTIVE["out"]
            num = (o - REF[li]).norm().item()
            den = REF[li].norm().item()
            alpha = num / den / d
            print(f"L{li:2d}  delta={d:4.2f}: layer-out rel err {num/den*100:6.2f}%  alpha = {alpha:5.3f}", flush=True)
        ACTIVE["phase"] = None


if __name__ == "__main__":
    main()
