"""Measure how much upstream compression perturbs layer-L activations.

Runs the model twice (compressed 0..L-1 vs original) on the same token
stream and reports ||x_comp - x_orig|| / ||x_orig|| at layer L input,
plus per-row stats. If the perturbation is tiny, e2e calibration is
pointless; if large, it is the main lever.
"""
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
L_HOOK = 5


def main():
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

    from dsv4_generate_ttt import get_int4x, int4x_forward  # compressed decode

    CAPTURE: dict[int, torch.Tensor] = {}
    CUR: dict[str, torch.Tensor] = {}

    import torch.nn.functional as F
    from dsv4_experts import (HASH_LAYERS, ROUTER_BIAS, ROUTED_SCALE,
                              ROUTER_TID, ROUTER_W, SWIGLU_LIMIT, TOP_K, ffn, load_selected_experts)

    shared_cache = {}

    def get_shared(li):
        if li not in shared_cache:
            from safetensors import safe_open
            fp = os.path.join(de.LOSSLESS, f"layers_{li}_ffn.safetensors")
            base = f"layers.{li}.ffn.shared_experts"
            d = {}
            with safe_open(fp, framework="pt", device="cuda") as f:
                for proj in ("w1", "w2", "w3"):
                    w = f.get_tensor(f"{base}.{proj}.weight")
                    s = f.get_tensor(f"{base}.{proj}.scale")
                    d[proj] = de._decode(base, w, s).to(torch.bfloat16)
            shared_cache[li] = d
        return shared_cache[li]

    state = {"compressed": False}

    def make_hook(li):
        def hook(module, args_, kwargs, output):
            x = args_[0]
            B, S, D = x.shape
            flat = x.reshape(-1, D).float()
            if li == L_HOOK:
                CAPTURE[0 if state["compressed"] else 1] = flat.detach().cpu()
                return None  # keep original FFN for layer >= L_HOOK? NO - must run it
            if not state["compressed"]:
                return None  # original path (module forward already ran)
            logits = flat @ ROUTER_W[li].T
            scores = F.softplus(logits).sqrt()
            if li in HASH_LAYERS:
                indices = ROUTER_TID[li][CUR["ids"].reshape(-1)]
            else:
                indices = torch.topk(scores + ROUTER_BIAS[li], TOP_K, dim=-1).indices
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * ROUTED_SCALE
            sw = get_shared(li)
            flatb = flat.to(torch.bfloat16)
            out = ffn(flatb, sw["w1"], sw["w2"], sw["w3"]).float()
            sel = indices.unique().tolist()
            experts = load_selected_experts(li, sel)
            for k in sel:
                d = get_int4x(li, k)
                m_any = (indices == k).any(dim=1)
                if d is not None:
                    h = int4x_forward(d, flat[m_any])
                    out_k = (h.to(torch.bfloat16) @ d["w2b"].T).float()
                    pos = torch.cumsum(m_any.long(), 0) - 1
                    for kk in range(TOP_K):
                        m = indices[:, kk] == k
                        if m.any():
                            out[m] += weights[m, kk, None] * out_k[pos[m]]
                    del h, out_k
                else:
                    w1, w2, w3 = experts[k]
                    ek = ffn(flatb, w1, w2, w3).float()
                    for kk in range(TOP_K):
                        m = indices[:, kk] == k
                        if m.any():
                            out[m] += weights[m, kk, None] * ek[m]
                    del ek
            del experts
            return out.to(x.dtype).reshape(B, S, D)
        return hook

    handles = [model.model.layers[li].mlp.register_forward_hook(make_hook(li), with_kwargs=True) for li in range(L_HOOK + 1)]

    ids = torch.arange(0, 2048, device="cuda")
    CUR["ids"] = ids
    with torch.no_grad():
        state["compressed"] = False
        model(input_ids=ids.unsqueeze(0), use_cache=False)
        state["compressed"] = True
        model(input_ids=ids.unsqueeze(0), use_cache=False)
    for h in handles:
        h.remove()

    x_orig, x_comp = CAPTURE[1], CAPTURE[0]
    rel = ((x_comp - x_orig).norm(dim=1) / x_orig.norm(dim=1).clamp_min(1e-9))
    cos = F.cosine_similarity(x_comp, x_orig, dim=1)
    print(f"L{L_HOOK} input perturbation after 0..{L_HOOK-1} compressed (2048 tokens):")
    print(f"  rel err : mean {rel.mean()*100:.2f}%  median {rel.median()*100:.2f}%  p90 {rel.quantile(0.9)*100:.2f}%")
    print(f"  cos sim : mean {cos.mean():.4f}  min {cos.min():.4f}")


if __name__ == "__main__":
    main()
