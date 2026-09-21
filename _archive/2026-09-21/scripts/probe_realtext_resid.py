"""Direct probe: expert output error on REAL TEXT rows (no TTT machinery).

Runs a real text passage through the model with compressed hooks (I4X_LAYERS),
and for hot experts at probe layers measures
  resid = ||h @ w2b.T - y_teacher||^2 / ||y_teacher||^2
with y_teacher = ORIGINAL FP4 expert on the SAME drifted rows.
This bypasses TTT capture entirely - a clean cross-check of the 100% val-resid
anomaly seen in TTT consolidation logs.

Usage: I4X_LAYERS=0,1,... python scripts/probe_realtext_resid.py [n_tokens]
"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import stub_import_tf  # noqa: F401
import dsv4_experts as de
from dsv4_experts import (HASH_LAYERS, ROUTER_BIAS, ROUTER_TID, ROUTER_W,
                          TOP_K, ffn, load_router)
import dsv4_generate_ttt as gen
from dsv4_generate_ttt import get_dequant, get_int4x

PROBE_LAYERS = [int(x) for x in os.environ.get("PROBE_LAYERS", "5,11,20,29,35,42").split(",")]
TEXT = ("The theory of general relativity states that gravity is not a force but "
        "a curvature of spacetime caused by mass and energy. This theory has been "
        "confirmed by many experiments and observations, including the bending of "
        "light around massive objects and the precise orbit of Mercury. In the "
        "modern formulation, Einstein's field equations relate the geometry of "
        "spacetime to the distribution of matter and energy within it.")

CUR_IDS = None
ACC: dict[int, dict[str, list]] = {}
COMP = set(int(x) for x in os.environ.get("I4X_LAYERS", "").split(",") if x != "")


def main():
    n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]
    load_router(snap, wm)
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
    from transformers import AutoTokenizer

    tok_path = r"C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if os.environ.get("RANDOM_TOKENS") == "1":
        torch.manual_seed(4321)
        ids = torch.randint(0, 129280, (1, n_tokens))
    else:
        ids = tokenizer(TEXT, return_tensors="pt").input_ids
    ids = ids[:, :n_tokens]
    print(f"{'RANDOM' if os.environ.get('RANDOM_TOKENS') == '1' else 'real'} text: {ids.shape[1]} tokens", flush=True)

    torch.set_default_device("cuda")
    model = DeepseekV4ForCausalLM.from_pretrained(
        "C:/HAGI_v2/dsv4_shared_only", torch_dtype=torch.bfloat16)
    torch.set_default_device("cpu")
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = "eager"
    model.config.gradient_checkpointing = False
    global CUR_IDS

    def make_hook(li):
        def hook(module, args_, kwargs, output):
            x = args_[0]
            B, S, D = x.shape
            flat = x.reshape(-1, D).float()
            import torch.nn.functional as F
            logits = flat @ ROUTER_W[li].T
            scores = F.softplus(logits).sqrt()
            if li in HASH_LAYERS:
                indices = ROUTER_TID[li][CUR_IDS.reshape(-1)]
            else:
                indices = torch.topk(scores + ROUTER_BIAS[li], TOP_K, dim=-1).indices
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)

            flatb = flat.to(torch.bfloat16)
            sw = gen.get_shared_dequant(li)
            out = ffn(flatb, sw["w1"], sw["w2"], sw["w3"]).float()

            comp = li in COMP
            if li in PROBE_LAYERS:
                for k in indices.reshape(-1).unique().tolist():
                    m_any = (indices == k).any(dim=1)
                    n_rows = int(m_any.sum())
                    if n_rows < 3:
                        continue
                    x_k = flat[m_any]
                    w1, w2, w3 = get_dequant(li, k)
                    y_t = ffn(x_k.to(torch.bfloat16), w1, w2, w3).float()
                    d = get_int4x(li, k)
                    if d is None:
                        continue
                    h = gen.int4x_forward(d, x_k)
                    y_c = (h.to(torch.bfloat16) @ d["w2b"].T).float()
                    err = (y_c - y_t).pow(2).sum() / y_t.pow(2).sum().clamp_min(1e-30)
                    # pre-activation stats: soft_lim knee check (knee=7.5, lim=10)
                    zb = (x_k.to(torch.bfloat16) - d["mu"]) @ d["P"]
                    gg = (zb @ d["w1"].T + d["b1"]).float()
                    uu = (zb @ d["w3"].T + d["b3"]).float()
                    frac_sat = ((gg.abs() > 7.5).float().mean() + (uu.abs() > 7.5).float().mean()) / 2
                    ACC.setdefault(li, {}).setdefault(k, []).append(
                        (n_rows, err.item(), frac_sat.item()))
            if comp:
                _z = None
                for k in indices.unique().tolist():
                    d = get_int4x(li, k)
                    m_any = (indices == k).any(dim=1)
                    if _z is None and d is not None:
                        _z = (flatb - d["mu"]) @ d["P"]
                    if d is None:
                        w1, w2, w3 = get_dequant(li, k)
                        y_k = ffn(flatb, w1, w2, w3).float()
                        out_k = y_k
                    else:
                        h = gen.int4x_forward_z(d, _z[m_any])
                        out_k = (h.to(torch.bfloat16) @ d["w2b"].T).float()
                    pos = torch.cumsum(m_any.long(), 0) - 1
                    for kk in range(TOP_K):
                        m = indices[:, kk] == k
                        if m.any():
                            w_row = weights[m, kk, None]
                            out[m] += w_row * out_k[pos[m]]
            else:
                for k in indices.unique().tolist():
                    w1, w2, w3 = get_dequant(li, k)
                    m_any = (indices == k).any(dim=1)
                    y_k = ffn(flatb, w1, w2, w3).float()
                    pos = torch.cumsum(m_any.long(), 0) - 1
                    for kk in range(TOP_K):
                        m = indices[:, kk] == k
                        if m.any():
                            out[m] += weights[m, kk, None] * y_k[pos[m]]
            return out.to(x.dtype).reshape(B, S, D)
        return hook

    def e_loader(li, k):
        ep = os.path.join("dsv4_reduced", f"layer_{li}", f"expert_{k}.pt")
        if os.path.exists(ep):
            ee = torch.load(ep, map_location="cpu", weights_only=False)
            return ee.get("residual", float("nan"))
        return float("nan")

    e = {}
    for li in range(43):
        model.model.layers[li].mlp.register_forward_hook(make_hook(li), with_kwargs=True)

    with torch.no_grad():
        CUR_IDS = ids.reshape(-1)
        model(ids.cuda())

    print("\n=== per-layer expert resid on REAL TEXT (compressed fwd vs teacher) ===")
    for li in PROBE_LAYERS:
        if li not in ACC:
            continue
        items = sorted(ACC[li].items(), key=lambda kv: -kv[1][0][0])[:6]
        for k, v in items:
            n, r, fs = v[0]
            ck = e_loader(li, k)
            print(f"L{li:2d} k{k:3d} rows={n:3d}  realtext_resid={r * 100:7.2f}%  "
                  f"fit_resid={ck * 100:.3f}%  sat>7.5={fs * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
