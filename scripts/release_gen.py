"""Release generation wrapper: runs the VALIDATED dsv4_generate_ttt.py path
against the RELEASE layout (dsv4_release/{skeleton,layers,routers.safetensors}).

The release banks are bitwise identical to dsv4_reduced (verified per-tensor),
and the skeleton's shared experts are bitwise identical to lossless_layers
decode (verified), so results match the local validated pipeline.

Usage:
  python scripts/release_gen.py "<prompt>" [max_new] [--evolve ...]
All extra args are passed through to dsv4_generate_ttt.py (TTT is ON by
default - the model adapts during generation; use --no-ttt to disable).
"""
import os
import sys

RELEASE = os.path.abspath("dsv4_release")
SKELETON = os.path.join(RELEASE, "skeleton")

sys.argv[0] = "scripts/dsv4_generate_ttt.py"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv4_generate_ttt as T  # noqa: E402

# Re-point the module's data roots at the release layout. Router tables ship
# in routers.safetensors (not in the skeleton index); expert banks are the
# release safetensors files re-exposed as per-expert tensors.
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


def _patch():
    T.MODEL_DIR = SKELETON
    tok_snap = os.path.join(RELEASE, "tokenizer")
    T.TOKENIZER = os.path.join(tok_snap, "tokenizer.json")

    # --- routers: load from routers.safetensors ---
    rt = load_file(os.path.join(RELEASE, "routers.safetensors"), device="cpu")
    for li in range(T.N_LAYERS):
        T.ROUTER_W[li] = rt[f"router_w.{li}"].cuda().float()
        if li in T.HASH_LAYERS:
            T.ROUTER_TID[li] = rt[f"router_tid.{li}"].cuda().long()
        else:
            T.ROUTER_BIAS[li] = rt[f"router_bias.{li}"].cuda().float()

    # --- shared experts: skeleton bf16 weights are the pre-decoded
    # lossless values (verified bitwise); wrap them in the same dict
    # shape get_shared_dequant returns ---
    import json
    idx = json.load(open(os.path.join(SKELETON, "model.safetensors.index.json")))
    wm = idx["weight_map"]

    def _sk(k):
        from safetensors import safe_open
        with safe_open(os.path.join(SKELETON, wm[k]), framework="pt") as f:
            return f.get_tensor(k).cuda()

    for li in range(T.N_LAYERS):
        base = f"layers.{li}.ffn.shared_experts"
        T.SHARED_DEQUANT[li] = {
            "w1": _sk(f"{base}.w1.weight"), "w2": _sk(f"{base}.w2.weight"),
            "w3": _sk(f"{base}.w3.weight"),
        }

    # --- experts: expose release banks as per-expert dict via loader ---
    bank_cache: dict[int, dict] = {}

    def _bank(li):
        if li not in bank_cache:
            t = load_file(os.path.join(RELEASE, "layers", f"layer{li}.safetensors"),
                          device="cuda")
            bank_cache[li] = t
        return bank_cache[li]

    def _fake_expert(li, k):
        t = _bank(li)
        return {
            "w1a": t["w13t"][k, :2048], "w1a_scale": t["s13"][k, :2048],
            "w3a": t["w13t"][k, 2048:], "w3a_scale": t["s13"][k, 2048:],
            "bias1a": t["b13"][k, :2048], "bias3a": t["b13"][k, 2048:],
            "w2a": t["w2a"][k], "w2a_scale": t["s2"][k],
            "mode": "terni4", "n_real": 0, "n_val": 0, "residual": None,
        }

    import collections
    T.I4X_PACKED.clear()

    def _prewarm_remap(paths):
        # ignore ttt's dsv4_reduced paths; load from release banks instead
        loaded = 0
        for li in range(T.N_LAYERS):
            t = _bank(li)
            for k in range(256):
                e = _fake_expert(li, k)
                T.I4X_PACKED[(li, k)] = {kk: (v if torch.is_tensor(v) else v)
                                         for kk, v in e.items()}
                loaded += 1
        print(f"prewarm(release): {loaded} experts resident", flush=True)

    T.prewarm_packed = _prewarm_remap

    # POD: fp32 P/mu from dsv4_reduced is NOT part of the release; the banks
    # carry P f16 + mu f32. ttt loads P fp32 from dsv4_reduced. For the
    # release wrapper, use dsv4_reduced if present (local dev), else bank P.
    red_ok = os.path.isdir(T.REDUCED)

    def _load_pod(li):
        if li in T.POD_CACHE:
            return T.POD_CACHE[li]
        if red_ok:
            P = torch.load(os.path.join(T.REDUCED, f"layer_{li}", "P.pt"),
                           map_location="cuda").float()
            mu = torch.load(os.path.join(T.REDUCED, f"layer_{li}", "mu.pt"),
                            map_location="cuda").float()
        else:
            t = _bank(li)
            P = t["P"].cuda().float()
            mu = t["mu"].cuda().float().reshape(1, -1)
        T.POD_CACHE[li] = (P, mu)
        return P, mu

    T.load_pod = _load_pod


_patch()

if __name__ == "__main__":
    T.main()
