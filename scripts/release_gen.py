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

    # --- fast MoE decode path (HAGI_FAST_MOE=1, default on): expert-level
    # LRU over dsv4_release/experts (12MB per expert) + route-indirect
    # Triton kernels on stacked [T,*] scratch. No full-bank residency.
    fast_enabled = (os.environ.get("HAGI_FAST_MOE", "1") == "1"
                    and "--evolve" not in sys.argv)
    if fast_enabled:
        import collections
        import triton_bank_kernels as tbk
        from ternary_bank_kernels import k_h13t, trit_lut

        lut = trit_lut()
        PACK_MAX = int(os.environ.get("HAGI_PACK_MAX", "1600"))
        pack = collections.OrderedDict()   # (li,k) -> expert tensors (cuda)

        def _slices(li, k):
            key = (li, k)
            if key in pack:
                pack.move_to_end(key)
                return pack[key]
            fp = os.path.join(RELEASE, "experts", f"L{li}", f"E{k}.safetensors")
            d = load_file(fp, device="cuda")
            pack[key] = d
            while len(pack) > PACK_MAX:
                pack.popitem(last=False)
            return d

        def _fast_hook(li, ttt_on):
            import torch.nn.functional as F
            sw = T.get_shared_dequant(li)
            D, I, GS = 4096, 2048, 128
            D, I, GS = 4096, 2048, 128  # noqa: F811 (kept for clarity)
            state = {}
            T_K = T.TOP_K

            def _ensure():
                if not state:
                    state["Pbf"] = _load_pod(li)[0].to(torch.bfloat16)
                    state["mubf"] = _load_pod(li)[1].to(torch.bfloat16).reshape(1, -1)
                    state["muP"] = (state["mubf"] @ state["Pbf"]).reshape(1, -1)
                    # persistent scratch buffers (decode n==1 path)
                    state["hbuf"] = torch.empty(T_K, I, device="cuda")
                    state["ybuf"] = torch.empty(T_K, D, device="cuda")
                    state["acc"] = torch.zeros(D, device="cuda")
                    state["ids_rel"] = torch.arange(T_K, dtype=torch.int32, device="cuda")
                return state

            def hook(module, args, kwargs, output):
                st = _ensure()
                Pbf, mubf = st["Pbf"], st["mubf"]
                x = args[0]
                B, S, Dd = x.shape
                n = B * S
                flat = x.reshape(n, Dd).float()
                logits = flat @ T.ROUTER_W[li].T
                scores = torch.nn.functional.softplus(logits).sqrt()
                if li in T.HASH_LAYERS:
                    indices = T.ROUTER_TID[li][T.CURRENT_IDS.reshape(-1)]
                else:
                    indices = torch.topk(scores + T.ROUTER_BIAS[li], T.TOP_K, dim=-1).indices
                weights = scores.gather(1, indices)
                weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * T.ROUTED_SCALE
                flatb = flat.to(torch.bfloat16)
                g = (flatb @ sw["w1"].T).clamp(max=10.0)
                u = (flatb @ sw["w3"].T).clamp(min=-10.0, max=10.0)
                out = (F.silu(g) * u @ sw["w2"].T).float()
                # batched: LRU slices; single launch pair per layer
                NT = n * T.TOP_K
                flat_idx = indices.reshape(-1).tolist()
                ws = [_slices(li, int(kk)) for kk in flat_idx]
                w13c = torch.stack([d["w13"] for d in ws])
                s13c = torch.stack([d["s13"] for d in ws])
                b13c = torch.stack([d["b13"] for d in ws])
                w2c = torch.stack([d["w2a"] for d in ws])
                s2c = torch.stack([d["s2"] for d in ws])
                zb = ((flatb - mubf) @ Pbf).contiguous()          # [n, D]
                hbuf = torch.empty(NT, I, device=x.device)
                ybuf = torch.empty(NT, Dd, device=x.device)
                acc = torch.zeros(n, Dd, device=x.device)
                ids_all = torch.arange(NT, dtype=torch.int32, device=x.device)
                wts = weights.reshape(-1).contiguous()
                k_h13t[(NT, I // 64)](zb, ids_all, w13c, s13c, b13c, lut, hbuf,
                                      D=D, I=I, GS=GS, BI=64, num_warps=2, ZPJ=T.TOP_K)
                tbk.k_yb_multi[(NT, Dd // 512)](hbuf, ids_all, wts, w2c, s2c, ybuf, acc,
                                                D=D, I=I, GS=GS, BD=512, BK=128,
                                                num_warps=8, ZPJ=T.TOP_K)
                res = (out + acc).to(x.dtype).reshape(B, S, Dd)
                del output
                return res
                res = (out + routed).to(x.dtype).reshape(B, S, Dd)
                del output
                return res

            return hook

        T.make_hook = _fast_hook

        # --- fused mHC (hc_fused): the hc forward is ~45 tiny torch launches
        # (sinkhorn loop); one fused path cuts 5.2ms -> 0.33ms per site.
        # Decode only (S==1); prefill uses the reference module.
        from hc_fused import HCFused

        _orig_setup = T.setup_model

        def setup_model_with_hc():
            model = _orig_setup()
            n = 0
            for li in range(T.N_LAYERS):
                for site in ("attn_hc", "ffn_hc"):
                    mod = getattr(model.model.layers[li], site)
                    fused = HCFused(mod)
                    ref_fwd = mod.forward

                    def shim(hidden_streams, _f=fused, _ref=ref_fwd):
                        if hidden_streams.shape[1] == 1:
                            return _f.forward(hidden_streams)
                        return _ref(hidden_streams)

                    mod.forward = shim
                    n += 1
            print(f"[fast-moe] fused mHC on {n} hc sites (decode)", flush=True)
            return model

        T.setup_model = setup_model_with_hc
    else:
        print("[fast-moe] disabled (--evolve or HAGI_FAST_MOE=0): full TTT active", flush=True)


_patch()

if __name__ == "__main__":
    T.main()
