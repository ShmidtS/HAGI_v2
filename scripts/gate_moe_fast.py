"""Fast MoE path for latent-gate training (v2, sync-free).

Measured problems of v1 (400s/chunk):
  - boolean-mask indexing (z[m_any], (idx==k).any()) forces a device SYNC
    per expert per pass; under AMD_SERIALIZE_KERNEL=3 each sync drains the
    whole pipeline -> hundreds of seconds per chunk.
  - pass-2 recomputes pass-1's MoE forward although inputs differ only by
    the (zero-init, tiny) gate perturbation -> second-order error.

v2 design:
  1. SORT-SCATTER: one torch.sort over the n*K assignments per layer;
     per-expert work uses contiguous slices + index_select + index_add_.
     Two syncs per layer (uq.tolist + counts.tolist), zero boolean masks.
  2. PASS REUSE: train_gate sets PHASE=1 (no-grad pass) then PHASE=2
     (grad pass). Phase-1 stores each layer's MoE output; phase-2 returns
     the cached tensor (detached anyway) instead of recomputing. The
     attention/residual path in phase-2 is EXACT (fresh forward, grad
     flows through it); only MoE outputs are frozen (approximation,
     error = J_moe * delta_gate, second-order while the gate is small).
  3. RESIDENT BANKS: packed terni4 of all compressed layers on GPU
     (~57GB for 27 layers); hot UNPACKED weights cached with an LRU byte
     budget; FP4 tail streamed from mmap safetensors with an LRU dequant
     cache (no clear-all thrash).

Env:
  GATE_REUSE_MLP=0  disables pass-2 reuse (exact but 2x slower)
"""
import os, sys, glob
from collections import OrderedDict
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv4_generate_ttt as gen

REDUCED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dsv4_reduced")

BANKS: dict[int, dict[int, dict]] = {}   # li -> k -> packed tensors (GPU)
POD: dict[int, dict] = {}                # li -> {"P", "mu"} bf16
GPUBYTES = 0
HOT: "OrderedDict[tuple, tuple]" = OrderedDict()   # (li,k) -> (w1,w3,w2) bf16
HOT_BYTES = 0
HOT_BUDGET = 6_000_000_000
FP4_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()  # (li,k) -> (w1,w2,w3) bf16
FP4_BYTES = 0
FP4_BUDGET = 6_000_000_000

PHASE = 1          # set by train_gate: 1 = collect+cache, 2 = grad pass
CHUNK_ID = -1
REUSE = {}         # (chunk_id, li) -> output tensor [n, D] float
REUSE_ON = os.environ.get("GATE_REUSE_MLP", "1") == "1"
STATS = {"reuse": 0, "compute": 0, "hot_hit": 0, "hot_miss": 0}


def load_bank(li: int, verbose: bool = False) -> int:
    global GPUBYTES
    if li in BANKS:
        return 0
    bank = {}
    for fp in sorted(glob.glob(os.path.join(REDUCED, f"layer_{li}", "expert_*.pt"))):
        k = int(fp.split("expert_")[1].split(".")[0])
        e = torch.load(fp, map_location="cpu", weights_only=False)
        if e.get("mode") != "terni4":
            continue
        bank[k] = {
            "w1a": e["w1a"].cuda(non_blocking=True),
            "w1s": e["w1a_scale"].cuda(non_blocking=True),
            "w3a": e["w3a"].cuda(non_blocking=True),
            "w3s": e["w3a_scale"].cuda(non_blocking=True),
            "w2a": e["w2a"].cuda(non_blocking=True),
            "w2s": e["w2a_scale"].cuda(non_blocking=True),
            "b1": e["bias1a"].cuda(non_blocking=True),
            "b3": e["bias3a"].cuda(non_blocking=True),
        }
        GPUBYTES += sum(t.numel() * t.element_size() for t in bank[k].values())
        del e
    BANKS[li] = bank
    if verbose:
        free, _ = torch.cuda.mem_get_info()
        print(f"bank L{li}: {len(bank)} experts, banks {GPUBYTES/1e9:.1f}GB, "
              f"free {free/1e9:.1f}GB", flush=True)
    return len(bank)


def load_pod(li: int):
    if li in POD:
        return POD[li]
    P = torch.load(os.path.join(REDUCED, f"layer_{li}", "P.pt"),
                   map_location="cuda").to(torch.bfloat16)
    mu = torch.load(os.path.join(REDUCED, f"layer_{li}", "mu.pt"),
                    map_location="cuda").to(torch.bfloat16)
    POD[li] = {"P": P, "mu": mu.reshape(1, -1)}
    return POD[li]


_TRIT_LUT = None


def _lut():
    """uint8 -> 5 ternary trits, LUT [256, 5] int8 (-1/0/1). Gather beats
    the int32 divmod chain 3x (measured 1.3ms vs 4.0ms per [2048,820])."""
    global _TRIT_LUT
    if _TRIT_LUT is None:
        lut = torch.zeros(256, 5, dtype=torch.int8)
        for v in range(256):
            t = v
            for i in range(5):
                lut[v, i] = (t % 3) - 1
                t //= 3
        _TRIT_LUT = lut.cuda()
    return _TRIT_LUT


def _tern_w(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    t = _lut()[a.long()].to(torch.float32)
    t = t.reshape(a.shape[0], a.shape[1] * 5)[:, : s.shape[1] * 128]
    return (t * s.repeat_interleave(128, dim=1)).to(torch.bfloat16)


def _int4_w(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    t = a.to(torch.int16)
    lo = ((t & 15) - 8).to(torch.bfloat16)
    hi = ((t >> 4) - 8).to(torch.bfloat16)
    out = torch.empty(t.shape[0], t.shape[1] * 2, dtype=torch.bfloat16, device=a.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    gs = out.shape[1] // s.shape[1]
    return out * s.repeat_interleave(gs, dim=1).to(torch.bfloat16)


def _hot_get(li: int, k: int):
    """Unpacked (w1, w3, w2) bf16 for expert k of compressed layer li, LRU."""
    global HOT_BYTES
    key = (li, k)
    if key in HOT:
        HOT.move_to_end(key)
        STATS["hot_hit"] += 1
        return HOT[key]
    STATS["hot_miss"] += 1
    d = BANKS[li][k]
    w1 = _tern_w(d["w1a"], d["w1s"])
    w3 = _tern_w(d["w3a"], d["w3s"])
    w2 = _int4_w(d["w2a"], d["w2s"])
    val = (w1, w3, w2)
    HOT[key] = val
    HOT_BYTES += sum(t.numel() * t.element_size() for t in val)
    while HOT_BYTES > HOT_BUDGET and len(HOT) > 1:
        _, ev = HOT.popitem(last=False)
        HOT_BYTES -= sum(t.numel() * t.element_size() for t in ev)
    return val


def _route(li: int, flat: torch.Tensor):
    """Router scores/indices/weights; identical math to gen.make_hook."""
    logits = flat @ gen.ROUTER_W[li].T
    scores = F.softplus(logits).sqrt()
    if li in gen.HASH_LAYERS:
        indices = gen.ROUTER_TID[li][gen.CURRENT_IDS.reshape(-1)]
    else:
        indices = torch.topk(scores + gen.ROUTER_BIAS[li], gen.TOP_K, dim=-1).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * gen.ROUTED_SCALE
    return indices.long(), weights


def _sorted_assign(indices: torch.Tensor, weights: torch.Tensor):
    """-> (uq_list, counts_list, row_pos_sorted [m], w_sorted [m]) with
    exactly 2 device syncs."""
    n, K = indices.shape
    se, perm = torch.sort(indices.reshape(-1))
    row_pos = perm // K
    w_sorted = weights.reshape(-1)[perm]
    uq, counts = torch.unique_consecutive(se, return_counts=True)
    return uq.tolist(), counts.tolist(), row_pos, w_sorted


def _shared_out(li: int, flatb: torch.Tensor) -> torch.Tensor:
    sw = gen.get_shared_dequant(li)
    return gen.ffn(flatb, sw["w1"], sw["w2"], sw["w3"]).float()


def _batched_experts(li, fired, z, row_pos, w_sorted, cnt, uq):
    """All fired experts of one layer in O(few) big ops.

    cat packed W13 rows of every fired expert -> ONE LUT unpack -> ONE
    z@W1_all.T / z@W3_all.T GEMM (bf16) -> one soft_lim(+bias) / silu ->
    per-expert W2 GEMM on slices (block sizes differ, cannot batch) with
    index_add_ scatter. Replaces per-expert unpack+3 GEMMs (~10 launches
    each) with ~10 big launches total per layer.
    """
    soft_lim = gen.soft_lim
    bank = BANKS[li]
    n = z.shape[0]
    # offsets into the sorted assignment arrays per fired expert
    offs = {}
    o = 0
    for k, c in zip(uq, cnt):
        if k in bank:
            offs[k] = (o, o + c)
        o += c
    # --- batched W13: cat packed rows, unpack once ---
    w1a = torch.cat([bank[k]["w1a"] for k in fired])          # [F*I, n/5]
    w1s = torch.cat([bank[k]["w1s"] for k in fired])
    w3a = torch.cat([bank[k]["w3a"] for k in fired])
    w3s = torch.cat([bank[k]["w3s"] for k in fired])
    b1 = torch.cat([bank[k]["b1"] for k in fired])            # [F*I] fp32
    b3 = torch.cat([bank[k]["b3"] for k in fired])
    W1 = _tern_w(w1a, w1s)                                    # [F*I, D] bf16
    W3 = _tern_w(w3a, w3s)
    del w1a, w3a
    G = soft_lim(z @ W1.T + b1.to(torch.bfloat16)[None, :])   # [n, F*I]
    U = soft_lim(z @ W3.T + b3.to(torch.bfloat16)[None, :])
    del W1, W3
    H = (F.silu(G.float()) * U.float()).to(torch.bfloat16)   # [n, F*I]
    del G, U
    # --- per-expert W2 GEMM + scatter ---
    acc = torch.zeros(n, z.shape[1], dtype=torch.float32, device=z.device)
    I_ = bank[fired[0]]["w1a"].shape[0]
    for k in fired:
        o0, o1 = offs[k]
        rpos = row_pos[o0:o1]
        wsel = w_sorted[o0:o1]
        h_k = H[:, k * I_:(k + 1) * I_]                        # [n, I] bf16
        # only the rows routed to this expert
        h_sel = h_k.index_select(0, rpos)
        w2 = _int4_w(bank[k]["w2a"], bank[k]["w2s"])
        y = (h_sel @ w2.T).float()
        acc.index_add_(0, rpos, wsel[:, None] * y)
    return acc


def make_fast_hook(li: int, detach: bool = True):
    soft_lim = gen.soft_lim

    def hook(module, args, kwargs, output):
        x = args[0]
        B, S, D = x.shape
        # ---- pass-2 reuse: MoE output frozen from phase 1 ----
        ck = (CHUNK_ID, li)
        if REUSE_ON and PHASE == 2 and ck in REUSE:
            STATS["reuse"] += 1
            out = REUSE.pop(ck)
            return out.reshape(B, S, D).to(x.dtype)
        x = args[0]
        flat = x.reshape(-1, D).float()
        flatb = flat.to(torch.bfloat16)
        indices, weights = _route(li, flat)
        out = _shared_out(li, flatb)

        if li not in BANKS:
            load_bank(li)
        pod = load_pod(li)
        z = (flatb - pod["mu"]) @ pod["P"]          # hoisted once per layer

        uq, cnt, row_pos, w_sorted = _sorted_assign(indices, weights)
        bank = BANKS[li]
        fired = [k for k in uq if k in bank]
        out = out + _batched_experts(li, fired, z, row_pos, w_sorted, cnt, uq)
        STATS["compute"] += 1
        if REUSE_ON and PHASE == 1:
            REUSE[ck] = out.detach()
        if detach and out.requires_grad:
            out = out.detach()
        return out.reshape(B, S, D).to(x.dtype)

    return hook


def make_fp4_hook(li: int, detach: bool = True):
    from safetensors import safe_open
    from dsv4_experts import dequant_fp4
    global FP4_BYTES
    LOSSLESS = gen.LOSSLESS if hasattr(gen, "LOSSLESS") else "C:/HAGI_v2/lossless_layers"
    base_fp = os.path.join(LOSSLESS, f"layers_{li}_ffn.safetensors")

    def hook(module, args, kwargs, output):
        global FP4_BYTES
        x = args[0]
        B, S, D = x.shape
        ck = (CHUNK_ID, li)
        if REUSE_ON and PHASE == 2 and ck in REUSE:
            STATS["reuse"] += 1
            out = REUSE.pop(ck)
            return out.reshape(B, S, D).to(x.dtype)
        flat = x.reshape(-1, D).float()
        flatb = flat.to(torch.bfloat16)
        indices, weights = _route(li, flat)
        out = _shared_out(li, flatb)

        uq, cnt, row_pos, w_sorted = _sorted_assign(indices, weights)
        acc = torch.zeros_like(out)
        o = 0
        with safe_open(base_fp, framework="pt", device="cpu") as f:
            base = f"layers.{li}.ffn.experts"
            for k, c in zip(uq, cnt):
                rpos = row_pos[o:o + c]
                wsel = w_sorted[o:o + c]
                o += c
                key = (li, k)
                if key in FP4_CACHE:
                    FP4_CACHE.move_to_end(key)
                    w1, w2, w3 = FP4_CACHE[key]
                else:
                    w1 = dequant_fp4(f.get_tensor(f"{base}.{k}.w1.weight").cuda(),
                                     f.get_tensor(f"{base}.{k}.w1.scale").cuda()).to(torch.bfloat16)
                    w2 = dequant_fp4(f.get_tensor(f"{base}.{k}.w2.weight").cuda(),
                                     f.get_tensor(f"{base}.{k}.w2.scale").cuda()).to(torch.bfloat16)
                    w3 = dequant_fp4(f.get_tensor(f"{base}.{k}.w3.weight").cuda(),
                                     f.get_tensor(f"{base}.{k}.w3.scale").cuda()).to(torch.bfloat16)
                    val = (w1, w2, w3)
                    FP4_CACHE[key] = val
                    FP4_BYTES += sum(t.numel() * t.element_size() for t in val)
                    while FP4_BYTES > FP4_BUDGET and len(FP4_CACHE) > 1:
                        _, ev = FP4_CACHE.popitem(last=False)
                        FP4_BYTES -= sum(t.numel() * t.element_size() for t in ev)
                ek = gen.ffn(flatb.index_select(0, rpos), w1, w2, w3).float()
                acc.index_add_(0, rpos, wsel[:, None] * ek)
        out = out + acc
        STATS["compute"] += 1
        if REUSE_ON and PHASE == 1:
            REUSE[ck] = out.detach()
        if detach and out.requires_grad:
            out = out.detach()
        return out.reshape(B, S, D).to(x.dtype)

    return hook


def install_hooks(model, compressed_layers, detach: bool = True):
    handles = []
    n_lay = len(model.model.layers)
    for li in range(n_lay):
        if li in compressed_layers:
            h = model.model.layers[li].mlp.register_forward_hook(
                make_fast_hook(li, detach), with_kwargs=True)
        else:
            h = model.model.layers[li].mlp.register_forward_hook(
                make_fp4_hook(li, detach), with_kwargs=True)
        handles.append(h)
    return handles
