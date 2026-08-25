"""Generation with int4x-compressed experts + TTT (online RLS) and weight save-on-improvement.

Builds on dsv4_generate_fast.py (custom MoE routing hook). Differences:
  - experts with an int4x checkpoint (dsv4_reduced/layer_L/expert_k.pt) run COMPRESSED:
    z = (x - mu) @ P, y = (silu(soft_lim(z@W1q^T+b1)) * soft_lim(z@W3q^T+b3)) @ (int4W2*s2)^T
  - TTT engine (default ON for checkpointed layers): during generation each active
    expert accumulates anchored-RLS statistics G/C against the ORIGINAL FP4 expert
    (teacher pass on the same routed activations, subsampled 1/TTT_SAMPLE tokens);
    exact refit every TTT_REFIT accumulated rows, guarded on the buffered residual.
  - save-on-improvement: at the end of the run (after the chain of thought) each
    adapted expert is consolidated (snap to int4) and, if its HOLDOUT residual
    (last 20% of the accumulated buffer, never refit against... accumulated after
    the last refit) improved by >= TTT_MIN_GAIN vs the static int4 readout, the
    updated W2 is written back into the expert file ATOMICALLY (tmp + os.replace),
    with a `ttt` metadata block (tokens seen, residuals, timestamp).

Usage:
    python scripts/dsv4_generate_ttt.py "<prompt>" [max_new_tokens] [--no-ttt] [--no-save]
"""

from __future__ import annotations

import collections
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stub_import_tf  # noqa: F401  (must precede transformers: stubs torch.distributed)
import dsv4_experts as de
import gigatoken
from dsv4_experts import unpack_binary, unpack_int4
from dsv4_refit_experts import soft_lim
from safetensors import safe_open
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

MODEL_DIR = "C:/HAGI_v2/dsv4_shared_only"
TOKENIZER = r"C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json"
LOSSLESS = "C:/HAGI_v2/lossless_layers"
REDUCED = "dsv4_reduced"

N_LAYERS = 43
HASH_LAYERS = {0, 1, 2}
TOP_K = 6
ROUTED_SCALE = 1.5
SWIGLU_LIMIT = 10.0
BOS_ID = 0
EOS_ID = 1
GPU_HEADROOM = 8 * 1024**3

# --- TTT knobs ---
TTT_SAMPLE = 4  # update stats on every Nth activation row (token budget guard)
TTT_REFIT = 64  # exact solve after this many accumulated rows per expert
TTT_ALPHA = 1000.0  # anchor weight (in "token" units) toward the int4 init
TTT_LAM = 0.9995  # forgetting factor
TTT_HOLDOUT = 0.2  # tail fraction of the buffer used for save decision
TTT_MIN_GAIN = 0.02  # save only if holdout resid improves by >= 2% relative
TTT_MAX_EXPERTS = 16  # live RLS states (G/C ~48MB each) - LRU
TTT_ROWS_MAX = 2048  # max rows kept per expert for holdout eval

CURRENT_IDS: torch.Tensor | None = None
ROUTER_W: dict[int, torch.Tensor] = {}
ROUTER_BIAS: dict[int, torch.Tensor] = {}
ROUTER_TID: dict[int, torch.Tensor] = {}

PACKED_CACHE: collections.OrderedDict[tuple, dict] = collections.OrderedDict()
CACHE_BYTES = 0
SHARED_DEQUANT: dict[int, dict] = {}
HIT = 0
MISS = 0

INT4X: dict[tuple, dict] = {}  # (L,k) -> dequant-int4x expert (frozen parts)
TTT_STATE: collections.OrderedDict[tuple, dict] = collections.OrderedDict()
TTT_ROWS = 0  # global update counter for TTT_SAMPLE
SAVE_LOG: list[str] = []


def ffn(x, w1, w2, w3):
    gate = (x @ w1.T).clamp(max=SWIGLU_LIMIT)
    up = (x @ w3.T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return (torch.nn.functional.silu(gate) * up) @ w2.T


def load_router(snap, wm):
    for li in range(N_LAYERS):
        p = f"layers.{li}.ffn.gate"
        ROUTER_W[li] = de.read_tensor(snap, wm, f"{p}.weight", device="cuda").to(torch.float32)
        if li in HASH_LAYERS:
            ROUTER_TID[li] = de.read_tensor(snap, wm, f"{p}.tid2eid", device="cuda").to(torch.long)
        else:
            ROUTER_BIAS[li] = de.read_tensor(snap, wm, f"{p}.bias", device="cuda").to(torch.float32)


def get_expert_packed(li, k):
    global CACHE_BYTES, HIT, MISS
    key = (li, k)
    if key in PACKED_CACHE:
        PACKED_CACHE.move_to_end(key)
        HIT += 1
        return PACKED_CACHE[key]
    MISS += 1
    fp = os.path.join(LOSSLESS, f"layers_{li}_ffn.safetensors")
    base = f"layers.{li}.ffn.experts.{k}"
    with safe_open(fp, framework="pt", device="cuda") as f:
        d = {}
        for proj in ("w1", "w2", "w3"):
            d[f"{proj}.weight"] = f.get_tensor(f"{base}.{proj}.weight")
            d[f"{proj}.scale"] = f.get_tensor(f"{base}.{proj}.scale")
    PACKED_CACHE[key] = d
    PACKED_CACHE.move_to_end(key)
    CACHE_BYTES += sum(t.numel() * t.element_size() for t in d.values())
    free, _ = torch.cuda.mem_get_info()
    lim = max(0, free - GPU_HEADROOM)
    while CACHE_BYTES > lim and len(PACKED_CACHE) > 1:
        _, ev = PACKED_CACHE.popitem(last=False)
        CACHE_BYTES -= sum(t.numel() * t.element_size() for t in ev.values())
        del ev
    return d


def get_shared_dequant(li):
    if li in SHARED_DEQUANT:
        return SHARED_DEQUANT[li]
    fp = os.path.join(LOSSLESS, f"layers_{li}_ffn.safetensors")
    base = f"layers.{li}.ffn.shared_experts"
    d = {}
    with safe_open(fp, framework="pt", device="cuda") as f:
        for proj in ("w1", "w2", "w3"):
            w = f.get_tensor(f"{base}.{proj}.weight")
            s = f.get_tensor(f"{base}.{proj}.scale")
            d[proj] = de._decode(base, w, s)
    SHARED_DEQUANT[li] = d
    return d


def load_pod(li):
    red = os.path.join(REDUCED, f"layer_{li}")
    P = torch.load(os.path.join(red, "P.pt"), map_location="cuda").float()
    mu = torch.load(os.path.join(red, "mu.pt"), map_location="cuda").float()
    return P, mu


def get_int4x(li, k):
    """Dequantized int4x expert: returns frozen parts + current (possibly adapted) W2."""
    key = (li, k)
    if key in INT4X:
        return INT4X[key]
    fp = os.path.join(REDUCED, f"layer_{li}", f"expert_{k}.pt")
    if not os.path.exists(fp):
        return None
    e = torch.load(fp, map_location="cpu", weights_only=False)
    if e.get("mode") != "int4x":
        return None
    P, mu = load_pod(li)
    dev = "cuda"
    q1 = unpack_binary(e["w1a"]).float().to(dev)
    q3 = unpack_binary(e["w3a"]).float().to(dev)
    q2 = unpack_int4(e["w2a"]).float().to(dev)
    s1 = e["w1a_scale"].float().to(dev)
    s3 = e["w3a_scale"].float().to(dev)
    s2 = e["w2a_scale"].float().to(dev)
    w2 = q2 * s2[:, None]  # continuous readout (adapted in place by TTT)
    d = {
        "P": P,
        "mu": mu,
        "w1": q1 * s1[:, None],
        "w3": q3 * s3[:, None],
        "b1": e["bias1a"].float().to(dev),
        "b3": e["bias3a"].float().to(dev),
        "w2": w2,
        "residual": e.get("residual", float("inf")),
    }
    INT4X[key] = d
    return d


def int4x_forward(d, x_rows):
    """x_rows [n, 4096] raw -> y [n, 4096] via POD rotation + int4x expert."""
    z = (x_rows - d["mu"]) @ d["P"]
    g = soft_lim(z @ d["w1"].T + d["b1"])
    u = soft_lim(z @ d["w3"].T + d["b3"])
    return torch.nn.functional.silu(g) * u  # h


def _teacher_y(li, k, x_rows):
    p = get_expert_packed(li, k)
    w1 = de.dequant_fp4(p["w1.weight"], p["w1.scale"])
    w2 = de.dequant_fp4(p["w2.weight"], p["w2.scale"])
    w3 = de.dequant_fp4(p["w3.weight"], p["w3.scale"])
    y = ffn(x_rows, w1, w2, w3)
    del w1, w2, w3
    return y


def ttt_touch(li, k):
    key = (li, k)
    if key not in TTT_STATE:
        if len(TTT_STATE) >= TTT_MAX_EXPERTS:
            TTT_STATE.popitem(last=False)  # drop coldest state (weights stay adapted)
        Fd = INT4X[key]["w2"].shape[1]
        Dm = INT4X[key]["w2"].shape[0]
        alpha = None  # set on first row (needs h energy scale)
        TTT_STATE[key] = {
            "G": None,
            "C": None,
            "alpha": alpha,
            "Fd": Fd,
            "D": Dm,
            "H": [],
            "Y": [],
            "rows": 0,
            "since_refit": 0,
            "refits": 0,
        }
    TTT_STATE.move_to_end(key)
    return TTT_STATE[key]


def ttt_update(li, k, h_rows, y_rows):
    """Accumulate anchored RLS stats; exact refit every TTT_REFIT rows (guarded)."""
    st = ttt_touch(li, k)
    w2 = INT4X[(li, k)]["w2"]
    if st["G"] is None:
        Fd = st["Fd"]
        st["alpha"] = (h_rows.T @ h_rows).diagonal().mean().item() / max(1, h_rows.shape[0]) * TTT_ALPHA
        st["G"] = torch.eye(Fd, device="cuda") * st["alpha"]
        st["C"] = st["alpha"] * w2.T.clone()
    G, C = st["G"], st["C"]
    n = h_rows.shape[0]
    # pre-update residual on these rows (prequential)
    G.mul_(TTT_LAM**n).add_(h_rows.T @ h_rows)
    C.mul_(TTT_LAM**n).add_(h_rows.T @ y_rows)
    st["rows"] += n
    st["since_refit"] += n
    st["H"].append(h_rows.detach())
    st["Y"].append(y_rows.detach())
    if sum(t.shape[0] for t in st["H"]) > TTT_ROWS_MAX:
        st["H"].pop(0)
        st["Y"].pop(0)
    if st["since_refit"] >= TTT_REFIT:
        st["since_refit"] = 0
        st["refits"] += 1
        Hb = torch.cat(st["H"])
        Yb = torch.cat(st["Y"])
        r_before = _resid(w2, Hb, Yb)
        reg = G.diagonal().mean() * 1e-3
        w2_new = torch.linalg.solve(G + reg * torch.eye(st["Fd"], device="cuda"), C).T.contiguous()
        r_after = _resid(w2_new, Hb, Yb)
        if r_after < r_before:
            INT4X[(li, k)]["w2"].copy_(w2_new)
            del w2_new
            return r_before, r_after
        del w2_new
    return None


def _resid(w2, h, y):
    yh = h @ w2.T
    return (((yh - y) ** 2).sum() / (y**2).sum().clamp_min(1e-12)).item()


def snap_int4(w2):
    sg = w2.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 7.0
    q = (w2 / sg).round().clamp(-7, 7)
    num = (q * w2).sum(dim=1)
    den = (q * q).sum(dim=1).clamp_min(1e-9)
    a = num / den
    a = a.sign() * a.abs().clamp_min(1e-6)
    return q, a


def consolidate_and_save(li, k, enabled=True):
    """End-of-run: snap adapted W2 to int4; save if holdout improves vs static."""
    key = (li, k)
    if key not in TTT_STATE:
        return
    st = TTT_STATE.pop(key)
    if not st["H"]:
        return
    Hb = torch.cat(st["H"])
    Yb = torch.cat(st["Y"])
    w2 = INT4X[key]["w2"]
    # static reference: the ORIGINAL checkpoint readout (rebuild from file pattern)
    e = torch.load(os.path.join(REDUCED, f"layer_{li}", f"expert_{k}.pt"), map_location="cpu", weights_only=False)
    q2_0 = unpack_int4(e["w2a"]).float().cuda()
    w2_static = q2_0 * e["w2a_scale"].float().cuda()[:, None]
    del q2_0
    # holdout = tail of the accumulated buffer
    n_hold = max(64, int(Hb.shape[0] * TTT_HOLDOUT))
    Hh, Yh = Hb[-n_hold:], Yb[-n_hold:]
    r_static = _resid(w2_static, Hh, Yh)
    r_cont = _resid(w2, Hh, Yh)
    q, a = snap_int4(w2)
    r_cons = _resid(q * a[:, None], Hh, Yh)
    best_w2 = w2.contiguous()
    if r_cons > r_static:
        q, a = snap_int4(w2_static)
        r_cons = r_static
        best_w2 = w2_static
    improved = (r_static - r_cons) / r_static
    print(
        f"  [ttt] L{li} k{k}: tokens={st['rows']} refits={st['refits']} "
        f"holdout static={r_static*100:.3f}% cont={r_cont*100:.3f}% cons={r_cons*100:.3f}% "
        f"(gain {improved*100:+.1f}%)",
        flush=True,
    )
    if not enabled or improved < TTT_MIN_GAIN:
        return
    # atomic save: pattern + scale back into the expert file
    fp = os.path.join(REDUCED, f"layer_{li}", f"expert_{k}.pt")
    e["w2a"] = de.pack_int4(q.cpu())
    e["w2a_scale"] = a.cpu()
    e["residual"] = r_cons
    e["ttt"] = {"tokens": st["rows"], "refits": st["refits"], "static": r_static, "cons": r_cons}
    tmp = fp + ".tmp"
    torch.save(e, tmp)
    os.replace(tmp, fp)
    SAVE_LOG.append(f"L{li}_{k}: {r_static*100:.3f}% -> {r_cons*100:.3f}%")
    print(f"  [ttt] SAVED {fp}", flush=True)


def make_hook(li, ttt_on):
    def hook(module, args, kwargs, output):
        x = args[0]
        B, S, D = x.shape
        flat = x.reshape(-1, D).float()

        logits = flat @ ROUTER_W[li].T
        scores = torch.nn.functional.softplus(logits).sqrt()
        if li in HASH_LAYERS:
            indices = ROUTER_TID[li][CURRENT_IDS.reshape(-1)]
        else:
            indices = torch.topk(scores + ROUTER_BIAS[li], TOP_K, dim=-1).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * ROUTED_SCALE

        sw = get_shared_dequant(li)
        out = ffn(flat, sw["w1"], sw["w2"], sw["w3"])

        global TTT_ROWS
        for k in indices.unique().tolist():
            d = get_int4x(li, k) if ttt_on or True else None
            if d is not None:
                h = None
                out_k = None
                for kk in range(TOP_K):
                    m = indices[:, kk] == k
                    if not m.any():
                        continue
                    xm = flat[m]
                    if h is None:
                        h = int4x_forward(d, xm)
                        out_k = h @ d["w2"].T
                    out[m] += weights[m, kk, None] * out_k
                    if ttt_on:
                        TTT_ROWS += 1
                        if TTT_ROWS % TTT_SAMPLE == 0:
                            y_t = _teacher_y(li, k, xm)
                            ttt_update(li, k, h.float(), y_t.float())
                            del y_t
                del h, out_k
            else:
                p = get_expert_packed(li, k)
                w1 = de.dequant_fp4(p["w1.weight"], p["w1.scale"])
                w2 = de.dequant_fp4(p["w2.weight"], p["w2.scale"])
                w3 = de.dequant_fp4(p["w3.weight"], p["w3.scale"])
                ek = ffn(flat, w1, w2, w3)
                del w1, w2, w3
                for kk in range(TOP_K):
                    m = indices[:, kk] == k
                    if m.any():
                        out[m] += weights[m, kk, None] * ek[m]
                del ek

        correct = out.to(x.dtype).reshape(B, S, D)
        del out, flat, scores, logits, indices, weights
        return correct

    return hook


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    ttt_on = "--no-ttt" not in sys.argv
    save_on = "--no-save" not in sys.argv

    print("loading tokenizer...", flush=True)
    tok = gigatoken.Tokenizer.from_json(open(TOKENIZER, "rb").read())
    ids = [BOS_ID] + list(tok.encode(prompt))

    print("loading router...", flush=True)
    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]
    load_router(snap, wm)

    print("loading model skeleton...", flush=True)
    t0 = time.time()
    torch.set_default_device("cuda")
    model: DeepseekV4ForCausalLM = DeepseekV4ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    torch.set_default_device("cpu")
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = "eager"
    model.config.gradient_checkpointing = False
    free, total = torch.cuda.mem_get_info()
    print(f"model loaded in {time.time() - t0:.1f}s, GPU free={free / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)
    n_int4x = sum(1 for li in range(N_LAYERS) for k in range(256) if os.path.exists(os.path.join(REDUCED, f"layer_{li}", f"expert_{k}.pt")))
    print(f"int4x checkpoints available: {n_int4x} experts; ttt={'on' if ttt_on else 'off'} save={'on' if save_on else 'off'}", flush=True)

    handles = [model.model.layers[li].mlp.register_forward_hook(make_hook(li, ttt_on), with_kwargs=True) for li in range(N_LAYERS)]

    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    global CURRENT_IDS
    generated = list(ids)
    t0 = time.time()
    with torch.no_grad():
        CURRENT_IDS = input_ids
        out = model(input_ids=input_ids, use_cache=True, past_key_values=None)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        generated.append(nxt)
        for _ in range(max_new - 1):
            if nxt == EOS_ID:
                break
            CURRENT_IDS = torch.tensor([[nxt]], device="cuda", dtype=torch.long)
            out = model(input_ids=CURRENT_IDS, use_cache=True, past_key_values=past)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax().item())
            generated.append(nxt)

    for h in handles:
        h.remove()

    dt = time.time() - t0
    n_tok = len(generated) - len(ids)
    print(f"generated {n_tok} tokens in {dt:.1f}s ({n_tok / max(dt, 1e-9):.2f} tok/s)", flush=True)

    print("consolidating TTT experts...", flush=True)
    for (li, k) in list(TTT_STATE.keys()):
        consolidate_and_save(li, k, enabled=save_on)
    if SAVE_LOG:
        print(f"saved {len(SAVE_LOG)} expert(s): " + "; ".join(SAVE_LOG), flush=True)
    else:
        print("no expert met the save threshold", flush=True)

    text = tok.decode(generated)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    print("=== OUTPUT ===", flush=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
