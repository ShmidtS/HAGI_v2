"""Self-contained loader for the HAGI-compressed DeepSeek-V4-Flash.

Loads the released artifact set (skeleton + routers + terni4 expert
banks) from a local directory and returns a generative model. No
HAGI-repo paths; everything resolves from `root`.

    from hagi_load import load_compressed_model, generate
    model, tok = load_compressed_model("dsv4_release")
    text = generate(model, tok, "The capital of France is Paris.", 60)

The compressed experts replace the skeleton's (absent) MoE weights at
runtime via forward-hook output replacement; skeleton attention and
embeddings stay untouched. The routed expert math is the validated
terni4 recipe: z = (x - mu) @ P; h = silu(soft_lim(z @ W1^T + b1)) *
soft_lim(z @ W3^T + b3); y = h @ W2^T, where soft_lim is the
smooth-clamp activation the original experts use (swiglu_limit).

Optional env:
  HAGI_BANK_BUDGET   bytes; LRU eviction of resident layer banks
                     (default 60 GB — the empirically stable footprint
                     on a 107 GB APU; raise on larger GPUs)
  HAGI_KV_INT8=1     int8 KV-cache (2x KV memory, quality-neutral)

Requirements: torch, transformers, safetensors.
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F

N_LAYERS = 43
D, I, GS = 4096, 2048, 128
HASH_LAYERS = (0, 1, 2)
TOP_K = 6
ROUTED_SCALE = 1.5             # routed_scaling_factor (original config)
SWIGLU_LIMIT = 10.0            # swiglu_limit (original config)
SOFT_KNEE = 0.75               # exact identity below knee*limit, tanh above


def soft_lim(x: torch.Tensor) -> torch.Tensor:
    """The original experts' soft clamp: exact identity below knee*limit,
    smooth tanh rolloff to the limit above (bounded signal, no dead
    gradients). Must match the refit recipe bit-for-bit in behavior."""
    th = SOFT_KNEE * SWIGLU_LIMIT
    tail = SWIGLU_LIMIT - th
    ax = x.abs()
    y_abs = torch.where(ax <= th, ax, th + tail * torch.tanh((ax - th) / tail))
    return y_abs * torch.sign(x)


# ---------------------------------------------------------------- unpack ---

_TRIT_LUT = None


def _lut() -> torch.Tensor:
    """uint8 -> 5 ternary trits, LUT [256, 5] int8 (-1/0/1)."""
    global _TRIT_LUT
    if _TRIT_LUT is None:
        lut = torch.empty(256, 5, dtype=torch.int8)
        for v in range(256):
            x = v
            for j in range(5):          # little-end trits, 3^5 = 243 <= 256
                lut[v, j] = (x % 3) - 1
                x //= 3
        _TRIT_LUT = lut.cuda()
    return _TRIT_LUT


def _tern_w(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """uint8 [R, ceil(I/5)] -> bf16 ternary [R, I] x per-(row,group) scale."""
    t = _lut()[a.long()].to(torch.float32)
    t = t.reshape(a.shape[0], a.shape[1] * 5)[:, : s.shape[1] * 128]
    return (t * s.repeat_interleave(128, dim=1)).to(torch.bfloat16)


def _int4_w(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """uint8 int4 packed (+8, lo-nibble = even col) -> bf16 [R, 2C]."""
    t = a.to(torch.int16)
    lo = ((t & 15) - 8).to(torch.bfloat16)
    hi = ((t >> 4) - 8).to(torch.bfloat16)
    out = torch.empty(t.shape[0], t.shape[1] * 2, dtype=torch.bfloat16, device=a.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    gs = out.shape[1] // s.shape[1]
    return out * s.repeat_interleave(gs, dim=1).to(torch.bfloat16)


# ----------------------------------------------------------------- model ---

@torch.no_grad()
def load_compressed_model(root: str, device: str = "cuda"):
    """-> (model, tokenizer). Raises if the release set is incomplete."""
    root = os.path.abspath(root)
    for sub in ("skeleton", "layers", "routers.safetensors", "tokenizer"):
        if not os.path.exists(os.path.join(root, sub)):
            raise FileNotFoundError(f"incomplete release: missing {sub}")

    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4ForCausalLM)
    from safetensors.torch import load_file

    model = DeepseekV4ForCausalLM.from_pretrained(
        os.path.join(root, "skeleton"), torch_dtype=torch.float32)
    model.eval()
    model = model.to(torch.bfloat16).to(device)
    model.config._experts_implementation = "eager"
    for p in model.parameters():
        p.requires_grad_(False)

    # --- router tables (shipped separately from the skeleton) ---
    rt = load_file(os.path.join(root, "routers.safetensors"), device="cpu")
    ROUTER_W, ROUTER_BIAS, ROUTER_TID = {}, {}, {}
    for li in range(N_LAYERS):
        ROUTER_W[li] = rt[f"router_w.{li}"].to(device).float()
        if li in HASH_LAYERS:
            ROUTER_TID[li] = rt[f"router_tid.{li}"].to(device)
        else:
            ROUTER_BIAS[li] = rt[f"router_bias.{li}"].to(device).float()
    del rt

    # --- lazy bank store with LRU eviction ---
    bank_files = sorted(glob.glob(os.path.join(root, "layers", "layer*.safetensors")))
    if len(bank_files) != N_LAYERS:
        raise FileNotFoundError(f"expected {N_LAYERS} layer banks, found {len(bank_files)}")
    BANKS: dict[int, dict] = {}
    BANK_BYTES: dict[int, int] = {}
    POD: dict[int, dict] = {}
    held = 0
    budget = int(os.environ.get("HAGI_BANK_BUDGET", str(60_000_000_000)))

    def load_layer(li: int):
        nonlocal held
        if li in BANKS:
            return BANKS[li], POD[li]
        t = load_file(bank_files[li], device="cpu")
        w13t = t["w13t"].to(device, non_blocking=True)
        s13 = t["s13"].to(device, non_blocking=True)
        b13 = t["b13"].to(device, non_blocking=True)
        w2a = t["w2a"].to(device, non_blocking=True)
        s2 = t["s2"].to(device, non_blocking=True)
        P = t["P"].to(device).to(torch.bfloat16)
        mu = t["mu"].to(device).to(torch.bfloat16).reshape(1, -1)
        by = sum(x.numel() * x.element_size() for x in (w13t, s13, b13, w2a, s2))
        bank = {}
        for k in range(256):
            bank[k] = {
                "w1a": w13t[k, :I], "w3a": w13t[k, I:],
                "w1s": s13[k, :I], "w3s": s13[k, I:],
                "b1": b13[k, :I], "b3": b13[k, I:],
                "w2a": w2a[k], "w2s": s2[k],
            }
        BANKS[li] = bank
        POD[li] = {"P": P, "mu": mu}
        held += by
        while held > budget and len(BANKS) > 1:
            oldest = next(iter(BANKS))
            if oldest == li:
                break
            BANKS.pop(oldest)
            POD.pop(oldest, None)
            held -= BANK_BYTES.pop(oldest, 0)
        BANK_BYTES[li] = by
        return bank, POD[li]

    state = {"cur_ids": None}

    def route(li: int, flat: torch.Tensor, n: int):
        logits = flat @ ROUTER_W[li].T
        scores = F.softplus(logits).sqrt()
        if li in HASH_LAYERS:
            indices = ROUTER_TID[li][state["cur_ids"]]
        else:
            indices = torch.topk(scores + ROUTER_BIAS[li], TOP_K, dim=-1).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * ROUTED_SCALE
        return indices.long(), weights

    def make_forward(li: int):
        def mlp_forward(x, input_ids=None, **kw):
            B, S, Dh = x.shape
            n = B * S
            flat = x.reshape(n, Dh).float()
            flatb = flat.to(torch.bfloat16)
            if input_ids is not None:
                state["cur_ids"] = input_ids.reshape(-1)[:n]
            if li in HASH_LAYERS and (state["cur_ids"] is None or state["cur_ids"].shape[0] != n):
                raise RuntimeError("hash layers route by token id: pass input_ids")

            bank, pod = load_layer(li)
            indices, weights = route(li, flat, n)

            # sync-free routing scatter (one sort, 2 host syncs)
            se, perm = torch.sort(indices.reshape(-1))
            row_pos = perm // TOP_K
            w_sorted = weights.reshape(-1)[perm]
            uq, counts = torch.unique_consecutive(se, return_counts=True)

            z = (flatb - pod["mu"]) @ pod["P"]          # POD rotation, hoisted
            acc = torch.zeros(n, Dh, dtype=torch.float32, device=x.device)
            o = 0
            for k, c in zip(uq.tolist(), counts.tolist()):
                rpos = row_pos[o:o + c]
                wsel = w_sorted[o:o + c]
                o += c
                d = bank[k]
                w1 = _tern_w(d["w1a"], d["w1s"])
                w3 = _tern_w(d["w3a"], d["w3s"])
                g = soft_lim(z[rpos] @ w1.T + d["b1"].to(torch.bfloat16)[None, :])
                u = soft_lim(z[rpos] @ w3.T + d["b3"].to(torch.bfloat16)[None, :])
                h = F.silu(g.float()) * u.float()
                w2 = _int4_w(d["w2a"], d["w2s"])
                y = (h.to(torch.bfloat16) @ w2.T).float()
                acc.index_add_(0, rpos, wsel[:, None] * y)
            return acc.to(x.dtype).reshape(B, S, Dh)
        return mlp_forward

    for li in range(N_LAYERS):
        model.model.layers[li].mlp.forward = make_forward(li)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer"))
    return model, tok


def install_int8_kv(model) -> None:
    """HAGI_KV_INT8=1: patch the cache to store KV as int8 (2x memory)."""
    if os.environ.get("HAGI_KV_INT8", "0") != "1":
        return
    from safetensors.torch import load_file
    root = os.environ.get("HAGI_KV_SCALES", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "kv_int8_scales.pt"))
    scales = torch.load(root, map_location="cpu", weights_only=True)
    store = _Int8Store(scales)
    orig_update = model._cache_class_update if False else None  # patched per-cache
    _patch_cache_class(store)


class _Int8Store:
    def __init__(self, scales):
        self.scales = scales

    def compress(self, kv, li):
        s = self.scales[li].to(device=kv.device)
        return (kv.float() / s).round().clamp(-127, 127).to(torch.int8)

    def decompress(self, q, li):
        s = self.scales[li].to(device=q.device, dtype=torch.float32)
        return (q.float() * s).to(torch.bfloat16)


def _patch_cache_class(store) -> None:
    from transformers.cache_utils import DynamicCache

    if getattr(DynamicCache, "_hagi_int8", False):
        return
    orig_init = DynamicCache.__init__

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        if getattr(self, "layers", None):
            for layer in self.layers:
                layer.update = _int8_update(layer.update, store)

    def _int8_update(orig, st):
        def upd(key_states, value_states, *a, **kw):
            li = a[0] if a else kw.get("layer_idx", 0)
            qk, qv = st.compress(key_states, li), st.compress(value_states, li)
            k, v = orig(qk, qv, *a, **kw)
            return st.decompress(k, li), st.decompress(v, li)
        return upd

    DynamicCache.__init__ = init
    DynamicCache._hagi_int8 = True


def _load_kv_scales():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_int8_scales.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(
            "HAGI_KV_INT8=1 but kv_int8_scales.pt is missing from the release")
    return torch.load(p, map_location="cpu", weights_only=True)


def install_int8_kv(model) -> None:  # noqa: F811 - single clean entry
    if os.environ.get("HAGI_KV_INT8", "0") != "1":
        return
    store = _Int8Store(_load_kv_scales())
    _patch_cache_class(store)


@torch.no_grad()
def generate(model, tok, prompt: str, max_new: int = 64, temp: float = 0.0) -> str:
    """Greedy (temp=0) or top-p=0.95 sampling continuation."""
    ids = tok.encode(prompt)
    if not ids:
        ids = [0]
    input_ids = torch.tensor([ids], device=model.device, dtype=torch.long)
    generated = list(ids)
    logits = None
    past = None
    for _ in range(max_new):
        out = model(input_ids=input_ids if past is None else input_ids[:, -1:],
                    use_cache=True, past_key_values=past)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        if temp <= 0:
            nxt = int(logits.argmax())
        else:
            probs = F.softmax(logits / temp, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cum = sp.cumsum(0)
            sp[cum - sp > 0.95] = 0
            nxt = int(si[torch.multinomial(sp / sp.sum(), 1)])
        generated.append(nxt)
        input_ids = torch.tensor([[nxt]], device=model.device, dtype=torch.long)
    return tok.decode(generated)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("prompt")
    ap.add_argument("max_new", nargs="?", type=int, default=64)
    args = ap.parse_args()
    model, tok = load_compressed_model(args.root)
    install_int8_kv(model)
    print(generate(model, tok, args.prompt, args.max_new))
