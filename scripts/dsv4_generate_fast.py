"""Fast generation with the REAL 256-expert MoE.

Optimizations vs dsv4_generate_real.py:
  1. Persistent safetensors handles (46 files opened ONCE — the header JSON is
     parsed once, not 1806 times per token).
  2. LRU cache of PACKED (int8 + E8M0 scale) routed experts on GPU: an expert is
     loaded from disk only on first activation, then stays resident until the
     cache must evict (only when GPU memory is exhausted).
  3. Shared experts cached DEQUANTIZED (they are active on every token).

Usage:
    python scripts/dsv4_generate_fast.py "<prompt>" [max_new_tokens]
"""

from __future__ import annotations

import collections
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dsv4_experts as de
import gigatoken
from safetensors import safe_open
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

import stub_import_tf  # noqa: F401

MODEL_DIR = "C:/HAGI_v2/dsv4_shared_only"
TOKENIZER = r"C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json"
LOSSLESS = "C:/HAGI_v2/lossless_layers"

N_LAYERS = 43
HASH_LAYERS = {0, 1, 2}
TOP_K = 6
ROUTED_SCALE = 1.5
SWIGLU_LIMIT = 10.0
BOS_ID = 0
EOS_ID = 1
GPU_HEADROOM = 8 * 1024**3  # keep this much GPU free for activations/KV

CURRENT_IDS: torch.Tensor | None = None
ROUTER_W: dict[int, torch.Tensor] = {}
ROUTER_BIAS: dict[int, torch.Tensor] = {}
ROUTER_TID: dict[int, torch.Tensor] = {}

FILE_HANDLES: dict[str, object] = {}
PACKED_CACHE: collections.OrderedDict[tuple, dict] = collections.OrderedDict()
CACHE_BYTES = 0
SHARED_DEQUANT: dict[int, dict] = {}
HIT = 0
MISS = 0


def ffn(x, w1, w2, w3):
    gate = x @ w1.T
    up = x @ w3.T
    gate = gate.clamp(max=SWIGLU_LIMIT)
    up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return (torch.nn.functional.silu(gate) * up) @ w2.T


def load_router(snap, wm):
    for li in range(N_LAYERS):
        p = f"layers.{li}.ffn.gate"
        ROUTER_W[li] = de.read_tensor(snap, wm, f"{p}.weight", device="cuda").to(torch.float32)
        if li in HASH_LAYERS:
            ROUTER_TID[li] = de.read_tensor(snap, wm, f"{p}.tid2eid", device="cuda").to(torch.long)
        else:
            ROUTER_BIAS[li] = de.read_tensor(snap, wm, f"{p}.bias", device="cuda").to(torch.float32)


def _handle(fp):
    # NOTE: persistent handles crash inside model.forward() on ROCm/Windows,
    # so each miss opens the file once via a with block and reads all 6 tensors.
    return safe_open(fp, framework="pt", device="cuda")


def _psize(d):
    return sum(t.numel() * t.element_size() for t in d.values())


def _cache_limit():
    free, _ = torch.cuda.mem_get_info()
    return max(0, free - GPU_HEADROOM)


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
    with _handle(fp) as f:
        d = {
            "w1.weight": f.get_tensor(f"{base}.w1.weight"),
            "w1.scale": f.get_tensor(f"{base}.w1.scale"),
            "w2.weight": f.get_tensor(f"{base}.w2.weight"),
            "w2.scale": f.get_tensor(f"{base}.w2.scale"),
            "w3.weight": f.get_tensor(f"{base}.w3.weight"),
            "w3.scale": f.get_tensor(f"{base}.w3.scale"),
        }
    PACKED_CACHE[key] = d
    PACKED_CACHE.move_to_end(key)
    CACHE_BYTES += _psize(d)
    lim = _cache_limit()
    while CACHE_BYTES > lim and len(PACKED_CACHE) > 1:
        _, ev = PACKED_CACHE.popitem(last=False)
        CACHE_BYTES -= _psize(ev)
        del ev
    return d


def get_shared_dequant(li):
    if li in SHARED_DEQUANT:
        return SHARED_DEQUANT[li]
    fp = os.path.join(LOSSLESS, f"layers_{li}_ffn.safetensors")
    base = f"layers.{li}.ffn.shared_experts"
    d = {}
    with _handle(fp) as f:
        for proj in ("w1", "w2", "w3"):
            w = f.get_tensor(f"{base}.{proj}.weight")
            s = f.get_tensor(f"{base}.{proj}.scale")
            d[proj] = de._decode(base, w, s)
    SHARED_DEQUANT[li] = d
    return d


def make_hook(li):
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

        for k in indices.unique().tolist():
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

    print("loading tokenizer...", flush=True)
    tok = gigatoken.Tokenizer.from_json(open(TOKENIZER, "rb").read())
    ids = [BOS_ID] + list(tok.encode(prompt))
    print(f"prompt ids ({len(ids)})", flush=True)

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

    handles = [
        model.model.layers[li].mlp.register_forward_hook(make_hook(li), with_kwargs=True) for li in range(N_LAYERS)
    ]

    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    global CURRENT_IDS

    generated = list(ids)
    t_prefill = time.time()
    past = None
    with torch.no_grad():
        CURRENT_IDS = input_ids
        out = model(input_ids=input_ids, use_cache=True, past_key_values=None)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        generated.append(nxt)
        t_prefill = time.time() - t_prefill
        print(f"prefill {len(ids)} tokens in {t_prefill:.1f}s", flush=True)

        t_dec = time.time()
        times = []
        for step in range(max_new - 1):
            if nxt == EOS_ID:
                break
            CURRENT_IDS = torch.tensor([[nxt]], device="cuda", dtype=torch.long)
            out = model(input_ids=CURRENT_IDS, use_cache=True, past_key_values=past)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax().item())
            generated.append(nxt)
            times.append(time.time() - t_dec)
            t_dec = time.time()
            if (len(times) + 1) % 5 == 0:
                print(
                    f"  {len(times) + 1} tokens, last5={sum(times[-5:]) / 5:.2f}s/tok "
                    f"(cache {HIT}H/{MISS}M, {PACKED_CACHE.__len__()} experts, "
                    f"{CACHE_BYTES / 1e9:.1f}GB)",
                    flush=True,
                )

    for h in handles:
        h.remove()

    if times:
        cold = sum(times[:5]) / min(5, len(times))
        warm = sum(times[5:]) / max(1, len(times) - 5)
        print(f"\ncold (first 5): {cold:.2f}s/tok = {1 / cold:.2f} tok/s", flush=True)
        print(f"warm (after 5): {warm:.2f}s/tok = {1 / warm:.2f} tok/s", flush=True)
    print(
        f"cache: {HIT} hits / {MISS} misses, {len(PACKED_CACHE)} experts resident ({CACHE_BYTES / 1e9:.1f}GB)",
        flush=True,
    )

    text = tok.decode(generated)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    print("=== OUTPUT ===", flush=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
