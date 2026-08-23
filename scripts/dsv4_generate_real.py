"""Generate with the REAL 256-expert MoE (loaded on-the-fly from lossless_layers)
with KV-cache enabled (O(n) decode instead of O(n^2)).

Loads the compact skeleton (correct non-expert weights), replaces each MoE
block's output via forward hooks with the exact router + top-6 routed experts
+ shared expert computed from lossless_layers, and decodes autoregressively
with past_key_values.

Usage:
    python scripts/dsv4_generate_real.py "<prompt>" [max_new_tokens]
"""

from __future__ import annotations

import os
import sys
import time
from typing import cast

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dsv4_experts as de
import gigatoken
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

CURRENT_IDS: torch.Tensor | None = None
ROUTER_W: dict[int, torch.Tensor] = {}
ROUTER_BIAS: dict[int, torch.Tensor] = {}
ROUTER_TID: dict[int, torch.Tensor] = {}


def ffn(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    gate = x @ w1.T
    up = x @ w3.T
    gate = gate.clamp(max=SWIGLU_LIMIT)
    up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return (torch.nn.functional.silu(gate) * up) @ w2.T


def load_router(snap: str, wm: dict, device: str) -> None:
    for li in range(N_LAYERS):
        p = f"layers.{li}.ffn.gate"
        ROUTER_W[li] = de.read_tensor(snap, wm, f"{p}.weight", device=device).to(torch.float32)
        if li in HASH_LAYERS:
            ROUTER_TID[li] = de.read_tensor(snap, wm, f"{p}.tid2eid", device=device).to(torch.long)
        else:
            ROUTER_BIAS[li] = de.read_tensor(snap, wm, f"{p}.bias", device=device).to(torch.float32)


def make_hook(li: int):
    def hook(module, args, kwargs, output):
        x = args[0]  # [B, S, D] bf16
        B, S, D = x.shape
        flat = x.reshape(-1, D).float()
        N = flat.shape[0]

        logits = flat @ ROUTER_W[li].T
        scores = torch.nn.functional.softplus(logits).sqrt()
        if li in HASH_LAYERS:
            ids_flat = CURRENT_IDS.reshape(-1)
            indices = ROUTER_TID[li][ids_flat]
        else:
            indices = torch.topk(scores + ROUTER_BIAS[li], TOP_K, dim=-1).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * ROUTED_SCALE

        fp = os.path.join(LOSSLESS, f"layers_{li}_ffn.safetensors")
        prefix = f"layers.{li}.ffn"
        shared = de.load_shared_file(fp, prefix, device="cuda")
        out = ffn(flat, shared["w1"], shared["w2"], shared["w3"])
        del shared

        for k in indices.unique().tolist():
            E = de.load_expert_file(fp, prefix, k, device="cuda")
            ek = ffn(flat, E["w1"], E["w2"], E["w3"])
            del E
            for kk in range(TOP_K):
                m = indices[:, kk] == k
                if m.any():
                    out[m] += weights[m, kk, None] * ek[m]
            del ek

        correct = out.to(x.dtype).reshape(B, S, D)
        del out, flat, scores, logits, indices, weights
        return correct

    return hook


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print("loading tokenizer...", flush=True)
    tok = gigatoken.Tokenizer.from_json(open(TOKENIZER, "rb").read())
    ids = [BOS_ID] + list(tok.encode(prompt))
    print(f"prompt ids ({len(ids)})", flush=True)

    print("loading router weights...", flush=True)
    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]
    load_router(snap, wm, device="cuda")

    print("loading model skeleton...", flush=True)
    t0 = time.time()
    torch.set_default_device("cuda")
    model: DeepseekV4ForCausalLM = DeepseekV4ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    torch.set_default_device("cpu")
    model.eval()
    model = cast(DeepseekV4ForCausalLM, cast(torch.nn.Module, model).to(torch.bfloat16))
    model.config._experts_implementation = "eager"
    model.config.gradient_checkpointing = False
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    handles = [
        model.model.layers[li].mlp.register_forward_hook(make_hook(li), with_kwargs=True) for li in range(N_LAYERS)
    ]

    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    global CURRENT_IDS

    print(f"prefill + generating {max_new} tokens...", flush=True)
    generated = list(ids)
    t_prefill = time.time()
    past = None
    with torch.no_grad():
        # prefill
        CURRENT_IDS = input_ids
        out = model(input_ids=input_ids, use_cache=True, past_key_values=None)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        generated.append(nxt)
        t_prefill = time.time() - t_prefill
        print(f"  prefill {len(ids)} tokens in {t_prefill:.1f}s", flush=True)

        # decode loop
        t_dec = time.time()
        n_dec = 0
        for step in range(max_new - 1):
            if nxt == EOS_ID:
                break
            CURRENT_IDS = torch.tensor([[nxt]], device="cuda", dtype=torch.long)
            out = model(input_ids=CURRENT_IDS, use_cache=True, past_key_values=past)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax().item())
            generated.append(nxt)
            n_dec += 1
            if (n_dec + 1) % 5 == 0:
                print(f"  {n_dec + 1} tokens, {time.time() - t_dec:.1f}s", flush=True)
        t_dec = time.time() - t_dec

    for h in handles:
        h.remove()

    print(f"decoded {n_dec} tokens in {t_dec:.1f}s = {n_dec / t_dec:.2f} tok/s", flush=True)
    text = tok.decode(generated)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    print("=== OUTPUT ===", flush=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
