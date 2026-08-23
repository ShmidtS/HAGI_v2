"""Generate with real 256-expert MoE + int8 KV-cache (2x KV memory win).

Same real-expert decode loop as dsv4_generate_real, but KV is stored as int8
(via dsv4_kvcache_int8.install_int8_compression) instead of bf16. Reports the
cache dtype (proof the method applied), KV byte counts, throughput and text.

Usage:
    python scripts/dsv4_generate_int8kv.py "<prompt>" [max_new_tokens]
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dsv4_experts as de
import dsv4_generate_real as gr
import dsv4_kvcache_int8 as ki
import gigatoken
from transformers.cache_utils import DynamicCache
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

import stub_import_tf  # noqa: F401

MODEL_DIR = "C:/HAGI_v2/dsv4_shared_only"
TOKENIZER = r"C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json"
SCALES_PATH = "C:/HAGI_v2/checkpoints_dsv4/kv_int8_scales.pt"
SKELETON_KV = "checkpoints_dsv4/attention_skeleton"

N_LAYERS = 43
BOS_ID = 0
EOS_ID = 1


def kv_bytes(cache) -> int:
    n = 0
    for li in range(len(cache.layers)):
        n += cache.layers[li].keys.numel() * cache.layers[li].keys.element_size()
        n += cache.layers[li].values.numel() * cache.layers[li].values.element_size()
    return n


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    print("loading tokenizer...", flush=True)
    tok = gigatoken.Tokenizer.from_json(open(TOKENIZER, "rb").read())
    ids = [BOS_ID] + list(tok.encode(prompt))
    print(f"prompt ids ({len(ids)})", flush=True)

    print("loading router weights...", flush=True)
    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]
    gr.load_router(snap, wm, device="cuda")

    print("loading model skeleton...", flush=True)
    t0 = time.time()
    torch.set_default_device("cuda")
    model: DeepseekV4ForCausalLM = DeepseekV4ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    torch.set_default_device("cpu")
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = "eager"
    model.config.gradient_checkpointing = False
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    handles = [
        model.model.layers[li].mlp.register_forward_hook(gr.make_hook(li), with_kwargs=True) for li in range(N_LAYERS)
    ]

    print("loading int8 KV scales...", flush=True)
    if os.path.exists(SCALES_PATH):
        scales = torch.load(SCALES_PATH, map_location="cpu", weights_only=True)
    else:
        scales = ki.compute_scales(SKELETON_KV)
    store = ki.Int8KVStore(scales)
    print(f"scales: {len(store)} layers", flush=True)

    cache = DynamicCache(config=model.config)
    ki.install_int8_compression(cache, store)

    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    gr.CURRENT_IDS = input_ids

    generated = list(ids)
    t_prefill = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, past_key_values=cache)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        generated.append(nxt)
        t_prefill = time.time() - t_prefill

        dtypes = {str(cache.layers[li].keys.dtype) for li in range(len(cache.layers))}
        print(f"cache key dtypes after prefill: {sorted(dtypes)}", flush=True)
        print(f"KV bytes: int8={kv_bytes(cache)}, bf16-equivalent={kv_bytes(cache) * 2}", flush=True)
        print(f"prefill {len(ids)} tokens in {t_prefill:.1f}s", flush=True)

        t_dec = time.time()
        n_dec = 0
        for _ in range(max_new - 1):
            if nxt == EOS_ID:
                break
            gr.CURRENT_IDS = torch.tensor([[nxt]], device="cuda", dtype=torch.long)
            out = model(input_ids=gr.CURRENT_IDS, use_cache=True, past_key_values=past)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax().item())
            generated.append(nxt)
            n_dec += 1
        t_dec = time.time() - t_dec

    for h in handles:
        h.remove()

    dtypes = {str(cache.layers[li].keys.dtype) for li in range(len(cache.layers))}
    print(f"final cache key dtypes: {sorted(dtypes)}", flush=True)
    if n_dec > 0:
        print(f"decoded {n_dec} tokens in {t_dec:.1f}s = {n_dec / t_dec:.2f} tok/s", flush=True)
    text = tok.decode(generated)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    print("=== OUTPUT ===", flush=True)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
