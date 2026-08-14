"""End-to-end test: KV-POD compressed cache + YaRN 32 on the reduced model.

Reuses dsv4_generate_reduced (as `gr`) for loading router/shared/reduced
experts + skeleton. Then:
  1. Pre-creates DynamicCache(config=model.config) (eager, all 43 layers)
  2. install_kv_compression -> sliding KV stored at 256 dims (P_kv from pod_reduced)
  3. patch_yarn_factor(32) -> 1M -> 2M positional
  4. Runs a short generation, reports cache dims + memory per token + output.

Usage:
    python scripts/dsv4_test_kvpod.py "<prompt>" [max_new_tokens]
"""
from __future__ import annotations

import sys
import time
import os

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import stub_import_tf  # noqa: F401
from transformers.cache_utils import DynamicCache
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

import dsv4_generate_reduced as gr
import dsv4_kvcache_pod as kpod
import dsv4_experts as de
import gigatoken


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'The capital of France is'
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    tok = gigatoken.Tokenizer.from_json(open(gr.TOKENIZER, 'rb').read())
    ids = [gr.BOS_ID] + list(tok.encode(prompt))
    print(f'prompt ids ({len(ids)})', flush=True)

    snap = de.default_snapshot()
    wm = de.load_index(snap)['weight_map']
    gr.load_router(snap, wm)
    print('loading shared experts...', flush=True)
    shared_cache = gr.load_shared()
    print('loading reduced experts (all layers)...', flush=True)
    t0 = time.time()
    red_cache = {li: gr.load_reduced_layer(li) for li in range(gr.N_LAYERS)}
    print(f'reduced experts loaded in {time.time()-t0:.1f}s', flush=True)

    torch.set_default_device('cuda')
    # Patch YaRN factor 16 -> 32 BEFORE module construction (inv_freq baked at init)
    config = DeepseekV4Config.from_pretrained(gr.MODEL_DIR)
    kpod.patch_yarn_factor(config, factor=32)
    model = DeepseekV4ForCausalLM.from_pretrained(gr.MODEL_DIR, config=config, torch_dtype=torch.float32)
    torch.set_default_device('cpu')
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = 'eager'
    model.config.gradient_checkpointing = False

    handles = [model.model.layers[li].mlp.register_forward_hook(
        gr.make_reduced_hook(li, red_cache, shared_cache), with_kwargs=True)
        for li in range(gr.N_LAYERS)]

    # --- KV-POD: compressed cache + YaRN 32 ---
    kvc = kpod.KVCompressor('checkpoints_dsv4/pod_reduced', rank=256)
    print(f'KVCompressor: {len(kvc)} bases loaded', flush=True)
    past = DynamicCache(config=model.config)
    kpod.install_kv_compression(past, kvc)
    print(f'cache layers: {len(past.layers)}; YaRN factor={model.config.rope_parameters["compress"]["factor"]}', flush=True)

    input_ids = torch.tensor([ids], device='cuda', dtype=torch.long)
    generated = list(ids)
    with torch.no_grad():
        gr.CURRENT_IDS = input_ids
        out = model(input_ids=input_ids, use_cache=True, past_key_values=past)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        generated.append(nxt)
        for _ in range(max_new - 1):
            if nxt == gr.EOS_ID:
                break
            gr.CURRENT_IDS = torch.tensor([[nxt]], device='cuda', dtype=torch.long)
            out = model(input_ids=gr.CURRENT_IDS, use_cache=True, past_key_values=past)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax().item())
            generated.append(nxt)
    for h in handles:
        h.remove()

    # --- report cache dims + memory ---
    dims = sorted({tuple(l.keys.shape) for l in past.layers if hasattr(l, 'keys') and l.keys.numel()})
    dims_desc = '; '.join(f'{d}' for d in dims[:3]) + (f' ... ({len(dims)} unique)' if len(dims) > 3 else '')
    sliding_dim = [l.keys.shape[-1] for l in past.layers if hasattr(l, 'keys') and l.keys.numel()]
    total_tokens = sum(l.keys.shape[-2] for l in past.layers if hasattr(l, 'keys') and l.keys.numel())
    total_elems = sum(l.keys.numel() + l.values.numel() for l in past.layers if hasattr(l, 'keys'))
    mem_gb = total_elems * 2 / 1e9  # bf16
    print(f'cache key shapes: {dims_desc}', flush=True)
    print(f'sliding key head_dim: {set(sliding_dim)}', flush=True)
    print(f'cache tokens (sum): {total_tokens}; elems: {total_elems}; mem: {mem_gb:.3f} GB', flush=True)

    text = tok.decode(generated)
    print(f'OUTPUT: {text}', flush=True)


if __name__ == '__main__':
    main()
