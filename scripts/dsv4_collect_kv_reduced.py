"""Collect KV (and x_L) activations on the FULL reduced model (all 43 layers).

Combines the reduced-FFN hooks from dsv4_generate_reduced.py with the
attention KV hook from dsv4_collect_attention.py, so the collected KV/x_L are
computed through the actual reduced experts (not the placeholder skeleton).

Outputs to checkpoints_dsv4/attention_reduced/:
  kv_L{L}.pt  [N, 512]  (kv_norm output, pre-RoPE, K==V)
  x_L{L}.pt   [N, 4096] (mlp input = cross-layer FFN input)

Usage:
    python scripts/dsv4_collect_kv_reduced.py [--max-tokens 3000]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import stub_import_tf  # noqa: F401
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
import gigatoken
import dsv4_generate_reduced as gr

TOKENIZER = r'C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json'
OUT_DIR = 'checkpoints_dsv4/attention_reduced'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-tokens', type=int, default=3000)
    args = ap.parse_args()

    import dsv4_collect_attention as ca
    tok = gigatoken.Tokenizer.from_json(open(TOKENIZER, 'rb').read())
    ids = [0] + torch.randperm(ca.VOCAB, dtype=torch.long).tolist()
    ids = ids[: args.max_tokens]
    n_tok = len(ids)
    print(f'tokens: {n_tok}', flush=True)

    # router + shared + reduced experts (same as generation path)
    snap = gr.de.default_snapshot()
    wm = gr.de.load_index(snap)['weight_map']
    gr.load_router(snap, wm)
    print('loading shared experts...', flush=True)
    shared_cache = gr.load_shared()
    print('loading reduced experts (all layers)...', flush=True)
    t0 = time.time()
    red_cache = {}
    for li in range(gr.N_LAYERS):
        red_cache[li] = gr.load_reduced_layer(li)
    print(f'reduced experts loaded in {time.time()-t0:.1f}s', flush=True)

    torch.set_default_device('cuda')
    model = DeepseekV4ForCausalLM.from_pretrained(gr.MODEL_DIR, torch_dtype=torch.float32)
    torch.set_default_device('cpu')
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = 'eager'
    model.config.gradient_checkpointing = False
    print(f'layers: {len(model.model.layers)}, model ready', flush=True)

    collected: dict[str, torch.Tensor] = {}
    handles = []

    def kv_hook(key):
        def hook(module, args, kwargs, output):
            collected[key] = output.detach().float().cpu().reshape(-1, output.shape[-1])
            return output
        return hook

    def xin_hook(key):
        def hook(module, args, kwargs):
            collected[key] = args[0].detach().float().cpu().reshape(-1, args[0].shape[-1])
        return hook

    gr.CURRENT_IDS = torch.tensor([ids], device='cuda', dtype=torch.long)
    for li in range(gr.N_LAYERS):
        layer = model.model.layers[li]
        handles.append(layer.self_attn.kv_norm.register_forward_hook(kv_hook(f'kv_L{li}'), with_kwargs=True))
        handles.append(layer.mlp.register_forward_pre_hook(xin_hook(f'x_L{li}'), with_kwargs=True))
        handles.append(layer.mlp.register_forward_hook(gr.make_reduced_hook(li, red_cache, shared_cache), with_kwargs=True))

    os.makedirs(OUT_DIR, exist_ok=True)
    input_ids = torch.tensor([ids], device='cuda', dtype=torch.long)
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)
    print(f'forward done', flush=True)

    for h in handles:
        h.remove()

    for key, val in collected.items():
        torch.save(val, os.path.join(OUT_DIR, f'{key}.pt'))
    print(f'saved {len(collected)} tensors to {OUT_DIR}', flush=True)
    for key in sorted(collected):
        print(f'  {key}: {tuple(collected[key].shape)}', flush=True)


if __name__ == '__main__':
    main()
