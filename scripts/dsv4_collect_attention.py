"""Collect attention + cross-layer activations for POD basis construction.

Hooks every decoder layer and captures:
  - KV vector: `self_attn.kv_norm` output      -> [N, 512]      (pre-RoPE, K==V)
  - Q vector:  `self_attn.q_b_norm` output     -> [N, heads, 512] (pre-RoPE)
  - layer FFN input x_L: `mlp` input args[0]   -> [N, 4096]
  - layer output x_{L+1}: decoder layer output -> [N, 4096]

The first layer's activations are exact even on the placeholder-expert
skeleton (layer-0 attention sees the real embedding). Later layers are only
valid on a model whose experts approximate the original (the reduced model),
so Q collection is restricted to a configurable layer subset to bound RAM.

Usage:
    python scripts/dsv4_collect_attention.py                    # skeleton, layer-0 Q only
    python scripts/dsv4_collect_attention.py --model dsv4_reduced --q-layers 0,21,42
    python scripts/dsv4_collect_attention.py --max-tokens 6000 --q-layers all
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
import dsv4_experts as de

TOKENIZER = r'C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json'
OUT_DIR = 'checkpoints_dsv4/attention'
VOCAB = 129280

TEXT = (
    'The transformer architecture processes a sequence of tokens through stacked layers. Each layer applies multi-head attention followed by a feed-forward network. '
    'Mixture-of-experts models replace the dense feed-forward network with expert networks and a router that selects a small subset per token. The router scores each token and selects the top few. '
    'def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2) '
    'import numpy as np; x = np.linspace(0, 1, 100); y = np.sin(x) + np.cos(2*x) '
    'The theory of relativity transformed our understanding of space and time. Quantum mechanics describes matter at the smallest scales. '
    'Rust is a systems programming language with fearless concurrency and memory safety without a garbage collector. '
    'Машинное обучение учится на данных через итеративную оптимизацию функции потерь. Градиентный спуск обновляет параметры в направлении уменьшения ошибки. '
    'Reinforcement learning trains agents through rewards and environment interaction. The agent learns a policy that maximizes expected return. '
    'A neural network is a composition of affine transformations and nonlinearities. Training minimizes a loss over a dataset. '
    'The attention mechanism computes weighted averages of values based on query-key similarity. '
) * 20


def parse_q_layers(spec: str, n_layers: int) -> list[int]:
    if spec == 'all':
        return list(range(n_layers))
    return [int(x) for x in spec.split(',') if x.strip() != '']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None, help='model dir (default: original DeepSeek-V4-Flash snapshot)')
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--max-tokens', type=int, default=3000)
    ap.add_argument('--q-layers', default='0', help="comma list, 'all', or 'none'")
    args = ap.parse_args()

    model_dir = args.model or de.default_snapshot()
    tok = gigatoken.Tokenizer.from_json(open(TOKENIZER, 'rb').read())
    ids = [0] + torch.randperm(VOCAB, dtype=torch.long).tolist()
    ids = ids[: args.max_tokens]
    n_tok = len(ids)
    print(f'tokens: {n_tok} (shuffled vocab)', flush=True)

    torch.set_default_device('cuda')
    model = DeepseekV4ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    torch.set_default_device('cpu')
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = 'eager'
    model.config.gradient_checkpointing = False
    n_layers = len(model.model.layers)
    print(f'layers: {n_layers}, model ready', flush=True)

    q_layers = set(parse_q_layers(args.q_layers, n_layers))
    collected: dict[str, torch.Tensor] = {}
    handles = []

    def mk_hook(key):
        def hook(module, args, kwargs, output):
            if isinstance(output, tuple):
                output = output[0]
            collected[key] = output.detach().float().cpu().reshape(-1, output.shape[-1])
            return output
        return hook

    def mk_input_hook(key):
        def hook(module, args, kwargs):
            x = args[0]
            collected[key] = x.detach().float().cpu().reshape(-1, x.shape[-1])
        return hook

    for li in range(n_layers):
        layer = model.model.layers[li]
        handles.append(layer.self_attn.kv_norm.register_forward_hook(mk_hook(f'kv_L{li}'), with_kwargs=True))
        if li in q_layers:
            handles.append(layer.self_attn.q_b_norm.register_forward_hook(mk_hook(f'q_L{li}'), with_kwargs=True))
        handles.append(layer.mlp.register_forward_pre_hook(mk_input_hook(f'x_L{li}'), with_kwargs=True))
        handles.append(layer.register_forward_hook(mk_hook(f'xout_L{li}'), with_kwargs=True))

    os.makedirs(args.out, exist_ok=True)
    input_ids = torch.tensor([ids], device='cuda', dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)
    print(f'forward done in {time.time()-t0:.1f}s', flush=True)

    for h in handles:
        h.remove()

    for key, val in collected.items():
        path = os.path.join(args.out, f'{key}.pt')
        torch.save(val, path)
    print(f'saved {len(collected)} tensors to {args.out}', flush=True)
    for key in sorted(collected):
        print(f'  {key}: {tuple(collected[key].shape)}', flush=True)


if __name__ == '__main__':
    main()
