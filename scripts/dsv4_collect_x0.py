"""Regenerate ONLY x_0 (layer-0 FFN input) — minimal, no expert dequant.

Collects args[0] of layer 0's mlp (post-attention residual) via one skeleton
forward. Placeholder experts are used for ALL layers (output does not matter —
we only need layer 0's mlp INPUT). Fast: no FP4 dequant.

Usage: python scripts/dsv4_collect_x0.py
"""
from __future__ import annotations

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

MODEL_DIR = 'C:/HAGI_v2/dsv4_shared_only'
TOKENIZER = r'C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json'
XPATH = 'checkpoints_dsv4/pod/x_layer0.pt'

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


def main():
    tok = gigatoken.Tokenizer.from_json(open(TOKENIZER, 'rb').read())
    ids = [0] + list(tok.encode(TEXT))
    ids = ids[:3000]
    print(f'tokens: {len(ids)}', flush=True)

    torch.set_default_device('cuda')
    model = DeepseekV4ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    torch.set_default_device('cpu')
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = 'eager'
    model.config.gradient_checkpointing = False
    print('model ready', flush=True)

    collected = {}

    def hook(module, args, kwargs, output):
        x = args[0]
        collected['x0'] = x.reshape(-1, x.shape[-1]).float().detach().cpu()
        return output

    h = model.model.layers[0].mlp.register_forward_hook(hook, with_kwargs=True)

    input_ids = torch.tensor([ids], device='cuda', dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)
    print(f'forward done in {time.time()-t0:.1f}s', flush=True)
    h.remove()

    torch.save(collected['x0'], XPATH)
    print(f'saved {XPATH}: {tuple(collected["x0"].shape)}', flush=True)


if __name__ == '__main__':
    main()
