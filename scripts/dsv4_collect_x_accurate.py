"""Per-expert activation collection on the ORIGINAL model.

Collects, for EVERY one of the 11008 experts (43 layers × 256):
  - x_k: the real MLP input tokens routed to expert k (post routing, original model)
  - y_k: the original FP4 expert k output on those tokens  (teacher, for comparison)

Plus:
  - x_layer{L}.pt : per-layer MLP input [N, 4096] (post attention, pre MoE)
  - x_global.pt   : one global activation set [N*43, 4096] (concatenated x_L,
                    for extrapolation / global POD basis)

Real FP4 experts are loaded one layer at a time inside a forward hook, so
x_L[li] is exact (true layers 0..li-1 before it).

Outputs (checkpoints_dsv4/pod_accurate/):
  x_layer{L}.pt                       [N, 4096]
  acts_layer{L}.pt  -> {k: (x_k, y_k)}  for all 256 experts of layer L
  x_global.pt                         [N*43, 4096]

Usage: python scripts/dsv4_collect_x_accurate.py [--max-tokens 3000]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import stub_import_tf  # noqa: F401
from safetensors import safe_open
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
import dsv4_experts as de
import gigatoken

MODEL_DIR = 'C:/HAGI_v2/dsv4_shared_only'
TOKENIZER = r'C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json'
LOSSLESS = 'C:/HAGI_v2/lossless_layers'
OUT = 'checkpoints_dsv4/pod_accurate'

N_LAYERS = 43
HASH_LAYERS = {0, 1, 2}
TOP_K = 6
ROUTED_SCALE = 1.5
SWIGLU_LIMIT = 10.0

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


ROUTER_W = {}
ROUTER_BIAS = {}
ROUTER_TID = {}


def load_router(snap, wm):
    for li in range(N_LAYERS):
        p = f'layers.{li}.ffn.gate'
        ROUTER_W[li] = de.read_tensor(snap, wm, f'{p}.weight', device='cuda').to(torch.float32)
        if li in HASH_LAYERS:
            ROUTER_TID[li] = de.read_tensor(snap, wm, f'{p}.tid2eid', device='cuda').to(torch.long)
        else:
            ROUTER_BIAS[li] = de.read_tensor(snap, wm, f'{p}.bias', device='cuda').to(torch.float32)


def load_shared():
    shared = {}
    for li in range(N_LAYERS):
        fp = os.path.join(LOSSLESS, f'layers_{li}_ffn.safetensors')
        prefix = f'layers.{li}.ffn'
        sh = de.load_shared_file(fp, prefix, device='cuda')
        shared[li] = {kk: sh[kk].float() for kk in ('w1', 'w2', 'w3')}
    return shared


def dequant_fp4_batch(w, scale):
    """Batched fp4 decode: w [B, out, in//2] int8, scale [B, out, in//32] -> [B, out, in] fp32."""
    table = de.FP4_TABLE.to(w.device)
    u = w.to(torch.uint8)
    low = table[(u & 0x0F).long()]
    high = table[((u >> 4) & 0x0F).long()]
    B, out, in2 = u.shape
    v = torch.empty((B, out, in2 * 2), dtype=torch.float32, device=w.device)
    v[:, :, 0::2] = low
    v[:, :, 1::2] = high
    s = scale.to(torch.float32).repeat_interleave(de.FP4_BLOCK, dim=2)
    return v * s


def load_selected_experts(li, ids):
    """Load + batched-dequant ONLY the routed experts (ids list) -> dict k:(w1,w2,w3)."""
    fp = os.path.join(LOSSLESS, f'layers_{li}_ffn.safetensors')
    ids = sorted(ids)
    experts = {}
    with safe_open(fp, framework='pt', device='cpu') as f:
        for proj in ('w1', 'w2', 'w3'):
            w_list = [f.get_tensor(f'layers.{li}.ffn.experts.{k}.{proj}.weight') for k in ids]
            s_list = [f.get_tensor(f'layers.{li}.ffn.experts.{k}.{proj}.scale') for k in ids]
            w_stack = torch.stack(w_list).to('cuda')
            s_stack = torch.stack(s_list).to('cuda')
            decoded = dequant_fp4_batch(w_stack, s_stack)
            for i, k in enumerate(ids):
                experts.setdefault(k, {})[proj] = decoded[i]
    return {k: (v['w1'], v['w2'], v['w3']) for k, v in experts.items()}


def ffn(xin, w1, w2, w3):
    g = (xin @ w1.T).clamp(max=SWIGLU_LIMIT)
    u = (xin @ w3.T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return (F.silu(g) * u) @ w2.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-tokens', type=int, default=3000)
    args = ap.parse_args()

    tok = gigatoken.Tokenizer.from_json(open(TOKENIZER, 'rb').read())
    ids = [0] + list(tok.encode(TEXT))
    ids = ids[: args.max_tokens]
    print(f'tokens: {len(ids)}', flush=True)

    snap = de.default_snapshot()
    wm = de.load_index(snap)['weight_map']
    load_router(snap, wm)
    print('loading shared experts...', flush=True)
    shared_cache = load_shared()

    torch.set_default_device('cuda')
    model = DeepseekV4ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    torch.set_default_device('cpu')
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = 'eager'
    model.config.gradient_checkpointing = False
    print('skeleton ready', flush=True)

    os.makedirs(OUT, exist_ok=True)
    CURRENT_IDS = torch.tensor(ids, device='cuda', dtype=torch.long)
    global_x = []

    def make_hook(li):
        def hook(module, args_, kwargs, output):
            x = args_[0]
            B, S, D = x.shape
            flat = x.reshape(-1, D).float()
            torch.save(flat.detach().cpu(), os.path.join(OUT, f'x_layer{li}.pt'))
            global_x.append(flat.detach().cpu())

            t0 = time.time()
            logits = flat @ ROUTER_W[li].T
            scores = F.softplus(logits).sqrt()
            if li in HASH_LAYERS:
                indices = ROUTER_TID[li][CURRENT_IDS.reshape(-1)]
            else:
                indices = torch.topk(scores + ROUTER_BIAS[li], TOP_K, dim=-1).indices
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * ROUTED_SCALE

            # load ONLY the routed experts (6-10, not all 256)
            sel = indices.unique().tolist()
            experts = load_selected_experts(li, sel)

            sh = shared_cache[li]
            out = (F.silu((flat @ sh['w1'].T).clamp(max=SWIGLU_LIMIT))
                   * (flat @ sh['w3'].T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)) @ sh['w2'].T

            # per-expert: x_k (input tokens routed to k) + y_k (original output)
            acts = {}
            for k in indices.unique().tolist():
                pos = (indices == k).any(dim=1).nonzero().flatten()  # [n_k] flat indices
                x_k = flat[pos]                 # [n_k, 4096]
                w1, w2, w3 = experts[k]
                y_k = ffn(x_k, w1, w2, w3)      # [n_k, 4096] original output
                acts[str(k)] = (x_k.detach().cpu(), y_k.detach().cpu())
                for kk in range(TOP_K):
                    m = indices[:, kk] == k      # [N] bool
                    if m.any():
                        m_idx = m.nonzero().flatten()          # [n_kk] flat indices
                        rel = torch.searchsorted(pos, m_idx)    # position in y_k
                        out[m_idx] += weights[m_idx, kk, None] * y_k[rel]
            torch.save(acts, os.path.join(OUT, f'acts_layer{li}.pt'))
            del experts, acts
            print(f'  layer {li}: {len(indices.unique())} experts hit, '
                  f'{time.time()-t0:.1f}s', flush=True)
            return out.to(x.dtype).reshape(B, S, D)
        return hook

    handles = [model.model.layers[li].mlp.register_forward_hook(make_hook(li), with_kwargs=True)
               for li in range(N_LAYERS)]

    input_ids = torch.tensor([ids], device='cuda', dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)
    print(f'forward done in {time.time()-t0:.1f}s', flush=True)
    for h in handles:
        h.remove()

    torch.save(torch.cat(global_x, dim=0), os.path.join(OUT, 'x_global.pt'))
    print(f'saved x_layer/acts_layer (0..{N_LAYERS-1}) + x_global to {OUT}', flush=True)


if __name__ == '__main__':
    main()
