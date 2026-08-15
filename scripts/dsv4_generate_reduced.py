"""Generate with the full REDUCED model: skeleton + reduced ternary experts (all 43 layers).

Loads the compact skeleton (correct non-expert weights), replaces each MoE
block's output via forward hooks with the reduced experts (POD input proj +
ternary SwiGLU + int8 output basis Q) computed from dsv4_reduced/, and decodes
autoregressively with KV-cache.

Usage:
    python scripts/dsv4_generate_reduced.py "<prompt>" [max_new_tokens]
"""
from __future__ import annotations

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
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
import dsv4_experts as de
import gigatoken

MODEL_DIR = 'C:/HAGI_v2/dsv4_shared_only'
TOKENIZER = r'C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json'
LOSSLESS = 'C:/HAGI_v2/lossless_layers'
REDUCED = 'C:/HAGI_v2/dsv4_reduced'

N_LAYERS = 43
HASH_LAYERS = {0, 1, 2}
TOP_K = 6
ROUTED_SCALE = 1.5
SWIGLU_LIMIT = 10.0
BOS_ID = 0
EOS_ID = 1
K = 512
Kp = 8
INTER = 128

CURRENT_IDS: torch.Tensor | None = None
ROUTER_W: dict[int, torch.Tensor] = {}
ROUTER_BIAS: dict[int, torch.Tensor] = {}
ROUTER_TID: dict[int, torch.Tensor] = {}


def unpack_ternary(q: torch.Tensor) -> torch.Tensor:
    t = q.to(torch.int32)
    out, n = t.shape
    trits = torch.zeros(out, n * 5, dtype=torch.int32, device=t.device)
    for i in range(5):
        trits[:, i::5] = (t // (3 ** i)) % 3
    return trits - 1


def load_reduced_layer(li: int) -> dict:
    P = torch.load(os.path.join(REDUCED, f'layer_{li}', 'P.pt'), map_location='cuda')
    mu_path = os.path.join(REDUCED, f'layer_{li}', 'mu.pt')
    mu = torch.load(mu_path, map_location='cuda') if os.path.exists(mu_path) else None
    experts = {}
    for k in range(256):
        e = torch.load(os.path.join(REDUCED, f'layer_{li}', f'expert_{k}.pt'), map_location='cuda')
        experts[k] = {
            'w1': e['w1'], 'w1s': e['w1_scale'].to('cuda'),
            'w3': e['w3'], 'w3s': e['w3_scale'].to('cuda'),
            'w2': e['w2'], 'w2s': e['w2_scale'].to('cuda'),
            'Q': e['Q'].to('cuda'), 'Qs': e['Q_scale'].to('cuda'),
        }
    return {'P': P, 'mu': mu, 'experts': experts}


def reduced_ffn_z(z, e):
    zz = z.to(torch.bfloat16)
    w1 = unpack_ternary(e['w1'])[:, :K].to(torch.bfloat16) * e['w1s'].to(torch.bfloat16)[:, None]
    w3 = unpack_ternary(e['w3'])[:, :K].to(torch.bfloat16) * e['w3s'].to(torch.bfloat16)[:, None]
    w2 = unpack_ternary(e['w2'])[:, :INTER].to(torch.bfloat16) * e['w2s'].to(torch.bfloat16)[:, None]
    Q = e['Q'].to(torch.bfloat16) * e['Qs'].to(torch.bfloat16)[None, :]
    gate = (zz @ w1.T).clamp(max=SWIGLU_LIMIT)
    up = (zz @ w3.T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    h = F.silu(gate) * up
    yc = h @ w2.T
    return (yc @ Q.T).float()


def make_reduced_hook(li: int, red_cache: dict, shared_cache: dict):
    def hook(module, args, kwargs, output):
        x = args[0]
        B, S, D = x.shape
        flat = x.reshape(-1, D).float()
        logits = flat @ ROUTER_W[li].T
        scores = F.softplus(logits).sqrt()
        if li in HASH_LAYERS:
            indices = ROUTER_TID[li][CURRENT_IDS.reshape(-1)]
        else:
            indices = torch.topk(scores + ROUTER_BIAS[li], TOP_K, dim=-1).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * ROUTED_SCALE

        sh = shared_cache[li]
        out = (F.silu((flat @ sh['w1'].T).clamp(max=SWIGLU_LIMIT))
               * (flat @ sh['w3'].T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)) @ sh['w2'].T

        red = red_cache[li]
        # Fix A: center before projecting (mu stored at reduce time).
        if red.get('mu') is not None:
            z = (flat - red['mu']) @ red['P']
        else:
            z = flat @ red['P']
        for k in indices.unique().tolist():
            y = reduced_ffn_z(z, red['experts'][k])
            for kk in range(TOP_K):
                m = indices[:, kk] == k
                if m.any():
                    out[m] += weights[m, kk, None] * y[m]
        return out.to(x.dtype).reshape(B, S, D)
    return hook


def load_router(snap: str, wm: dict) -> None:
    for li in range(N_LAYERS):
        p = f'layers.{li}.ffn.gate'
        ROUTER_W[li] = de.read_tensor(snap, wm, f'{p}.weight', device='cuda').to(torch.float32)
        if li in HASH_LAYERS:
            ROUTER_TID[li] = de.read_tensor(snap, wm, f'{p}.tid2eid', device='cuda').to(torch.long)
        else:
            ROUTER_BIAS[li] = de.read_tensor(snap, wm, f'{p}.bias', device='cuda').to(torch.float32)


def load_shared() -> dict:
    shared = {}
    for li in range(N_LAYERS):
        fp = os.path.join(LOSSLESS, f'layers_{li}_ffn.safetensors')
        prefix = f'layers.{li}.ffn'
        sh = de.load_shared_file(fp, prefix, device='cuda')
        shared[li] = {kk: sh[kk].float() for kk in ('w1', 'w2', 'w3')}
    return shared


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'Hello'
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    tok = gigatoken.Tokenizer.from_json(open(TOKENIZER, 'rb').read())
    ids = [BOS_ID] + list(tok.encode(prompt))
    print(f'prompt ids ({len(ids)})', flush=True)

    snap = de.default_snapshot()
    wm = de.load_index(snap)['weight_map']
    load_router(snap, wm)
    print('loading shared experts...', flush=True)
    shared_cache = load_shared()
    print('loading reduced experts (all layers)...', flush=True)
    t0 = time.time()
    red_cache = {}
    for li in range(N_LAYERS):
        red_cache[li] = load_reduced_layer(li)
    print(f'reduced experts loaded in {time.time()-t0:.1f}s', flush=True)

    torch.set_default_device('cuda')
    model = DeepseekV4ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    torch.set_default_device('cpu')
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = 'eager'
    model.config.gradient_checkpointing = False

    handles = [model.model.layers[li].mlp.register_forward_hook(make_reduced_hook(li, red_cache, shared_cache), with_kwargs=True)
               for li in range(N_LAYERS)]

    input_ids = torch.tensor([ids], device='cuda', dtype=torch.long)
    global CURRENT_IDS
    generated = list(ids)
    past = None
    with torch.no_grad():
        CURRENT_IDS = input_ids
        out = model(input_ids=input_ids, use_cache=True, past_key_values=None)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        generated.append(nxt)
        for step in range(max_new - 1):
            if nxt == EOS_ID:
                break
            CURRENT_IDS = torch.tensor([[nxt]], device='cuda', dtype=torch.long)
            out = model(input_ids=CURRENT_IDS, use_cache=True, past_key_values=past)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax().item())
            generated.append(nxt)
    for h in handles:
        h.remove()
    text = tok.decode(generated)
    print(f'OUTPUT: {text}', flush=True)


if __name__ == '__main__':
    main()
