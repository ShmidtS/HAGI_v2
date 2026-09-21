"""Collect per-expert activations over the FULL vocab + random token stream.

Feeds every token id in the vocabulary (129280) plus random uniform sequences
and accumulates per-expert activations, flushed to disk in small batches so GPU
and RAM stay bounded.

Memory design (GPU is the bottleneck on this APU):
  - skeleton model loaded in bf16 (not fp32)
  - shared FFN weights kept on CPU in bf16, moved to GPU per layer inside the hook
  - per-expert (x_k, y_k) flushed to part files every --flush-batches forwards
  - each expert is capped at --cap samples (rare experts get honest coverage)

Token stream:
  1. every token id in vocab, in order (--no-vocab to skip)
  2. random uniform sequences to fill the rest of --max-tokens

Output (checkpoints_dsv4/pod_all_tokens/):
  acts_layer{L}.pt   {k: (x_k, y_k)}  (bf16, merged from part files)
  x_layer{L}.pt      [N, 4096]         (bf16, merged from part files)

Usage: python scripts/dsv4_collect_all_tokens.py [--max-tokens 258560]
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
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import dsv4_experts as de
import gigatoken
from dsv4_experts import (
    HASH_LAYERS,
    LOSSLESS,
    N_LAYERS,
    ROUTED_SCALE,
    ROUTER_BIAS,
    ROUTER_TID,
    ROUTER_W,
    SWIGLU_LIMIT,
    TOP_K,
    ffn,
    load_router,
    load_selected_experts,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

import stub_import_tf  # noqa: F401

MODEL_DIR = "C:/HAGI_v2/dsv4_shared_only"
TOKENIZER = r"C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json"

OUT = "checkpoints_dsv4/pod_all_tokens"
SEQ = 256
BATCH = 4  # tokens per forward = SEQ * BATCH = 1024
VOCAB = 129280


def load_shared_gpu():
    """Load shared FFN weights for all layers onto GPU in bf16."""
    shared = {}
    for li in range(N_LAYERS):
        fp = os.path.join(LOSSLESS, f"layers_{li}_ffn.safetensors")
        prefix = f"layers.{li}.ffn"
        sh = de.load_shared_file(fp, prefix, device="cpu")
        shared[li] = {kk: sh[kk].to(torch.bfloat16).to("cuda") for kk in ("w1", "w2", "w3")}
    return shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=VOCAB * 2, help="total token budget (default: vocab + 1x random)")
    ap.add_argument("--cap", type=int, default=2048, help="max samples kept per expert")
    ap.add_argument("--flush-batches", type=int, default=2, help="flush accumulators to disk every N forward passes")
    ap.add_argument("--no-vocab", action="store_true", help="skip ordered vocab pass")
    args = ap.parse_args()

    tok = gigatoken.Tokenizer.from_json(open(TOKENIZER, "rb").read())

    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]
    load_router(snap, wm)
    print("loading shared experts (GPU bf16)...", flush=True)
    shared_gpu = load_shared_gpu()

    torch.set_default_device("cuda")
    model = DeepseekV4ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    torch.set_default_device("cpu")
    model.eval()
    model.config._experts_implementation = "eager"
    model.config.gradient_checkpointing = False
    model = model.to(torch.bfloat16)
    print("skeleton ready (bf16)", flush=True)

    os.makedirs(OUT, exist_ok=True)

    stream = []
    if not args.no_vocab:
        # uniformly shuffled vocab (all manifolds mixed) — matches POD coverage goal
        stream.append(torch.randperm(VOCAB, dtype=torch.long))
    n_rand = max(0, args.max_tokens - sum(int(s.numel()) for s in stream))
    if n_rand > 0:
        stream.append(torch.randint(0, VOCAB, (n_rand,), dtype=torch.long))
    stream = torch.cat(stream)[: args.max_tokens]
    n_tokens = stream.numel()
    print(f"token stream: {n_tokens} tokens (vocab + random)", flush=True)

    # per-layer accumulators (CPU bf16 parts), cleared on flush
    ACC = {li: {str(k): ([], []) for k in range(256)} for li in range(N_LAYERS)}
    X_ACC = {li: [] for li in range(N_LAYERS)}
    COUNTS = {li: {str(k): 0 for k in range(256)} for li in range(N_LAYERS)}
    CUR = {"ids": None}
    PART = [0]

    def make_hook(li):
        def hook(module, args_, kwargs, output):
            x = args_[0]
            B, S, D = x.shape
            flat = x.reshape(-1, D).float()
            X_ACC[li].append(flat.detach().cpu().to(torch.bfloat16))

            logits = flat @ ROUTER_W[li].T
            scores = F.softplus(logits).sqrt()
            if li in HASH_LAYERS:
                indices = ROUTER_TID[li][CUR["ids"].reshape(-1)]
            else:
                indices = torch.topk(scores + ROUTER_BIAS[li], TOP_K, dim=-1).indices
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * ROUTED_SCALE

            sel = indices.unique().tolist()
            experts = load_selected_experts(li, sel)
            sh = {kk: v.float() for kk, v in shared_gpu[li].items()}

            out = (
                F.silu((flat @ sh["w1"].T).clamp(max=SWIGLU_LIMIT))
                * (flat @ sh["w3"].T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            ) @ sh["w2"].T

            for k in indices.unique().tolist():
                key = str(k)
                pos = (indices == k).any(dim=1).nonzero().flatten()
                x_k = flat[pos]
                w1, w2, w3 = experts[k]
                y_k = ffn(x_k, w1, w2, w3)
                if COUNTS[li][key] < args.cap:
                    take = min(args.cap - COUNTS[li][key], pos.numel())
                    ACC[li][key][0].append(x_k[:take].detach().cpu().to(torch.bfloat16))
                    ACC[li][key][1].append(y_k[:take].detach().cpu().to(torch.bfloat16))
                    COUNTS[li][key] += take
                for kk in range(TOP_K):
                    m = indices[:, kk] == k
                    if m.any():
                        m_idx = m.nonzero().flatten()
                        rel = torch.searchsorted(pos, m_idx)
                        out[m_idx] += weights[m_idx, kk, None] * y_k[rel]
            del experts, sh
            return out.to(x.dtype).reshape(B, S, D)

        return hook

    handles = [
        model.model.layers[li].mlp.register_forward_hook(make_hook(li), with_kwargs=True) for li in range(N_LAYERS)
    ]

    def flush():
        PART[0] += 1
        p = PART[0]
        for li in range(N_LAYERS):
            acts = {k: (torch.cat(v[0]), torch.cat(v[1])) for k, v in ACC[li].items() if v[0]}
            if acts:
                torch.save(acts, os.path.join(OUT, f"acts_layer{li}_part{p}.pt"))
                ACC[li] = {str(k): ([], []) for k in range(256)}
            if X_ACC[li]:
                torch.save(torch.cat(X_ACC[li], dim=0), os.path.join(OUT, f"x_layer{li}_part{p}.pt"))
                X_ACC[li] = []

    t0 = time.time()
    n_batches = (n_tokens + SEQ * BATCH - 1) // (SEQ * BATCH)
    for bi in range(n_batches):
        s = bi * SEQ * BATCH
        e = min(s + SEQ * BATCH, n_tokens)
        chunk = stream[s:e]
        if chunk.numel() == 0:
            continue
        pad = SEQ * BATCH - chunk.numel()
        if pad:
            chunk = torch.cat([chunk, torch.zeros(pad, dtype=torch.long)])
        input_ids = chunk.view(BATCH, SEQ).to("cuda")
        CUR["ids"] = input_ids
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=False)
        if (bi + 1) % args.flush_batches == 0 or bi == n_batches - 1:
            flush()
            covered = sum(1 for li in range(N_LAYERS) for c in COUNTS[li].values() if c > 0)
            print(f"  {e}/{n_tokens} tokens, {covered}/11008 experts covered, {time.time() - t0:.0f}s", flush=True)

    for h in handles:
        h.remove()

    print("merging parts...", flush=True)
    total = 0
    for li in range(N_LAYERS):
        merged = {}
        for f in sorted(os.listdir(OUT)):
            if f.startswith(f"acts_layer{li}_part") and f.endswith(".pt"):
                d = torch.load(os.path.join(OUT, f), map_location="cpu", weights_only=False)
                for k, (x, y) in d.items():
                    if k in merged:
                        mx, my = merged[k]
                        merged[k] = (torch.cat([mx, x]), torch.cat([my, y]))
                    else:
                        merged[k] = (x, y)
        if merged:
            torch.save(merged, os.path.join(OUT, f"acts_layer{li}.pt"))
            total += sum(x.shape[0] for x, _ in merged.values())
            print(f"  layer {li}: {len(merged)} experts", flush=True)
        xm = [
            torch.load(os.path.join(OUT, f), map_location="cpu", weights_only=False)
            for f in sorted(os.listdir(OUT))
            if f.startswith(f"x_layer{li}_part") and f.endswith(".pt")
        ]
        if xm:
            torch.save(torch.cat(xm, dim=0), os.path.join(OUT, f"x_layer{li}.pt"))
    print(f"done: {total} total samples in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
