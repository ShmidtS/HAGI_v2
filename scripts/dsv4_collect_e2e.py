"""E2E-calibrated activation collection for layer L_HOOK (memory-bounded v2).

Design (after the v1 OOM crash - load_selected_experts dequantized up to
256 experts x fp32 = ~13GB per layer, unbounded):
  - layers 0..L_HOOK-1 run COMPRESSED via the generator's bounded LRU caches
  - the hook at L_HOOK captures (X, routing indices) and ABORTS the forward
    (custom exception) - layers > L_HOOK never compute
  - y_k per expert computed OFFLINE, one expert at a time (bounded)
Output: checkpoints_dsv4/pod_e2e/acts_layer{L}.pt  {k: (x_k, y_k)} bf16.
"""
from __future__ import annotations

import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import stub_import_tf  # noqa: F401  (must precede transformers)
import dsv4_experts as de
import dsv4_generate_ttt as gt  # bounded caches + decode helpers
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

L_HOOK = 5
VOCAB = 129280
OUT = "checkpoints_dsv4/pod_e2e"
CAPTURE: dict[str, list] = {"x": [], "idx": []}


class AbortForward(Exception):
    pass


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    os.makedirs(OUT, exist_ok=True)

    model = gt.setup_model()  # router + bf16 skeleton, bounded

    import torch.nn.functional as F

    def make_hook(li):
        def hook(module, args_, kwargs, output):
            x = args_[0]
            B, S, D = x.shape
            flat = x.reshape(-1, D).float()
            logits = flat @ gt.ROUTER_W[li].T
            scores = F.softplus(logits).sqrt()
            if li in gt.HASH_LAYERS:
                indices = gt.ROUTER_TID[li][gt.CURRENT_IDS.reshape(-1)]
            else:
                indices = torch.topk(scores + gt.ROUTER_BIAS[li], gt.TOP_K, dim=-1).indices
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * gt.ROUTED_SCALE
            sw = gt.get_shared_dequant(li)
            flatb = flat.to(torch.bfloat16)
            out = gt.ffn(flatb, sw["w1"], sw["w2"], sw["w3"]).float()
            for k in indices.unique().tolist():
                d = gt.get_int4x(li, k)  # bounded LRU
                if d is None:
                    raise RuntimeError(f"L{li} k{k}: no i1i4 ckpt")
                m_any = (indices == k).any(dim=1)
                h = gt.int4x_forward(d, flat[m_any])
                out_k = (h.to(torch.bfloat16) @ d["w2b"].T).float()
                pos = torch.cumsum(m_any.long(), 0) - 1
                for kk in range(gt.TOP_K):
                    m = indices[:, kk] == k
                    if m.any():
                        out[m] += weights[m, kk, None] * out_k[pos[m]]
                del h, out_k, m_any
            if li == L_HOOK - 1:
                # NEXT layer's input == this layer's output already computed
                # by the transformer between hooks; capture at L_HOOK instead.
                pass
            return out.to(x.dtype).reshape(B, S, D)
        return hook

    def make_capture_hook(li):
        def hook(module, args_, kwargs, output):
            x = args_[0]
            B, S, D = x.shape
            flat = x.reshape(-1, D).float()
            logits = flat @ gt.ROUTER_W[li].T
            scores = F.softplus(logits).sqrt()
            if li in gt.HASH_LAYERS:
                indices = gt.ROUTER_TID[li][gt.CURRENT_IDS.reshape(-1)]
            else:
                indices = torch.topk(scores + gt.ROUTER_BIAS[li], gt.TOP_K, dim=-1).indices
            CAPTURE["x"].append(flat.detach().cpu().to(torch.bfloat16))
            CAPTURE["idx"].append(indices.detach().cpu())
            raise AbortForward
        return hook

    handles = [model.model.layers[li].mlp.register_forward_hook(make_hook(li), with_kwargs=True) for li in range(L_HOOK)]
    handles.append(model.model.layers[L_HOOK].mlp.register_forward_hook(make_capture_hook(L_HOOK), with_kwargs=True))

    B = 4
    SEQ = 256
    N_TOKENS = 32768
    # EXACTLY the original collector's stream: shuffled vocab sequences
    torch.manual_seed(1234)
    stream = torch.cat([torch.randperm(VOCAB, dtype=torch.long) for _ in range(1)])
    n_rand = max(0, N_TOKENS - stream.numel())
    stream = torch.cat([stream, torch.randint(0, VOCAB, (n_rand,), dtype=torch.long)])
    n_batches = stream.numel() // (SEQ * B)
    t0 = time.time()
    for bi in range(n_batches):
        chunk = stream[bi * SEQ * B:(bi + 1) * SEQ * B]
        ids = chunk.view(B, SEQ).to("cuda")
        gt.CURRENT_IDS = ids
        try:
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
        except AbortForward:
            pass
        if (bi + 1) % 8 == 0:
            print(f"batch {bi+1}/{n_batches} ({time.time()-t0:.0f}s, x-rows={sum(t.shape[0] for t in CAPTURE['x'])})", flush=True)
    for h in handles:
        h.remove()

    X = torch.cat(CAPTURE["x"])            # [N, 4096] bf16
    IDX = torch.cat(CAPTURE["idx"])        # [N, 6]
    print(f"captured X {tuple(X.shape)}, routing {tuple(IDX.shape)} in {time.time()-t0:.0f}s", flush=True)
    torch.save({"x": X, "idx": IDX}, os.path.join(OUT, f"cap_layer{L_HOOK}.pt"))

    # offline: y_k per expert, one expert at a time (bounded)
    snap = de.default_snapshot()
    acts = {}
    fp_layer = os.path.join(de.LOSSLESS, f"layers_{L_HOOK}_ffn.safetensors")
    t1 = time.time()
    uniq = sorted(set(IDX.reshape(-1).tolist()))
    for n, k in enumerate(uniq):
        E = de.load_expert_file(fp_layer, f"layers.{L_HOOK}.ffn", k)
        m = (IDX == k).any(dim=1)
        x_k = X[m].float().cuda()
        y_k = de.ffn(x_k, E["w1"].cuda(), E["w2"].cuda(), E["w3"].cuda()).cpu().to(torch.bfloat16)
        acts[str(k)] = (X[m], y_k)
        del E, x_k, y_k
        if (n + 1) % 64 == 0:
            print(f"  y-compute {n+1}/{len(uniq)} ({time.time()-t1:.0f}s)", flush=True)
    torch.save(acts, os.path.join(OUT, f"acts_layer{L_HOOK}.pt"))
    tot = sum(v[0].shape[0] for v in acts.values())
    print(f"DONE: {len(acts)} experts, {tot} rows -> acts_layer{L_HOOK}.pt (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
