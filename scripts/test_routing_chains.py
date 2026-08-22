"""Expert transition chains from SAVED activations (no model forward).

Uses checkpoints_dsv4/pod_all_tokens/x_layer{L}.pt (per-layer MLP inputs,
258560 tokens) + router weights. Computes top-k indices per layer per token,
builds directed transition counts T[li][next, cur], compares to chance
(36/256 = 0.1406 directed pairs per token for two random 6-of-256 sets).

Hash layers 0-2 excluded from analysis (routing is token-id based there).

Output: checkpoints_dsv4/routing_chains.pt {li: indices int16 [N, TOP_K]}
Usage:  python scripts/test_routing_chains.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import dsv4_experts as de
from dsv4_experts import N_LAYERS, ROUTER_BIAS, ROUTER_W, TOP_K, load_router

import stub_import_tf  # noqa: F401

ACTS = "checkpoints_dsv4/pod_all_tokens"
OUT = "checkpoints_dsv4/routing_chains.pt"
K = 256


def main():
    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]
    load_router(snap, wm)

    all_idx = {}
    for li in range(N_LAYERS):
        x = torch.load(os.path.join(ACTS, f"x_layer{li}.pt"), map_location="cuda", weights_only=False).float()
        logits = x @ ROUTER_W[li].T
        scores = F.softplus(logits).sqrt()
        idx = torch.topk(scores + ROUTER_BIAS[li], TOP_K, dim=-1).indices
        all_idx[li] = idx.to(torch.int16).cpu()
        del x, logits, scores
        print(f"layer {li}: routed {idx.shape[0]} tokens", flush=True)

    torch.save(all_idx, OUT)

    print("\n=== transition analysis (layers 3+, topk routing) ===", flush=True)
    exp_one = TOP_K * TOP_K / K  # 0.1406 chance directed pairs per token
    rows = []
    for li in range(3, N_LAYERS - 1):
        a = all_idx[li].long().cuda()
        b = all_idx[li + 1].long().cuda()
        Tm = torch.zeros(K * K, dtype=torch.float32, device="cuda")
        for kk in range(TOP_K):
            for jj in range(TOP_K):
                Tm += torch.bincount(a[:, kk] * K + b[:, jj], minlength=K * K).float()
        Tm /= a.shape[0]
        mx, pair = Tm.max(dim=0)
        ka, kb = divmod(int(pair.item()), K)
        excess = (Tm - exp_one).clamp(min=0).sum().item()
        frac = excess / Tm.sum().item()
        # chi-square-like: total excess vs sqrt(N) noise scale
        n_tok = a.shape[0]
        noise = (exp_one * TOP_K * n_tok) ** 0.5 / n_tok / TOP_K
        z = (mx.item() - exp_one) / noise
        rows.append((frac, z))
        print(
            f"L{li}->L{li + 1}: mean={Tm.mean().item():.4f} exp={exp_one:.4f} "
            f"max={mx.item():.3f} ({ka}->{kb}) sigma={noise:.4f} "
            f"z={z:.1f} excess_mass={frac * 100:.2f}%",
            flush=True,
        )

    fr = max(r[0] for r in rows)
    zz = max(r[1] for r in rows)
    print(f"\nMAX over layers: excess_mass={fr * 100:.2f}%  z={zz:.1f}")
    if zz < 8:
        print("VERDICT: transitions indistinguishable from random -> no sequential expert pipelines.")
    else:
        print("VERDICT: significant transitions found -> inspect hot pairs.")


if __name__ == "__main__":
    main()
