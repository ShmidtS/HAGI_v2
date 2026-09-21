"""Compare ONE binary-refit expert vs ORIGINAL FP4 expert on its real vocab activations.

Runs original:  y_orig = ffn(x, w_orig)
Runs copy:      z = (x - mu) @ P ; y_copy = ffn(z, w_bin)
Reports residual (normalized MSE) vs the collected ground-truth y_true.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dsv4_experts as de
import torch
import torch.nn.functional as F

L = 0
K = 0

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REDUCED = os.path.join(ROOT, "dsv4_reduced")
POD = os.path.join(ROOT, "checkpoints_dsv4", "pod_all_tokens")


def resid(yp, yt):
    return ((yp - yt).pow(2).sum() / yt.pow(2).sum()).item()


def main():
    torch.cuda.empty_cache()
    free0 = torch.cuda.mem_get_info()[0] / 2**30

    # original FP4 expert
    w1o, w2o, w3o = de.load_selected_experts(L, [K])[K]  # fp32 cuda

    # binary expert (packed -> unpack -> *scale)
    e = torch.load(os.path.join(REDUCED, f"layer_{L}", f"expert_{K}.pt"), map_location="cuda", weights_only=False)
    w1q = de.unpack_binary(e["w1"].cuda()).float() * e["w1_scale"].cuda().float()[:, None]
    w3q = de.unpack_binary(e["w3"].cuda()).float() * e["w3_scale"].cuda().float()[:, None]
    w2q = de.unpack_binary(e["w2"].cuda()).float() * e["w2_scale"].cuda().float()[:, None]

    # activations
    acts = torch.load(os.path.join(POD, f"acts_layer{L}.pt"), map_location="cuda", weights_only=False)
    x, y_true = acts[str(K)]
    x = x.float().cuda()
    y_true = y_true.float().cuda()
    n = x.shape[0]

    # original forward
    y_orig = de.ffn(x, w1o, w2o, w3o)

    # binary forward via POD
    P = torch.load(os.path.join(REDUCED, f"layer_{L}", "P.pt"), map_location="cuda").float()
    mu = torch.load(os.path.join(REDUCED, f"layer_{L}", "mu.pt"), map_location="cuda").float()
    z = (x - mu) @ P
    y_copy = de.ffn(z, w1q, w2q, w3q)

    r_orig = resid(y_orig, y_true)
    r_copy = resid(y_copy, y_true)
    cos = F.cosine_similarity(y_copy.flatten().unsqueeze(0), y_true.flatten().unsqueeze(0)).item()

    free1 = torch.cuda.mem_get_info()[0] / 2**30
    print(f"layer {L} expert {K}  (n={n})")
    print(f"  resid(original vs y_true) = {r_orig * 100:.4f}%")
    print(f"  resid(binary    vs y_true) = {r_copy * 100:.4f}%")
    print(f"  stored residual           = {e['residual'] * 100:.4f}%")
    print(f"  cosine(binary, y_true)    = {cos:.4f}")
    print(f"  GPU free: {free0:.1f} -> {free1:.1f} GB (delta {free0 - free1:.2f} GB)")


if __name__ == "__main__":
    main()
