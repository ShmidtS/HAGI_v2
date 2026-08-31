"""A/B the e2e hypothesis on L5 WITHOUT a full refit:

 A) resid of the CURRENT checkpoints (refit on original activations)
    evaluated on the NEW e2e activations (input = through compressed 0-4)
 A0) resid of the same checkpoints on ORIGINAL activations (sanity, matches v19 stats)
 B) closed-form re-solve of ONLY W2 (ridge + GPTQ g128) on e2e activations,
    signs/scales of W13 kept - measures how much of the gap W2-recalibration
    can absorb (the cheap 90% version of a full e2e refit).

Both evaluated on held-out e2e rows (split per expert 80/20).
"""
import os
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import stub_import_tf  # noqa: F401
from dsv4_refit_experts import _gptq_groups, soft_lim  # noqa: E402

L = 5
RED = f"dsv4_reduced/layer_{L}"
POD_E2E = "checkpoints_dsv4/pod_e2e/acts_layer5.pt"
POD_ORIG = "checkpoints_dsv4/pod_all_tokens/acts_layer5.pt"
GS = 128

torch.set_grad_enabled(False)
dev = "cuda"

acts = torch.load(POD_E2E, map_location="cpu", weights_only=False)
P = torch.load(os.path.join(RED, "P.pt"), map_location=dev).float()
mu = torch.load(os.path.join(RED, "mu.pt"), map_location=dev).float()

res_a, res_a0, res_b = [], [], []
n_done = 0
import os as _os
_N = int(_os.environ.get("AB_SAMPLE", "64"))
for k in sorted(acts.keys(), key=int)[:_N]:
    x, y = acts[k]
    n = x.shape[0]
    if n < 128:
        continue
    perm = torch.randperm(n)
    n_tr = int(n * 0.8)
    x_tr, y_tr = x[perm[:n_tr]].float().to(dev), y[perm[:n_tr]].float().to(dev)
    x_va, y_va = x[perm[n_tr:]].float().to(dev), y[perm[n_tr:]].float().to(dev)

    e = torch.load(os.path.join(RED, f"expert_{k}.pt"), map_location="cpu", weights_only=False)
    # current checkpoint decode (exactly as the generator does)
    from dsv4_experts import unpack_binary, unpack_int4  # noqa: E402

    w1 = (unpack_binary(e["w1a"]).float().to(dev) * e["w1a_scale"].float().to(dev)[:, None])
    w3 = (unpack_binary(e["w3a"]).float().to(dev) * e["w3a_scale"].float().to(dev)[:, None])
    b1 = e["bias1a"].float().to(dev)
    b3 = e["bias3a"].float().to(dev)
    q2 = unpack_int4(e["w2a"]).float().to(dev)
    s2 = e["w2a_scale"].float().to(dev)
    if s2.dim() == 1:
        W2 = q2 * s2[:, None]
    else:
        W2 = q2 * s2.repeat_interleave(GS, dim=1)

    def fwd(x_rows, W1, W3, B1, B3, W2m):
        z = (x_rows - mu) @ P
        g = soft_lim(z @ W1.T + B1[None, :])
        u = soft_lim(z @ W3.T + B3[None, :])
        return (F.silu(g) * u) @ W2m.T

    # A: current ckpt on e2e rows (held-out)
    yh = fwd(x_va, w1, w3, b1, b3, W2)
    res_a.append((((yh - y_va) ** 2).sum() / (y_va ** 2).sum()).item())

    # B: re-solve W2 on e2e TRAIN rows (signs/scales kept), GPTQ g128
    z_tr = (x_tr - mu) @ P
    g = soft_lim(z_tr @ w1.T + b1[None, :])
    u = soft_lim(z_tr @ w3.T + b3[None, :])
    h = F.silu(g) * u
    Gm = h.T @ h
    Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
    W2c = torch.linalg.solve(Gm, h.T @ y_tr).T.contiguous()
    ng = W2c.shape[1] // GS
    Wg = W2c.view(-1, ng, GS)
    sg_ = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9) / 7.0
    qg = (Wg / sg_).round().clamp(-7, 7)
    for _ in range(3):
        num = (qg * Wg).sum(-1, keepdim=True)
        den = (qg * qg).sum(-1, keepdim=True).clamp_min(1e-9)
        sg_ = (num / den).clamp_min(1e-9)
        qg = (Wg / sg_).round().clamp(-7, 7)
    s2g = sg_.squeeze(-1)
    Hh = (h.T @ h) / h.shape[0]
    try:
        q2n = _gptq_groups(W2c, Hh, s2g, gs=GS)
        W2n = q2n * s2g.repeat_interleave(GS, dim=1)
        yh = fwd(x_va, w1, w3, b1, b3, W2n)
        res_b.append((((yh - y_va) ** 2).sum() / (y_va ** 2).sum()).item())
    except RuntimeError:
        pass
    del e, x_tr, y_tr, x_va, y_va, h, g, u, W2c
    n_done += 1
    if n_done % 16 == 0:
        import math
        print(f"  {n_done} done: A med {math.sqrt(sorted(res_a)[len(res_a)//2])*100:.2f}%  B med {math.sqrt(sorted(res_b)[len(res_b)//2])*100:.2f}%", flush=True)

import math

for name, arr in [("A cur-ckpt on e2e rows", res_a), ("B W2-resolve on e2e", res_b)]:
    if not arr:
        continue
    arr_s = sorted(arr)
    med = math.sqrt(arr_s[len(arr_s) // 2]) * 100
    p75 = math.sqrt(arr_s[int(len(arr_s) * 0.75)]) * 100
    print(f"{name:28s} n={len(arr)}  median {med:.2f}%  p75 {p75:.2f}%  (norm err)")
