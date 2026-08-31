"""Honest A/B eval on layer 1: residual of a checkpoint variant measured on
DRIFTED rows (collected through compressed layer 0) vs the FP4 teacher.

Usage: python scripts/eval_ab_layer1.py <variant_dir> [n_experts]
Variant dirs: dsv4_reduced_ab/{clean8k, drift8k, orig}.
Copies variant -> dsv4_reduced/layer_1, evaluates, restores nothing (caller
manages), prints per-expert residuals + summary.
"""
import os
import shutil
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stub_import_tf  # noqa: F401
import dsv4_generate_ttt as gen

variant = sys.argv[1]
n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 64

L = 1
# place variant checkpoints
dst = f"dsv4_reduced/layer_{L}"
tmp = "dsv4_reduced_ab/_staged"
if os.path.exists(tmp):
    shutil.rmtree(tmp)
shutil.copytree(dst, tmp)
for f in os.listdir(variant):
    if f.startswith("expert_") and f.endswith(".pt"):
        shutil.copy(os.path.join(variant, f), dst)

gen.INT4X.clear()
gen.PACKED_CACHE.clear()

acts = torch.load("checkpoints_dsv4/seq8k_drift/acts_layer1.pt", map_location="cpu", weights_only=False)
# hottest experts first (most rows = most impact), then fill with the rest
order = sorted(acts.keys(), key=lambda k: -acts[k][0].shape[0])
sample = order[:n_sample]

res = []
with torch.no_grad():
    for k in sample:
        x_k, _ = acts[k]
        x = x_k.float().cuda()
        d = gen.get_int4x(L, int(k))
        h = gen.int4x_forward(d, x)
        y_hat = (h.to(torch.bfloat16) @ d["w2b"].T).float()
        y_t = gen._teacher_y(L, int(k), x)
        r = ((y_hat - y_t).norm() / y_t.norm()).item() * 100
        res.append((int(k), x.shape[0], r))

hot = [r for _, n, r in res if n >= 1024]
allr = [r for _, _, r in res]
print(f"variant={variant}")
print(f"  n={len(res)}  median={statistics.median(allr):.2f}%  mean={statistics.mean(allr):.2f}%  max={max(allr):.2f}%")
if hot:
    print(f"  hot(n>=1024): {len(hot)}  median={statistics.median(hot):.2f}%  max={max(hot):.2f}%")

# restore staged original dir content
for f in os.listdir(dst):
    os.remove(os.path.join(dst, f))
for f in os.listdir(tmp):
    shutil.copy(os.path.join(tmp, f), dst)
shutil.rmtree(tmp)
