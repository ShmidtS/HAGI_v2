"""Smoke: unifold (±5σ) через одного эксперта — замерить residual сходимости."""
import os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv4_refit_experts as r
from dsv4_collect_x_accurate import load_selected_experts, ffn as ffn_exact

ap = argparse.ArgumentParser()
ap.add_argument('--layer', type=int, default=0)
ap.add_argument('--expert', type=int, default=3)
ap.add_argument('--steps', type=int, default=1000)
ap.add_argument('--inter', type=int, default=2048)
ap.add_argument('--kp', type=int, default=512)
args = ap.parse_args()

L, k = args.layer, args.expert
P = torch.load(os.path.join(r.REDUCED, f'layer_{L}', 'P.pt'), map_location='cuda').float()
mu = torch.load(os.path.join(r.REDUCED, f'layer_{L}', 'mu.pt'), map_location='cuda').float()
acts = torch.load(os.path.join(r.POD, f'acts_layer{L}.pt'), map_location='cpu', weights_only=False)
print(f'layer {L}: {len(acts)} experts in acts; expert {k} present={str(k) in acts}', flush=True)

zs = []
for x_k, _ in acts.values():
    if x_k.shape[0] > 0:
        zs.append((x_k.float().cuda() - mu) @ P)
del acts
torch.cuda.empty_cache()
z_pool = torch.cat(zs, dim=0)
global_sigma = z_pool.std(dim=0, unbiased=False).clamp(0.1, 2.0)
print(f'z_pool {tuple(z_pool.shape)}  sigma mean={global_sigma.mean().item():.3f} '
      f'median={global_sigma.median().item():.3f}', flush=True)

z_uni = r.universal_signal(torch.zeros(0, r.K, device='cuda'),
                           sigma_override=global_sigma, z_proxy=z_pool)
x_uni = (mu + z_uni @ P.T).float()
print(f'unifold z {tuple(z_uni.shape)} (min={z_uni.min().item():.2f} max={z_uni.max().item():.2f})', flush=True)

experts = load_selected_experts(L, [k])
w1, w2, w3 = experts[k]
y_uni = ffn_exact(x_uni, w1, w2, w3)
del experts
torch.cuda.empty_cache()
print(f'y_uni {tuple(y_uni.shape)}', flush=True)

Q0 = r.safe_svd_q(y_uni, args.kp)
if Q0.shape[1] < args.kp:
    Q0 = torch.cat([Q0, torch.zeros(r.D, args.kp - Q0.shape[1], device='cuda')], dim=1)

t0 = time.time()
results = r.train_batch([(z_uni, y_uni, Q0[:, :args.kp])], args.inter, args.steps,
                        kp=args.kp, stop_threshold=None, n_real=0)
w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs = results[0]
resid = r.resid_weights_full(z_uni, y_uni, Qq, Qs, w1q, w1s, w3q, w3s, w2q, w2s)
print(f'FINAL resid={resid*100:.4f}%  ({time.time()-t0:.0f}s)', flush=True)
