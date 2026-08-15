"""Sweep inter x KP for the reduced expert with learnable Q + Muon.

Standalone bench, separate from the main dsv4_refit_experts pipeline.
For each (inter, KP) it retrains ternary core + learnable Q from scratch
(warm-started from the top-KP SVD columns) against the full 4096-dim target,
then reports the full-space residual and the packed expert size in MB.

Usage:
  python scripts/dsv4_bench_compress2.py --layer 0 --n-experts 4 --steps 400
"""
import torch, torch.nn.functional as F, os, sys, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv4_reduce_layer import ternarize
from dsv4_refit_experts import qste_bf16, zeropower, quantize_q

K = 512
D = 4096
POD = 'checkpoints_dsv4/pod_accurate'
REDUCED = 'dsv4_reduced'
TRIT_BYTES = 0.2  # 5 trits/byte pack


def size_mb(inter, kp, q_bytes=1):
    w13 = 2 * (inter * K * TRIT_BYTES) / 1024 / 1024
    w2 = (kp * inter * TRIT_BYTES) / 1024 / 1024
    q = (kp * D * q_bytes) / 1024 / 1024
    return w13 + w2 + q


def resid_full(z, y_full, w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs):
    w1 = w1q * w1s[:, None]
    w3 = w3q * w3s[:, None]
    w2 = w2q * w2s[:, None]
    Q = Qq.float() * Qs[None, :]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        g = (z @ w1.T).clamp(max=10.0)
        u = (z @ w3.T).clamp(min=-10.0, max=10.0)
        yp = ((F.silu(g) * u) @ w2.T) @ Q.T
    return (F.mse_loss(yp.float(), y_full) /
            F.mse_loss(y_full, torch.zeros_like(y_full))).item()


def train_sweep(pairs, inter, kp, steps, q_dtype='int8', check_every=100):
    """pairs: [(z[n,K], y[n,D], Q0[D,kp])] -> [(w1q,w1s,w3q,w3s,w2q,w2s,Qq,Qs)]."""
    G = len(pairs)
    Nmax = max(z.shape[0] for z, _, _ in pairs)
    Z = torch.zeros(G, Nmax, K, device='cuda', dtype=torch.bfloat16)
    Y = torch.zeros(G, Nmax, D, device='cuda', dtype=torch.bfloat16)
    Q = torch.zeros(G, D, kp, device='cuda', dtype=torch.float32)
    M = torch.zeros(G, Nmax, 1, device='cuda')
    for i, (z, y, Q0) in enumerate(pairs):
        Z[i, :z.shape[0]] = z.to(torch.bfloat16)
        Y[i, :y.shape[0]] = y.to(torch.bfloat16)
        Q[i] = Q0.float()
        M[i, :z.shape[0]] = 1.0

    W1 = torch.nn.Parameter(torch.randn(G, inter, K, device='cuda') * K**-0.5)
    W3 = torch.nn.Parameter(torch.randn(G, inter, K, device='cuda') * K**-0.5)
    W2 = torch.nn.Parameter(torch.randn(G, kp, inter, device='cuda') * inter**-0.5)
    Qp = torch.nn.Parameter(Q)
    e1 = torch.zeros_like(W1)
    e3 = torch.zeros_like(W3)
    e2 = torch.zeros_like(W2)
    eQ = torch.zeros_like(Qp)
    MU = 0.95
    bs = min(Nmax, 1024)
    t0 = time.time()
    for st in range(steps):
        if Nmax > bs:
            idx = torch.randint(0, Nmax, (bs,), device='cuda')
            Zb, Yb, Mb = Z[:, idx], Y[:, idx], M[:, idx]
        else:
            Zb, Yb, Mb = Z, Y, M
        with torch.autocast('cuda', dtype=torch.bfloat16):
            g = torch.bmm(Zb, qste_bf16(W1).transpose(1, 2)).clamp(max=10.0)
            u = torch.bmm(Zb, qste_bf16(W3).transpose(1, 2)).clamp(min=-10.0, max=10.0)
            h = F.silu(g) * u
            yp_r = torch.bmm(h, qste_bf16(W2).transpose(1, 2))
            yp = torch.bmm(yp_r, Qp.transpose(1, 2))
        num = (Mb * (yp - Yb) ** 2).sum(dim=(1, 2)).float()
        den = (Mb * Yb ** 2).sum(dim=(1, 2)).float().clamp_min(1e-12)
        resid = num / den
        loss = num.sum() / (Mb.sum() * D + 1e-12)
        W1.grad = None; W3.grad = None; W2.grad = None; Qp.grad = None
        loss.backward()
        with torch.no_grad():
            lr_m = 0.05 * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * st / steps)).item())
            e1.mul_(MU).add_(W1.grad, alpha=1 - MU)
            e3.mul_(MU).add_(W3.grad, alpha=1 - MU)
            e2.mul_(MU).add_(W2.grad, alpha=1 - MU)
            eQ.mul_(MU).add_(Qp.grad, alpha=1 - MU)
            W1.data -= lr_m * zeropower(e1)
            W3.data -= lr_m * zeropower(e3)
            W2.data -= lr_m * zeropower(e2)
            Qp.data -= lr_m * zeropower(eQ)
        if (st + 1) % check_every == 0:
            print(f'    step {st+1}/{steps}  loss={loss.item():.6f}  '
                  f'resid med={resid.median().item()*100:.4f}%  '
                  f'ETA {(time.time()-t0)/(st+1)*(steps-st)/60:.1f} min', flush=True)

    w1q, w1s = ternarize(W1.detach())
    w3q, w3s = ternarize(W3.detach())
    w2q, w2s = ternarize(W2.detach())
    out = []
    for i in range(G):
        Qd = Qp.detach()[i]
        if q_dtype == 'int8':
            Qq, Qs = quantize_q(Qd)
        elif q_dtype == 'bf16':
            Qq, Qs = Qd.to(torch.bfloat16), torch.ones(kp, device='cuda')
        else:  # fp32
            Qq, Qs = Qd.float(), torch.ones(kp, device='cuda')
        out.append((w1q[i], w1s[i], w3q[i], w3s[i], w2q[i], w2s[i], Qq, Qs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--n-experts', type=int, default=4)
    ap.add_argument('--steps', type=int, default=1600)
    ap.add_argument('--q-dtype', choices=['int8', 'bf16', 'fp32'], default='int8')
    args = ap.parse_args()

    acts = torch.load(os.path.join(POD, f'acts_layer{args.layer}.pt'),
                      map_location='cpu', weights_only=False)
    P = torch.load(os.path.join(REDUCED, f'layer_{args.layer}', 'P.pt'),
                   map_location='cuda').float()
    mu = torch.load(os.path.join(REDUCED, f'layer_{args.layer}', 'mu.pt'),
                    map_location='cuda').float()

    keys = list(acts.keys())[:args.n_experts]
    pairs_full = []
    for k in keys:
        x_k, y_k = acts[k]
        n_k = x_k.shape[0]
        if n_k > 1024:
            idx = torch.randperm(n_k)[:1024]
            x_k = x_k[idx]
            y_k = y_k[idx]
        z = (x_k.float().cuda() - mu) @ P
        y_full = y_k.float().cuda()
        e = torch.load(os.path.join(REDUCED, f'layer_{args.layer}', f'expert_{k}.pt'),
                       map_location='cpu', weights_only=False)
        Qfull = e['Q'].float() * e['Q_scale'].float()[None, :]  # [4096, 384]
        pairs_full.append((k, z, y_full, Qfull))

    configs = [
        (128, 16),
        (128, 8),
        (128, 4),
        (64, 16),
        (64, 8),
        (64, 4),
        (32, 8),
        (32, 4),
    ]

    print(f'layer {args.layer}, {len(pairs_full)} experts, {args.steps} steps', flush=True)
    print(f'{"inter":>6} {"kp":>5} {"MB":>7} {"resid%":>10}', flush=True)
    for inter, kp in configs:
        pairs = [(z, y, Qfull[:, :kp]) for _, z, y, Qfull in pairs_full]
        t0 = time.time()
        results = train_sweep(pairs, inter, kp, args.steps, args.q_dtype)
        meds = []
        for (_, z, y, _), r in zip(pairs_full, results):
            meds.append(resid_full(z, y, *r))
        med = torch.tensor(meds).median().item() * 100
        q_bytes = {'int8': 1, 'bf16': 2, 'fp32': 4}[args.q_dtype]
        print(f'{inter:>6} {kp:>5} {size_mb(inter, kp, q_bytes):>7.2f} {med:>10.4f}  '
              f'({time.time()-t0:.0f}s)', flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
