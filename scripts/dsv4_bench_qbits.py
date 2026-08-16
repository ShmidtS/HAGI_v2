"""Sweep quantization divisor (scale = max/divisor) for int4 Q with QAT.

Larger divisor -> smaller scale -> more clipping of outliers but finer levels
for the bulk of values. Sweeps divisor beyond the base level count (7 for int4).
"""
import torch, torch.nn.functional as F, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv4_refit_experts import K, INTER, KP, D, qste_bf16, zeropower, ternarize, pack_ternary

POD = 'checkpoints_dsv4/pod_accurate'
REDUCED = 'dsv4_reduced'
SWIGLU_LIMIT = 10.0
LEVELS = 7  # int4


def load_expert(L, k):
    P = torch.load(os.path.join(REDUCED, f'layer_{L}', 'P.pt'), map_location='cuda').float()
    mu = torch.load(os.path.join(REDUCED, f'layer_{L}', 'mu.pt'), map_location='cuda').float()
    acts = torch.load(os.path.join(POD, f'acts_layer{L}.pt'), map_location='cpu', weights_only=False)
    x_k, y_k = acts[k]
    n = x_k.shape[0]
    if n > 1024:
        idx = torch.randperm(n)[:1024]
        x_k = x_k[idx]
        y_k = y_k[idx]
    z = (x_k.float().cuda() - mu) @ P
    y = y_k.float().cuda()
    return z, y


def qat_q(Qp, divisor):
    """Qp [1, D, KP] fp32 -> STE-quantized with scale = max/divisor."""
    scale = Qp.detach().abs().max(dim=1, keepdim=True)[0].clamp_min(1e-9) / divisor
    q = (Qp / scale).round().clamp(-LEVELS, LEVELS)
    return Qp + (q * scale - Qp).detach()


def train_one(z, y, steps, divisor):
    Q0 = torch.svd_lowrank(y, q=KP, niter=2)[2]
    if Q0.shape[1] < KP:
        Q0 = torch.cat([Q0, torch.zeros(D, KP - Q0.shape[1], device='cuda')], dim=1)
    Z = z.to(torch.bfloat16).unsqueeze(0)
    Y = y.to(torch.bfloat16).unsqueeze(0)
    W1 = torch.nn.Parameter(torch.randn(1, INTER, K, device='cuda') * K**-0.5)
    W3 = torch.nn.Parameter(torch.randn(1, INTER, K, device='cuda') * K**-0.5)
    W2 = torch.nn.Parameter(torch.randn(1, KP, INTER, device='cuda') * INTER**-0.5)
    Qp = torch.nn.Parameter(Q0.float().unsqueeze(0))
    e1 = torch.zeros_like(W1); e3 = torch.zeros_like(W3)
    e2 = torch.zeros_like(W2); eQ = torch.zeros_like(Qp)
    MU = 0.95
    bs = min(Z.shape[1], 1024)
    for st in range(steps):
        if Z.shape[1] > bs:
            idx = torch.randint(0, Z.shape[1], (bs,), device='cuda')
            Zb, Yb = Z[:, idx], Y[:, idx]
        else:
            Zb, Yb = Z, Y
        with torch.autocast('cuda', dtype=torch.bfloat16):
            g = torch.bmm(Zb, qste_bf16(W1).transpose(1, 2)).clamp(max=SWIGLU_LIMIT)
            u = torch.bmm(Zb, qste_bf16(W3).transpose(1, 2)).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            h = F.silu(g) * u
            yp_r = torch.bmm(h, qste_bf16(W2).transpose(1, 2))
            yp = torch.bmm(yp_r, qat_q(Qp, divisor).transpose(1, 2))
        loss = F.mse_loss(yp.float(), Yb.float())
        W1.grad = None; W3.grad = None; W2.grad = None; Qp.grad = None
        loss.backward()
        with torch.no_grad():
            lr = 0.05 * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * st / steps)).item())
            e1.mul_(MU).add_(W1.grad, alpha=1 - MU)
            e3.mul_(MU).add_(W3.grad, alpha=1 - MU)
            e2.mul_(MU).add_(W2.grad, alpha=1 - MU)
            eQ.mul_(MU).add_(Qp.grad, alpha=1 - MU)
            W1.data -= lr * zeropower(e1)
            W3.data -= lr * zeropower(e3)
            W2.data -= lr * zeropower(e2)
            Qp.data -= lr * zeropower(eQ)
    w1q, w1s = ternarize(W1.detach())
    w3q, w3s = ternarize(W3.detach())
    w2q, w2s = ternarize(W2.detach())
    w1q, w1s = w1q[0], w1s[0]
    w3q, w3s = w3q[0], w3s[0]
    w2q, w2s = w2q[0], w2s[0]
    return w1q, w1s, w3q, w3s, w2q, w2s, Qp.detach()[0].float()


def quantize_bits(Qfull, divisor):
    scale = Qfull.abs().max(0)[0].clamp_min(1e-9) / divisor
    return (Qfull / scale[None, :]).round().clamp(-LEVELS, LEVELS).to(torch.int8), scale


def resid_for(z, y, w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs):
    w1 = w1q * w1s[:, None]
    w3 = w3q * w3s[:, None]
    w2 = w2q * w2s[:, None]
    Q = Qq.float() * Qs[None, :]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        g = (z.to(torch.bfloat16) @ w1.to(torch.bfloat16).T).clamp(max=SWIGLU_LIMIT)
        u = (z.to(torch.bfloat16) @ w3.to(torch.bfloat16).T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        h = F.silu(g) * u
        yp = (h @ w2.to(torch.bfloat16).T) @ Q.to(torch.bfloat16).T
    return (F.mse_loss(yp.float(), y) / F.mse_loss(y, torch.zeros_like(y))).item()


def main():
    acts = torch.load(os.path.join(POD, 'acts_layer0.pt'), map_location='cpu', weights_only=False)
    k = max(acts.items(), key=lambda kv: kv[1][0].shape[0])[0]
    print(f'layer 0 expert {k}: {acts[k][0].shape[0]} samples, int4 Q, {STEPS} steps', flush=True)
    z, y = load_expert(0, k)

    print(f'\n=== int4 Q divisor sweep (scale = max/divisor) ===', flush=True)
    for div in (28, 56, 112, 224, 448, 896):
        t0 = time.time()
        w1q, w1s, w3q, w3s, w2q, w2s, Qfull = train_one(z, y, STEPS, div)
        Qq, Qs = quantize_bits(Qfull, div)
        resid = resid_for(z, y, w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs)
        print(f'  divisor={div:>2}: resid={resid*100:.4f}%  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    STEPS = 5000
    main()
