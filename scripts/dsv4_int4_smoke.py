"""Long smoke: co-adaptation for S=1 binary W13 + W2 int4 (6.0MB) vs champion S=4 binary (6.0MB).

Config A (int4): binary features (sign STE, Muon, per-row scales), W2 = dual-ridge Theta
snapped to int4 with per-output-channel LS scale, re-solved every 25 steps; features train
THROUGH the snapped W2 (co-adaptation). Rounds x steps, log per round.
Config B (champion parity): same synthetic teacher, binary4x trainer (train_batch) same steps.
"""
import sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
import dsv4_refit_experts as R

torch.manual_seed(0)
DEV = "cuda"
n, K, D, I = 2048, 4096, 4096, 2048
z = torch.randn(n, K, device=DEV, dtype=torch.bfloat16)
W1 = torch.randn(I, K, device=DEV) * K**-0.5
W3 = torch.randn(I, K, device=DEV) * K**-0.5
W2 = torch.randn(D, I, device=DEV) * I**-0.5
g = F.silu((z.float() @ W1.T).clamp(max=10.0))
u = (z.float() @ W3.T).clamp(min=-10.0, max=10.0)
y = (g * u @ W2.T)
zc = z.float()

def q_bin(w):
    wq = w.sign()
    return w + (wq - w).detach()

def theta_of(hf):
    Kk = hf @ hf.T
    Kk = Kk + Kk.diagonal().mean() * 1e-2 * torch.eye(n, device=DEV)
    return hf.T @ torch.linalg.solve(Kk, y)

def snap_int4(th):
    qmax = 7.0
    s = (th.abs().amax(dim=0, keepdim=True) / qmax).clamp_min(1e-9)
    q = (th / s).round().clamp(-qmax, qmax)
    a = (q * th).sum(dim=0) / (q * q).sum(dim=0).clamp_min(1e-9)
    a = a.sign() * a.abs().clamp_min(1e-7)
    return q * a.unsqueeze(0)

s1 = W1.abs().mean(dim=1).detach().clone().requires_grad_()
s3 = W3.abs().mean(dim=1).detach().clone().requires_grad_()
w1s = W1.clone().requires_grad_()
w3s = W3.clone().requires_grad_()
mu1 = torch.zeros_like(w1s); mu3 = torch.zeros_like(w3s)
w2q = torch.zeros(D, I, device=DEV)

def feats():
    gg = F.silu((zc @ (q_bin(w1s) * s1.unsqueeze(1)).T).clamp(max=10.0))
    uu = (zc @ (q_bin(w3s) * s3.unsqueeze(1)).T).clamp(min=-10.0, max=10.0)
    return gg * uu

t0 = time.time()
ROUNDS, STEPS = 15, 400
for rnd in range(ROUNDS):
    for st in range(STEPS):
        hf = feats()
        loss = ((hf @ w2q.T - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for w, mom in ((w1s, mu1), (w3s, mu3)):
                mom.mul_(0.95).add_(w.grad)
                upd = R.zeropower(mom.unsqueeze(0)).squeeze(0)
                w.data.add_(upd, alpha=-0.02)
            s1 -= 0.02 * s1.grad; s3 -= 0.02 * s3.grad
            s1.clamp_min_(1e-6); s3.clamp_min_(1e-6)
        w1s.grad = None; w3s.grad = None; s1.grad = None; s3.grad = None
        if (st + 1) % 25 == 0:
            with torch.no_grad():
                w2q = snap_int4(theta_of(feats())).T.contiguous()
    with torch.no_grad():
        hf = feats()
        r = (((hf @ w2q.T - y) ** 2).sum() / (y ** 2).sum()).item()
    print(f"[int4] round {rnd}: {r*100:.3f}% ({time.time()-t0:.0f}s)", flush=True)

# champion parity: binary4x trainer, same total steps
res = R.train_batch([(z, y.to(torch.bfloat16))], I, ROUNDS * STEPS, stop_threshold=None,
                    init=(W1.to(torch.bfloat16), W3.to(torch.bfloat16), W2.to(torch.bfloat16)))
r = R.resid_weights_full(z, y, res[0])
print(f"[champ-bin4x] {ROUNDS*STEPS} steps: {r*100:.3f}%", flush=True)
