"""Parallel activation refit with LEARNABLE Q (joint ternary + Q).

Retrains the ternary core AND a learnable Q jointly from scratch on exact
per-expert activations, against the FULL 4096-dim target (warm-started from
the SVD Q). This replaces the old fixed-SVD-Q refit (reduced 384-dim loss)
and yields ~5x lower residual at the same model size.

Run multiple processes over disjoint layer ranges (--start-layer/--end-layer).
Resumable via a per-process --done-log (skip by full-space residual threshold).
"""
import torch, torch.nn.functional as F, os, sys, time, argparse, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv4_reduce_layer import qste, ternarize, pack_ternary
from dsv4_generate_reduced import unpack_ternary

K = 512
INTER = 4096
KP = 384
D = 4096
POD = 'checkpoints_dsv4/pod_accurate'
REDUCED = 'dsv4_reduced'


def current_resid(z, y_full, e):
    w1 = unpack_ternary(e['w1'])[:, :K].float().cuda() * e['w1_scale'].float().cuda()[:, None]
    w3 = unpack_ternary(e['w3'])[:, :K].float().cuda() * e['w3_scale'].float().cuda()[:, None]
    w2 = unpack_ternary(e['w2'])[:, :INTER].float().cuda() * e['w2_scale'].float().cuda()[:, None]
    Q = e['Q'].float().cuda() * e['Q_scale'].float().cuda()[None, :]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        g = (z @ w1.T).clamp(max=10.0)
        u = (z @ w3.T).clamp(min=-10.0, max=10.0)
        yp = ((F.silu(g) * u) @ w2.T) @ Q.T
    return (F.mse_loss(yp.float(), y_full) /
            F.mse_loss(y_full, torch.zeros_like(y_full))).item()


def resid_weights_full(z, y_full, Qq, Qs, w1q, w1s, w3q, w3s, w2q, w2s):
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


def qste_bf16(W):
    """Ternary STE in bf16 (half the allocation vs fp32). W [G,out,in] fp32 -> [G,out,in] bf16."""
    Wb = W.to(torch.bfloat16)
    s = Wb.detach().abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
    q = (Wb / s).clamp(-1, 1).round() * s
    return Wb + (q - Wb).detach()


def zeropower(G, steps=3):
    """Newton-Schulz orthogonalization of each [m,n] matrix (Muon core)."""
    a, m, n = G.shape
    X = G.to(torch.bfloat16)
    X = X / (X.norm(dim=(1, 2), keepdim=True).clamp_min(1e-7))
    if m <= n:
        for _ in range(steps):
            X = 1.5 * X - 0.5 * (X @ X.transpose(1, 2)) @ X
    else:
        for _ in range(steps):
            X = 1.5 * X - 0.5 * X @ (X.transpose(1, 2) @ X)
    return X.float()


def quantize_q(Q):
    """Q [4096, KP] float -> (int8 [4096, KP], scale [KP])."""
    scale = Q.abs().max(dim=0)[0].clamp_min(1e-9) / 127.0
    q = (Q / scale[None, :]).round().clamp(-127, 127)
    return q.to(torch.int8), scale


def train_batch(pairs, inter, steps, check_every=25):
    """pairs: list of (z_k [n_k,K], y_k [n_k,4096], Q0_k [4096,KP])
    -> list of (w1q,w1s,w3q,w3s,w2q,w2s,Qq,Qs).
    Jointly trains ternary core + learnable Q against full 4096-dim target."""
    G = len(pairs)
    Nmax = max(z.shape[0] for z, _, _ in pairs)
    Z = torch.zeros(G, Nmax, K, device='cuda', dtype=torch.bfloat16)
    Y = torch.zeros(G, Nmax, D, device='cuda', dtype=torch.bfloat16)
    Q = torch.zeros(G, D, KP, device='cuda', dtype=torch.float32)
    M = torch.zeros(G, Nmax, 1, device='cuda')
    for i, (z, y, Q0) in enumerate(pairs):
        Z[i, :z.shape[0]] = z.to(torch.bfloat16)
        Y[i, :y.shape[0]] = y.to(torch.bfloat16)
        Q[i] = Q0.float()
        M[i, :z.shape[0]] = 1.0

    W1 = torch.nn.Parameter(torch.randn(G, inter, K, device='cuda') * K**-0.5)
    W3 = torch.nn.Parameter(torch.randn(G, inter, K, device='cuda') * K**-0.5)
    W2 = torch.nn.Parameter(torch.randn(G, KP, inter, device='cuda') * inter**-0.5)
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
            yp_r = torch.bmm(h, qste_bf16(W2).transpose(1, 2))  # [G,bs,KP]
            yp = torch.bmm(yp_r, Qp.transpose(1, 2))            # [G,bs,4096]
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
        Qq, Qs = quantize_q(Qp.detach()[i])
        out.append((w1q[i], w1s[i], w3q[i], w3s[i], w2q[i], w2s[i], Qq, Qs))
    return out


def run_refit(start_layer, end_layer, group, steps, all_flag, done_log):
    for L in range(start_layer, end_layer):
        p_path = os.path.join(REDUCED, f'layer_{L}', 'P.pt')
        if not os.path.exists(p_path):
            continue
        P = torch.load(p_path, map_location='cuda').float()
        mu = torch.load(os.path.join(REDUCED, f'layer_{L}', 'mu.pt'), map_location='cuda').float()
        acts = torch.load(os.path.join(POD, f'acts_layer{L}.pt'), map_location='cpu', weights_only=False)
        print(f'layer {L}: loading {len(acts)} experts...', flush=True)

        todo = []
        for k, (x_k, y_k) in acts.items():
            key = f'{L}_{k}'
            ep = os.path.join(REDUCED, f'layer_{L}', f'expert_{k}.pt')
            if not os.path.exists(ep):
                continue
            n_k = x_k.shape[0]
            if n_k > 1024:
                idx = torch.randperm(n_k)[:1024]
                x_k = x_k[idx]
                y_k = y_k[idx]
            e = torch.load(ep, map_location='cpu', weights_only=False)
            rc = e.get('residual', float('inf'))
            if isinstance(rc, (int, float)) and rc < 1.5e-4:
                with open(done_log, 'a') as f:
                    f.write(f'{key}\n')
                continue
            Q0 = e['Q'].float().cuda() * e['Q_scale'].float().cuda()[None, :]
            z = (x_k.float().cuda() - mu) @ P
            y_full = y_k.float().cuda()
            if not all_flag:
                resid = current_resid(z, y_full, e)
                if resid <= 1.5e-4:
                    with open(done_log, 'a') as f:
                        f.write(f'{key}\n')
                    continue
            todo.append((k, z, y_full, Q0))

        del acts
        torch.cuda.empty_cache()
        print(f'layer {L}: collected {len(todo)} experts to refit', flush=True)

        if not todo:
            print(f'layer {L}: nothing to refit', flush=True)
            continue

        todo.sort(key=lambda t: t[1].shape[0])
        t0 = time.time()
        n_fixed = 0
        for i in range(0, len(todo), group):
            chunk = todo[i:i + group]
            pairs = [(z, yf, Q0) for _, z, yf, Q0 in chunk]
            results = train_batch(pairs, INTER, steps)
            for (k, z, yf, Q0), (w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs) in zip(chunk, results):
                resid = resid_weights_full(z, yf, Qq, Qs, w1q, w1s, w3q, w3s, w2q, w2s)
                ep = os.path.join(REDUCED, f'layer_{L}', f'expert_{k}.pt')
                e = torch.load(ep, map_location='cpu', weights_only=False)
                e['w1'] = pack_ternary(w1q).cpu()
                e['w1_scale'] = w1s.cpu()
                e['w3'] = pack_ternary(w3q).cpu()
                e['w3_scale'] = w3s.cpu()
                e['w2'] = pack_ternary(w2q).cpu()
                e['w2_scale'] = w2s.cpu()
                e['Q'] = Qq.cpu()
                e['Q_scale'] = Qs.cpu()
                e['residual'] = resid
                torch.save(e, ep)
                with open(done_log, 'a') as f:
                    f.write(f'{L}_{k}\n')
                n_fixed += 1
            print(f'  layer {L} chunk {i // group}: {len(chunk)} experts '
                  f'({time.time()-t0:.0f}s total)', flush=True)
            torch.cuda.empty_cache()
        print(f'layer {L}: refit {n_fixed} experts in {time.time()-t0:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-layer', type=int, default=0)
    ap.add_argument('--end-layer', type=int, default=43)
    ap.add_argument('--group', type=int, default=64)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--all', action='store_true', help='refit all experts, not only > 0.1%')
    ap.add_argument('--done-log', default='refit_done_q.txt')
    ap.add_argument('--n-procs', type=int, default=4,
                    help='split layers across N parallel processes (0=auto=min(4,n_layers))')
    args = ap.parse_args()

    n_procs = max(1, args.n_procs if args.n_procs > 0 else min(4, args.end_layer - args.start_layer))

    if n_procs == 1:
        run_refit(args.start_layer, args.end_layer, args.group, args.steps, args.all, args.done_log)
        return

    per = (args.end_layer - args.start_layer + n_procs - 1) // n_procs
    procs = []
    for i in range(n_procs):
        s = args.start_layer + i * per
        e = min(args.start_layer + (i + 1) * per, args.end_layer)
        if s >= e:
            break
        cmd = [sys.executable, '-u', os.path.abspath(__file__),
               '--start-layer', str(s), '--end-layer', str(e),
               '--group', str(args.group), '--steps', str(args.steps),
               '--done-log', args.done_log, '--n-procs', '1']
        if args.all:
            cmd.append('--all')
        with open(f'refit_q_p{i}.log', 'w') as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        procs.append(p)
        print(f'proc {i}: layers {s}-{e} -> refit_q_p{i}.log (pid {p.pid})', flush=True)
    for p in procs:
        p.wait()
    print('all procs done', flush=True)


if __name__ == '__main__':
    main()
