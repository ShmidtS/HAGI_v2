"""Parallel activation refit: retrain ternary core FROM SCRATCH on exact
per-expert activations (x_k -> y_k), batched with masked padding.

Run multiple processes over disjoint layer ranges (--start-layer/--end-layer).
--all refits every expert with collected activations (target ~0.003%).
Resumable via a per-process --done-log.
"""
import torch, torch.nn.functional as F, os, sys, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv4_reduce_layer import qste, ternarize, pack_ternary
from dsv4_generate_reduced import unpack_ternary

K = 512
INTER = 4096
KP = 384
POD = 'checkpoints_dsv4/pod_accurate'
REDUCED = 'dsv4_reduced'


def current_resid(z, target, e):
    w1 = unpack_ternary(e['w1'])[:, :K].float().cuda() * e['w1_scale'].float().cuda()[:, None]
    w3 = unpack_ternary(e['w3'])[:, :K].float().cuda() * e['w3_scale'].float().cuda()[:, None]
    w2 = unpack_ternary(e['w2'])[:, :INTER].float().cuda() * e['w2_scale'].float().cuda()[:, None]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        g = (z @ w1.T).clamp(max=10.0)
        u = (z @ w3.T).clamp(min=-10.0, max=10.0)
        yp = (F.silu(g) * u) @ w2.T
    return (F.mse_loss(yp.float(), target) /
            F.mse_loss(target, torch.zeros_like(target))).item()


def resid_weights(z, target, w1q, w1s, w3q, w3s, w2q, w2s):
    w1 = w1q * w1s[:, None]
    w3 = w3q * w3s[:, None]
    w2 = w2q * w2s[:, None]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        g = (z @ w1.T).clamp(max=10.0)
        u = (z @ w3.T).clamp(min=-10.0, max=10.0)
        yp = (F.silu(g) * u) @ w2.T
    return (F.mse_loss(yp.float(), target) /
            F.mse_loss(target, torch.zeros_like(target))).item()


def train_batch(pairs, inter, steps):
    """pairs: list of (z_k [n_k,K], target_k [n_k,KP]) -> list of (w1q,w1s,w3q,w3s,w2q,w2s)."""
    G = len(pairs)
    Nmax = max(z.shape[0] for z, _ in pairs)
    Z = torch.zeros(G, Nmax, K, device='cuda')
    T = torch.zeros(G, Nmax, KP, device='cuda')
    M = torch.zeros(G, Nmax, 1, device='cuda')
    for i, (z, t) in enumerate(pairs):
        Z[i, :z.shape[0]] = z
        T[i, :t.shape[0]] = t
        M[i, :z.shape[0]] = 1.0

def qste_bf16(W):
    """Ternary STE in bf16 (half the allocation vs fp32). W [G,out,in] fp32 -> [G,out,in] bf16."""
    Wb = W.to(torch.bfloat16)
    s = Wb.detach().abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
    q = (Wb / s).clamp(-1, 1).round() * s
    return Wb + (q - Wb).detach()


def _ternary_eff(W):
    s = W.detach().abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
    q = (W / s).clamp(-1, 1).round()
    return q.to(torch.bfloat16) * s.to(torch.bfloat16)


def train_batch(pairs, inter, steps, err_thresh=1e-4, check_every=25):
    """pairs: list of (z_k [n_k,K], target_k [n_k,KP]) -> list of (w1q,w1s,w3q,w3s,w2q,w2s).
    Freezes each expert IMMEDIATELY once its (EMA) sample residual < err_thresh."""
    G = len(pairs)
    Nmax = max(z.shape[0] for z, _ in pairs)
    Z = torch.zeros(G, Nmax, K, device='cuda', dtype=torch.bfloat16)
    T = torch.zeros(G, Nmax, KP, device='cuda')
    M = torch.zeros(G, Nmax, 1, device='cuda')
    for i, (z, t) in enumerate(pairs):
        Z[i, :z.shape[0]] = z.to(torch.bfloat16)
        T[i, :t.shape[0]] = t
        M[i, :z.shape[0]] = 1.0

    W1 = torch.nn.Parameter(torch.randn(G, inter, K, device='cuda') * K**-0.5)
    W3 = torch.nn.Parameter(torch.randn(G, inter, K, device='cuda') * K**-0.5)
    W2 = torch.nn.Parameter(torch.randn(G, KP, inter, device='cuda') * inter**-0.5)
    o = torch.optim.Adam([W1, W3, W2], lr=2e-3)
    bs = min(Nmax, 2048)
    frozen = torch.zeros(G, dtype=torch.bool, device='cuda')
    ema = torch.full((G,), 1.0, device='cuda')
    t0 = time.time()
    for st in range(steps):
        if frozen.all():
            break
        lr = 2e-3 * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * st / steps)).item())
        for g in o.param_groups:
            g['lr'] = lr
        if Nmax > bs:
            idx = torch.randint(0, Nmax, (bs,), device='cuda')
            Zb, Tb, Mb = Z[:, idx], T[:, idx], M[:, idx]
        else:
            Zb, Tb, Mb = Z, T, M
        with torch.autocast('cuda', dtype=torch.bfloat16):
            g = torch.bmm(Zb, qste_bf16(W1).transpose(1, 2)).clamp(max=10.0)
            u = torch.bmm(Zb, qste_bf16(W3).transpose(1, 2)).clamp(min=-10.0, max=10.0)
            h = F.silu(g) * u
            yp = torch.bmm(h, qste_bf16(W2).transpose(1, 2))
        # per-expert sample residual (cheap, already computed) -> freeze immediately
        num = (Mb * (yp.float() - Tb) ** 2).sum(dim=(1, 2))
        den = (Mb * Tb ** 2).sum(dim=(1, 2)).clamp_min(1e-12)
        ema = 0.8 * ema + 0.2 * (num / den)
        newly = (ema < err_thresh) & ~frozen
        frozen |= newly
        loss = num.sum() / (Mb.sum() * KP + 1e-12)
        o.zero_grad()
        loss.backward()
        if frozen.any():
            W1.grad[frozen] = 0.0
            W3.grad[frozen] = 0.0
            W2.grad[frozen] = 0.0
        o.step()
        if (st + 1) % check_every == 0:
            print(f'    step {st+1}/{steps}  loss={loss.item():.6f}  '
                  f'frozen={int(frozen.sum())}/{G}  '
                  f'ETA {(time.time()-t0)/(st+1)*(steps-st)/60:.1f} min', flush=True)

    w1q, w1s = ternarize(W1.detach())
    w3q, w3s = ternarize(W3.detach())
    w2q, w2s = ternarize(W2.detach())
    return [(w1q[i], w1s[i], w3q[i], w3s[i], w2q[i], w2s[i]) for i in range(G)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-layer', type=int, default=0)
    ap.add_argument('--end-layer', type=int, default=43)
    ap.add_argument('--group', type=int, default=512)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--all', action='store_true', help='refit all experts, not only > 0.1%')
    ap.add_argument('--done-log', default='refit_done.txt')
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.done_log):
        done = set(open(args.done_log).read().split())

    for L in range(args.start_layer, args.end_layer):
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
            if key in done:
                continue
            ep = os.path.join(REDUCED, f'layer_{L}', f'expert_{k}.pt')
            if not os.path.exists(ep):
                continue
            # cap n_k to keep the batched forward cheap (1024 tokens is plenty)
            n_k = x_k.shape[0]
            if n_k > 1024:
                idx = torch.randperm(n_k)[:1024]
                x_k = x_k[idx]
                y_k = y_k[idx]
            e = torch.load(ep, map_location='cpu', weights_only=False)
            Q = e['Q'].float().cuda() * e['Q_scale'].float().cuda()[None, :]
            z = (x_k.float().cuda() - mu) @ P
            target = y_k.float().cuda() @ Q
            if not args.all:
                resid = current_resid(z, target, e)
                if resid <= 0.001:
                    with open(args.done_log, 'a') as f:
                        f.write(f'{key}\n')
                    continue
            todo.append((k, z, target))

        del acts
        torch.cuda.empty_cache()
        print(f'layer {L}: collected {len(todo)} experts to refit', flush=True)

        if not todo:
            print(f'layer {L}: nothing to refit', flush=True)
            continue

        todo.sort(key=lambda t: t[1].shape[0])
        t0 = time.time()
        n_fixed = 0
        for i in range(0, len(todo), args.group):
            chunk = todo[i:i + args.group]
            pairs = [(z, tgt) for _, z, tgt in chunk]
            results = train_batch(pairs, INTER, args.steps)
            for (k, z, tgt), (w1q, w1s, w3q, w3s, w2q, w2s) in zip(chunk, results):
                resid = resid_weights(z, tgt, w1q, w1s, w3q, w3s, w2q, w2s)
                ep = os.path.join(REDUCED, f'layer_{L}', f'expert_{k}.pt')
                e = torch.load(ep, map_location='cpu', weights_only=False)
                e['w1'] = pack_ternary(w1q).cpu()
                e['w1_scale'] = w1s.cpu()
                e['w3'] = pack_ternary(w3q).cpu()
                e['w3_scale'] = w3s.cpu()
                e['w2'] = pack_ternary(w2q).cpu()
                e['w2_scale'] = w2s.cpu()
                e['residual'] = resid
                torch.save(e, ep)
                with open(args.done_log, 'a') as f:
                    f.write(f'{L}_{k}\n')
                n_fixed += 1
            print(f'  layer {L} chunk {i // args.group}: {len(chunk)} experts '
                  f'({time.time()-t0:.0f}s total)', flush=True)
            torch.cuda.empty_cache()
        print(f'layer {L}: refit {n_fixed} experts in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
