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
from dsv4_collect_x_accurate import load_selected_experts, ffn as ffn_exact
import dsv4_experts as de

K = 512
INTER = 1024
KP = 512
MAX_KP = 768
Q_BITS = 4
Q_DIVISOR = 28
Q_LEVELS = 7
REFIT_CFGS = [
    (1024, 512),
]
D = 4096
POD = 'checkpoints_dsv4/pod_accurate'
REDUCED = 'dsv4_reduced'
M_SYNTH = 2048  # universal test-signal samples per expert (multi-tone + white noise)


_ROUTER_W_CACHE = {}


def load_router_weight(L):
    """Load the router gate weight [256, 4096] for layer L (cached)."""
    if L not in _ROUTER_W_CACHE:
        snap = de.default_snapshot()
        wm = de.load_index(snap)['weight_map']
        p = f'layers.{L}.ffn.gate'
        _ROUTER_W_CACHE[L] = de.read_tensor(snap, wm, f'{p}.weight', device='cuda').to(torch.float32)
    return _ROUTER_W_CACHE[L]


def universal_signal(z_real, M=M_SYNTH, seed=0, sigma_override=None, z_proxy=None):
    """Comm-theory universal test signal in POD coordinates, confined to the real manifold.

    Bootstrap + 10% jitter around real points (manifold coverage). Multi-tone axis
    probes were dropped: they excite the expert on pure axes the ternary kernel
    cannot match and inflated the mixed residual without improving real-data fit.
    For experts with no real samples, bootstrap from z_proxy (other experts of the
    layer, same input manifold); falls back to white noise when no proxy exists.
    """
    n, K = z_real.shape
    g = torch.Generator(device=z_real.device).manual_seed(seed)
    if sigma_override is not None:
        sigma = sigma_override
    elif n >= 8:
        sigma = z_real.std(dim=0).clamp(0.3, 2.0)
    else:
        sigma = torch.ones(K, device=z_real.device)
    if n == 0:
        if z_proxy is not None and z_proxy.shape[0] > 0:
            idx = torch.randint(0, z_proxy.shape[0], (M,), generator=g, device=z_real.device)
            eps = torch.randn(M, K, generator=g, device=z_real.device) * (0.1 * sigma[None, :])
            return z_proxy[idx] + eps
        return torch.randn(M, K, generator=g, device=z_real.device) * sigma[None, :]
    idx = torch.randint(0, n, (M,), generator=g, device=z_real.device)
    z_base = z_real[idx]  # [M, K]
    eps = torch.randn(M, K, generator=g, device=z_real.device) * (0.1 * sigma[None, :])
    return z_base + eps


def safe_svd_q(Y, q):
    """Top-q right singular vectors [D, q] of Y, robust to ill-conditioned inputs."""
    try:
        _, _, V = torch.svd_lowrank(Y, q=q, niter=2)
    except Exception:
        _, _, V = torch.linalg.svd(Y.float(), full_matrices=False)
        V = V[:, :q]
    return V


def pick_config(n_k):
    """Adaptive (inter, kp) by activation count.

    Residual correlates with n_k (Pearson r=0.92 on layer 33): experts with
    many activations have a higher-rank output that the fixed 1024/512 kernel
    cannot express. Scale width (inter) and output rank (kp) with data volume.
    """
    if n_k < 200:
        return 1024, 512
    if n_k < 400:
        return 2048, 512
    return 4096, 768


def current_resid(z, y_full, e):
    Kk = z.shape[1]
    w1 = unpack_ternary(e['w1']).float().cuda()[:, :Kk] * e['w1_scale'].float().cuda()[:, None]
    w3 = unpack_ternary(e['w3']).float().cuda()[:, :Kk] * e['w3_scale'].float().cuda()[:, None]
    w2 = unpack_ternary(e['w2']).float().cuda() * e['w2_scale'].float().cuda()[:, None]
    Q = unpack_int4(e['Q']).float().cuda() * e['Q_scale'].float().cuda()[None, :]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        g = (z @ w1.T).clamp(max=10.0)
        u = (z @ w3.T).clamp(min=-10.0, max=10.0)
        w2 = w2[:, :g.shape[1]]
        yp = ((F.silu(g) * u) @ w2.T) @ Q.T
    return (F.mse_loss(yp.float(), y_full) /
            F.mse_loss(y_full, torch.zeros_like(y_full))).item()


def resid_weights_full(z, y_full, Qq, Qs, w1q, w1s, w3q, w3s, w2q, w2s):
    Kk = z.shape[1]
    w1 = w1q[:, :Kk] * w1s[:, None]
    w3 = w3q[:, :Kk] * w3s[:, None]
    w2 = w2q * w2s[:, None]
    Q = unpack_int4(Qq).float() * Qs[None, :]
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


def pack_int4(q):
    """q [D, KP] int8 (-7..7) -> uint8 [D, KP//2] (two nibbles per byte, +8 offset)."""
    q = q[:, :q.shape[1] // 2 * 2]
    even = q[:, 0::2]
    odd = q[:, 1::2]
    return ((even + 8) & 0x0F) | (((odd + 8) & 0x0F) << 4)


def unpack_int4(p):
    """uint8 [D, KP//2] -> int8 [D, KP] (-7..7)."""
    even = (p & 0x0F).to(torch.int16) - 8
    odd = ((p >> 4) & 0x0F).to(torch.int16) - 8
    return torch.stack([even, odd], dim=2).reshape(p.shape[0], p.shape[1] * 2).to(torch.int8)


def qat_q(Qp, divisor):
    """Qp [G, D, KP] fp32 -> STE-quantized int4 (scale = max/divisor)."""
    scale = Qp.detach().abs().max(dim=1, keepdim=True)[0].clamp_min(1e-9) / divisor
    q = (Qp / scale).round().clamp(-Q_LEVELS, Q_LEVELS)
    return Qp + (q * scale - Qp).detach()


def quantize_q(Q):
    """Q [4096, KP] float -> (int4 packed uint8 [4096, KP//2], scale [KP])."""
    scale = Q.abs().max(dim=0)[0].clamp_min(1e-9) / Q_DIVISOR
    q = (Q / scale[None, :]).round().clamp(-Q_LEVELS, Q_LEVELS).to(torch.int8)
    return pack_int4(q), scale


def train_batch(pairs, inter, steps, kp=KP, check_every=25, patience=1000, stop_threshold=None, n_real=0):
    """pairs: list of (z_k [n_k,K], y_k [n_k,4096], Q0_k [4096,kp])
    -> list of (w1q,w1s,w3q,w3s,w2q,w2s,Qq,Qs).
    Jointly trains ternary core + learnable Q against full 4096-dim target.
    n_real: first n_real rows of each pair are the REAL samples; when set, the
    early-stop threshold is evaluated on those rows only (honest metric)."""
    G = len(pairs)
    if isinstance(n_real, (list, tuple)):
        n_real = [int(v) for v in n_real]
    else:
        n_real = [int(n_real)] * G
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
    best = None
    stall = 0
    ema = None
    best_state = None
    best_stop = torch.full((G,), float('inf'), device='cuda')
    n_real_t = torch.tensor(n_real, device='cuda')
    done_mask = torch.zeros(G, dtype=torch.bool, device='cuda')
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
            Quse = qat_q(Qp, Q_DIVISOR)
            yp = torch.bmm(yp_r, Quse.transpose(1, 2))           # [G,bs,4096]
        num = (Mb * (yp - Yb) ** 2).sum(dim=(1, 2)).float()
        den = (Mb * Yb ** 2).sum(dim=(1, 2)).float().clamp_min(1e-12)
        resid = num / den
        active = (~done_mask).float()
        if active.sum() == 0:
            break
        loss = (num * active).sum() / ((Mb.sum(dim=(1, 2)) * active).sum() * D + 1e-12)
        W1.grad = None; W3.grad = None; W2.grad = None; Qp.grad = None
        loss.backward()
        if done_mask.any():
            with torch.no_grad():
                W1.grad[done_mask] = 0
                W3.grad[done_mask] = 0
                W2.grad[done_mask] = 0
                Qp.grad[done_mask] = 0
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
        resid_med = resid.median().item()
        ema = resid_med if ema is None else 0.9 * ema + 0.1 * resid_med
        if stop_threshold is None:
            # Stall-based early stop (synthetic/uncovered experts only).
            if best is None or ema < best - max(5e-4, best * 5e-2):
                best = ema
                stall = 0
            else:
                stall += 1
            if stall >= patience:
                print(f'    stalled at step {st+1}/{steps} (best resid {best*100:.4f}%)', flush=True)
                break
        # Honest early-stop metric: per-expert residual on the REAL rows only.
        # Computed every check_every steps (not every step) to roughly halve the
        # forward cost; best-state tracks the per-expert minimum on that grid.
        honest = (n_real_t.sum() == 0) or (st == steps - 1) or ((st + 1) % check_every == 0)
        stop_resid = resid  # [G] synthetic residual (fallback when not honest)
        if n_real_t.sum() > 0 and honest:
            nrm = int(n_real_t.max().item())
            Zc, Yc = Z[:, :nrm], Y[:, :nrm]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                gr = torch.bmm(Zc, qste_bf16(W1).transpose(1, 2)).clamp(max=10.0)
                ur = torch.bmm(Zc, qste_bf16(W3).transpose(1, 2)).clamp(min=-10.0, max=10.0)
                ypr = torch.bmm(torch.bmm(F.silu(gr) * ur, qste_bf16(W2).transpose(1, 2)), Quse.transpose(1, 2))
            real_mask = torch.arange(nrm, device='cuda')[None, :] < n_real_t[:, None]  # [G, nrm]
            num_r = (real_mask[:, :, None] * (ypr - Yc) ** 2).sum(dim=(1, 2)).float()
            den_r = (real_mask[:, :, None] * Yc ** 2).sum(dim=(1, 2)).float().clamp_min(1e-12)
            stop_resid = num_r / den_r  # [G]
        # Per-expert best-state (min honest stop_resid), not the final one.
        if honest:
            better = stop_resid < best_stop  # [G]
            if better.any():
                best_stop = torch.where(better, stop_resid, best_stop)
                W1d, W3d, W2d, Qpd = W1.detach(), W3.detach(), W2.detach(), Qp.detach()
                if best_state is None:
                    best_state = (W1d.clone(), W3d.clone(), W2d.clone(), Qpd.clone())
                best_state[0][better] = W1d[better]
                best_state[1][better] = W3d[better]
                best_state[2][better] = W2d[better]
                best_state[3][better] = Qpd[better]
        if stop_threshold is not None and honest:
            done_mask |= (stop_resid <= stop_threshold)
            if done_mask.all():
                print(f'    early stop at step {st+1}/{steps} (all {G} experts <= {stop_threshold*100:.3f}%)', flush=True)
                break
        if (st + 1) % check_every == 0:
            print(f'    step {st+1}/{steps}  loss={loss.item():.6f}  '
                  f'resid med={resid.median().item()*100:.4f}%  '
                  f'ETA {(time.time()-t0)/(st+1)*(steps-st)/60:.1f} min', flush=True)

    # Restore the best state (min honest residual) instead of the final one.
    if best_state is not None:
        W1.data.copy_(best_state[0])
        W3.data.copy_(best_state[1])
        W2.data.copy_(best_state[2])
        Qp.data.copy_(best_state[3])
    w1q, w1s = ternarize(W1.detach())
    w3q, w3s = ternarize(W3.detach())
    w2q, w2s = ternarize(W2.detach())
    out = []
    for i in range(G):
        Qq, Qs = quantize_q(Qp.detach()[i])
        out.append((w1q[i], w1s[i], w3q[i], w3s[i], w2q[i], w2s[i], Qq, Qs))
    return out


def run_refit(start_layer, end_layer, group, steps, all_flag, done_log, refit_threshold=2e-4):
    """Router-based refit: covered experts learn on their REAL routed activations
    (honest channel). Uncovered (noisy) experts learn on a proxy manifold built from
    the nearest covered experts by router-weight cosine similarity."""
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
            n_k = x_k.shape[0]
            if n_k > 1024:
                idx = torch.randperm(n_k)[:1024]
                x_k = x_k[idx]
                y_k = y_k[idx]
            e = None
            if os.path.exists(ep):
                e = torch.load(ep, map_location='cpu', weights_only=False)
                rc = e.get('residual', float('inf'))
                if (isinstance(rc, (int, float)) and rc < refit_threshold
                        and e.get('Q_bits', 0) == Q_BITS and not all_flag):
                    with open(done_log, 'a') as f:
                        f.write(f'{key}\n')
                    continue
                Qfull = unpack_int4(e['Q']).float() * e['Q_scale'].float()[None, :]
                if Qfull.shape[1] >= MAX_KP:
                    Q0 = Qfull[:, :MAX_KP].cuda()
                else:
                    Q0 = safe_svd_q(y_k.float().cuda(), MAX_KP)
                    if Q0.shape[1] < MAX_KP:
                        Q0 = torch.cat([Q0, torch.zeros(D, MAX_KP - Q0.shape[1], device='cuda')], dim=1)
            else:
                Q0 = safe_svd_q(y_k.float().cuda(), MAX_KP)
                if Q0.shape[1] < MAX_KP:
                    Q0 = torch.cat([Q0, torch.zeros(D, MAX_KP - Q0.shape[1], device='cuda')], dim=1)
            z = (x_k.float().cuda() - mu) @ P
            y_full = y_k.float().cuda()
            if e is not None and not all_flag:
                resid = current_resid(z, y_full, e)
                if resid <= refit_threshold:
                    with open(done_log, 'a') as f:
                        f.write(f'{key}\n')
                    continue
            todo.append((k, z, y_full, Q0))

        covered = set(acts.keys())
        for k in range(256):
            sk = str(k)
            if sk not in covered:
                todo.append((sk, None, None, None))

        del acts
        torch.cuda.empty_cache()
        print(f'layer {L}: collected {len(todo)} experts to refit '
              f'({len(covered)} real + {len(todo) - len(covered)} synthetic)', flush=True)

        if not todo:
            print(f'layer {L}: nothing to refit', flush=True)
            continue

        todo.sort(key=lambda t: 0 if t[1] is None else t[1].shape[0])
        zs = [t[1] for t in todo if t[1] is not None]
        z_by_key = {t[0]: t[1] for t in todo if t[1] is not None}
        if zs:
            global_sigma = torch.stack([zz.float().std(dim=0, unbiased=False) for zz in zs]).mean(dim=0).clamp(0.1, 2.0)
        else:
            global_sigma = torch.ones(K, device='cuda')
        # Router weights for uncovered-expert proxy selection.
        rw = load_router_weight(L)  # [256, 4096] float32 cuda
        rw_n = rw / (rw.norm(dim=1, keepdim=True).clamp_min(1e-8))
        covered_int = sorted(int(kk) for kk in covered)
        covered_rw = rw_n[covered_int]  # [n_covered, 4096]

        t0 = time.time()
        n_fixed = 0

        def save_expert(k, resid, w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs, inter):
            ep = os.path.join(REDUCED, f'layer_{L}', f'expert_{k}.pt')
            e = torch.load(ep, map_location='cpu', weights_only=False) if os.path.exists(ep) else {}
            e['w1'] = pack_ternary(w1q).cpu()
            e['w1_scale'] = w1s.cpu()
            e['w3'] = pack_ternary(w3q).cpu()
            e['w3_scale'] = w3s.cpu()
            e['w2'] = pack_ternary(w2q).cpu()
            e['w2_scale'] = w2s.cpu()
            e['Q'] = Qq.cpu()
            e['Q_scale'] = Qs.cpu()
            e['Q_bits'] = Q_BITS
            e['inter'] = inter
            e['residual'] = resid
            torch.save(e, ep)
            with open(done_log, 'a') as f:
                f.write(f'{L}_{k}\n')

        covered_todo = [t for t in todo if t[1] is not None]
        missing_todo = [t for t in todo if t[1] is None]

        # --- Covered: one expert at a time (memory-bound iGPU: G=1 beats batching) ---
        for (k, z, yf, Q0) in covered_todo:
            inter, kp = pick_config(z.shape[0])
            z_synth = universal_signal(z)
            x_synth = (mu + z_synth @ P.T).float()
            experts = load_selected_experts(L, [int(k)])
            w1, w2, w3 = experts[int(k)]
            y_synth = ffn_exact(x_synth, w1, w2, w3)
            del experts, x_synth
            z_all = torch.cat([z, z_synth])
            y_all = torch.cat([yf, y_synth])
            results = train_batch([(z_all, y_all, Q0[:, :kp])], inter, steps, kp=kp,
                                  stop_threshold=refit_threshold, n_real=z.shape[0])
            w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs = results[0]
            resid = resid_weights_full(z, yf, Qq, Qs, w1q, w1s, w3q, w3s, w2q, w2s)
            save_expert(k, resid, w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs, inter)
            n_fixed += 1
            print(f'  layer {L} expert {k}: resid={resid*100:.4f}%  '
                  f'({n_fixed} total, {time.time()-t0:.0f}s)', flush=True)
            torch.cuda.empty_cache()

        # --- Missing: proxy + synthetic, one expert at a time (13%) ---
        for (k, z, yf, Q0) in missing_todo:
            inter, kp = 2048, 512
            sim = covered_rw @ rw_n[int(k)]  # [n_covered]
            top = sim.topk(min(8, sim.shape[0])).indices
            z_proxy = torch.cat([z_by_key[str(covered_int[j])] for j in top.tolist()], dim=0)
            z_synth = universal_signal(torch.zeros(0, K, device='cuda'),
                                       sigma_override=global_sigma, z_proxy=z_proxy)
            x_synth = (mu + z_synth @ P.T).float()
            experts = load_selected_experts(L, [int(k)])
            w1, w2, w3 = experts[int(k)]
            y_synth = ffn_exact(x_synth, w1, w2, w3)
            del experts, x_synth
            Q0m = safe_svd_q(y_synth, kp)
            if Q0m.shape[1] < kp:
                Q0m = torch.cat([Q0m, torch.zeros(D, kp - Q0m.shape[1], device='cuda')], dim=1)
            results = train_batch([(z_synth, y_synth, Q0m[:, :kp])], inter, steps, kp=kp,
                                  stop_threshold=None, n_real=0)
            w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs = results[0]
            resid = resid_weights_full(z_synth, y_synth, Qq, Qs, w1q, w1s, w3q, w3s, w2q, w2s)
            save_expert(k, resid, w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs, inter)
            n_fixed += 1
            print(f'  layer {L} expert {k}: resid={resid*100:.4f}%  '
                  f'({n_fixed} total, {time.time()-t0:.0f}s)', flush=True)
            torch.cuda.empty_cache()
        print(f'layer {L}: refit {n_fixed} experts in {time.time()-t0:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-layer', type=int, default=0)
    ap.add_argument('--end-layer', type=int, default=43)
    ap.add_argument('--group', type=int, default=64)
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--all', action='store_true', help='refit all experts, not only > 0.1%')
    ap.add_argument('--done-log', default='refit_done_q.txt')
    ap.add_argument('--refit-threshold', type=float, default=1e-4,
                    help='accept an expert when its residual <= this (fraction, 0.01%% default)')
    ap.add_argument('--n-procs', type=int, default=4,
                    help='split layers across N parallel processes (0=auto=min(4,n_layers))')
    args = ap.parse_args()

    n_procs = max(1, args.n_procs if args.n_procs > 0 else min(4, args.end_layer - args.start_layer))

    if n_procs == 1:
        run_refit(args.start_layer, args.end_layer, args.group, args.steps, args.all,
                  args.done_log, args.refit_threshold)
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
               '--done-log', args.done_log, '--n-procs', '1',
               '--refit-threshold', str(args.refit_threshold)]
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
