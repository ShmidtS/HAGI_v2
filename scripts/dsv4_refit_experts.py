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


def universal_signal(z_real, M=M_SYNTH, seed=0, sigma_override=None, z_proxy=None, jitter=0.1):
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
            eps = torch.randn(M, K, generator=g, device=z_real.device) * (jitter * sigma[None, :])
            return z_proxy[idx] + eps
        return torch.randn(M, K, generator=g, device=z_real.device) * sigma[None, :]
    idx = torch.randint(0, n, (M,), generator=g, device=z_real.device)
    z_base = z_real[idx]  # [M, K]
    eps = torch.randn(M, K, generator=g, device=z_real.device) * (jitter * sigma[None, :])
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
    """Adaptive (inter, kp) by activation count — fallback when no saved config.

    Residual correlates with n_k (Pearson r=0.92 on layer 33): experts with
    many activations have a higher-rank output that the fixed 1024/512 kernel
    cannot express. A size/error sweep showed inter=4096 OVERFITS at finite
    time (worse than 2048 at n<=120), so it is dropped.
    """
    if n_k < 200:
        return 1024, 512
    return 2048, 768


def config_candidates(n_k):
    """Pass-1 config search candidates: the size/error Pareto points to pick from.
    n<8 (tier C): real residual is noise on 1..7 rows — no ranking, fixed default.
    n<200: compact (1024,512) vs accurate (2048,768).
    n>=200: accurate (2048,768) vs max-width (4096,768)."""
    if n_k < 8:
        return [pick_config(n_k)]
    if n_k < 200:
        return [(1024, 512), (2048, 768)]
    return [(2048, 768), (4096, 768)]


def read_config(e, n_k):
    """Recover (inter, kp) from a saved expert file (pass 2 = known config).
    kp is encoded in the packed int4 Q shape: Q is [D, kp//2]."""
    if e is not None and 'inter' in e and 'Q' in e:
        return int(e['inter']), int(e['Q'].shape[1] * 2)
    return pick_config(n_k)


def config_search(z, yf, z_all, y_all, Q0, e_prev, n_k, config_steps, config_margin,
                  tier_thr, use_compile, warm, rw_eff):
    """Pass 1: try candidate (inter,kp) at short steps, return the smallest-size
    config whose real residual is within config_margin of the best (Pareto knee).
    Returns (inter, kp, results_tuple, resid)."""
    cands = config_candidates(n_k)
    scored = []
    for inter, kp in cands:
        init = warm_init_from(e_prev, inter, kp) if (warm and e_prev is not None) else None
        results = train_batch([(z_all, y_all, Q0[:, :kp])], inter, config_steps, kp=kp,
                              stop_threshold=tier_thr, n_real=n_k, use_compile=use_compile,
                              init=init, real_weight=rw_eff)
        w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs = results[0]
        resid = resid_weights_full(z, yf, Qq, Qs, w1q, w1s, w3q, w3s, w2q, w2s)
        scored.append((resid, inter, kp, results[0]))
        print(f'    [config] inter={inter} kp={kp} -> resid={resid*100:.4f}%', flush=True)
    # knee: smallest size (inter*kp proxy) within margin of the best residual
    scored.sort(key=lambda t: (t[1] * t[2], t[0]))
    best_resid = min(s[0] for s in scored)
    for resid, inter, kp, res in scored:
        if resid <= best_resid * (1.0 + config_margin):
            return inter, kp, res, resid
    resid, inter, kp, res = scored[-1]
    return inter, kp, res, resid


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
    """Newton-Schulz orthogonalization of each [m,n] matrix (Muon core).
    steps=3: measured identical live throughput to steps=2 in the 4-process
    refit (~46-47 steps/s each), so keep the tighter orthogonalization."""
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


def train_batch_fwd(Zb, W1, W3, W2, Qp):
    """Module-level (dynamo in this torch build cannot trace closures):
    explicit bf16 casts — numerically identical to the autocast version
    (autocast casts bmm inputs to bf16 anyway)."""
    g = torch.bmm(Zb, qste_bf16(W1).transpose(1, 2)).clamp(max=10.0)
    u = torch.bmm(Zb, qste_bf16(W3).transpose(1, 2)).clamp(min=-10.0, max=10.0)
    h = F.silu(g) * u
    yp_r = torch.bmm(h, qste_bf16(W2).transpose(1, 2))  # [G,bs,KP]
    Quse = qat_q(Qp, Q_DIVISOR)
    yp = torch.bmm(yp_r, Quse.to(torch.bfloat16).transpose(1, 2))  # [G,bs,4096]
    return yp, Quse


def train_batch(pairs, inter, steps, kp=KP, check_every=25, patience=4000, stop_threshold=None, n_real=0, use_compile=False,
                 init=None, stall_checks=200, stall_tol=0.02, real_weight=1.0):
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
    if init is not None:
        # Warm-start the ternary core from a previous pass (packed ternary unpacked
        # to float). Shapes must match: (G,inter,K)/(G,inter,K)/(G,kp,inter).
        iW1, iW3, iW2 = init
        if iW1.shape == W1.shape and iW3.shape == W3.shape and iW2.shape == W2.shape:
            W1.data.copy_(iW1)
            W3.data.copy_(iW3)
            W2.data.copy_(iW2)
            print('    [warm] ternary core initialized from previous pass', flush=True)
        else:
            print(f'    [warm] shape mismatch ({tuple(iW1.shape)} vs {tuple(W1.shape)}) — random init', flush=True)
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
    # Honest-EMA stall tracking (covered experts that plateau above the threshold):
    # stop after stall_checks honest checks without >stall_tol relative improvement.
    ema_h = None
    ref_h = None
    stall_ct = torch.zeros(G, device='cuda')
    n_real_t = torch.tensor(n_real, device='cuda')
    done_mask = torch.zeros(G, dtype=torch.bool, device='cuda')

    def update_best(metric, valid=None):
        nonlocal best_state, best_stop
        if valid is None:
            valid = torch.ones(G, dtype=torch.bool, device='cuda')
        if valid.any():
            cand = torch.where(valid, metric, torch.full_like(best_stop, float('inf')))
            better = cand < best_stop
            if better.any():
                best_stop = torch.where(better, cand, best_stop)
                W1d, W3d, W2d, Qpd = W1.detach(), W3.detach(), W2.detach(), Qp.detach()
                if best_state is None:
                    best_state = (W1d.clone(), W3d.clone(), W2d.clone(), Qpd.clone())
                best_state[0][better] = W1d[better]
                best_state[1][better] = W3d[better]
                best_state[2][better] = W2d[better]
                best_state[3][better] = Qpd[better]

    fwd_plain = train_batch_fwd  # non-compiled: full-batch eval below varies N per expert
    if use_compile:
        fwd = torch.compile(train_batch_fwd)
        print('    [compile] torch.compile enabled (first steps compile the graph)', flush=True)
    else:
        fwd = train_batch_fwd

    t0 = time.time()
    for st in range(steps):
        if Nmax > bs:
            idx = torch.randint(0, Nmax, (bs,), device='cuda')
            Zb, Yb, Mb = Z[:, idx], Y[:, idx], M[:, idx]
        else:
            idx = torch.arange(Nmax, device='cuda')
            Zb, Yb, Mb = Z, Y, M
        yp, Quse = fwd(Zb, W1, W3, W2, Qp)
        diff2 = (yp - Yb) ** 2
        yb2 = Yb ** 2
        num = (Mb * diff2).sum(dim=(1, 2)).float()
        den = (Mb * yb2).sum(dim=(1, 2)).float().clamp_min(1e-12)
        resid = num / den
        active = (~done_mask).float()
        if active.sum() == 0:
            break
        # Weighted loss: real rows (idx < n_real) count real_weight x more.
        # The synthetic manifold (2048 rows) otherwise drowns the handful of
        # real rows (~1% of data) — model hits 0.7% on synth but 22% on real.
        if real_weight != 1.0:
            rw = torch.where(idx[None, :] < n_real_t[:, None],
                             torch.full((1,), real_weight, device='cuda'),
                             torch.ones((1,), device='cuda')).float()  # [G, bs]
            rw = rw[..., None]
            num_w = (Mb * rw * diff2).sum(dim=(1, 2)).float()
            den_w = ((Mb * rw).sum(dim=(1, 2)) * D).float().clamp_min(1e-12)
            loss = (num_w * active).sum() / (den_w * active).sum()
        else:
            loss = (num * active).sum() / ((Mb.sum(dim=(1, 2)) * active).sum() * D + 1e-12)
        # Free stochastic real-residual from the training batch (no extra forward):
        # NOTE: no stochastic minibatch-subset best-state update here. A subset
        # of real rows (sampled idx < n_real) is a noisy, biased-low estimator;
        # it drove best_stop to ~10% while the full 24-row residual was ~22%.
        # The honest full-row check (below, every check_every steps) is the only
        # driver of best_stop/best_state.
        W1.grad = None; W3.grad = None; W2.grad = None; Qp.grad = None
        loss.backward()
        if use_compile and ((st + 1) % check_every == 0 or st == 0):
            # ROCm torch.compile bug guard: non-finite grads would silently skip
            # updates (see BENCHMARKS compile+batch>=320). Fail loudly instead.
            if not (torch.isfinite(W1.grad).all() and torch.isfinite(W3.grad).all()
                    and torch.isfinite(W2.grad).all() and torch.isfinite(Qp.grad).all()):
                raise RuntimeError(
                    f'non-finite grads at step {st+1} under torch.compile '
                    '(known ROCm buffer-reuse bug) — rerun without --compile')
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
            # Gate on the PER-EXPERT BEST (what actually gets saved), not on the
            # lagging EMA of the batch median — the median oscillates while the
            # min-state still improves, causing premature stops.
            cur_best = best_stop.min().item()
            if not (cur_best < float('inf')):
                stall = 0  # best_stop not measured yet (nan/inf) — don't count
            elif best is None or cur_best < best - max(1e-5, best * stall_tol):
                best = cur_best
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
        if stop_threshold is not None and n_real_t.sum() > 0 and honest:
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
        if honest and stop_threshold is None and ((st + 1) % check_every == 0 or st == steps - 1):
            # tier C / missing: a 1-row real metric is noise — select best state
            # by the deterministic full-batch manifold residual instead (same
            # metric the stall gate watches).
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                ypf, _ = fwd_plain(Z, W1, W3, W2, Qp)
                d2 = (M * (ypf - Y) ** 2).sum(dim=(1, 2)).float()
                y2 = (M * Y ** 2).sum(dim=(1, 2)).float().clamp_min(1e-12)
                update_best(d2 / y2)
        elif honest:
            update_best(stop_resid)
        if stop_threshold is not None and honest:
            done_mask |= (stop_resid <= stop_threshold)
            if done_mask.all():
                print(f'    early stop at step {st+1}/{steps} (all {G} experts <= {stop_threshold*100:.3f}%)', flush=True)
                break
            # Stall-stop: covered experts stuck on a plateau above the threshold
            # burn the whole budget otherwise. EMA of the honest residual must
            # improve by >stall_tol (relative) within stall_checks honest checks.
            with torch.no_grad():
                if ema_h is None:
                    ema_h = stop_resid.clone()
                    ref_h = stop_resid.clone()
                else:
                    ema_h = 0.9 * ema_h + 0.1 * stop_resid
                    improved = ema_h < ref_h * (1.0 - stall_tol)
                    ref_h = torch.minimum(ref_h, ema_h)
                    stall_ct = torch.where(improved & (~done_mask), torch.zeros_like(stall_ct), stall_ct + 1.0)
                    newly = (~done_mask) & (stall_ct >= stall_checks) & (best_stop > stop_threshold)
                    if newly.any():
                        done_mask |= newly
                        print(f'    stall-stop {int(newly.sum())} expert(s) at step {st+1} '
                              f'(plateau best {best_stop[newly].min().item()*100:.4f}%)', flush=True)
                        if done_mask.all():
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
    if os.environ.get('REFIT_DEBUG_QUANT') == '1':
        nrm = int(n_real_t.max().item()) if n_real_t.numel() else 0
        if nrm > 0:
            Zc, Yc = Z[:, :nrm], Y[:, :nrm]
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                gr = torch.bmm(Zc, qste_bf16(W1).transpose(1, 2)).clamp(max=10.0)
                ur = torch.bmm(Zc, qste_bf16(W3).transpose(1, 2)).clamp(min=-10.0, max=10.0)
                ypr = torch.bmm(torch.bmm(F.silu(gr) * ur, qste_bf16(W2).transpose(1, 2)),
                                qat_q(Qp, Q_DIVISOR).transpose(1, 2))
            rmask = torch.arange(nrm, device='cuda')[None, :] < n_real_t[:, None]
            r_bf16 = ((rmask[:, :, None] * (ypr - Yc) ** 2).sum(dim=(1, 2)).float() /
                      (rmask[:, :, None] * Yc ** 2).sum(dim=(1, 2)).float().clamp_min(1e-12))
            tq1 = ternarize(W1.detach()); tq3 = ternarize(W3.detach()); tq2 = ternarize(W2.detach())
            w1f = tq1[0][0] * tq1[1][0][:, None]; w3f = tq3[0][0] * tq3[1][0][:, None]; w2f = tq2[0][0] * tq2[1][0][:, None]
            _Qq, _Qs = quantize_q(Qp.detach()[0])
            Qf = unpack_int4(_Qq).float() * _Qs[None, :]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                g = (Zc[0] @ w1f[:, :K].T).clamp(max=10.0)
                u = (Zc[0] @ w3f[:, :K].T).clamp(min=-10.0, max=10.0)
                yp = ((F.silu(g) * u) @ w2f.T) @ Qf.T
            r_fp32 = (F.mse_loss(yp.float(), Yc[0].float()) /
                      F.mse_loss(Yc[0].float(), torch.zeros_like(Yc[0].float()))).item()
            print(f'    [quant-debug] best_stop={best_stop[0].item()*100:.4f}%  '
                  f'bf16_resid={r_bf16[0].item()*100:.4f}%  fp32_resid={r_fp32*100:.4f}%', flush=True)
    w1q, w1s = ternarize(W1.detach())
    w3q, w3s = ternarize(W3.detach())
    w2q, w2s = ternarize(W2.detach())
    out = []
    for i in range(G):
        Qq, Qs = quantize_q(Qp.detach()[i])
        # --- Post-hoc least-squares rescale of the kp output channels ---
        # y = (yc * a) @ B with B = Q^T: J(a) is quadratic, normal equations
        # are tiny (kp x kp). Fold the optimal a into w2_scale — format
        # unchanged, free residual gain on the deployment (quantized) weights.
        try:
            w1 = w1q[i] * w1s[i][:, None]
            w3 = w3q[i] * w3s[i][:, None]
            w2 = w2q[i] * w2s[i][:, None]
            B = (unpack_int4(Qq).float() * Qs[None, :]).T  # [kp, D]
            # Fit/verify on the deployment metric: REAL rows only when present
            # (the synthetic-dominated mixture made the gate 95% synthetic and
            # let a channel rescale pass while degrading real rows — 10% -> 21%).
            nri = int(n_real_t[i])
            if nri > 0:
                zi = Z[i, :nri]
                yi = Y[i, :nri]
            else:
                zi = Z[i]
                yi = Y[i]
            yi = yi.float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                g = (zi @ w1.T).clamp(max=10.0)
                u = (zi @ w3.T).clamp(min=-10.0, max=10.0)
                yc = (F.silu(g) * u) @ w2.T  # [N, kp]
            yc = yc.float()
            rw = torch.ones(zi.shape[0], device='cuda')
            ycw = yc * rw[:, None]
            yB = yi @ B.T  # [N, kp]: entry (n, k) = <y_n, B_k>
            Mm = (ycw.T @ yc) * (B @ B.T)  # [kp, kp]
            rhs = (ycw * yB).sum(dim=0)  # rhs_k = sum_n w_n * yc_nk * <y_n, B_k>
            # Ridge toward 1 (shrinkage): free per-channel fit overfits when the
            # real-row count is tiny (tier C). mu blends LS solution with a=1.
            mu = 0.05 * Mm.diagonal().abs().mean().clamp_min(1e-12)
            lam = 1e-6 * Mm.diagonal().abs().mean().clamp_min(1e-12)
            eye = torch.eye(Mm.shape[0], device='cuda')
            a = torch.linalg.solve(Mm + (lam + mu) * eye, rhs + mu)
            a = a.clamp(0.5, 2.0)
            # Self-verification gate: apply only if it measurably improves the
            # weighted residual on these very rows — regression is impossible.
            r_before = ((yc @ B - yi) ** 2 * rw[:, None]).sum()
            r_after = (((yc * a[None, :]) @ B - yi) ** 2 * rw[:, None]).sum()
            if r_after < r_before:
                w2s[i] = w2s[i] * a
        except Exception:
            pass  # rescale is an optimization, never a blocker
        out.append((w1q[i], w1s[i], w3q[i], w3s[i], w2q[i], w2s[i], Qq, Qs))
    return out


def warm_init_from(e, inter, kp):
    """Unpack a saved expert's ternary core into float init tensors for train_batch.
    Returns None if shapes/keys are unusable."""
    try:
        if 'w1' not in e or 'w3' not in e or 'w2' not in e or e.get('inter') != inter:
            return None
        w1 = unpack_ternary(e['w1']).float() * e['w1_scale'].float()[:, None]   # [inter,K']
        w3 = unpack_ternary(e['w3']).float() * e['w3_scale'].float()[:, None]   # [inter,K']
        w2 = unpack_ternary(e['w2']).float() * e['w2_scale'].float()[:, None]   # [rows,KP']
        if w2.shape[0] != kp:
            return None
        # trim base-3 packing pads: K' is rounded up to a multiple of 5
        w1 = w1[:, :K]
        w3 = w3[:, :K]
        w2 = w2[:, :e['inter']]
        if w1.shape[1] != K or w2.shape[1] != e['inter']:
            return None
        return (w1[None], w3[None], w2[None])
    except Exception:
        return None


def run_refit(start_layer, end_layer, group, steps, all_flag, done_log, refit_threshold=2e-4, use_compile=False, warm=False, seed=0,
              real_weight=800.0, jitter=0.2, mode='refine', config_steps=2000, config_margin=0.05):
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
        skipped_z = {}  # z of experts skipped by resume — missing-branch proxy needs them
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
                if (mode == 'refine' and not all_flag
                        and isinstance(rc, (int, float)) and rc < refit_threshold
                        and e.get('Q_bits', 0) == Q_BITS):
                    with open(done_log, 'a') as f:
                        f.write(f'{key}\n')
                    skipped_z[k] = (x_k.float().cuda() - mu) @ P
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
            if e is not None and not all_flag and mode == 'refine':
                resid = current_resid(z, y_full, e)
                if resid <= refit_threshold:
                    with open(done_log, 'a') as f:
                        f.write(f'{key}\n')
                    skipped_z[k] = z
                    continue
            todo.append((k, z, y_full, Q0, e))

        covered = set(acts.keys())
        for k in range(256):
            sk = str(k)
            if sk not in covered:
                todo.append((sk, None, None, None, None))

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
        z_by_key.update(skipped_z)
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

        def save_expert(k, resid, w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs, inter, n_real=None):
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
            if n_real is not None:
                e['n_real'] = int(n_real)
            torch.save(e, ep)
            with open(done_log, 'a') as f:
                f.write(f'{L}_{k}\n')

        covered_todo = [t for t in todo if t[1] is not None]
        missing_todo = [t for t in todo if t[1] is None]

        # --- Covered: one expert at a time (memory-bound iGPU: G=1 beats batching) ---
        for (k, z, yf, Q0, e_prev) in covered_todo:
            n_k = z.shape[0]
            # n_k tiers: honest residual is only meaningful with enough real rows.
            if n_k >= 128:
                tier_thr, tier_steps, tier = refit_threshold, steps, 'A'
            elif n_k >= 8:
                tier_thr, tier_steps, tier = refit_threshold, steps, 'B'
            else:
                # n_k < 8: honest residual is noise on a handful of rows — no stop
                # threshold, full budget (stall-stop still guards plateaus).
                tier_thr, tier_steps, tier = None, steps, 'C'
            # Real-row up-weight only where the honest metric is meaningful (n>=8).
            # For tier C the 1..7 real rows are noise — up-weighting them would
            # overfit the noise and wreck the manifold the model is fit on.
            rw_eff = real_weight if n_k >= 8 else 1.0
            if n_k < 8:
                # tier C (n<8): bootstrap the synthetic manifold from router-weight
                # neighbours (like the missing/n=0 branch), NOT from the <=7 noisy
                # real rows. Those rows sit at sensitive FFN points (silu+clamp) and
                # give a meaningless noise manifold (real resid ~70-150%, synth ~9%).
                # Neighbours' z carry the layer's true input manifold.
                sim = covered_rw @ rw_n[int(k)]
                sim = sim.clone()
                sim[covered_int.index(int(k))] = -1.0  # exclude self
                top = sim.topk(min(8, sim.shape[0])).indices
                z_proxy = torch.cat([z_by_key[str(covered_int[j])] for j in top.tolist()], dim=0)
                z_synth = universal_signal(torch.zeros(0, K, device='cuda'),
                                           sigma_override=global_sigma, z_proxy=z_proxy,
                                           seed=seed + int(k), jitter=jitter)
            else:
                z_synth = universal_signal(z, seed=seed + int(k), jitter=jitter)
            x_synth = (mu + z_synth @ P.T).float()
            experts = load_selected_experts(L, [int(k)])
            w1, w2, w3 = experts[int(k)]
            y_synth = ffn_exact(x_synth, w1, w2, w3)
            del experts, x_synth
            z_all = torch.cat([z, z_synth])
            y_all = torch.cat([yf, y_synth])
            if mode == 'config':
                # Pass 1: pick the size/error-optimal (inter, kp) at short steps.
                inter, kp, res, resid = config_search(
                    z, yf, z_all, y_all, Q0, e_prev, n_k, config_steps, config_margin,
                    tier_thr, use_compile, warm, rw_eff)
                w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs = res
            else:
                # Pass 2: known config from file, full budget to minimize error.
                inter, kp = read_config(e_prev, n_k)
                init = warm_init_from(e_prev, inter, kp) if (warm and e_prev is not None) else None
                results = train_batch([(z_all, y_all, Q0[:, :kp])], inter, tier_steps, kp=kp,
                                      stop_threshold=tier_thr, n_real=z.shape[0], use_compile=use_compile,
                                      init=init, real_weight=rw_eff)
                w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs = results[0]
                resid = resid_weights_full(z, yf, Qq, Qs, w1q, w1s, w3q, w3s, w2q, w2s)
            save_expert(k, resid, w1q, w1s, w3q, w3s, w2q, w2s, Qq, Qs, inter, n_real=n_k)
            n_fixed += 1
            print(f'  layer {L} expert {k} [tier {tier}, n={n_k}]: resid={resid*100:.4f}%  '
                  f'({n_fixed} total, {time.time()-t0:.0f}s)', flush=True)
            torch.cuda.empty_cache()

        # --- Missing: proxy + synthetic, one expert at a time (13%) ---
        # Fixed config (2048,512) — no config search needed, so pass 1 (config)
        # skips them; they train only in pass 2 (refine).
        if mode == 'refine':
            for (k, z, yf, Q0, _e_prev) in missing_todo:
                inter, kp = 2048, 512
                sim = covered_rw @ rw_n[int(k)]  # [n_covered]
                top = sim.topk(min(8, sim.shape[0])).indices
                z_proxy = torch.cat([z_by_key[str(covered_int[j])] for j in top.tolist()], dim=0)
                z_synth = universal_signal(torch.zeros(0, K, device='cuda'),
                                           sigma_override=global_sigma, z_proxy=z_proxy, seed=seed + int(k), jitter=jitter)
                x_synth = (mu + z_synth @ P.T).float()
                experts = load_selected_experts(L, [int(k)])
                w1, w2, w3 = experts[int(k)]
                y_synth = ffn_exact(x_synth, w1, w2, w3)
                del experts, x_synth
                Q0m = safe_svd_q(y_synth, kp)
                if Q0m.shape[1] < kp:
                    Q0m = torch.cat([Q0m, torch.zeros(D, kp - Q0m.shape[1], device='cuda')], dim=1)
                results = train_batch([(z_synth, y_synth, Q0m[:, :kp])], inter, steps, kp=kp,
                                      stop_threshold=None, n_real=0, use_compile=use_compile)
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
    ap.add_argument('--all', action='store_true', help='refit all experts, not only > 0.1%%')
    ap.add_argument('--done-log', default='refit_done_q.txt')
    ap.add_argument('--refit-threshold', type=float, default=1e-4,
                    help='accept an expert when its residual <= this (fraction, 0.01%% default)')
    ap.add_argument('--n-procs', type=int, default=4,
                    help='split layers across N parallel processes (0=auto=min(4,n_layers))')
    ap.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True,
                    help='torch.compile the training forward step (default: on; --no-compile to disable. '
                         'Guard raises on non-finite grads)')
    ap.add_argument('--warm', action='store_true',
                    help='warm-start the ternary core from the saved expert file (iterative refinement passes)')
    ap.add_argument('--seed', type=int, default=0,
                    help='unifold signal seed offset (vary per refinement pass for data diversity)')
    ap.add_argument('--real-weight', type=float, default=800.0,
                    help='loss weight multiplier for REAL rows (vs 1.0 synthetic). '
                         'Fixes the synth-dominance: 24 real rows = 1%% of loss otherwise')
    ap.add_argument('--jitter', type=float, default=0.2,
                    help='unifold bootstrap jitter magnitude (fraction of per-dim sigma)')
    ap.add_argument('--mode', choices=['config', 'refine'], default='refine',
                    help='config = pass 1 (pick optimal inter/kp per expert); refine = pass 2 (known config, full budget)')
    ap.add_argument('--config-steps', type=int, default=500,
                    help='pass-1 config-search training steps per candidate (short; ranking is stable at ~100-500)')
    ap.add_argument('--config-margin', type=float, default=0.05,
                    help='pass-1 knee: pick smallest config within this relative margin of the best residual')
    args = ap.parse_args()

    n_procs = max(1, args.n_procs if args.n_procs > 0 else min(4, args.end_layer - args.start_layer))

    if n_procs == 1:
        run_refit(args.start_layer, args.end_layer, args.group, args.steps, args.all,
                  args.done_log, args.refit_threshold, use_compile=args.compile, warm=args.warm, seed=args.seed,
                  real_weight=args.real_weight, jitter=args.jitter, mode=args.mode,
                  config_steps=args.config_steps, config_margin=args.config_margin)
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
        if args.compile:
            cmd.append('--compile')
        if args.warm:
            cmd.append('--warm')
        cmd.extend(['--seed', str(args.seed), '--real-weight', str(args.real_weight),
                    '--jitter', str(args.jitter), '--mode', args.mode,
                    '--config-steps', str(args.config_steps), '--config-margin', str(args.config_margin)])
        # Per-process inductor/triton cache: 4 workers sharing one cache dir on
        # Windows hit PermissionError on kernel .hsaco writes (file locking).
        cache_root = os.path.join(os.environ.get('TEMP', '.'), f'torchinductor_p{i}')
        env = {**os.environ,
               'TORCHINDUCTOR_CACHE_DIR': cache_root,
               'TRITON_CACHE_DIR': os.path.join(cache_root, 'triton')}
        with open(f'refit_q_p{i}.log', 'w') as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        procs.append(p)
        print(f'proc {i}: layers {s}-{e} -> refit_q_p{i}.log (pid {p.pid})', flush=True)
    for p in procs:
        p.wait()
    print('all procs done', flush=True)


if __name__ == '__main__':
    main()
