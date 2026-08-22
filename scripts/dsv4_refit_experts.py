"""Parallel activation refit with LEARNABLE Q (joint ternary + Q).

Retrains the ternary core AND a learnable Q jointly from scratch on exact
per-expert activations, against the FULL 4096-dim target (warm-started from
the SVD Q). This replaces the old fixed-SVD-Q refit (reduced 384-dim loss)
and yields ~5x lower residual at the same model size.

Run multiple processes over disjoint layer ranges (--start-layer/--end-layer).
Resumable via a per-process --done-log (skip by full-space residual threshold).
"""

import argparse
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv4_experts as de
from dsv4_experts import (
    ffn as ffn_exact,
)
from dsv4_experts import (
    load_selected_experts,
    pack_ternary,
    ternarize,
    unpack_ternary,
)

K = 4096
INTER = 1024
KP = 4096
MAX_KP = 4096
Q_BITS = 4
Q_DIVISOR = 28
Q_LEVELS = 7
REFIT_CFGS = [
    (1024, 512),
]
D = 4096
POD = "checkpoints_dsv4/pod_all_tokens"
REDUCED = "dsv4_reduced"
DEAD_LOG = "refit_dead.txt"
M_SYNTH = 2048  # universal test-signal samples per expert (multi-tone + white noise)


_ROUTER_W_CACHE = {}


def load_router_weight(L):
    """Load the router gate weight [256, 4096] for layer L (cached)."""
    if L not in _ROUTER_W_CACHE:
        snap = de.default_snapshot()
        wm = de.load_index(snap)["weight_map"]
        p = f"layers.{L}.ffn.gate"
        _ROUTER_W_CACHE[L] = de.read_tensor(snap, wm, f"{p}.weight", device="cuda").to(torch.float32)
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
        return 2048, 4096
    return 2048, 4096


def config_candidates(n_k):
    """Fixed config (2048, 4096): kp == D (identity output), inter == original.
    No size sweep — measurements showed lowering inter/kp only adds loss."""
    return [(512, 4096), (768, 4096), (1024, 4096), (1536, 4096), (2048, 4096)]


def read_config(e, n_k):
    """Recover (inter, kp) from a saved expert file (pass 2 = known config).
    kp is now the fixed shared-KP (output basis is layer-global, not per-expert)."""
    if e is not None and "inter" in e:
        return int(e["inter"]), KP
    return pick_config(n_k)


def config_search(
    z, yf, z_all, y_all, Q_shared, e_prev, n_k, config_steps, config_margin, tier_thr, use_compile, warm, rw_eff
):
    """Pass 1: try candidate (inter,kp) at short steps, return the smallest-size
    config whose real residual is within config_margin of the best (Pareto knee).
    Returns (inter, kp, results_tuple, resid)."""
    cands = config_candidates(n_k)
    scored = []
    for inter, kp in cands:
        init = warm_init_from(e_prev, inter, kp) if (warm and e_prev is not None) else None
        results = train_batch(
            [(z_all, y_all)],
            inter,
            config_steps,
            kp=kp,
            Q_shared=Q_shared,
            stop_threshold=tier_thr,
            n_real=n_k,
            use_compile=use_compile,
            init=init,
            real_weight=rw_eff,
        )
        w1q, w1s, w1q2, w1s2, w3q, w3s, w3q2, w3s2, w2q, w2s, w2q2, w2s2 = results[0]
        resid = resid_weights_full(z, yf, Q_shared, w1q, w1s, w1q2, w1s2, w3q, w3s, w3q2, w3s2, w2q, w2s, w2q2, w2s2)
        scored.append((resid, inter, kp, results[0]))
        print(f"    [config] inter={inter} kp={kp} -> resid={resid * 100:.4f}%", flush=True)
    # knee: smallest size (inter*kp proxy) within margin of the best residual
    scored.sort(key=lambda t: (t[1] * t[2], t[0]))
    best_resid = min(s[0] for s in scored)
    for resid, inter, kp, res in scored:
        if resid <= best_resid * (1.0 + config_margin):
            return inter, kp, res, resid
    resid, inter, kp, res = scored[-1]
    return inter, kp, res, resid


def resid_weights_full(z, y_full, Q_shared, w1q, w1s, w1q2, w1s2, w3q, w3s, w3q2, w3s2, w2q, w2s, w2q2, w2s2):
    Kk = z.shape[1]
    w1 = w1q[:, :Kk] * w1s[:, None] + w1q2[:, :Kk] * w1s2[:, None]
    w3 = w3q[:, :Kk] * w3s[:, None] + w3q2[:, :Kk] * w3s2[:, None]
    w2 = w2q * w2s[:, None] + w2q2 * w2s2[:, None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        g = (z @ w1.T).clamp(max=10.0)
        u = (z @ w3.T).clamp(min=-10.0, max=10.0)
        yp = (F.silu(g) * u) @ w2.T
        if Q_shared is not None:
            yp = yp @ Q_shared.T
    return (F.mse_loss(yp.float(), y_full) / F.mse_loss(y_full, torch.zeros_like(y_full))).item()


def qste_bf16(W):
    """Ternary STE in bf16 (half the allocation vs fp32). W [G,out,in] fp32 -> [G,out,in] bf16."""
    Wb = W.to(torch.bfloat16)
    s1 = Wb.detach().abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
    q1 = (Wb / s1).clamp(-1, 1).round()
    r = Wb - q1 * s1
    s2 = r.detach().abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
    q2 = (r / s2).clamp(-1, 1).round()
    recon = q1 * s1 + q2 * s2
    return Wb + (recon - Wb).detach()


def qste_lsq(W, s1, s2):
    """2-level ternary STE with LEARNABLE per-output-channel scales (LSQ).
    forward = q1*s1 + q2*s2, dW = identity (STE), ds1 = q1, ds2 = q2.
    Returns bf16 (matches Zb in bmm)."""
    Wb = W.to(torch.bfloat16)
    s1b = s1.to(torch.bfloat16)
    s2b = s2.to(torch.bfloat16)
    q1 = (Wb.detach() / s1b.detach()).clamp(-1, 1).round()
    r = Wb.detach() - q1 * s1b.detach()
    q2 = (r / s2b.detach()).clamp(-1, 1).round()
    recon = q1 * s1b + q2 * s2b  # s1b/s2b differentiable → grad flows to scales
    return Wb + (recon - Wb).detach()


def ternarize2_fixed(W, s1, s2):
    """Export quantize with FIXED (learned) scales → (q1, s1, q2, s2), s [G,out]."""
    q1 = (W / s1).clamp(-1, 1).round()
    r = W - q1 * s1
    q2 = (r / s2).clamp(-1, 1).round()
    return q1, s1.squeeze(2), q2, s2.squeeze(2)


def ternarize2(W):
    """2-level ternary quantize -> (q1, s1, q2, s2), each q in {-1,0,1}, s [G,out]."""
    s1 = W.abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
    q1 = (W / s1).clamp(-1, 1).round()
    r = W - q1 * s1
    s2 = r.abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
    q2 = (r / s2).clamp(-1, 1).round()
    return q1, s1.squeeze(2), q2, s2.squeeze(2)


def lloyd_2t_scale(W, n_iter=20):
    """Lloyd (k-means) closed-form scales for 2-level ternary, W [G,out,in].
    s1 = (q1·W)/(q1·q1), s2 likewise — minimizes weight error better than mean|W|."""
    s1 = W.abs().mean(dim=2, keepdim=True).clamp_min(1e-8)
    for _ in range(n_iter):
        q1 = (W / s1).clamp(-1, 1).round()
        s1 = (q1 * W).sum(dim=2, keepdim=True) / (q1 * q1).sum(dim=2, keepdim=True).clamp_min(1e-8)
    q1 = (W / s1).clamp(-1, 1).round()
    r = W - q1 * s1
    s2 = r.abs().mean(dim=2, keepdim=True).clamp_min(1e-8)
    for _ in range(n_iter):
        q2 = (r / s2).clamp(-1, 1).round()
        s2 = (q2 * r).sum(dim=2, keepdim=True) / (q2 * q2).sum(dim=2, keepdim=True).clamp_min(1e-8)
    return s1, s2


def zeropower(G, steps=2):
    """Newton-Schulz orthogonalization of each [m,n] matrix (Muon core).
    steps=2: at K=4096/kp=2048, steps=3 is 30-40% slower (Qp [4096,2048]:
    51ms vs 36ms) with no measurable quality gain. The old K=512 4-process
    refit showed identical throughput for steps=2 vs 3 — so 2 is enough.
    Returns bf16 (measured: cos 0.99982 vs fp32-out, rel diff ~0; weights are
    ternary anyway) — the fp32 cast cost 14 ms/step of pure overhead."""
    a, m, n = G.shape
    X = G.to(torch.bfloat16)
    X = X / (X.norm(dim=(1, 2), keepdim=True).clamp_min(1e-7))
    if m <= n:
        for _ in range(steps):
            X = 1.5 * X - 0.5 * (X @ X.transpose(1, 2)) @ X
    else:
        for _ in range(steps):
            X = 1.5 * X - 0.5 * X @ (X.transpose(1, 2) @ X)
    return X


def pack_int4(q):
    """q [D, KP] int8 (-7..7) -> uint8 [D, KP//2] (two nibbles per byte, +8 offset)."""
    q = q[:, : q.shape[1] // 2 * 2]
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


def train_batch_fwd(Zb, W1, W3, W2, Q_shared=None, scales=None):
    """Module-level forward. Q_shared [D, kp] is the FIXED shared output basis
    (int8 * per-column scale), identical for all experts in the layer.
    None = identity (kp == D), so the output projection is skipped.
    scales = (s1_1,s2_1,s1_3,s2_3,s1_2,s2_2) learnable LSQ scales (None → mean|W|)."""
    if scales is not None:
        s1_1, s2_1, s1_3, s2_3, s1_2, s2_2 = scales
        g = torch.bmm(Zb, qste_lsq(W1, s1_1, s2_1).transpose(1, 2)).clamp(max=10.0)
        u = torch.bmm(Zb, qste_lsq(W3, s1_3, s2_3).transpose(1, 2)).clamp(min=-10.0, max=10.0)
        h = F.silu(g) * u
        yp = torch.bmm(h, qste_lsq(W2, s1_2, s2_2).transpose(1, 2))  # [G,bs,kp]
    else:
        g = torch.bmm(Zb, qste_bf16(W1).transpose(1, 2)).clamp(max=10.0)
        u = torch.bmm(Zb, qste_bf16(W3).transpose(1, 2)).clamp(min=-10.0, max=10.0)
        h = F.silu(g) * u
        yp = torch.bmm(h, qste_bf16(W2).transpose(1, 2))  # [G,bs,kp]
    if Q_shared is not None:
        yp = torch.bmm(yp, Q_shared.to(torch.bfloat16).T.unsqueeze(0).expand(yp.shape[0], -1, -1))
    return yp


def train_batch(
    pairs,
    inter,
    steps,
    kp=KP,
    Q_shared=None,
    check_every=25,
    patience=4000,
    stop_threshold=None,
    n_real: int | list[int] | tuple[int, ...] = 0,
    use_compile=False,
    init=None,
    stall_checks=200,
    stall_tol=0.02,
    real_weight=1.0,
):
    """pairs: list of (z_k [n_k,K], y_k [n_k,4096])
    -> list of (w1q,w1s,w3q,w3s,w2q,w2s).
    Trains the ternary core against the full 4096-dim target through the FIXED
    shared output basis Q_shared [D,kp] (one per layer, not per expert).
    n_real: first n_real rows of each pair are the REAL samples; when set, the
    early-stop threshold is evaluated on those rows only (honest metric)."""
    G = len(pairs)
    if isinstance(n_real, (list, tuple)):
        n_real = [int(v) for v in n_real]
    else:
        n_real = [int(n_real)] * G
    Nmax = max(z.shape[0] for z, _ in pairs)
    Z = torch.zeros(G, Nmax, K, device="cuda", dtype=torch.bfloat16)
    Y = torch.zeros(G, Nmax, D, device="cuda", dtype=torch.bfloat16)
    M = torch.zeros(G, Nmax, 1, device="cuda")
    for i, (z, y) in enumerate(pairs):
        Z[i, : z.shape[0]] = z.to(torch.bfloat16)
        Y[i, : y.shape[0]] = y.to(torch.bfloat16)
        M[i, : z.shape[0]] = 1.0

    W1 = torch.nn.Parameter(torch.randn(G, inter, K, device="cuda") * K**-0.5)
    W3 = torch.nn.Parameter(torch.randn(G, inter, K, device="cuda") * K**-0.5)
    W2 = torch.nn.Parameter(torch.randn(G, kp, inter, device="cuda") * inter**-0.5)
    if init is not None:
        # Warm-start the ternary core from a previous pass (packed ternary unpacked
        # to float). Shapes must match: (G,inter,K)/(G,inter,K)/(G,kp,inter).
        iW1, iW3, iW2 = init
        if iW1.shape == W1.shape and iW3.shape == W3.shape and iW2.shape == W2.shape:
            W1.data.copy_(iW1)
            W3.data.copy_(iW3)
            W2.data.copy_(iW2)
            print("    [warm] ternary core initialized from previous pass", flush=True)
        else:
            print(f"    [warm] shape mismatch ({tuple(iW1.shape)} vs {tuple(W1.shape)}) — random init", flush=True)
    # Learnable per-output-channel scales (LSQ) — Lloyd closed-form init (better than mean|W|).
    s1_1 = torch.nn.Parameter(lloyd_2t_scale(W1.detach())[0])
    s2_1 = torch.nn.Parameter(lloyd_2t_scale(W1.detach())[1])
    s1_3 = torch.nn.Parameter(lloyd_2t_scale(W3.detach())[0])
    s2_3 = torch.nn.Parameter(lloyd_2t_scale(W3.detach())[1])
    s1_2 = torch.nn.Parameter(lloyd_2t_scale(W2.detach())[0])
    s2_2 = torch.nn.Parameter(lloyd_2t_scale(W2.detach())[1])
    scales = [s1_1, s2_1, s1_3, s2_3, s1_2, s2_2]
    e1 = torch.zeros_like(W1)
    e3 = torch.zeros_like(W3)
    e2 = torch.zeros_like(W2)
    es = [torch.zeros_like(p) for p in scales]
    MU = 0.95
    bs = min(Nmax, 1024)
    best = None
    stall = 0
    ema = None
    best_state = None
    best_stop = torch.full((G,), float("inf"), device="cuda")
    # Honest-EMA stall tracking (covered experts that plateau above the threshold):
    # stop after stall_checks honest checks without >stall_tol relative improvement.
    ema_h = None
    ref_h = None
    stall_ct = torch.zeros(G, device="cuda")
    n_real_t = torch.tensor(n_real, device="cuda")
    done_mask = torch.zeros(G, dtype=torch.bool, device="cuda")

    def update_best(metric, valid=None):
        nonlocal best_state, best_stop
        with torch.no_grad():
            if valid is None:
                valid = torch.ones(G, dtype=torch.bool, device="cuda")
            if valid.any():
                cand = torch.where(valid, metric.detach(), torch.full_like(best_stop, float("inf")))
                better = cand < best_stop
                if better.any():
                    best_stop = torch.where(better, cand, best_stop)
                    W1d, W3d, W2d = W1.detach(), W3.detach(), W2.detach()
                    sd = [p.detach() for p in scales]
                    if best_state is None:
                        best_state = (W1d.clone(), W3d.clone(), W2d.clone()) + tuple(p.clone() for p in sd)
                    best_state[0][better] = W1d[better]
                    best_state[1][better] = W3d[better]
                    best_state[2][better] = W2d[better]
                    for bi, p in enumerate(sd):
                        best_state[3 + bi][better] = p[better]

    fwd_plain = train_batch_fwd  # non-compiled: full-batch eval below varies N per expert
    if use_compile:
        fwd = torch.compile(train_batch_fwd)
        print("    [compile] torch.compile enabled (first steps compile the graph)", flush=True)
    else:
        fwd = train_batch_fwd

    t0 = time.time()
    for st in range(steps):
        if Nmax > bs:
            idx = torch.randint(0, Nmax, (bs,), device="cuda")
            Zb, Yb, Mb = Z[:, idx], Y[:, idx], M[:, idx]
        else:
            idx = torch.arange(Nmax, device="cuda")
            Zb, Yb, Mb = Z, Y, M
        yp = fwd(Zb, W1, W3, W2, Q_shared, scales)
        diff2 = (yp - Yb) ** 2
        yb2 = Yb**2
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
            rw = torch.where(
                idx[None, :] < n_real_t[:, None],
                torch.full((1,), real_weight, device="cuda"),
                torch.ones((1,), device="cuda"),
            ).float()  # [G, bs]
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
        W1.grad = None
        W3.grad = None
        W2.grad = None
        loss.backward()
        g1, g3, g2 = W1.grad, W3.grad, W2.grad
        assert g1 is not None and g3 is not None and g2 is not None
        if use_compile and ((st + 1) % check_every == 0 or st == 0):
            # ROCm torch.compile bug guard: non-finite grads would silently skip
            # updates (see BENCHMARKS compile+batch>=320). Fail loudly instead.
            if not (torch.isfinite(g1).all() and torch.isfinite(g3).all() and torch.isfinite(g2).all()):
                raise RuntimeError(
                    f"non-finite grads at step {st + 1} under torch.compile "
                    "(known ROCm buffer-reuse bug) — rerun without --compile"
                )
        if done_mask.any():
            with torch.no_grad():
                g1[done_mask] = 0
                g3[done_mask] = 0
                g2[done_mask] = 0
        with torch.no_grad():
            lr_m = 0.05 * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * st / steps)).item())
            e1.mul_(MU).add_(g1, alpha=1 - MU)
            e3.mul_(MU).add_(g3, alpha=1 - MU)
            e2.mul_(MU).add_(g2, alpha=1 - MU)
            W1.data -= lr_m * zeropower(e1)
            W3.data -= lr_m * zeropower(e3)
            W2.data -= lr_m * zeropower(e2)
            for p, ep in zip(scales, es):
                if p.grad is not None:
                    ep.mul_(MU).add_(p.grad, alpha=1 - MU)
                    p.data -= lr_m * ep
        resid_med = resid.median().item()
        ema = resid_med if ema is None else 0.9 * ema + 0.1 * resid_med
        if stop_threshold is None:
            # Stall-based early stop (synthetic/uncovered experts only).
            # Gate on the PER-EXPERT BEST (what actually gets saved), not on the
            # lagging EMA of the batch median — the median oscillates while the
            # min-state still improves, causing premature stops.
            cur_best = best_stop.min().item()
            if not (cur_best < float("inf")):
                stall = 0  # best_stop not measured yet (nan/inf) — don't count
            elif best is None or cur_best < best - max(1e-5, best * stall_tol):
                best = cur_best
                stall = 0
            else:
                stall += 1
            if stall >= patience:
                assert best is not None  # stall only increments after best set
                print(f"    stalled at step {st + 1}/{steps} (best resid {best * 100:.4f}%)", flush=True)
                break
        # Honest early-stop metric: per-expert residual on the REAL rows only.
        # Computed every check_every steps (not every step) to roughly halve the
        # forward cost; best-state tracks the per-expert minimum on that grid.
        honest = (n_real_t.sum() == 0) or (st == steps - 1) or ((st + 1) % check_every == 0)
        stop_resid = resid  # [G] synthetic residual (fallback when not honest)
        if stop_threshold is not None and n_real_t.sum() > 0 and honest:
            nrm = int(n_real_t.max().item())
            Zc, Yc = Z[:, :nrm], Y[:, :nrm]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                gr = torch.bmm(Zc, qste_lsq(W1, s1_1, s2_1).transpose(1, 2)).clamp(max=10.0)
                ur = torch.bmm(Zc, qste_lsq(W3, s1_3, s2_3).transpose(1, 2)).clamp(min=-10.0, max=10.0)
                ypr = torch.bmm(F.silu(gr) * ur, qste_lsq(W2, s1_2, s2_2).transpose(1, 2))
                if Q_shared is not None:
                    ypr = torch.bmm(ypr, Q_shared.to(torch.bfloat16).T.unsqueeze(0).expand(gr.shape[0], -1, -1))
            real_mask = torch.arange(nrm, device="cuda")[None, :] < n_real_t[:, None]  # [G, nrm]
            num_r = (real_mask[:, :, None] * (ypr - Yc) ** 2).sum(dim=(1, 2)).float()
            den_r = (real_mask[:, :, None] * Yc**2).sum(dim=(1, 2)).float().clamp_min(1e-12)
            stop_resid = num_r / den_r  # [G]
        # Per-expert best-state (min honest stop_resid), not the final one.
        if honest and stop_threshold is None and ((st + 1) % check_every == 0 or st == steps - 1):
            # tier C / missing: a 1-row real metric is noise — select best state
            # by the deterministic full-batch manifold residual instead (same
            # metric the stall gate watches).
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ypf = fwd_plain(Z, W1, W3, W2, Q_shared, scales)
                d2 = (M * (ypf - Y) ** 2).sum(dim=(1, 2)).float()
                y2 = (M * Y**2).sum(dim=(1, 2)).float().clamp_min(1e-12)
                update_best(d2 / y2)
        elif honest:
            update_best(stop_resid)
        if stop_threshold is not None and honest:
            done_mask |= stop_resid <= stop_threshold
            if done_mask.all():
                print(
                    f"    early stop at step {st + 1}/{steps} (all {G} experts <= {stop_threshold * 100:.3f}%)",
                    flush=True,
                )
                break
            # Stall-stop: covered experts stuck on a plateau above the threshold
            # burn the whole budget otherwise. EMA of the honest residual must
            # improve by >stall_tol (relative) within stall_checks honest checks.
            with torch.no_grad():
                if ema_h is None:
                    ema_h = stop_resid.clone()
                    ref_h = stop_resid.clone()
                else:
                    assert ref_h is not None  # set together with ema_h
                    ema_h = 0.9 * ema_h + 0.1 * stop_resid
                    improved = ema_h < ref_h * (1.0 - stall_tol)
                    ref_h = torch.minimum(ref_h, ema_h)
                    stall_ct = torch.where(improved & (~done_mask), torch.zeros_like(stall_ct), stall_ct + 1.0)
                    newly = (~done_mask) & (stall_ct >= stall_checks) & (best_stop > stop_threshold)
                    if newly.any():
                        done_mask |= newly
                        print(
                            f"    stall-stop {int(newly.sum())} expert(s) at step {st + 1} "
                            f"(plateau best {best_stop[newly].min().item() * 100:.4f}%)",
                            flush=True,
                        )
                        if done_mask.all():
                            break
        if (st + 1) % check_every == 0:
            if (st + 1) % (check_every * 20) == 0:
                import os as _os

                import psutil

                _p = psutil.Process(_os.getpid())
                print(
                    f"    [mem] step {st + 1} rss={_p.memory_info().rss / 2**30:.2f}GB "
                    f"alloc={torch.cuda.memory_allocated() / 2**30:.2f}GB "
                    f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GB",
                    flush=True,
                )
            print(
                f"    step {st + 1}/{steps}  loss={loss.item():.6f}  "
                f"resid med={resid.median().item() * 100:.4f}%  "
                f"ETA {(time.time() - t0) / (st + 1) * (steps - st) / 60:.1f} min",
                flush=True,
            )

    # Restore the best state (min honest residual) instead of the final one.
    if best_state is not None:
        W1.data.copy_(best_state[0])
        W3.data.copy_(best_state[1])
        W2.data.copy_(best_state[2])
        for bi, p in enumerate(scales):
            p.data.copy_(best_state[3 + bi])
    if os.environ.get("REFIT_DEBUG_QUANT") == "1":
        nrm = int(n_real_t.max().item()) if n_real_t.numel() else 0
        if nrm > 0:
            Zc, Yc = Z[:, :nrm], Y[:, :nrm]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                gr = torch.bmm(Zc, qste_lsq(W1, s1_1, s2_1).transpose(1, 2)).clamp(max=10.0)
                ur = torch.bmm(Zc, qste_lsq(W3, s1_3, s2_3).transpose(1, 2)).clamp(min=-10.0, max=10.0)
                ypr = torch.bmm(F.silu(gr) * ur, qste_lsq(W2, s1_2, s2_2).transpose(1, 2))
                if Q_shared is not None:
                    ypr = torch.bmm(ypr, Q_shared.to(torch.bfloat16).T.unsqueeze(0).expand(gr.shape[0], -1, -1))
            rmask = torch.arange(nrm, device="cuda")[None, :] < n_real_t[:, None]
            r_bf16 = (rmask[:, :, None] * (ypr - Yc) ** 2).sum(dim=(1, 2)).float() / (rmask[:, :, None] * Yc**2).sum(
                dim=(1, 2)
            ).float().clamp_min(1e-12)
            tq1 = ternarize(W1.detach())
            tq3 = ternarize(W3.detach())
            tq2 = ternarize(W2.detach())
            w1f = tq1[0][0] * tq1[1][0][:, None]
            w3f = tq3[0][0] * tq3[1][0][:, None]
            w2f = tq2[0][0] * tq2[1][0][:, None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                g = (Zc[0] @ w1f[:, :K].T).clamp(max=10.0)
                u = (Zc[0] @ w3f[:, :K].T).clamp(min=-10.0, max=10.0)
                yp = (F.silu(g) * u) @ w2f.T
                if Q_shared is not None:
                    yp = yp @ Q_shared.float().T
            r_fp32 = (
                F.mse_loss(yp.float(), Yc[0].float()) / F.mse_loss(Yc[0].float(), torch.zeros_like(Yc[0].float()))
            ).item()
            print(
                f"    [quant-debug] best_stop={best_stop[0].item() * 100:.4f}%  "
                f"bf16_resid={r_bf16[0].item() * 100:.4f}%  fp32_resid={r_fp32 * 100:.4f}%",
                flush=True,
            )
    w1q, w1s, w1q2, w1s2 = ternarize2_fixed(W1.detach(), s1_1.detach(), s2_1.detach())
    w3q, w3s, w3q2, w3s2 = ternarize2_fixed(W3.detach(), s1_3.detach(), s2_3.detach())
    w2q, w2s, w2q2, w2s2 = ternarize2_fixed(W2.detach(), s1_2.detach(), s2_2.detach())
    out = []
    for i in range(G):
        out.append(
            (w1q[i], w1s[i], w1q2[i], w1s2[i], w3q[i], w3s[i], w3q2[i], w3s2[i], w2q[i], w2s[i], w2q2[i], w2s2[i])
        )
    return out


def warm_init_from(e, inter, kp):
    """Unpack a saved expert's ternary core into float init tensors for train_batch.
    Returns None if shapes/keys are unusable."""
    try:
        if "w1" not in e or "w3" not in e or "w2" not in e or e.get("inter") != inter:
            return None
        w1 = (
            unpack_ternary(e["w1"]).float() * e["w1_scale"].float()[:, None]
            + unpack_ternary(e["w1_q2"]).float() * e["w1_scale2"].float()[:, None]
        )
        w3 = (
            unpack_ternary(e["w3"]).float() * e["w3_scale"].float()[:, None]
            + unpack_ternary(e["w3_q2"]).float() * e["w3_scale2"].float()[:, None]
        )
        w2 = (
            unpack_ternary(e["w2"]).float() * e["w2_scale"].float()[:, None]
            + unpack_ternary(e["w2_q2"]).float() * e["w2_scale2"].float()[:, None]
        )
        if w2.shape[0] != kp:
            return None
        # trim base-3 packing pads: K' is rounded up to a multiple of 5
        w1 = w1[:, :K]
        w3 = w3[:, :K]
        w2 = w2[:, : e["inter"]]
        if w1.shape[1] != K or w2.shape[1] != e["inter"]:
            return None
        return (w1[None], w3[None], w2[None])
    except Exception:
        return None


def run_refit(
    start_layer,
    end_layer,
    group,
    steps,
    all_flag,
    done_log,
    refit_threshold=2e-4,
    use_compile=False,
    warm=False,
    seed=0,
    real_weight=800.0,
    jitter=0.2,
    mode="refine",
    config_steps=2000,
    config_margin=0.05,
):
    """Router-based refit: covered experts learn on their REAL routed activations
    (honest channel). Uncovered (noisy) experts learn on a proxy manifold built from
    the nearest covered experts by router-weight cosine similarity."""
    for L in range(start_layer, end_layer):
        p_path = os.path.join(REDUCED, f"layer_{L}", "P.pt")
        if not os.path.exists(p_path):
            continue
        P = torch.load(p_path, map_location="cuda").float()
        mu = torch.load(os.path.join(REDUCED, f"layer_{L}", "mu.pt"), map_location="cuda").float()
        acts = torch.load(os.path.join(POD, f"acts_layer{L}.pt"), map_location="cuda", weights_only=False)
        print(f"layer {L}: loading {len(acts)} experts...", flush=True)

        todo = []
        skipped_z = {}  # z of experts skipped by resume — missing-branch proxy needs them
        for k, (x_k, y_k) in acts.items():
            key = f"{L}_{k}"
            ep = os.path.join(REDUCED, f"layer_{L}", f"expert_{k}.pt")
            n_k = x_k.shape[0]
            if n_k > 1024:
                idx = torch.randperm(n_k)[:1024]
                x_k = x_k[idx]
                y_k = y_k[idx]
            e = None
            if os.path.exists(ep):
                e = torch.load(ep, map_location="cuda", weights_only=False)
                rc = e.get("residual", float("inf"))
                if (
                    mode == "refine"
                    and not all_flag
                    and isinstance(rc, (int, float))
                    and rc < refit_threshold
                    and e.get("Q_bits", 0) == Q_BITS
                ):
                    with open(done_log, "a") as f:
                        f.write(f"{key}\n")
                    skipped_z[k] = (x_k.float().cuda() - mu) @ P
                    continue
            z = (x_k.float().cuda() - mu) @ P
            y_full = y_k.float().cuda()
            todo.append((k, z, y_full, e))

        covered = set(acts.keys())
        for k in range(256):
            sk = str(k)
            if sk not in covered:
                todo.append((sk, None, None, None))

        # --- kp == D (4096): output basis is identity, no Q needed ---
        Q_shared = None
        print(f"layer {L}: kp=D -> identity output (no Q)", flush=True)

        del acts
        torch.cuda.empty_cache()
        print(
            f"layer {L}: collected {len(todo)} experts to refit "
            f"({len(covered)} real + {len(todo) - len(covered)} synthetic)",
            flush=True,
        )

        if not todo:
            print(f"layer {L}: nothing to refit", flush=True)
            continue

        todo.sort(key=lambda t: 0 if t[1] is None else t[1].shape[0])
        zs = [t[1] for t in todo if t[1] is not None]
        z_by_key = {t[0]: t[1] for t in todo if t[1] is not None}
        z_by_key.update(skipped_z)
        if zs:
            global_sigma = torch.stack([zz.float().std(dim=0, unbiased=False) for zz in zs]).mean(dim=0).clamp(0.1, 2.0)
        else:
            global_sigma = torch.ones(K, device="cuda")
        # Router weights for uncovered-expert proxy selection.
        rw = load_router_weight(L)  # [256, 4096] float32 cuda
        rw_n = rw / (rw.norm(dim=1, keepdim=True).clamp_min(1e-8))
        covered_int = sorted(int(kk) for kk in covered)
        covered_rw = rw_n[covered_int]  # [n_covered, 4096]

        t0 = time.time()
        n_fixed = 0

        def save_expert(k, resid, w1q, w1s, w1q2, w1s2, w3q, w3s, w3q2, w3s2, w2q, w2s, w2q2, w2s2, inter, n_real=None):
            ep = os.path.join(REDUCED, f"layer_{L}", f"expert_{k}.pt")
            e = torch.load(ep, map_location="cuda", weights_only=False) if os.path.exists(ep) else {}
            e["w1"] = pack_ternary(w1q).cpu()
            e["w1_scale"] = w1s.cpu()
            e["w1_q2"] = pack_ternary(w1q2).cpu()
            e["w1_scale2"] = w1s2.cpu()
            e["w3"] = pack_ternary(w3q).cpu()
            e["w3_scale"] = w3s.cpu()
            e["w3_q2"] = pack_ternary(w3q2).cpu()
            e["w3_scale2"] = w3s2.cpu()
            e["w2"] = pack_ternary(w2q).cpu()
            e["w2_scale"] = w2s.cpu()
            e["w2_q2"] = pack_ternary(w2q2).cpu()
            e["w2_scale2"] = w2s2.cpu()
            e["inter"] = inter
            e["residual"] = resid
            if n_real is not None:
                e["n_real"] = int(n_real)
            torch.save(e, ep)
            with open(done_log, "a") as f:
                f.write(f"{L}_{k}\n")

        covered_todo = [t for t in todo if t[1] is not None]
        missing_todo = [t for t in todo if t[1] is None]

        # --- Covered: batch tier-A experts (G=8) — measured 2.1x faster than
        # G=1 on this iGPU (33 vs 68 ms/expert-step), VRAM ~3 GB. Tier B stays
        # G=1 (synthetic manifold differs per expert, real_weight upweighting).
        # Pre-compute per-expert data first, then train in groups of BATCH_G.
        def _prep_covered(t):
            (k, z, yf, e_prev) = t
            n_k = z.shape[0]
            if n_k >= 128:
                tier_thr, tier_steps, tier = refit_threshold, steps, "A"
                rw_eff = 1.0
                z_synth = z[:0]
            elif n_k >= 8:
                tier_thr, tier_steps, tier = refit_threshold, steps, "B"
                rw_eff = real_weight
                z_synth = universal_signal(z, seed=seed + int(k), jitter=jitter)
            else:
                return None  # tier C: skip (noise)
            x_synth = (mu + z_synth @ P.T).float()
            experts = load_selected_experts(L, [int(k)])
            w1, w2, w3 = experts[int(k)]
            y_synth = ffn_exact(x_synth, w1, w2, w3)
            fp4_init = (w1[None].contiguous(), w3[None].contiguous(), w2[None].contiguous())
            del experts, x_synth
            z_all = torch.cat([z, z_synth])
            y_all = torch.cat([yf, y_synth])
            inter_e, kp_e = read_config(e_prev, n_k)
            # RAM/VRAM fix: staging ALL experts' data grew memory by tens of GB
            # (init alone ~100 MB fp32/expert x 250). Stage only the small fp4_init
            # (bf16 on CPU); warm init is re-loaded from file at chunk time.
            init = tuple(t.to(torch.bfloat16).cpu() for t in fp4_init)
            has_warm = bool(warm and e_prev is not None)
            del fp4_init
            return dict(
                k=k,
                n_k=n_k,
                tier=tier,
                tier_thr=tier_thr,
                tier_steps=tier_steps,
                rw_eff=rw_eff,
                inter=inter_e,
                kp=kp_e,
                z_rows=z.shape[0],
                z_all_rows=z_all.shape[0],
                has_warm=has_warm,
                init=init,
            )

        BATCH_G = int(os.environ.get("REFIT_BATCH_G", "8"))
        prepped_A = []
        prepped_B = []
        for t in covered_todo:
            p = _prep_covered(t)
            if p is None:
                continue
            (prepped_B if p["tier"] == "B" else prepped_A).append(p)

        def _run_group(group, todo_data, mode_refine=True):
            nonlocal n_fixed
            if not group:
                return
            # group by identical (inter,kp) so shapes match inside one G-batch
            from collections import defaultdict

            by_cfg = defaultdict(list)
            for p in group:
                by_cfg[(p["inter"], p["kp"])].append(p)
            for cfg, items in sorted(by_cfg.items()):
                for i in range(0, len(items), BATCH_G):
                    chunk = items[i : i + BATCH_G]
                    # Lazy staging: pull (z_all, y_all, init) from the per-layer
                    # todo store only for the active chunk, then free. This is
                    # what keeps RAM/VRAM flat instead of growing with the layer.
                    pairs = []
                    for p in chunk:
                        z_all_gpu, y_all_gpu = todo_data[p["k"]]
                        n_rows = p["z_all_rows"]
                        pairs.append((z_all_gpu[:n_rows], y_all_gpu[:n_rows]))
                    if chunk[0]["has_warm"]:
                        warm_inits = []
                        for p in chunk:
                            ep = os.path.join(REDUCED, f"layer_{L}", f"expert_{p['k']}.pt")
                            e_prev = (
                                torch.load(ep, map_location="cpu", weights_only=False) if os.path.exists(ep) else None
                            )
                            wi = warm_init_from(e_prev, p["inter"], p["kp"]) if e_prev is not None else None
                            warm_inits.append(wi if wi is not None else tuple(t for t in p["init"]))
                        init_gpu = tuple(torch.cat([wi[j].cpu() for wi in warm_inits]).cuda().float() for j in range(3))
                    else:
                        init_gpu = tuple(torch.cat([p["init"][j] for p in chunk]).cuda().float() for j in range(3))
                    results = train_batch(
                        pairs,
                        chunk[0]["inter"],
                        chunk[0]["tier_steps"],
                        kp=chunk[0]["kp"],
                        Q_shared=None,
                        stop_threshold=chunk[0]["tier_thr"],
                        n_real=[p["n_k"] for p in chunk],
                        use_compile=use_compile,
                        init=init_gpu,
                        real_weight=chunk[0]["rw_eff"],
                    )
                    for p, res in zip(chunk, results):
                        w1q, w1s, w1q2, w1s2, w3q, w3s, w3q2, w3s2, w2q, w2s, w2q2, w2s2 = res
                        z_h, yf_h = todo_data[p["k"]]
                        resid = resid_weights_full(
                            z_h, yf_h, None, w1q, w1s, w1q2, w1s2, w3q, w3s, w3q2, w3s2, w2q, w2s, w2q2, w2s2
                        )
                        save_expert(
                            p["k"],
                            resid,
                            w1q,
                            w1s,
                            w1q2,
                            w1s2,
                            w3q,
                            w3s,
                            w3q2,
                            w3s2,
                            w2q,
                            w2s,
                            w2q2,
                            w2s2,
                            p["inter"],
                            n_real=p["n_k"],
                        )
                        n_fixed += 1
                        print(
                            f"  layer {L} expert {p['k']} [tier {p['tier']}, n={p['n_k']}]: "
                            f"resid={resid * 100:.4f}%  ({n_fixed} total, {time.time() - t0:.0f}s)",
                            flush=True,
                        )
                    torch.cuda.empty_cache()

        if mode == "refine":
            # Lazy staging: todo_data holds the GPU (z, yf) per expert key from
            # the load phase; _run_group builds (z_all, y_all) per chunk. Tier A
            # has no synthetic rows; tier B is trained one-at-a-time below with
            # its own synthetic manifold.
            todo_data = {t[0]: (t[1], t[2]) for t in covered_todo}
            _run_group(prepped_A, todo_data)
            # Tier B: per-expert synthetic + upweighting — keep G=1 semantics.
            # Synth rows are built here (chunk time) and appended lazily.
            for p in prepped_B:
                z, yf = todo_data[p["k"]]
                z_synth = universal_signal(z, seed=seed + int(p["k"]), jitter=jitter)
                x_synth = (mu + z_synth @ P.T).float()
                experts = load_selected_experts(L, [int(p["k"])])
                w1, w2, w3 = experts[int(p["k"])]
                y_synth = ffn_exact(x_synth, w1, w2, w3)
                del experts, x_synth
                todo_data[p["k"]] = (torch.cat([z, z_synth]), torch.cat([yf, y_synth]))
                p["z_all_rows"] = todo_data[p["k"]][0].shape[0]
                _run_group([p], todo_data)
        else:
            # config pass: unchanged per-expert path (searches differ per expert)
            for t in covered_todo:
                (k, z, yf, e_prev) = t
                n_k = z.shape[0]
                if n_k >= 128:
                    tier_thr, tier_steps, tier = refit_threshold, steps, "A"
                    rw_eff = 1.0
                    z_synth = z[:0]
                elif n_k >= 8:
                    tier_thr, tier_steps, tier = refit_threshold, steps, "B"
                    rw_eff = real_weight
                    z_synth = universal_signal(z, seed=seed + int(k), jitter=jitter)
                else:
                    continue
                x_synth = (mu + z_synth @ P.T).float()
                experts = load_selected_experts(L, [int(k)])
                w1, w2, w3 = experts[int(k)]
                y_synth = ffn_exact(x_synth, w1, w2, w3)
                del experts, x_synth
                z_all = torch.cat([z, z_synth])
                y_all = torch.cat([yf, y_synth])
                inter, kp, res, resid = config_search(
                    z,
                    yf,
                    z_all,
                    y_all,
                    Q_shared,
                    e_prev,
                    n_k,
                    config_steps,
                    config_margin,
                    tier_thr,
                    use_compile,
                    warm,
                    rw_eff,
                )
                w1q, w1s, w1q2, w1s2, w3q, w3s, w3q2, w3s2, w2q, w2s, w2q2, w2s2 = res
                save_expert(
                    k, resid, w1q, w1s, w1q2, w1s2, w3q, w3s, w3q2, w3s2, w2q, w2s, w2q2, w2s2, inter, n_real=n_k
                )
                n_fixed += 1
                print(
                    f"  layer {L} expert {k} [tier {tier}, n={n_k}]: resid={resid * 100:.4f}%  "
                    f"({n_fixed} total, {time.time() - t0:.0f}s)",
                    flush=True,
                )
                torch.cuda.empty_cache()

        # --- Missing (n=0, never activated): skip — do not transfer to reduced ---
        if mode == "refine":
            for k, z, yf, _e_prev in missing_todo:
                with open(DEAD_LOG, "a") as f:
                    f.write(f"{L}_{k}\n")
                print(f"  layer {L} expert {k}: DEAD (0 activations) — skipped", flush=True)
        print(f"layer {L}: refit {n_fixed} experts in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=43)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--all", action="store_true", help="refit all experts, not only > 0.1%%")
    ap.add_argument("--done-log", default="refit_done_q.txt")
    ap.add_argument(
        "--refit-threshold",
        type=float,
        default=1e-4,
        help="accept an expert when its residual <= this (fraction, 0.01%% default)",
    )
    ap.add_argument(
        "--n-procs", type=int, default=4, help="split layers across N parallel processes (0=auto=min(4,n_layers))"
    )
    ap.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile the training forward step (default: off. Measured SLOWER on iGPU: 114ms vs 67ms/step). "
        "Guard raises on non-finite grads)",
    )
    ap.add_argument(
        "--warm",
        action="store_true",
        help="warm-start the ternary core from the saved expert file (iterative refinement passes)",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="unifold signal seed offset (vary per refinement pass for data diversity)"
    )
    ap.add_argument(
        "--real-weight",
        type=float,
        default=800.0,
        help="loss weight multiplier for REAL rows (vs 1.0 synthetic). "
        "Fixes the synth-dominance: 24 real rows = 1%% of loss otherwise",
    )
    ap.add_argument(
        "--jitter", type=float, default=0.2, help="unifold bootstrap jitter magnitude (fraction of per-dim sigma)"
    )
    ap.add_argument(
        "--mode",
        choices=["config", "refine"],
        default="refine",
        help="config = pass 1 (pick optimal inter/kp per expert); refine = pass 2 (known config, full budget)",
    )
    ap.add_argument(
        "--config-steps",
        type=int,
        default=500,
        help="pass-1 config-search training steps per candidate (short; ranking is stable at ~100-500)",
    )
    ap.add_argument(
        "--config-margin",
        type=float,
        default=0.05,
        help="pass-1 knee: pick smallest config within this relative margin of the best residual",
    )
    args = ap.parse_args()

    n_procs = max(1, args.n_procs if args.n_procs > 0 else min(4, args.end_layer - args.start_layer))

    if n_procs == 1:
        # Single-process runs get the same HIP-crash supervisor as the
        # multiprocess launcher: retry in-process up to 10 times (resume via
        # done-log makes restarts free). Root cause of the crash is a Windows
        # AMD driver fault (Event 41/6008 pattern), not our code.
        MAX_RETRIES = 10
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                run_refit(
                    args.start_layer,
                    args.end_layer,
                    args.group,
                    args.steps,
                    args.all,
                    args.done_log,
                    args.refit_threshold,
                    use_compile=args.compile,
                    warm=args.warm,
                    seed=args.seed,
                    real_weight=args.real_weight,
                    jitter=args.jitter,
                    mode=args.mode,
                    config_steps=args.config_steps,
                    config_margin=args.config_margin,
                )
                return
            except torch.AcceleratorError as ex:
                print(f"HIP crash (attempt {attempt}/{MAX_RETRIES}): {ex}", flush=True)
                if attempt == MAX_RETRIES:
                    raise
                print("cooling down 90s before restart (resume from done-log)...", flush=True)
                time.sleep(90)
                torch.cuda.empty_cache()
        return

    per = (args.end_layer - args.start_layer + n_procs - 1) // n_procs
    procs = []
    for i in range(n_procs):
        s = args.start_layer + i * per
        e = min(args.start_layer + (i + 1) * per, args.end_layer)
        if s >= e:
            break
        cmd = [
            sys.executable,
            "-u",
            os.path.abspath(__file__),
            "--start-layer",
            str(s),
            "--end-layer",
            str(e),
            "--group",
            str(args.group),
            "--steps",
            str(args.steps),
            "--done-log",
            args.done_log,
            "--n-procs",
            "1",
            "--refit-threshold",
            str(args.refit_threshold),
        ]
        if args.all:
            cmd.append("--all")
        if args.compile:
            cmd.append("--compile")
        if args.warm:
            cmd.append("--warm")
        cmd.extend(
            [
                "--seed",
                str(args.seed),
                "--real-weight",
                str(args.real_weight),
                "--jitter",
                str(args.jitter),
                "--mode",
                args.mode,
                "--config-steps",
                str(args.config_steps),
                "--config-margin",
                str(args.config_margin),
            ]
        )
        # Per-process inductor/triton cache: 4 workers sharing one cache dir on
        # Windows hit PermissionError on kernel .hsaco writes (file locking).
        cache_root = os.path.join(os.environ.get("TEMP", "."), f"torchinductor_p{i}")
        env = {
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": cache_root,
            "TRITON_CACHE_DIR": os.path.join(cache_root, "triton"),
        }
        with open(f"refit_q_p{i}.log", "a") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        procs.append((p, cmd, env))
        print(f"proc {i}: layers {s}-{e} -> refit_q_p{i}.log (pid {p.pid})", flush=True)
    # HIP driver faults (unspecified launch failure) kill a worker mid-run;
    # resume logic (done-log + threshold) makes restarts safe, so retry each
    # worker up to MAX_RETRIES times with a cooldown.
    MAX_RETRIES = 10
    retries = [0] * len(procs)
    while procs:
        alive = []
        for idx, (p, cmd, env) in enumerate(procs):
            rc = p.wait()
            if rc == 0:
                continue
            retries[idx] += 1
            if retries[idx] > MAX_RETRIES:
                print(f"proc {idx}: giving up after {MAX_RETRIES} retries (rc={rc})", flush=True)
                continue
            print(f"proc {idx}: crashed rc={rc}, retry {retries[idx]}/{MAX_RETRIES} in 60s", flush=True)
            time.sleep(60)
            with open(f"refit_q_p{idx}.log", "a") as f:
                f.write(f"\n=== supervisor restart {retries[idx]} ===\n")
                p2 = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            alive.append((p2, cmd, env))
        procs = alive
    print("all procs done", flush=True)


if __name__ == "__main__":
    main()
