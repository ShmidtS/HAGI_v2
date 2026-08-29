"""Parallel 1-bit BINARY refit (q = sign(W), learnable per-output-channel scale).

Retrains a binary core + learnable scale (LSQ) from scratch on exact
per-expert activations, against the FULL 4096-dim target (warm-started from
the FP4 expert weights). Binary weights = 1 bit/weight (~8x smaller than the
FP4 source, ~3.2x smaller than the old two-level ternary).

Run multiple processes over disjoint layer ranges (--start-layer/--end-layer).
Resumable via a per-process --done-log (skip by full-space residual threshold).
"""

import argparse
import math
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
    pack_binary,
    pack_int4,
)

K = 4096
INTER = 2048
D = 4096
NSUB = 1  # int4x: single full-K sub; W13 binary (1 bit), W2 int4 (4 bits) - 6.3 MB/expert
POD = "checkpoints_dsv4/pod_all_tokens"
REDUCED = "dsv4_reduced"
DEAD_LOG = "refit_bin_dead.txt"
M_SYNTH = 2048  # universal test-signal samples per expert (multi-tone + white noise)
MODE_MARKER = "int4x"  # binary W13 + int4 W2, single sub (post-hoc binary S=4+W2-1b was 21.1% vs this 9.9% pre-train)
CROSSED = True  # crossed pairing: approximates off-diagonal cross terms at zero extra bits
SCALE_LR_FACTOR = 0.5  # LSQ scales: lower LR than Muon weights (raw grad, ~INTERx more sensitive)
LR_BASE = 0.02  # Muon base LR (cosined); 0.04 measured worse on synthetic (35% vs 23% @300)
JITTER_BANK = 8  # precomputed (idx, Zj, Yb) minibatches: kills the per-step teacher
# forward (3 bmm); bank is regenerated every JITTER_REGEN steps for row coverage
JITTER_REGEN = 200
AM_WEIGHT = 0.0  # activation-matching (g/u/h) loss weight vs the output loss
DECORR_ALPHA = 0.25  # W2 replica init decorrelation noise (0 = identical copies)
S2_EVERY = 25  # exact per-channel s2 refit cadence: convex at fixed signs, churn-free,
# cheap (~0.2s) - can run often without disturbing feature co-adaptation
SIGN_EVERY = 25  # W2 pattern refresh cadence (dual-ridge Theta + binarize, guarded). RARE on
# purpose: each accepted flip changes the gradient path for features - frequent refresh
# keeps Muon momentum (0.95, ~20-step memory) perpetually stale (measured: every-10 worse)


SOFT_LIM = 10.0  # soft saturation: replaces hard clamp (normalization, no dead gradients)
SOFT_KNEE = 0.75  # exact identity below knee*lim, smooth tanh rolloff to lim above


def u_slice(s):
    """Input-slice index for the u (W3) branch of sub s (crossed pairing)."""
    return (s + 1) % NSUB if CROSSED else s


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

    Bootstrap + 10% jitter around real points (manifold coverage). For experts
    with no real samples, bootstrap from z_proxy (other experts of the layer);
    falls back to white noise when no proxy exists.
    """
    n, Kk = z_real.shape
    g = torch.Generator(device=z_real.device).manual_seed(seed)
    if sigma_override is not None:
        sigma = sigma_override
    elif n >= 8:
        sigma = z_real.std(dim=0).clamp(0.3, 2.0)
    else:
        sigma = torch.ones(Kk, device=z_real.device)
    if n == 0:
        if z_proxy is not None and z_proxy.shape[0] > 0:
            idx = torch.randint(0, z_proxy.shape[0], (M,), generator=g, device=z_real.device)
            eps = torch.randn(M, Kk, generator=g, device=z_real.device) * (jitter * sigma[None, :])
            return z_proxy[idx] + eps
        return torch.randn(M, Kk, generator=g, device=z_real.device) * sigma[None, :]
    idx = torch.randint(0, n, (M,), generator=g, device=z_real.device)
    z_base = z_real[idx]  # [M, K]
    eps = torch.randn(M, Kk, generator=g, device=z_real.device) * (jitter * sigma[None, :])
    return z_base + eps


def qste_bin(W, s):
    """1-bit binary STE with learnable per-output-channel scale.
    forward = sign(W) * s, dW = identity (STE, scaled by s), ds = sign(W).
    Returns bf16 (matches Zb in bmm)."""
    Wb = W.to(torch.bfloat16)
    sb = s.to(torch.bfloat16)
    q = torch.sign(Wb)
    q_ste = q.detach() + Wb - Wb.detach()  # value = sign(Wb), grad(dW) = identity
    return q_ste * sb


def qste_int4(W, s):
    """4-bit {-7..7} STE with learnable per-output-channel scale (W2 readout).
    forward = round(W).clamp(-7,7) * s, dW = identity (STE). Returns bf16."""
    Wb = W.to(torch.bfloat16)
    sb = s.to(torch.bfloat16)
    q = Wb.round().clamp(-7.0, 7.0)
    q_ste = q.detach() + Wb - Wb.detach()
    return q_ste * sb


def qint4_fixed(W):
    """int4 snap without STE (refresh/solve/export paths; W2 params are grid units)."""
    return W.round().clamp(-7.0, 7.0)


def binarize_fixed(W, s):
    """Export quantize with FIXED (learned) scale -> (q, s), q in {-1,+1}, s [G,out]."""
    q = torch.sign(W)
    return q, s.squeeze(2)


def zeropower(G, steps=2):
    """Newton-Schulz orthogonalization of each [m,n] matrix (Muon core). Returns bf16."""
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


def soft_lim(x, lim=SOFT_LIM, knee=SOFT_KNEE):
    """Signal normalization replacing hard clamp: exact identity below knee*lim,
    smooth tanh rolloff to lim above -> bounded signal, no dead gradients."""
    th = knee * lim
    tail = lim - th
    ax = x.abs()
    y_abs = torch.where(ax <= th, ax, th + tail * torch.tanh((ax - th) / tail))
    return y_abs * torch.sign(x)


def ptq_closed_form(w1_rot, w3_rot, w2, z_rows, y_rows, bias1, bias3, cd_rounds=3):
    # NOTE: w2 unused here (the readout is re-solved); kept for signature clarity.
    """Closed-form int4x encoder (AngelSlim STQ-style, no gradient steps):

    signs = sign(W_rot) (STE-free), LS scales, exact ridge solve for W2 on the
    given activations (imatrix analog: the functional error IS the metric),
    CD snap of W2, then one re-LS of W1/W3 scales + biases at the quantized W2.
    Returns (res, resid_before_snap_W2) in trainer format."""
    q1 = torch.sign(w1_rot)
    q3 = torch.sign(w3_rot)
    s1 = w1_rot.abs().mean(dim=1).clamp_min(1e-9)
    s3 = w3_rot.abs().mean(dim=1).clamp_min(1e-9)
    w1 = q1 * s1[:, None]
    w3 = q3 * s3[:, None]
    g = soft_lim(z_rows @ w1.T + bias1[None, :])
    u = soft_lim(z_rows @ w3.T + bias3[None, :])
    h = F.silu(g) * u  # [n, inter]
    # exact ridge solve for the continuous W2 readout
    Gm = h.T @ h
    Gm.diagonal().add_((Gm.diagonal().mean() * 1e-2))
    W2c = torch.linalg.solve(Gm, h.T @ y_rows).T.contiguous()  # [4096, inter]
    # CD snap of W2 to int4 (weighted by feature energy = the true metric)
    cw = (h ** 2).sum(dim=0)
    cw = cw / cw.mean().clamp_min(1e-12)
    sg = W2c.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 7.0
    q2 = (W2c / sg).round().clamp(-7, 7)
    for _ in range(cd_rounds):
        num = (cw * q2 * W2c).sum(dim=1, keepdim=True)
        den = (cw * q2 * q2).sum(dim=1, keepdim=True).clamp_min(1e-9)
        sg = num / den
        sg = torch.where(sg.abs() < 1e-9, torch.full_like(sg, 1e-9), sg)
        q2 = (W2c / sg).round().clamp(-7, 7)
    num = (cw * q2 * W2c).sum(dim=1)
    den = (cw * q2 * q2).sum(dim=1).clamp_min(1e-9)
    s2 = (num / den)
    s2 = s2.sign() * s2.abs().clamp_min(1e-6)
    # one re-LS of W1/W3 scales + biases at the QUANTIZED W2
    bounds = [0, z_rows.shape[1]]
    res = (q1, s1, q3, s3, q2, s2, bias1, bias3, bounds)
    return res, W2c


def resid_weights_full(z, y_full, res):
    """Pyramidal binary residual: sub s reads its equal-energy input slice
    (POD rank), full width, full output; sum of sub outputs vs exact target."""
    bounds = res[-1]
    wqs = res[: 6 * NSUB]  # 12 (q, s) pairs: w1,w3,w2 per sub, interleaved
    bss = res[6 * NSUB : -1]  # 8 biases: b1, b3 per sub
    yp = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(NSUB):
            w1q, w1s = wqs[6 * s], wqs[6 * s + 1]
            w3q, w3s = wqs[6 * s + 2], wqs[6 * s + 3]
            w2q, w2s = wqs[6 * s + 4], wqs[6 * s + 5]
            b1 = bss[2 * s].to(torch.bfloat16)
            b3 = bss[2 * s + 1].to(torch.bfloat16)
            zs = z[:, bounds[s] : bounds[s + 1]]
            us = z[:, bounds[u_slice(s)] : bounds[u_slice(s) + 1]]  # crossed pairing
            w1 = w1q * w1s[:, None]
            w3 = w3q * w3s[:, None]
            w2 = w2q * w2s[:, None]
            g = soft_lim(zs @ w1.T + b1)
            u = soft_lim(us @ w3.T + b3)
            ys = (F.silu(g) * u) @ w2.T
            yp = ys if yp is None else yp + ys
    assert yp is not None
    return (F.mse_loss(yp.float(), y_full) / F.mse_loss(y_full, torch.zeros_like(y_full))).item()


def train_batch_fwd(Zb, weights, scales, biases, bounds, max_sub=None, want_guh=True):
    """Pyramidal binary forward: sub s reacts to its EQUAL-ENERGY input slice
    (POD quantile bounds) at FULL hidden width, full output; y = sum of subs.
    max_sub=<s> restricts the sum to subs 0..s (boosting stages).
    want_guh=False skips the g/u/h stacks (unused when AM_WEIGHT == 0).
    weights = [W1_0, W3_0, W2_0, ...] ([G,inter,len_s] / [G,D,inter]),
    scales likewise [G,out,1], biases = [b1_0, b3_0, ...], bounds = int list.
    Returns (yp, g, u, h); g/u/h are sums over active subs (full inter)."""
    top = NSUB if max_sub is None else max_sub + 1
    yp = None
    gs, us, hs = [], [], []
    for s in range(top):
        W1, W3, W2 = weights[3 * s], weights[3 * s + 1], weights[3 * s + 2]
        s1, s3, s2 = scales[3 * s], scales[3 * s + 1], scales[3 * s + 2]
        b1 = biases[2 * s] if biases else None
        b3 = biases[2 * s + 1] if biases else None
        Zs = Zb[..., bounds[s] : bounds[s + 1]].contiguous()  # 1 copy vs 2 implicit bmm copies
        Zu = Zb[..., bounds[u_slice(s)] : bounds[u_slice(s) + 1]].contiguous()  # crossed pairing
        g = torch.bmm(Zs, qste_bin(W1, s1).transpose(1, 2))
        if b1 is not None:
            g = g + b1.to(torch.bfloat16)
        g = soft_lim(g)
        u = torch.bmm(Zu, qste_bin(W3, s3).transpose(1, 2))
        if b3 is not None:
            u = u + b3.to(torch.bfloat16)
        u = soft_lim(u)
        h = F.silu(g) * u
        ys = torch.bmm(h, qste_int4(W2, s2).transpose(1, 2))  # [G,bs,D]
        yp = ys if yp is None else yp + ys
        gs.append(g)
        us.append(u)
        hs.append(h)
    assert yp is not None
    if want_guh:
        return yp, torch.stack(gs).sum(0), torch.stack(us).sum(0), torch.stack(hs).sum(0)
    return yp, None, None, None


def train_batch(
    pairs,
    inter,
    steps,
    check_every=25,
    patience=4000,
    stop_threshold=None,
    n_real: int | list[int] | tuple[int, ...] = 0,
    use_compile=False,
    init=None,
    stall_checks=400,
    stall_tol=0.02,
    real_weight=1.0,
    bias1=None,
    bias3=None,
    jitter=0.05,
):
    """pairs: list of (z_k [n_k,K], y_k [n_k,4096]) -> list of (w1q,w1s,w3q,w3s,w2q,w2s).
    Trains the binary core against the full 4096-dim target. n_real: first n_real
    rows of each pair are the REAL samples (honest early-stop metric)."""
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

    # Pyramidal (equal-energy) input split: POD coords are energy-ordered, so
    # slice boundaries sit at cumulative-energy quantiles -> every sub carries
    # equal signal mass and each W2 replica (8.4 Mbit) serves equal energy.
    Zf = Z.float()
    Mf = M.float()
    cnt = Mf.sum(dim=1, keepdim=True).clamp_min(1.0)
    z_mean = (Zf * Mf).sum(dim=1, keepdim=True) / cnt
    z_var = ((Zf - z_mean).pow(2) * Mf).sum(dim=1, keepdim=True) / cnt
    z_std = z_var.sqrt().clamp_min(1e-6).to(torch.bfloat16)  # [G,K,1], jitter scale too
    # Equal quarters: the POD spectrum is flat (q10/q50/q90 = 0.197/0.208/0.222),
    # so quantile bounds are noise ([1027,1022,1026,1021]); fixed equal slices are
    # measured-equivalent AND give all subs identical shapes.
    bounds = [K * s // NSUB for s in range(NSUB + 1)]
    widths = [bounds[s + 1] - bounds[s] for s in range(NSUB)]
    print(f"    [pyramid] input slice widths {widths}", flush=True)

    weights = []
    for s in range(NSUB):
        w_in = widths[s]
        u_in = widths[u_slice(s)]  # crossed pairing: W3_s acts on the u-slice
        p1 = torch.nn.Parameter(torch.randn(G, inter, w_in, device="cuda") * w_in**-0.5)
        p3 = torch.nn.Parameter(torch.randn(G, inter, u_in, device="cuda") * u_in**-0.5)
        p2 = torch.nn.Parameter(torch.randn(G, D, inter, device="cuda") * inter**-0.5)
        p2.requires_grad_(False)  # convex subproblem: solved exactly by LS refresh, not Muon
        weights.extend([p1, p3, p2])
    if init is not None:
        iW1, iW3, iW2 = init
        if iW1.shape[-1] == K and iW3.shape[-1] == K and iW2.shape == (G, D, inter):
            for s in range(NSUB):
                weights[3 * s].data.copy_(iW1[..., bounds[s] : bounds[s + 1]])
                weights[3 * s + 1].data.copy_(iW3[..., bounds[u_slice(s)] : bounds[u_slice(s) + 1]])
                if s == 0 or DECORR_ALPHA <= 0.0:
                    weights[3 * s + 2].data.copy_(iW2)
                else:
                    # decorrelated replicas: identical inits start in the same
                    # basin; multiplicative noise breaks the symmetry gently
                    weights[3 * s + 2].data.copy_(iW2 * (1.0 + DECORR_ALPHA * torch.randn_like(iW2)))
            print(
                f"    [warm] pyramidal binary core init (POD slices, W2 replicas decorrelated alpha={DECORR_ALPHA})",
                flush=True,
            )
        else:
            print(f"    [warm] shape mismatch ({tuple(iW1.shape)}) - random init", flush=True)
    # W2 params live in int4 GRID units (round->{-7..7}); normalize so amax per
    # output channel lands on the grid edge (random init values would all round to 0)
    w2_grid_scale = []
    with torch.no_grad():
        for s in range(NSUB):
            p2 = weights[3 * s + 2]
            sg = p2.data.abs().amax(dim=2, keepdim=True).clamp_min(1e-9) / 7.0
            p2.data.div_(sg)
            w2_grid_scale.append(sg.detach())  # [G, D, 1] original amplitude per channel
    # Jacobian-aligned scale init per sub: per-channel LS matching the original's
    # response (Z@W^T vs Z@sign(W)^T) on each sub's OWN input quarter. No range clamp.
    with torch.no_grad():
        Mb = M.to(torch.bfloat16)

        def _ls(A, B):
            num = (Mb * A * B).sum(dim=1)  # [G, out]
            den = (Mb * B * B).sum(dim=1)  # [G, out]
            return (num / den.clamp_min(1e-6)).clamp_min(1e-5).unsqueeze(2)  # [G, out, 1]

        scale_vals = []
        for s in range(NSUB):
            W1d = weights[3 * s].detach().to(torch.bfloat16)
            W3d = weights[3 * s + 1].detach().to(torch.bfloat16)
            W2d = weights[3 * s + 2].detach().to(torch.bfloat16)
            Zs = Z[..., bounds[s] : bounds[s + 1]]
            Zu = Z[..., bounds[u_slice(s)] : bounds[u_slice(s) + 1]]
            gA = torch.bmm(Zs, W1d.transpose(1, 2))
            gB = torch.bmm(Zs, W1d.sign().transpose(1, 2))
            scale_vals.append(_ls(gA, gB))
            uA = torch.bmm(Zu, W3d.transpose(1, 2))
            uB = torch.bmm(Zu, W3d.sign().transpose(1, 2))
            scale_vals.append(_ls(uA, uB))
            h_o = F.silu(soft_lim(gA)) * soft_lim(uA)
            yA = torch.bmm(h_o, (W2d * w2_grid_scale[s]).to(torch.bfloat16).transpose(1, 2))
            yB = torch.bmm(h_o, qint4_fixed(W2d).to(torch.bfloat16).transpose(1, 2))
            scale_vals.append(_ls(yA, yB))
    scales = [torch.nn.Parameter(v.float()) for v in scale_vals]
    # Original (rotated, FULL-K) FP4 weights for on-the-fly target generation.
    # bf16 is exact for FP4-decoded values and ~2x faster on ROCm than fp32.
    if init is not None:
        oW1 = init[0].to(torch.bfloat16)
        oW3 = init[1].to(torch.bfloat16)
        oW2 = init[2].to(torch.bfloat16)
    else:
        with torch.no_grad():
            oW1 = torch.zeros(G, inter, K, device="cuda", dtype=torch.bfloat16)
            oW3 = torch.zeros(G, inter, K, device="cuda", dtype=torch.bfloat16)
            for s in range(NSUB):
                oW1[..., bounds[s] : bounds[s + 1]] = weights[3 * s].detach().to(torch.bfloat16)
                oW3[..., bounds[s] : bounds[s + 1]] = weights[3 * s + 1].detach().to(torch.bfloat16)
            oW2 = weights[2].detach().to(torch.bfloat16)
    # Folded target biases (mu @ W.T), FULL; every sub contributes to the SAME
    # channels, so each starts at bias/NSUB and the sum reconstructs the original.
    t_b1 = bias1.to(torch.bfloat16).unsqueeze(1) if bias1 is not None else None  # [G,1,inter]
    t_b3 = bias3.to(torch.bfloat16).unsqueeze(1) if bias3 is not None else None
    if bias1 is not None and bias3 is not None:
        biases = []
        for _ in range(NSUB):
            biases.append(torch.nn.Parameter((bias1 / NSUB).unsqueeze(1).clone()))  # [G,1,inter]
            biases.append(torch.nn.Parameter((bias3 / NSUB).unsqueeze(1).clone()))
        ebs = [torch.zeros_like(p) for p in biases]
    else:
        biases = []
        ebs = []
    # per-dim std (jitter scale) computed above with the pyramid bounds
    ews = [torch.zeros_like(w) for w in weights]
    es = [torch.zeros_like(p) for p in scales]
    MU = 0.95
    bs = min(Nmax, 1024)
    best = None
    stall = 0
    best_state = None
    best_stop = torch.full((G,), float("inf"), device="cuda")
    ema_h = None
    ref_h = None
    stall_ct = torch.zeros(G, device="cuda")
    n_real_t = torch.tensor(n_real, device="cuda")
    done_mask = torch.zeros(G, dtype=torch.bool, device="cuda")
    nonlocal_guard_ct = [0]  # non-finite-loss guard hits (boxed for closure mutability)

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
                    snap = [t.detach() for t in weights + scales + biases]
                    if best_state is None:
                        best_state = tuple(t.clone() for t in snap)
                    for j, t in enumerate(snap):
                        best_state[j][better] = t[better]

    fwd_plain = train_batch_fwd
    if use_compile:
        fwd = torch.compile(train_batch_fwd)
        print("    [compile] torch.compile enabled (first steps compile the graph)", flush=True)
    else:
        fwd = train_batch_fwd

    # jittered-target bank: JITTER_BANK precomputed minibatches (Zj, teacher Yb),
    # regenerated every JITTER_REGEN steps; saves the per-step teacher forward
    jbank: list = []

    def _rebuild_jbank():
        jbank.clear()
        with torch.no_grad():
            for _ in range(JITTER_BANK):
                idx_b = torch.randint(0, Nmax, (bs,), device="cuda")
                Zj = Z[:, idx_b] + jitter * z_std * torch.randn_like(Z[:, idx_b])
                g_o = soft_lim(torch.bmm(Zj, oW1.transpose(1, 2)) + t_b1)
                u_o = soft_lim(torch.bmm(Zj, oW3.transpose(1, 2)) + t_b3)
                Yb_b = torch.bmm(F.silu(g_o) * u_o, oW2.transpose(1, 2))
                jbank.append((idx_b, Zj, Yb_b))

    t0 = time.time()
    for st in range(steps):
        g_o = u_o = h_o = None
        if t_b1 is not None and t_b3 is not None:
            if not jbank or st % JITTER_REGEN == 0:
                _rebuild_jbank()
            idx, Zb, Yb = jbank[st % len(jbank)]
            Mb = torch.ones_like(M[:, idx])
        else:
            if Nmax > bs:
                idx = torch.randint(0, Nmax, (bs,), device="cuda")
                Zb, Mb = Z[:, idx], M[:, idx]
            else:
                idx = torch.arange(Nmax, device="cuda")
            Zb, Mb = Z, M
            Yb = Y[:, idx]
        yp, g_bin, u_bin, h_bin = fwd(Zb, weights, scales, biases, bounds, want_guh=AM_WEIGHT > 0)
        diff2 = (yp - Yb) ** 2
        yb2 = Yb**2
        num = (Mb * diff2).sum(dim=(1, 2)).float()
        den = (Mb * yb2).sum(dim=(1, 2)).float().clamp_min(1e-12)
        resid = num / den
        active = (~done_mask).float()
        if active.sum() == 0:
            break
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
        # Activation matching: match intermediate g/u/h against the original's,
        # decoupling W1/W3/W2 learning (richer, direct signal than output-only).
        if AM_WEIGHT > 0 and g_o is not None and u_o is not None and h_o is not None:
            cnt = (Mb.sum(dim=(1, 2)) * active).sum() + 1e-12
            loss_g = ((Mb * (g_bin - g_o) ** 2).sum(dim=(1, 2)).float() * active).sum() / (cnt * inter)
            loss_u = ((Mb * (u_bin - u_o) ** 2).sum(dim=(1, 2)).float() * active).sum() / (cnt * inter)
            loss_h = ((Mb * (h_bin - h_o) ** 2).sum(dim=(1, 2)).float() * active).sum() / (cnt * inter)
            loss = loss + AM_WEIGHT * (loss_g + loss_u + loss_h)
        for w in weights:
            w.grad = None
        loss.backward()
        # Inactive boosting subs are off-graph -> grad stays None; update skips them.
        grads: list[torch.Tensor | None] = [w.grad for w in weights]
        # Non-finite guard: binary weights are bounded (sign), so divergence
        # channels through the continuous scales/biases (raw-SGD momentum).
        # Roll back to the last honest best_state and reset momentum so the
        # explosion cannot poison the run (finite blowups end as inf -> caught).
        if not torch.isfinite(loss):
            with torch.no_grad():
                if best_state is not None:
                    for p, saved in zip(weights + scales + biases, best_state):
                        p.data.copy_(saved)
                for t in ews + es + ebs:
                    t.zero_()
            nonlocal_guard_ct[0] += 1
            if nonlocal_guard_ct[0] <= 3 or nonlocal_guard_ct[0] % 100 == 0:
                print(
                    f"    [guard] non-finite loss at step {st + 1} "
                    f"(#{nonlocal_guard_ct[0]}) - rollback to best_state, momentum reset",
                    flush=True,
                )
            continue
        if use_compile and ((st + 1) % check_every == 0 or st == 0):
            if not all(torch.isfinite(g).all() for g in grads if g is not None):
                raise RuntimeError(
                    f"non-finite grads at step {st + 1} under torch.compile "
                    "(known ROCm buffer-reuse bug) - rerun without --compile"
                )
        if done_mask.any():
            with torch.no_grad():
                for g in grads:
                    if g is not None:
                        g[done_mask] = 0
                for p in scales + biases:  # freeze done experts fully (not just weights)
                    if p.grad is not None:
                        p.grad[done_mask] = 0
        with torch.no_grad():
            lr_m = LR_BASE * 0.5 * (1 + math.cos(math.pi * st / steps))  # CPU scalar, no GPU sync
            lr_s = lr_m * SCALE_LR_FACTOR
            for j in range(NSUB):
                for wi, (w, ew, g) in enumerate(zip(weights[3 * j : 3 * j + 3], ews[3 * j : 3 * j + 3], grads[3 * j : 3 * j + 3])):
                    if g is None:
                        continue
                    ew.mul_(MU).add_(g, alpha=1 - MU)
                    w.data -= lr_m * zeropower(ew)
                for si, (p, ep) in enumerate(zip(scales[3 * j : 3 * j + 3], es[3 * j : 3 * j + 3])):
                    if si == 2:
                        continue  # s2: exact-LS only (S2_EVERY); gradient drift destabilizes (measured)
                    if p.grad is not None:
                        p.data -= lr_s * p.grad  # plain SGD: momentum on scales amplifies the
                        # post-refresh amplitude kick (measured: 1.5M% buffer spikes)
                for p, ep in zip(biases[2 * j : 2 * j + 2], ebs[2 * j : 2 * j + 2]):
                    if p.grad is not None:
                        p.data -= lr_s * p.grad  # plain SGD, same reason
        # --- periodic exact LS refresh (ALS): scales/biases are convex subproblems ---
        do_s2 = S2_EVERY and ((st + 1) % S2_EVERY == 0 or st == steps - 1)
        do_signs = SIGN_EVERY and ((st + 1) % SIGN_EVERY == 0 or st == steps - 1)
        if (do_s2 or do_signs) and not done_mask.any():
            with torch.no_grad():

                def _resid_now(sc):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        ypr, _, _, _ = train_batch_fwd(Z, weights, sc, biases, bounds, want_guh=False)
                    d2 = (M * (ypr.float() - Y.float()) ** 2).sum(dim=(1, 2))
                    y2 = (M * Y.float() ** 2).sum(dim=(1, 2)).clamp_min(1e-12)
                    return d2 / y2

                r0 = _resid_now(scales)
                Mfl = M.float()
                if do_s2 or do_signs:
                    hs = []
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        for s_ in range(NSUB):
                            s1v, s3v = scales[3 * s_], scales[3 * s_ + 1]
                            b1v = biases[2 * s_] if biases else None
                            b3v = biases[2 * s_ + 1] if biases else None
                            Zs = Z[..., bounds[s_] : bounds[s_ + 1]].contiguous()
                            Zu = Z[..., bounds[u_slice(s_)] : bounds[u_slice(s_) + 1]].contiguous()
                            gg = torch.bmm(Zs, (torch.sign(weights[3 * s_]) * s1v).to(torch.bfloat16).transpose(1, 2))
                            if b1v is not None:
                                gg = gg + b1v.to(torch.bfloat16)
                            gg = soft_lim(gg)
                            uu = torch.bmm(Zu, (torch.sign(weights[3 * s_ + 1]) * s3v).to(torch.bfloat16).transpose(1, 2))
                            if b3v is not None:
                                uu = uu + b3v.to(torch.bfloat16)
                            uu = soft_lim(uu)
                            hs.append(F.silu(gg) * uu)

                    def _solve_s2():
                        """Per-channel joint (4x4) s2 for the CURRENT W2 signs. Convex:
                        cannot be worse than the incumbent scales."""
                        A2 = []
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            for s_ in range(NSUB):
                                A2.append(
                                    torch.bmm(hs[s_], qint4_fixed(weights[3 * s_ + 2]).to(torch.bfloat16).transpose(1, 2)).float()
                                )
                        Mf2 = M.float()[:, :, 0]
                        Gm = A2[0].new_zeros(G, D, NSUB, NSUB)
                        for i in range(NSUB):
                            for j in range(i, NSUB):
                                v = (Mf2[:, :, None] * A2[i] * A2[j]).sum(dim=1)
                                Gm[:, :, i, j] = v
                                Gm[:, :, j, i] = v
                        rhs2 = A2[0].new_zeros(G, D, NSUB)
                        for i in range(NSUB):
                            rhs2[:, :, i] = (Mf2[:, :, None] * A2[i] * Y.float()).sum(dim=1)
                        reg2 = Gm.diagonal(dim1=2, dim2=3).mean(dim=(1, 2)).clamp_min(1e-6) * 1e-6
                        Gm = Gm + reg2[:, None, None, None] * torch.eye(NSUB, device="cuda")[None, None]
                        x2 = torch.linalg.solve(Gm.view(G * D, NSUB, NSUB), rhs2.view(G * D, NSUB)).view(G, D, NSUB)
                        for s_ in range(NSUB):
                            v = x2[:, :, s_]
                            scales[3 * s_ + 2].data = (v.sign() * v.abs().clamp_min(1e-5)).unsqueeze(2)

                    # (b0) s2-only: optimal at current signs - apply unconditionally
                    _solve_s2()
                    r1 = _resid_now(scales)  # <= r0 by construction
                    if do_signs:
                        snap1 = [t.detach().clone() for t in weights + scales + biases]
                        # (b1) sign refresh from dual-ridge Theta (guarded)
                        H = torch.cat(hs, dim=2).float() * Mfl  # [G,n,S*inter]
                        # DUAL ridge solve: F = S*inter = 8192 features > n samples -> primal
                        # normal equations are rank-deficient (measured: garbage signs, resid 99.7%).
                        # Minimum-norm solution via the sample-space system [n,n].
                        Ym = Y.float() * Mfl  # [G,n,D]
                        Kk = torch.bmm(H, H.transpose(1, 2))  # [G,n,n]
                        regk = Kk.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-6) * 1e-2  # F==n square in int4x: needs a real ridge
                        Kk = Kk + regk[:, None, None] * torch.eye(Kk.shape[1], device="cuda")[None]
                        alpha = torch.linalg.solve(Kk, Ym)  # [G,n,D]
                        Theta = torch.bmm(H.transpose(1, 2), alpha)  # [G,F,D]
                        for s_ in range(NSUB):
                            blk = Theta[:, s_ * inter : (s_ + 1) * inter, :]  # [G,inter,D]
                            sg = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 7.0  # [G,1,D]
                            weights[3 * s_ + 2].data = ((blk / sg).round().clamp(-7.0, 7.0)).transpose(1, 2).contiguous()
                        _solve_s2()
                        r2_raw = _resid_now(scales)
                        if os.environ.get("NO_GUARD") is None and (r2_raw >= r1).any():
                            for t, sv in zip(weights + scales + biases, snap1):
                                t.data.copy_(sv)
                        else:
                            r1 = r2_raw
                if (r1 < r0).any():
                    for t in es + ebs:
                        t.zero_()  # scales/biases optimal: kill stale momentum
                    # sign refreshes are rare: always log; s2-only steps log big gains only
                    if do_signs or (r0 - r1).mean() > r0.mean() * 0.005:
                        tag = "als+signs" if do_signs else "als-s2"
                        print(f"    [{tag}] step {st + 1}: {r0.mean() * 100:.3f}% -> {r1.mean() * 100:.3f}%", flush=True)
        if stop_threshold is None:
            cur_best = best_stop.min().item()
            if not (cur_best < float("inf")):
                stall = 0
            elif best is None or cur_best < best - max(1e-5, best * stall_tol):
                best = cur_best
                stall = 0
            else:
                stall += 1
            if stall >= patience:
                assert best is not None
                print(f"    stalled at step {st + 1}/{steps} (best resid {best * 100:.4f}%)", flush=True)
                break
        honest = (n_real_t.sum() == 0) or (st == steps - 1) or ((st + 1) % check_every == 0)
        stop_resid = resid
        if stop_threshold is not None and n_real_t.sum() > 0 and honest:
            nrm = int(n_real_t.max().item())
            Zc, Yc = Z[:, :nrm], Y[:, :nrm]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ypr, _, _, _ = train_batch_fwd(Zc, weights, scales, biases, bounds, want_guh=False)
            real_mask = torch.arange(nrm, device="cuda")[None, :] < n_real_t[:, None]  # [G, nrm]
            num_r = (real_mask[:, :, None] * (ypr - Yc) ** 2).sum(dim=(1, 2)).float()
            den_r = (real_mask[:, :, None] * Yc**2).sum(dim=(1, 2)).float().clamp_min(1e-12)
            stop_resid = num_r / den_r
        if honest and stop_threshold is None and ((st + 1) % check_every == 0 or st == steps - 1):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ypf, _, _, _ = fwd_plain(Z, weights, scales, biases, bounds, want_guh=False)
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
            with torch.no_grad():
                if ema_h is None:
                    ema_h = stop_resid.clone()
                    ref_h = stop_resid.clone()
                else:
                    assert ref_h is not None
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
                import psutil

                _p = psutil.Process(os.getpid())
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

    if best_state is not None:
        for p, saved in zip(weights + scales + biases, best_state):
            p.data.copy_(saved)

    # Post-hoc LS refit of the W2 OUTPUT scales (signs frozen): closed-form
    # per-channel least squares on the full training buffer. Applied per
    # expert only when it lowers the buffer residual (Goodhart guard).
    with torch.no_grad():

        def _buf_resid(sc):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yp, _, _, _ = train_batch_fwd(Z, weights, sc, biases, bounds, want_guh=False)
            d2 = (M * (yp.float() - Y.float()) ** 2).sum(dim=(1, 2))
            y2 = (M * Y.float() ** 2).sum(dim=(1, 2)).clamp_min(1e-12)
            return d2 / y2

        r_before = _buf_resid(scales)
        A = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(NSUB):
                W1d, W3d, W2d = weights[3 * s], weights[3 * s + 1], weights[3 * s + 2]
                s1, s3 = scales[3 * s], scales[3 * s + 1]
                b1 = biases[2 * s] if biases else None
                b3 = biases[2 * s + 1] if biases else None
                Zs = Z[..., bounds[s] : bounds[s + 1]].contiguous()
                Zu = Z[..., bounds[u_slice(s)] : bounds[u_slice(s) + 1]].contiguous()
                g = torch.bmm(Zs, (torch.sign(W1d) * s1).to(torch.bfloat16).transpose(1, 2))
                if b1 is not None:
                    g = g + b1.to(torch.bfloat16)
                g = soft_lim(g)
                u = torch.bmm(Zu, (torch.sign(W3d) * s3).to(torch.bfloat16).transpose(1, 2))
                if b3 is not None:
                    u = u + b3.to(torch.bfloat16)
                u = soft_lim(u)
                h = F.silu(g) * u
                # scale-free output basis: sign(W2) columns, s2 folded out
                A.append(torch.bmm(h, qint4_fixed(W2d).to(torch.bfloat16).transpose(1, 2)).float())  # [G,N,D]
        Mf = M.float()[:, :, 0]  # [G,N]
        # Per-channel normal equations: Gram must aggregate over n ONLY (not over d),
        # matching rhs — a scalar Gram was silently off by ~D in scale and the old
        # "apply if better" guard was hiding it (LS never actually fired).
        Gm = A[0].new_zeros(G, D, NSUB, NSUB)
        for i in range(NSUB):
            for j in range(i, NSUB):
                v = (Mf[:, :, None] * A[i] * A[j]).sum(dim=1)  # [G, D]
                Gm[:, :, i, j] = v
                Gm[:, :, j, i] = v
        rhs = A[0].new_zeros(G, D, NSUB)
        for i in range(NSUB):
            rhs[:, :, i] = (Mf[:, :, None] * A[i] * Y.float()).sum(dim=1)  # [G, D]
        reg = Gm.diagonal(dim1=2, dim2=3).mean(dim=(1, 2)).clamp_min(1e-6) * 1e-6
        Gm = Gm + reg[:, None, None, None] * torch.eye(NSUB, device=Gm.device)[None, None]
        x = torch.linalg.solve(Gm.view(G * D, NSUB, NSUB), rhs.view(G * D, NSUB)).view(G, D, NSUB)
        new_s2 = [(v.sign() * v.abs().clamp_min(1e-5)).unsqueeze(2) for v in (x[:, :, s] for s in range(NSUB))]
        cand_scales = list(scales)
        for s in range(NSUB):
            cand_scales[3 * s + 2] = new_s2[s]
        r_after = _buf_resid(cand_scales)
        better = r_after < r_before
        if better.any():
            for s in range(NSUB):
                scales[3 * s + 2].data = torch.where(better[:, None, None], new_s2[s], scales[3 * s + 2].data)
            print(
                f"    [ls] W2 scale refit: buffer resid {r_before[better].mean() * 100:.3f}% "
                f"-> {r_after[better].mean() * 100:.3f}% ({int(better.sum())}/{G} experts)",
                flush=True,
            )

    fixed = []
    for j, (w, sc) in enumerate(zip(weights, scales)):
        if j % 3 == 2:  # W2: int4 grid
            fixed.append((qint4_fixed(w.detach()), sc.detach().squeeze(2)))
        else:  # W13: binary
            fixed.append(binarize_fixed(w.detach(), sc.detach()))
    bf = [p.detach().squeeze(1) for p in biases]  # each [G, inter]
    out = []
    for i in range(G):
        items = []
        for q, sc in fixed:  # 12 pairs: w1,w3,w2 per sub (q, scale) interleaved
            items.append(q[i])
            items.append(sc[i])
        for b in bf:  # 8 biases: b1, b3 per sub
            items.append(b[i])
        items.append(bounds)  # pyramidal slice boundaries (shared ints)
        out.append(tuple(items))
    return out


def run_refit(
    start_layer,
    end_layer,
    steps,
    all_flag,
    done_log,
    refit_threshold=2e-4,
    use_compile=False,
    seed=0,
    real_weight=800.0,
    jitter=0.2,
):
    """Router-based binary refit: covered experts learn on their REAL routed
    activations; uncovered (noisy) experts learn on a proxy manifold."""
    for L in range(start_layer, end_layer):
        p_path = os.path.join(REDUCED, f"layer_{L}", "P.pt")
        if not os.path.exists(p_path):
            continue
        P = torch.load(p_path, map_location="cuda").float()
        mu = torch.load(os.path.join(REDUCED, f"layer_{L}", "mu.pt"), map_location="cuda").float()
        acts = torch.load(os.path.join(POD, f"acts_layer{L}.pt"), map_location="cpu", weights_only=False)
        print(f"layer {L}: indexed {len(acts)} experts (activations stay on CPU, moved per expert)", flush=True)

        todo = []
        for k, (x_k, y_k) in acts.items():
            key = f"{L}_{k}"
            ep = os.path.join(REDUCED, f"layer_{L}", f"expert_{k}.pt")
            e = None
            if os.path.exists(ep):
                e = torch.load(ep, map_location="cpu", weights_only=False)
                rc = e.get("residual", float("inf"))
                # Resume only when the file is already a BINARY refit and good
                # enough. Old ternary files (no MODE_MARKER) are always refit.
                if (
                    not all_flag
                    and e.get("mode") == MODE_MARKER
                    and isinstance(rc, (int, float))
                    and rc < refit_threshold
                ):
                    with open(done_log, "a") as f:
                        f.write(f"{key}\n")
                    continue
            todo.append((k, x_k, y_k, e))

        covered = set(acts.keys())
        for k in range(256):
            sk = str(k)
            if sk not in covered:
                todo.append((sk, None, None, None))

        del acts
        print(
            f"layer {L}: collected {len(todo)} experts to refit "
            f"({len(covered)} real + {len(todo) - len(covered)} synthetic)",
            flush=True,
        )

        if not todo:
            print(f"layer {L}: nothing to refit", flush=True)
            continue

        todo.sort(key=lambda t: 0 if t[1] is None else t[1].shape[0])

        t0 = time.time()
        n_fixed = 0

        SUB_IDS = "abcdefgh"[:NSUB]
        W_NAMES = [f"w{m}{c}" for c in SUB_IDS for m in ("1", "3", "2")]
        B_NAMES = [f"bias{m}{c}" for c in SUB_IDS for m in ("1", "3")]

        def save_expert(k, resid, res, inter, n_real=None):
            ep = os.path.join(REDUCED, f"layer_{L}", f"expert_{k}.pt")
            e = {}
            for j, nm in enumerate(W_NAMES):  # res: (q, scale) pairs, interleaved
                q = res[2 * j]
                e[nm] = (pack_int4(q) if nm.startswith("w2") else pack_binary(q)).cpu()
                e[f"{nm}_scale"] = res[2 * j + 1].cpu()
            for j, nm in enumerate(B_NAMES):  # then 8 biases (b1, b3 per sub)
                e[nm] = res[6 * NSUB + j].cpu()
            e["bounds"] = list(res[-1])  # pyramidal input-slice boundaries
            e["inter"] = inter
            e["residual"] = resid
            e["mode"] = MODE_MARKER
            if n_real is not None:
                e["n_real"] = int(n_real)
            torch.save(e, ep)
            with open(done_log, "a") as f:
                f.write(f"{L}_{k}\n")

        covered_todo = [t for t in todo if t[1] is not None]
        missing_todo = [t for t in todo if t[1] is None]

        def _prep_covered(t):
            (k, x_k, y_k, e_prev) = t
            z = (x_k.float().cuda() - mu) @ P
            yf = y_k.float().cuda()
            n_k = z.shape[0]
            if n_k >= 128:
                tier_thr, tier_steps, tier = refit_threshold, steps, "A"
                rw_eff = 1.0
                z_synth = z[:0]
            else:
                # Train ALL experts — no tier-C skip. Few/no real samples get
                # synthetic activations generated from the ORIGINAL FP4 expert
                # (built later in the tier-B loop; no signal is generated here).
                tier_thr, tier_steps, tier = refit_threshold, steps, "B"
                rw_eff = real_weight
            experts = load_selected_experts(L, [int(k)])
            w1, w2, w3 = experts[int(k)]
            # Fold POD rotation P + mean mu into W1/W3 so the binary core operates in z-space:
            # target g = z @ (W@P).T + mu@W.T. W2 (intermediate->output) is unchanged.
            w1_rot = w1 @ P  # [inter, K] -> [inter, K]
            w3_rot = w3 @ P
            bias1 = mu.reshape(-1) @ w1.T  # [inter] (mu is [1,K] -> flatten)
            bias3 = mu.reshape(-1) @ w3.T
            # PTQ FIRST, gradient on top: replace the FP4 W2 with the exact
            # ridge-solved readout on this expert's own rows (~1s), then let the
            # gradient pass refine signs/scales/W2 from a fitted starting point.
            _res0, w2c = ptq_closed_form(w1_rot, w3_rot, w2, z, yf, bias1, bias3)
            fp4_init = (w1_rot[None].contiguous(), w3_rot[None].contiguous(), w2c[None].contiguous())
            del experts
            z_all_rows = z.shape[0]
            todo_data[k] = (z, yf)
            init = tuple(t.to(torch.bfloat16).cpu() for t in fp4_init)
            bias1_cpu = bias1.cpu()
            bias3_cpu = bias3.cpu()
            del fp4_init, w1_rot, w3_rot
            return dict(
                k=k,
                n_k=n_k,
                tier=tier,
                tier_thr=tier_thr,
                tier_steps=tier_steps,
                rw_eff=rw_eff,
                inter=INTER,
                z_rows=z.shape[0],
                z_all_rows=z_all_rows,
                init=init,
                bias1=bias1_cpu,
                bias3=bias3_cpu,
            )

        todo_data = {}
        prepped_A = []
        prepped_B = []
        for t in covered_todo:
            p = _prep_covered(t)
            if p is None:
                continue
            (prepped_B if p["tier"] == "B" else prepped_A).append(p)

        # Layer-wide activation pool: the residual stream that feeds ALL experts
        # of the layer. For experts with few/no routed rows this is a far better
        # manifold than bootstrapping around their own 2-3 points (measured:
        # n=2 tier-B converged to 38-66% resid - wasted 67s each).
        n_pool_rows = 8192
        z_parts = []
        n_have = 0
        rng = torch.Generator().manual_seed(1234 + L)
        cand = list(todo_data.keys())
        cand.sort(key=lambda kk: todo_data[kk][0].shape[0], reverse=True)
        for kk in cand:
            z_kk = todo_data[kk][0]
            if n_have + z_kk.shape[0] > n_pool_rows:
                take = n_pool_rows - n_have
                if take > 0:
                    z_parts.append(z_kk[:take].cpu())
                    n_have += take
                break
            z_parts.append(z_kk.cpu())
            n_have += z_kk.shape[0]
        z_pool = torch.cat(z_parts) if z_parts else torch.zeros(1, INTER)
        del z_parts
        print(f"layer {L}: activation pool for tier-B/dead: {z_pool.shape[0]} rows", flush=True)

        PTQ_FALLBACK = 0.10  # gradient result worse than this -> closed-form PTQ

        def _run_expert(p):
            """Train ONE expert (no batching): G=1 tensors in, single result out."""
            nonlocal n_fixed
            z_all_gpu, y_all_gpu = todo_data[p["k"]]
            n_rows = p["z_all_rows"]
            init_gpu = tuple(t.cuda().float() for t in p["init"])
            bias1_gpu = p["bias1"].cuda().float().unsqueeze(0)  # [1, inter]
            bias3_gpu = p["bias3"].cuda().float().unsqueeze(0)
            res = train_batch(
                [(z_all_gpu[:n_rows], y_all_gpu[:n_rows])],
                INTER,
                p["tier_steps"],
                stop_threshold=p["tier_thr"],
                n_real=[p["n_k"]],
                use_compile=use_compile,
                init=init_gpu,
                real_weight=p["rw_eff"],
                bias1=bias1_gpu,
                bias3=bias3_gpu,
            )[0]
            z_h, yf_h = todo_data[p["k"]]
            resid = resid_weights_full(z_h, yf_h, res)
            if resid > PTQ_FALLBACK:
                # stalled gradient run (thin coverage: STE sign oscillation) ->
                # AngelSlim-style closed-form PTQ in ~1s; keep whichever is better
                w1f, w3f, w2f = init_gpu[0][0], init_gpu[1][0], init_gpu[2][0]
                try:
                    res_ptq, _w2c = ptq_closed_form(
                        w1f, w3f, w2f, z_all_gpu[:n_rows], y_all_gpu[:n_rows],
                        p["bias1"].cuda().float(), p["bias3"].cuda().float(),
                    )
                    resid_ptq = resid_weights_full(z_h, yf_h, res_ptq)
                    if resid_ptq < resid:
                        res, resid = res_ptq, resid_ptq
                        print(f"    [ptq-fallback] {p['k']}: {resid*100:.3f}%", flush=True)
                except Exception as ex:  # noqa: BLE001 - fallback must never kill the run
                    print(f"    [ptq-fallback] {p['k']} failed: {ex}", flush=True)
            save_expert(p["k"], resid, res, p["inter"], n_real=p["n_k"])
            n_fixed += 1
            print(
                f"  layer {L} expert {p['k']} [tier {p['tier']}, n={p['n_k']}]: "
                f"resid={resid * 100:.4f}%  ({n_fixed} total, {time.time() - t0:.0f}s)",
                flush=True,
            )
            todo_data.pop(p["k"], None)  # stream: free GPU activations
            torch.cuda.empty_cache()

        for p in prepped_A:
            _run_expert(p)
        for p in prepped_B:
            z, yf = todo_data[p["k"]]
            if z.shape[0] < 128:  # thin real coverage -> bootstrap from the LAYER pool
                z_synth = universal_signal(
                    z, seed=seed + int(p["k"]), jitter=jitter,
                    z_proxy=z_pool.cuda() if z.shape[0] < 8 else None,
                )
            else:
                z_synth = universal_signal(z, seed=seed + int(p["k"]), jitter=jitter)
            x_synth = (mu + z_synth @ P.T).float()
            experts = load_selected_experts(L, [int(p["k"])])
            w1, w2, w3 = experts[int(p["k"])]
            y_synth = ffn_exact(x_synth, w1, w2, w3)
            del experts, x_synth
            todo_data[p["k"]] = (torch.cat([z, z_synth]), torch.cat([yf, y_synth]))
            p["z_all_rows"] = todo_data[p["k"]][0].shape[0]
            # PTQ warm-init on the FULL (real+synthetic) rows: the continuous W2
            # starts exact-fitted instead of the raw FP4 readout
            try:
                b1g = p["bias1"].cuda().float()
                b3g = p["bias3"].cuda().float()
                z_all, y_all = todo_data[p["k"]]
                _r0, w2c = ptq_closed_form(w1 @ P, w3 @ P, w2, z_all, y_all, b1g, b3g)
                init_l = list(p["init"])
                init_l[2] = w2c[None].to(torch.bfloat16).cpu()
                p["init"] = tuple(init_l)
                del w2c
            except Exception as ex:  # noqa: BLE001 - init stays FP4 on failure
                print(f"    [ptq-init] {p['k']} failed: {ex}", flush=True)
            _run_expert(p)

        for k, z, yf, _e_prev in missing_todo:
            # 0 routed rows in the collection, but the expert IS part of the model:
            # distill it on the layer activation pool against its original weights.
            g = torch.Generator().manual_seed(777 + int(k))
            idx = torch.randint(0, z_pool.shape[0], (4096,), generator=g)
            z_rows = z_pool[idx].cuda().float()
            x_rows = (mu + z_rows @ P.T).float()
            experts = load_selected_experts(L, [int(k)])
            w1, w2, w3 = experts[int(k)]
            y_rows = ffn_exact(x_rows, w1, w2, w3)
            bias1 = mu.reshape(-1) @ w1.T
            bias3 = mu.reshape(-1) @ w3.T
            _r0, w2c0 = ptq_closed_form(w1 @ P, w3 @ P, w2, z_rows, y_rows, bias1.float(), bias3.float())
            init = (w1 @ P, w3 @ P, w2c0)
            del experts, x_rows
            res = train_batch(
                [(z_rows, y_rows)],
                INTER,
                steps,
                stop_threshold=refit_threshold,
                n_real=[0],
                use_compile=use_compile,
                init=tuple(t[None].contiguous().float() for t in init),
                real_weight=1.0,
                bias1=bias1.float()[None],
                bias3=bias3.float()[None],
            )[0]
            resid = resid_weights_full(z_rows, y_rows, res)
            if resid > PTQ_FALLBACK:
                try:
                    res_ptq, _w2c = ptq_closed_form(
                        (w1 @ P), (w3 @ P), w2, z_rows, y_rows,
                        bias1.float(), bias3.float(),
                    )
                    resid_ptq = resid_weights_full(z_rows, y_rows, res_ptq)
                    if resid_ptq < resid:
                        res, resid = res_ptq, resid_ptq
                        print(f"    [ptq-fallback] dead {k}: {resid*100:.3f}%", flush=True)
                except Exception as ex:  # noqa: BLE001
                    print(f"    [ptq-fallback] dead {k} failed: {ex}", flush=True)
            save_expert(k, resid, res, INTER, n_real=0)
            n_fixed += 1
            print(
                f"  layer {L} expert {k} [dead->distill, n=0]: resid={resid * 100:.4f}%  "
                f"({n_fixed} total, {time.time() - t0:.0f}s)",
                flush=True,
            )
            torch.cuda.empty_cache()
        print(f"layer {L}: refit {n_fixed} experts in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=43)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--all", action="store_true", help="refit all experts, not only <= threshold")
    ap.add_argument("--done-log", default="refit_bin_done.txt")
    ap.add_argument(
        "--refit-threshold",
        type=float,
        default=1e-3,
        help="accept an expert when its residual <= this (fraction, 0.1%% default)",
    )
    ap.add_argument(
        "--n-procs", type=int, default=4, help="split layers across N parallel processes (0=auto=min(4,n_layers))"
    )
    ap.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile the training forward step (default: off)",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="unifold signal seed offset (vary per refinement pass for data diversity)"
    )
    ap.add_argument(
        "--real-weight",
        type=float,
        default=800.0,
        help="loss weight multiplier for REAL rows (vs 1.0 synthetic)",
    )
    ap.add_argument(
        "--jitter", type=float, default=0.2, help="unifold bootstrap jitter magnitude (fraction of per-dim sigma)"
    )
    args = ap.parse_args()

    n_procs = max(1, args.n_procs if args.n_procs > 0 else min(4, args.end_layer - args.start_layer))

    def _run():
        run_refit(
            args.start_layer,
            args.end_layer,
            args.steps,
            args.all,
            args.done_log,
            args.refit_threshold,
            use_compile=args.compile,
            seed=args.seed,
            real_weight=args.real_weight,
            jitter=args.jitter,
        )

    if n_procs == 1:
        MAX_RETRIES = 10
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                _run()
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
        cmd.extend(
            [
                "--seed",
                str(args.seed),
                "--real-weight",
                str(args.real_weight),
                "--jitter",
                str(args.jitter),
            ]
        )
        cache_root = os.path.join(os.environ.get("TEMP", "."), f"torchinductor_p{i}")
        env = {
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": cache_root,
            "TRITON_CACHE_DIR": os.path.join(cache_root, "triton"),
        }
        with open(f"refit_bin_p{i}.log", "a") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        procs.append((p, cmd, env))
        print(f"proc {i}: layers {s}-{e} -> refit_bin_p{i}.log (pid {p.pid})", flush=True)
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
            with open(f"refit_bin_p{idx}.log", "a") as f:
                f.write(f"\n=== supervisor restart {retries[idx]} ===\n")
                p2 = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            alive.append((p2, cmd, env))
        procs = alive
    print("all procs done", flush=True)


if __name__ == "__main__":
    main()
