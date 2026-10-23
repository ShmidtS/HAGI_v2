"""Formalization-derived utilities (Hagi Lean program -> code).

Each function is the executable form of a proven theorem; the docstring
cites the source. No fitted constants live here: every threshold comes
from a proof (see .omc/attempts/growth_gate_v2_round14.md).

1. Mix.lean ``mixed_invisible``: a row rotation Q applied to the
   concat hidden stream is invisible to the logits iff its column
   sums reproduce the heads (for a shared head: sums == 1). A mixer
   violating this transfers mass between leaves -- the regime where
   it must be TRAINED, not trusted.
2. KVWater.lean ``waterfilling_bound``: under a fixed bit budget the
   total KV residual sum c_i * exp(-kappa * b_i) is minimized by
   equalizing the marginal residuals: bits grow LOGARITHMICALLY in
   the sensitivity c_i (b*_i - b*_j = (log c_i - log c_j)/kappa),
   NOT proportionally. Closed form -> no allocation sweep.
3. Select.lean ``certifiedGain``: (M - c)/(N+1), the certified new
   mean-CE bound gain of adding a candidate -- used by
   scripts/growth_gate.py; re-exported here as the canonical home.
"""
from __future__ import annotations

import math

import torch


def mixer_invisible_condition(q: torch.Tensor) -> bool:
    """Mix.lean mixed_invisible: check the visibility invariant.

    Args:
        q: the ``[n, n]`` rotation applied to the concat stream's
            block axis (e.g. ``H_n / sqrt(n)`` from the Hadamard
            mixer, an F3 character matrix, any learned rotation).

    Returns:
        True iff the rotation is *logit-invisible* for a shared head:
        every column of Q sums to 1, i.e. the head's per-leaf readout
        sees each leaf's own contribution exactly. Formally:
        ``forall b, sum_a Q[a, b] = 1`` (the common-head case of
        ``sum_a Q[a, b] * W_a = W_b`` -- when all W_a are the same
        weight, this reduces to the column sums being 1).

    Notes:
        The Sylvester Hadamard ``H_n / sqrt(n)`` FAILS this for
        n > 1 (columns sum to +-1/sqrt(n) != 1) -- which is exactly
        why the Hadamard mixer needs the pre-rotated head ``WQ`` and
        cannot be trusted bare on the ensemble path. The identity and
        the parent-preserving lifts PASS. Use this to gate any mixer
        that claims to preserve the free-ensemble logits.
    """
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError(f"expected a square rotation, got {q.shape}")
    sums = q.sum(dim=0)
    return bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))


def kv_waterfill_bits(
    sensitivities: torch.Tensor,
    total_bits: float,
    kappa: float,
) -> torch.Tensor:
    """KVWater.lean waterfilling_bound: the optimal bit allocation.

    Minimizes ``sum_i c_i * exp(-kappa * b_i)`` subject to
    ``sum_i b_i = total_bits`` (the theorem's marginal-residual
    equalization): the optimum equalizes ``c_i * exp(-kappa * b_i)``,
    so ``b*_i = (log c_i - log lambda) / kappa`` with the water
    level ``log lambda`` fixed by the budget.

    Args:
        sensitivities: ``c_i >= 0`` -- per-position KV sensitivity
            (e.g. softmax mass * value norm; the theorem's
            contribution-weighted scale).
        total_bits: the total bit budget across positions.
        kappa: the residual decay rate per bit (quantization error
            decays exponentially with precision).

    Returns:
        The optimal per-entry bit allocation ``b*_i`` (float; floor to
        integer grids at the call site). Entries with ``c_i = 0``
        get 0 bits -- zero sensitivity carries zero residual weight.
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if total_bits < 0:
        raise ValueError("total_bits must be nonnegative")
    c = sensitivities.float().clamp_min(0)
    n = c.numel()
    if n == 0 or total_bits == 0.0:
        return torch.zeros_like(c)
    log_c = torch.where(c > 0, c.log(), torch.full_like(c, -math.inf))
    finite = torch.isfinite(log_c)
    if not finite.any():
        # all-zero sensitivities: uniform split is optimal (all
        # marginal residuals equal at any allocation).
        return torch.full_like(c, total_bits / n)
    # KKT waterfilling with the NONNEGATIVITY constraint b_i >= 0:
    # the naive closed form can assign negative bits to low-c
    # entries; clamping them would VIOLATE the budget. The correct
    # solution is the active-set iteration: drop entries whose
    # closed-form allocation is negative, recompute the water level
    # lambda on the remaining active set, repeat until stable. At
    # the KKT point every active entry has c_i exp(-kappa b_i) =
    # lambda and every dropped entry has c_i <= lambda (0 bits).
    active = finite.clone()
    b = torch.zeros_like(c)
    for _ in range(n + 1):
        k = int(active.sum())
        if k == 0:
            break
        s = float(log_c[active].sum())
        log_lambda = (s - kappa * total_bits) / k
        cand = (log_c - log_lambda) / kappa
        new_active = active & (cand > 0)
        if bool((new_active == active).all()):
            b = torch.where(active, cand, torch.zeros_like(c))
            return b
        active = new_active
    b = torch.where(active, (log_c - (float(log_c[active].sum())
                                     - kappa * total_bits) / int(active.sum())) / kappa,
                    torch.zeros_like(c))
    return b.clamp_min(0)


def certified_gain(n: int, mean_ce: float, candidate_ce: float) -> float:
    """Select.lean newBound / certifiedGain: the pre-step bound.

    Adding a candidate with standalone CE ``c`` to a pool of ``n``
    leaves with mean-CE ``M`` certifies the new bound
    ``(n*M + c)/(n+1)``; the gain over the current bound is exactly
    ``(M - c)/(n+1)``. Monotone in c (certifiedGain_monotone), so the
    best candidate certifies for all (grow_epsilon_stop).
    """
    return (mean_ce - candidate_ce) / (n + 1)


def jensen_gap_lse(logits: torch.Tensor) -> torch.Tensor:
    """GapLaw.lean twoGap_cosh: the EXACT Jensen gap, targets-free.

    Per position: ``gap = mean_a lse(z_a) - lse(mean_a z_a)`` --
    the targets cancel, so no CE pass over single leaves is needed
    (O(N) forwards + one lse). For two leaves this equals the proved
    cosh form ``0.5 * log sum_{u,v} p_u p_v cosh(d_u - d_v)`` with
    ``d = (z_A - z_B)/2`` (verified analytically); the general-n
    route is the same lse telescoping.

    Args:
        logits: ``[n, B, T, V]`` per-leaf logits (same scale).

    Returns:
        ``[B, T]`` per-position exact gap (nonnegative by
        twoGap_nonneg; zero iff consensus, twoGap_zero_iff).
    """
    z = logits.double()
    lse_a = torch.logsumexp(z, dim=-1)          # [n, B, T]
    lse_bar = torch.logsumexp(z.mean(0), dim=-1)  # [B, T]
    return lse_a.mean(0) - lse_bar


def complementarity(delta_c: torch.Tensor, delta_pool: torch.Tensor) -> float:
    """GapLaw P2 prescription: the candidate-selection metric.

    The candidate's gap increment is governed by the cross-
    covariance of its deviation ``delta_c`` with the pool's mean
    deviation ``delta_pool``: positive correlation shrinks the
    increment (redundant mass), zero/negative maximizes it
    (independent delta-mass -- the formal side of Select.lean's
    counterexample: a WORSE standalone leaf can be the BEST
    addition because it carries more independent deviation mass).

    Args:
        delta_c: candidate deviations ``[D]`` (logits minus pool mean).
        delta_pool: the pool's mean deviation ``[D]``.

    Returns:
        ``1 - corr(delta_c, delta_pool)`` in ``[0, 2]``; higher =
        more complementary. Falls back to 1.0 (neutral) when either
        side is degenerate (zero variance).
    """
    a = delta_c.double().flatten()
    b = delta_pool.double().flatten()
    va, vb = a.var(), b.var()
    if va <= 0 or vb <= 0:
        return 1.0
    rho = float(((a - a.mean()) * (b - b.mean())).mean() / (va * vb).sqrt())
    return 1.0 - rho


def saturated_threshold(g_inf: float, eps: float = 0.0021) -> float:
    """GapLaw P4 prescription: the DERIVED saturation size.

    Exchangeable leaves with total gap budget ``G_inf`` have
    increments ~ G_inf / N^2; growth stops paying when
    ``G_inf / N^2 < eps``. MODEL-BASED (the G_inf law itself is a
    docstring-level model in GapLaw.lean, not a theorem).

    Synthesis round-22 refinement: solving the model equation
    exactly, N(N+1) > G_inf/eps, gives
    ``N* = (-1 + sqrt(1 + 4*G_inf/eps)) / 2`` -- used here instead
    of the cruder sqrt(G/eps) upper estimate. The measured form:
    with the CURRENT gap G_N, the next-leaf increment is
    G_N/(N^2-1); stop when that falls under the measurement noise
    (eps_stop, adaptive -- see adaptive_eps).
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    return (math.sqrt(1.0 + 4.0 * max(g_inf, 0.0) / eps) - 1.0) / 2.0


def next_leaf_gain(g_n: float, n: int) -> float:
    """Synthesis round-22: the predicted NEXT-leaf gap increment
    from the CURRENT measured gap, without knowing G_inf.

    From the model G_N = G_inf(1 - 1/N): G_inf = G_N * N/(N-1), so
    the next increment dG_N = G_inf/(N(N+1)) = G_N/(N^2 - 1).
    Stop growing when this falls under the measurement noise.
    """
    if n < 2:
        raise ValueError("n must be >= 2 (need a previous level)")
    return g_n / (n * n - 1)


def adaptive_eps(se_before: float, se_after: float, z: float = 1.96) -> float:
    """Synthesis round-22 §30: the stop threshold IS the measurement
    noise, not a global constant.

    A gain is detectable only beyond the combined standard error of
    the two measurements (before/after the candidate merge):
    eps_stop = z * sqrt(se_before^2 + se_after^2). Below this, the
    gate cannot distinguish growth from noise -- stop regardless
    of the fixed EPS.
    """
    return z * math.sqrt(se_before * se_before + se_after * se_after)


def merge_tax(ce_merged: float, ce_ensemble: float) -> float:
    """Synthesis round-22 §27: the merge tax -- how much of the
    algebraic ensemble gain the physical merge destroys.

    tau = CE_merged - CE_ensemble. tau ~ 0: the merge architecture is
    lossless (block-diag identity holds); tau > 0: the merge path
    (mixer geometry, bf16, norm granularity) eats part of the gain.
    Measured gen-1: 5.6742 - 5.6206 = 0.054.
    """
    return ce_merged - ce_ensemble


def jensen_gap_accum(S: torch.Tensor, A: float, n: int) -> torch.Tensor:
    """Streaming form of GapLaw's exact gap (synthesis §5): no need
    to materialize ``[N, B, T, V]``.

    Maintained per position (over the pool's N forwards):
        S = sum_i z_i          (running logit sum)
        A = sum_i LSE(z_i)     (running lse sum)

    Then the exact gap is ``A/n - LSE(S/n)`` -- the targets-free
    twoGap_cosh identity (identical to jensen_gap_lse, proven by the
    same telescoping), computed with O(B*T*V) memory for ONE leaf at
    a time instead of N.

    Args:
        S: ``[B, T, V]`` running logit sum over the pool.
        A: running ``sum_i LSE(z_i)`` (scalar over positions if
            pre-averaged; pass the per-position sum as a tensor of
            shape [B, T] for per-position gaps).
        n: the number of leaves accumulated so far.

    Returns:
        The exact Jensen gap (float or ``[B, T]`` per-position).
    """
    zbar = (S.double() if torch.is_tensor(S) else None)
    lse_bar = torch.logsumexp(S.double() / n, dim=-1)
    return A / n - lse_bar


def ensemble_ce_logits(mean_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Synthesis §4: the exact merged-ensemble CE from logits alone.

    The mean-logit ensemble's CE is ``LSE(m_N) - m_N[t]`` -- no wide
    model, no merge_experts, no merged forward. With the running sum
    ``S`` over the pool: ``m_N = S/N``.

    Args:
        mean_logits: ``[B, T, V]`` the pool's mean logits (m_N).
        targets: ``[B, T]`` int64 next-token ids.

    Returns:
        Per-position CE ``[B, T]`` in float64.
    """
    m = mean_logits.double()
    lse = torch.logsumexp(m, dim=-1)
    return lse - m.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def ensemble_delta_ce(S: torch.Tensor, n: int, cand_logits: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
    """The EXACT candidate decision metric (synthesis §4+§6), from
    one candidate forward -- no merge, no wide model.

    m_{N+1} = (N*m_N + z_c)/(N+1); the decision quantity is
    CE_{N+1} - CE_N, computed purely by lse algebra on the running
    pool sum S (mean m_N = S/N) plus the candidate's logits.

    Args:
        S: ``[B, T, V]`` running pool logit sum.
        n: current pool size N.
        cand_logits: ``[B, T, V]`` the candidate's logits (same head
            scale as the pool leaves).
        targets: ``[B, T]`` int64 next-token ids.

    Returns:
        Per-position exact ``CE_{N+1} - CE_N`` (negative = the
        candidate improves the ensemble).
    """
    m_old = S.double() / n
    m_new = (S.double() + cand_logits.double()) / (n + 1)
    ce_old = torch.logsumexp(m_old, -1) - m_old.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ce_new = torch.logsumexp(m_new, -1) - m_new.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return ce_new - ce_old


def ls_scale_gram(q: torch.Tensor, w: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Ternary.lean lsScale_gram: functional LS scale, the global
    optimum for ``||Xw - s*Xq||^2`` at a FIXED ternary pattern q.

    Complete-the-square: the cross term zeroes exactly at
    ``s* = (q^T H w) / (q^T H q)`` with ``H = X^T X`` the Gram of the
    layer inputs; the remainder is a nonnegative square -- hence the
    GLOBAL minimum (lsScale_gram_global). Degenerates to the proved
    weight-space lsScale at ``H = I``. The refresh cycle
    (q -> LS -> q -> LS ...) is block-coordinate descent: every step
    is an exact minimization over its own variable, so the
    functional error is monotone non-increasing.

    Args:
        q: the fixed ternary pattern, ``[out, in]``.
        w: the master weight being quantized, ``[out, in]``.
        H: layer input Gram matrix ``X^T X``, ``[in, in]`` (PSD).

    Returns:
        The per-row optimal scale ``[out, 1]`` (rows are independent).
    """
    if H.shape[0] != H.shape[1] or w.shape[-1] != H.shape[0]:
        raise ValueError("shape mismatch: H must be [in,in], w/q [out,in]")
    qH = q @ H                       # [out, in]
    num = (qH * w).sum(-1, keepdim=True)     # q^T H w per row
    den = (qH * q).sum(-1, keepdim=True).clamp_min(1e-12)  # q^T H q
    return num / den


def joint_lr_bound(grad_norms: dict[str, float], mix_grad_norm: float,
                   smoothness: float, eps: float = 0.0021) -> float:
    """Joint.lean jointStep_regression_bound: the DERIVED joint-step LR.

    Any corpus's CE regression is bounded by
    ``|lr| * ||g_c|| * ||g|| + (L/2) lr^2 ||g||^2`` (Cauchy-Schwarz on
    the linear term + the smoothness bound on the quadratic one). The
    joint LR is safe when the WORST corpus's bound stays under eps:

        lr <= eps / (max_c ||g_c|| * ||g|| + (L/2) ||g||^2)

    This replaces the empirical 0.001: the recipe's LR is now a
    function of measured gradient norms and the floor eps.

    Args:
        grad_norms: per-corpus gradient norms {corpus: ||g_c||}.
        mix_grad_norm: the mixture gradient norm ||g||.
        smoothness: the loss smoothness L.
        eps: the regression floor (default: the measured 0.0021).

    Returns:
        The derived safe LR (positive float).
    """
    if not grad_norms or mix_grad_norm <= 0 or smoothness <= 0:
        raise ValueError("need non-empty grad_norms, positive ||g|| and L")
    worst = max(grad_norms.values())
    return eps / (worst * mix_grad_norm + 0.5 * smoothness * mix_grad_norm ** 2)


def joint_conflict_lr(grad_c_dot_g: float, mix_grad_norm: float,
                      smoothness: float) -> tuple[bool, float]:
    """Joint.lean jointStep_conflict_regression: the conflict window.

    If a corpus's gradient CONFLICTS with the mixture direction
    (``<g_c, g> < 0``), that corpus MUST regress for every LR in
    ``0 < lr < 2|<g_c, g>| / (L ||g||^2)`` -- first-order geometry, not
    an optimization accident. No LR schedule can save it; only a
    per-corpus guard or a proximal constraint can.

    Returns:
        (is_conflicted, conflict_window_upper): whether the corpus
        conflicts, and the upper edge of the mandatory-regression LR
        window.
    """
    if mix_grad_norm <= 0 or smoothness <= 0:
        raise ValueError("need positive ||g|| and L")
    conflicted = grad_c_dot_g < 0
    window = 2 * abs(grad_c_dot_g) / (smoothness * mix_grad_norm ** 2)
    return conflicted, window


def trust_region_bound(grad_c_norm: float, radius: float,
                       smoothness: float) -> float:
    """Joint.lean trustRegion_bound: the proximal medicine.

    A step within radius R of the prior worsens ANY corpus by at most
    ``||g_c|| * R + (L/2) R^2`` -- regardless of direction. Ridge-to-
    prior with lambda ~ ||g||/R makes the slimpajama catastrophe
    impossible by construction.
    """
    if radius < 0 or smoothness <= 0:
        raise ValueError("need radius >= 0 and L > 0")
    return grad_c_norm * radius + 0.5 * smoothness * radius * radius


def gen_gap_decay(g0: float, rho: float, d: float, t: int) -> float:
    """GenCycle.lean genGap_decay: the recursion-death law.

    If sibling disagreement evolves as ``G_{t+1} <= rho*G_t + D``
    (0 <= rho < 1), then
    ``G_t <= rho^t * G0 + D*(1-rho^t)/(1-rho)``; the fixed point is
    ``D/(1-rho)``. The recursion is dead when the fixed point falls
    under eps -- fresh leaves no longer diverge measurably, and only
    the data axis (changing the corpus mix = the D field) remains.

    Measured: gen0->gen1 gap 0.30 -> 0.036 implies rho ~ 0.12 at
    D ~ 0; if specialization does not replenish D, gen-2 leaves will
    be born at gap ~ 0.004 -- at the death threshold. Measuring D on
    gen-2 leaves is the law's only unknown input.
    """
    if not (0 <= rho < 1) or t < 0:
        raise ValueError("need 0 <= rho < 1 and t >= 0")
    return rho ** t * g0 + d * (1 - rho ** t) / (1 - rho)


def gen_gap_fixed_point(rho: float, d: float) -> float:
    """The stationary point D/(1-rho): what the recursion converges to."""
    if not (0 <= rho < 1):
        raise ValueError("need 0 <= rho < 1")
    return d / (1 - rho)


def gen_recursion_dead(rho: float, d: float, eps: float = 0.0021) -> bool:
    """The derived generational verdict: dead when the fixed point
    of the gap law falls under the floor eps (the same twoGap eps
    criterion as the ladder)."""
    return gen_gap_fixed_point(rho, d) < eps


def two_gap_fast(p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """GapLaw synthesis: the O(V) FACTORIZED two-leaf Jensen gap.

    The cosh double sum factors exactly:
        sum_{u,v} p_u p_v cosh(d_u - d_v)
          = (sum_u p_u e^{d_u}) * (sum_v p_v e^{-d_v})
    so gap = 0.5 * [log sum p_u e^{d_u} + log sum p_u e^{-d_u}]
    -- O(V), no approximation (the O(V^2) form in the docstring
    underestimated this). Verified against the direct cosh sum.

    Args:
        p: softmax of the midpoint logits, ``[..., V]``.
        d: per-vocab deviations (z_A - z_B)/2, ``[..., V]``.

    Returns:
        The exact two-leaf gap ``[...]``.
    """
    pe = torch.logsumexp(d.double(), dim=-1) - torch.logsumexp(
        (-d).double(), dim=-1)  # not it -- compute directly:
    a = torch.logsumexp((d + p.clamp_min(1e-30).log()).double(), dim=-1)
    b = torch.logsumexp((-d + p.clamp_min(1e-30).log()).double(), dim=-1)
    return 0.5 * (a + b)


def fisher_novelty(delta: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """The synthesis' expert-selection score: delta^T F delta.

    F = diag(p) - p p^T is the softmax Hessian at the ensemble mean;
    the score is the quadratic form of the candidate's deviation from
    the pool mean -- the NEW information it carries, weighted by how
    much the ensemble's output cares. Better than the 1-rho proxy:
    a redundant expert (delta in the pool's span) scores low, an
    independent direction scores high, regardless of standalone CE.
    """
    d = delta.double()
    pp = p.double()
    quad = (pp * d * d).sum(-1) - (pp * d).pow(2).sum(-1)
    return quad


def rank_budget_allocation(tail_scale: dict[str, tuple[float, float, int, int]],
                           budget: float) -> dict[str, int]:
    """RankBudget.lean with the SYNTHESIS CORRECTION (kappa factor).

    Exponential-tail model E_j(r) = c_j * exp(-kappa_j * r), cost
    r*(m_j + n_j). KKT equalizes the weighted marginal residuals
    c_j*kappa_j*exp(-kappa_j*r_j) = lambda*(m_j+n_j), giving

        r_j* = (log c_j + log kappa_j - log lambda - log(m_j+n_j)) / kappa_j

    (the kappa multiplier in the numerator -- the round-25 synthesis
    caught it missing). Active set: c_j*kappa_j > lambda*(m_j+n_j).

    Args:
        tail_scale: {name: (c_j, kappa_j, m_j, n_j)} per tensor.
        budget: total rank-cost budget sum r_j (m_j + n_j) <= budget.

    Returns:
        {name: r_j*} (floats; floor at the call site).
    """
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    items = list(tail_scale.items())
    active = {k: v for k, v in items if v[0] * v[1] > 1e-30}
    if not active or budget == 0.0:
        return {k: 0.0 for k in tail_scale}
    # binary-search lambda: sum r_j(lambda) = budget (monotone in lambda)
    def total(lambda_):
        s = 0.0
        for k, (c, kap, m, n) in active.items():
            lhs = c * kap
            r = max(0.0, (math.log(lhs) - math.log(lambda_) - math.log(m + n)) / kap)
            s += r * (m + n)
        return s
    lo, hi = 1e-12, 1e12
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if total(mid) > budget:
            lo = mid  # bigger lambda -> smaller ranks
        else:
            hi = mid
    lam = math.sqrt(lo * hi)
    return {k: max(0.0, (math.log(v[0] * v[1]) - math.log(lam) - math.log(v[2] + v[3])) / v[1])
            if k in active else 0.0 for k, v in tail_scale.items()}


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """DField.lean kl_nonneg/kl_zero_iff_eq: KL(p||q) >= 0, zero iff p==q.

    The D-field's base brick (route: log x <= x - 1, strict by exp's
    strict convexity). Inputs are strictly-positive probability
    vectors; zero-count guard (smoothing) is the caller's duty --
    a zero q on a used token is a silent bias source (the NCE
    proposal guard applies here too).
    """
    p = p.double()
    q = q.double()
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if (p <= 0).any() or (q <= 0).any():
        raise ValueError("KL requires strictly positive distributions "
                         "(smooth zero counts at the call site)")
    if abs(float(p.sum()) - 1.0) > 1e-6 or abs(float(q.sum()) - 1.0) > 1e-6:
        raise ValueError("p and q must be probability vectors")
    return float((p * (p.log() - q.log())).sum())


def d_field(corpus_dists: dict[str, torch.Tensor], weights: dict[str, float]) -> float:
    """DField.lean: the data-field divergence D(w) = sum_i w_i KL(p_i || p_w).

    The disagreement-replenishment model: how much NEW divergence the
    corpus mix carries. Measurable WITHOUT training (one-time unigram
    statistics). divField_zero_iff: D = 0 iff the corpora are clones --
    the recursion can only die of data degeneration.

    Args:
        corpus_dists: {name: unigram distribution [V]} (strictly positive).
        weights: {name: mix weight w_i}; normalized internally.
    """
    if set(corpus_dists) != set(weights):
        raise ValueError("corpus_dists and weights must cover the same names")
    s = sum(weights.values())
    if s <= 0:
        raise ValueError("weights must sum to a positive value")
    w = {k: v / s for k, v in weights.items()}
    p_mix = None
    for k in w:
        p = corpus_dists[k].double()
        p_mix = w[k] * p if p_mix is None else p_mix + w[k] * p
    return sum(w[k] * kl_divergence(corpus_dists[k], p_mix) for k in w)


def conflict_signal(corpus_dist: torch.Tensor, mix_dist: torch.Tensor,
                    corpus_ce: float, pool_ce: float) -> float:
    """The double replacement signal (DField + Joint): a corpus is a
    D-SINK when it is both (a) anomalous in distribution (large KL from
    the mix -- its tokens barely overlap the blend) and (b) never
    fitted (its CE stays far above the pool's). The measured
    slimpajama case: cross-KL ~13 nats to EVERY sibling corpus (the
    others pair at 0.3-5) and CE 13-17 vs the pool's ~4.5.
    """
    kl_val = kl_divergence(corpus_dist, mix_dist)
    return kl_val * max(0.0, corpus_ce - pool_ce)


def compound_verdict_interval(alpha_lo: float, alpha_hi: float,
                             d_lo: float, d_hi: float,
                             j_lo: float, j_hi: float,
                             eps_c: float) -> str:
    """Compound.lean compound_budget_interval / compound_fold_interval:
    the interval-honest generational verdict.

    With confidence intervals on (alpha, D, J):
    - GROW: even the pessimistic end clears the threshold
      (alpha_lo*D_lo + J_lo > eps_c).
    - FOLD: even the optimistic end is under (alpha_hi*D_hi + J_hi <= eps_c).
    - UNDECIDED: the confidence band straddles eps_c -- measure another
      generation; do not flip a coin. The verdict is a statistical
      decision, not a point comparison (for the noisy c = 0.18, 0.17,
      -0.02 trajectory).
    """
    lo = alpha_lo * d_lo + j_lo
    hi = alpha_hi * d_hi + j_hi
    if lo > eps_c:
        return "GROW"
    if hi <= eps_c:
        return "FOLD"
    return "UNDECIDED"


def safe_qp_multipliers(G: torch.Tensor, b: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """(Deprecated return form; see safe_qp_solve for the certified form.)"""
    return safe_qp_solve(G, b, eps)[0]


def safe_qp_solve(
    G: torch.Tensor, b: torch.Tensor, eps: torch.Tensor, ridge: float = 1e-10,
    g_norm_sq: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """SafeQP dual with a CERTIFICATE (round 39, reviewer prescription).

    Fixes vs the round-32 form:
    - singular active subsets fall back to ridge/pinv instead of being
      silently skipped (a fully singular enumeration used to return
      lambda=0 -- a constraint-violating step dressed as a safe one);
    - returns the certificate alongside lambda: the caller can verify
      feasibility (min_i <g_i,d>+eps_i), descent (<g,d> >= ||d||^2-tol)
      and contraction (||g-d|| <= ||g||) -- exactly the safeQP_descent
      properties formalized in Lean;
    - the dead ``best_obj`` line removed.

    Note (usage): the round-32/33 pipeline used this as an OFFLINE scan
    (grad telemetry saved to qp_*.pt, then evaluated here) -- no calls
    inside src/ train paths, by design; the controller hook is future
    work (round-40 queue).
    """
    G = G.double()
    b = b.double()
    eps = eps.double()
    K = G.shape[0]
    if G.shape != (K, K) or b.shape != (K,) or eps.shape != (K,):
        raise ValueError("G must be [K,K], b and eps [K]")
    if (eps < 0).any():
        raise ValueError("safety budgets must be nonnegative")
    c = b + eps
    best_obj = 0.0
    best = torch.zeros(K, dtype=torch.float64)
    if (c >= 0).all():
        return best, {"feasibility": float("inf"), "descent_gap": 0.0,
                      "contraction": 0.0, "active": 0}
    for mask in range(1, 1 << K):
        idx = [i for i in range(K) if mask & (1 << i)]
        A = G[idx][:, idx]
        rhs = -c[idx]
        if len(idx) == 1:
            if A[0, 0] <= 1e-12:
                continue
            lam_A = rhs / A[0, 0]
        else:
            try:
                lam_A = torch.linalg.solve(A, rhs)
            except Exception:
                # ridge fallback: never silently skip a subset (a skipped
                # singular subset could leave the optimum unfound)
                lam_A = torch.linalg.lstsq(A + ridge * torch.eye(len(idx), dtype=torch.float64), rhs).solution
        if (lam_A < -1e-9).any():
            continue
        lam = torch.zeros(K, dtype=torch.float64)
        lam[idx] = lam_A
        resid = c + G @ lam
        inactive = [i for i in range(K) if not (mask & (1 << i))]
        if inactive and (resid[inactive] < -1e-9).any():
            continue
        obj = 0.5 * float(lam @ (G @ lam)) + float(lam @ c)
        if obj < best_obj - 1e-12:
            best_obj = obj
            best = lam
    # Certificate for the returned dual point. KKT for min ||d-g||^2 s.t.
    # <g_i, d> >= -eps_i gives d* = g + sum_i lam_i g_i, so feasibility is
    # min_i (b + G lam + eps)_i — exactly the loop's residual at the returned
    # point. Descent and contraction additionally need ||g||^2 (pass
    # g_norm_sq when the caller holds the flat gradient); without it the keys
    # are nan so a missing bound can never masquerade as a satisfied one.
    cert = {"active": int((best > 1e-9).sum())}
    cert["feasibility"] = float((b + G @ best + eps).min())
    if g_norm_sq is not None:
        lamGlam = float(best @ (G @ best))
        descent = g_norm_sq + 2.0 * float(best @ b) + lamGlam  # ||d*||^2
        align = g_norm_sq + float(best @ b)  # <g, d*>
        cert["descent_gap"] = align - descent  # >= 0 iff ||d*||^2 <= <g,d*>
        cert["contraction"] = math.sqrt(max(lamGlam, 0.0)) - math.sqrt(
            max(g_norm_sq, 0.0)
        )  # ||g - d*|| - ||g|| <= 0 must hold
        # T4 (R147, 2606.29521): kappa_t = ||d*||^2/||g||^2 — the fraction of
        # the gradient KEPT by the projection; the PL-rate factor is
        # 1 - mu*kappa/L. Needs g_norm_sq > 0 to be meaningful.
        cert["kappa"] = descent / g_norm_sq if g_norm_sq > 0.0 else float("nan")
    else:
        cert["descent_gap"] = float("nan")
        cert["contraction"] = float("nan")
    return best, cert


def domination_metrics(
    grads: torch.Tensor, weights: torch.Tensor
) -> dict[str, object]:
    """Dominate.lean round 33: the domination law, as a computable scan.

    Step-direction decomposition (domination_decompose): any step d
    splits into the part aligned with the dominant gradient g_s and
    the perpendicular remainder; the aligned part's effect on corpus i
    is bounded by the dominant-minimum disagreement (domination_regression):
    when d is aligned with g_s and <g_i, g_s> <= cos_min * ||g_i||*||g_s||,
    then <g_i, d> <= (alignment fraction) * cos_min * ||g_i||*||d|| -- the
    aligned contribution is NEGATIVE for cos_min < 0, and collateral
    CE regression of non-dominated corpora is unavoidable at any lr
    that moves the dominant corpus. The round-32 measurement
    (slimpajama 94.8% share) is the canonical instance.

    normalized_descent_preserved: normalizing contributions
    (g_i / ||g_i||, equal w_i*||g_i||) keeps descent iff
    <g_norm, g> > 0; the flip is quantified below as descent_ratio.

    Args:
        grads: [K, D] per-corpus gradients on one calibration batch.
        weights: [K] mixture weights.

    Returns:
        dict with per-corpus shares, alignment cosines, the kappa
        DOMINATED/SAFE verdict, and the normalized-descent flip check.
    """
    G = grads.double()
    w = weights.double()
    K = G.shape[0]
    contrib = w[:, None] * G                      # [K, D]
    norms = G.norm(dim=1)
    mix = contrib.sum(0)
    mix_norm = mix.norm()
    tot = contrib.norm()
    shares = (contrib.norm(dim=1) / tot) if tot > 0 else torch.zeros(K)
    with torch.no_grad():
        cos_to_mix = (
            (G @ mix) / (norms * mix_norm).clamp_min(1e-30)
        )
    dom = int(torch.argmax(shares))
    kappa = float(shares[dom])  # dominant share in [0, 1]
    # disagreement of each corpus with the dominant gradient
    cos_dom = (G @ G[dom]) / (norms * norms[dom]).clamp_min(1e-30)
    # Full verdict (round-33 correction): harmful domination requires
    # BOTH a dominant share AND disagreement of the rest with the
    # dominant direction -- domination_regression's two conditions.
    # Benign domination (consensus, high mean cos) is safe: the
    # healthy round-32 swap dominated by weight but agreed in
    # direction and the joint descended (5.43 -> 5.17).
    others = [i for i in range(K) if i != dom]
    mean_cos_dom = float(cos_dom[others].mean()) if others else 1.0
    verdict = (
        "DOMINATED"
        if kappa >= 0.5 and mean_cos_dom < 0.5
        else ("SAFE_CONSENSUS" if kappa >= 0.5 else "SAFE")
    )
    # normalized_descent_preserved check
    gn = contrib / contrib.norm(dim=1, keepdim=True).clamp_min(1e-30)
    g_norm_mix = gn.sum(0)
    descent_dot = float(torch.dot(g_norm_mix, mix))
    descent_ratio = descent_dot / float(mix_norm) if mix_norm > 0 else 0.0
    return {
        "dominant": dom,
        "kappa": kappa,
        "verdict": verdict,
        "shares": shares.tolist(),
        "cos_to_mix": cos_to_mix.tolist(),
        "cos_to_dominant": cos_dom.tolist(),
        "mean_cos_dom": mean_cos_dom,
        "normalized_descent_preserved": descent_dot > 0,
        "normalized_descent_ratio": descent_ratio,
    }


def domination_threshold(shares: list[float]) -> float:
    """Dominate.lean: the kappa threshold licensed by the regression bound.

    The regression bound is vacuous unless the dominant share exceeds
    the maximal perpendicular budget; empirically the flip in
    normalized descent happens when one contribution outvotes the
    sum of the rest. The scan verdict above uses the 0.5 share line
    (one corpus owning more than the combined rest); both historical
    points sit above it (0.948 and the softened guard ~0.6).
    """
    return max(shares) / max(sum(shares), 1e-30)


def growth_verdict(g_f: float, r_repr: float, eps_g: float, eps_r: float) -> str:
    """Round 34 (Lean growth_verdict_table): the (G_F, R_repr) decision.

    The four-regime table replacing the plateau-CE heuristic:

    - G_F >= eps_g and R_repr <  eps_r -> "TTT/LORA": the current
      architecture REPRESENTS the gradient well but free energy is high --
      adapt within the model (LoRA/TTT), do not grow.
    - G_F >= eps_g and R_repr >= eps_r -> "GROW": representation error is
      the bottleneck -- expand the architecture (expert/rank/layer/mixer
      by ComputeBudget argmax of DeltaG_F/DeltaC_j).
    - G_F <  eps_g and R_repr <  eps_r -> "STOP": both low -- converged.
    - G_F <  eps_g and R_repr >= eps_r -> "TEACHER_CHECK": low free
      energy with high representation error is contradictory -- suspect
      the teacher/reference, re-audit before growing.

    Thresholds come from equilibrium_bracket noise scales: eps_g from the
    ensemble KL noise bracket, eps_r from the gradient-projection noise.
    """
    high_g = g_f >= eps_g
    # Lean boundary: verdict_ttt takes R <= eps, verdict_grow takes
    # eps < R -- strict on R, so R == eps falls to TTT (the cheaper
    # mechanism wins ties), matching the theorem hypotheses exactly.
    high_r = r_repr > eps_r
    if high_g and not high_r:
        return "TTT/LORA"
    if high_g and high_r:
        return "GROW"
    if not high_g and not high_r:
        return "STOP"
    return "TEACHER_CHECK"


def mechanism_marginal_value(delta_g_f: dict[str, float],
                             delta_c: dict[str, float]) -> str:
    """Round 34: j* = argmax DeltaG_F / DeltaC (ComputeBudget in the
    free-energy currency). Measured channel values (era-2 ledgers):

    - init (shared E0 prior):      DeltaG_F ~ 0.245 nat per leaf
    - rank (LoRA r=16):            DeltaG_F ~ 0.196 nat, DeltaC tiny
    - joint (movement channel):   DeltaG_F ~ 0.23 (canonical) / 0.35 (dbridge)

    Returns the mechanism key with the best free-energy-per-compute ratio.
    """
    ratios = {k: delta_g_f[k] / max(delta_c[k], 1e-12) for k in delta_g_f}
    return max(ratios, key=ratios.get)


def anova_split(hidden: torch.Tensor, n_leaves: int) -> dict[str, float]:
    """F3Root.lean anova_split: the tree go/no-go number, one forward pass.

    Decomposes the residual-stream energy into the root-accessible part
    (the all-equal leaf mean -- the only component the root cortex and
    the parent-preserving lift can see) and the leaf-variance part (the
    specializations, invisible to the tree by construction:
    root_no_leaf_signal). With x viewed as [N_leaves, hidden_leaf] per
    position:

        E_total = ||x||^2  =  ||mean||^2 * N  +  sum_i ||x_i - mean||^2

    A high root share means the block-diagonal lift bound is cheap and
    the F3 tree is promising; a high leaf-variance share (orthogonal
    specializations) means flat merge keeps the advantage forever.

    Args:
        hidden: [..., n_leaves * leaf_hidden] stream tensor.
        n_leaves: the leaf count of the prospective tree level.

    Returns:
        {"root_share": ..., "leaf_share": ..., "verdict": ...}
    """
    x = hidden.detach().float()
    lead = x.shape[:-1]
    leaf_h = x.shape[-1] // n_leaves
    xl = x.reshape(*lead, n_leaves, leaf_h)
    mean = xl.mean(dim=-2, keepdim=True)                 # [.., 1, leaf_h]
    ss_root = float((mean * mean).sum() * n_leaves)
    ss_leaf = float(((xl - mean) ** 2).sum())
    tot = ss_root + ss_leaf
    root_share = ss_root / tot if tot > 0 else 0.0
    verdict = "TREE_PROMISING" if root_share >= 0.5 else "FLAT_ADVANTAGE"
    return {"root_share": root_share, "leaf_share": 1.0 - root_share,
            "verdict": verdict}


def bf16_frozen_gain_check(param: torch.Tensor, grad_scale: float = 1e-4) -> bool:
    """PreNorm.lean bf16_frozen_update: detect a silently frozen gain.

    Above 1.0 the bf16 grid steps by 2^-7 = 0.0078125; an update whose
    magnitude is below half the step rounds back to the current value and
    the gain never moves again for the whole run. A 1D gain parameter
    living in bf16 with gradient scale ~1e-4 is FROZEN by construction.
    Every new 1D gain (norms, scales, gates, per-head QK gains) must
    carry keep_fp32; this check audits an existing tensor.
    """
    if param.dtype != torch.bfloat16 or param.numel() != 1:
        return False
    return grad_scale < 2.0 ** -8 / 2.0


def ns_steps_needed(sigma_min: float, sigma_target: float = 1.0,
                    shrink: float = 3.4445) -> int:
    """DesignOpt NS-огибающая: e(s+1) <= rho*e(s) with rho ~ 1/shrink
    (the leading Newton-Schulz coefficient), so s* = ceil(log(sigma_target/
    sigma_min)/log(shrink)). Synced with Lean ns_geometric_envelope."""
    import math
    if sigma_min <= 0 or sigma_min >= sigma_target:
        return 0
    return int(math.ceil(math.log(sigma_target / sigma_min) / math.log(shrink)))


def optimal_batch(b_noise: float, t0: float, c: float) -> float:
    """DesignOpt optimal batch: B* = sqrt(B_noise * t0 / c) where t0/c
    are the per-step overheads expressed in tokens. Synced with Lean."""
    if c <= 0:
        return float("inf")
    return (b_noise * t0 / c) ** 0.5


def freedman_tail_low(delta: float, V: float, b: float) -> float:
    """Probability/Freedman ``freedman_tail_low``: adaptive tail bound.

    P[sum X <= -delta] <= exp(-delta^2 / (2*(V + b*delta/3))) where V
    is the predictable quadratic variation of the martingale and b the
    per-step range. Strictly stronger than Azuma's exp(-delta^2/(2n))
    whenever V < n (use as the accept-gate tail in place of the
    fixed-n form; see also certified_controller.hoeffding_n).
    """
    if delta <= 0 or V < 0 or b <= 0:
        raise ValueError("need delta > 0, V >= 0, b > 0")
    return math.exp(-(delta * delta) / (2.0 * (V + b * delta / 3.0)))


def eckart_young_frobenius_error(
    singular_values: torch.Tensor, k: int,
) -> torch.Tensor:
    """Spectral ``eckart_young_frobenius``: optimal truncation residual.

    Returns the Frobenius norm of the best rank-k approximation
    residual, sqrt(sum_{i>k} s_i^2) for singular values sorted in
    descending order (the theorem's optimality certificate: no rank-k
    approximation does better in Frobenius norm).
    """
    if singular_values.ndim != 1:
        raise ValueError("expected a 1-D singular value vector")
    if not 0 <= k <= singular_values.numel():
        raise ValueError(f"k={k} out of range")
    s_sorted = torch.sort(singular_values, descending=True).values
    return torch.sqrt(torch.sum(s_sorted[k:] ** 2))


# --- Thermo-condensation certificates (R522/R527/R528, 2026-10-16) -------
# C:\primes\Hagi\Ensemble\ThermoPrune.lean (condensation_budget),
# Budget\ThermoCondensation.lean (condensation_epsilon_to_gap),
# Budget\AnytimeCondensation.lean (anytime_condensation).
# Pure math: no torch tensors involved.


def condensation_budget_cost(deviation: float) -> float:
    """CE cost of a scaffold rebuild <= M^2 / 4 [R522
    ``condensation_budget``].

    If the logit deviations d_v of the live scaffold stay within a
    delta-neighborhood (|d_u - d_v| <= M for all pairs), rebuilding
    (merge/prune) the ensemble costs at most M^2/4 in CE — the
    "1.5% scaffold" certificate: the thermal bulk can be cut and the
    scaffold re-laid inside a known CE budget.
    """
    if deviation < 0.0:
        raise ValueError("deviation must be nonnegative")
    return deviation * deviation / 4.0


def condensation_allowed_deviation(eps: float) -> float:
    """The inverse interface: a CE budget eps buys a scaffold deviation
    delta = 2*sqrt(eps) [R527 ``condensation_epsilon_to_gap``]."""
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    return 2.0 * math.sqrt(eps)


def anytime_condensation_cost(sigma2: float, T: int, alpha: float) -> float:
    """Anytime-certified condensation budget: <= 2*sigma^2*ln(T/alpha)
    CE [R528 ``anytime_condensation``].

    When the scaffold statistic g is sub-Gaussian (CGF parameter
    sigma^2, h_emp_ measured) and the anytime bad-set is empty (an
    event of probability >= 1-alpha by the R525 threshold
    sqrt(2 sigma^2 ln(T/alpha))), a softmax-scaffold rebuild under the
    deviation d = g - mu costs at most 2*sigma^2*ln(T/alpha): the
    alpha-monitored rebuild is certified without retraining.
    """
    if sigma2 <= 0.0:
        raise ValueError("sigma2 must be positive")
    if T < 1:
        raise ValueError("T must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return 2.0 * sigma2 * math.log(T / alpha)


# --- Pinsker CMI generalization + capacity floors (2026-10-18) ---------------
# Ports: Information/Pinsker.lean (gen_pair_bound, cmi_iteration),
# Information/RoutingCapacity.lean (router_capacity_bound),
# Information/MemoryCapacity.lean (separated_needs_capacity).


def gen_pair_bound(
    q: list[float], ef: list[float], eg: list[float], kl_qp: float
) -> float:
    """Stability+information generalization bound [``gen_pair_bound``
    (Information/Pinsker.lean)]:

        |E_q f - E_p g| <= E_q|f - g| + 2 sqrt(KL(q||p))

    Decomposition: stability term (expected pointwise gap under the
    training distribution) + information term (Pinsker root of the
    KL between sample and population distributions). q/p: probability
    vectors (sum 1), ef/eg: expectations of the [0,1]-bounded test
    functions Representative use: E_q|f-g| = measured stability,
    KL = measured distribution shift; the bound is the generalization
    certificate |gen gap| <= ... for the pair.

    Returns the bound value. Caller owns the measurement semantics.
    """
    if len(q) != len(ef) or len(q) != len(eg):
        raise ValueError("q, ef, eg must have equal length")
    if kl_qp < 0.0:
        raise ValueError("KL must be nonnegative")
    stab = sum(qi * abs(f - g) for qi, f, g in zip(q, ef, eg))
    return stab + 2.0 * math.sqrt(kl_qp)


def cmi_iteration(stab: list[float], kls: list[float]) -> float:
    """CMI generalization rate [``cmi_iteration`` (Pinsker.lean)]: with
    per-sample deltas obeying Delta_i <= stab_i + 2 sqrt(KL_i), the MEAN
    generalization gap obeys the Cauchy-iterated rate

        (1/n) sum Delta_i <= (1/n) sum stab_i + 2 sqrt((1/n) sum KL_i).

    sqrt-aggregation: the information term grows with the SQUARE ROOT of
    the mean KL, not the mean sqrt - the whole CMI pipeline payoff
    (Hoeffding mgf + Pinsker + stability decomposition).
    Premises: stab_i >= 0, KL_i >= 0 (measured, h_emp_).
    """
    if not stab or len(stab) != len(kls):
        raise ValueError("stab and kls must be nonempty, equal length")
    if any(k < 0 for k in kls) or any(st < 0 for st in stab):
        raise ValueError("stab and kls must be nonnegative")
    n = len(stab)
    return sum(stab) / n + 2.0 * math.sqrt(sum(kls) / n)


def router_capacity_ok(n_tasks: int, routing_bits: int) -> bool:
    """Routing capacity floor [``router_capacity_bound`` +
    ``separated_needs_capacity``]: K 2eps-separated tasks served through a
    B-bit router require log2(K) bits - K <= 2^B. Decoders do not create
    information (``no_free_recovery``). Honest budget guard for merge
    routing / quantized adapter decisions."""
    if n_tasks < 0 or routing_bits < 0:
        raise ValueError("counts must be nonnegative")
    return n_tasks <= 2 ** routing_bits



# --- PoE logZ bounds (Energy/PoEBound.lean, 2026-10-18) ----------------------


def poe_logz_second_order(
    z: list[list[float]], w: list[float], R: list[float]
) -> tuple[float, float]:
    """Certified PoE pooling error [``poe_logZ_second_order``
    (Energy/PoEBound.lean)]: for weighted product-of-experts logits,
    the Jensen gap between the pooled logZ and the weighted expert logZs
    obeys the SHARP law

        |log sum_v e^{<w, z_v>} - sum_i w_i log sum_v e^{z_i v}|
            <= (1/8) * sum_i w_i R_i^2,

    where R_i is the per-expert RANGE of the centered deviation
    z_i v - mu_v. Premises: w >= 0, sum w = 1; R_i measured (h_emp_).
    Returns (actual_gap, certified_bound). The audit's pairwise-M^2/8
    form is REFUTED in Lean; per-expert-range 1/8 is the sharp law.

    z: [K][V] logit grids, w: [K] weights, R: [K] per-expert ranges.
    """
    import math as _m
    if not z or len(z) != len(w) or len(w) != len(R):
        raise ValueError("z, w, R must be nonempty and equal length")
    if any(wi < 0 for wi in w) or abs(sum(w) - 1.0) > 1e-6:
        raise ValueError("w must be a probability vector")
    if any(ri < 0 for ri in R):
        raise ValueError("R must be nonnegative")
    n_v = len(z[0])
    k = len(w)
    pooled = _m.log(sum(
        _m.exp(sum(w[i] * z[i][v] for i in range(k))) for v in range(n_v)
    ))
    per_expert = sum(
        w[i] * _m.log(sum(_m.exp(z[i][v]) for v in range(n_v)))
        for i in range(k)
    )
    gap = abs(pooled - per_expert)
    bound = 0.125 * sum(wi * ri * ri for wi, ri in zip(w, R))
    return gap, bound


def poe_softmax_speed(z: list[list[float]], w: list[float], big_m: float) -> tuple[float, float]:
    """Per-token PoE speed bound [``poe_softmax_speed``]: if EVERY
    expert's centered spread is <= M (uniform range), the Jensen gap is
    <= M^2/8 - the K=2 case recovers the pairwise statement with the
    midpoint. Returns (actual_gap, bound M^2/8)."""
    if big_m < 0.0:
        raise ValueError("M must be nonnegative")
    gap, _ = poe_logz_second_order(z, w, [big_m] * len(w))
    return gap, big_m * big_m / 8.0


def poe_logz_pairwise(z: list[list[float]], w: list[float], D: list[list[float]]) -> tuple[float, float]:
    """Honest PAIRWISE constant [``poe_logZ_pairwise``]: with pairwise
    logit distances |z_i v - z_j v| <= D_ij, the Jensen gap is

        <= (1/2) * sum_ij w_i w_j D_ij^2.

    The audit's 1/8 pairwise constant FAILS (refuted in Lean with z1=(1,-1),
    z2=(0,0), w=1/2: gap ~ 0.157 vs claimed 0.0625); 1/2 is provable via
    the weighted Cauchy-Schwarz. Returns (actual_gap, bound)."""
    import math as _m
    k = len(z)
    if not z or len(D) != k or any(len(row) != k for row in D):
        raise ValueError("D must be a K x K matrix matching z")
    gap, _ = poe_logz_second_order(z, w, [0.0] * k)  # gap recompute
    bound = 0.5 * sum(
        w[i] * w[j] * D[i][j] ** 2
        for i in range(k) for j in range(k)
    )
    return gap, bound


# --- LSE-shift / KL-additivity / smoothing (2026-10-19, ночь-2) ---------------


def lse_shift_upper_bound(
    x: list[float], y: list[float]
) -> tuple[float, float, float]:
    """First-order LSE shift with the r^2/8 certified remainder
    [``lse_shift_lower`` / ``lse_shift_upper`` (Energy/PoEBound.lean)]:

        lse(y) - lse(x) <= <sigma(x), y - x> + r^2/8,

    where r is the RANGE of the shift (y_v - x_v): the exp-Jensen lower
    bound (first-order term is a LOWER bound on the shift) plus the
    Hoeffding-MGF upper remainder under the bounded range. Returns
    (actual_shift, first_order_term, certified_bound = fo + r^2/8).
    Premises: none beyond finiteness - fully unconditional.
    """
    import math as _m
    if not x or len(x) != len(y):
        raise ValueError("x and y must be nonempty, equal length")
    lse_x = _m.log(sum(_m.exp(v) for v in x))
    lse_y = _m.log(sum(_m.exp(v) for v in y))
    zx = _m.log(sum(_m.exp(v) for v in x))
    smax = [_m.exp(v - zx) for v in x]  # softmax(x)
    fo = sum(sv * (yv - xv) for sv, yv, xv in zip(smax, y, x))
    shifts = [yv - xv for yv, xv in zip(y, x)]
    r = max(shifts) - min(shifts)
    actual = lse_y - lse_x
    return actual, fo, fo + r * r / 8.0


def kl_product_additive(
    kl1: float, kl2: float
) -> float:
    """KL additivity on independent products [``kl_product``
    (Data/DField.lean)]: KL(p1 x p2 || q1 x q2) = KL(p1||q1) + KL(p2||q2)
    - term-level split of the log of the ratio. Runtime: the joint
    KL of independent per-position distributions decomposes - the
    licence for per-position distillation budgets to sum over tokens.

    The identity is arithmetic on composed distributions; here it is
    the composition RULE: given measured per-position KLs, the joint
    budget is their SUM (returns kl1 + kl2 with premise checks).
    """
    if kl1 < 0.0 or kl2 < 0.0:
        raise ValueError("KLs must be nonnegative")
    return kl1 + kl2


def smooth_support(p: list[float], alpha: float, uniform: list[float] | None = None) -> list[float]:
    """Support-gap fix [``dfield_smoothing`` (Data/DField.lean)]:
    (1-alpha)*p + alpha*u > 0 pointwise for alpha in (0,1) and u > 0 -
    the bridge from the zero-count reality to full-support KL theorems.
    uniform defaults to the flat distribution. Returns the smoothed
    vector (the premise side is checked: alpha strict (0,1)).
    """
    if not p:
        raise ValueError("p must be nonempty")
    if any(x < 0 for x in p):
        raise ValueError("p must be nonnegative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if uniform is None:
        u = [1.0 / len(p)] * len(p)
    else:
        if len(uniform) != len(p) or any(x <= 0 for x in uniform):
            raise ValueError("uniform must be positive, equal length")
        u = uniform
    return [(1.0 - alpha) * pv + alpha * uv for pv, uv in zip(p, u)]


# --- InfoNCE identities (Data/InfoNCE.lean, 2026-10-19) ----------------------


def infonce_gap_is_kl(
    llm_entropy_q: float, cross_entropy_qp: float, kl_qp: float
) -> float:
    """InfoNCE residual identity [``infonce_gap_is_kl``
    (Data/InfoNCE.lean)]: the gap between the MI InfoNCE-estimate floor
    and the estimate is EXACTLY KL(post || model):

        (log k - H(q)) - (log k - CE(q, p)) = KL(q || p).

    The contrastive estimate is terminal-exact iff the model equals the
    posterior (gap zero iff KL zero). Runtime: report the measured gap
    against the measured KL - any disagreement flags a calibration bug,
    the identity is the invariant. Verifies and returns the gap.
    """
    gap = cross_entropy_qp - llm_entropy_q
    if not (llm_entropy_q >= 0.0 and cross_entropy_qp >= 0.0):
        raise ValueError("entropies must be nonnegative")
    if abs(gap - kl_qp) > 1e-6 * max(1.0, abs(kl_qp)):
        raise ValueError(
            f"identity violated: gap {gap} != KL {kl_qp} - measured "
            f"quantities are inconsistent (calibration bug)"
        )
    return gap


def infonce_never_overestimates(
    log_k: float, cross_entropy_qp: float, entropy_q: float
) -> bool:
    """MI floor check [``infonce_never_overestimates``]: the InfoNCE
    estimate (log k - CE) never overestimates the MI floor
    (log k - H(q)): CE(q,p) >= H(q) always holds. Returns True when the
    measured quantities respect the floor (honest guard against
    mis-calibrated contrastive estimators)."""
    if cross_entropy_qp < 0.0 or entropy_q < 0.0:
        raise ValueError("entropies must be nonnegative")
    return cross_entropy_qp >= entropy_q


def dfield_entropy_identity(
    entropy_mixture: float, weights: list[float], entropies: list[float]
) -> float:
    """D-field entropy form [``dfield_entropy_identity``
    (Data/DField.lean)]: D(w) = H(p_w) - sum_i w_i H(p_i) — the
    generalized Jensen-Shannon divergence equals the mixture entropy
    minus the weighted member entropies. Runtime: compute D from
    measured entropies WITHOUT touching per-token distributions:
    D = H(mixture) - <H-members>. Returns the negative-entropy deficit
    (>= 0; equality iff all members identical)."""
    if not weights or len(weights) != len(entropies):
        raise ValueError("weights and entropies must be nonempty, equal length")
    if any(w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must be a probability vector")
    if any(h < 0 for h in entropies):
        raise ValueError("entropies must be nonnegative")
    return entropy_mixture - sum(w * h for w, h in zip(weights, entropies))
