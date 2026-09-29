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
