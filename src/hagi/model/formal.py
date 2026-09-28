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
    ``G_inf / N^2 < eps``, i.e. ``N > sqrt(G_inf / eps)``. Replaces
    the fitted TOL_GAP with a function of the measured pool
    covariance (via G_inf) and the floor eps.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    return math.sqrt(max(g_inf, 0.0) / eps)


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
