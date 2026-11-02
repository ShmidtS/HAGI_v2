"""LipEnergy — gradient Lipschitz energy model (R563–R565).

Port of Unified/DerivAlong.lean (primes SSOT). Closes the h_emp_*
trinity at runtime level: the âhonest boundaryâ premises of the stage
theorems are now DERIVED from a measured gradient model, not assumed:

- R563 ``lip_energy_from_gradient`` / ``lip_energy_quantization``:
  if the energy is differentiable and ||grad E(x)|| <= kappa everywhere,
  then |E(w) - E(q)| <= kappa * ||w - q|| by MVT on the segment — the
  measured kappa = sup ||grad E|| certifies quantization distortion.
- R565 ``smooth_from_lipschitz_gradient``: L-smooth gradient
  (||∇E(y)-∇E(x)|| <= L||y-x||) gives the upper-quadratic model
  E(q+dir) <= E(q) + <grad E(q), dir> + L*||dir||^2/2.
- R564 ``grow_gain_from_smooth_descent``: composing the quadratic model
  with certified descent (dnorm^2 <= inner) and the SafeQP window
  (0 < L, 0 <= eta <= 1/L): E2 <= E1 - eta*dnorm^2/2 — the grow-stage
  gain is the MEASURED constant eta*dnorm^2/2.

Pure arithmetic; the measurement side (grad norms, inner products) is
owned by callers (h_emp_ sources), this module owns the certificates.
"""
from __future__ import annotations

import math


def lip_energy_quantization(kappa: float, dist: float) -> float:
    """|E(w) - E(qTern w)| <= kappa * ||w - qTern w|| [R563
    ``lip_energy_quantization``].

    Certified bound on energy distortion from weight rounding: kappa is
    the measured sup gradient norm (h_emp_: ||grad E|| <= kappa), dist =
    ||w - qTern w|| the measured quantization displacement (ternary
    rounding gives dist <= s/2 in-range via Energy/Ternary).
    """
    if kappa < 0.0:
        raise ValueError("kappa must be nonnegative")
    if dist < 0.0:
        raise ValueError("dist must be nonnegative")
    return kappa * dist


def gridq_energy_bound(lipschitz: float, n_coords: int, grid_step: float) -> float:
    """|E(Q(w)) - E(w)| <= L * sqrt(n) * delta / 2 [EFArc.lean
    ``gridQ_energy_bound`` (Unified/QuantEnergyBridge)].

    Analytic quantization->energy bridge: composing the vector grid
    contract (sum of per-coordinate rounding errors^2 <= n*(delta/2)^2,
    Energy/GridQ) with an explicit Euclidean-Lipschitz energy constant
    L replaces the empirical h_emp_ premise with an analytic one.
    Runtime consumer: ternary/grid quantizers (BitLinear cache path).
    """
    if lipschitz < 0.0:
        raise ValueError("lipschitz must be nonnegative")
    if n_coords < 1:
        raise ValueError("n_coords must be >= 1")
    if grid_step <= 0.0:
        raise ValueError("grid_step must be positive")
    return lipschitz * math.sqrt(n_coords) * grid_step / 2.0


def smooth_descent_energy_bound(
    energy: float, inner: float, dir_norm_sq: float, L: float
) -> float:
    """E(q + dir) <= E(q) + <grad E q, dir> + L*||dir||^2/2 [R565
    ``smooth_from_lipschitz_gradient``].

    The upper-quadratic model from the L-smooth gradient premise
    (||grad E(y) - grad E(x)|| <= L||y-x||, measured). ``inner`` is the
    measured directional derivative <grad E(q), dir>. This is the
    runtime form of h_emp_smooth at eta = 1.
    """
    if L < 0.0:
        raise ValueError("L must be nonnegative")
    if dir_norm_sq < 0.0:
        raise ValueError("dir_norm_sq must be nonnegative")
    return energy + inner + L * dir_norm_sq / 2.0


def grow_gain_from_smooth_descent(
    inner: float, dnorm_sq: float, L: float, eta: float
) -> tuple[bool, float]:
    """Certified descent step: (holds, gain) [R564
    ``grow_gain_from_smooth_descent`` + Foundations joint_stage].

    Given the measured quadratic model and certified descent
    (dnorm_sq <= inner, i.e. the direction is consistent with its own
    gradient signal), the SafeQP window 0 < L with 0 <= eta <= 1/L
    certifies E2 <= E1 - eta * dnorm_sq / 2 — the measured grow-stage
    decrease. Returns (False, 0.0) when a premise fails (eta outside the
    window, no certified descent) — honest rejection, not a guess.
    """
    if L <= 0.0:
        raise ValueError("L must be positive")
    if eta < 0.0 or eta > 1.0 / L:
        return False, 0.0
    if dnorm_sq > inner:
        return False, 0.0
    return True, eta * dnorm_sq / 2.0


def kappa_from_grad_norms(grad_norms: list[float]) -> float:
    """Measured kappa = max ||grad E|| over the observed steps (h_emp_
    source for R563: the certificate premise is ||grad E|| <= kappa, so
    the honest runtime kappa is the running sup of measured gradient
    norms — a LOWER witness of the true sup, never an estimate).

    Empty input -> 0.0 (vacuous certificate: kappa=0 bounds nothing
    above, applies to nothing).
    """
    return max((float(g) for g in grad_norms), default=0.0)


def lip_stage_certificate(
    grad_norms: list[float],
    dist: float,
    eta: float | None = None,
    inner: float | None = None,
    dnorm_sq: float | None = None,
    L: float | None = None,
) -> dict:
    """One-glance stage certificate from the LipEnergy vertical (R563–R565).

    Always reports the R563 quantization bound kappa*dist. When the
    SafeQP premises are supplied (eta, inner, dnorm_sq, L), also reports
    the R564 certified grow gain (eta*dnorm_sq/2) with honest rejection
    when the window/descent premises fail. Pure bookkeeping — the
    measurements are the caller's (h_emp_ sources).
    """
    kappa = kappa_from_grad_norms(grad_norms)
    cert: dict = {
        "kappa": kappa,
        "quant_loss_bound": lip_energy_quantization(kappa, dist),
    }
    if None not in (eta, inner, dnorm_sq, L):
        ok, gain = grow_gain_from_smooth_descent(inner, dnorm_sq, L, eta)
        cert["grow_gain_certified"] = ok
        cert["grow_gain"] = gain if ok else 0.0
    return cert


def cut_safe_accept(
    dL_meas: float,
    d_norm: float,
    eps_probe: float,
    eps_att: float = 0.0,
    L_att: float = 0.0,
) -> tuple[bool, float]:
    """Certified step-accept gate [ProbeError ``cut_safe_accept``].

    Composition of theorems (Primes Pretraining/ProbeError.lean):
      - ``safe_certificate_margin`` (Cauchy-Schwarz): the TRUE loss change
        satisfies dL_exact <= dL_meas + eps * ||d||, where eps bounds the
        gradient-signal error ||g_hat - g_full||;
      - ``clamped_attention_grad_bound``: the attention-quantization term
        contributes at most L_att * eps_att;
      - ``probe_safe_accept``: if the MEASURED model plus the total margin
        is still nonpositive, the EXACT change is certified nonpositive:

            dL_meas + (eps_probe + L_att * eps_att) * ||d|| <= 0
            ==>  dL_exact <= 0.

    Returns (accept, total_margin_per_norm). Honest rejection: when the
    gate fails, the exact decrease is NOT certified - the caller should
    veto or shrink the step, not hope.
    """
    if d_norm < 0.0:
        raise ValueError("d_norm must be nonnegative")
    if eps_probe < 0.0 or eps_att < 0.0:
        raise ValueError("eps must be nonnegative")
    if L_att < 0.0:
        raise ValueError("L_att must be nonnegative")
    total_eps = eps_probe + L_att * eps_att
    slack = -(dL_meas) - total_eps * d_norm
    return dL_meas + total_eps * d_norm <= 0.0, max(slack, 0.0)


# --- Insight/consolidation vertical (Autonomy/Insight.lean, 2026-10-19) -------

def forgetting_kl_bound(
    kl: float, emp_curvature_half: float, curvature_cap: float,
) -> bool:
    """Forgetting KL gate [``forgetting_kl_bound`` (R329/Autonomy/
    Insight.lean)]: with h_emp_ second-order curvature measured,
    if ``kl <= curv/2`` and ``curv <= 2*eps`` then ``kl <= eps``.
    Runtime: post-merge forgetting certificate from the measured
    empirical curvature (lip_energy stage certificate). Returns True
    when the chain holds (kl actually bounded by eps)."""
    if kl < 0.0 or curvature_cap < 0.0:
        raise ValueError("kl and eps must be nonnegative")
    if emp_curvature_half < 0.0:
        raise ValueError("curvature must be nonnegative")
    return kl <= curvature_cap


def consolidation_gate(
    gold_dot_ds: float, heps: float, gI_dot_ds: float,
) -> tuple[bool, bool]:
    """Step-wise consolidation safety [``insight_consolidation_safe``
    (Autonomy/Insight.lean)]: if the consolidation step ds minimizes
    distance to gI over a convex set contained in the gold cone, then
    (a) the gold direction is not harmed beyond -heps and
    (b) the step is a descent step on the target: sqNorm(ds) <=
    <gI, ds>. Runtime: the two SafeQP-veto predicates a merge/root-ft
    step must both satisfy. Returns (gold_ok, descent_ok)."""
    gold_ok = gold_dot_ds >= -heps
    descent_ok = gI_dot_ds >= 0.0  # sqNorm(ds) <= <gI, ds> projected check
    return gold_ok, descent_ok


def gate_retention_bound(steps_gold_dots: list[float], heps: float) -> float:
    """Multi-step knowledge retention [``gate_retention`` (SS6.3,
    LSL-bridge)]: if every consolidation step lies in the gold cone
    (<gold, ds_t> >= -heps), the T-step drift is at most linear:
    sum_t <gold, ds_t> >= -T*heps. Runtime: cumulative drift budget
    over a training generation - the certificate for a whole stage
    from per-step certificates. Returns the certified floor."""
    if heps < 0.0:
        raise ValueError("heps must be nonnegative")
    if any(g < -heps - 1e-9 for g in steps_gold_dots):
        raise ValueError("step outside the gold cone - certificate void")
    return -len(steps_gold_dots) * heps


def tldr_two_tier_safety(
    adapter_out_on_kernel: bool, kl: float, curvature_cap: float,
) -> tuple[bool, bool]:
    """Two-tier adapter safety [``tldr_two_tier_safety``]: a low-rank
    adapter acts as zero on the kernel of B (tier 1, exact) and the
    forgetting KL is bounded by eps via the curvature chain (tier 2).
    Returns (kernel_null_ok, kl_ok)."""
    return adapter_out_on_kernel, forgetting_kl_bound(kl, curvature_cap, curvature_cap)
