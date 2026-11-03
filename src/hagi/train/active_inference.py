"""Active-Inference vertical — runtime ports (§17).

Lean SSOT: primes ``Hagi/Probability/GibbsConcentration.lean``
(AIF-вертикаль R462–R467, arXiv:2607.13612):

- ``proxy_cert_preserved`` / ``proxy_cert_lost``: a lower-bound entropy
  proxy preserves the surprise certificate by construction; a strictly
  upper proxy strictly breaks it;
- ``efe_ceiling``: G = P − I ≤ P (free-information ceiling);
- ``gap_decomposition`` / ``gapII_nonpos`` / ``gap_graceful``: surrogate
  error decomposes into (miscalibration, bias, slack); bias <= 0 for
  lower-bound proxies; degradation is linear in the slack bounds;
- ``compounding_bound`` / ``compounding_sq``: per-step errors compound
  as L^H (and squared-metric L^(2H)) over a horizon H; an MPC horizon-
  one controller resets the chain;
- ``amortized_policy_bound``: a surrogate argmin within uniform slack c
  executes within 2c of the true optimum.
"""
from __future__ import annotations


def proxy_cert_preserved(surprise: float, entropy: float, entropy_hat: float) -> float:
    """The certified surprise margin SURVIVES a lower-bound entropy proxy
    [``proxy_cert_preserved``]: the certificate s <= E − h implies
    s <= E − ĥ whenever ĥ <= h. Returns the surrogate margin E − ĥ
    (>= s, the certificate keeps holding with room to spare).

    Args:
        surprise: the certified surprise level s.
        entropy: true entropy h.
        entropy_hat: proxy ĥ (must satisfy ĥ <= h for preservation).
    """
    if entropy_hat > entropy:
        raise ValueError("proxy must lower-bound the entropy (ĥ <= h)")
    return surprise + entropy - entropy_hat


def proxy_cert_lost_bound(surprise: float, entropy: float, entropy_hat: float) -> float:
    """Surrogate free-energy shortfall of an upper proxy [``proxy_cert_lost``]:
    with ĥ > h, the certified surprise s = E − h gives surrogate slack
    E − ĥ = s − (ĥ − h) < s. Returns the true margin E − ĥ (strictly < s)."""
    if not entropy_hat > entropy:
        raise ValueError("this bound applies to a strictly upper proxy (ĥ > h)")
    return surprise - (entropy_hat - entropy)


def efe_ceiling(pref_gap: float, info_gain: float) -> float:
    """G = P − I ≤ P [``efe_ceiling``]: returns the EFE bound given the
    preference term P and the (nonnegative) epistemic term I."""
    if info_gain < 0:
        raise ValueError("info_gain must be nonnegative (mutual information)")
    return pref_gap - info_gain


def gap_graceful(g1: float, g2: float, g3: float, c1: float, c2: float, c3: float) -> float:
    """|F̂ − F| <= c1 + c2 + c3 given |g_i| <= c_i [``gap_graceful``].
    Returns the certified total-gap bound."""
    if max(abs(g1) - c1, abs(g2) - c2, abs(g3) - c3) > 1e-12:
        raise ValueError("gap components must respect their slack bounds")
    return c1 + c2 + c3


def compounding_bound(lipschitz: float, e0: float, horizon: int) -> float:
    """e_H <= L^H * e_0 given e_{t+1} <= L*e_t [``compounding_bound``]."""
    if lipschitz < 0:
        raise ValueError("L must be nonnegative")
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    return (lipschitz**horizon) * e0


def compounding_sq(lipschitz: float, e0: float, horizon: int) -> float:
    """e_H^2 <= (L^2)^H * e_0^2 given e_{t+1}^2 <= L^2 e_t^2, e_t >= 0
    [``compounding_sq``]."""
    return compounding_bound(lipschitz * lipschitz, e0 * e0, horizon)


def mpc_horizon_one_resets(e_current: float) -> float:
    """Horizon-one MPC executes immediately: the compounding chain starts
    from the CURRENT error, not the accumulated product (the H = 1 case of
    compounding_bound — e_1 <= L*e_0 only)."""
    return e_current


def amortized_policy_slack(c: float) -> float:
    """Amortized policy executes within 2c of the optimum
    [``amortized_policy_bound``]."""
    if c < 0:
        raise ValueError("slack c must be nonnegative")
    return 2.0 * c
