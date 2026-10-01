"""Analytic batch size from the AM-GM step law.

``Hagi/Audit/Exactness.lean`` (``amgm_equality``, ``amgm_uniqueness``,
``batch_T_min``). The per-step time of a training step with batch size
B splits into

    T(B) = (1 + B_n / B) * (t0 + c * B)

    - the first factor is the gradient-accumulation bubble: with a
      fixed token budget ``B_n`` spread over B-sized batches there are
      ``B_n / B`` micro-steps, and the fixed per-step overhead
      ``t0`` (kernel launches, the one-time costs the ROCm build
      amortizes badly) is paid once per micro-step;
    - the second is the compute term: a fixed ``t0`` per step plus
      ``c * B`` of actual matmul work.

Expanding and applying AM-GM to ``c*B + B_n*t0/B``:

    T(B) >= t0 + c*B_n + 2*sqrt(c * B_n * t0)

with equality iff ``B^2 = B_n*t0/c``. ``amgm_equality`` proves the
bound is tight at that point and ``amgm_uniqueness`` proves the
minimizer is UNIQUE there. So the optimal batch is a closed form:

    B* = sqrt(B_n * t0 / c)

not a swept hyperparameter. Note the consequence: ``B*`` GROWS as
``sqrt(t0)`` -- the worse the per-step overhead, the LARGER the batch
should be, because a bigger batch amortizes the fixed cost over more
tokens. On a bandwidth-bound accelerator with high launch overhead
this is not a marginal effect.
"""

from __future__ import annotations

import math


def optimal_batch(token_budget: int, overhead: float, per_sample: float) -> float:
    """``B* = sqrt(B_n * t0 / c)`` (``amgm_equality`` + ``amgm_uniqueness``).

    Args:
        token_budget: ``B_n`` -- tokens consumed per optimizer step.
        overhead: ``t0`` -- the fixed per-micro-step cost (seconds).
        per_sample: ``c`` -- the marginal compute per sample (seconds).

    Returns:
        The unique minimizing batch size.

    Raises:
        ValueError: when any argument is non-positive.
    """
    if token_budget <= 0 or overhead <= 0 or per_sample <= 0:
        raise ValueError("optimal_batch needs positive token_budget, overhead, per_sample")
    return math.sqrt(token_budget * overhead / per_sample)


def step_time(
    batch: float, token_budget: float, overhead: float, per_sample: float
) -> float:
    """``T(B) = (1 + B_n/B) * (t0 + c*B)`` -- the measured step time."""
    return (1.0 + token_budget / batch) * (overhead + per_sample * batch)


def time_lower_bound(
    token_budget: float, overhead: float, per_sample: float
) -> float:
    """``t0 + c*B_n + 2*sqrt(c*B_n*t0)`` -- the AM-GM floor (``batch_T_min``)."""
    return (
        overhead
        + per_sample * token_budget
        + 2.0 * math.sqrt(per_sample * token_budget * overhead)
    )


def speedup_vs_batch(
    batch: float,
    token_budget: float,
    overhead: float,
    per_sample: float,
) -> float:
    """Speedup factor of the optimal batch over ``batch``, at least 1.

    Returns ``T(batch) / T(B*)``. At ``batch = B*`` the ratio is exactly
    1.0. Because the ratio is a fraction of the SLOWER time, small
    batches score far below 1 (a batch of 1 can be thousands of times
    slower) -- read it as "the optimal batch is N times faster", so
    multiply by ``batch``-side time or invert if that reads better.
    """
    optimal = optimal_batch(token_budget, overhead, per_sample)
    return step_time(batch, token_budget, overhead, per_sample) / step_time(
        optimal, token_budget, overhead, per_sample
    )
