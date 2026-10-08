"""R176 BitAlloc — runtime port.

* bit allocation (``transfer_exact`` / ``imbalance_yields_gain`` /
  ``stable_factor_two`` / ``two_layer_equalize`` /
  ``distill_quant_composite``): bits want to live where the error is
  large; greedy transfer stabilizes factor-2 balanced
"""
from __future__ import annotations

import torch

from hagi.train.bit_alloc import (
    apply_transfer,
    distill_quant_composite,
    greedy_transfer,
    layer_error,
    total_error,
    transfer_delta,
    two_layer_equalize,
)


class TestBitAlloc:
    def test_layer_error_halves(self):
        # layerError_antitone: one more bit halves the error
        assert abs(layer_error(3.0, 4) - 3.0 / 16) < 1e-12
        assert abs(layer_error(3.0, 5) - layer_error(3.0, 4) / 2) < 1e-12

    def test_transfer_identity_exact(self):
        # transfer_exact: moving a bit k -> j changes the total by
        # EXACTLY e_k - e_j/2
        c = [1.0, 4.0, 2.0]
        f = [4, 2, 3]
        j, k = 1, 0
        before = total_error(c, f)
        after = total_error(c, apply_transfer(f, j, k))
        assert abs((after - before) - transfer_delta(c, f, j, k)) < 1e-12

    def test_transfer_nonincreasing_iff_receiver_2x(self):
        # the exchange improves exactly when e_k <= e_j/2
        c = [1.0, 5.0]
        f = [4, 2]
        # e_0 = 1/16 = 0.0625, e_1 = 5/4 = 1.25: 2*e_0 = 0.125 < 1.25 -> improving
        assert transfer_delta(c, f, j=1, k=0) < 0
        # reverse: e_1/2 = 0.625 > e_0 -> worsens
        assert transfer_delta(c, f, j=0, k=1) > 0

    def test_greedy_reaches_factor2_balance(self):
        # imbalance_yields_gain + stable_factor_two: greedy transfer
        # stabilizes with every error within a factor 2
        c = [1.0, 4.0, 16.0]
        f = [8, 2, 1]
        g, rounds = greedy_transfer(c, f)
        assert sum(g) == sum(f)  # budget preserved
        errs = [layer_error(ci, bi) for ci, bi in zip(c, g)]
        for ej in errs:
            for ek in errs:
                if ek > 0:
                    assert ej <= 2.0 * ek + 1e-12

    def test_greedy_improves_total(self):
        c = [1.0, 8.0, 2.0]
        f = [7, 1, 2]
        g, _ = greedy_transfer(c, f)
        assert total_error(c, g) <= total_error(c, f) + 1e-12
        assert total_error(c, g) < total_error(c, f)

    def test_two_layer_balancing_dominates(self):
        # two_layer_equalize: e(x) + e(x+2d) >= 2*e(x+d)
        for x, d in ((2, 0), (2, 3), (0, 5), (1, 1)):
            gap = two_layer_equalize(4.0, x, d)
            assert gap >= -1e-12

    def test_composite_certificate(self):
        # distill_quant_composite: S_n + n*g <= E_0 + delta_n + totalError
        s_n, e0, delta_n, n, g = 1.0, 2.0, 0.1, 3, 0.2
        c, f = [1.0, 2.0], [4, 4]
        bound, slack = distill_quant_composite(s_n, e0, delta_n, n, g, c, f)
        assert slack >= 0  # certificate holds on this instance
        assert bound == e0 + delta_n + total_error(c, f)

    def test_domain_errors(self):
        for bad in (
            lambda: layer_error(1.0, -1),
            lambda: transfer_delta([1.0], [1], 0, 0),
            lambda: transfer_delta([1.0, 1.0], [1, 0], 0, 1),  # giver empty
            lambda: total_error([1.0], [1, 2]),
        ):
            try:
                bad()
                raised = False
            except ValueError:
                raised = True
            assert raised

