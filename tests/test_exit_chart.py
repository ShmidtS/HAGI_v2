"""Tests for EXITChartHalt."""

from __future__ import annotations

import pytest

from hagi.model.exit_chart import EXITChartHalt


class TestEXITChartHalt:
    def test_init_not_halted(self):
        h = EXITChartHalt(0.05, 10, 3)
        assert not h.halted and h.n_observations == 0

    def test_below_min_steps(self):
        h = EXITChartHalt(0.05, 10, 3)
        for _ in range(9):
            assert not h.observe(0.01)
        assert not h.halted

    def test_halted_after_low(self):
        h = EXITChartHalt(0.05, 10, 2)
        for _ in range(11):
            h.observe(0.01)
        assert h.halted

    def test_halted_sticky(self):
        h = EXITChartHalt(0.05, 5, 2)
        for _ in range(7):
            h.observe(0.01)
        assert h.halted
        assert h.observe(1.0)

    def test_reset(self):
        h = EXITChartHalt(0.05, 5, 2)
        for _ in range(7):
            h.observe(0.01)
        h.reset()
        assert not h.halted and h.n_observations == 0

    def test_high_novelty_no_halt(self):
        h = EXITChartHalt(0.05, 10, 3)
        for _ in range(20):
            h.observe(1.0)
        assert not h.halted

    def test_mixed_no_halt(self):
        h = EXITChartHalt(0.05, 10, 3)
        for i in range(20):
            h.observe(0.01 if i % 2 == 0 else 1.0)
        assert not h.halted

    def test_rejects_zero_threshold(self):
        with pytest.raises(ValueError):
            EXITChartHalt(0.)

    def test_rejects_zero_min_steps(self):
        with pytest.raises(ValueError):
            EXITChartHalt(min_steps=0)

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError):
            EXITChartHalt(window=0)
