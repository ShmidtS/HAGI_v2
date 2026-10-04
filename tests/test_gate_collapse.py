"""T5 live monitor tests: gate_collapse_report (Pinsker floor).

The report's job: turn the three gate-window measurements (CE,
H_model, H_data) into the honest R108/T5 collapse alarm. On fixed
windows KL(data||model) = CE - H_data is an identity, the floor is
``H_data - pinsker_floor_correction(KL, V)`` at nu=1, and margin =
H_model - floor with ``collapsed = margin < 0``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest  # noqa: E402

from hagi.train.distill_recursion import gate_collapse_report  # noqa: E402


class TestGateCollapseReport:
    def test_perfectly_fitted_window_kl_zero(self):
        # CE == H_data: the model matches the window distribution
        # exactly; KL = 0, correction = 0, floor = H_data.
        r = gate_collapse_report(ce=8.0, h_model=8.0, h_data=8.0, v=32768)
        assert r["kl_model_data"] == pytest.approx(0.0)
        assert r["floor"] == pytest.approx(8.0)
        assert r["margin"] == pytest.approx(0.0)
        assert r["collapsed"] is False  # margin == 0 is not below

    def test_confident_model_below_floor_is_alarm(self):
        # CE == H_data (no KL slack) but the model is CONFIDENT: its
        # output entropy 5.5 sits below the floor 8.0. This is the
        # honest reading: a low-entropy output on a high-entropy
        # window is exactly what a collapsing model looks like.
        r = gate_collapse_report(ce=8.0, h_model=5.5, h_data=8.0, v=32768)
        assert r["floor"] == pytest.approx(8.0)
        assert r["margin"] == pytest.approx(5.5 - 8.0)
        assert r["collapsed"] is True

    def test_calibrated_spread_model_no_alarm(self):
        # CE above H_data by 0.2 nats: floor = 8 - corr(0.2, 32768)
        # ≈ 8 - (sqrt(0.1)*ln(32767) + h2(0.316)) ≈ 8 - 2.75 ≈ 5.25;
        # h_model 7.8 is above it.
        r = gate_collapse_report(ce=8.2, h_model=7.8, h_data=8.0, v=32768)
        assert r["kl_model_data"] == pytest.approx(0.2)
        assert r["floor"] < 8.0
        assert r["margin"] > 0.0
        assert r["collapsed"] is False

    def test_deep_collapse_alarm(self):
        # KL = 0.5: floor = 8 - (sqrt(0.25)*ln(32767) + h2(0.354))
        # ≈ 8 - (0.354*10.397 + 0.933) ≈ 8 - 4.61 ≈ 3.39; h_model 2.0
        # is far below it.
        r = gate_collapse_report(ce=8.5, h_model=2.0, h_data=8.0, v=32768)
        assert r["collapsed"] is True
        assert r["margin"] < 0.0
        assert r["floor"] < 4.7  # the correction moved the floor well down

    def test_ce_below_h_data_raises(self):
        # CE cannot be below the window's own entropy; the measurement
        # itself is broken (e.g. a wrong window or a sliced head).
        with pytest.raises(ValueError):
            gate_collapse_report(ce=7.9, h_model=5.0, h_data=8.0, v=32768)

    def test_report_keys_complete(self):
        r = gate_collapse_report(ce=8.2, h_model=7.8, h_data=8.0, v=32768)
        assert set(r) == {"kl_model_data", "floor", "margin", "collapsed"}
        assert math.isfinite(r["floor"])
        assert isinstance(r["collapsed"], bool)
