"""Tests: R388 thermostat gate integration in the growth cycle
selector (scripts/growth/growth_cycle.py::_thermostat_gate).

The gate is opt-in (CLI flags, default off); these tests pin the
contract without touching the GPU line.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "growth_cycle", ROOT / "scripts" / "growth" / "growth_cycle.py")
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)


def test_exploit_beyond_horizon():
    # gen74-like delta; cooling from gen 100: at gen 200 the law
    # says exploration is exhausted (competitor mass -> 0 << theta)
    rec = gc._thermostat_gate(200, 0.0354, 1.0, 0.9, 0.05, 100)
    assert rec["active"] and rec["phase"] == "exploit"
    assert rec["competitor_mass"] <= rec["theta"]
    assert rec["t"] == 100


def test_explore_early():
    rec = gc._thermostat_gate(101, 0.0354, 1.0, 0.9, 0.05, 100)
    assert rec["phase"] == "explore"
    assert rec["competitor_mass"] > rec["theta"]


def test_inactive_when_delta_unmeasured():
    rec = gc._thermostat_gate(101, None, 1.0, 0.9, 0.05, 100)
    assert rec == {"active": False,
                   "reason": "delta unmeasured (no accepted generation yet)"}


def test_inactive_on_nonpositive_delta():
    # a rejected generation (delta <= 0) is not an energy gap
    rec = gc._thermostat_gate(101, 0.0, 1.0, 0.9, 0.05, 100)
    assert rec["active"] is False


def test_selector_meta_carries_thermostat():
    # _select_next_action with thermostat_cfg logs the phase record
    meta = gc._select_next_action(10**6, ["math"], (1.0, 0.9, 0.05, 100))
    assert "thermostat" in meta  # opt-in path populated
    meta_off = gc._select_next_action(10**6, ["math"])
    assert "thermostat" not in meta_off  # default off, untouched
