"""Tests for the supervisor's §10/§13/§17 gate wiring and distill lane."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "scripts" / "growth"), str(_REPO), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml

import growth_supervisor as gs  # noqa: E402


def _lane(meta: dict | None = None) -> gs.Lane:
    return gs.Lane(
        name="t", expert_configs=[], joint_config="configs/x.yaml",
        meta=meta or {},
    )


def _log_with_ce(tmp_path: Path, ces: list[float]) -> Path:
    log = tmp_path / "joint.log"
    log.write_text(
        "\n".join(f"step | ce={c:.4f}" for c in ces), encoding="utf-8"
    )
    return log


def test_ignition_gate_inputs_missing_logs_and_skips(tmp_path, caplog) -> None:
    lane = _lane({})  # no gamma/k
    log = _log_with_ce(tmp_path, [4.0, 3.9, 3.8])
    with caplog.at_level("INFO", logger="growth"):
        out = gs.ignition_gate_check(lane, log)
    assert out is None
    assert any("inputs_missing" in r.message for r in caplog.records)


def test_ignition_gate_measures_and_advises(tmp_path) -> None:
    # A growing capability trajectory with a plausible frontier dynamics;
    # meta supplies only the cone parameters, the fit must do the rest.
    ces = [4.0, 3.9, 3.8, 3.7, 3.6]
    log = _log_with_ce(tmp_path, ces)
    lane = _lane({"gamma": 0.1, "k": 0.5})
    out = gs.ignition_gate_check(lane, log)
    assert out is not None
    assert out["bifurcation"] in ("grow", "decay", "inject")
    assert "advice" in out
    assert out["C_final"] > 0.0


def test_ignition_gate_meta_overrides_beat_the_fit(tmp_path) -> None:
    ces = [4.0, 3.9, 3.8, 3.7, 3.6]
    log = _log_with_ce(tmp_path, ces)
    lane = _lane({"gamma": 0.1, "k": 0.5, "beta": 0.0, "rho": 0.5})
    out = gs.ignition_gate_check(lane, log)
    assert out is not None
    # beta = 0 and rho < 1: the decay branch (frontier_decay_no_growth)
    assert out["bifurcation"] == "decay"


def test_ignition_gate_saturation_and_window(tmp_path) -> None:
    ces = [4.0, 3.9, 3.8, 3.7, 3.6]
    log = _log_with_ce(tmp_path, ces)
    lane = _lane({
        "gamma": 0.1, "k": 0.5, "beta": 5.0, "rho": 0.9,
        "Cstar": 1.0, "sigma": 0.5, "alpha": 0.1,
    })
    out = gs.ignition_gate_check(lane, log)
    assert out is not None
    assert "lifecycle" in out
    assert "takeoff_window" in out
    assert out["takeoff_window"]["certified_factor_bound"] > 1.0


def test_parse_ce_series_skips_nan_and_inf(tmp_path) -> None:
    log = tmp_path / "l.log"
    log.write_text("a | ce=1.0\nb | ce=nan\nc | ce=2.0\nd | ce=inf\n",
                   encoding="utf-8")
    assert gs.parse_ce_series(log) == [1.0, 2.0]


# --- §17 distill lane ------------------------------------------------------


def test_make_distill_config_derives_from_parent(tmp_path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text(yaml.safe_dump({
        "model": {"vocab_size": 100, "hidden_size": 32},
        "train": {"checkpoint_dir": "checkpoints/gen7_joint",
                  "max_steps": 10},
        "merge": {"distill": False, "distill_teacher": "self"},
        "distill": {"teachers": ["ck/joint/best.pt", "ck/sib/best.pt"]},
    }), encoding="utf-8")
    out_path = tmp_path / "derived.yaml"
    made = gs.make_distill_config(parent, out_path)
    raw = yaml.safe_load(made.read_text(encoding="utf-8"))
    assert raw["train"]["checkpoint_dir"] == "checkpoints/gen7_joint_distill"
    assert raw["train"]["init_from"] == "checkpoints/gen7_joint/best.pt"
    assert raw["merge"]["distill"] is True
    assert raw["merge"]["distill_teacher"] == "ck/joint/best.pt"
    assert raw["merge"]["distill_disagreement_quantile"] == 0.95
    # the advisory top-level section is consumed, not copied (load_config
    # would reject an unknown key)
    assert "distill" not in raw


def test_make_distill_config_without_teachers_keeps_kd_off(tmp_path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text(yaml.safe_dump({
        "train": {"checkpoint_dir": "checkpoints/j"},
    }), encoding="utf-8")
    made = gs.make_distill_config(parent, tmp_path / "d.yaml")
    raw = yaml.safe_load(made.read_text(encoding="utf-8"))
    assert not raw["merge"].get("distill")
    assert not raw["merge"].get("distill_disagreement_quantile")


def _report(mean: float) -> dict:
    return {"domains": {"a": {"exact_ce": mean}, "b": {"exact_ce": mean}}}


def test_distill_leak_from_evals_gates_on_measured_c_and_delta() -> None:
    lane = _lane()
    # teacher 3.0, student 3.2 -> delta = 0.2; incumbent 3.5 -> c = 0.5
    out = gs.distill_leak_from_evals(
        lane, _report(3.2), _report(3.0), _report(3.5))
    assert out is not None
    assert out["c_k"] == 0.5
    assert out["delta_k"] == 0.2
    assert out["continue_recursion"] is True

    # leak: student worse than teacher by more than the cycle gain
    out2 = gs.distill_leak_from_evals(
        lane, _report(3.8), _report(3.0), _report(3.5))
    assert out2["continue_recursion"] is False
    assert "STOP" in out2["note"]


def test_distill_leak_from_evals_missing_inputs_returns_none() -> None:
    lane = _lane()
    assert gs.distill_leak_from_evals(lane, None, _report(3.0), None) is None
    assert gs.distill_leak_from_evals(lane, _report(3.0), None, None) is None


def test_load_plan_accepts_distill_lane_type(tmp_path) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text(yaml.safe_dump([
        {"name": "d0", "type": "distill", "joint": "configs/j.yaml",
         "teacher_report": "reports/g.json"},
        {"name": "g0", "experts": ["configs/a.yaml"],
         "joint": "configs/j.yaml"},
    ]), encoding="utf-8")
    lanes = gs.load_plan(plan)
    assert lanes[0].lane_type == "distill"
    assert lanes[0].expert_configs == []
    assert lanes[0].meta["teacher_report"] == "reports/g.json"
    assert lanes[1].lane_type == "growth"
