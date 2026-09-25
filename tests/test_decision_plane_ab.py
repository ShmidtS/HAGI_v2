"""Tests for the bounded synthetic DecisionPlane experiment runner."""

from __future__ import annotations

import math

import pytest
import torch

from hagi.model.model import HAGI
from scripts.decision_plane_ab import (
    LANE_SPECS,
    _common_state,
    decision_metrics,
    load_common_state,
    majority_logits,
    make_config,
    run,
    run_experiment,
    run_seed,
    synthetic_partition,
    validate_fixtures,
)


def test_decision_metrics_match_closed_form():
    logits = torch.tensor(
        [[4.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 0, 2])
    metrics = decision_metrics(logits, targets, confidence_bins=10)

    probs = logits.detach().softmax(dim=-1)
    expected_nll = -torch.log(probs.gather(1, targets[:, None]).squeeze(1)).mean()
    assert math.isclose(metrics["nll"], float(expected_nll), abs_tol=1e-7)
    assert metrics["accuracy"] == pytest.approx(2 / 3, abs=1e-7)
    # predictions [0,1,2] vs targets [0,0,2]: F1=(2/3,0,1,0),
    # therefore macro-F1=5/12 including the absent class.
    assert metrics["macro_f1"] == pytest.approx(5 / 12, abs=1e-7)
    assert len(metrics["bins"]) == 10
    assert sum(int(item["count"]) for item in metrics["bins"]) == 3
    assert 0.0 <= metrics["brier"] <= 2.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_decision_metrics_rejects_invalid_shapes_ranges_and_empty_inputs():
    good = torch.zeros(2, 4)
    targets = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="shapes"):
        decision_metrics(good, targets[:1], confidence_bins=10)
    with pytest.raises(ValueError, match="N >= 1"):
        decision_metrics(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), confidence_bins=10)
    with pytest.raises(ValueError, match="K >= 2"):
        decision_metrics(torch.zeros(2, 1), targets, confidence_bins=10)
    with pytest.raises(ValueError, match="confidence_bins"):
        decision_metrics(good, targets, confidence_bins=0)
    with pytest.raises(ValueError, match="option range"):
        decision_metrics(good, torch.tensor([0, 4]), confidence_bins=10)


def test_majority_baseline_uses_training_histogram_for_holdout_rows():
    logits = majority_logits(torch.tensor([0, 0, 1, 2]), 4, rows=7)
    assert logits.shape == (7, 4)
    assert logits.argmax(dim=-1).tolist() == [0] * 7


def test_synthetic_fixture_is_disjoint_nonuniform_and_not_last_token_copy():
    train_ids, train_labels = synthetic_partition(
        seed=11,
        rows=32,
        vocab_size=64,
        seq_len=8,
        num_options=4,
        domain=0,
    )
    holdout_ids, holdout_labels = synthetic_partition(
        seed=12,
        rows=20,
        vocab_size=64,
        seq_len=8,
        num_options=4,
        domain=1,
    )
    assert validate_fixtures(train_ids, train_labels, holdout_ids, holdout_labels, num_options=4) == 0
    assert torch.equal(train_labels, train_ids[:, 0].remainder(4))
    assert not torch.equal(train_labels, train_ids[:, -1].remainder(4))
    assert len(torch.unique(train_labels)) == 4
    assert len(torch.unique(holdout_labels)) == 4
    train_rows = {tuple(row) for row in train_ids.tolist()}
    assert all(tuple(row) not in train_rows for row in holdout_ids.tolist())


def test_fixture_rejects_overlap_and_invalid_labels():
    ids = torch.tensor([[1, 2], [3, 4]])
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="overlap"):
        validate_fixtures(ids, labels, ids, labels, num_options=2)
    with pytest.raises(ValueError, match="invalid option labels"):
        validate_fixtures(ids, labels, ids, torch.tensor([0, 2]), num_options=2)


def test_common_state_omits_fresh_decision_head():
    model = HAGI(make_config(seed=3, steps=1, num_options=4, freeze_base=False))
    with torch.no_grad():
        assert model.decision_head is not None
        model.decision_head.weight.fill_(3.0)
    state = _common_state(model)
    assert not any(name.startswith("decision_head.") for name in state)

    torch.manual_seed(99)
    other = HAGI(make_config(seed=4, steps=1, num_options=4, freeze_base=False))
    assert other.decision_head is not None
    with torch.no_grad():
        assert torch.count_nonzero(other.decision_head.weight) == 0
    load_common_state(other, state)
    assert torch.equal(other.encoder.embedding.weight, model.encoder.embedding.weight)
    assert torch.count_nonzero(other.decision_head.weight) == 0


def test_common_state_loader_rejects_missing_or_shaped_drift():
    model = HAGI(make_config(seed=3, steps=1, num_options=4, freeze_base=False))
    state = _common_state(model)
    with pytest.raises(ValueError, match="missing"):
        load_common_state(model, {name: value for name, value in state.items() if name != "encoder.embedding.weight"})
    malformed = dict(state)
    malformed["encoder.embedding.weight"] = torch.zeros(
        model.cfg.model.vocab_size,
        model.cfg.model.hidden_size + 1,
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        load_common_state(model, malformed)
    with pytest.raises(ValueError, match="lane state mismatch"):
        load_common_state(
            model,
            {**state, "decision_head.weight": torch.zeros(4, model.cfg.model.hidden_size)},
        )


def test_smoke_reports_exact_budget_and_fail_closed_quality():
    report = run_seed(
        seed=17,
        steps=1,
        num_options=4,
        device="cpu",
    )

    assert tuple(report["lane_specs"]) == LANE_SPECS
    assert [lane["name"] for lane in report["lanes"]] == list(LANE_SPECS)
    assert report["dropped_rows"] == 0
    assert report["synthetic"] is True
    assert report["quality_supported"] is False
    assert report["mechanism_supported"] is True
    assert report["runtime_checks"] == {
        "finite_metrics": True,
        "all_updates_applied": True,
        "common_initial_decision_metrics_exact": True,
        "common_initial_lm_ce_exact": True,
    }
    for lane in report["lanes"][1:]:
        assert lane["steps_completed"] == 1
        assert lane["updates_applied"] == 1
        assert lane["rejected_updates"] == 0
        assert lane["decision_head_max_abs_update"] > 0.0
    initial = report["lanes"][1]["initial"]
    assert all(lane["initial"] == initial for lane in report["lanes"][1:])


def test_multi_seed_report_is_fail_closed_and_pre_registered():
    from scripts.decision_plane_ab import parse_args

    report = run_experiment(
        parse_args(["--steps", "1", "--num-seeds", "1", "--device", "cpu"])
    )
    assert report["execution_completed"] is True
    assert report["quality_supported"] is False
    assert report["promotion_status"] == "research-only"
    assert report["pre_registered_gates"] == {
        "minimum_seeds": 3,
        "required_strict_majority_wins": 1,
        "max_ece": 0.15,
        "max_accuracy_regression": 0.0,
    }
    assert report["overall_checks"]["at_least_three_seeds"] is False


def test_run_rejects_nonpositive_steps():
    with pytest.raises(ValueError, match="steps must be >= 1"):
        run(seed=1, steps=0)
