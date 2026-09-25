"""Tests for the bounded tied-vs-untied embedding/head runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hagi.model.model import HAGI
from scripts.embedding_tie_ab import (
    LANE_SPECS,
    MAX_MEMORY_RATIO,
    PROTECTED_DOMAINS,
    lane_state,
    load_exact_state,
    make_config,
    model_cpu_state,
    parse_args,
    real_corpus_batches,
    run_experiment,
    run_seed,
    synthetic_batches,
    unigram_ce,
)


def test_config_and_lane_specs_are_minimal():
    tied = make_config(seed=3, steps=2, tie_lm_head=True)
    untied = make_config(seed=3, steps=2, tie_lm_head=False)
    assert tied.model.embedding.tie_lm_head is True
    assert untied.model.embedding.tie_lm_head is False
    assert tied.model.decision.enabled is False
    assert tied.model.cortex.enabled is False
    assert tied.model.adapters.enabled is False
    assert LANE_SPECS == (("tied", True), ("untied", False))


def test_lane_state_copies_codebook_to_untied_projection():
    base = model_cpu_state(HAGI(make_config(seed=5, steps=1, tie_lm_head=True)))
    assert "head.projection.weight" not in base

    untied_cfg = make_config(seed=5, steps=1, tie_lm_head=False)
    state = lane_state(untied_cfg, base, seed=5)
    assert "head.projection.weight" in state
    assert torch.equal(state["head.projection.weight"], base["encoder.embedding.weight"])


def test_exact_state_loader_rejects_key_and_shape_drift():
    model = HAGI(make_config(seed=6, steps=1, tie_lm_head=True))
    state = model_cpu_state(model)
    with pytest.raises(ValueError, match="missing"):
        load_exact_state(model, {key: value for key, value in state.items() if key != "encoder.embedding.weight"})
    malformed = dict(state)
    malformed["encoder.embedding.weight"] = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_exact_state(model, malformed)


def test_synthetic_batches_are_fixed_and_shareable():
    cfg = make_config(seed=7, steps=3, tie_lm_head=True)
    first, holdouts = synthetic_batches(
        cfg,
        steps=3,
        seed=8,
        device=torch.device("cpu"),
    )
    second, second_holdouts = synthetic_batches(
        cfg,
        steps=3,
        seed=8,
        device=torch.device("cpu"),
    )
    assert len(first) == 3
    assert set(holdouts) == {"synthetic"}
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left["input_ids"], right["input_ids"])
        assert torch.equal(left["targets"], right["targets"])
    for key in ("input_ids", "targets"):
        assert torch.equal(holdouts["synthetic"][key], second_holdouts["synthetic"][key])


def test_unigram_reference_matches_closed_form():
    targets = torch.tensor([0, 1, 1, 2])
    value = unigram_ce(
        {"targets": targets.reshape(1, 4)},
        vocab_size=3,
    )
    probabilities = (torch.bincount(targets, minlength=3).double() + 1.0) / 7.0
    expected = float(-probabilities[targets].log().mean())
    assert value == pytest.approx(expected)


def _write_packed_corpus(root: Path, *, tokens: int = 256, vocab_size: int = 32) -> None:
    for index, name in enumerate(PROTECTED_DOMAINS):
        values = (np.arange(tokens, dtype=np.uint32) + index) % vocab_size
        (root / f"{name}.bin").write_bytes(values.tobytes())
    (root / "mix.json").write_text(
        json.dumps({"sources": [{"name": name, "ratio": 1} for name in PROTECTED_DOMAINS]}),
        encoding="utf-8",
    )


def test_real_corpus_loader_uses_fixed_disjoint_protected_batches(tmp_path):
    _write_packed_corpus(tmp_path)
    cfg = make_config(seed=9, steps=2, tie_lm_head=True, vocab_size=32)
    cfg.train.data.seq_len = 8
    train, holdouts, metadata = real_corpus_batches(
        cfg,
        data_dir=str(tmp_path),
        steps=2,
        seed=10,
        start_offset=16,
        holdout_offset=128,
        device=torch.device("cpu"),
    )
    assert len(train) == 2
    assert set(holdouts) == set(PROTECTED_DOMAINS)
    train_windows = {tuple(row) for batch in train for row in batch["input_ids"].tolist()}
    for domain, batch in holdouts.items():
        holdout_windows = {tuple(row) for row in batch["input_ids"].tolist()}
        assert train_windows.isdisjoint(holdout_windows), domain
    for batch in [*train, *holdouts.values()]:
        assert tuple(batch["input_ids"].shape) == (2, 8)
        assert tuple(batch["targets"].shape) == (2, 8)
        assert int(batch["input_ids"].max()) < 32
        assert batch["doc_ids"].shape == batch["input_ids"].shape
    assert metadata["real_packed_stream"] is True
    assert metadata["manifest_validated"] is False
    train_end_by_source = metadata["train_end_by_source"]
    assert all(value >= 16 for value in train_end_by_source.values())
    assert any(value > 16 for value in train_end_by_source.values())
    assert all(value < 128 for value in train_end_by_source.values())


def test_real_corpus_loader_rejects_overlapping_holdout(tmp_path):
    _write_packed_corpus(tmp_path)
    cfg = make_config(seed=10, steps=1, tie_lm_head=True, vocab_size=32)
    cfg.train.data.seq_len = 8
    with pytest.raises(ValueError, match="must be greater"):
        real_corpus_batches(
            cfg,
            data_dir=str(tmp_path),
            steps=1,
            seed=11,
            start_offset=16,
            holdout_offset=16,
            device=torch.device("cpu"),
        )


def test_run_seed_preserves_exact_step_zero_observables():
    report = run_seed(
        seed=12,
        steps=1,
        device=torch.device("cpu"),
        timing_warmup=0,
        real_corpus=False,
        data_dir="data",
        start_offset=0,
        holdout_offset=1,
    )
    assert report["execution_completed"] is True
    assert report["per_seed_gates"]["initial_traces_exact"] is True
    assert report["comparisons"]["untied_over_tied_training_state_bytes"] > 1.0
    for lane in report["lanes"]:
        assert lane["steps_completed"] == 1
        assert lane["updates_applied"] == 1
        assert lane["rejected_updates"] == 0
        assert lane["initial_projection_exact_copy"] is True


def test_experiment_keeps_quality_and_promotion_fail_closed(tmp_path):
    args = parse_args(
        [
            "--steps",
            "1",
            "--num-seeds",
            "1",
            "--timing-warmup",
            "0",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    report = run_experiment(args)
    assert report["execution_completed"] is True
    assert report["quality_supported"] is False
    assert report["promotion_status"] == "research-only"
    assert report["promotion_eligible"] is False
    assert report["overall_gate_checks"]["at_least_three_seeds"] is False
    assert report["pre_registered_gates"]["max_training_state_memory_ratio"] == MAX_MEMORY_RATIO


@pytest.mark.parametrize(
    "argv,match",
    [
        (["--steps", "0"], "positive"),
        (["--num-seeds", "0"], "positive"),
        (["--steps", "1", "--timing-warmup", "1"], "timing-warmup"),
    ],
)
def test_argument_validation(argv, match):
    with pytest.raises(ValueError, match=match):
        parse_args(argv)
