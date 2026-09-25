"""CLI contract for the bounded self-improvement entry point.

These tests pin the *persistence policy* of ``scripts/self_improve.py``, not
model quality. An accepted update is an operational result, never quality
evidence: the loop scores its own generated trajectory, and that window is
discarded before the verdict, so a rejected or fully rolled-back run must not
leave a checkpoint behind for a later run to pick up as a parent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.self_improve as self_improve_cli
from hagi.train.self_improve import SelfImproveResult, SelfImproveStats
from tests.conftest import tiny_config

_SCRIPT_DIR = str(Path(__file__).resolve().parent.parent / "scripts")


@pytest.fixture
def run_cli(monkeypatch, tmp_path: Path):
    """Run ``main()`` with a stubbed loop so no training happens in tests."""
    import scripts.self_improve as module

    real_self_improve = module.self_improve

    def _restore() -> None:
        module.self_improve = real_self_improve

    def _run(accepted_updates: int, *, argv: list[str] | None = None):
        captured: dict[str, object] = {}

        def fake_self_improve(**kwargs):
            applied = bool(accepted_updates)
            result = SelfImproveResult(
                iteration=0,
                generated_ids=[1, 2],
                pre_ce=9.2,
                post_ce=9.1 if applied else 9.2,
                kl_div=0.001 if applied else 0.0,
                update_applied=applied,
            )
            stats = SelfImproveStats(
                iterations=[result] * accepted_updates if applied else [result],
                stopped="max_iterations" if applied else "kl_bound",
                best_ce=result.post_ce if applied else None,
                accepted_updates=accepted_updates,
            )
            captured["stats"] = stats
            return stats

        monkeypatch.setattr(module, "self_improve", fake_self_improve)
        monkeypatch.setattr(module, "load_config", lambda _path: tiny_config())
        monkeypatch.setattr(
            module.sys, "argv", ["self_improve.py", *(argv if argv is not None else _default_argv(tmp_path))]
        )
        code = module.main()
        return code, captured["stats"]

    yield _run
    _restore()


def _default_argv(tmp_path: Path) -> list[str]:
    return [
        "--config",
        str(Path(__file__).resolve().parent.parent / "configs" / "level0_ab" / "ru_baseline.yaml"),
        "--prompt-ids",
        "1",
        "2",
        "3",
        "--n-new-tokens",
        "4",
        "--max-iterations",
        "1",
        "--device",
        "cpu",
        "--checkpoint-dir",
        str(tmp_path / "ckpt"),
    ]


def test_rejected_run_writes_no_checkpoint(run_cli, tmp_path: Path) -> None:
    """No accepted update -> no ``step-*.pt`` and an explicit null checkpoint."""
    code, stats = run_cli(accepted_updates=0)

    assert stats.accepted_updates == 0
    assert code == 2, "a fully rejected run must fail closed, not report success"
    assert list((tmp_path / "ckpt").glob("step-*.pt")) == []


def test_rejected_run_report_marks_no_checkpoint_and_no_quality(run_cli, capsys) -> None:
    code, _ = run_cli(accepted_updates=0)

    assert code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["checkpoint"] is None
    assert report["quality_supported"] is False
    assert report["accepted_updates"] == 0


def test_accepted_run_saves_checkpoint_and_stays_research_only(run_cli, tmp_path: Path, capsys) -> None:
    code, _ = run_cli(accepted_updates=1)

    assert code == 0
    saved = list((tmp_path / "ckpt").glob("step-*.pt"))
    assert len(saved) == 1, "an accepted update must persist exactly one checkpoint"
    report = json.loads(capsys.readouterr().out)
    assert report["checkpoint"] == str(saved[0])
    assert report["quality_supported"] is False, "operational success is not quality evidence"
    assert report["accepted_updates"] == 1


def test_rls_warmup_without_applied_delta_persists_nothing(monkeypatch, tmp_path: Path, capsys) -> None:
    """RLS warmup may accumulate rows while every ``lora_B`` stays unchanged.

    That is in-memory fitter progress, but model-only checkpoints cannot persist
    those accumulators. Treating the warmup as durable progress would hand a
    later run an unchanged parent with a new step number.
    """
    result = SelfImproveResult(
        iteration=0,
        generated_ids=[1, 2],
        pre_ce=9.2,
        post_ce=9.2,
        kl_div=0.0,
        update_applied=False,
        delta_rms_frac=0.0,
    )
    stats = SelfImproveStats(
        iterations=[result],
        stopped="max_iterations",
        best_ce=9.2,
        accepted_updates=1,
    )
    monkeypatch.setattr(self_improve_cli, "self_improve", lambda **kwargs: stats)
    monkeypatch.setattr(self_improve_cli, "load_config", lambda _p: tiny_config())
    monkeypatch.setattr(
        self_improve_cli.sys, "argv",
        ["self_improve.py", *_rls_argv(tmp_path)],
    )

    assert self_improve_cli.main() == 2
    assert list((tmp_path / "ckpt").glob("step-*.pt")) == []
    report = json.loads(capsys.readouterr().out)
    assert report["checkpoint"] is None
    assert report["accepted_updates"] == 1
    assert report["applied_updates"] == 0


def test_report_shape_is_stable_for_downstream_automation(run_cli, capsys) -> None:
    """Automation must be able to branch on accepted/rejected without parsing prose."""
    run_cli(accepted_updates=0)
    report = json.loads(capsys.readouterr().out)
    assert set(report) >= {
        "checkpoint",
        "step",
        "accepted_updates",
        "applied_updates",
        "quality_supported",
        "stopped",
        "iterations",
    }


def test_cli_is_importable_without_executing_main() -> None:
    """The module must stay importable for tests (``raise SystemExit`` is guarded)."""
    assert callable(self_improve_cli.main)
    assert sys.modules["scripts.self_improve"] is self_improve_cli


def _rls_argv(tmp_path: Path) -> list[str]:
    return [
        *_default_argv(tmp_path),
        "--mode",
        "rls",
    ]


def test_rls_mode_selects_the_ttt_lora_contour_only(monkeypatch, tmp_path: Path) -> None:
    """RLS fits ``lora_B`` through the ridge solve, so the CLI must not leave
    the pyramid contour enabled -- ``self_improve`` rejects both at once."""
    captured: dict[str, object] = {}

    def fake_self_improve(**kwargs):
        captured["cfg"] = kwargs["cfg"]
        captured["mode"] = kwargs["mode"]
        return SelfImproveStats(
            iterations=[SelfImproveResult(
                iteration=0, generated_ids=[1], pre_ce=9.2, post_ce=9.2,
                kl_div=0.0, update_applied=False, delta_rms_frac=0.0,
            )],
            stopped="kl_bound",
            accepted_updates=0,
        )

    monkeypatch.setattr(self_improve_cli, "self_improve", fake_self_improve)
    monkeypatch.setattr(self_improve_cli, "load_config", lambda _p: tiny_config())
    monkeypatch.setattr(
        self_improve_cli.sys, "argv",
        ["self_improve.py", *_rls_argv(tmp_path)],
    )
    self_improve_cli.main()

    cfg = captured["cfg"]
    assert captured["mode"] == "rls"
    assert cfg.model.adapters.ttt_lora.enabled is True
    assert cfg.model.adapters.pyramid.enabled is False


def test_rls_mode_never_builds_a_trainer_or_optimizer(monkeypatch, tmp_path: Path) -> None:
    """RLS has no optimizer. Creating one would silently add state the RLS
    transaction never snapshots or restores."""
    def explode(*args, **kwargs):
        raise AssertionError("RLS mode must not construct a Trainer/optimizer")

    monkeypatch.setattr(self_improve_cli, "Trainer", explode)
    monkeypatch.setattr(self_improve_cli, "self_improve", lambda **kwargs: SelfImproveStats(
        iterations=[SelfImproveResult(
            iteration=0, generated_ids=[1], pre_ce=9.2, post_ce=9.2,
            kl_div=0.0, update_applied=False, delta_rms_frac=0.0,
        )],
        stopped="kl_bound", accepted_updates=0,
    ))
    monkeypatch.setattr(self_improve_cli, "load_config", lambda _p: tiny_config())
    monkeypatch.setattr(
        self_improve_cli.sys, "argv",
        ["self_improve.py", *_rls_argv(tmp_path)],
    )
    assert self_improve_cli.main() == 2


def test_rls_mode_rejects_resume_before_any_work(monkeypatch, tmp_path: Path) -> None:
    """``save_checkpoint`` persists no RLS accumulators, so resuming would
    restore ``lora_B`` while silently dropping ``G``/``C``/buffers."""
    monkeypatch.setattr(
        self_improve_cli, "self_improve",
        lambda **kwargs: pytest.fail("RLS resume must fail before the loop runs"),
    )
    monkeypatch.setattr(self_improve_cli, "load_config", lambda _p: tiny_config())
    monkeypatch.setattr(
        self_improve_cli.sys, "argv",
        ["self_improve.py", *_rls_argv(tmp_path), "--resume"],
    )
    with pytest.raises(SystemExit) as excinfo:
        self_improve_cli.main()
    assert "resume" in str(excinfo.value)


def test_report_records_mode_and_rls_step_size(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(self_improve_cli, "self_improve", lambda **kwargs: SelfImproveStats(
        iterations=[SelfImproveResult(
            iteration=0, generated_ids=[1], pre_ce=9.2, post_ce=9.1,
            kl_div=0.0, update_applied=True, delta_rms_frac=0.004,
        )],
        stopped="max_iterations", accepted_updates=1, best_ce=9.1,
    ))
    monkeypatch.setattr(self_improve_cli, "load_config", lambda _p: tiny_config())
    monkeypatch.setattr(
        self_improve_cli.sys, "argv",
        ["self_improve.py", *_rls_argv(tmp_path)],
    )
    assert self_improve_cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "rls"
    assert report["delta_rms_frac"] == 0.004


def test_gradient_mode_remains_the_default_contract(monkeypatch, tmp_path: Path, capsys) -> None:
    """No ``--mode`` must keep the existing pyramid + Trainer + optimizer
    semantics, so the default path is not silently redefined."""
    captured: dict[str, object] = {}

    def fake_self_improve(**kwargs):
        captured["mode"] = kwargs["mode"]
        captured["trainer"] = kwargs["trainer"]
        return SelfImproveStats(
            iterations=[SelfImproveResult(
                iteration=0, generated_ids=[1], pre_ce=9.2, post_ce=9.1,
                kl_div=0.0, update_applied=True,
            )],
            stopped="max_iterations", accepted_updates=1, best_ce=9.1,
        )

    monkeypatch.setattr(self_improve_cli, "self_improve", fake_self_improve)
    monkeypatch.setattr(self_improve_cli, "load_config", lambda _p: tiny_config())
    monkeypatch.setattr(
        self_improve_cli.sys, "argv",
        ["self_improve.py", *_default_argv(tmp_path)],
    )
    assert self_improve_cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert captured["mode"] == "gradient"
    assert captured["trainer"] is not None
    assert report["mode"] == "gradient"


@pytest.mark.parametrize(
    "optimizer_state",
    [None, {}, {"unexpected": {"state": {}}}],
)
def test_gradient_resume_requires_valid_optimizer_state(
    monkeypatch, tmp_path: Path, optimizer_state
) -> None:
    """A resumed gradient run cannot silently discard its optimizer history."""
    checkpoint_dir = tmp_path / "resume"
    checkpoint_dir.mkdir()
    monkeypatch.setattr(
        self_improve_cli,
        "latest_checkpoint",
        lambda _directory: checkpoint_dir / "step-0000003.pt",
    )
    def fail_load_model(*_args, **_kwargs):
        pytest.fail("invalid optimizer state must fail before loading model weights")

    monkeypatch.setattr(self_improve_cli, "load_model", fail_load_model)
    monkeypatch.setattr(
        self_improve_cli,
        "load_payload",
        lambda *_args, **_kwargs: {"optimizer": optimizer_state},
    )
    monkeypatch.setattr(
        self_improve_cli,
        "self_improve",
        lambda **kwargs: pytest.fail("invalid resume must fail before the loop"),
    )
    monkeypatch.setattr(
        self_improve_cli.sys,
        "argv",
        [
            "self_improve.py",
            *_default_argv(tmp_path),
            "--resume",
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
    )

    with pytest.raises(SystemExit, match="optimizer"):
        self_improve_cli.main()
    assert list(checkpoint_dir.glob("step-*.pt")) == []


def test_gradient_resume_rejects_incompatible_optimizer_state(monkeypatch, tmp_path: Path) -> None:
    """Outer schema alone is insufficient; the real optimizer load must validate it."""
    checkpoint_dir = tmp_path / "resume"
    checkpoint_dir.mkdir()
    monkeypatch.setattr(
        self_improve_cli,
        "latest_checkpoint",
        lambda _directory: checkpoint_dir / "step-0000003.pt",
    )
    monkeypatch.setattr(
        self_improve_cli,
        "load_model",
        lambda *_args, **_kwargs: (3, tiny_config()),
    )
    monkeypatch.setattr(
        self_improve_cli,
        "load_payload",
        lambda *_args, **_kwargs: {
            "optimizer": {"muon": {}, "adamw": {}}
        },
    )
    monkeypatch.setattr(
        self_improve_cli,
        "self_improve",
        lambda **kwargs: pytest.fail("incompatible optimizer must fail before the loop"),
    )
    monkeypatch.setattr(
        self_improve_cli.sys,
        "argv",
        [
            "self_improve.py",
            *_default_argv(tmp_path),
            "--resume",
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
    )

    with pytest.raises(SystemExit, match="optimizer state is incompatible"):
        self_improve_cli.main()
    assert list(checkpoint_dir.glob("step-*.pt")) == []


def test_gradient_resume_without_checkpoint_fails_closed(monkeypatch, tmp_path: Path) -> None:
    """``--resume`` means resume, not "start a fresh run".

    Falling through when the directory is empty makes a missing parent look
    like successful recovery and lets a new checkpoint appear under the same
    automation command that expected an existing lineage.
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(
        self_improve_cli, "self_improve",
        lambda **kwargs: pytest.fail("resume without a checkpoint must fail before work"),
    )
    monkeypatch.setattr(self_improve_cli, "load_config", lambda _p: tiny_config())
    monkeypatch.setattr(
        self_improve_cli.sys, "argv",
        [
            "self_improve.py",
            *_default_argv(tmp_path),
            "--resume",
            "--checkpoint-dir",
            str(empty_dir),
        ],
    )
    with pytest.raises(SystemExit, match="resume"):
        self_improve_cli.main()
    assert list(empty_dir.glob("step-*.pt")) == []

