"""Falsifier tests for the pinned external task-suite evaluation.

Each test asserts a property a near-identity or non-discriminating
implementation would fail: digest pins must bite, the promotion gate must
actually discriminate, and the champion bytes must survive rejection.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from hagi.config import CHECKPOINT_FORMAT_VERSION
from hagi.model.model import HAGI
from hagi.orchestrator.external_eval import (
    ExternalSuite,
    apply_promotion,
    build_external_suite,
    protocol_payload,
    row_digest,
    score_checkpoint,
    select_champion,
    suite_payload,
    verified_suite_source,
)
from hagi.orchestrator.state import canonical_json_bytes, sha256_bytes
from hagi.train.checkpoint import config_to_dict
from tests.conftest import tiny_config

VOCAB = 16
SCORER = "exact_ce_packed_v1"


def _manifest(source_bytes: bytes) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "dataset",
        "dataset": "synthetic-external",
        "split": "unseen",
        "vocab_size": VOCAB,
        "sources": [
            {
                "name": "A",
                "ratio": 1.0,
                "dataset": "synthetic-external",
                "revision": "r1",
                "license": "test",
                "source_url": "local://synthetic",
                "retrieval_timestamp": "2026-09-25T00:00:00Z",
                "tokenizer_version": "synthetic-packed-v1",
                "filter_policy_version": "v1",
                "dedup_policy_version": "v1",
                "byte_count": len(source_bytes),
                "token_count": 12,
                "input_sha256": sha256_bytes(b"external-a"),
                "output_sha256": sha256_bytes(b"external-a-out"),
            }
        ],
        "files": [],
    }


def _suite(tmp_path: Path) -> ExternalSuite:
    return build_external_suite(
        tmp_path / "suite",
        suite_id="tiny-unseen",
        split="test",
        tokenizer_name="synthetic-packed-v1",
        vocab_size=VOCAB,
        sources={"A": [[1, 2, 3, 4], [5, 6, 7, 8]], "B": [[9, 10, 11, 12], [13, 14, 15, 0]]},
        manifest=_manifest(b"placeholder"),
        scorer_id=SCORER,
    )


def _write_checkpoint(path: Path, seed: int, *, scale: float = 1.0, steps: int = 0) -> str:
    """Write a tiny random-weight checkpoint; larger scale => sharper logits."""
    cfg = tiny_config(
        **{
            "model.vocab_size": VOCAB,
            "model.hidden_size": 16,
            "model.num_layers": 1,
            "model.attention.num_query_heads": 1,
            "model.attention.num_kv_heads": 1,
            "model.attention.head_dim": 16,
            "model.attention.max_seq_len": 8,
            "model.embedding.conv_kernel": 1,
            "model.embedding.tie_lm_head": False,
            "model.ternary.enabled": False,
            "model.adapters.enabled": False,
            "model.cortex.enabled": False,
            "model.decision.enabled": False,
            "model.loop_depth": 2,
            "model.head.unigram_prior": False,
            "model.head.unigram_path": "",
            "train.data.seq_len": 4,
            "train.max_steps": 1,
            "train.compile_model": False,
        }
    )
    torch.manual_seed(seed)
    model = HAGI(cfg)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(scale)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "config": config_to_dict(cfg),
        "completed_steps": steps,
    }
    torch.save(payload, path)
    return sha256_bytes(path.read_bytes())


def test_tampered_source_bytes_are_rejected(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    good = verified_suite_source(suite)
    data = json.loads(Path(suite.source_path).read_text(encoding="utf-8"))
    data["rows"]["A"][0]["token_ids"] = [1, 2, 3, 9]
    Path(suite.source_path).write_bytes(canonical_json_bytes(data))
    # Digest pin must fire before any row-level check.
    with pytest.raises(ValueError, match="digest mismatch"):
        verified_suite_source(suite)
    # And even a re-pinned-but-stale row digest must be refused.
    repinned = replace(suite, source_sha256=sha256_bytes(Path(suite.source_path).read_bytes()))
    with pytest.raises(ValueError, match="row digest mismatch"):
        verified_suite_source(repinned)
    assert good.row_ids_sha256[0] == row_digest([1, 2, 3, 4])


def test_wrong_manifest_is_rejected(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    manifest = json.loads(Path(suite.manifest_path).read_text(encoding="utf-8"))
    manifest["artifact_type"] = "checkpoint"
    Path(suite.manifest_path).write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError):
        verified_suite_source(suite)
    # Same bytes re-pinned, but the manifest is schema-invalid.
    repinned = replace(suite, manifest_sha256=sha256_bytes(Path(suite.manifest_path).read_bytes()))
    with pytest.raises(ValueError, match="artifact_type"):
        verified_suite_source(repinned)
    # A manifest naming a different suite is also refused.
    other = _suite(tmp_path / "other")
    with pytest.raises(ValueError, match="digest mismatch"):
        verified_suite_source(replace(suite, manifest_sha256=other.manifest_sha256))


def test_wrong_checkpoint_digest_is_rejected(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    path = tmp_path / "c.pt"
    _write_checkpoint(path, 7)
    with pytest.raises(ValueError, match="digest mismatch"):
        score_checkpoint(path, "0" * 64, suite)


def test_worse_challenger_is_rejected_and_champion_bytes_unchanged(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path)
    # Swap the roles: the better checkpoint is now the champion, so the
    # challenger (the measured worse one) must be rejected on external evidence.
    worse, worse_sha, better, better_sha = _select_roles(tmp_path, suite)
    champion, champion_sha = better, better_sha
    challenger, challenger_sha = worse, worse_sha
    before = champion.read_bytes()
    verdict = apply_promotion(
        champion,
        champion_sha,
        challenger,
        challenger_sha,
        suite,
        margin=0.0,
        max_source_regression=1.0,
    )
    assert verdict["decision"] == "retain_champion"
    assert float(verdict["external_gain"]) <= 0.0
    assert champion.read_bytes() == before
    assert sha256_bytes(champion.read_bytes()) == champion_sha


def _select_roles(tmp_path: Path, suite: ExternalSuite) -> tuple[Path, str, Path, str]:
    """Return (champion, sha, challenger, sha) with the champion measurably worse.

    Roles are assigned from the measured CE rather than from seed luck, so the
    tests assert gate behaviour instead of a property of a random draw.
    """
    scored: list[tuple[float, Path, str]] = []
    for name, seed, scale in (("a", 31, 1.0), ("b", 41, 1.0)):
        path = tmp_path / f"{name}.pt"
        digest = _write_checkpoint(path, seed, scale=scale)
        verdict = score_checkpoint(path, digest, suite)
        scored.append((float(verdict["macro_exact_ce"]), path, digest))
    scored.sort(key=lambda item: item[0], reverse=True)
    worst, best = scored[0], scored[1]
    if worst[0] <= best[0]:
        pytest.skip("degenerate fixture: the two checkpoints scored identically")
    return worst[1], worst[2], best[1], best[2]


def test_better_challenger_is_promoted_only_with_margin(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    champion, champion_sha, challenger, challenger_sha = _select_roles(tmp_path, suite)
    # The declared ceiling is part of the caller's contract: with a ceiling of
    # 1.0 the macro improvement decides, with the default 0.0 a per-source
    # regression blocks promotion even when the macro is better.
    verdict = select_champion(
        champion,
        champion_sha,
        challenger,
        challenger_sha,
        suite,
        margin=0.0,
        max_source_regression=1.0,
    )
    assert float(verdict["external_gain"]) > 0.0
    assert verdict["decision"] == "promote"
    # A margin above the observed gain must flip the decision.
    strict = select_champion(
        champion,
        champion_sha,
        challenger,
        challenger_sha,
        suite,
        margin=float(verdict["external_gain"]) + 1.0,
        max_source_regression=1.0,
    )
    assert strict["decision"] == "retain_champion"
    # Same candidate, default zero-regression ceiling.
    tight = select_champion(
        champion,
        champion_sha,
        challenger,
        challenger_sha,
        suite,
        margin=0.0,
    )
    assert tight["worst_source_regression"] == float(verdict["worst_source_regression"])
    assert (tight["worst_source_regression"] > 0.0) == (
        tight["decision"] == "retain_champion"
    )


def test_equal_score_challenger_rejected_at_zero_margin(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    champion = tmp_path / "champion.pt"
    challenger = tmp_path / "challenger.pt"
    # Byte-different but weight-identical: same CE, different bytes. The
    # completed_steps field differs, so a byte-comparing implementation would
    # see "a different checkpoint"; a CE-comparing one must see zero gain.
    champion_sha = _write_checkpoint(champion, 55, steps=0)
    challenger_sha = _write_checkpoint(challenger, 55, steps=1)
    assert challenger_sha != champion_sha
    verdict = select_champion(
        champion, champion_sha, challenger, challenger_sha, suite, margin=0.0
    )
    assert verdict["decision"] == "retain_champion"
    assert float(verdict["external_gain"]) == pytest.approx(0.0, abs=1e-6)


def test_missing_margin_fails_closed(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    champion = tmp_path / "champion.pt"
    challenger = tmp_path / "challenger.pt"
    champion_sha = _write_checkpoint(champion, 61)
    challenger_sha = _write_checkpoint(challenger, 71)
    with pytest.raises(ValueError, match="margin must be supplied"):
        select_champion(
            champion, champion_sha, challenger, challenger_sha, suite, margin=None
        )
    with pytest.raises(ValueError, match="margin"):
        apply_promotion(
            champion, champion_sha, challenger, challenger_sha, suite, margin=None
        )


def test_source_regression_blocks_promotion(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    champion, champion_sha, challenger, challenger_sha = _select_roles(tmp_path, suite)
    worst = float(
        select_champion(
            champion,
            champion_sha,
            challenger,
            challenger_sha,
            suite,
            margin=0.0,
            max_source_regression=1e9,
        )["worst_source_regression"]
    )
    assert worst >= 0.0
    strict = select_champion(
        champion,
        champion_sha,
        challenger,
        challenger_sha,
        suite,
        margin=0.0,
        max_source_regression=0.0,
    )
    # If any source regressed, the zero-regression ceiling must block it.
    assert (worst > 0.0) == (strict["decision"] == "retain_champion")
    looser = select_champion(
        champion,
        champion_sha,
        challenger,
        challenger_sha,
        suite,
        margin=0.0,
        max_source_regression=1e9,
    )
    assert looser["decision"] == "promote"


def test_verdict_digest_changes_with_any_pinned_field(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    path = tmp_path / "c.pt"
    checkpoint_sha = _write_checkpoint(path, 99)
    base = score_checkpoint(path, checkpoint_sha, suite)
    other_protocol = protocol_payload(SCORER + "-b", vocab_size=VOCAB)
    mutations = {
        "suite_id": replace(suite, suite_id="tiny-unseen-2"),
        "split": replace(suite, split="holdout"),
        "tokenizer_name": replace(suite, tokenizer_name="synthetic-packed-v2"),
        "scorer_id": replace(
            suite,
            scorer_id=SCORER + "-b",
            protocol_payload=other_protocol,
            protocol_sha256=sha256_bytes(other_protocol),
        ),
        "source_sha256": replace(suite, source_sha256="1" * 64),
        "manifest_sha256": replace(suite, manifest_sha256="2" * 64),
        "protocol_sha256": replace(
            suite,
            protocol_payload=other_protocol,
            protocol_sha256=sha256_bytes(other_protocol),
        ),
        "source_path": replace(suite, source_path=str(tmp_path / "elsewhere.json")),
        "manifest_path": replace(suite, manifest_path=str(tmp_path / "elsewhere-manifest.json")),
    }
    seen = {suite.binding_digest}
    for field, mutated in mutations.items():
        assert mutated.binding_digest != suite.binding_digest, field
        seen.add(mutated.binding_digest)
    # Distinct mutations must not collide into one binding digest.
    assert len(seen) == len(mutations) + 1
    # A protocol digest that does not match its payload is refused outright.
    with pytest.raises(ValueError, match="protocol payload"):
        replace(suite, protocol_sha256="3" * 64)
    assert base["suite_binding_sha256"] == suite.binding_digest
    # Scoring on a mutated-but-unwritten pin also fails closed.
    with pytest.raises(ValueError):
        score_checkpoint(path, checkpoint_sha, mutations["source_sha256"])
    with pytest.raises(ValueError):
        score_checkpoint(path, checkpoint_sha, mutations["manifest_sha256"])


def test_suite_row_ids_digest_covers_every_row(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    assert len(suite.row_ids_sha256) == 4
    payload = suite_payload(
        suite_id="tiny-unseen",
        split="test",
        tokenizer_name="synthetic-packed-v1",
        vocab_size=VOCAB,
        sources={"A": [[1, 2, 3, 4], [5, 6, 7, 8]], "B": [[9, 10, 11, 12], [13, 14, 15, 0]]},
    )
    assert sha256_bytes(payload) == suite.source_sha256


def test_duplicate_rows_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct"):
        suite_payload(
            suite_id="dup",
            split="test",
            tokenizer_name="synthetic-packed-v1",
            vocab_size=VOCAB,
            sources={"A": [[1, 2], [1, 2]]},
        )


def test_suite_contract_rejects_broken_digests(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    for kwargs in (
        {"source_sha256": "nope"},
        {"protocol_sha256": "4" * 64},
        {"suite_id": ""},
    ):
        with pytest.raises(ValueError):
            replace(suite, **kwargs)


def test_verdict_payload_is_json_serializable_and_digested(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    path = tmp_path / "c.pt"
    checkpoint_sha = _write_checkpoint(path, 123)
    verdict: dict[str, Any] = dict(score_checkpoint(path, checkpoint_sha, suite))
    assert verdict["schema"] == "hagi_external_verdict_v1"
    assert verdict["scored_rows"] == 4
    assert len(str(verdict["evidence_sha256"])) == 64
    payload = {key: value for key, value in verdict.items() if key != "source_metrics"}
    json.dumps(payload, allow_nan=False)
    assert float(verdict["macro_exact_ce"]) > 0.0
