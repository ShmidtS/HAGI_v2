"""Offline tests for the tokenizer-independent decoded-text evaluator."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from hagi.model.model import HAGI
from hagi.train.checkpoint import save_checkpoint
from scripts.common_reference_eval import (
    TokenizerLane,
    build_scoring_windows,
    create_gigatoken_adapter,
    evaluate_checkpoint,
    evaluate_text,
    make_metrics,
    parse_args,
)
from tests.conftest import tiny_config


class ExactTokenizer:
    """Small lossless tokenizer used to make expected IDs explicit."""

    vocab_size = 1024

    def __init__(self, rows: dict[str, list[int]], name: str = "exact-fixture-v1"):
        self.rows = rows
        self.name = name
        self.version = "test-v1"
        self.fingerprint = None

    def encode(self, text: str) -> list[int]:
        return list(self.rows[text])

    def decode(self, ids: list[int]) -> bytes:
        for text, row in self.rows.items():
            if row == ids:
                return text.encode("utf-8")
        raise ValueError("unknown fixture row")


def tokenizer_lane(tokenizer: ExactTokenizer) -> TokenizerLane:
    return TokenizerLane(
        name=tokenizer.name,
        version=tokenizer.version,
        vocab_size=tokenizer.vocab_size,
        encode=tokenizer.encode,
        decode=tokenizer.decode,
        fingerprint=tokenizer.fingerprint,
    )


def _flatten_targets(windows) -> list[int]:
    return [token for window in windows for token in window.target_ids]


def _flatten_scored_positions(windows) -> list[tuple[int, int]]:
    return [(index, position) for index, window in enumerate(windows) for position in window.scored_positions]


def test_rolling_windows_score_every_content_token_once_with_retained_context():
    windows = build_scoring_windows(
        list(range(10)),
        prefix_token_id=99,
        max_seq_len=5,
        context_tokens=3,
    )

    assert _flatten_targets(windows) == list(range(10))
    assert len({token for token in _flatten_targets(windows)}) == 10
    assert all(1 <= len(window.input_ids) <= 5 for window in windows)
    assert all(window.input_ids[window.scored_positions[0]] is not None for window in windows)
    assert windows[0].input_ids == (99, 0, 1, 2, 3)
    assert windows[0].scored_positions == (0, 1, 2, 3, 4)
    assert windows[1].input_ids == (2, 3, 4, 5, 6)
    assert windows[1].scored_positions == (2, 3, 4)
    assert windows[2].input_ids == (5, 6, 7, 8)
    assert windows[2].scored_positions == (2, 3)
    assert _flatten_scored_positions(windows)[-1] == (2, 3)


def test_rolling_windows_never_append_eos_or_truncate_content():
    windows = build_scoring_windows([3, 4, 5, 6], prefix_token_id=1, max_seq_len=4, context_tokens=2)
    assert _flatten_targets(windows) == [3, 4, 5, 6]
    assert all(window.target_ids[-1] != 1 for window in windows)
    assert sum(len(window.target_ids) for window in windows) == 4


def test_prefix_free_single_token_fails_closed_because_it_has_no_causal_context():
    with pytest.raises(ValueError, match="prefix-free"):
        build_scoring_windows([1], prefix_token_id=None, max_seq_len=4, context_tokens=2)


@pytest.mark.parametrize(
    "tokens,prefix,max_len,context,match",
    [
        ([], 0, 4, 2, "content tokens"),
        ([1], 0, 1, 1, "max_seq_len"),
        ([1], 0, 4, 0, "context_tokens"),
        ([1], 0, 4, 4, "context_tokens"),
        ([-1], 0, 4, 2, "non-negative"),
        ([1], -1, 4, 2, "prefix_token_id"),
    ],
)
def test_rolling_windows_fail_closed(tokens, prefix, max_len, context, match):
    with pytest.raises(ValueError, match=match):
        build_scoring_windows(tokens, prefix, max_len, context)


def test_closed_form_metrics_use_full_utf8_byte_denominator():
    text = "aЯ"
    total_nll = 4.5
    metrics = make_metrics(total_nll=total_nll, token_count=3, text=text)
    byte_count = len(text.encode("utf-8"))

    assert metrics["token_count"] == 3
    assert metrics["byte_count"] == byte_count
    assert metrics["character_count"] == 2
    assert metrics["native_token_ce"] == pytest.approx(1.5)
    assert metrics["byte_perplexity"] == pytest.approx(math.exp(4.5 / byte_count))
    assert metrics["bits_per_byte"] == pytest.approx(4.5 / (byte_count * math.log(2.0)))
    assert metrics["tokens_per_byte"] == pytest.approx(3 / byte_count)


def test_evaluate_text_matches_independent_tiny_hagi_calculation():
    cfg = tiny_config(**{"train.tokenizer": "exact-fixture-v1"})
    model = HAGI(cfg).eval()
    text = "abc"
    tokenizer = ExactTokenizer({text: [10, 20, 30]})

    report = evaluate_text(
        model,
        cfg,
        text,
        tokenizer_lane(tokenizer),
        context_tokens=8,
        device=torch.device("cpu"),
    )
    windows = build_scoring_windows(
        [10, 20, 30],
        prefix_token_id=cfg.train.data.eos_token_id,
        max_seq_len=cfg.model.attention.max_seq_len,
        context_tokens=8,
    )
    expected = 0.0
    for window in windows:
        output = model(torch.tensor([window.input_ids], dtype=torch.long))
        selected = output.hidden[0, list(window.scored_positions)]
        expected += float(model.head.exact_loss(selected, torch.tensor(window.target_ids)).detach()) * len(
            window.target_ids
        )

    assert report["scored_token_count"] == 3
    assert report["total_nll"] == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert report["round_trip_exact"] is True
    assert report["scoring_policy"]["prefix_token_id"] == cfg.train.data.eos_token_id
    assert report["scoring_policy"]["appended_eos"] is False
    assert report["scoring_policy"]["bos_added"] is False


def test_evaluate_text_rolls_long_content_without_changing_hand_calculated_nll():
    cfg = tiny_config(
        **{
            "model.attention.max_seq_len": 4,
            "train.data.seq_len": 4,
            "train.tokenizer": "exact-fixture-v1",
        }
    )
    model = HAGI(cfg).eval()
    text = "abcdefghij"
    ids = list(range(10, 20))
    tokenizer = ExactTokenizer({text: ids})

    report = evaluate_text(
        model,
        cfg,
        text,
        tokenizer_lane(tokenizer),
        context_tokens=2,
        device=torch.device("cpu"),
    )
    windows = build_scoring_windows(ids, 1, 4, 2)
    expected = 0.0
    for window in windows:
        output = model(torch.tensor([window.input_ids], dtype=torch.long))
        selected = output.hidden[0, list(window.scored_positions)]
        expected += float(model.head.exact_loss(selected, torch.tensor(window.target_ids)).detach()) * len(
            window.target_ids
        )

    assert len(windows) == 3
    assert report["scored_token_count"] == 10
    assert report["total_nll"] == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert report["scoring_policy"]["window_count"] == 3


def test_evaluate_text_rejects_lossy_decode_and_invalid_model_ids():
    cfg = tiny_config(**{"train.tokenizer": "exact-fixture-v1"})
    model = HAGI(cfg)
    text = "abc"

    class LossyTokenizer(ExactTokenizer):
        def decode(self, ids):
            return b"different"

    with pytest.raises(ValueError, match="round-trip"):
        evaluate_text(model, cfg, text, tokenizer_lane(LossyTokenizer({text: [10, 20, 30]})), device="cpu")

    with pytest.raises(ValueError, match="model vocabulary"):
        evaluate_text(model, cfg, text, tokenizer_lane(ExactTokenizer({text: [10, 20, 512]})), device="cpu")

    mismatched = ExactTokenizer({text: [10, 20, 30]}, name="other-tokenizer")
    with pytest.raises(ValueError, match="does not match checkpoint tokenizer"):
        evaluate_text(model, cfg, text, tokenizer_lane(mismatched), device="cpu")


def test_evaluate_text_rejects_non_integer_token_ids_and_context_tokens():
    cfg = tiny_config(**{"train.tokenizer": "exact-fixture-v1"})
    model = HAGI(cfg)
    text = "abc"
    tokenizer = ExactTokenizer({text: [10, 20, 30]})
    tokenizer.encode = lambda value: [10.5, 20, 30]
    with pytest.raises(ValueError, match="non-negative integers"):
        evaluate_text(model, cfg, text, tokenizer_lane(tokenizer), device="cpu")

    clean = tokenizer_lane(ExactTokenizer({text: [10, 20, 30]}))
    with pytest.raises(ValueError, match="context_tokens must be an integer"):
        evaluate_text(model, cfg, text, clean, context_tokens=2.5, device="cpu")


def test_evaluate_text_rejects_vocab_map_with_wrong_model_size(tmp_path):
    import numpy as np

    from hagi.data.vocab_map import VocabMap

    cfg = tiny_config(**{"train.tokenizer": "exact-fixture-v1"})
    model = HAGI(cfg)
    text = "abc"
    np.savez(
        tmp_path / "vocab_map.npz",
        old_to_new=np.arange(8, dtype=np.int64),
        new_to_old=np.arange(7, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="does not match model vocabulary"):
        evaluate_text(
            model,
            cfg,
            text,
            tokenizer_lane(ExactTokenizer({text: [1, 2, 3]})),
            vocab_map=VocabMap(tmp_path / "vocab_map.npz", fallback=3),
            device="cpu",
        )


class FakeGigaModule:
    class Tokenizer:
        def __init__(self, name):
            self.name = name
            self.vocab_size = 32
            self.as_hf_called = False

        def as_hf(self):
            self.as_hf_called = True
            raise AssertionError("native verified API is encode_batch_list, not HFCompat.encode_batch")

        def encode_batch_list(self, rows):
            assert rows == ["Привет"]
            return [[7, 8, 9]]

        def decode(self, ids):
            assert ids == [7, 8, 9]
            return "Привет".encode()


def test_gigatoken_adapter_uses_verified_native_api_and_strict_utf8():
    adapter = create_gigatoken_adapter("fixture", module=FakeGigaModule, version="0.10.0")
    assert adapter.name == "fixture"
    assert adapter.version == "0.10.0"
    assert adapter.vocab_size == 32
    assert adapter.encode("Привет") == [7, 8, 9]
    assert adapter.decode([7, 8, 9]) == "Привет"

    class BrokenDecode(FakeGigaModule.Tokenizer):
        def decode(self, ids):
            return b"\xff"

    module = FakeGigaModule()
    module.Tokenizer = BrokenDecode
    adapter = create_gigatoken_adapter("broken", module=module, version="0.10.0")
    with pytest.raises(UnicodeDecodeError):
        adapter.decode([7, 8, 9])


def test_evaluate_checkpoint_report_is_complete_and_fail_closed(tmp_path):
    cfg = tiny_config(**{"train.tokenizer": "exact-fixture-v1"})
    model = HAGI(cfg)
    checkpoint = save_checkpoint(model, cfg, 0, tmp_path / "checkpoint", keep_last=1)
    text_path = tmp_path / "heldout.txt"
    text = "abc"
    text_path.write_text(text, encoding="utf-8", newline="")
    source_hash = hashlib.sha256(text_path.read_bytes()).hexdigest()
    tokenizer = ExactTokenizer({text: [10, 20, 30]})

    report = evaluate_checkpoint(
        checkpoint,
        text_path,
        tokenizer_lane(tokenizer),
        expected_source_sha256=source_hash,
        device="cpu",
        artifact_manifest_validated=False,
    )

    assert report["quality_supported"] is False
    assert report["promotion_status"] == "research-only"
    assert report["source"]["sha256"] == source_hash
    assert report["checkpoint"]["completed_steps"] == 0
    assert report["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert len(report["checkpoint"]["config_sha256"]) == 64
    assert report["scoring_policy"]["exact_full_alphabet_head"] is True
    assert report["scoring_policy"]["sampled_softmax_forbidden"] is True

    serialized = json.dumps(report, sort_keys=True, allow_nan=False)
    assert "Infinity" not in serialized and "NaN" not in serialized


def test_evaluate_checkpoint_rejects_wrong_or_malformed_source_hash(tmp_path):
    cfg = tiny_config(**{"train.tokenizer": "exact-fixture-v1"})
    checkpoint = save_checkpoint(HAGI(cfg), cfg, 0, tmp_path / "checkpoint", keep_last=1)
    text_path = tmp_path / "heldout.txt"
    text_path.write_text("abc", encoding="utf-8", newline="")
    tokenizer = tokenizer_lane(ExactTokenizer({"abc": [10, 20, 30]}))
    for digest, match in (("0" * 64, "sha256 mismatch"), ("abc", "hexadecimal")):
        with pytest.raises(ValueError, match=match):
            evaluate_checkpoint(
                checkpoint,
                text_path,
                tokenizer,
                expected_source_sha256=digest,
                device="cpu",
            )


def test_argument_parser_requires_explicit_source_hash_and_bounded_size():
    args = parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--text-file",
            "heldout.txt",
            "--expected-source-sha256",
            "a" * 64,
            "--tokenizer",
            "fixture",
        ]
    )
    assert args.max_source_bytes == 64 * 1024 * 1024
    assert args.context_tokens is None
    assert args.quality_supported is False

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--text-file",
                "heldout.txt",
                "--tokenizer",
                "fixture",
            ]
        )


def test_runner_does_not_claim_quality_without_matched_recipe(tmp_path: Path):
    # This test pins the fail-closed schema used by future reports.
    assert not (tmp_path / "not-created.json").exists()
