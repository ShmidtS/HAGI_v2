"""Offline contracts for the pinned Banking77 real DecisionPlane runner."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.decision_plane_banking77 as banking77_module
from hagi.data.artifacts import file_entry
from hagi.model.adaptive import freeze_base_in_place
from scripts.decision_plane_banking77 import (
    EXPECTED_MANIFEST_SHA256,
    DecisionData,
    DecisionDocument,
    _source_provenance,
    _tokenizer_fingerprint,
    _update_metrics_finite,
    build_packed_shard_payloads,
    build_train_vocabulary,
    encode_decision_documents,
    make_exact_length_batches,
    make_real_config,
    parse_args,
    repeat_epoch_batches,
    run_seed,
    smoothed_label_prior_logits,
    verify_pinned_token_streams,
)


def _document(content: tuple[int, ...], label: int) -> DecisionDocument:
    return DecisionDocument(content_ids=content, label=label)


def _small_data() -> DecisionData:
    return DecisionData(
        train=(
            _document((3, 4), 0),
            _document((3, 5), 1),
            _document((6, 7), 0),
        ),
        test=(_document((3, 8), 1), _document((2, 3), 0)),
        num_options=2,
        native_vocab_size=16,
        train_native_unique=8,
        test_token_count=4,
        test_oov_token_count=1,
        manifest_sha256=EXPECTED_MANIFEST_SHA256,
        train_text_sha256="fixture-train",
        test_text_sha256="fixture-test",
        tokenizer_version="fixture-0",
        tokenizer_fingerprint={
            "vocab_size": 0,
            "merges_count": 0,
            "vocab_sha256": "0" * 64,
            "merges_sha256": "0" * 64,
            "fingerprint_sha256": "0" * 64,
        },
        quality_eligible=False,
    )


def test_train_only_vocabulary_reserves_pad_eos_unk_and_maps_oov():
    native_train = [[3, 4], [5, 4]]
    vocab = build_train_vocabulary(native_train, native_vocab_size=12)
    assert vocab.pad_id == 0
    assert vocab.eos_id == 1
    assert vocab.unk_id == 2
    assert vocab.vocab_size == 6
    assert np.asarray(vocab.old_to_new)[3:6].tolist() == [3, 4, 5]
    assert np.asarray(vocab.old_to_new)[6:].tolist() == [-1] * 6
    assert vocab.map_ids([3, 4, 5, 11]).tolist() == [3, 4, 5, 2]
    assert vocab.counts.tolist() == [0, 2, 0, 1, 2, 1]
    assert len(vocab.sha256) == 64


def test_exact_length_batches_have_prefix_targets_and_no_padding():
    vocab = build_train_vocabulary([[3, 4], [5, 6], [7, 8]], native_vocab_size=16)
    train = encode_decision_documents([[3, 4], [5, 6], [7, 8]], [0, 1, 0], vocab)
    batches = make_exact_length_batches(train, max_batch_size=2, seed=1234)
    assert len(batches) == 2
    assert [tuple(batch["input_ids"].shape) for batch in batches] == [(2, 3), (1, 3)]
    for batch in batches:
        assert torch.all(batch["input_ids"][:, 0] == vocab.eos_id)
        assert torch.all(batch["input_ids"][:, -1] != vocab.eos_id)
        assert torch.all(batch["targets"][:, -1] == vocab.eos_id)
        assert torch.all(batch["input_ids"] != vocab.pad_id)
        assert batch["input_ids"].shape == batch["targets"].shape
        assert batch["decision_targets"].shape == (batch["input_ids"].shape[0],)


def test_epoch_repetition_uses_every_document_once_per_epoch():
    vocab = build_train_vocabulary([[3, 4], [5, 6], [7]], native_vocab_size=16)
    train = encode_decision_documents([[3, 4], [5, 6], [7]], [0, 1, 1], vocab)
    batches = repeat_epoch_batches(train, epochs=2, max_batch_size=2, base_seed=1234)
    observed = []
    for batch in batches:
        for ids, label in zip(batch["content_ids"], batch["decision_targets"].tolist(), strict=True):
            observed.append((tuple(ids), label))
    assert len(observed) == 6
    assert sorted(observed) == sorted([((3, 4), 0), ((5, 6), 1), ((7,), 1)] * 2)


def test_smoothed_majority_prior_is_finite_and_histogram_argmax():
    logits = smoothed_label_prior_logits(torch.tensor([0, 0, 1]), num_options=3, rows=5)
    expected = torch.log(torch.tensor([3.0, 2.0, 1.0]) / 8.0)
    assert torch.allclose(logits, expected)
    assert logits.argmax().item() == 0
    assert bool(torch.isfinite(logits).all())


def test_pinned_token_stream_replay_requires_eos_appended_shards(tmp_path):
    native_rows = {"train": [[3, 4], [5, 6]], "test": [[7, 8]]}
    payloads = {
        split: build_packed_shard_payloads(rows, split=split)
        for split, rows in native_rows.items()
    }
    entries = []
    sources = []
    for split, split_payloads in payloads.items():
        digest = hashlib.sha256()
        for path, payload in sorted(split_payloads.items()):
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            entries.append(file_entry(path, payload))
            digest.update(payload)
        sources.append(
            {
                "name": split,
                "ratio": 1.0,
                "dataset": "fixture",
                "revision": "fixture",
                "license": "fixture",
                "source_url": "fixture",
                "retrieval_timestamp": "fixed",
                "tokenizer_version": "fixture",
                "filter_policy_version": "fixture",
                "dedup_policy_version": "fixture",
                "byte_count": 1,
                "token_count": 1,
                "input_sha256": "0" * 64,
                "output_sha256": digest.hexdigest(),
            }
        )
    manifest = {"files": entries, "sources": sources}
    verified = verify_pinned_token_streams(tmp_path, manifest, native_rows=native_rows)
    assert verified == {
        "train": {
            "verified": True,
            "token_count": 6,
            "sha256": hashlib.sha256(payloads["train"]["train/shard-000000.bin"]).hexdigest(),
            "shard_count": 1,
        },
        "test": {
            "verified": True,
            "token_count": 3,
            "sha256": hashlib.sha256(payloads["test"]["test/shard-000000.bin"]).hexdigest(),
            "shard_count": 1,
        },
    }
    (tmp_path / "test/shard-000000.bin").write_bytes(b"\x00" * 12)
    with pytest.raises(ValueError, match="tokenization drift"):
        verify_pinned_token_streams(tmp_path, manifest, native_rows=native_rows)


def test_real_config_matches_preregistered_geometry_and_objectives(tmp_path):
    counts = tmp_path / "unigram.npy"
    np.save(counts, np.array([0, 3, 0, 2, 2, 1, 1, 1, 0, 0], dtype=np.int64))
    cfg = make_real_config(
        seed=1234,
        max_steps=7,
        num_options=77,
        vocab_size=10,
        counts_path=counts,
        freeze_base=False,
    )
    assert cfg.model.vocab_size == 10
    assert cfg.model.hidden_size == 128
    assert cfg.model.num_layers == 4
    assert cfg.model.attention.max_seq_len == 128
    assert cfg.model.attention.sink_len == 0
    assert cfg.model.ffn.intermediate_size == 256
    assert cfg.model.embedding.tie_lm_head is True
    assert cfg.model.embedding.conv_kernel == 4
    assert cfg.model.ternary.enabled is True
    assert cfg.model.head.unigram_prior is True
    assert cfg.model.head.sampled_softmax_k == 0
    assert cfg.model.decision.num_options == 77
    assert cfg.train.precision == "fp32"
    assert cfg.train.use_muon is False
    assert cfg.train.adam.body_lr_scale == 1.0
    assert cfg.train.learning_rate == 0.003
    assert cfg.train.max_steps == 7
    assert cfg.train.z_loss_weight == 0.0
    assert cfg.model.head.z_loss_weight == 0.0


def test_one_seed_real_runner_executes_matched_lanes_but_is_not_quality_eligible(tmp_path):
    data = _small_data()
    counts = tmp_path / "unigram.npy"
    vocab = build_train_vocabulary(
        [[3, 4], [5, 6], [7, 8], [9, 10]],
        native_vocab_size=data.native_vocab_size,
    )
    np.save(counts, vocab.counts)
    report = run_seed(
        data,
        seed=1234,
        counts_path=counts,
        epochs=3,
        max_batch_size=4,
        grad_accum_steps=1,
        device="cpu",
    )
    assert [lane["name"] for lane in report["lanes"]] == [
        "majority_or_uniform",
        "frozen_probe",
        "decision_only",
        "end_to_end",
    ]
    assert report["mechanism_supported"] is True
    assert report["quality_supported"] is False
    assert report["preregistered_budget"] is False
    assert report["artifact_validated"] is False
    assert report["runtime_checks"]["frozen_base_unchanged"] is True
    assert report["runtime_checks"]["body_trainable_updates_observed"] is True
    by_name = {lane["name"]: lane for lane in report["lanes"]}
    assert by_name["frozen_probe"]["base_required_groups"] == []
    assert by_name["frozen_probe"]["base_trainable_parameter_count"] == 0
    assert by_name["decision_only"]["base_changed_parameter_count"] > 0
    assert by_name["end_to_end"]["base_changed_parameter_count"] > 0
    for lane in report["lanes"][1:]:
        ledger = lane["step_finiteness"]
        assert ledger["all_steps_finite"] is True
        assert ledger["record_count"] == report["optimizer_steps"]
        assert all(
            count == ledger["record_count"]
            for count in ledger["counts"].values()
        )
        assert hashlib.sha256(Path(ledger["path"]).read_bytes()).hexdigest() == ledger["sha256"]
    assert all(lane["finite"] for lane in report["lanes"])


def test_unexpected_base_freeze_fails_ownership_gate(monkeypatch, tmp_path):
    data = _small_data()
    counts = tmp_path / "unigram.npy"
    vocab = build_train_vocabulary(
        [[3, 4], [5, 6], [7, 8], [9, 10]],
        native_vocab_size=data.native_vocab_size,
    )
    np.save(counts, vocab.counts)

    class UnexpectedlyFrozenTrainer(banking77_module.Trainer):
        def __init__(self, model, cfg, start_step=0):
            freeze_base_in_place(model)
            super().__init__(model, cfg, start_step)

    monkeypatch.setattr(banking77_module, "Trainer", UnexpectedlyFrozenTrainer)
    report = run_seed(
        data,
        seed=1234,
        counts_path=counts,
        epochs=1,
        max_batch_size=4,
        grad_accum_steps=1,
        device="cpu",
    )
    assert report["runtime_checks"]["base_ownership_verified"] is False
    assert report["runtime_checks"]["body_trainable_updates_observed"] is False
    assert report["mechanism_supported"] is False
    by_name = {lane["name"]: lane for lane in report["lanes"]}
    for lane_name in ("decision_only", "end_to_end"):
        assert by_name[lane_name]["base_trainable_parameter_count"] == 0
        assert by_name[lane_name]["base_expected_trainable_parameter_count"] > 0
        assert by_name[lane_name]["base_ownership_matches"] is False


def test_tokenizer_fingerprint_is_deterministic_and_content_bound():
    class Backend:
        vocab = {1: b"a", 0: b"<pad>"}
        merges = [(b"a", b"b")]

    class Tokenizer:
        _backend = Backend()

    first = _tokenizer_fingerprint(Tokenizer())
    assert first == _tokenizer_fingerprint(Tokenizer())
    assert first["vocab_size"] == 2
    assert first["merges_count"] == 1
    Backend.vocab = {1: b"c", 0: b"<pad>"}
    changed_vocab = _tokenizer_fingerprint(Tokenizer())
    assert changed_vocab["fingerprint_sha256"] != first["fingerprint_sha256"]
    Backend.vocab = {1: b"a", 0: b"<pad>"}
    Backend.merges = [(b"a", b"c")]
    changed_merges = _tokenizer_fingerprint(Tokenizer())
    assert changed_merges["fingerprint_sha256"] != first["fingerprint_sha256"]


def test_update_finiteness_ignores_only_unused_muon_gradient_aliases():
    metrics = {
        "loss": 1.0,
        "grad_norm": float("nan"),
        "body_grad_norm": float("nan"),
        "rest_grad_norm": 0.5,
        "update_applied": True,
    }
    assert _update_metrics_finite(metrics, use_muon=False) is True
    assert _update_metrics_finite(metrics, use_muon=True) is False
    assert _update_metrics_finite({**metrics, "rest_grad_norm": float("nan")}, use_muon=False) is False


def test_source_provenance_preserves_exact_files_and_tree_hash(tmp_path):
    provenance = _source_provenance(tmp_path / "snapshot")
    assert len(provenance["tree_sha256"]) == 64
    assert len(provenance["files"]) >= 20
    for relative, metadata in provenance["files"].items():
        snapshot = tmp_path / "snapshot" / relative
        assert snapshot.is_file()
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == metadata["sha256"]
        assert snapshot.stat().st_size == metadata["byte_count"]


def test_cli_requires_explicit_manifest_hash():
    args = parse_args(
        [
            "--artifact",
            "artifact",
            "--expected-manifest-sha256",
            EXPECTED_MANIFEST_SHA256,
            "--device",
            "cpu",
        ]
    )
    assert args.expected_manifest_sha256 == EXPECTED_MANIFEST_SHA256
    assert args.num_seeds == 3
    assert args.epochs == 3
    with pytest.raises(SystemExit):
        parse_args(["--artifact", "artifact"])
    with pytest.raises(ValueError, match="num-seeds must be in"):
        parse_args(
            [
                "--artifact",
                "artifact",
                "--expected-manifest-sha256",
                EXPECTED_MANIFEST_SHA256,
                "--num-seeds",
                "4",
            ]
        )
    with pytest.raises(ValueError, match="three-seed quality protocol requires --seed 1234"):
        parse_args(
            [
                "--artifact",
                "artifact",
                "--expected-manifest-sha256",
                EXPECTED_MANIFEST_SHA256,
                "--seed",
                "999",
            ]
        )
