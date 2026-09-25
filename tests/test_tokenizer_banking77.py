"""Fast offline contracts for the pinned tokenizer frontier experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts import tokenizer_banking77 as subject
from scripts.tokenizer_banking77 import (
    EXPECTED_CANDIDATE_TOKENIZER_SHA256,
    EXPECTED_MANIFEST_SHA256,
    ByteNormalizedTrainer,
    GigaAdapter,
    HFAdapter,
    build_padded_exposure,
    build_source_blocks,
    byte_normalized_loss,
    document_stream_sha256,
    evaluate_gate_input,
    make_matched_config,
    parse_args,
    parse_csv_rows,
    write_jsonl_ledger,
)


def test_strict_csv_preserves_multiline_rows_and_rejects_bad_hash():
    text = 'text,category\n"two\nlines",a\n'
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rows, text_hash = parse_csv_rows(text, "train", expected_sha256=expected)
    assert rows == [{"text": "two\nlines", "category": "a"}]
    assert text_hash == expected
    with pytest.raises(ValueError, match="sha256 mismatch"):
        parse_csv_rows(text, "train", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="unexpected columns"):
        parse_csv_rows("text,other\nx,a\n", "train")
    with pytest.raises(ValueError, match="empty text"):
        parse_csv_rows("text,category\n,a\n", "train")


def test_document_stream_hash_is_ordered_and_length_prefixed():
    documents = [("Я",), ("a\nb",), ("😀",)]
    baseline = document_stream_sha256(documents)
    assert len(baseline) == 64
    assert baseline == document_stream_sha256(documents)
    assert baseline != document_stream_sha256(list(reversed(documents)))
    assert baseline != document_stream_sha256([("ab",)])


def test_fake_gigatoken_adapter_uses_verified_batch_and_decode_api():
    class Native:
        vocab_size = 262_144

        @staticmethod
        def encode_batch_list(texts):
            return [[1, 2] if text == "ok" else [3, 4] for text in texts]

        @staticmethod
        def decode(ids):
            return {2: "ok", 4: "no"}.get(ids[-1], "").encode("utf-8")

    adapter = GigaAdapter(Native, version="0.10.0", expected_sha256=None)
    assert adapter.encode_batch(["ok", "no"]) == [[1, 2], [3, 4]]
    assert adapter.decode([1, 2]) == "ok"


def test_hf_adapter_uses_no_special_tokens_and_verified_batch(monkeypatch):
    class Encoding:
        ids = [1, 2]
        special_tokens_mask = [0, 0]

    class Native:
        def get_vocab(self, with_added_tokens=True):
            assert with_added_tokens
            return {
                "a": 1,
                "b": 2,
                "<|im_end|>": 3,
                "<|endoftext|>": 4,
            }

        @staticmethod
        def encode_batch(texts, *, add_special_tokens):
            assert add_special_tokens is False
            assert texts == ["ok", "ok"]
            return [Encoding(), Encoding()]

        @staticmethod
        def decode(ids):
            return {2: "ok"}[ids[-1]]

    monkeypatch.setattr(subject, "EXPECTED_CANDIDATE_ACTIVE_VOCAB", 4)
    monkeypatch.setattr(subject, "EXPECTED_CANDIDATE_EOS", 3)
    monkeypatch.setattr(subject, "EXPECTED_CANDIDATE_PAD", 4)
    adapter = HFAdapter(
        Native(),
        tokenizer_path=Path("tokenizer.json"),
        config_path=Path("tokenizer_config.json"),
        version="0.23.2",
        expected_sha256=None,
        expected_config_sha256=None,
    )
    assert adapter.encode_batch(["ok", "ok"]) == [[1, 2], [1, 2]]
    assert adapter.decode([1, 2]) == "ok"
    assert adapter.eos_id == 3
    assert adapter.pad_id == 4


def test_padded_exposure_alignment_rejects_eos_and_truncation():
    docs = [(4, 5), (6, 7, 8)]
    batch = build_padded_exposure(docs, eos_id=99, pad_id=0, vocab_size=16)
    assert batch.input_ids.tolist() == [[99, 4, 0], [99, 6, 7]]
    assert batch.targets.tolist() == [[4, 5, 0], [6, 7, 8]]
    assert batch.loss_mask.tolist() == [[1, 1, 0], [1, 1, 1]]
    assert batch.lengths == (2, 3)
    with pytest.raises(ValueError, match="EOS"):
        build_padded_exposure([(4, 99)], eos_id=99, pad_id=0, vocab_size=16)
    with pytest.raises(ValueError, match="empty"):
        build_padded_exposure([()], eos_id=99, pad_id=0, vocab_size=16)
    with pytest.raises(ValueError, match="outside"):
        build_padded_exposure([(16,)], eos_id=99, pad_id=0, vocab_size=16)


def test_right_padding_matches_exact_hagi_hidden_states_and_total_ce():
    torch.manual_seed(7)
    cfg = make_matched_config(seed=1234, tokenizer_name="fixture", max_updates=1)
    cfg.model.vocab_size = 32
    cfg.model.hidden_size = 16
    cfg.model.num_layers = 2
    cfg.model.loop_depth = 1
    cfg.model.attention.num_query_heads = 2
    cfg.model.attention.num_kv_heads = 1
    cfg.model.attention.head_dim = 8
    cfg.model.attention.max_seq_len = 16
    cfg.model.attention.sink_len = 0
    cfg.model.sliding.window = 0
    cfg.model.embedding.init_std = 0.02
    cfg.model.ffn.intermediate_size = 32
    cfg.model.ffn.multiple_of = 16
    cfg.model.head.ce_chunk_rows = 4
    cfg.train.data.seq_len = 16
    cfg.train.data.eos_token_id = 31
    cfg.train.data.pad_token_id = 0
    subject.validate_config(cfg)
    model = subject.HAGI(cfg).eval()
    docs = [(3, 4, 5), (6, 7)]
    batch = build_padded_exposure(docs, eos_id=31, pad_id=0, vocab_size=32)
    with torch.no_grad():
        padded = model(batch.input_ids, batch.targets, loss_mask=batch.loss_mask)
        exact = [
            model(
                torch.tensor([[31, *row[:-1]]], dtype=torch.long),
                torch.tensor([row], dtype=torch.long),
            )
            for row in docs
        ]
    for index, row in enumerate(docs):
        torch.testing.assert_close(
            padded.hidden[index, : len(row)],
            exact[index].hidden[0, : len(row)],
            rtol=0.0,
            atol=1e-6,
        )
    assert padded.n_tokens == sum(exact_output.n_tokens for exact_output in exact)
    padded_total_nll = float(padded.ce) * padded.n_tokens
    exact_total_nll = sum(
        float(exact_output.ce) * exact_output.n_tokens for exact_output in exact
    )
    assert padded_total_nll == pytest.approx(exact_total_nll, rel=0.0, abs=1e-5)


def test_source_blocks_use_three_permutations_and_keep_last_19():
    rows = tuple(range(10_003))
    blocks = build_source_blocks(rows, seed=1234, epochs=3, block_size=64)
    assert len(blocks) == 471
    assert all(len(block) == 64 for index, block in enumerate(blocks) if index % 157 != 156)
    assert [len(blocks[index]) for index in (156, 313, 470)] == [19, 19, 19]
    epochs = [blocks[start : start + 157] for start in (0, 157, 314)]
    flattened = [
        [row for block in epoch for row in block]
        for epoch in epochs
    ]
    assert all(sorted(rows_for_epoch) == list(rows) for rows_for_epoch in flattened)
    assert epochs[0] != epochs[1] != epochs[2] != epochs[0]


def test_byte_normalized_loss_uses_native_nll_sum():
    output = type("Output", (), {"ce": torch.tensor(2.0), "n_tokens": 7})()
    assert float(byte_normalized_loss(output, block_bytes=14)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="count"):
        byte_normalized_loss(output, block_bytes=14, expected_tokens=6)
    with pytest.raises(ValueError, match="finite"):
        byte_normalized_loss(type("Output", (), {"ce": torch.tensor(float("nan")), "n_tokens": 7})(), 14)


def test_optimizer_partition_matches_production_semantics_and_is_not_fused():
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 3), nn.LayerNorm(4))
    model[0].is_channel_weight = True
    cfg = make_matched_config(seed=1234, tokenizer_name="fixture", max_updates=471)
    trainer = ByteNormalizedTrainer(model, cfg, max_steps=471)
    groups = trainer.optimizer.param_groups
    body = [p for group in groups if group.get("_body") for p in group["params"]]
    decay = [p for group in groups if not group.get("_body") and group["weight_decay"] for p in group["params"]]
    no_decay = [p for group in groups if group["weight_decay"] == 0 for p in group["params"]]
    assert body == [model[0].weight]
    assert decay == [model[1].weight]
    assert no_decay == [model[1].bias, *model[2].parameters()]
    assert len({id(p) for group in groups for p in group["params"]}) == len(list(model.parameters()))
    assert all(group.get("fused") is False for group in groups)


def test_optimizer_state_digest_is_recursive_and_strict():
    state = {"step": 1, "exp_avg": {"0": torch.tensor([1.0, 2.0])}, "step_tensor": torch.tensor(3)}
    digest = subject.state_digest(state)
    assert len(digest) == 64
    assert digest != subject.state_digest({**state, "step": 2})


def test_matched_config_has_exact_geometry_and_no_optional_objectives():
    cfg = make_matched_config(seed=1234, tokenizer_name="fixture", max_updates=471)
    assert cfg.model.vocab_size == 262_144
    assert cfg.model.hidden_size == 128
    assert cfg.model.num_layers == 4
    assert cfg.model.attention.num_query_heads == 4
    assert cfg.model.attention.num_kv_heads == 2
    assert cfg.model.attention.head_dim == 32
    assert cfg.model.attention.max_seq_len == 128
    assert cfg.model.ffn.intermediate_size == 256
    assert cfg.model.embedding.conv_kernel == 4
    assert cfg.model.embedding.tie_lm_head is True
    assert cfg.model.ternary.enabled is True
    assert cfg.model.head.unigram_prior is False
    assert cfg.model.head.sampled_softmax_k == 0
    assert cfg.train.precision == "fp32"
    assert cfg.train.use_muon is False
    assert cfg.train.compile_model is False


def test_gate_boundaries_and_reduced_runs_fail_closed():
    deltas = [0.01, 0.01, 0.01]
    report = evaluate_gate_input(
        seeds=[1234, 2243, 3252],
        updates_per_seed=[471] * 3,
        bpb_improvements=deltas,
        timing_ratios=[1.25, 1.25, 1.25],
        memory_ratios=[1.25, 1.25, 1.25],
        roundtrip_exact=True,
        runs_finite=True,
        matched_configs=True,
        provenance_stable=True,
        deterministic=True,
    )
    assert report["status"] == "eligible-for-next-slice"
    assert report["quality_supported"] is True
    assert report["systems_supported"] is True
    assert report["execution_valid"] is True
    assert report["checks"]["current_runs_finite"] is True
    assert report["checks"]["six_complete_finite_runs"] is True
    one_seed_regression = evaluate_gate_input(
        seeds=[1234, 2243, 3252],
        updates_per_seed=[471] * 3,
        bpb_improvements=[-0.001, 0.02, 0.02],
        timing_ratios=[1.0] * 3,
        memory_ratios=[1.0] * 3,
        roundtrip_exact=True,
        runs_finite=True,
        matched_configs=True,
        provenance_stable=True,
        deterministic=True,
    )
    assert one_seed_regression["execution_valid"] is True
    assert one_seed_regression["checks"]["bpb_all_seeds_positive"] is False
    assert one_seed_regression["quality_supported"] is False
    systems_failure = evaluate_gate_input(
        seeds=[1234, 2243, 3252],
        updates_per_seed=[471] * 3,
        bpb_improvements=deltas,
        timing_ratios=[1.251, 1.0, 1.0],
        memory_ratios=[1.0] * 3,
        roundtrip_exact=True,
        runs_finite=True,
        matched_configs=True,
        provenance_stable=True,
        deterministic=True,
    )
    assert systems_failure["quality_supported"] is True
    assert systems_failure["systems_supported"] is False
    assert systems_failure["status"] == "research-only"
    for invalid_name, invalid_vector in (
        ("zero timing", ([0.0, 1.0, 1.0], [1.0, 1.0, 1.0])),
        ("negative memory", ([1.0, 1.0, 1.0], [-1.0, 1.0, 1.0])),
    ):
        timing, memory = invalid_vector
        invalid_systems = evaluate_gate_input(
            seeds=[1234, 2243, 3252],
            updates_per_seed=[471] * 3,
            bpb_improvements=deltas,
            timing_ratios=timing,
            memory_ratios=memory,
            roundtrip_exact=True,
            runs_finite=True,
            matched_configs=True,
            provenance_stable=True,
            deterministic=True,
        )
        assert invalid_systems["execution_valid"] is True, invalid_name
        assert invalid_systems["checks"]["systems_ratios_positive"] is False
        assert invalid_systems["systems_supported"] is False
    report = evaluate_gate_input(
        seeds=[1234, 2243, 3252],
        updates_per_seed=[471] * 3,
        bpb_improvements=[0.009999, 0.009999, 0.009999],
        timing_ratios=[1.25, 1.25, 1.25],
        memory_ratios=[1.25, 1.25, 1.25],
        roundtrip_exact=True,
        runs_finite=True,
        matched_configs=True,
        provenance_stable=True,
        deterministic=True,
    )
    assert report["status"] == "research-only"
    assert report["quality_supported"] is False
    assert report["execution_valid"] is True
    incomplete_vectors = evaluate_gate_input(
        seeds=[1234, 2243, 3252],
        updates_per_seed=[471] * 3,
        bpb_improvements=[0.02, 0.02, 0.02],
        timing_ratios=[1.0, 1.0],
        memory_ratios=[1.0, 1.0, 1.0],
        roundtrip_exact=True,
        runs_finite=True,
        matched_configs=True,
        provenance_stable=True,
        deterministic=True,
    )
    assert incomplete_vectors["execution_valid"] is False
    assert incomplete_vectors["checks"]["vector_shapes_valid"] is False
    smoke_metadata = subject._protocol_metadata(
        requested_seeds=[1234],
        updates_per_seed=1,
        full_protocol_executed=False,
    )
    assert smoke_metadata == {
        "requested_protocol": {"seeds": [1234], "updates_per_seed": 1},
        "target_protocol": {
            "seeds": [1234, 2243, 3252],
            "updates_per_seed": 471,
        },
        "full_protocol_executed": False,
    }
    report = evaluate_gate_input(
        seeds=[1234],
        updates_per_seed=[1],
        bpb_improvements=[1.0],
        timing_ratios=[0.1],
        memory_ratios=[0.1],
        roundtrip_exact=True,
        runs_finite=True,
        matched_configs=True,
        provenance_stable=True,
        deterministic=True,
    )
    assert report["status"] == "research-only"
    assert report["execution_valid"] is False
    assert report["checks"]["current_runs_finite"] is True
    assert report["checks"]["six_complete_finite_runs"] is False


def test_json_finiteness_and_ledger_hash_are_strict(tmp_path):
    path = tmp_path / "ledger.jsonl"
    summary = write_jsonl_ledger(path, [{"step": 0, "value": 1.25}, {"step": 1, "value": 2.0}])
    assert summary == {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "record_count": 2,
    }
    with pytest.raises(ValueError, match="finite JSON"):
        write_jsonl_ledger(tmp_path / "bad.jsonl", [{"value": float("nan")}])


def test_ledger_failure_preserves_existing_target_and_removes_temporary(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("existing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite JSON"):
        write_jsonl_ledger(path, [{"value": float("inf")}])
    assert path.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_byte_normalized_trainer_applies_update_without_global_clipping(monkeypatch):
    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 4)
            self.output = nn.Linear(4, 16, bias=False)

        def forward(self, input_ids, targets, loss_mask):
            logits = self.output(self.embedding(input_ids))
            losses = nn.functional.cross_entropy(
                logits.reshape(-1, 16), targets.reshape(-1), reduction="none"
            )
            valid = loss_mask.reshape(-1)
            return type(
                "Output",
                (),
                {
                    "ce": losses[valid].mean(),
                    "n_tokens": int(valid.sum().item()),
                },
            )()

    clip_calls = []

    def fail_if_clipped(*args, **kwargs):
        clip_calls.append((args, kwargs))
        raise AssertionError("global gradient clipping is forbidden")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fail_if_clipped)
    model = TinyLM()
    cfg = make_matched_config(seed=1234, tokenizer_name="fixture", max_updates=1)
    trainer = ByteNormalizedTrainer(model, cfg, max_steps=1)
    exposure = build_padded_exposure(
        [(1, 2), (3, 4)], eos_id=0, pad_id=15, vocab_size=16
    )
    record = trainer.train_step(exposure, block_bytes=4)
    assert record["update_applied"] is True
    assert record["post_update_state_finite"] is True
    assert record["metrics_finite"] is True
    assert record["gradients_finite"] is True
    assert clip_calls == []


def test_no_replace_publication_never_overwrites_existing_target(tmp_path):
    temporary = tmp_path / ".result.tmp"
    target = tmp_path / "result.json"
    temporary.write_bytes(b"complete-report")
    subject._publish_file_no_replace(temporary, target)
    assert target.read_bytes() == b"complete-report"
    assert not temporary.exists()

    competing = tmp_path / ".competing.tmp"
    competing.write_bytes(b"competing-report")
    with pytest.raises(FileExistsError, match="already exists"):
        subject._publish_file_no_replace(competing, target)
    assert target.read_bytes() == b"complete-report"


def test_reservation_rolls_back_on_base_exception(tmp_path, monkeypatch):
    run_path = tmp_path / "run"
    output_path = run_path / "report.json"

    def interrupt_fsync(_descriptor):
        raise KeyboardInterrupt("synthetic reservation interrupt")

    monkeypatch.setattr(subject.os, "fsync", interrupt_fsync)
    with pytest.raises(KeyboardInterrupt, match="reservation interrupt"):
        subject.prepare_run_paths(run_path, output_path)
    assert not run_path.exists()
    assert not list(tmp_path.glob(".*.lock"))


def test_failed_run_preserves_explicit_terminal_failure_without_report(tmp_path, monkeypatch):
    run_path = tmp_path / "run"
    output_path = run_path / "report.json"

    def fail_run(_args):
        raise RuntimeError("synthetic runner failure")

    monkeypatch.setattr(subject, "run_experiment", fail_run)
    with pytest.raises(RuntimeError, match="synthetic runner failure"):
        subject.main([
            "--artifact", "artifact",
            "--baseline-tokenizer", "baseline",
            "--candidate-tokenizer", "candidate",
            "--seeds", "1234",
            "--max-updates-per-seed", "1",
            "--device", "cpu",
            "--run", str(run_path),
            "--output", str(output_path),
        ])
    assert not output_path.exists()
    failure = json.loads((run_path / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed-before-report"
    assert failure["quality_supported"] is False
    assert failure["systems_supported"] is False
    assert failure["production_promotion"] is False
    assert failure["error_type"] == "RuntimeError"
    assert failure["error_message"] == "synthetic runner failure"
    assert not (run_path / f".{output_path.name}.lock").exists()
    assert run_path.is_dir()


def test_success_calls_runner_once_and_interrupt_after_link_is_not_failure(tmp_path, monkeypatch):
    run_path = tmp_path / "run"
    output_path = run_path / "report.json"
    calls = []
    report = {"execution_valid": True, "quality_supported": False, "value": 1}

    def successful_run(_args):
        calls.append("run")
        return report

    original_write_report = subject.write_report

    def interrupt_after_link(path, payload):
        original_write_report(path, payload)
        calls.append("link")
        raise KeyboardInterrupt("synthetic post-link interrupt")

    monkeypatch.setattr(subject, "run_experiment", successful_run)
    monkeypatch.setattr(subject, "write_report", interrupt_after_link)
    exit_code = subject.main([
        "--artifact", "artifact",
        "--baseline-tokenizer", "baseline",
        "--candidate-tokenizer", "candidate",
        "--seeds", "1234",
        "--max-updates-per-seed", "1",
        "--device", "cpu",
        "--run", str(run_path),
        "--output", str(output_path),
    ])
    assert exit_code == 0
    assert calls == ["run", "link"]
    assert output_path.read_bytes() == subject._report_payload(report)
    assert not (run_path / "failure.json").exists()
    assert not (run_path / f".{output_path.name}.lock").exists()


def test_prepare_run_paths_exclusively_reserves_nested_report(tmp_path):
    run_path = tmp_path / "run"
    output_path = run_path / "result.json"
    prepared_run, prepared_output, lock_path = subject.prepare_run_paths(
        run_path, output_path
    )
    assert prepared_run == run_path
    assert prepared_output == output_path
    assert lock_path.is_file()
    subject.assert_output_reserved(output_path, lock_path, run_path)
    with pytest.raises(FileExistsError, match="not exclusively reservable"):
        subject.prepare_run_paths(run_path, output_path)

    output_path.write_text("appeared", encoding="utf-8")
    with pytest.raises(FileExistsError, match="appeared during run"):
        subject.assert_output_reserved(output_path, lock_path, run_path)
    output_path.unlink()
    lock_path.unlink()
    with pytest.raises(FileNotFoundError, match="lock is missing"):
        subject.assert_output_reserved(output_path, lock_path, run_path)

    with pytest.raises(ValueError, match="direct child"):
        subject.prepare_run_paths(tmp_path / "other-run", tmp_path / "outside.json")
    outside = tmp_path / "outside.json"
    outside.write_text("existing", encoding="utf-8")
    existing_run = tmp_path / "existing-run"
    existing_run.mkdir()
    with pytest.raises(FileExistsError, match="not exclusively reservable"):
        subject.prepare_run_paths(existing_run, existing_run / "result.json")


def test_cli_defaults_to_exact_protocol_and_has_no_hash_override():
    args = parse_args([
        "--artifact", "artifact",
        "--baseline-tokenizer", "baseline-tokenizer",
        "--candidate-tokenizer", "candidate-tokenizer",
        "--run", "run",
        "--output", "out.json",
        "--device", "cpu",
    ])
    assert args.baseline_tokenizer == Path("baseline-tokenizer")
    assert args.candidate_tokenizer == Path("candidate-tokenizer")
    assert args.seeds == [1234, 2243, 3252]
    assert args.max_updates_per_seed == 471
    assert not hasattr(args, "expected_manifest_sha256")
    assert not hasattr(args, "revision")
    assert not hasattr(args, "model_id")
    with pytest.raises(SystemExit):
        parse_args([
            "--artifact", "artifact",
            "--baseline-tokenizer", "baseline-tokenizer",
            "--run", "run",
            "--output", "out.json",
            "--device", "cpu",
        ])
    assert EXPECTED_MANIFEST_SHA256.startswith("9927589c")
    assert EXPECTED_CANDIDATE_TOKENIZER_SHA256.startswith("0997f410")
