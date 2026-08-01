"""Packed data pipeline: document ids, window continuity, mix weights.

``doc_ids`` is the only thing standing between packed batches and cross-document
attention leakage, and it is computed with a cumsum whose off-by-one is invisible
in any shape check. So the derivation is asserted position by position.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from hagi.data.dataset import PackedMixDataset, PackedStream, dataset_path, load_mix


def write_bin(path, tokens):
    np.asarray(tokens, dtype=np.uint32).tofile(path)


@pytest.fixture
def corpus(tmp_path):
    """Two sources plus a mix.json, one source deliberately absent from disk."""
    write_bin(tmp_path / "a.bin", list(range(1, 201)))
    write_bin(tmp_path / "b.bin", list(range(500, 700)))
    (tmp_path / "mix.json").write_text(
        json.dumps(
            {
                "sources": [
                    {"name": "a", "ratio": 0.75},
                    {"name": "b", "ratio": 0.25},
                    {"name": "missing", "ratio": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


class TestPaths:
    def test_resolves_bin(self, tmp_path):
        assert dataset_path(tmp_path, "edu").name == "edu.bin"

    def test_prefers_compact_bin(self, tmp_path):
        write_bin(tmp_path / "edu.bin", [1, 2, 3])
        write_bin(tmp_path / "edu.compact.bin", [1, 2, 3])
        assert dataset_path(tmp_path, "edu").name == "edu.compact.bin"

    def test_falls_back_to_raw_when_no_compact(self, tmp_path):
        write_bin(tmp_path / "edu.bin", [1, 2, 3])
        assert dataset_path(tmp_path, "edu").name == "edu.bin"

    @pytest.mark.parametrize("name", ["", "../escape", "sub/dir", "back\\slash"])
    def test_rejects_traversal(self, tmp_path, name):
        with pytest.raises(ValueError):
            dataset_path(tmp_path, name)


class TestLoadMix:
    def test_normalizes_and_drops_missing(self, corpus):
        weights = load_mix(corpus)
        assert set(weights) == {"a", "b"}
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["a"] > weights["b"]

    def test_overrides_replace_the_file(self, corpus):
        weights = load_mix(corpus, {"b": 1.0})
        assert weights == {"b": 1.0}

    def test_zero_weight_source_dropped(self, corpus):
        assert set(load_mix(corpus, {"a": 1.0, "b": 0.0})) == {"a"}

    def test_no_usable_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mix(tmp_path, {"nothing": 1.0})


class TestPackedStream:
    def test_window_is_seq_len_plus_one(self, corpus):
        stream = PackedStream(corpus / "a.bin", 8, 1, np.random.default_rng(0))
        assert stream.next_window().shape == (9,)

    def test_windows_advance_by_seq_len(self, corpus):
        """The extra token is the last target, so windows overlap by exactly one."""
        stream = PackedStream(corpus / "a.bin", 8, 1, np.random.default_rng(0))
        first = stream.next_window()
        second = stream.next_window()
        assert int(first[-1]) == int(second[0])

    def test_wraps_at_end_of_file(self, corpus):
        stream = PackedStream(corpus / "a.bin", 8, 1, np.random.default_rng(0))
        for _ in range(60):
            window = stream.next_window()
            assert window.shape == (9,)

    def test_too_short_file_raises(self, tmp_path):
        write_bin(tmp_path / "tiny.bin", [1, 2, 3])
        with pytest.raises(ValueError):
            PackedStream(tmp_path / "tiny.bin", 8, 1, np.random.default_rng(0))


class TestDocIds:
    def take(self, dataset, n=1):
        it = iter(dataset)
        return [next(it) for _ in range(n)]

    def test_doc_id_increments_after_each_eos(self, tmp_path):
        # EOS = 1; the stream is deliberately laid out so boundaries are known.
        write_bin(tmp_path / "s.bin", [5, 6, 1, 7, 8, 1, 9, 9, 9, 1] * 8)
        dataset = PackedMixDataset(str(tmp_path), seq_len=10, eos_token_id=1, weights={"s": 1.0})
        item = self.take(dataset)[0]
        ids = item["input_ids"]
        docs = item["doc_ids"]
        expected = torch.cumsum((ids == 1).long(), 0) - (ids == 1).long()
        assert torch.equal(docs, expected)

    def test_eos_belongs_to_the_document_it_terminates(self, tmp_path):
        write_bin(tmp_path / "s.bin", [5, 1, 7, 1] * 20)
        dataset = PackedMixDataset(str(tmp_path), seq_len=8, eos_token_id=1, weights={"s": 1.0})
        item = self.take(dataset)[0]
        ids, docs = item["input_ids"], item["doc_ids"]
        for pos in range(len(ids)):
            if int(ids[pos]) == 1 and pos > 0:
                assert int(docs[pos]) == int(docs[pos - 1]), "EOS must close its own document"

    def test_doc_ids_are_non_decreasing(self, tmp_path):
        write_bin(tmp_path / "s.bin", list(range(1, 400)))
        dataset = PackedMixDataset(str(tmp_path), seq_len=32, eos_token_id=1, weights={"s": 1.0})
        docs = self.take(dataset)[0]["doc_ids"]
        assert bool((docs[1:] >= docs[:-1]).all())

    def test_cross_doc_attention_omits_doc_ids(self, tmp_path):
        write_bin(tmp_path / "s.bin", list(range(1, 200)))
        dataset = PackedMixDataset(
            str(tmp_path), seq_len=8, eos_token_id=1, weights={"s": 1.0}, cross_doc_attention=True
        )
        assert "doc_ids" not in self.take(dataset)[0]


class TestTargets:
    def test_targets_are_inputs_shifted_by_one(self, tmp_path):
        write_bin(tmp_path / "s.bin", list(range(100, 300)))
        dataset = PackedMixDataset(str(tmp_path), seq_len=16, eos_token_id=1, weights={"s": 1.0})
        item = next(iter(dataset))
        assert torch.equal(item["input_ids"][1:], item["targets"][:-1])

    def test_no_padding_is_ever_emitted(self, tmp_path):
        """Packing's entire point: every scored position carries a real token."""
        write_bin(tmp_path / "s.bin", list(range(100, 400)))
        dataset = PackedMixDataset(str(tmp_path), seq_len=16, eos_token_id=1, weights={"s": 1.0})
        it = iter(dataset)
        for _ in range(20):
            item = next(it)
            assert int(item["input_ids"].min()) > 0

    def test_dtype_is_int64(self, tmp_path):
        write_bin(tmp_path / "s.bin", list(range(1, 200)))
        dataset = PackedMixDataset(str(tmp_path), seq_len=8, eos_token_id=1, weights={"s": 1.0})
        item = next(iter(dataset))
        assert item["input_ids"].dtype == torch.int64
        assert item["targets"].dtype == torch.int64


class TestMixing:
    def test_both_sources_appear(self, corpus):
        """Proportional interleaving, not sequential cycling: the V30 log showed a
        step change in ce at every dataset boundary under sequential cycling."""
        dataset = PackedMixDataset(
            str(corpus), seq_len=8, eos_token_id=1, weights={"a": 0.5, "b": 0.5}, seed=3
        )
        it = iter(dataset)
        seen_low = seen_high = False
        for _ in range(60):
            first = int(next(it)["input_ids"][0])
            if first < 300:
                seen_low = True
            else:
                seen_high = True
        assert seen_low and seen_high, "one source never appeared in 60 windows"

    def test_seed_is_reproducible(self, corpus):
        def first_windows(seed):
            dataset = PackedMixDataset(
                str(corpus), seq_len=8, eos_token_id=1, weights={"a": 0.5, "b": 0.5}, seed=seed
            )
            it = iter(dataset)
            return [next(it)["input_ids"].tolist() for _ in range(5)]

        assert first_windows(11) == first_windows(11)
        assert first_windows(11) != first_windows(12)
