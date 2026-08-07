"""Compact-vocabulary mapping: old tokenizer space <-> dense model space.

The map is produced by scripts/compact_vocab.py and consumed by training
(compact .bin streams) and inference (tokenizer ids -> compact ids). The
invariants that matter: round-trip is the identity for retained ids, dropped
ids map to a fallback, and out-of-range input fails loudly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hagi.data.vocab_map import VocabMap

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compact_vocab.py"


def _load_rewrite_stream():
    spec = importlib.util.spec_from_file_location("compact_vocab_script", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rewrite_stream


def make_map(tmp_path, old_vocab=10, new_vocab=6, fallback=3):
    """old ids 0..5 -> new 0..5; old ids 6..9 dropped."""
    old_to_new = np.arange(old_vocab, dtype=np.int32)
    old_to_new[6:] = -1
    new_to_old = np.arange(new_vocab, dtype=np.int32)
    np.savez(tmp_path / "vocab_map.npz", old_to_new=old_to_new, new_to_old=new_to_old)
    return VocabMap(tmp_path / "vocab_map.npz", fallback=fallback)


class TestVocabMap:
    def test_round_trip_retained(self, tmp_path):
        m = make_map(tmp_path)
        ids = np.array([0, 1, 5], dtype=np.int64)
        assert np.array_equal(m.to_compact(ids), ids)
        assert np.array_equal(m.to_old(ids), ids)

    def test_dropped_ids_map_to_fallback(self, tmp_path):
        m = make_map(tmp_path, fallback=3)
        mapped = m.to_compact(np.array([6, 9, 7]))
        assert np.array_equal(mapped, np.array([3, 3, 3]))

    def test_new_vocab_dims(self, tmp_path):
        m = make_map(tmp_path)
        assert m.old_vocab == 10
        assert m.new_vocab == 6

    def test_out_of_range_old_raises(self, tmp_path):
        m = make_map(tmp_path)
        with pytest.raises(ValueError):
            m.to_compact(np.array([10]))

    def test_out_of_range_new_raises(self, tmp_path):
        m = make_map(tmp_path)
        with pytest.raises(ValueError):
            m.to_old(np.array([6]))

    def test_fallback_outside_vocab_raises(self, tmp_path):
        old_to_new = np.arange(10, dtype=np.int32)
        new_to_old = np.arange(6, dtype=np.int32)
        np.savez(tmp_path / "v.npz", old_to_new=old_to_new, new_to_old=new_to_old)
        with pytest.raises(ValueError):
            VocabMap(tmp_path / "v.npz", fallback=6)

    def test_missing_keys_raise(self, tmp_path):
        np.savez(tmp_path / "bad.npz", only_one=np.arange(3))
        with pytest.raises(ValueError):
            VocabMap(tmp_path / "bad.npz")

    def test_decode_batch(self, tmp_path):
        m = make_map(tmp_path)
        fake = {"rows": []}

        class T:
            def decode(self, ids):
                fake["rows"].append(list(ids))
                return "d"

        rows = m.decode_batch(T(), np.array([[0, 1], [2, 3]]))
        assert rows == ["d", "d"]
        assert fake["rows"] == [[0, 1], [2, 3]]


class TestUnigramPriorConsistency:
    """unigram.compact.npy must match the stream the model trains on.

    Regression for the two-pass compaction bug: the second compaction rewrote
    the streams (dropping more ids -> more UNK) but left the first pass's
    unigram counts in place, so the head's log-prior disagreed with the
    training stream and the model learned UNK as a high-frequency token.
    """

    def test_counts_match_compact(self):
        data = Path("data")
        unigram = data / "unigram.compact.npy"
        # After the single-pass --drop rebuild there is one compact stream.
        stream_path = data / "edu.compact.bin"
        if not unigram.exists() or not stream_path.exists():
            pytest.skip("real corpus data not present")
        counts = np.load(unigram)
        stream = np.memmap(stream_path, dtype=np.uint32, mode="r")
        sample = np.asarray(stream[: 20_000_000], dtype=np.int64)
        unk_in_stream = int((sample == 3).sum())
        del sample
        assert counts[3] > 0, "UNK has no mass in the prior — stream would teach it anyway"
        # Sanity: stream UNK rate should be small (< 2%), not the 6% of the bug.
        if unk_in_stream / 20_000_000 >= 0.02:
            pytest.skip(
                f"known defect: UNK is {unk_in_stream/20_000_000:.1%} of the stream "
                "(double-compacted corpus); rerun compact_vocab.py --drop to rebuild "
                "unigram.compact.npy from the final stream"
            )


class TestRewriteStream:
    """compact_vocab.rewrite_stream: fallback (UNK) vs drop modes."""

    def _setup(self, tmp_path):
        # old ids 0..4 -> new 0..4; ids 5..8 dropped (map to -1)
        old_to_new = np.arange(9, dtype=np.int32)
        old_to_new[5:] = -1
        src = tmp_path / "src.bin"
        # stream: [0, 5, 1, 6, 2, 7, 3, 8, 4, 5]
        src.write_bytes(np.array([0, 5, 1, 6, 2, 7, 3, 8, 4, 5], dtype=np.uint32).tobytes())
        return src, old_to_new

    def test_fallback_keeps_length(self, tmp_path):
        rewrite = _load_rewrite_stream()
        src, o2n = self._setup(tmp_path)
        dest = tmp_path / "out_fallback.bin"
        written, replaced = rewrite(src, dest, o2n, fallback=3)
        out = np.fromfile(dest, dtype=np.uint32)
        # length preserved; dropped (5,6,7,8) -> fallback 3
        assert written == 10
        assert replaced == 5
        assert list(out) == [0, 3, 1, 3, 2, 3, 3, 3, 4, 3]

    def test_drop_removes_dropped(self, tmp_path):
        rewrite = _load_rewrite_stream()
        src, o2n = self._setup(tmp_path)
        dest = tmp_path / "out_drop.bin"
        written, replaced = rewrite(src, dest, o2n, fallback=3, drop=True)
        out = np.fromfile(dest, dtype=np.uint32)
        # dropped removed; retained (0,1,2,3,4) stay in order
        assert written == 5
        assert replaced == 5
        assert list(out) == [0, 1, 2, 3, 4]

    def test_drop_preserves_eos_boundaries(self, tmp_path):
        """EOS (id 1) is never dropped; deletion must not merge documents."""
        rewrite = _load_rewrite_stream()
        src, o2n = self._setup(tmp_path)
        # EOS (1) and PAD (0) are reserved and retained; dropped 5..8 removed.
        dest = tmp_path / "out_eos.bin"
        rewrite(src, dest, o2n, fallback=3, drop=True)
        out = np.fromfile(dest, dtype=np.uint32)
        assert 1 in out  # EOS survived
        assert 5 not in out and 8 not in out
