"""The corpus guard must fail closed, not merely warn.

Reproduces the exact failure mode the project hit: a mix that resolves to a
raw ``.bin`` stream (source vocabulary, 248320 ids) instead of a compacted
one, which trains a 32768-vocabulary model on out-of-range embedding indices.
The old ``dataset_path`` returned that stream silently.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from hagi.data.dataset import _assert_ids_in_vocabulary, dataset_path  # noqa: E402

VOCAB = 32768


def test_raw_stream_is_rejected_by_the_training_guard(tmp_path: Path):
    raw = tmp_path / "bad.bin"
    raw.write_bytes(np.array([0, 1, 2, 255_998], dtype=np.uint32).tobytes())
    with pytest.raises(ValueError, match="outside the model"):
        _assert_ids_in_vocabulary(tmp_path, {"bad": 1.0}, VOCAB)


def test_compacted_stream_is_accepted(tmp_path: Path):
    ok = tmp_path / "good.compact.bin"
    ok.write_bytes(np.array([0, 5, VOCAB - 1], dtype=np.uint32).tobytes())
    _assert_ids_in_vocabulary(tmp_path, {"good": 1.0}, VOCAB)  # must not raise


def test_require_compacted_refuses_the_raw_fallback(tmp_path: Path):
    (tmp_path / "only_raw.bin").write_bytes(np.zeros(4, dtype=np.uint32).tobytes())
    # Default behaviour is unchanged: legacy callers still get the raw stream.
    assert dataset_path(tmp_path, "only_raw").name == "only_raw.bin"
    with pytest.raises(FileNotFoundError, match="no compacted stream"):
        dataset_path(tmp_path, "only_raw", require_compacted=True)


def test_empty_corpus_fails_closed(tmp_path: Path):
    (tmp_path / "void.compact.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="empty corpus"):
        _assert_ids_in_vocabulary(tmp_path, {"void": 1.0}, VOCAB)


def test_cli_exits_nonzero_on_out_of_range(tmp_path: Path):
    (tmp_path / "corrupt.bin").write_bytes(
        np.array([0, 99_999], dtype=np.uint32).tobytes()
    )
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "check_corpus_ids.py"),
         "--data-dir", str(tmp_path), "--vocab-size", str(VOCAB)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAIL" in proc.stderr
