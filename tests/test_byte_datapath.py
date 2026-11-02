# Byte-level datapath: ByteHNet input_mode with the uint8 corpus stream
# (2026-11-02). Corpus: raw UTF-8 bytes, eos = 0x0A, vocab 256, no tokenizer.

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.data.dataset import (  # noqa: E402
    PackedStream,
    build_dataloader,
    dataset_path,
)


def _byte_corpus(tmp_path: Path, n_bytes: int = 20_000) -> Path:
    rng = np.random.default_rng(0)
    # readable-ASCII text with newlines: exercises eos doc boundaries
    words = [b"alpha", b"beta", b"gamma", b"delta"]
    buf = bytearray()
    while len(buf) < n_bytes:
        buf += b" ".join(rng.choice(words, size=8).tolist()) + b"\n"
    d = tmp_path / "data_bytes"
    d.mkdir(exist_ok=True)
    (d / "text_main.bin").write_bytes(bytes(buf[:n_bytes]))
    import json

    (d / "mix.json").write_text(
        json.dumps({"sources": [{"name": "text_main", "ratio": 1.0}]})
    )
    return d


def test_byte_stream_windows_are_raw_bytes(tmp_path):
    d = _byte_corpus(tmp_path)
    rng = np.random.default_rng(1)
    s = PackedStream(dataset_path(d, "text_main"), 64, 10, rng, token_dtype=np.uint8)
    w = s.next_window()
    assert w is not None and len(w) == 65
    assert w.dtype == np.int64
    assert w.max() < 256 and w.min() >= 0


def test_byte_dataloader_eos_and_shapes(tmp_path):
    import dataclasses

    d = _byte_corpus(tmp_path)

    @dataclasses.dataclass
    class Data:
        data_dir: str = ""
        seq_len: int = 64
        eos_token_id: int = 0
        seed: int = 7
        num_workers: int = 0
        cross_doc_attention: bool = False
        weights: dict | None = None

    @dataclasses.dataclass
    class Train:
        data: Data
        batch_size: int = 4

    @dataclasses.dataclass
    class Model:
        input_mode: str = "bytes_hnet"

    @dataclasses.dataclass
    class Cfg:
        model: Model
        train: Train

    cfg = Cfg(model=Model(), train=Train(data=Data(
        # byte mode appends _bytes to the dir
        data_dir=str(d).replace("_bytes", ""),
        eos_token_id=0,  # must be overridden to 10 in byte mode
    )))
    dl = build_dataloader(cfg)
    batch = next(iter(dl))
    assert batch["input_ids"].shape == (4, 64)
    assert batch["targets"].shape == (4, 64)
    assert batch["input_ids"].max() < 256
    assert "doc_ids" in batch
