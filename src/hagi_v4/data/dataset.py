"""Memmap dataset for HAGI V4 — reads .bin files with uint16 tokens."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    """Binary memmap dataset. Token IDs stored as uint16 or uint32.

    Files: e:\\HAGI_v2\\data\\*.bin
    Format: flat array, no headers. dtype detected from vocab_size:
      vocab <= 65535 → uint16 (2 bytes)
      vocab >  65535 → uint32 (4 bytes)
    """

    def __init__(self, path: str, seq_len: int = 512, vocab_size: int = 49154, dtype: str = "auto"):
        self.path = Path(path)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        if dtype == "auto":
            self._dtype = torch.uint16 if vocab_size <= 65535 else torch.uint32
            self._byte_width = 2 if vocab_size <= 65535 else 4
        elif dtype == "uint32":
            self._dtype = torch.uint32
            self._byte_width = 4
        else:
            self._dtype = torch.uint16
            self._byte_width = 2
        file_size = self.path.stat().st_size
        self.num_tokens = file_size // self._byte_width
        self._data = None

    def _load(self):
        if self._data is None:
            with open(self.path, "rb") as f:
                self._data = torch.frombuffer(f.read(), dtype=self._dtype).long().clone()

    def __len__(self) -> int:
        return max(0, self.num_tokens - self.seq_len - 1)

    def __getitem__(self, idx: int) -> dict:
        self._load()
        chunk = self._data[idx : idx + self.seq_len + 1]
        ids = chunk[: self.seq_len].clone()
        tgt = chunk[: self.seq_len].clone()
        ids[ids >= self.vocab_size] = 0
        tgt[tgt >= self.vocab_size] = 0
        return {"input_ids": ids, "targets": tgt}


class MixedDataset(Dataset):
    """Mix multiple .bin datasets according to ratios from mix.json.

    Samples from each dataset proportionally, drawing random starting
    positions from each source.
    """

    def __init__(
        self,
        data_dir: str,
        mix_path: str | None = None,
        seq_len: int = 512,
        vocab_size: int = 49154,
        num_samples: int = 100000,
    ):
        import json

        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples

        if mix_path is None:
            mix_path = self.data_dir / "mix.json"
        with open(mix_path) as f:
            mix = json.load(f)

        self.sources: list[tuple[str, float, MemmapDataset]] = []
        total_ratio = 0.0
        for src in mix["sources"]:
            name = src["name"]
            ratio = src["ratio"]
            path = self.data_dir / f"{name}.bin"
            if path.exists():
                ds = MemmapDataset(str(path), seq_len, vocab_size)
                if len(ds) > 0:
                    self.sources.append((name, ratio, ds))
                    total_ratio += ratio

        self.sources = [(n, r / total_ratio, d) for n, r, d in self.sources]
        self._cumulative = []
        acc = 0.0
        for n, r, d in self.sources:
            acc += r
            self._cumulative.append((acc, n, d))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        import random

        r = random.random()
        for threshold, name, ds in self._cumulative:
            if r < threshold:
                pos = random.randint(0, max(0, len(ds) - 1))
                item = ds[pos]
                return item
        return self._cumulative[-1][2][0]


def build_dataloader(
    data_dir: str = "data",
    mix_path: str | None = None,
    batch_size: int = 8,
    seq_len: int = 512,
    vocab_size: int = 49154,
    num_samples: int = 100000,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader from .bin files in data_dir."""
    dataset = MixedDataset(data_dir, mix_path, seq_len, vocab_size, num_samples)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
