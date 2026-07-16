"""Memmap dataset for HAGI V4 — reads .bin files with uint16 tokens."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def dataset_path(data_dir: str | Path, name: str) -> Path:
    if not isinstance(name, str) or not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
        raise ValueError(f"invalid dataset name: {name!r}")
    return Path(data_dir) / f"{name}.bin"


def validate_terminal_eos(
    input_ids: torch.Tensor,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Return valid targets through exactly one terminal EOS, excluding padding."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B,T]")
    if eos_token_id == pad_token_id:
        raise ValueError("eos_token_id and pad_token_id must be distinct")
    eos = input_ids.eq(eos_token_id)
    if not eos.sum(dim=1).eq(1).all():
        raise ValueError("each row must contain exactly one terminal EOS")
    eos_index = eos.to(torch.int64).argmax(dim=1)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    valid = positions <= eos_index.unsqueeze(1)
    if (input_ids.eq(pad_token_id) & valid).any() or ((input_ids != pad_token_id) & ~valid).any():
        raise ValueError("each row must contain exactly one terminal EOS followed only by padding")
    return valid


class MemmapDataset(Dataset):
    """Binary token dataset. Token IDs stored as uint16 or uint32.

    Files: <data_dir>/*.bin
    Format: flat array, no headers. dtype detected from vocab_size:
      vocab <= 65535 -> uint16 (2 bytes)
      vocab >  65535 -> uint32 (4 bytes)

    With EOS metadata, samples are indexed as EOS-delimited records. Long
    records are split into deterministic non-overlapping content chunks.
    """

    def __init__(
        self,
        path: str,
        seq_len: int = 512,
        vocab_size: int = 49154,
        dtype: str = "auto",
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ):
        self.path = Path(path)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        if type(seq_len) is not int or seq_len < 2:
            raise ValueError("seq_len must be an integer of at least 2")
        if (eos_token_id is None) != (pad_token_id is None):
            raise ValueError("eos_token_id and pad_token_id must be provided together")
        if eos_token_id is not None and eos_token_id == pad_token_id:
            raise ValueError("eos_token_id and pad_token_id must be distinct")
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
        if file_size % self._byte_width:
            raise ValueError("dataset file size must be aligned to its token dtype")
        self.num_tokens = file_size // self._byte_width
        np_dtype = np.uint16 if self._byte_width == 2 else np.uint32
        self._data = np.memmap(self.path, dtype=np_dtype, mode="r", shape=(self.num_tokens,))
        self._records: list[tuple[int, int]] | None = None
        if self.eos_token_id is not None:
            self._records = []
            start = 0
            content_limit = self.seq_len - 1
            scan_block = 1_048_576
            for block_start in range(0, self.num_tokens, scan_block):
                block = self._data[block_start : block_start + scan_block]
                for relative_eos in np.flatnonzero(block == self.eos_token_id):
                    end = block_start + int(relative_eos)
                    for chunk_start in range(start, end, content_limit):
                        self._records.append((chunk_start, min(chunk_start + content_limit, end)))
                    if start == end:
                        self._records.append((start, end))
                    start = end + 1
            if start < self.num_tokens:
                for chunk_start in range(start, self.num_tokens, content_limit):
                    self._records.append((chunk_start, min(chunk_start + content_limit, self.num_tokens)))

    def _load(self):
        return None

    def __len__(self) -> int:
        if self._records is not None:
            return len(self._records)
        return max(0, self.num_tokens - self.seq_len + 1)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample for masked LM (same-position prediction).

        input_ids and targets are the same normalized chunk. The model predicts
        the original token at each selected position, including terminal EOS.
        """
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if self._records is not None:
            start, end = self._records[idx]
            ids = torch.from_numpy(np.asarray(self._data[start:end]).astype(np.int64, copy=True))
        else:
            ids = torch.from_numpy(np.asarray(self._data[idx : idx + self.seq_len]).astype(np.int64, copy=True))
        ids[ids >= self.vocab_size] = 0
        if self.eos_token_id is None or self.pad_token_id is None:
            return {"input_ids": ids, "targets": ids.clone()}

        normalized = torch.full((self.seq_len,), self.pad_token_id, dtype=torch.long)
        content = ids[ids != self.pad_token_id][: self.seq_len - 1]
        normalized[: content.numel()] = content
        normalized[content.numel()] = self.eos_token_id
        valid = validate_terminal_eos(
            normalized.unsqueeze(0),
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        ).squeeze(0)
        return {"input_ids": normalized, "targets": normalized.clone(), "valid_target_mask": valid}


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
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
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
        if not mix["sources"]:
            raise ValueError("mix sources must be nonempty")

        self.sources: list[tuple[str, float, MemmapDataset]] = []
        total_ratio = 0.0
        for src in mix["sources"]:
            name = src["name"]
            ratio = src["ratio"]
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 0:
                raise ValueError("mix source ratios must be finite and nonnegative")
            path = dataset_path(self.data_dir, name)
            if path.exists():
                ds = MemmapDataset(
                    str(path),
                    seq_len,
                    vocab_size,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
                if len(ds) > 0:
                    self.sources.append((name, ratio, ds))
                    total_ratio += ratio

        if total_ratio <= 0:
            raise ValueError("mix source total ratio must be greater than zero")
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
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader from .bin files in data_dir."""
    dataset = MixedDataset(
        data_dir,
        mix_path,
        seq_len,
        vocab_size,
        num_samples,
        eos_token_id,
        pad_token_id,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
