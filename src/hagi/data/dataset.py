"""Packed-corpus data pipeline.

Documents are concatenated into one flat token stream and cut into fixed-length
windows. No padding: every position in every batch carries a real token, and each
window is annotated with ``doc_ids`` so the attention mask can be block-diagonal
per document.

Why this replaced per-document batching. The V28/V30 loader padded each document
to ``seq_len``. Measured utilization on this corpus:

    tinystories   T=512 -> 0.41    T=2048 -> 0.10
    wikipedia_en  T=512 -> 0.55    T=2048 -> 0.20
    edu           T=512 -> 0.80    T=2048 -> 0.43
    slimpajama    T=512 -> 0.78    T=2048 -> 0.40

At T=2048 roughly three quarters of every forward pass was spent on PAD tokens —
attention over them, FFN over them, and their gradient contributions masked out
at the end. Packing recovers all of it, which is a 2-9x effective throughput gain
depending on source, larger than any architectural change in this rewrite.

Sampling is proportional and interleaved. Each window draws its source from the
mix weights, so every optimizer step sees the full corpus distribution. Sequential
per-dataset cycling (V28) meant the model spent 11k consecutive steps on
tinystories and then met python_instruct for the first time — the resulting
forgetting is visible in the V30 log as a step change in ce at every dataset
boundary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

logger = logging.getLogger(__name__)

TOKEN_DTYPE = np.uint32


def dataset_path(data_dir: str | Path, name: str) -> Path:
    """Resolve ``<data_dir>/<name>.bin``, preferring a compacted stream.

    ``scripts/compact_vocab.py`` writes ``<name>.compact.bin`` against a dense id
    map. When present, the compact stream is used so the model never sees dropped
    ids; otherwise the raw stream is used. Rejects traversal in ``name``.
    """
    if not isinstance(name, str) or not name or Path(name).name != name or "\\" in name:
        raise ValueError(f"invalid dataset name: {name!r}")
    base = Path(data_dir) / f"{name}.bin"
    compact = base.with_suffix(".compact.bin")
    if compact.exists():
        return compact
    return base


def load_mix(data_dir: str | Path, overrides: dict[str, float] | None = None) -> dict[str, float]:
    """Load per-source sampling weights, normalized to sum to 1.

    Sources with no ``.bin`` on disk are dropped with a warning rather than
    failing: a partially-downloaded corpus should still train.

    Args:
        data_dir: directory holding ``mix.json`` and the ``.bin`` files.
        overrides: explicit weights replacing ``mix.json`` entirely.

    Returns:
        ``{name: weight}`` summing to 1.

    Raises:
        FileNotFoundError: no usable source found.
    """
    root = Path(data_dir)
    if overrides:
        raw = dict(overrides)
    else:
        with open(root / "mix.json", encoding="utf-8") as fh:
            spec = json.load(fh)
        raw = {src["name"]: float(src.get("ratio", 1.0)) for src in spec["sources"]}

    present = {}
    for name, weight in raw.items():
        if weight <= 0:
            continue
        path = dataset_path(root, name)
        if not path.exists():
            logger.warning("data mix: %s.bin not found, skipping", name)
            continue
        present[name] = weight
    if not present:
        raise FileNotFoundError(f"no dataset .bin files found in {root}")
    total = sum(present.values())
    return {name: weight / total for name, weight in present.items()}


class PackedStream:
    """A memory-mapped token stream that yields packed windows.

    Reading is sequential from a randomized start offset. Sequential access keeps
    the OS page cache effective — a random seek per window on a 5 GB file makes
    the loader disk-bound rather than compute-bound.

    Args:
        path: ``.bin`` of flat uint32 tokens.
        seq_len: window length.
        eos_token_id: document delimiter, used to derive ``doc_ids``.
        rng: seeded generator for the start offset.
    """

    def __init__(self, path: Path, seq_len: int, eos_token_id: int, rng: np.random.Generator) -> None:
        self.path = path
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.tokens = np.memmap(path, dtype=TOKEN_DTYPE, mode="r")
        self.n_tokens = len(self.tokens)
        if self.n_tokens < seq_len + 1:
            raise ValueError(f"{path.name} holds {self.n_tokens} tokens, need at least {seq_len + 1}")
        self.cursor = int(rng.integers(0, self.n_tokens - seq_len - 1))

    def next_window(self) -> np.ndarray:
        """Return ``seq_len + 1`` contiguous tokens, wrapping at end of file.

        The extra token is the final position's target, so no window loses its
        last prediction to the boundary.
        """
        end = self.cursor + self.seq_len + 1
        if end > self.n_tokens:
            self.cursor = 0
            end = self.seq_len + 1
        window = np.asarray(self.tokens[self.cursor : end], dtype=np.int64)
        self.cursor += self.seq_len
        return window


class PackedMixDataset(IterableDataset):
    """Infinite iterator over proportionally-mixed packed windows.

    Iterable rather than indexed: the corpus is a stream with no meaningful
    length, and an indexed dataset would need either a random seek per item
    (disk-bound) or a precomputed index of every window (gigabytes of state).

    Each worker gets its own generator seed and its own start offsets, so workers
    do not duplicate each other's windows.

    Args:
        data_dir: corpus directory.
        seq_len: window length.
        eos_token_id: document delimiter.
        weights: per-source sampling weights (normalized).
        seed: base seed; each worker adds its id.
        cross_doc_attention: when True, ``doc_ids`` is omitted and attention is
            allowed to cross document boundaries (cheaper, slightly wrong).
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        eos_token_id: int,
        weights: dict[str, float],
        seed: int = 1234,
        cross_doc_attention: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.weights = weights
        self.seed = seed
        self.cross_doc_attention = cross_doc_attention

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = np.random.default_rng(self.seed + worker_id * 7919)

        names = list(self.weights)
        probs = np.array([self.weights[n] for n in names], dtype=np.float64)
        probs /= probs.sum()
        streams = [
            PackedStream(dataset_path(self.data_dir, n), self.seq_len, self.eos_token_id, rng)
            for n in names
        ]

        while True:
            window = streams[int(rng.choice(len(streams), p=probs))].next_window()
            input_ids = torch.from_numpy(window[:-1])
            targets = torch.from_numpy(window[1:])
            item = {"input_ids": input_ids, "targets": targets}
            if not self.cross_doc_attention:
                # A new document starts after each EOS, so the running count of
                # EOS tokens to the left is the document index. Computed on the
                # input positions, which is what the mask indexes.
                is_eos = (input_ids == self.eos_token_id).to(torch.int32)
                doc_ids = torch.cumsum(is_eos, dim=0) - is_eos
                item["doc_ids"] = doc_ids
            yield item


def build_dataloader(cfg, data_dir: str | None = None) -> DataLoader:
    """Build the training dataloader from ``cfg.train.data``.

    Args:
        cfg: top-level config.
        data_dir: overrides ``cfg.train.data.data_dir``.

    Returns:
        A :class:`DataLoader` yielding dicts of ``[B, seq_len]`` int64 tensors.
    """
    dc = cfg.train.data
    root = data_dir or dc.data_dir
    weights = load_mix(root, dc.weights or None)
    logger.info(
        "packed mix over %d sources: %s",
        len(weights),
        ", ".join(f"{n}={w:.3f}" for n, w in sorted(weights.items(), key=lambda kv: -kv[1])),
    )
    dataset = PackedMixDataset(
        data_dir=root,
        seq_len=dc.seq_len,
        eos_token_id=dc.eos_token_id,
        weights=weights,
        seed=dc.seed,
        cross_doc_attention=dc.cross_doc_attention,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=dc.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=dc.num_workers > 0,
        prefetch_factor=4 if dc.num_workers > 0 else None,
    )
