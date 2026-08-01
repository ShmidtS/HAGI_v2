"""Compact-vocabulary id mapping (old tokenizer space <-> dense model space).

``scripts/compact_vocab.py`` rewrites the corpus into a dense id range
``[0, V_new)`` and saves ``data/vocab_map.npz`` with two arrays:

* ``old_to_new [V_old]`` — old tokenizer id -> compact id, ``-1`` for dropped.
* ``new_to_old [V_new]`` — compact id -> original tokenizer id.

Training reads the compact streams directly, so the model never sees a dropped
id. Inference encodes with the tokenizer (old space) and must map every token
into the compact space before the forward pass; decoding maps back. Dropped ids
(never in the corpus, so never in a compact stream) fall back to a reserved id.

The map is a pure data artifact: it is loaded from disk, never written here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VocabMap:
    """Bidirectional compact-vocab mapping with a fallback for dropped ids.

    Args:
        path: ``vocab_map.npz`` produced by scripts/compact_vocab.py.
        fallback: compact id to use for dropped ids (typically UNK).
    """

    def __init__(self, path: str | Path, fallback: int = 3) -> None:
        data = np.load(path)
        if "old_to_new" not in data or "new_to_old" not in data:
            raise ValueError(f"{path} is not a vocab map (missing old_to_new/new_to_old)")
        old_to_new = np.asarray(data["old_to_new"], dtype=np.int64)
        new_to_old = np.asarray(data["new_to_old"], dtype=np.int64)
        if old_to_new.ndim != 1 or new_to_old.ndim != 1:
            raise ValueError(f"vocab map arrays must be 1D, got {old_to_new.shape}/{new_to_old.shape}")
        self.old_vocab = int(old_to_new.shape[0])
        self.new_vocab = int(new_to_old.shape[0])
        self._old_to_new = old_to_new
        self._new_to_old = new_to_old
        self.fallback = int(fallback)
        if not 0 <= self.fallback < self.new_vocab:
            raise ValueError(f"fallback id {fallback} is outside the compact vocabulary {self.new_vocab}")

    def to_compact(self, ids) -> np.ndarray:
        """Map old-space token ids to compact ids; dropped ids -> fallback."""
        arr = np.asarray(ids, dtype=np.int64)
        if arr.size and (arr.min() < 0 or arr.max() >= self.old_vocab):
            raise ValueError(f"ids outside old vocabulary [0, {self.old_vocab}): "
                             f"min={arr.min()} max={arr.max()}")
        mapped = self._old_to_new[arr]
        mapped = np.where(mapped < 0, self.fallback, mapped)
        return mapped.astype(np.int64)

    def to_old(self, ids) -> np.ndarray:
        """Map compact ids back to old-space ids (for the tokenizer)."""
        arr = np.asarray(ids, dtype=np.int64)
        if arr.size and (arr.min() < 0 or arr.max() >= self.new_vocab):
            raise ValueError(f"ids outside compact vocabulary [0, {self.new_vocab}): "
                             f"min={arr.min()} max={arr.max()}")
        return self._new_to_old[arr].astype(np.int64)

    def decode_batch(self, tokenizer, ids) -> list[str]:
        """Decode compact ids through the old tokenizer (batch of rows or flat)."""
        old = self.to_old(ids)
        rows = old.reshape(ids.shape).tolist() if getattr(ids, "ndim", 0) > 1 else old.tolist()
        if isinstance(rows, list) and rows and isinstance(rows[0], list):
            return [tokenizer.decode(row) for row in rows]
        return [tokenizer.decode(rows)]
