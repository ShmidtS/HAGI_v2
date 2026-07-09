"""TinyStories dataset loader — reads pre-tokenized binary file.

The file data/tinystories.bin contains 60M uint16 SmolLM2 token IDs,
267K stories separated by EOS (token 0).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class TinyStoriesConfig:
    n_stories: int = 10000
    seq_len: int = 256
    val_frac: float = 0.05
    seed: int = 42
    bin_path: str = "data/tinystories.bin"
    eos_token_id: int = 0


class TinyStoriesDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, self.token_ids.shape[0] - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.token_ids[idx : idx + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "targets": chunk[1:]}


def load_tinystories(cfg: TinyStoriesConfig) -> tuple[TinyStoriesDataset, TinyStoriesDataset]:
    import os

    if not os.path.exists(cfg.bin_path):
        raise FileNotFoundError(f"TinyStories binary not found: {cfg.bin_path}")

    logger.info("Loading pre-tokenized data from %s", cfg.bin_path)
    data = np.fromfile(cfg.bin_path, dtype=np.uint16)
    logger.info("Loaded %d tokens, max ID=%d", len(data), data.max())

    eos = cfg.eos_token_id
    story_boundaries = np.where(data == eos)[0]
    n_total = len(story_boundaries)
    logger.info("Found %d stories", n_total)

    rng = random.Random(cfg.seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    selected = sorted(indices[: cfg.n_stories])

    all_tokens: list[int] = []
    for si in selected:
        start = story_boundaries[si - 1] + 1 if si > 0 else 0
        end = story_boundaries[si] + 1
        all_tokens.extend(data[start:end].tolist())

    token_tensor = torch.tensor(all_tokens, dtype=torch.long)
    n_val = max(cfg.seq_len + 1, int(len(token_tensor) * cfg.val_frac))
    val_tokens = token_tensor[:n_val]
    train_tokens = token_tensor[n_val:]

    train_ds = TinyStoriesDataset(train_tokens, cfg.seq_len)
    val_ds = TinyStoriesDataset(val_tokens, cfg.seq_len)
    logger.info("Train: %d, Val: %d", len(train_ds), len(val_ds))
    return train_ds, val_ds
