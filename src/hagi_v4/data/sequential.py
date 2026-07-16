"""Sequential cycling dataset iterator — ported from HAGI v1.

Cycles through datasets in list order (curriculum: easy->hard),
N cycles per dataset before advancing to the next. Loops forever.

Stage 1: curriculum_order from config (default: tinystories -> edu)
Stage 2 (at step threshold): stage2_datasets from config
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from hagi_v4.data.dataset import MemmapDataset, dataset_path

logger = logging.getLogger(__name__)


class RandomSubsetDataset(Dataset):
    """Draws n_samples random batches from a MemmapDataset."""

    def __init__(self, base: MemmapDataset, n_samples: int):
        self.base = base
        self.n_samples = n_samples
        self._indices = None

    def _ensure_indices(self):
        if self._indices is None:
            length = len(self.base)
            if length == 0:
                self._indices = []
            else:
                self._indices = [random.randint(0, length - 1) for _ in range(self.n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        self._ensure_indices()
        if not self._indices:
            return self.base[0]
        return self.base[self._indices[idx % len(self._indices)]]


class SequentialCyclingIterator:
    """Cycles through datasets in order, N cycles each, loops forever.

    Checkpointable: state_dict / load_state_dict preserve position.
    """

    def __init__(
        self,
        entries: list[tuple[str, str]],
        seq_len: int,
        vocab_size: int,
        cycles_per_dataset: int = 3,
        batch_size: int = 8,
        samples_per_cycle: int = 5000,
        num_workers: int = 0,
        dtype: str = "auto",
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ):
        self.entries = entries  # [(name, path), ...]
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.cycles_per_dataset = cycles_per_dataset
        self.batch_size = batch_size
        self.samples_per_cycle = samples_per_cycle
        self.num_workers = num_workers
        self.dtype = dtype
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.current_idx = 0
        self.current_cycle = 0
        self._current_loader = None
        self._current_iter = None

        logger.info(
            f"SequentialCycling: {len(entries)} datasets, "
            f"{cycles_per_dataset} cycles each, "
            f"{samples_per_cycle} samples/cycle"
        )

    def _build_loader(self) -> DataLoader:
        name, path = self.entries[self.current_idx]
        ds = MemmapDataset(
            path,
            self.seq_len,
            self.vocab_size,
            dtype=self.dtype,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        subset = RandomSubsetDataset(ds, self.samples_per_cycle)
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        logger.info(f"SequentialCycling: dataset={name} cycle={self.current_cycle + 1}/{self.cycles_per_dataset}")
        return loader

    def _ensure_iter(self):
        if self._current_iter is None:
            self._current_loader = self._build_loader()
            self._current_iter = iter(self._current_loader)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        self._ensure_iter()
        try:
            return next(self._current_iter)
        except StopIteration:
            self.current_cycle += 1
            if self.current_cycle >= self.cycles_per_dataset:
                self.current_idx = (self.current_idx + 1) % len(self.entries)
                self.current_cycle = 0
            self._current_iter = None
            self._ensure_iter()
            return next(self._current_iter)

    def state_dict(self) -> dict:
        return {"current_idx": self.current_idx, "current_cycle": self.current_cycle}

    def load_state_dict(self, state: dict) -> None:
        self.current_idx = state.get("current_idx", 0)
        self.current_cycle = state.get("current_cycle", 0)
        self._current_iter = None


class CurriculumBatchProvider:
    """Two-stage curriculum: switches from stage1 to stage2 at step threshold."""

    def __init__(
        self,
        stage1_iter: SequentialCyclingIterator,
        stage2_iter: SequentialCyclingIterator | None,
        stage2_start_step: int,
    ):
        self.stage1 = stage1_iter
        self.stage2 = stage2_iter
        self.stage2_start = stage2_start_step
        self._optimizer_step = 0
        self._switched = False

    def set_optimizer_step(self, step: int) -> None:
        self._optimizer_step = step

    def _active_iter(self):
        if self.stage2 is not None and self._optimizer_step >= self.stage2_start:
            if not self._switched:
                logger.info(f"Curriculum: switching to stage 2 at step {self._optimizer_step}")
                self._switched = True
            return self.stage2
        return self.stage1

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        return next(self._active_iter())

    def state_dict(self) -> dict:
        return {
            "stage": 2 if self._switched else 1,
            "optimizer_step": self._optimizer_step,
            "stage1_state": self.stage1.state_dict(),
            "stage2_state": self.stage2.state_dict() if self.stage2 is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        self._optimizer_step = state.get("optimizer_step", 0)
        self._switched = state.get("stage", 1) == 2
        if "stage1_state" in state:
            self.stage1.load_state_dict(state["stage1_state"])
        if self.stage2 is not None and state.get("stage2_state") is not None:
            self.stage2.load_state_dict(state["stage2_state"])


def resolve_sequential_entries(
    mix_paths: list[dict],
    data_dir: str,
) -> list[tuple[str, str]]:
    """Resolve mix_paths config into (name, path) tuples in curriculum order."""
    entries = []
    for entry in mix_paths:
        name = entry["name"]
        path = str(dataset_path(data_dir, name))
        if Path(path).exists():
            entries.append((name, path))
        else:
            logger.warning(f"SequentialCycling: {name}.bin not found — skipping")
    return entries


def build_sequential_dataloader(
    cfg,
    data_dir: str = "data",
) -> CurriculumBatchProvider:
    """Build curriculum dataloader from config."""

    with open(Path(data_dir) / "mix.json") as f:
        mix = json.load(f)

    # Build stage1 entries in curriculum order (easy -> hard)
    curriculum_order = cfg.train.curriculum_order
    available = {s["name"]: s for s in mix["sources"]}
    stage1_entries = []
    for name in curriculum_order:
        if name in available:
            path = str(dataset_path(data_dir, name))
            if Path(path).exists():
                stage1_entries.append((name, path))

    if not stage1_entries:
        expected = ", ".join(f"{name}.bin" for name in curriculum_order)
        raise ValueError(f"no stage1 datasets found in {data_dir}; expected: {expected}")

    cycles = cfg.train.sequential_cycles
    samples_per_cycle = max(1000, cfg.train.max_steps * cfg.train.batch_size // (len(stage1_entries) * cycles))
    dtype = getattr(cfg.train, "data_dtype", "auto")

    stage1 = SequentialCyclingIterator(
        entries=stage1_entries,
        seq_len=cfg.train.seq_len,
        vocab_size=cfg.model.vocab_size,
        cycles_per_dataset=cycles,
        batch_size=cfg.train.batch_size,
        samples_per_cycle=samples_per_cycle,
        dtype=dtype,
        eos_token_id=cfg.train.eos_token_id,
        pad_token_id=cfg.train.pad_token_id,
    )

    # Stage 2: hard-reasoning subset from config
    stage2_names = set(cfg.train.stage2_datasets)
    stage2_entries = [(n, p) for n, p in stage1_entries if n in stage2_names]
    stage2 = None
    stage2_start = cfg.train.curriculum_stage2_start if cfg.train.curriculum_enabled else None
    if stage2_entries and stage2_start is not None:
        stage2 = SequentialCyclingIterator(
            entries=stage2_entries,
            seq_len=cfg.train.seq_len,
            vocab_size=cfg.model.vocab_size,
            cycles_per_dataset=2,
            batch_size=cfg.train.batch_size,
            samples_per_cycle=samples_per_cycle,
            dtype=dtype,
            eos_token_id=cfg.train.eos_token_id,
            pad_token_id=cfg.train.pad_token_id,
        )

    return CurriculumBatchProvider(
        stage1_iter=stage1,
        stage2_iter=stage2,
        stage2_start_step=stage2_start if stage2_start is not None else 0,
    )
