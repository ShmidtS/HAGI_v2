"""Proportional mixed dataset iterator — interleaved sampling from all datasets.

Replaces sequential cycling (catastrophic forgetting root cause). Each batch
draws randomly from all datasets weighted by their ratios. Within each stage
(stage1 / stage2) all datasets are mixed proportionally, preventing the
optimizer from specializing on one distribution at a time.

Curriculum: stage1 datasets at stage1 ratios → stage2 datasets at stage2
ratios after step threshold. No "cycles" — continuous mixed sampling.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from hagi.data.dataset import MemmapDataset, dataset_path

logger = logging.getLogger(__name__)


class ProportionalMixDataset(Dataset):
    """Draws samples from multiple MemmapDatasets weighted by ratios.

    Each __getitem__ picks a dataset according to the ratio distribution,
    then draws a random position from that dataset. This interleaves all
    datasets within every batch — no forgetting.
    """

    def __init__(
        self,
        datasets: list[tuple[str, float, MemmapDataset]],
        n_samples: int,
    ):
        self.datasets = datasets  # [(name, weight, ds), ...]
        self.n_samples = n_samples
        self._names = [n for n, _, _ in datasets]
        self._weights = [w for _, w, _ in datasets]
        self._dss = [d for _, _, d in datasets]
        total = sum(self._weights)
        self._cumulative = []
        acc = 0.0
        for w in self._weights:
            acc += w / total
            self._cumulative.append(acc)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        r = random.random()
        for i, threshold in enumerate(self._cumulative):
            if r < threshold:
                ds = self._dss[i]
                if len(ds) == 0:
                    continue
                pos = random.randint(0, len(ds) - 1)
                return ds[pos]
        # Fallback: last dataset
        return self._dss[-1][random.randint(0, len(self._dss[-1]) - 1)]


def build_proportional_dataloader(
    cfg,
    data_dir: str = "data",
) -> DataLoader:
    """Build a DataLoader that interleaves ALL datasets proportionally.

    Stage1 and stage2 are separate mix proportions applied at the
    curriculum threshold. Within each stage, datasets are interleaved
    randomly, weighted by their configured ratios.
    """
    with open(Path(data_dir) / "mix.json") as f:
        mix = json.load(f)

    available = {s["name"]: s for s in mix["sources"]}
    stage1_order = cfg.train.curriculum.stage1_order

    # Build stage1 datasets
    stage1_datasets: list[tuple[str, float, MemmapDataset]] = []
    for name in stage1_order:
        if name not in available:
            continue
        path = str(dataset_path(data_dir, name))
        if not Path(path).exists():
            logger.warning(f"ProportionalMix: {name}.bin not found — skipping")
            continue
        src = available[name]
        ratio = float(src.get("ratio", 1.0))
        ds = MemmapDataset(
            path,
            cfg.train.seq_len,
            cfg.model.vocab_size,
            dtype=cfg.train.data_dtype,
            eos_token_id=cfg.train.eos_token_id,
            pad_token_id=cfg.train.pad_token_id,
        )
        if len(ds) > 0:
            stage1_datasets.append((name, ratio, ds))

    if not stage1_datasets:
        expected = ", ".join(f"{n}.bin" for n in stage1_order)
        raise ValueError(f"no stage1 datasets found in {data_dir}; expected: {expected}")

    names1 = ", ".join(f"{n}({w:.1f})" for n, w, _ in stage1_datasets)
    logger.info(f"ProportionalMix stage1: {names1}")

    # Build stage2 datasets (hard-reasoning subset)
    stage2_names = set(cfg.train.curriculum.stage2_datasets)
    stage2_datasets = [(n, w, d) for n, w, d in stage1_datasets if n in stage2_names]

    if stage2_datasets:
        names2 = ", ".join(f"{n}({w:.1f})" for n, w, _ in stage2_datasets)
        logger.info(f"ProportionalMix stage2: {names2}")

    # Total samples per epoch = max_steps * grad_accum_steps
    n_samples = cfg.train.max_steps * cfg.train.grad_accum_steps
    stage2_start = cfg.train.curriculum.stage2_start if cfg.train.curriculum.enabled else cfg.train.max_steps + 1

    class CurriculumMixDataset(Dataset):
        """Dataset wrapper that switches from stage1 to stage2 at threshold."""

        def __init__(self):
            self.stage1 = ProportionalMixDataset(stage1_datasets, n_samples)
            self.stage2 = (
                ProportionalMixDataset(stage2_datasets, n_samples)
                if stage2_datasets
                else self.stage1
            )
            self._step = 0
            self._switched = False

        def set_optimizer_step(self, step: int) -> None:
            self._step = step
            if step >= stage2_start and not self._switched and stage2_datasets:
                logger.info(f"Curriculum: switching to stage 2 at step {step}")
                self._switched = True

        def __len__(self) -> int:
            return n_samples

        def _active(self):
            return self.stage2 if self._switched else self.stage1

        def __getitem__(self, idx: int) -> dict:
            return self._active()[idx]

        def state_dict(self) -> dict:
            return {
                "optimizer_step": self._step,
                "stage": 2 if self._switched else 1,
            }

        def load_state_dict(self, state: dict) -> None:
            self._step = state.get("optimizer_step", 0)
            self._switched = state.get("stage", 1) == 2

    dataset = CurriculumMixDataset()
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
