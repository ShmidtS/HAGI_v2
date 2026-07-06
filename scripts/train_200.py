"""HAGI V4 — 200-step training test with distillation. All params from YAML."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _train_utils import log_summary, run_training_steps, setup_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main():
    cfg, model, optimizer, device, n_params = setup_training()

    from hagi_v4.train.distillation import DistillationTeacher, transfer_embeddings

    transfer_embeddings(model, cfg.train.distill_embed_teacher)
    teacher = DistillationTeacher(cfg.train.distill_teacher)
    teacher.load()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    n_steps = 200
    losses, total_time = run_training_steps(
        model, optimizer, cfg, device, n_steps=n_steps, teacher=teacher, log_interval=20
    )

    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    teacher.free()
    log_summary(n_params, losses, total_time, n_steps, peak_vram)


if __name__ == "__main__":
    main()
