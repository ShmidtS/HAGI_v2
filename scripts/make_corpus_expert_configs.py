"""Generate per-corpus expert configs for the recursive-growth pipeline.

Each corpus is a separate source. From each corpus, experts (H=128) are trained
sequentially until the data is exhausted — each expert trains on a contiguous
slice (start_offset = previous expert's consumed tokens) to saturation OR the
end of the slice. All level-0 experts are then block-diagonally merged into
H=N*128, and the merged model becomes the shared prior for the next level.

Corpus -> single source (each expert trains on exactly one .bin):
    edu, slimpajama, wikipedia_ru, oscar_ru, openwebmath, smoltalk,
    tinystories, wikipedia_en, python_instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
EXPERTS = CONFIGS / "corpus_experts"

# Each corpus is a separate source; one expert config per corpus.
CORPORA: list[str] = [
    "edu",
    "slimpajama",
    "wikipedia_ru",
    "oscar_ru",
    "openwebmath",
    "smoltalk",
    "tinystories",
    "wikipedia_en",
    "python_instruct",
]

EXPERT_BASE = "v46_1b.yaml"

# Safe max batch under torch.compile on ROCm (see make_expert_configs.py).
SAFE_BATCH = 256


def _load(name: str) -> dict:
    with open(CONFIGS / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dump(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, width=100)


def make_corpus_expert_configs() -> dict[str, Path]:
    """Write one config per corpus. Returns {corpus: path}."""
    paths: dict[str, Path] = {}
    for corpus in CORPORA:
        cfg = _load(EXPERT_BASE)
        # max_steps is a hard safety ceiling, NOT the training target. The
        # expert trains until it reaches a validation-CE plateau (saturation)
        # OR the corpus slice is exhausted (finite PackedStream).
        cfg["train"]["max_steps"] = 100000
        cfg["train"]["batch_size"] = SAFE_BATCH
        # Same-initialization: every expert must start from the same random
        # weights (the method's core requirement).
        cfg["model"]["init_seed"] = 1234
        # Each expert trains on exactly one corpus.
        cfg["train"]["data"]["weights"] = {corpus: 1.0}
        # Train to saturation (validation-CE plateau), not a fixed step count.
        cfg["train"]["logging"]["exact_ce_interval"] = 100
        cfg["train"]["logging"]["exact_ce_rows"] = 512
        cfg["train"]["saturation_patience"] = 20
        cfg["train"]["saturation_tol"] = 0.01
        cfg["train"]["saturation_min_steps"] = 2000
        cfg["train"]["checkpoint_dir"] = f"checkpoints_corpus/{corpus}"
        cfg["train"]["checkpoint_interval"] = 1000
        cfg["train"]["checkpoint_keep_last"] = 2
        cfg["train"]["schedule"]["warmup_steps"] = 500
        path = EXPERTS / f"expert_{corpus}.yaml"
        _dump(cfg, path)
        paths[corpus] = path
    return paths


def main() -> int:
    paths = make_corpus_expert_configs()
    print("corpus expert configs:")
    for corpus, p in paths.items():
        print(f"  {corpus:20s} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
