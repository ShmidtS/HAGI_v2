"""Offline gate_ce evaluation of an arbitrary checkpoint.

The in-run gate (loop.py ``_gate_ce``) reads fixed 2048-token tail
windows per corpus and weights them canonically -- but only runs
inside a training loop, at gate_eval_interval cadence, so any
checkpoint saved BEFORE the gate was enabled has no comparable
number. The gen7 merged run was killed at step ~1040 with exactly
that question open: is best.pt (step 850, first post-resume gate
sample) really the best state, or an artifact of the first sample?

This script reproduces the gate EXACTLY (same tail window, same
weights, same 2048-token read, same exact_loss) on any checkpoint,
so gate numbers are comparable across checkpoints and runs.

Usage:
    python scripts/gate_score.py --config configs/dbridge_gen7_merged.yaml \
        --ckpt checkpoints/dbridge_gen7_merged/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import load_config  # noqa: E402
from hagi.model.factory import build_model_for_config  # noqa: E402
from hagi.train.checkpoint import load_model  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402


def gate_windows(cfg) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """The loop.py gate: last-2M tail, 2048 tokens, canonical weights."""
    windows = []
    data_dir = Path(cfg.train.data.data_dir)
    for name, w in sorted((cfg.train.data.weights or {}).items()):
        p = data_dir / f"{name}.compact.bin"
        if not p.exists():
            continue
        total = p.stat().st_size // 4
        start = max(0, total - 2_000_000)
        with p.open("rb") as fh:
            fh.seek(start * 4)
            T = np.frombuffer(fh.read(2048 * 4), dtype=np.uint32).astype(np.int64)
        ids = torch.from_numpy(T[:2048]).reshape(2, 1024)
        windows.append((ids[:, :-1], ids[:, 1:], float(w)))
    wsum = sum(w for *_, w in windows) or 1.0
    return [(x, y, w / wsum) for x, y, w in windows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    configure_runtime()
    cfg = load_config(args.config)
    model = build_model_for_config(cfg).to(args.device)
    steps, _ = load_model(args.ckpt, model, args.device)
    model.eval()

    windows = gate_windows(cfg)
    total = 0.0
    with torch.no_grad():
        for x, y, w in windows:
            x, y = x.to(args.device), y.to(args.device)
            o = model(x, y)
            f = o.hidden.reshape(-1, o.hidden.shape[-1])
            total += w * float(model.head.exact_loss(f, y.reshape(-1)))
    print(f"ckpt={args.ckpt} step={steps} gate_ce={total:.4f}")


if __name__ == "__main__":
    main()
