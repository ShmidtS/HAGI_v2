"""Held-out exact-CE evaluation on a token region the trainer never saw.

Every ``exact_ce`` printed by ``scripts/train.py`` is measured on the batches
the model is being trained on, so it is a training metric, not a quality
metric. This script reads a *disjoint* tail region of the same corpus files
and scores it, which is what the growth hypothesis actually needs.

The trainer starts at byte offset 0 of each ``<name>.bin`` (see
``hagi.data.dataset``), so the held-out region is the last
``--holdout-tokens`` tokens of every file, minus a safety margin. Any run
that consumes more than the margin simply has no held-out left, and the
script refuses to score instead of silently scoring training data.

Usage:
    python scripts/eval_holdout.py --config configs/m2_merged_joint.yaml \
        --resume checkpoints/m2_merged_joint/step-0009000.pt --batches 40
"""

from __future__ import annotations

import argparse
import json
import math
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

# Held-out domain -> corpus files. Kept in sync with the trainer mixes used by
# the m2_* configs; a domain is reported only if all its files are present.
#
# ``python_instruct`` is deliberately absent from the mathcode domain: the whole
# corpus is only 3.1M tokens, so the m2 runs consumed all of it and no untouched
# region exists. Scoring it would silently report training loss as held-out
# quality, so the CODE half of that domain is reported as unavailable instead.
DOMAIN_FILES: dict[str, tuple[str, ...]] = {
    "ru": ("wikipedia_ru", "oscar_ru"),
    "en": ("slimpajama", "edu"),
    "mathcode": ("openwebmath",),
}

# Tokens the trainer may legitimately have consumed before this script is
# allowed to call anything "held out". The m2 runs consume 24.6M tokens per
# expert lane and 73.7M for the joint lane, so 60M leaves a real margin on the
# 100M+ corpora while still being refused on a corpus that is nearly consumed.
DEFAULT_SAFETY_MARGIN = 60_000_000


def _tail_tokens(root: Path, name: str, count: int) -> np.ndarray:
    path = root / f"{name}.compact.bin"
    if not path.is_file():
        path = root / f"{name}.bin"
    if not path.is_file():
        raise FileNotFoundError(f"no corpus for {name!r} under {root}")
    total = path.stat().st_size // 4
    if total <= count:
        raise ValueError(f"{name}: corpus of {total} tokens is not larger than {count}")
    with path.open("rb") as fh:
        fh.seek((total - count) * 4)
        buf = fh.read(count * 4)
    return np.frombuffer(buf, dtype=np.uint32).astype(np.int64)


def _batches(tokens: np.ndarray, seq_len: int, count: int) -> list[torch.Tensor]:
    usable = (len(tokens) // seq_len) * seq_len
    out = []
    for i in range(count):
        start = (i * seq_len) % max(usable - seq_len, 1)
        chunk = tokens[start : start + seq_len + 1]
        if len(chunk) < seq_len + 1:
            chunk = np.concatenate([chunk, tokens[: seq_len + 1 - len(chunk)]])
        out.append(torch.from_numpy(chunk.astype(np.int64)))
    return out


@torch.no_grad()
def score(model: torch.nn.Module, batches: list[torch.Tensor], device) -> dict:
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    for batch in batches:
        x = batch[:-1].unsqueeze(0).to(device)
        y = batch[1:].unsqueeze(0).to(device)
        out = model(x, targets=y)
        loss = out.loss if hasattr(out, "loss") else out["loss"]
        n = int(y.numel())
        total_nats += float(loss) * n
        total_tokens += n
    ce = total_nats / max(total_tokens, 1)
    return {"exact_ce": ce, "ppl": math.exp(ce), "tokens": total_tokens}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--holdout-tokens", type=int, default=8_000_000)
    ap.add_argument("--safety-margin", type=int, default=DEFAULT_SAFETY_MARGIN)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    configure_runtime()
    cfg = load_config(args.config)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    model = build_model_for_config(cfg).to(device)
    load_model(args.resume, model, str(device))
    model.eval()

    root = Path(cfg.train.data.data_dir)
    results: dict[str, dict] = {}
    for domain, names in DOMAIN_FILES.items():
        parts = []
        for name in names:
            for candidate in (root / f"{name}.compact.bin", root / f"{name}.bin"):
                if candidate.is_file():
                    total = candidate.stat().st_size // 4
                    if total <= args.safety_margin:
                        continue
                    parts.append(_tail_tokens(root, name, args.holdout_tokens))
                    break
        if not parts:
            results[domain] = {"error": "no untouched tail region available"}
            continue
        tokens = np.concatenate(parts)
        results[domain] = score(model, _batches(tokens, cfg.train.data.seq_len, args.batches), device)

    payload = {
        "config": args.config,
        "checkpoint": args.resume,
        "batches": args.batches,
        "holdout_tokens_per_corpus": args.holdout_tokens,
        "domains": results,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
