"""Control: is the trained model actually better than a random init?

Every quality number in this project is a delta between two models. If the
"trained" model were no better than a fresh random initialisation of the same
architecture, every delta would be measuring nothing.

This scores three models on identical held-out batches from the same corpus
tail, so the only difference between them is the weights:

  1. the trained checkpoint from --resume
  2. a fresh random initialisation of the SAME architecture (same config)
  3. optionally, a weight-scrambled copy of the trained model: every weight
     replaced by the trained weight of a different tensor, which keeps the
     value distribution but destroys the learned structure. It separates
     "the weights have the right scale" from "the weights are the right ones".

Usage:
    python scripts/trainability_control.py --config configs/m2_merged_joint.yaml \
        --resume checkpoints/m2_merged_joint/step-0009000.pt --batches 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_holdout import DOMAIN_FILES, _batches, _tail_tokens, score  # noqa: E402

from hagi.config import load_config  # noqa: E402
from hagi.model.merge import MergedHAGI, build_model_from_payload  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.checkpoint import config_from_dict, load_payload  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402

_DATA = Path(__file__).resolve().parent.parent / "data"


def fresh_model(cfg, device):
    """The same architecture with untouched random weights."""
    if cfg.merge.enabled:
        return MergedHAGI(cfg).to(device)
    return HAGI(cfg).to(device)


def scrambled_model(model: torch.nn.Module) -> torch.nn.Module:
    """Same weight shapes and value distribution, learned structure destroyed."""
    values = [p.detach().reshape(-1) for p in model.parameters()]
    out = fresh_model_like(model)
    params = list(out.parameters())
    for index, param in enumerate(params):
        donor = values[(index + 1) % len(values)]
        flat = donor[torch.randint(0, donor.numel(), (param.numel(),))]
        param.data.copy_(flat.reshape(param.shape))
    return out


def fresh_model_like(model: torch.nn.Module) -> torch.nn.Module:
    import copy

    clone = copy.deepcopy(model)
    for p in clone.parameters():
        p.data.zero_()
    return clone


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--holdout-tokens", type=int, default=8_000_000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    configure_runtime()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    cfg = load_config(args.config)
    payload = load_payload(args.resume)
    trained = build_model_from_payload(
        config_from_dict(payload["config"]), payload["model"], device=device
    )
    trained.eval()
    random_init = fresh_model(cfg, device)
    random_init.eval()
    scrambled = scrambled_model(trained)
    scrambled.eval()

    report = {"config": args.config, "checkpoint": args.resume, "domains": {}}
    for domain, names in DOMAIN_FILES.items():
        present = [n for n in names if ( _DATA / f"{n}.compact.bin").is_file() or (_DATA / f"{n}.bin").is_file()]
        if not present:
            continue
        tokens = _tail_tokens(_DATA, present[0], args.holdout_tokens)
        batches = _batches(tokens, args.seq_len, args.batches)
        trained_ce = score(trained, batches, device)
        random_ce = score(random_init, batches, device)
        scrambled_ce = score(scrambled, batches, device)
        report["domains"][domain] = {
            "trained": trained_ce,
            "random_init": random_ce,
            "scrambled": scrambled_ce,
            "trained_beats_random_by_nats": random_ce["exact_ce"] - trained_ce["exact_ce"],
        }
    out = Path(args.out) if args.out else None
    text = json.dumps(report, indent=2, sort_keys=True)
    if out:
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
