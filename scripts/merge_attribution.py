"""Is the M2 merged win from the MERGE, or from the extra 9000 steps?

The M2 result is the project's best quality evidence: the merged joint
model beats the baseline by -0.5623 nats on three domains, 3/3 in favour.
But the comparison is not clean:

  experts    trained 3000 steps each, one per domain
  baseline   trained 9000 steps
  merged     trained 9000 steps AFTER merging those experts

So the merged model got 9000 steps of post-merge training on top of three
3000-step experts, while the baseline got 9000 steps from scratch. The win
could be "merging worked", or it could be "9000 more steps on three
pre-trained experts beats 9000 steps from scratch" - which is distillation
with extra steps, not recursive growth.

This script measures the baseline at 4500 and 9000 steps and the merged at
the same two, so the step budget is matched. If the gap survives step
matching, the merge is doing real work. If it collapses, the original
number was a step-budget artifact and must be withdrawn.

Usage:
    python scripts/merge_attribution.py --out reports/merge_attribution.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_holdout import DOMAIN_FILES, _batches, _tail_tokens, score  # noqa: E402

from hagi.model.merge import build_model_from_payload  # noqa: E402
from hagi.train.checkpoint import config_from_dict, load_payload  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402

_DATA = Path(__file__).resolve().parent.parent / "data"

# (label, config, checkpoint)
ARMS = [
    ("baseline_4500", "configs/m2_baseline_merged.yaml", "checkpoints/m2_baseline_h1152/step-0004500.pt"),
    ("baseline_9000", "configs/m2_baseline_merged.yaml", "checkpoints/m2_baseline_h1152/step-0009000.pt"),
    ("merged_4500", "configs/m2_merged_joint.yaml", "checkpoints/m2_merged_joint/step-0004500.pt"),
    ("merged_9000", "configs/m2_merged_joint.yaml", "checkpoints/m2_merged_joint/step-0009000.pt"),
]


def evaluate(config_path: str, checkpoint_path: str, batches_n: int, seq_len: int, device) -> dict:
    # The checkpoint carries its own config; the path is kept in the report
    # for provenance and checked for existence by the caller.
    del config_path
    payload = load_payload(checkpoint_path)
    model = build_model_from_payload(
        config_from_dict(payload["config"]), payload["model"], device=device
    )
    model.eval()
    out: dict[str, dict] = {}
    for domain, names in DOMAIN_FILES.items():
        present = [
            n for n in names
            if (_DATA / f"{n}.compact.bin").is_file() or (_DATA / f"{n}.bin").is_file()
        ]
        if not present:
            continue
        tokens = _tail_tokens(_DATA, present[0], 8_000_000)
        bs = _batches(tokens, seq_len, batches_n)
        out[domain] = score(model, bs, device)
    out["_macro"] = {
        "exact_ce": st.mean(v["exact_ce"] for d, v in out.items() if not d.startswith("_"))
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    configure_runtime()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    report = {"arms": {}}
    for label, config_path, checkpoint in ARMS:
        if not Path(checkpoint).is_file():
            report["arms"][label] = {"error": f"missing {checkpoint}"}
            continue
        print(f"--- {label} ---", flush=True)
        result = evaluate(config_path, checkpoint, args.batches, args.seq_len, device)
        report["arms"][label] = result
        print(f"    macro exact_ce = {result['_macro']['exact_ce']:.6f}", flush=True)

    arms = report["arms"]
    if all("_macro" in arms.get(a[0], {}) for a in ARMS):
        for steps in ("4500", "9000"):
            b = arms[f"baseline_{steps}"]["_macro"]["exact_ce"]
            m = arms[f"merged_{steps}"]["_macro"]["exact_ce"]
            report[f"delta_at_{steps}"] = m - b
            print(f"delta at {steps} steps: {m - b:+.6f} nats (baseline {b:.6f} -> merged {m:.6f})")

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
