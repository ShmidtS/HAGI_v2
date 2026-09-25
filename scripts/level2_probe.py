"""Can HAGI merge a model that is itself already merged? Level-2 probe.

The project's claim is RECURSIVE growth: experts merge into a model, and
that model merges again with others. Level 1 works (M2: -0.57 nats,
step-budget matched). Level 2 has never been run.

The recursion is explicitly blocked in one path on purpose:

    src/hagi/model/merge.py:1264
        if drop_expert_mixers:
            raise ValueError("drop_expert_mixers=True is forbidden in recursive F3")

and the hierarchical path that is allowed lives in `merge_experts`, which
train.py:262 already calls with drop_expert_mixers=True. So the question is
not "is recursion supported" but "which assembly path does a level-2 merge
have to take, and does it preserve quality or destroy it".

This probe builds a level-2 model from three level-1 merged checkpoints,
scores it against each level-1 parent on the same held-out batches, and
reports the delta. It is a measurement, not a gate: the verdict rule is
decided after seeing the numbers, and stated before the run in
reports/LEVEL2_PROBE_20260926.md.

Usage:
    python scripts/level2_probe.py --out reports/level2_probe.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "scripts"))

from eval_holdout import DOMAIN_FILES, _batches, _tail_tokens, score  # noqa: E402

from hagi.model.merge import build_model_from_payload, merge_experts  # noqa: E402
from hagi.train.checkpoint import config_from_dict, config_to_dict, load_payload  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402

_DATA = _ROOT / "data"

# Three level-1 experts trained on different domains, at the same seed.
LEVEL1 = [
    ("ru", "checkpoints/m2_expert_ru_s5678/step-0003000.pt"),
    ("en", "checkpoints/m2_expert_en_s5678/step-0003000.pt"),
    ("mathcode", "checkpoints/m2_expert_mathcode_s5678/step-0003000.pt"),
]

# For a genuine LEVEL-2 merge the experts must themselves be merged models,
# which is what the hierarchical branch in merge_experts expects: per-head
# 2D QK gains to concatenate along the head axis, and `mixers.*` keys to
# drop and replace. Plain domain experts have 1D gains (one head each) and no
# mixers, so merging them tests a widening, not recursion.
#
# THREE-WAY, per the project's DFT3 basis: n_experts stays 3, so the
# merged mixer keeps using the ternary DFT3 lift. Only two distinct merged
# checkpoints exist per seed, so the third slot repeats the 4500 one. That
# is recorded as a limitation, not hidden: a repeated expert makes this a
# test that the level-2 DFT3 path builds and does not destroy the model,
# NOT a measurement of three independent generations.
LEVEL1_MERGED = [
    ("merged_4500", "checkpoints/m2_merged_joint_s5678/step-0004500.pt"),
    ("merged_9000", "checkpoints/m2_merged_joint_s5678/step-0009000.pt"),
    ("merged_4500_dup", "checkpoints/m2_merged_joint_s5678/step-0004500.pt"),
]


def build_batches(n: int, seq_len: int) -> dict[str, list[torch.Tensor]]:
    out = {}
    for domain, names in DOMAIN_FILES.items():
        present = [
            x for x in names
            if (_DATA / f"{x}.compact.bin").is_file() or (_DATA / f"{x}.bin").is_file()
        ]
        if present:
            out[domain] = _batches(_tail_tokens(_DATA, present[0], 8_000_000), seq_len, n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--experts", choices=("merged", "plain"), default="merged",
                    help="merged = level-1 merged models (true recursion); "
                         "plain = domain experts (widening, not recursion)")
    ap.add_argument("--force-three", action="store_true", default=True,
                    help="keep n_experts=3 so the merged mixer stays on the "
                         "ternary DFT3 lift (the project's basis)")
    args = ap.parse_args()

    configure_runtime()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    batches = build_batches(args.batches, args.seq_len)

    report: dict = {"level1": {}, "level2": None, "error": None}

    states, cfgs = [], []
    source = LEVEL1_MERGED if args.experts == "merged" else LEVEL1
    report["expert_set"] = args.experts
    n = len(source)
    if args.experts == "plain":
        n = 3
    for name, path in source:
        if not (_ROOT / path).is_file():
            report["error"] = f"missing {path}"
            print(report["error"])
            print(json.dumps(report, indent=2))
            return 2
        payload = load_payload(str(_ROOT / path))
        cfg = config_from_dict(payload["config"])
        model = build_model_from_payload(cfg, payload["model"], device=device)
        model.eval()
        states.append(payload["model"])
        cfgs.append(cfg)
        ce = {d: score(model, b, device)["exact_ce"] for d, b in batches.items()}
        report["level1"][name] = {
            "checkpoint": path,
            "domains": ce,
            "macro": st.mean(ce.values()),
        }
        print(f"level1 {name}: macro {st.mean(ce.values()):.6f}", flush=True)

    # A level-2 merge: three ALREADY-MERGED models become the experts of a
    # new one. Hidden size has to be N * expert_hidden, so the level-2 config
    # widens rather than nests; that widening is itself part of what is
    # being measured, not an implementation detail to hide.
    expert_hidden = cfgs[0].model.hidden_size
    level2_cfg = config_from_dict(config_to_dict(cfgs[0]))
    level2_cfg.model.hidden_size = expert_hidden * n
    level2_cfg.merge.n_experts = n
    # hidden_size must equal num_query_heads * head_dim, so widening the
    # hidden axis by n means widening the head count by n too. Recorded
    # here because it changes the model's shape, not just its wiring.
    level2_cfg.model.attention.num_query_heads *= n
    level2_cfg.model.attention.num_kv_heads *= n
    report["level2_shape"] = {
        "expert_hidden": expert_hidden,
        "level2_hidden": level2_cfg.model.hidden_size,
        "expert_heads": cfgs[0].model.attention.num_query_heads,
        "level2_heads": level2_cfg.model.attention.num_query_heads,
        "head_dim": cfgs[0].model.attention.head_dim,
        "n_experts": n,
    }
    print(f"level2 shape: {report['level2_shape']}", flush=True)

    try:
        level2 = merge_experts(
            level2_cfg, states, n_mixers=1,
            mixer_init_scale=level2_cfg.merge.mixer_init_scale,
            drop_expert_mixers=True,
        ).to(device)
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"level-2 merge FAILED: {report['error']}")
        ce = None
    else:
        level2.eval()
        ce = {d: score(level2, b, device)["exact_ce"] for d, b in batches.items()}
        report["level2"] = {
            "domains": ce,
            "macro": st.mean(ce.values()),
            "expert_hidden": expert_hidden,
            "level2_hidden": expert_hidden * 3,
        }
        print(f"level2: macro {st.mean(ce.values()):.6f}")

    if ce is not None:
        for name, rec in report["level1"].items():
            deltas = {d: ce[d] - rec["domains"][d] for d in ce}
            print(f"  level2 - {name}: macro {st.mean(deltas.values()):+.6f} {deltas}")

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
