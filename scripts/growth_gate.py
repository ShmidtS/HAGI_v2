"""Growth gate: the executable self-development trigger.

Closes the loop of HAGI's growth mechanics. Everything it decides on
has been measured this session (see .omc/attempts/):

- The free flat-N init-ensemble beats single leaves (lse-Jensen,
  confirmed: -0.21 nats) and post-merge training HURTS on
  homogeneous leaves (+0.39 nats) -- so growth = add leaves, assemble
  free, never fine-tune homogeneous.
- The derived ambiguity threshold eps = 0.0021 (measured floor) and
  tau = 2*atanh(eps) (Concat.lean C7: mass-gap <-> logit-gap) -- no
  magic constants.
- The exhaustion curve: ambiguity plateaus at N ~ 9-16 for
  homogeneous leaves (ladder_exhaustion_curve.md). When ambiguity
  stops falling while CE still falls, remaining error is
  SYSTEMATIC: new same-distribution copies will not fix it.

Decision rule (calibrated on the measured curve):
  measure flat-N ensemble ambiguity A_N on held-out windows
  - A_N dropping vs previous level  -> GROW: add leaves (variance
    reduction still paying)
  - A_N flat (|dA| < tol) AND CE dropping -> SATURATED-AMBIGUITY:
    variance is exhausted; grow element quality (data), not count
  - A_N flat AND CE flat -> EXHAUSTED: change the element's data
Exit code 0 + JSON on stdout for automation.

Usage:
    python scripts/growth_gate.py --configs configs/leafv3_s200{1..9}.yaml \
        --prev-ambiguity 0.056 --prev-ce 4.49
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hagi.config import load_config  # noqa: E402
from hagi.model.merge import merge_experts  # noqa: E402
from hagi.train.checkpoint import load_payload  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402

EPS = 0.0021  # measured floor (temperature_correction.md); not magic
TOL_AMB = 0.002  # ambiguity change below this = "flat"
TOL_CE = 0.01   # CE change below this = "flat"


def measure(configs: list[str]) -> dict:
    configure_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(configs)
    states, leaf_weights, leaf_cfg = [], None, None
    for c in configs:
        cc = load_config(c)
        ck = Path(cc.train.checkpoint_dir) / "step-0000800.pt"
        pl = load_payload(str(ck), "cpu")
        states.append(pl["model"])
        if leaf_cfg is None:
            leaf_cfg = cc
            leaf_weights = cc.train.data.weights
            s0 = float(pl["model"]["head.logit_scale"])
    # Build from the parent template (flat9_ft_m1 carries the right
    # per-block norm geometry) instead of widening the leaf config:
    # leaf configs have qk_norm weights whose merge needs the merged-
    # model's BlockRMSNorm layout, which merge_experts derives from
    # the config it is given -- the template has that geometry pinned.
    cfg = load_config(str(ROOT / "configs/flat9_ft_m1.yaml"))
    cfg.merge.mixer_type = "swiglu"
    cfg.merge.expert_weight_source = "effective_sparse"
    # the template's attention block may differ from the leaves' recipe
    # (rope_theta especially: leaves are theta-3000, a stale template
    # theta silently breaks the merged forward). Pin the leaf geometry.
    cfg.model.attention.rope_theta = leaf_cfg.model.attention.rope_theta
    cfg.model.attention.head_dim = leaf_cfg.model.attention.head_dim
    cfg.model.attention.sink_len = leaf_cfg.model.attention.sink_len
    cfg.model.attention.qk_norm = leaf_cfg.model.attention.qk_norm
    cfg.model.num_layers = leaf_cfg.model.num_layers
    cfg.model.loop_depth = leaf_cfg.model.loop_depth
    cfg.model.hidden_size = 128 * n
    cfg.model.attention.num_query_heads = 2 * n
    cfg.model.attention.num_kv_heads = n
    cfg.merge.n_experts = n
    cfg.train.data.weights = leaf_weights
    m = merge_experts(cfg, states, n_mixers=0).to(device).to(torch.bfloat16).eval()

    # held-out windows: tail of every configured domain corpus
    tot, ntok, amb, pos = 0.0, 0, 0, 0
    with torch.no_grad():
        m.head.logit_scale.fill_(s0 / n)
        for name, weight in cfg.train.data.weights.items():
            p = ROOT / f"data/{name}.compact.bin"
            if not p.is_file():
                continue
            total = p.stat().st_size // 4
            with p.open("rb") as fh:
                fh.seek((total - 2_000_000) * 4)
                T = np.frombuffer(fh.read(300_000 * 4), dtype=np.uint32).astype(np.int64)
            for w in range(2):
                ids = torch.from_numpy(T[w * 2048:(w + 1) * 2048]).reshape(4, 512)
                x, y = ids[:, :-1].to(device), ids[:, 1:].to(device)
                out = m(x, targets=y)
                tot += float(out.ce) * y.numel() * weight
                ntok += y.numel() * weight
                probs = m.head.logits(out.hidden).float().softmax(-1)
                top2 = probs.topk(2, dim=-1).values
                amb += int(((top2[..., 0] - top2[..., 1]) < EPS).sum())
                pos += y.numel()
    del m
    torch.cuda.empty_cache()
    return {"n_leaves": n, "ce": tot / ntok, "ambiguity": amb / pos,
            "eps": EPS, "tau": 2 * math.atanh(EPS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--prev-ambiguity", type=float, default=None,
                    help="ambiguity of the previous ladder level")
    ap.add_argument("--prev-ce", type=float, default=None)
    args = ap.parse_args()

    cur = measure(args.configs)
    if args.prev_ambiguity is None:
        verdict = "BASELINE"
        reason = "first level: record ce/ambiguity, compare at the next one"
    else:
        d_amb = cur["ambiguity"] - args.prev_ambiguity
        d_ce = cur["ce"] - args.prev_ce
        amb_flat = abs(d_amb) < TOL_AMB
        ce_flat = abs(d_ce) < TOL_CE
        if not amb_flat:
            verdict, reason = "GROW", (
                f"ambiguity {d_amb:+.4f}: variance reduction still paying, add leaves")
        elif not ce_flat:
            verdict, reason = "SATURATED-AMBIGUITY", (
                f"ambiguity flat ({d_amb:+.4f}) but CE {d_ce:+.4f}: residual error "
                f"is systematic; improve the ELEMENT (data), not the count")
        else:
            verdict, reason = "EXHAUSTED", (
                "both flat: this leaf recipe is done; change the element's data mix")
    print(json.dumps({**cur, "verdict": verdict, "reason": reason}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
