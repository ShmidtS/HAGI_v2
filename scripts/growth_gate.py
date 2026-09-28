"""Growth gate v2: the executable self-development trigger,
upgraded by the Lean program (Select/Concat/Ambig, lake build green).

Pre-quantified decision rules (all derived, no fitted constants):

- certifiedGain(N, M, c) = (M - c) / (N + 1): the certified new
  mean-CE bound after adding a candidate with standalone CE c to a
  pool of N leaves with mean-CE M (Select.lean newBound:
  new bound = (N*M + c)/(N+1)). Computable BEFORE the step -- the
  FGD U_t analogue.
- SATURATED = certifiedGain(best candidate) < EPS: if the BEST
  candidate (min c) cannot certify a gain, no candidate can
  (grow_epsilon_stop) -- the GROW axis is exhausted; only the
  EXHAUSTED axis (change the element's data) remains.
- Speed stop (Ambig.lean consensus_no_gain): expected next-leaf
  gain ~ (spread)^2 / (N(N+1)); stop when < EPS^2.
- Candidate RANKING is certified only by certifiedGain (a bound),
  never by standalone CE as the decision rule (Select.lean
  selection_hurts: a better-standalone leaf can be strictly worse
  in the ensemble). The pool decision needs an ensemble measure.

EPS = 0.0021 is the measured floor (temperature_correction.md);
tau = 2*atanh(EPS) is its logit form (Concat.lean C7).

Exit code 0 + JSON on stdout for automation.

Usage:
    python scripts/growth_gate.py --configs configs/v3leaf_s800*.yaml \
        [--candidate-configs extra_leaves.yaml ...] \
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
from hagi.model.formal import certified_gain  # noqa: E402
from hagi.model.merge import merge_experts  # noqa: E402
from hagi.train.checkpoint import load_payload  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402

EPS = 0.0021  # measured floor (temperature_correction.md); not magic
TOL_AMB = 0.002  # ambiguity change below this = "flat"
TOL_CE = 0.01   # CE change below this = "flat"

# Select.lean: the canonical certified_gain lives in hagi.model.formal;
# re-exported here for callers importing from growth_gate.
certified_gain = certified_gain


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
    ap.add_argument("--candidate-configs", nargs="+", default=None,
                    help="pool candidates: single leaves, each is ranked "
                         "by certifiedGain (a BOUND); if even the best "
                         "cannot certify >= EPS, GROW is exhausted")
    ap.add_argument("--prev-ambiguity", type=float, default=None,
                    help="ambiguity of the previous ladder level")
    ap.add_argument("--prev-ce", type=float, default=None)
    args = ap.parse_args()

    cur = measure(args.configs)
    result = {**cur}
    verdict_extra = ""

    # --- Pre-quantified GROW-axis check (Select.lean) ---
    if args.candidate_configs:
        n = cur["n_leaves"]
        mean_ce = cur["ce"]
        # NOTE: cur["ce"] is the merged-ensemble CE, which by
        # ensemble_ce_le_mean_general is <= the pool's mean single-CE.
        # Using it as M in certifiedGain is CONSERVATIVE (M smaller ->
        # gain smaller), so a certified GROW verdict is sound.
        cand_scores = {}
        for c in args.candidate_configs:
            cc = load_config(c)
            ck = Path(cc.train.checkpoint_dir) / "step-0000800.pt"
            if not ck.is_file():
                continue
            from hagi.model.model import HAGI  # noqa: E402
            from hagi.train.checkpoint import config_from_dict  # noqa: E402
            pl = load_payload(str(ck), "cpu")
            mm = HAGI(config_from_dict(pl["config"])).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            ).to(torch.bfloat16).eval()
            mm.load_state_dict(pl["model"], strict=True)
            # cheap single-leaf CE on the same windows
            tot_c, ntok_c = 0.0, 0
            with torch.no_grad():
                mm.head.logit_scale.data.fill_(
                    float(pl["model"]["head.logit_scale"]))
                for name, weight in cc.train.data.weights.items():
                    p = ROOT / f"data/{name}.compact.bin"
                    if not p.is_file():
                        continue
                    total = p.stat().st_size // 4
                    with p.open("rb") as fh:
                        fh.seek((total - 2_000_000) * 4)
                        T = np.frombuffer(fh.read(300_000 * 4),
                                          dtype=np.uint32).astype(np.int64)
                    dev = next(mm.parameters()).device
                    ids = torch.from_numpy(T[:2048]).reshape(4, 512)
                    x, y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
                    out = mm(x, targets=y)
                    tot_c += float(out.ce) * y.numel() * weight
                    ntok_c += y.numel() * weight
            cand_ce = tot_c / ntok_c
            cand_scores[c] = {"ce": cand_ce,
                               "certified_gain": certified_gain(n, mean_ce, cand_ce)}
            del mm
            torch.cuda.empty_cache()
        result["candidates"] = cand_scores
        best = min(cand_scores.values(), key=lambda s: s["ce"])
        result["best_candidate_gain"] = best["certified_gain"]
        if best["certified_gain"] < EPS:
            verdict_extra = (
                f"certified: even the best candidate (gain "
                f"{best['certified_gain']:.4f} < eps {EPS}) cannot move the "
                f"bound -- GROW axis exhausted (grow_epsilon_stop); only "
                f"EXHAUSTED (data) remains")

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
    # speed stop (Ambig.lean): expected next-leaf gain ~ spread^2/(N(N+1))
    spread = 0.0
    result["speed_stop"] = None
    if verdict == "GROW":
        # spread proxy: ambiguity is the measured disagreement floor
        spread = cur["ambiguity"]
        speed = spread ** 2 / (cur["n_leaves"] * (cur["n_leaves"] + 1))
        result["speed_stop"] = speed
        if speed < EPS ** 2:
            verdict, reason = "SATURATED-AMBIGUITY", (
                f"speed stop: next-leaf expected gain {speed:.2e} < eps^2 "
                f"{EPS**2:.2e} (consensus_no_gain) -- disagreement floor "
                f"reached; improve the ELEMENT, not the count")
    if verdict_extra:
        reason = (reason + "; " + verdict_extra) if verdict == "GROW" else verdict_extra
        if best["certified_gain"] < EPS and verdict == "GROW":
            verdict = "SATURATED-AMBIGUITY"
    print(json.dumps({**result, "verdict": verdict, "reason": reason}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
