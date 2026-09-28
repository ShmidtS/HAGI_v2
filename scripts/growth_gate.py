"""Growth gate v3: the self-development trigger, upgraded by the full
Lean program (Select/Concat/Ambig/Grow + the synthesis review).

The synthesis (post-formalization audit) established the core
distinction:

    growth = capacity expansion (identity-preserving, free)
           + information recombination (needs residual disagreement)

The Jensen gap -- mean(single CE) - merged CE on the SAME windows --
is the direct measurement of "how much disagreement the ensemble is
still averaging" (Concat.lean lse-Jensen: gap >= 0 always; Ambig.lean:
gap = 0 at consensus). This replaces the top-2 ambiguity heuristic as
the primary saturation signal: ambiguity is a proxy whose link to the
gap is NOT proven; the gap itself is the theorem's quantity.

Candidate selection follows Select.lean literally:

    standalone CE   -> cheap pre-filter (certifiedGain bound)
    ensemble dCE    -> the DECISION metric: merge the finalist into the
                       ladder and measure the real DeltaCE
    (selection_hurts: best leaf != best addition to the ensemble)

Exit code 0 + JSON on stdout.

Usage:
    python scripts/growth_gate.py --configs configs/v3leaf_s800*.yaml \
        [--candidate-configs extra.yaml ...] \
        [--ensemble-test-top K] \
        --prev-jensen-gap 0.34 --prev-ce 6.06
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
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.checkpoint import config_from_dict, load_payload  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402

EPS = 0.0021  # measured floor (temperature_correction.md); not magic
TOL_AMB = 0.002  # ambiguity change below this = "flat" (secondary)
TOL_CE = 0.01   # CE change below this = "flat"
TOL_GAP = 0.005  # Jensen-gap change below this = "disagreement exhausted"


def _windows(weights: dict) -> list[tuple[np.ndarray, float]]:
    """Tail windows of every configured corpus (the shared yardstick)."""
    out = []
    for name, weight in weights.items():
        p = ROOT / f"data/{name}.compact.bin"
        if not p.is_file():
            continue
        total = p.stat().st_size // 4
        with p.open("rb") as fh:
            fh.seek((total - 2_000_000) * 4)
            t = np.frombuffer(fh.read(300_000 * 4), dtype=np.uint32).astype(np.int64)
        out.append((t, float(weight)))
    return out


def _eval(model, wins, device) -> float:
    """Weighted exact CE over the shared windows."""
    tot, ntok = 0.0, 0.0
    with torch.no_grad():
        for t, w in wins:
            ids = torch.from_numpy(t[:2048]).reshape(4, 512)
            x, y = ids[:, :-1].to(device), ids[:, 1:].to(device)
            out = model(x, targets=y)
            tot += float(out.ce) * y.numel() * w
            ntok += y.numel() * w
    return tot / ntok


def _merged(cfg_template, leaf_cfg, states, n, device, s0):
    """Build the flat-n merged ensemble (clean path: swiglu +
    effective_sparse, no mixers, scale 1/n)."""
    cfg = load_config(str(cfg_template))
    cfg.merge.mixer_type = "swiglu"
    cfg.merge.expert_weight_source = "effective_sparse"
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
    m = merge_experts(cfg, states, n_mixers=0).to(device).to(torch.bfloat16).eval()
    with torch.no_grad():
        m.head.logit_scale.fill_(s0 / n)
    return m


def measure(configs: list[str], with_jensen: bool = True) -> dict:
    """Merged CE + ambiguity + (optionally) the Jensen gap: the mean
    single-leaf CE and the merged CE on the SAME windows; the gap is
    the theorem's quantity (>= 0 by lse-Jensen; 0 at consensus)."""
    configure_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(configs)
    states, s0 = [], None
    leaf_cfg = None
    for c in configs:
        cc = load_config(c)
        pl = load_payload(str(Path(cc.train.checkpoint_dir) / "step-0000800.pt"), "cpu")
        states.append(pl["model"])
        if leaf_cfg is None:
            leaf_cfg = cc
            s0 = float(pl["model"]["head.logit_scale"])
    wins = _windows(leaf_cfg.train.data.weights)
    m = _merged(ROOT / "configs/flat9_ft_m1.yaml", leaf_cfg, states, n, device, s0)
    ce_m = _eval(m, wins, device)
    # ambiguity (secondary signal; kept for continuity with v1/v2 logs)
    amb, pos = 0, 0
    with torch.no_grad():
        for t, _w in wins:
            ids = torch.from_numpy(t[:2048]).reshape(4, 512)
            x, y = ids[:, :-1].to(device), ids[:, 1:].to(device)
            out = m(x, targets=y)
            probs = m.head.logits(out.hidden).float().softmax(-1)
            top2 = probs.topk(2, dim=-1).values
            amb += int(((top2[..., 0] - top2[..., 1]) < EPS).sum())
            pos += y.numel()
    del m
    torch.cuda.empty_cache()
    res = {"n_leaves": n, "ce": ce_m, "ambiguity": amb / pos,
           "eps": EPS, "tau": 2 * math.atanh(EPS)}
    if not with_jensen:
        return res
    # Jensen gap: mean single CE on the SAME windows vs merged CE.
    single_ces = []
    for c in configs:
        cc = load_config(c)
        pl = load_payload(str(Path(cc.train.checkpoint_dir) / "step-0000800.pt"), "cpu")
        mm = HAGI(config_from_dict(pl["config"])).to(device).to(torch.bfloat16).eval()
        mm.load_state_dict(pl["model"], strict=True)
        with torch.no_grad():
            mm.head.logit_scale.data.fill_(float(pl["model"]["head.logit_scale"]))
        single_ces.append(_eval(mm, wins, device))
        del mm
        torch.cuda.empty_cache()
    res["mean_single_ce"] = float(np.mean(single_ces))
    res["jensen_gap"] = res["mean_single_ce"] - ce_m
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--candidate-configs", nargs="+", default=None,
                    help="pool candidates: ranked by certifiedGain (pre-filter), "
                         "then by the true ensemble dCE (decision metric)")
    ap.add_argument("--ensemble-test-top", type=int, default=2,
                    help="how many pre-filter finalists get the (expensive) "
                         "true ensemble-dCE merge test")
    ap.add_argument("--prev-ambiguity", type=float, default=None)
    ap.add_argument("--prev-ce", type=float, default=None)
    ap.add_argument("--prev-jensen-gap", type=float, default=None,
                    help="previous level's Jensen gap (primary saturation signal)")
    args = ap.parse_args()

    cur = measure(args.configs)
    result = {**cur}
    note = []

    # --- Candidate selection (Select.lean: two-stage) ---
    if args.candidate_configs:
        n = cur["n_leaves"]
        pool = []
        for c in args.candidate_configs:
            cc = load_config(c)
            ck = Path(cc.train.checkpoint_dir) / "step-0000800.pt"
            if not ck.is_file():
                continue
            pl = load_payload(str(ck), "cpu")
            mm = HAGI(config_from_dict(pl["config"])).to(
                "cuda" if torch.cuda.is_available() else "cpu").to(torch.bfloat16).eval()
            mm.load_state_dict(pl["model"], strict=True)
            wins = _windows(cc.train.data.weights)
            with torch.no_grad():
                mm.head.logit_scale.data.fill_(float(pl["model"]["head.logit_scale"]))
            ce_c = _eval(mm, wins, "cuda" if torch.cuda.is_available() else "cpu")
            pool.append({"config": c, "payload": pl, "ce": ce_c,
                         "certified_gain": certified_gain(n, cur["mean_single_ce"], ce_c)})
            del mm
            torch.cuda.empty_cache()
        # Stage 1 (pre-filter): certifiedGain bound, cheap.
        pool.sort(key=lambda p: p["ce"])
        finalists = pool[:max(1, args.ensemble_test_top)]
        # Stage 2 (decision): the TRUE ensemble dCE for each finalist
        # -- merge into the ladder, measure. selection_hurts makes this
        # the only sound decision metric; standalone CE is a pre-filter.
        leaf_cfg = load_config(args.configs[0])
        base_states = [load_payload(
            str(Path(load_config(c).train.checkpoint_dir) / "step-0000800.pt"), "cpu"
        )["model"] for c in args.configs]
        s0 = float(base_states[0]["head.logit_scale"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        wins = _windows(leaf_cfg.train.data.weights)
        for f in finalists:
            m = _merged(ROOT / "configs/flat9_ft_m1.yaml", leaf_cfg,
                        base_states + [f["payload"]["model"]], n + 1, device, s0)
            ce_new = _eval(m, wins, device)
            f["ensemble_ce"] = ce_new
            f["ensemble_dce"] = ce_new - cur["ce"]
            del m
            torch.cuda.empty_cache()
        result["candidates"] = [
            {k: v for k, v in f.items() if k not in ("payload",)} for f in pool]
        best_ens = min(finalists, key=lambda f: f["ensemble_dce"])
        result["best_candidate_ensemble_dce"] = best_ens["ensemble_dce"]
        result["best_candidate_config"] = best_ens["config"]
        if best_ens["ensemble_dce"] < -EPS:
            note.append(
                f"best candidate {Path(best_ens['config']).name} improves the "
                f"ensemble by {-best_ens['ensemble_dce']:.4f} nats (true dCE) -- "
                f"complementary leaf, GROW is real")
        else:
            note.append(
                f"no finalist improves the ensemble (best true dCE "
                f"{best_ens['ensemble_dce']:+.4f}) -- pool disagreement is "
                f"exhausted; only EXHAUSTED (data) remains")

    # --- Saturation verdicts ---
    if args.prev_ce is None:
        verdict = "BASELINE"
        reason = "first level: record ce/jensen_gap, compare at the next one"
    else:
        d_ce = cur["ce"] - args.prev_ce
        if args.prev_jensen_gap is not None:
            d_gap = cur["jensen_gap"] - args.prev_jensen_gap
        else:
            d_gap = None
        if d_gap is not None and abs(d_gap) < TOL_GAP and d_ce > -TOL_CE:
            verdict, reason = "SATURATED-DISAGREEMENT", (
                f"Jensen gap flat ({d_gap:+.4f}): the ensemble averages no new "
                f"disagreement -- grow the ELEMENT (data), not the count")
        elif d_ce < -TOL_CE:
            verdict, reason = "GROW", (
                f"CE improving ({d_ce:+.4f}) with live gap "
                f"({cur['jensen_gap']:.4f}): disagreement still paying")
        elif d_ce > TOL_CE:
            verdict, reason = "SATURATED-AMBIGUITY", (
                f"CE worsening ({d_ce:+.4f}): this level is past the optimum; "
                f"select leaves or improve the element")
        else:
            verdict, reason = "EXHAUSTED", (
                "CE flat and gap flat: this leaf recipe is done; change the data mix")
    if note:
        reason = reason + "; " + "; ".join(note)
    print(json.dumps({**result, "verdict": verdict, "reason": reason}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
