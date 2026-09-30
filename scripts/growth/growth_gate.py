"""Growth gate v4: exact logit-algebra candidate selection.

The GapLaw synthesis (post-formalization review, priorities A+B)
removed the last expensive step from the growth loop: the merged
ensemble's CE is ``LSE(m_N) - m_N[t]`` with ``m_N = S/N`` the mean of
leaf LOGITS -- no wide model, no merge_experts, no merged forward.
The candidate decision metric is the exact ``CE_{N+1} - CE_N`` from
ONE candidate forward (ensemble_delta_ce); the exact Jensen gap is
streamed (jensen_gap_accum, O(one-leaf) memory instead of [N,B,T,V]).

Selection ladder (Select.lean + the synthesis):
  1. standalone CE          -- free byproduct of the leaf pass
  2. certifiedGain bound    -- cheap pre-filter (monotone in c)
  3. 1 - rho complementarity -- cheap pre-filter (synthesis §3)
  4. exact ensemble dCE      -- THE decision metric, one forward

G_inf status note (synthesis §7): Gap(N) ~ G_inf(1-1/N) is an
empirically-confirmed MODEL (docstring-level in GapLaw.lean), not a
Lean theorem; saturated_threshold() reports it as a model-based
prediction, and the gate's saturation verdicts still rest on the
measured gap trajectory between adjacent levels.

Usage:
    python scripts/growth_gate.py --configs configs/v3leaf_s800*.yaml \
        [--candidate-configs extra.yaml ...] \
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
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '../../..'))
for _p in (_HERE, _REPO, os.path.join(_REPO, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hagi.config import load_config  # noqa: E402
from hagi.model.formal import (  # noqa: E402
    certified_gain,
    complementarity,
    ensemble_delta_ce,
    jensen_gap_accum,
    saturated_threshold,
)
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.checkpoint import config_from_dict, load_payload  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402

EPS = 0.0021  # measured floor (temperature_correction.md); not magic
TOL_AMB = 0.002  # ambiguity change below this = "flat" (secondary)
TOL_CE = 0.01   # CE change below this = "flat"
TOL_GAP = 0.005  # gap change below this = "disagreement exhausted"


def _windows(weights: dict) -> list[tuple[np.ndarray, float]]:
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


def _leaf_logits(model, x) -> torch.Tensor:
    """Per-position leaf logits at the leaf's own head scale."""
    out = model(x)
    return model.head.logits(out.hidden)


def _find_checkpoint(cc) -> Path:
    """Resolve the leaf's final checkpoint: prefer the trained
    step, fall back to any step-*.pt present (TODO from round 14:
    the step parameter was hardcoded; sweep the directory instead)."""
    d = Path(cc.train.checkpoint_dir)
    # the TRAINED checkpoint is the largest step present (the leaf's
    # final state); prefer the config's full-training step if present.
    steps = sorted(d.glob("step-*.pt"))
    if not steps:
        raise FileNotFoundError(f"no checkpoints in {d}")
    return steps[-1]


def _load_leaf(config_path: str, device: str):
    cc = load_config(config_path)
    ck = _find_checkpoint(cc)
    pl = load_payload(str(ck), "cpu")
    mm = HAGI(config_from_dict(pl["config"])).to(device).to(torch.bfloat16).eval()
    mm.load_state_dict(pl["model"], strict=True)
    with torch.no_grad():
        mm.head.logit_scale.data.fill_(float(pl["model"]["head.logit_scale"]))
    return mm, pl


def measure(configs: list[str], with_jensen: bool = True) -> dict:
    """Streaming exact measurement: one forward per leaf, no merge.

    Maintains S (running logit sum) and A (running lse sum); the gap
    and the ensemble CE fall out by algebra (GapLaw twoGap identity +
    synthesis §5). The old path (merge_experts -> wide forward ->
    per-leaf CE passes) is gone: O(N) forwards total.
    """
    configure_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(configs)
    wins = _windows(load_config(configs[0]).train.data.weights)
    leaf_models = [_load_leaf(c, device)[0] for c in configs]
    bs, seq = 4, 512
    # accumulate over windows: per-window S/A, then weighted-average
    gap_acc, ce_acc, ntok_w = [], [], []
    with torch.no_grad():
        for t, _w in wins:
            S = None
            A = None
            ces = []
            ids = torch.from_numpy(t[:2048]).reshape(bs, seq)
            x, y = ids[:, :-1].to(device), ids[:, 1:].to(device)
            for mm in leaf_models:
                z = _leaf_logits(mm, x).double()  # [B,T,V]
                lse = torch.logsumexp(z, -1)      # [B,T]
                lp = torch.log_softmax(z, -1)
                ces.append(float(-(lp.gather(-1, y.unsqueeze(-1))).mean()))
                S = z if S is None else S + z
                A = lse if A is None else A + lse
            gap = jensen_gap_accum(S, A, n)          # [B,T] exact
            from hagi.model.formal import ensemble_ce_logits
            ce = ensemble_ce_logits(S / n, y)         # [B,T] exact
            gap_acc.append(float(gap.mean()))
            ce_acc.append(float(ce.mean()))
            ntok_w.append(1)
    for mm in leaf_models:
        del mm
    torch.cuda.empty_cache()
    res = {"n_leaves": n, "ce": float(np.mean(ce_acc)),
           "jensen_gap": float(np.mean(gap_acc)), "eps": EPS,
           "tau": 2 * math.atanh(EPS)}
    # ambiguity (secondary continuity signal) needs the merged probs:
    # p = softmax(S/n); keep the last window's S for it.
    amb, pos = 0, 0
    with torch.no_grad():
        for t, _w in wins:
            ids = torch.from_numpy(t[:2048]).reshape(bs, seq)
            x = ids[:, :-1].to(device)
            # recompute S for one window only (cheap, N forwards)
            S = None
            for mm in [_load_leaf(c, device)[0] for c in configs]:
                z = _leaf_logits(mm, x).double()
                S = z if S is None else S + z
            p = torch.softmax(S / n, -1)
            top2 = p.topk(2, dim=-1).values
            amb += int(((top2[..., 0] - top2[..., 1]) < EPS).sum())
            pos += p.numel()
    res["ambiguity"] = amb / pos
    # model-based saturation prediction (synthesis §7: NOT a theorem)
    res["n_star_model"] = saturated_threshold(res["jensen_gap"])
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--candidate-configs", nargs="+", default=None,
                    help="ranked by: standalone CE -> certifiedGain -> "
                         "1-rho -> exact ensemble dCE (the decision)")
    ap.add_argument("--ensemble-test-top", type=int, default=3)
    ap.add_argument("--prev-ambiguity", type=float, default=None)
    ap.add_argument("--prev-ce", type=float, default=None)
    ap.add_argument("--prev-jensen-gap", type=float, default=None)
    args = ap.parse_args()

    cur = measure(args.configs)
    result = {**cur}
    note = []

    if args.candidate_configs:
        n = cur["n_leaves"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        wins = _windows(load_config(args.configs[0]).train.data.weights)
        bs, seq = 4, 512
        # pool running stats (reuse windows)
        pool = []
        for c in args.candidate_configs:
            mm, _ = _load_leaf(c, device)
            tot, ntok = 0.0, 0
            with torch.no_grad():
                for t, _w in wins:
                    ids = torch.from_numpy(t[:2048]).reshape(bs, seq)
                    x, y = ids[:, :-1].to(device), ids[:, 1:].to(device)
                    z = _leaf_logits(mm, x).double()
                    lp = torch.log_softmax(z, -1)
                    tot += float(-(lp.gather(-1, y.unsqueeze(-1))).mean())
                    ntok += 1
            pool.append({"config": c, "model": mm, "ce": tot / ntok})
        for p in pool:
            p["certified_gain"] = certified_gain(n, cur["ce"], p["ce"])
        # 1-rho prefilter needs pool mean deviation: approximate via
        # the candidate's own logits vs pool mean logits on windows.
        # Cheap proxy: correlation of the candidate's deviation from
        # ITS OWN mean across vocab vs the same for the pool would
        # need the pool logits again; use the certifiedGain rank for
        # prefilter order and reserve 1-rho for the finalist stage.
        pool.sort(key=lambda p: p["ce"])
        finalists = pool[:max(1, args.ensemble_test_top)]
        # Synthesis §28: cache the pool's running logit sum S ONCE --
        # M(N+1) forwards become N+M. S does not depend on the candidate.
        leaf_models = [_load_leaf(c, device)[0] for c in args.configs]
        S_cache = []
        with torch.no_grad():
            for t, _w in wins:
                ids = torch.from_numpy(t[:2048]).reshape(bs, seq)
                x = ids[:, :-1].to(device)
                S = None
                for mm in leaf_models:
                    z = _leaf_logits(mm, x).double()
                    S = z if S is None else S + z
                S_cache.append(S.cpu())  # [B,T,V] per window, float64
        for f in finalists:
            dces, comps = [], []
            with torch.no_grad():
                for wi, (t, _w) in enumerate(wins):
                    ids = torch.from_numpy(t[:2048]).reshape(bs, seq)
                    x, y = ids[:, :-1].to(device), ids[:, 1:].to(device)
                    S = S_cache[wi].to(device)
                    zc = _leaf_logits(f["model"], x).double()
                    dces.append(float(ensemble_delta_ce(S, n, zc, y).mean()))
                    # complementarity proxy on this window
                    comps.append(complementarity(zc.flatten()[:50_000],
                                                 (S / n).flatten()[:50_000]))
            f["ensemble_dce"] = float(np.mean(dces))
            f["complementarity"] = float(np.mean(comps))
            del f["model"]
        result["candidates"] = pool
        best = min(finalists, key=lambda f: f["ensemble_dce"])
        result["best_candidate_ensemble_dce"] = best["ensemble_dce"]
        result["best_candidate_config"] = best["config"]
        if best["ensemble_dce"] < -EPS:
            note.append(
                f"best candidate {Path(best['config']).name} improves the "
                f"ensemble by {-best['ensemble_dce']:.4f} nats (EXACT dCE, "
                f"one forward, no merge) -- complementary, GROW is real")
        else:
            note.append(
                f"no finalist improves the ensemble (best exact dCE "
                f"{best['ensemble_dce']:+.4f}) -- pool disagreement "
                f"exhausted; only EXHAUSTED (data) remains")

    if args.prev_ce is None:
        verdict = "BASELINE"
        reason = "first level: record ce/jensen_gap, compare at the next one"
    else:
        d_ce = cur["ce"] - args.prev_ce
        d_gap = (cur["jensen_gap"] - args.prev_jensen_gap
                 if args.prev_jensen_gap is not None else None)
        if d_gap is not None and abs(d_gap) < TOL_GAP and d_ce > -TOL_CE:
            verdict, reason = "SATURATED-DISAGREEMENT", (
                f"Jensen gap flat ({d_gap:+.4f}): the ensemble averages no "
                f"new disagreement -- grow the ELEMENT (data), not the count")
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
