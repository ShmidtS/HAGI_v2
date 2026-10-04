"""Growth gate v5: the (G_F, R_repr) verdict on gen6_joint (GrowthGate.lean).

Round 34 replaced the CE-plateau heuristic with a two-dimensional
verdict, and Phase B wired every port but this one -- growth_verdict
existed in formal.py with no caller. This script is the caller, on the
gen6 lineage (gen6_joint <- gen6_merged <- 3 gen4 experts):

G_F = tau * KL(q || p_E)     (FreeEnergy P1a/P2c: the joint/TTT
                              currency; q = gen6_joint, p_E = the
                              logit-mean ensemble of the 3 gen4
                              experts -- the consensus a LoRA contour
                              would distill toward). tau = 1, nats.

R_repr = ||g* - g|| / ||g||  (DesignOpt: the representation error --
                              how far the SafeQP projection d* = g +
                              sum lam_i g_i must move the mixed
                              gradient to keep every corpus safe.
                              No conflict -> lambda = 0 -> R_repr = 0:
                              the architecture already represents the
                              mixed gradient exactly).

Thresholds: split-half noise brackets (equilibrium_bracket's delta
estimated as the |A - B| difference between window halves), not hand
constants -- the same discipline as the eps certificate in
eval_domains.

Usage:
    python scripts/growth/gen7_gate.py \
        --student configs/dbridge_gen6_joint.yaml \
        --experts sib1=configs/dbridge_gen4_fresh_sib1.yaml \
                  lang=configs/dbridge_gen4_sib_lang.yaml \
                  code=configs/dbridge_gen4_sib_code.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "growth"))

from hagi.config import load_config  # noqa: E402
from hagi.model.formal import growth_verdict, safe_qp_solve  # noqa: E402
from hagi.model.merge import MergedHAGI  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.checkpoint import config_from_dict, load_payload  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402
from hagi.train.safeqp_controller import corpus_grad_gram  # noqa: E402


def _windows(weights: dict, n_win: int, span: int) -> list[tuple[np.ndarray, float]]:
    """Tail windows per corpus, canonical weights (growth_gate pattern)."""
    out = []
    for name, weight in weights.items():
        p = ROOT / f"data/{name}.compact.bin"
        if not p.is_file():
            continue
        total = p.stat().st_size // 4
        with p.open("rb") as fh:
            fh.seek((total - 2_000_000) * 4)
            t = np.frombuffer(fh.read(span * n_win * 4), dtype=np.uint32).astype(np.int64)
        for w in range(n_win):
            out.append((t[w * span:(w + 1) * span], float(weight)))
    return out


def _load(config_path: str):
    cc = load_config(config_path)
    d = Path(cc.train.checkpoint_dir)
    steps = sorted(d.glob("step-*.pt"))
    if not steps:
        raise FileNotFoundError(f"no checkpoints in {d}")
    pl = load_payload(str(steps[-1]), "cpu")
    model_cls = MergedHAGI if getattr(cc, "merge", None) and cc.merge.enabled else HAGI
    mm = model_cls(config_from_dict(pl["config"])).eval()
    mm.load_state_dict(pl["model"], strict=True)
    with torch.no_grad():
        mm.head.logit_scale.data.fill_(float(pl["model"]["head.logit_scale"]))
    return mm


def _logits(model, x) -> torch.Tensor:
    out = model(x)
    return model.head.logits(out.hidden)


def measure_g_f(student, experts, wins: list[tuple[np.ndarray, float]],
                bs: int = 2, seq: int = 1024) -> list[float]:
    """Per-window tau*KL(q || p_E): student vs the logit-mean ensemble.

    Streaming S pattern (growth_gate): one expert forward at a time,
    the ensemble logits are S/n, the student logits are its own pass.
    """
    n = len(experts)
    per_win = []
    with torch.no_grad():
        for t, _w in wins:
            ids = torch.from_numpy(t[:bs * seq]).reshape(bs, seq)
            x = ids[:, :-1]
            S = None
            for mm in experts:
                z = _logits(mm, x).double()
                S = z if S is None else S + z
            lp_e = torch.log_softmax(S / n, dim=-1)
            lq = torch.log_softmax(_logits(student, x).double(), dim=-1)
            q = lq.exp()
            kl = (q * (lq - lp_e)).sum(-1)  # [B, T-1]
            per_win.append(float(kl.mean()))
    return per_win


def measure_r_repr(student, wins: list[tuple[np.ndarray, float]],
                   weights: list[float], bs: int = 2, seq: int = 1024) -> float:
    """||g* - g||/||g|| from the SafeQP projection on corpus gradients.

    corpus_grad_gram returns (lambda, cos, norms) but not the mean
    gradient's norm -- both are recoverable from the Gram it reports:
    ||g||^2 = w'Gw (g = sum w_i g_i) and ||g - d*||^2 = lam'G lam
    (d* = g + sum lam_i g_i). No conflict -> lambda = 0 -> R_repr = 0.
    """
    names = [f"corpus{i}" for i in range(len(wins))]
    samples = []
    for (t, _w), name in zip(wins, names):
        ids = torch.from_numpy(t[:bs * seq]).reshape(bs, seq)
        samples.append((name, {"input_ids": ids[:, :-1], "targets": ids[:, 1:]}))
    res = corpus_grad_gram(student, samples, device="cpu",
                           corpus_weights=weights)
    if res["n_conflicts"] == 0:
        return 0.0
    lam = torch.tensor(res["safe_qp_lambda"], dtype=torch.float64)
    cos = torch.tensor(res["cos"], dtype=torch.float64)
    norms = torch.tensor(res["norms"], dtype=torch.float64)
    G = cos * norms[:, None] * norms[None, :]
    w = torch.tensor(weights, dtype=torch.float64)
    w = w / w.sum()
    g_norm = float(math.sqrt(max(float(w @ (G @ w)), 0.0)))
    return float(math.sqrt(max(float(lam @ (G @ lam)), 0.0)) / max(g_norm, 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", required=True)
    ap.add_argument("--experts", nargs="+", required=True, metavar="NAME=CONFIG")
    ap.add_argument("--n-win", type=int, default=4, help="windows per corpus")
    args = ap.parse_args()

    configure_runtime()
    cfg = load_config(args.student)
    weights = cfg.train.data.weights
    n_corpora = len([w for w in weights
                     if (ROOT / f"data/{w}.compact.bin").is_file()])

    student = _load(args.student)
    experts = [_load(s.split("=", 1)[1]) for s in args.experts]

    # G_F on 2*n_win windows per corpus; halves A/B for the noise bracket
    wins = _windows(weights, args.n_win * 2, 2048)
    kl = measure_g_f(student, experts, wins)
    half = len(kl) // 2
    gf_a, gf_b = float(np.mean(kl[:half])), float(np.mean(kl[half:]))
    g_f = float(np.mean(kl))
    eps_g = abs(gf_a - gf_b)

    # R_repr on one window per corpus (backward is expensive on CPU);
    # the A/B bracket comes from two disjoint window sets.
    span = 2 * 1024
    tail_wins = _windows(weights, 4, span)
    per_corpus = {}
    for t, w in tail_wins:
        per_corpus.setdefault(w, []).append(t)
    keys = sorted(per_corpus)
    set_a = [(per_corpus[k][0], k) for k in keys]
    set_b = [(per_corpus[k][1], k) for k in keys]
    wts = [k for _, k in set_a]
    r_a = measure_r_repr(student, set_a, wts)
    r_b = measure_r_repr(student, set_b, wts)
    r_repr = 0.5 * (r_a + r_b)
    eps_r = abs(r_a - r_b)

    verdict = growth_verdict(g_f, r_repr, eps_g, eps_r)
    out = {
        "student": args.student,
        "experts": args.experts,
        "n_windows_per_corpus": args.n_win * 2,
        "n_corpora": n_corpora,
        "g_f_nats": g_f,
        "g_f_half_a": gf_a,
        "g_f_half_b": gf_b,
        "eps_g": eps_g,
        "r_repr": r_repr,
        "r_repr_half_a": r_a,
        "r_repr_half_b": r_b,
        "eps_r": eps_r,
        "verdict": verdict,
    }
    print(json.dumps(out, indent=2))
    print(f"\nVERDICT: {verdict}")
    print(f"  G_F = {g_f:.4f} nats (eps_g {eps_g:.4f}) -- "
          f"{'HIGH' if g_f >= eps_g else 'low'} free energy")
    print(f"  R_repr = {r_repr:.4f} (eps_r {eps_r:.4f}) -- "
          f"{'HIGH' if r_repr >= eps_r else 'low'} representation error")
    if verdict == "TTT/LORA":
        print("  -> open the LoRA contour: distill q toward p_E (P1b")
        print("     on-policy RKL) instead of growing the ensemble.")
    elif verdict == "GROW":
        print("  -> grow (mechanism by ComputeBudget argmax DeltaG_F/DeltaC).")
    elif verdict == "STOP":
        print("  -> converged: both channels dead.")
    else:
        print("  -> teacher-check: contradiction, re-audit the ensemble.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
