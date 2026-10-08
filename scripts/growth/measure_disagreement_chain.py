"""R245 disagreement-chain diagnostics (DisagreementChain.lean).

The measured mixer gain deficit factors EXACTLY through the chain
(``disagreement_chain``):

    E_raw --align--> E_aligned --trunc--> E_kept --safe--> G_safe
         --cap--> G_cap

Each conversion factor is separately measurable, and
``deficit_localizes`` licenses measuring them stage by stage: if the
total product is below the required share rho, AT LEAST ONE alpha is
below rho^(1/4) — no need to audit all four exhaustively before
finding a culprit (pigeonhole on the product).

This script measures, from TWO expert checkpoints (and optionally a
merged one):

* ``alpha_align = E_aligned / E_raw`` — the latent alignment factor
  (``alignment_factor_le_one``: alignment compresses the MEASURED
  latent disagreement; the raw reading overestimates the merge loss
  by exactly the flip share). Implementation: SVD-factor both weight
  deltas, compare latent-coordinate disagreement before vs after an
  orthogonal Procrustes alignment of the top-r factors. The
  MATRATICES are unchanged — only the measured latent disagreement
  changes.
* ``alpha_kept`` — spectral truncation of the aligned disagreement at
  ``--energy``: the kept fraction of the original aligned
  disagreement energy retained by the truncation.
* ``alpha_safe`` — logit-space disagreement proxy on a probe batch
  (labelled ``safe_proxy`` unless a true SafeQP projection is
  available): normalized logit L2 disagreement between the experts.
* ``alpha_cap`` — capability conversion from eval CE JSONs
  (``--evala``/``--evalb``), else omitted.

Empirical anchors (gamma_req/gamma_meas defaults) are FLAGGED
scale-specific observations, not constants of nature.

Usage:
    python scripts/growth/measure_disagreement_chain.py \
        --config configs/gen7.yaml --a ckpt_a.pt --b ckpt_b.pt \
        [--merged merged.pt] [--rank 64] [--energy 0.95] \
        [--gamma-req 0.018] [--gamma-meas 0.002] [-o out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _top_factors(delta: torch.Tensor, rank: int) -> torch.Tensor:
    """Latent factor matrix U_r * S_r of the top-``rank`` SVD components."""
    d = delta.to(torch.float64)
    u, s, _ = torch.linalg.svd(d, full_matrices=False)
    r = min(int(rank), d.shape[1])
    return u[:, :r] * s[:r]


def alpha_align(delta_a: torch.Tensor, delta_b: torch.Tensor, rank: int) -> dict:
    """Latent alignment factor E_aligned/E_raw (R245 stage 1).

    Both deltas are factored (SVD, top-r); the RAW latent disagreement is
    the distance between the factor coordinate matrices, the ALIGNED one is
    the distance after an orthogonal Procrustes rotation of A's factors
    onto B's. Orthogonal Procrustes minimizes ||A R - B||_F over rotations,
    so alpha_align <= 1 (``alignment_factor_le_one``): the raw reading
    overestimates the merge-induced destruction by the flip share.
    """
    a = _top_factors(delta_a, rank)
    b = _top_factors(delta_b, rank)
    r = min(a.shape[1], b.shape[1])
    a, b = a[:, :r], b[:, :r]
    um, _, vmh = torch.linalg.svd(a.T @ b)
    aligned_factors = a @ (um @ vmh)
    raw = float(torch.linalg.matrix_norm(a - b, ord="fro"))
    aligned = float(torch.linalg.matrix_norm(aligned_factors - b, ord="fro"))
    if raw <= 0.0:
        return {"alpha_align": 1.0, "E_raw": 0.0, "E_aligned": 0.0, "D": b - b}
    return {
        "alpha_align": aligned / raw,
        "E_raw": raw,
        "E_aligned": aligned,
        "D": aligned_factors - b,
    }


def alpha_kept(aligned_disagreement: torch.Tensor, energy: float) -> dict:
    """Spectral truncation factor (R245 stage 2).

    Truncate the aligned disagreement at the smallest rank whose singular
    spectrum captures ``energy`` of the total Frobenius energy;
    ``alpha_kept`` is the kept fraction of the ORIGINAL energy — how much
    disagreement the merge would actually carry at that rank budget.
    """
    d = aligned_disagreement.to(torch.float64)
    s = torch.linalg.svdvals(d)
    total = float((s ** 2).sum())
    if total <= 0.0:
        return {"alpha_kept": 1.0, "rank_kept": 0}
    cum = torch.cumsum(s ** 2, dim=0) / total
    k = int(torch.searchsorted(cum, torch.tensor(energy - 1e-12)).item()) + 1
    k = min(k, s.numel())
    kept = float((s[:k] ** 2).sum()) / total
    return {"alpha_kept": kept, "rank_kept": k, "energy_target": float(energy)}


def _alpha_cap_from_evals(evala: dict, evalb: dict) -> float | None:
    """Capability conversion from eval CE JSONs (relative CE improvement)."""
    def _ce(payload: dict) -> float | None:
        for key in ("exact_ce", "ce", "cross_entropy", "lm_loss"):
            if key in payload:
                return float(payload[key])
        return None

    ca, cb = _ce(evala), _ce(evalb)
    if ca is None or cb is None or ca <= 0.0:
        return None
    return max(0.0, 1.0 - cb / ca)


def _load_states(path: str) -> tuple[dict | None, dict]:
    """Load a HAGI checkpoint payload or a bare state dict."""
    try:
        from hagi.train.checkpoint import load_payload

        payload = load_payload(path, "cpu")
        return payload.get("config"), payload["model"]
    except Exception:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            return state.get("config"), state["model"]
        return None, state


def _probe_logits(model_path: str, cfg, batch: dict, device: str) -> torch.Tensor:
    from hagi.model.merge import build_model_from_payload
    from hagi.train.checkpoint import config_from_dict

    config, state = _load_states(model_path)
    if config is None and cfg is not None:
        payload_cfg = config_to_plain(cfg)
        model = build_model_from_payload(
            config_from_dict(payload_cfg), state,
            n_mixers=int(getattr(cfg.merge, "n_mixers", 1)),
            mixer_init_scale=float(getattr(cfg.merge, "mixer_init_scale", 0.0)),
            device=device,
        )
    elif config is not None:
        c = config_from_dict(config)
        model = build_model_from_payload(
            c, state,
            n_mixers=int(getattr(c.merge, "n_mixers", 1)),
            mixer_init_scale=float(getattr(c.merge, "mixer_init_scale", 0.0)),
            device=device,
        )
    else:
        raise ValueError(f"cannot determine model geometry for {model_path}")
    model.eval()
    with torch.no_grad():
        out = model(
            batch["input_ids"].to(device),
            batch["targets"].to(device),
            doc_ids=batch.get("doc_ids"),
        )
        hidden = out.hidden.reshape(-1, out.hidden.shape[-1])
        return model.head.logits(hidden).float().cpu()


def config_to_plain(cfg) -> dict:
    import dataclasses

    return dataclasses.asdict(cfg)


def _probe_batch(cfg, seq_len: int = 1024) -> dict:
    """Deterministic tail-window probe batch from the heaviest corpus."""
    import numpy as np

    data_dir = Path(cfg.train.data.data_dir)
    weights = cfg.train.data.weights or {}
    if not weights:
        mix = data_dir / "mix.json"
        if mix.exists():
            spec = json.loads(mix.read_text(encoding="utf-8"))
            weights = {s["name"]: float(s.get("ratio", 1.0)) for s in spec["sources"]}
    if not weights:
        raise FileNotFoundError("no corpus weights for probe batch")
    name = max(weights, key=lambda n: weights[n])
    path = data_dir / f"{name}.compact.bin"
    if not path.exists():
        path = data_dir / f"{name}.bin"
    total = path.stat().st_size // 4
    with path.open("rb") as fh:
        fh.seek(max(total - 2_000_000, 0) * 4)
        raw = np.frombuffer(fh.read(2 * seq_len * 4), dtype=np.uint32).astype(np.int64)
    ids = torch.from_numpy(raw[: 2 * seq_len]).reshape(2, seq_len)
    return {"input_ids": ids[:, :-1], "targets": ids[:, 1:]}


def verdict(alphas: dict[str, float | None], gamma_req: float, gamma_meas: float) -> dict:
    """deficit_localizes: any alpha below the geometric share is a culprit."""
    measured = {k: v for k, v in alphas.items() if v is not None}
    if not measured or gamma_req <= 0.0 or gamma_meas <= 0.0:
        return {"geometric_share": None, "below_share": []}
    rho = gamma_meas / gamma_req
    share = rho ** (1.0 / max(1, len(measured)))
    below = sorted(k for k, v in measured.items() if v < share)
    return {"geometric_share": share, "below_share": below}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="data/model geometry YAML")
    p.add_argument("--a", required=True, help="expert A checkpoint")
    p.add_argument("--b", required=True, help="expert B checkpoint")
    p.add_argument("--merged", default=None, help="optional merged checkpoint")
    p.add_argument("--rank", type=int, default=64, help="top-r latent factors")
    p.add_argument("--energy", type=float, default=0.95, help="truncation energy")
    p.add_argument("--gamma-req", type=float, default=0.018,
                   help="required gain (flagged empirical anchor, not a constant)")
    p.add_argument("--gamma-meas", type=float, default=0.002,
                   help="measured gain (flagged empirical anchor)")
    p.add_argument("--evala", default=None, help="eval CE JSON for A (alpha_cap)")
    p.add_argument("--evalb", default=None, help="eval CE JSON for B (alpha_cap)")
    p.add_argument("-o", "--out", default=None, help="write JSON here")
    args = p.parse_args(argv)

    from hagi.config import load_config

    cfg = load_config(args.config)
    _, state_a = _load_states(args.a)
    _, state_b = _load_states(args.b)

    # alpha_align / alpha_kept per shared 2D tensor, energy-aggregated over
    # the tensor family: E = sqrt(sum of per-tensor energies). The two
    # experts' deltas against their common origin are approximated by the
    # two checkpoint tensors themselves (the deltas ARE the expert movements
    # when the shared prior is the zero reference; for init_from runs use
    # --a as the prior).
    totals = {"E_raw": 0.0, "E_aligned": 0.0, "kept": 0.0}
    for key, tb in state_b.items():
        ta = state_a.get(key)
        if (
            ta is None
            or not isinstance(tb, torch.Tensor)
            or ta.shape != tb.shape
            or tb.ndim != 2
            or not torch.is_floating_point(tb)
        ):
            continue
        delta_a = ta.double()
        delta_b = tb.double()
        al = alpha_align(delta_a, delta_b, args.rank)
        kept = alpha_kept(al["D"], args.energy)
        totals["E_raw"] += al["E_raw"] ** 2
        totals["E_aligned"] += al["E_aligned"] ** 2
        totals["kept"] += (kept["alpha_kept"] * al["E_aligned"]) ** 2
    e_raw = totals["E_raw"] ** 0.5
    e_aligned = totals["E_aligned"] ** 0.5
    alpha_align_total = e_aligned / e_raw if e_raw > 0 else 1.0
    alpha_kept_total = (totals["kept"] ** 0.5 / e_aligned) if e_aligned > 0 else 1.0

    result: dict = {
        "alpha_align": alpha_align_total,
        "alpha_kept": alpha_kept_total,
        "E_raw": e_raw,
        "E_aligned": e_aligned,
        "n_tensors": len([k for k in state_b if k in state_a]),
    }

    # alpha_safe: logit-space proxy on a probe batch.
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch = _probe_batch(cfg)
        za = _probe_logits(args.a, cfg, batch, device)
        zb = _probe_logits(args.b, cfg, batch, device)
        diff = torch.linalg.vector_norm(za - zb).item()
        scale = max(torch.linalg.vector_norm(za).item(),
                    torch.linalg.vector_norm(zb).item(), 1e-12)
        result["alpha_safe"] = diff / scale
        result["alpha_safe_kind"] = "safe_proxy"
        if args.merged:
            zm = _probe_logits(args.merged, cfg, batch, device)
            dm = max(
                torch.linalg.vector_norm(zm - za).item(),
                torch.linalg.vector_norm(zm - zb).item(),
            )
            result["merge_logit_disagreement"] = dm / scale
    except Exception as exc:  # probe is optional; weight-space still reported
        result["alpha_safe"] = None
        result["alpha_safe_error"] = str(exc)

    # alpha_cap from eval JSONs.
    if args.evala and args.evalb:
        with open(args.evala, encoding="utf-8") as fh:
            ea = json.load(fh)
        with open(args.evalb, encoding="utf-8") as fh:
            eb = json.load(fh)
        result["alpha_cap"] = _alpha_cap_from_evals(ea, eb)
    else:
        result["alpha_cap"] = None

    v = verdict(
        {
            "alpha_align": result["alpha_align"],
            "alpha_kept": result["alpha_kept"],
            "alpha_safe": result.get("alpha_safe"),
            "alpha_cap": result.get("alpha_cap"),
        },
        args.gamma_req,
        args.gamma_meas,
    )
    result.update(v)
    result["gamma_req"] = args.gamma_req
    result["gamma_meas"] = args.gamma_meas

    print(json.dumps(result, indent=2))
    if v["below_share"]:
        print(
            "verdict: deficit_localizes -> "
            + ", ".join(v["below_share"])
            + f" below geometric share {v['geometric_share']:.4f}"
        )
    else:
        print("verdict: no single stage below its geometric share")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
