#!/usr/bin/env python3
"""Generate the closed-form-solve oracle that the C++ TTT slice gets tested against.

Run: ``.venv/Scripts/python scripts/ttt_solve_oracle.py`` -> writes
``tests/fixtures/ttt_solve_oracle.json`` (manifest) and ``tests/fixtures/ttt_solve_oracle.npz``
(payload arrays). Both are committed output, so the fixture is reproducible from this file
alone.

Why the oracle is *computed by the shipped code*
------------------------------------------------
The point of a fixture is to catch a from-scratch reimplementation drifting. If this script
re-derived ``G``/``C``/``ridge``/``B`` by hand, a transcription slip here would be reproduced
faithfully by the C++ side and the test would pass while both were wrong. So the numbers come
from :meth:`hagi.train.ttt.TttRls.rls_step` itself — the same function the Python contour
calls — driven through a one-block duck-typed model (below). Only the *inputs* are synthetic
here.

The model stub is deliberately minimal: ``TttRls.__init__`` reads ``model.blocks``, and per
block ``block.adapters.ttt_lora``, ``block.mixer`` (for ``id()`` only), ``lora.r``,
``lora.hidden_size`` and ``lora.lora_B.device``. It never runs a forward pass, so a real
5120-hidden HAGI model — ~1.4 GiB of fp32 weights on this host — would buy nothing.
``TttLoraAdapter`` is constructed for real, so ``lora_A`` is the shipped QR draw and
``scaling`` is the shipped ``alpha/r``.

Math transcribed (all line numbers in ``src/hagi/train/ttt.py``)
---------------------------------------------------------------
Rows, from :meth:`TttRls._rows_for` (``:323-346``)::

    rms_stream = feat.pow(2).mean().sqrt().clamp_min(1e-12)      # :342
    rms_grad   = grad.pow(2).mean().sqrt().clamp_min(1e-12)      # :343
    target     = -stream_frac * (rms_stream / rms_grad) * grad   # :344
    phi        = feat @ lora.lora_A.to(torch.float32)            # :345
    y          = target / float(lora.scaling)                    # :346

``y`` is the *unscaled* target, so a solved ``B`` emits exactly the applied delta
(``_BlockRls`` docstring, ``:111-114``). Note the consequence for scale:
``rms(y * scaling) == stream_frac * rms(feat)`` regardless of the gradient's own magnitude —
the gradient only sets the direction. The fixture therefore feeds ``X``/``A`` and derives
``Y`` from a synthetic gradient rather than inventing ``Y`` directly.

Solve, from :meth:`TttRls.rls_step` (``:359-460``)::

    n_va   = n // 5 if (holdout and n >= 5) else ...             # :409
    split  = n - n_va                                            # :410
    nt     = tr_phi.shape[0]                                     # :421
    decay  = lam ** nt                                           # :422
    G     <- G * decay + tr_phi.T @ tr_phi                       # :423
    C     <- C * decay + tr_phi.T @ tr_y                         # :424
    if train_rows < refit_rows: return False, 0.0                # :428-429
    ridge  = G.diagonal().mean().clamp_min(1e-12) * reg          # :436
    B_cand = solve(G + ridge * I, C).T                           # :437-438
    if not (candidate_resid < current_resid): return False, 0.0  # :445-448
    lora_B.copy_(B_cand)                                         # :450
    stream = tr_phi.pow(2).mean().sqrt().clamp_min(1e-12)        # :452
    delta  = (scaling * (tr_phi @ B.T)).pow(2).mean().sqrt()     # :453
    frac   = delta / stream                                      # :454
    if frac > max_delta_rms_frac:                                # :455-459
        lora_B *= max_delta_rms_frac / frac ; frac = max_delta_rms_frac

Residual is :func:`hagi.train.ttt._resid` (``:149-152``): relative squared residual
``sum((phi @ B.T - y)^2) / clamp_min(sum(y^2), 1e-12)``, and with a zero ``B`` — the state
every case starts from — ``current_resid`` is exactly ``1.0``.

Constants are the shipped defaults of ``TttRls.__init__`` (``:182-188``): ``stream_frac=0.02``,
``reg=1e-3``, ``lam=0.9995``, ``refit_rows=64``, ``prior=1000.0``, ``max_delta_rms_frac=0.10``,
``rows_max=2048``; ``scaling = alpha/r = 1.0/8`` from ``TttLoraConfig``
(``src/hagi/config.py``). They are written into the manifest *with their line reference* so a
C++ author copies them, not guesses them.

Three cases, because one cannot prove a guard
---------------------------------------------
``T=64`` rows is not enough to reach the solve under the shipped defaults with holdout on:
``n_va = 64 // 5 = 12`` leaves ``nt = 52 < refit_rows = 64``, so the step returns
``(False, 0.0)`` and never solves. That is a real property of the shipped configuration and
case ``holdout_refit_gate`` pins it — a C++ solver that "helpfully" solves anyway is wrong.

The two solving cases run with ``holdout=False`` so all 64 rows enter ``G``/``C`` and
``nt == refit_rows``:

* ``default`` — ``stream_frac = 0.02``. The step is accepted and the delta cap does **not**
  bind (``prior = 1000`` shrinks the step far below ``stream_frac``; see the "Step bound"
  section of ``ttt.py``'s docstring), so this case exercises the solve and the accept path.
* ``cap_bound`` — ``stream_frac = 5.0`` (250x the default), chosen so the rescale at
  ``:455-459`` fires. At the default 0.02 — and even at 0.5 — the absolute ``prior`` of 1000
  holds ``delta_rms_frac`` at 4.5e-4 / 1.1e-2, below the 0.10 cap, so the branch never runs. Without this case the rescale branch — the part most likely to be
  dropped in a rewrite — would have no oracle at all.

Storage: ``.npz`` for the arrays, not JSON. ``X``/``Y`` are 64x5120 float32 (1.25 MiB each);
as JSON text they would cost ~14 MiB for the same bytes and round-trip through decimal repr,
which is a precision hazard the whole fixture exists to avoid. ``np.savez`` is bit-exact.
The manifest stays human-readable: constants, seeds, shapes, verdicts, scalars, and a sha256
per array so a tampered or stale payload is detectable without regenerating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hagi.config import TttLoraConfig  # noqa: E402
from hagi.model.adapters import TttLoraAdapter  # noqa: E402
from hagi.train.ttt import TttRls, _resid  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
MANIFEST_NAME = "ttt_solve_oracle.json"
PAYLOAD_NAME = "ttt_solve_oracle.npz"

#: Seed of record. ``A`` is ``TttLoraAdapter(..., init_seed=SEED)``, i.e. the shipped QR draw
#: (``src/hagi/model/adapters.py:212-218``) with ``torch.Generator().manual_seed(SEED)``.
SEED = 20260921
X_SEED = SEED + 1
GRAD_SEED = SEED + 2

HIDDEN = 5120  # text_config.hidden_size of Ternary-Bonsai-2-27B
RANK = 8  # matches the bootstrap adapter's rank and TttLoraConfig.rank
T_ROWS = 64
ALPHA = 1.0  # TttLoraConfig.alpha default -> scaling = 1/8

#: Shipped TttRls defaults (ttt.py:182-188), copied so the manifest states them explicitly.
DEFAULTS = {
    "stream_frac": 0.02,  # :182
    "reg": 1e-3,  # :183
    "lam": 0.9995,  # :184
    "refit_rows": 64,  # :185
    "prior": 1000.0,  # :186
    "max_delta_rms_frac": 0.10,  # :187
    "rows_max": 2048,  # :188
}
#: stream_frac for the case where the delta cap must bind. Measured: ``delta_rms_frac`` is
#: linear in ``stream_frac`` (target ∝ stream_frac → B ∝ stream_frac → delta ∝ stream_frac),
#: and 0.02 -> 4.5e-4, so 5.0 -> ~0.11 > ``max_delta_rms_frac`` = 0.10. Raising ``stream_frac``
#: is precisely the configuration ``ttt.py``'s "Step bound" section names as the one where the
#: guard binds ("It binds only when ``stream_frac`` is raised or ``prior`` lowered"), so this
#: case differs from ``default`` in that one knob and nothing else.
CAP_STREAM_FRAC = 5.0


class _StubBlock:
    """The ``block`` face ``TttRls.__init__`` reads: ``adapters.ttt_lora`` and ``mixer``."""

    def __init__(self, lora: TttLoraAdapter) -> None:
        self.adapters = SimpleNamespace(ttt_lora=lora)
        self.mixer = torch.nn.Identity()  # only id(block.mixer) is taken


class _StubModel:
    def __init__(self, blocks: list[_StubBlock]) -> None:
        self.blocks = blocks


def _lora(init_seed: int = SEED) -> TttLoraAdapter:
    return TttLoraAdapter(
        HIDDEN,
        TttLoraConfig(enabled=True, rank=RANK, alpha=ALPHA, dropout=0.0),
        init_seed=init_seed,
    )


def _rows(stream_frac: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic ``(X, grad, phi, y)`` for one block, per ``_rows_for`` (ttt.py:323-346)."""
    x = torch.randn(T_ROWS, HIDDEN, generator=torch.Generator().manual_seed(X_SEED), dtype=torch.float32)
    grad = torch.randn(
        T_ROWS, HIDDEN, generator=torch.Generator().manual_seed(GRAD_SEED), dtype=torch.float32
    )
    rms_stream = x.pow(2).mean().sqrt().clamp_min(1e-12)
    rms_grad = grad.pow(2).mean().sqrt().clamp_min(1e-12)
    target = -stream_frac * (rms_stream / rms_grad) * grad
    lora = _lora()
    phi = x @ lora.lora_A.to(dtype=torch.float32)
    return x, grad, phi, target / float(lora.scaling)


def _run_case(
    name: str,
    phi: torch.Tensor,
    y: torch.Tensor,
    *,
    holdout: bool,
    stream_frac: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    """One fresh adapter + one fresh fitter, driven through the shipped ``rls_step``.

    ``B`` and the returned verdict come from the shipped function. The pre-cap candidate is
    recovered from the shipped accumulators (``ttt.py:437-438``) *and validated*: in a case
    where the cap does not bind, the recovery must equal what the shipped code wrote. That
    turns the diagnostic into a checked transcription instead of a second opinion.
    """
    lora = _lora()
    fitter = TttRls(
        _StubModel([_StubBlock(lora)]),
        stream_frac=stream_frac,
        **{k: v for k, v in DEFAULTS.items() if k != "stream_frac"},
    )

    updated, frac = fitter.rls_step(0, phi, y, holdout=holdout)

    st = fitter._state[0]
    n = phi.shape[0]
    n_va = n // 5 if (holdout and n >= 5) else (1 if (holdout and n > 1) else 0)
    nt = n - n_va
    b_final = lora.lora_B.detach().to(torch.float32)

    scalars = {
        "updated": bool(updated),
        "delta_rms_frac": float(frac),
        "n_rows": int(n),
        "n_holdout": int(n_va),
        "n_train": int(nt),
        "decay": float(fitter.lam**nt),
        "scaling": float(lora.scaling),
        "stream_frac": stream_frac,
        "train_rows_after": int(st.train_rows),
        "refits": int(st.refits),
        "holdout_rows_counted": int(st.holdout_rows),
    }

    if not updated:
        # Say "refit_rows gate" only where it is provably the reason.
        scalars["reject_reason"] = (
            "refit_rows gate: n_train < refit_rows" if nt < fitter.refit_rows else "residual gate"
        )
        scalars["b_is_zero"] = bool(torch.count_nonzero(b_final) == 0)
        return scalars, {}

    Hb = torch.cat(st.phi_buf)
    Yb = torch.cat(st.y_buf)
    cap = fitter.max_delta_rms_frac
    ridge = float(st.G.diagonal().mean().clamp_min(1e-12) * fitter.reg)
    eye = torch.eye(st.G.shape[0], dtype=st.G.dtype)
    b_cand = torch.linalg.solve(st.G + ridge * eye, st.C).T.contiguous()

    # The guard's own arithmetic (ttt.py:452-454), on the candidate rather than the written B.
    stream = Hb.pow(2).mean().sqrt().clamp_min(1e-12)
    frac_pre = float(((float(lora.scaling) * (Hb @ b_cand.T)).pow(2).mean().sqrt() / stream).item())
    cap_bound = frac_pre > cap

    # Validation of the recovery: un-capped means the shipped write == the candidate.
    if not cap_bound and not torch.allclose(b_cand, b_final, rtol=1e-5, atol=1e-7):
        raise AssertionError("recovered candidate does not match what rls_step wrote")
    want_frac = cap if cap_bound else frac_pre
    if abs(want_frac - float(frac)) > 1e-9:
        raise AssertionError(f"guard verdict mismatch: recomputed {want_frac} vs shipped {frac}")

    scalars |= {
        "reject_reason": None,
        "b_is_zero": False,
        "window_rows": int(Hb.shape[0]),
        "ridge": ridge,
        "resid_gate_current": float(_resid(torch.zeros_like(b_cand), Hb, Yb)),
        "resid_gate_candidate": float(_resid(b_cand, Hb, Yb)),
        "resid_final": float(_resid(b_final, Hb, Yb)),
        "delta_rms_frac_pre_cap": frac_pre,
        "cap_bound": bool(cap_bound),
    }
    arrays = {
        f"B_{name}": b_final.numpy(),
        f"B_cand_{name}": b_cand.numpy(),
        f"G_{name}": st.G.detach().numpy(),
        f"C_{name}": st.C.detach().numpy(),
    }
    return scalars, arrays


def generate() -> tuple[dict, dict[str, np.ndarray]]:
    x, grad, phi_default, y_default = _rows(DEFAULTS["stream_frac"])
    _x, _g, _p, y_cap = _rows(CAP_STREAM_FRAC)

    gate_case, gate_arrays = _run_case(
        "holdout_refit_gate", phi_default, y_default, holdout=True, stream_frac=DEFAULTS["stream_frac"]
    )
    default_case, default_arrays = _run_case(
        "default", phi_default, y_default, holdout=False, stream_frac=DEFAULTS["stream_frac"]
    )
    cap_case, cap_arrays = _run_case(
        "cap_bound", phi_default, y_cap, holdout=False, stream_frac=CAP_STREAM_FRAC
    )

    arrays: dict[str, np.ndarray] = {
        "A": _lora().lora_A.detach().to(torch.float32).numpy(),  # (HIDDEN, RANK), as hagi stores it
        "X": x.numpy(),
        "grad": grad.numpy(),
        "phi": phi_default.numpy(),
        "Y_default": y_default.numpy(),
        "Y_cap_bound": y_cap.numpy(),
        **gate_arrays,
        **default_arrays,
        **cap_arrays,
    }

    manifest = {
        "kind": "hagi-ttt-solve-oracle",
        "generated_by": "scripts/ttt_solve_oracle.py",
        "source_math": {
            "rows": "src/hagi/train/ttt.py:323-346 (TttRls._rows_for)",
            "solve": "src/hagi/train/ttt.py:359-460 (TttRls.rls_step)",
            "residual": "src/hagi/train/ttt.py:149-152 (_resid)",
            "accumulators": "src/hagi/train/ttt.py:108-125 (_BlockRls), :246-250 (prior init)",
            "A_draw": "src/hagi/model/adapters.py:47-90 (_qr_orthonormal), :212-218 (init_seed)",
            "scaling": "src/hagi/model/adapters.py:210 (alpha/r), src/hagi/config.py TttLoraConfig",
        },
        "constants": {
            **DEFAULTS,
            "alpha": ALPHA,
            "scaling": ALPHA / RANK,
            "cap_stream_frac": CAP_STREAM_FRAC,
            "hidden": HIDDEN,
            "rank": RANK,
            "t_rows": T_ROWS,
        },
        "seeds": {
            "A_init_seed": SEED,
            "X": X_SEED,
            "grad": GRAD_SEED,
            "generator": "torch.Generator().manual_seed(n) on cpu, float32",
        },
        "storage_note": (
            "arrays live in " + PAYLOAD_NAME + ": 64x5120 float32 is 1.25 MiB binary but ~6 MiB of "
            "decimal JSON text per matrix, and JSON round-trips through repr instead of preserving "
            "the exact float32 pattern. np.savez is bit-exact."
        ),
        "payload": PAYLOAD_NAME,
        "arrays": {
            k: {
                "shape": list(v.shape),
                "dtype": np.dtype(v.dtype).name,
                "sha256": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
            }
            for k, v in sorted(arrays.items())
        },
        "cases": {
            "holdout_refit_gate": gate_case,
            "default": default_case,
            "cap_bound": cap_case,
        },
    }
    return manifest, arrays


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=FIXTURE_DIR)
    args = ap.parse_args(argv)

    manifest, arrays = generate()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / PAYLOAD_NAME, **arrays)
    (args.out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out_dir / MANIFEST_NAME} + {args.out_dir / PAYLOAD_NAME}")
    for name, case in manifest["cases"].items():
        print(
            f"  {name:20s} updated={case['updated']!s:5s} frac={case['delta_rms_frac']:.6g} "
            f"n_train={case['n_train']} " + (f"reason={case['reject_reason']}" if not case["updated"] else "")
        )
    print("  arrays: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in sorted(arrays.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
