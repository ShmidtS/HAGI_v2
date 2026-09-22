"""E7: rank backprop-free per-block credit producers by NUDGING, not by cosine.

Why
---
:mod:`scripts.e6_producer_validate` adjudicated producers with cosine against the
true autograd target and condemned the broadcast head error. That verdict is
annulled: arXiv 2606.21126 ("What Accuracy and Gradient Cosine Miss") shows on a
controlled audit that cosine-to-BP ranking *anti*-correlates with success --
their State Bridge scores cosine +0.322 (below FA's +0.423) yet beats every other
method by 99 pp of accuracy and 40-44x larger nudging effect, while Credit
Bridge has the *best* cosine (+0.679) and matches dead DFA. Cosine cannot
adjudicate producers. The functional metric is **nudging**: the test-loss change
from ONE step of size ``eta`` along each layer's credit direction. This script
ranks producers by that number on the small HAGI proxy, CPU only.

The contour this serves is a llama.cpp inference server with **no backward
pass**: it harvests per-block residual taps during the forward it already
computes, fits a rank-8 ``B`` closed-form, and writes it into the live adapter.
So the only question is whether a gradient-free target moves the loss.

Producers (each yields a per-block credit ``a_l``, l in 0..L-1, injected as
``h <- h - eta * a_l`` at block l's output):

* ``auto``   -- true autograd ``dCE/d(block-l output)``, from e6's
  :func:`true_targets` (which calls the shipped ``TttRls._harvest``). UPPER
  BOUND. If it does not win, the experiment's premise is wrong and this script
  says so.
* ``bcast``  -- e6's ``head_error`` vector ``e``, broadcast unchanged to every
  block. The currently-designed server producer.
* ``deep``   -- identity bridge / deep supervision:
  ``a_l = grad_{h_l} CE(head(out_norm(h_l)), y)``. Implemented by detaching each
  block output, re-running ONLY the ``out_norm -> _apply_mixers -> head.loss``
  tail on it, and taking one ``autograd.grad`` per block -- not by keeping the
  stack graph alive. Chosen because it is cheaper (L tail passes, not L backwards
  through L blocks) and because it is exactly the stated quantity: under this
  producer's own assumption the tail is the only graph connecting ``h_l`` to the
  loss.
* ``bridge_aff`` / ``bridge_lin`` -- the low-rank State Bridge, predictor fit
  closed-form (ridge least squares, no gradient descent for the predictor), with
  and without the intercept.
* ``rand``   -- same-norm random direction per block, seeded. The frozen-blocks
  analogue: **a producer that does not beat this is dead.**
* ``zero``   -- no credit. Sanity: must give ~0 loss change.
* ``bridge_aff_eqg`` / ``auto_eqb`` -- equal-norm controls (see "Norm honesty").

Bridge algebra, verified rather than trusted
--------------------------------------------
``A_l`` (r, d) is the block's frozen orthonormal ``lora_A`` transposed. e6's
``block_basis`` returns ``lora_A`` as ``[d, r]`` with orthonormal *columns*
(reduced QR of a Gaussian draw, ``model/adapters.py::_qr_orthonormal``), so
``A_l A_l^T = I_r`` is the claim under test; ``gram_dev`` below measures it, and
the whole reduction to a plain ``r x r`` least squares rests on it.

Row-vector form used here (``A`` = ``lora_A``, shape ``[d, r]``):
``X = h_l @ A``, ``Z = h_L @ A``, residual target ``R = Z - X``.

* linear:  ``W`` solves ``(X^T X + ridge*I) W = X^T R``;
  credit ``a = e + (e @ A) @ W @ A^T``, i.e. ``G = I + A^T W A`` and ``a = G^T e``.
  Identity check: ``a @ A == (e @ A) @ (I + W)``, exact iff ``A^T A == I_r``.
  (The brief wrote both ``G = I + A^T M^T A`` and ``a = e + A^T M (A e)``; those
  differ by a transpose on ``M``. The row form above is the one consistent with
  ``A G h = (I + M^T) X``, and it is checked numerically, not argued.)
* affine:  features and target centred per column, ``W`` fit on the centred
  features, intercept ``s = mean(Z) - mean(X) @ W`` kept separately;
  credit ``a = e + (e @ A) @ W @ A^T + s @ A^T``.

Pre-registered design choices (fixed BEFORE any result was seen)
---------------------------------------------------------------
1. **Primary eta = 0.01**, the paper's value. The sweep {0.001, 0.01, 0.1} is
   reported because our credit norms differ from theirs, but the ranking is read
   at 0.01.
2. **Affine form = (b), centre-and-keep-shift**, NOT (a) augmented constant
   column. Reason: appending a ones-row makes the augmented basis rows
   non-orthonormal (norm sqrt(d), and not orthogonal to ``span(A)``), which
   destroys exactly the ``A_l A_l^T = I_r`` step the r x r reduction rests on.
   (b) keeps the linear part an honest r x r solve on an orthonormal feature set
   and isolates the intercept as a separate r-vector, so ``bridge_aff`` vs
   ``bridge_lin`` measures the intercept alone with no basis confound.
3. **``h_L`` = the model's final hidden state as the head consumes it**, i.e.
   ``out_norm``-ed, pre-unembedding (``ModelOutput.hidden``). This matches the
   tuned-lens precedent (arXiv 2604.09885 sec. D/E: an affine map from an
   intermediate residual stream into the final layer *before unembedding*) and it
   puts ``Z_l`` in the same space as ``e``, which is what makes
   ``a = e + <lifted prediction>`` space-consistent. The pre-``out_norm`` target
   (last block output) is reported as a diagnostic rel-err line, not a producer.
4. **Falsifiable prediction from the precedent**: ``deep`` (identity bridge, no
   fit) LOSES to ``bridge_aff``. Intermediate-layer logits are "almost always
   significantly worse predictors than the final layer's", so a *fitted* bridge
   should beat the unfitted identity tail. If they tie, the bridge fit carries no
   information and the cheap producer is the right design.
5. **SHAPE prediction (stronger than the mean effect), pre-registered before any
   run in this session**: the broadcast's measured deficit is not uniform in
   depth -- e6 at 120 steps has cos(g_i,e) 0.507 at blk 0 rising to 0.912 at
   blk 7 (spearman +1.000, top-bottom +0.405, usable top-k >= 0.9 is 1 of 8),
   and +0.132 top-bottom at 50 steps. That deficit *is* by construction the
   failure to account for the REMAINING depth, which is exactly what the bridge
   corrects. So ``G_7 ~ I`` (nothing left to traverse after the last block) and
   ``G_l`` must deviate most from ``I`` at SHALLOW ``l``. Therefore
   **(bridge - bcast) improvement must DECREASE with block index and vanish at
   the last block**. Reading of the outcome, fixed in advance:

   * largest at shallow blocks, vanishing at the last -> mechanism CONFIRMED;
   * roughly FLAT across depth -> "better number, unconfirmed mechanism" (the
     bridge is fixing something other than remaining depth), NOT a design win;
   * largest at DEEP blocks -> defect, not a win: the fit is degenerate or the
     intercept dominates.

   Two settled facts, recorded so no effort is spent rediscovering them: mode 1
   (reference collapse) does NOT apply here -- ||g_i||/||e|| spread is 1.00-1.79x
   in every e6 regime, one global scale suffices; and e6's ``rel_resid`` (~78 at
   20 steps) is a scale-CONVENTION artifact (scale fixed at logit_scale/N, not
   optimal; 9.351e-05 / 1.16e-06 ~ 80.6), so it says nothing about producer
   quality. Raw norm ratios below are therefore labelled convention-dependent:
   only the equal-norm controls adjudicate quality.
6. **Regime discipline**: the 20-step regime is a TRANSIENT and the worst of the
   four (e6 cos/ctl 0.995 init -> 0.506 @20 -> 0.755 @50 -> 0.865 @120). The
   ladder is therefore reported at init, 20 and 120 steps, and 20 is annotated as
   the trough; adjudication reads 120.

``||G_l - I||_F / ||I||_F`` uses an exact identity rather than a build: with
``A^T A = I_r``, ``||A W A^T||_F = ||W||_F`` (the trace collapses through
``A^T A``), so the deviation is ``||W||_F / sqrt(d)``. Both forms are computed and
compared, which is a second, independent check of the orthonormality assumption.

Metric
------
``dLoss = CE(nudged) - CE(base)`` on the held-out measured window, NEGATIVE = the
credit direction helps. All allowed blocks are nudged in ONE forward, which is
the operational shape of a single adapter step; the depth ladder restricts which
blocks are allowed. Injection is a forward hook that ADDS ``-eta * a_l`` to the
block output and otherwise leaves the model's math untouched. ``zero`` installs
hooks that add exact 0.0, so its dLoss must be 0 to the bit -- that is the
harness check, not a claim.

Norm honesty
------------
A producer can look good at nudging purely by having a larger step, so
``||a_l|| / ||g_l||`` is reported by depth bucket for every producer, and if
``bridge`` beats ``auto`` the equal-norm controls (``bridge_aff_eqg`` rescaled to
``||g_l||``, ``auto_eqb`` rescaled to ``||bridge_aff_l||``) decide whether the
win is direction or magnitude.

Reuse
-----
Everything structural comes from e6 by import: ``build_cfg``, ``load_windows``,
``train_base``, ``true_targets``, ``head_error``, ``block_streams``,
``block_basis``, ``project``, ``parse_args``, ``peak_rss_bytes``, ``MIB``. No
model construction, hook capture, window loading, basis helper or arg parsing is
duplicated here.

Ridge convention
----------------
``ridge = reg * mean(diag(G)).clamp_min(1e-12)``, copied from
``train/ttt.py:436``. Two deliberate deviations, both because this is a one-shot
closed-form fit on one window rather than an online accumulator, cited here:
no ``prior`` diagonal (``train/ttt.py:247`` seeds ``G = eye * prior``) and no
``lam`` decay (``train/ttt.py:422``). With ``prior = 1000`` against ~1024 rows
the solve would be shrunk ~20x and the bridge would be measured through the
warmup artefact ``train/ttt.py`` documents rather than through its own predictive
power.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, autograd, no_grad

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e6_producer_validate import (  # noqa: E402
    MIB,
    block_basis,
    block_streams,
    build_cfg,
    head_error,
    load_windows,
    parse_args,
    peak_rss_bytes,
    project,
    spearman,
    train_base,
    true_targets,
)

from hagi.model.model import HAGI  # noqa: E402

# ---------------------------------------------------------------------------
# pre-registered constants
# ---------------------------------------------------------------------------

ETAS: tuple[float, ...] = (0.001, 0.01, 0.1)
PRIMARY_ETA = 0.01
REG = 1e-3  # TttRls default reg (train/ttt.py:185)
LADDER_REQ: tuple[int, ...] = (1, 4, 8, 16, 32)
MAIN: tuple[str, ...] = ("auto", "bcast", "deep", "bridge_aff", "bridge_lin", "rand", "zero")
EQE_SRC: tuple[str, ...] = ("auto", "deep", "bridge_aff", "bridge_lin", "rand")
ALL: tuple[str, ...] = (MAIN + tuple(f"{n}_eqe" for n in EQE_SRC)
                        + ("bridge_aff_eqg", "auto_eqb"))
LADDER_REGIMES: tuple[int, ...] = (0, 20, 120)


# ---------------------------------------------------------------------------
# injection forward (e6 hooks CAPTURE; these ADD)
# ---------------------------------------------------------------------------

def nudge_ce(model: HAGI, ids: Tensor, tgt: Tensor, credits: dict[int, Tensor], eta: float) -> float:
    """CE after ONE forward in which every listed block's output gets ``-eta * a_l``.

    The hook only adds a tensor; nothing else in the model's math changes, and an
    empty ``credits`` dict makes this exactly ``model(ids, tgt)``.
    """
    handles = []
    blocks = list(model.blocks)
    try:
        for blk, a in credits.items():
            delta = (-eta * a).unsqueeze(0)

            def hook(_m: object, _i: tuple, o: Tensor, *, d: Tensor = delta) -> Tensor:
                return o + d

            handles.append(blocks[blk].register_forward_hook(hook))
        with no_grad():
            out = model(ids, tgt)
    finally:
        for h in handles:
            h.remove()
    return float(out.ce.detach())


def credit_dict(credits: list[Tensor], k: int, L: int) -> dict[int, Tensor]:
    """Allow only the LAST ``k`` blocks to be nudged."""
    return {l: credits[l] for l in range(L - k, L)}


# ---------------------------------------------------------------------------
# producers
# ---------------------------------------------------------------------------

def deep_credits(model: HAGI, outs: list[Tensor], tgt: Tensor) -> list[Tensor]:
    """``a_l = grad_{h_l} CE(head(out_norm(h_l)), y)`` -- the identity bridge."""
    d = outs[0].shape[-1]
    flat_t = tgt.reshape(-1)
    res: list[Tensor] = []
    for h in outs:
        hh = h.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            z = model._apply_mixers(model.out_norm(hh))
            ce, _z = model.head.loss(z.reshape(-1, d), flat_t)
        (g,) = autograd.grad(ce, hh)
        res.append(g.detach().float().reshape(-1, d))
    return res


def _ridge_solve(x: Tensor, r: Tensor, reg: float) -> tuple[Tensor, float, Tensor]:
    """``(X^T X + ridge*I) W = X^T R`` with ttt.py's relative ridge.

    Returns ``(W, ridge, G)``.
    """
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    G = x.T @ x
    ridge = reg * float(G.diagonal().mean().clamp_min(1e-12))
    W = torch.linalg.solve(G + ridge * eye, x.T @ r)
    return W, ridge, G


def bridge_fit(basis: list[Tensor], outs: list[Tensor], hL: Tensor, preL: Tensor,
               e: Tensor, reg: float) -> tuple[dict[str, list[Tensor]], list[dict]]:
    """Closed-form ridge State Bridge per block; ``(credits, diagnostics)``.

    ``bridge_lin`` keeps only the fitted linear map, ``bridge_aff`` adds the
    lifted intercept. Diagnostics carry the Gram deviation, the identity check,
    and the prediction relative errors against both ``h_L`` targets.
    """
    credits: dict[str, list[Tensor]] = {"bridge_lin": [], "bridge_aff": []}
    diag: list[dict] = []
    ef = e.float()
    hLr = hL.reshape(-1, hL.shape[-1]).float()
    preLr = preL.reshape(-1, preL.shape[-1]).float()

    for l, (A, h) in enumerate(zip(basis, outs, strict=True)):
        d = h.shape[-1]
        r = A.shape[1]
        hf = h.reshape(-1, d).float()
        gram_dev = float(((A.T @ A) - torch.eye(r, dtype=A.dtype)).abs().max())
        Xe = project(hf, A)                     # [N, r]
        Ze = project(hLr, A)                    # post-out_norm target (producer)
        Zp = project(preLr, A)                  # pre-out_norm target (diagnostic)

        # --- linear fit, both targets -------------------------------------
        W, ridge, G = _ridge_solve(Xe, Ze - Xe, reg)
        rel_x = float((Xe - Ze).norm() / Ze.norm().clamp_min(1e-30))
        rel_lin = float(((Xe + Xe @ W) - Ze).norm() / Ze.norm().clamp_min(1e-30))
        Wp, _, _ = _ridge_solve(Xe, Zp - Xe, reg)
        rel_x_pre = float((Xe - Zp).norm() / Zp.norm().clamp_min(1e-30))
        rel_lin_pre = float(((Xe + Xe @ Wp) - Zp).norm() / Zp.norm().clamp_min(1e-30))

        # --- affine fit: centred features, shift kept separately ----------
        xm, zm = Xe.mean(0), Ze.mean(0)
        Xc, Zc = Xe - xm, Ze - zm
        Wa, ridge_a, Ga = _ridge_solve(Xc, Zc, reg)
        shift = zm - xm @ Wa
        rel_aff = float(((Xe @ Wa + shift) - Ze).norm() / Ze.norm().clamp_min(1e-30))

        # --- credits ------------------------------------------------------
        pe = project(ef, A)                     # [N, r] features of the head error
        lin = (pe @ W) @ A.T
        aff = (pe @ Wa + shift) @ A.T           # affine: Wa's linear part + intercept
        credits["bridge_lin"].append(ef + lin)
        credits["bridge_aff"].append(ef + aff)

        # identity check: a @ A == (e @ A) (I + W), exact iff A^T A == I_r
        lhs = credits["bridge_lin"][-1] @ A
        rhs = pe @ (torch.eye(r, dtype=pe.dtype) + W)
        lhs_a = credits["bridge_aff"][-1] @ A
        rhs_a = pe @ (torch.eye(r, dtype=pe.dtype) + Wa) + shift
        # ||G_l - I||_F / ||I||_F with G_l = I + A W A^T. The d-space operator's
        # Frobenius norm is built two ways: directly, and via ||W||_F/sqrt(d)
        # (exact iff A^T A == I_r). Agreement is a second orthonormality check.
        sqrt_d = float(d) ** 0.5
        dev = {
            "dev_lin_direct": float(torch.linalg.matrix_norm(A @ W @ A.T, "fro")) / sqrt_d,
            "dev_lin_w": float(W.norm()) / sqrt_d,
            "dev_aff_direct": float(torch.linalg.matrix_norm(A @ Wa @ A.T, "fro")) / sqrt_d,
            "dev_aff_w": float(Wa.norm()) / sqrt_d,
            "dev_bcast": 0.0,
        }
        diag.append({
            "gram_dev": gram_dev, "r": r, "n": Xe.shape[0],
            "cond": float(G.diagonal().max() / G.diagonal().mean()),
            "ridge": ridge, "ridge_aff": ridge_a,
            "rel_x": rel_x, "rel_lin": rel_lin, "rel_aff": rel_aff,
            "rel_x_pre": rel_x_pre, "rel_lin_pre": rel_lin_pre,
            "identity": float((lhs - rhs).norm() / rhs.norm().clamp_min(1e-30)),
            "identity_aff": float((lhs_a - rhs_a).norm() / rhs_a.norm().clamp_min(1e-30)),
            "lift_rel": float(lin.norm() / ef.norm().clamp_min(1e-30)),
            "bias_rel": float((shift @ A.T).norm() / ef.norm().clamp_min(1e-30)),
            **dev,
        })
    return credits, diag


def rand_credits(g: list[Tensor], seed: int) -> list[Tensor]:
    """Same-norm random direction per block, seeded -- the control that kills."""
    gg = torch.Generator().manual_seed(seed + 7)
    out = []
    for gi in g:
        v = torch.randn(gi.shape, generator=gg, dtype=torch.float32)
        out.append(v * (gi.norm() / v.norm().clamp_min(1e-30)))
    return out


def scale_to(src: list[Tensor], ref: list[Tensor]) -> list[Tensor]:
    """Rescale each block's credit so its Frobenius norm equals ``ref``'s."""
    return [a * (b.norm() / a.norm().clamp_min(1e-30)) for a, b in zip(src, ref, strict=True)]


def scale_all(src: list[Tensor], norm: Tensor) -> list[Tensor]:
    """Rescale every block's credit to the SAME Frobenius norm ``norm``.

    This is the lane the ranking is read on: at natural norms the comparison
    measures the logit_scale/N convention (step size), not the direction, and the
    autograd credit's effect is below the fp32 ulp of the loss, i.e. unmeasurable
    rather than merely small.
    """
    return [a * (norm / a.norm().clamp_min(1e-30)) for a in src]


def assemble(model: HAGI, ids: Tensor, tgt: Tensor, seed: int) -> tuple[dict[str, list[Tensor]], dict]:
    """All producers at the model's current weights, plus shared context."""
    g, ce_true = true_targets(model, ids, tgt)
    e, rel, _ce_head, scale = head_error(model, ids, tgt)
    if rel > 1e-4:
        raise SystemExit("closed-form head error does not match autograd at the head input")
    basis = block_basis(model)
    _ins, outs, _gap = block_streams(model, ids)
    with no_grad():
        hL = model(ids).hidden.detach()          # post-out_norm, pre-head (choice 3)
    preL = outs[-1]                              # last block output, no extra forward

    credits: dict[str, list[Tensor]] = {}
    br, diag = bridge_fit(basis, outs, hL, preL, e, REG)
    credits["auto"] = g
    credits["bcast"] = [e for _ in g]
    credits["deep"] = deep_credits(model, outs, tgt)
    credits.update(br)
    credits["rand"] = rand_credits(g, seed)
    credits["zero"] = [torch.zeros_like(x) for x in g]
    credits["bridge_aff_eqg"] = scale_to(credits["bridge_aff"], g)
    credits["auto_eqb"] = scale_to(g, credits["bridge_aff"])
    # equal-norm lane: every producer rescaled to ||e||, so a dLoss difference is a
    # direction difference and nothing else. bcast is already at ||e|| by identity.
    en = e.norm()
    for n in EQE_SRC:
        credits[f"{n}_eqe"] = scale_all(credits[n], en)
    return credits, {"g": g, "e": e, "ce": ce_true, "scale": scale, "diag": diag}


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------

def buckets(L: int) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"bottom": [], "middle": [], "top": []}
    for l in range(L):
        out["bottom" if l < L / 3 else ("middle" if l < 2 * L / 3 else "top")].append(l)
    return {k: v for k, v in out.items() if v}


def by_bucket(vals: np.ndarray, idx: dict[str, list[int]]) -> dict[str, float]:
    return {k: float(np.mean(vals[v])) for k, v in idx.items()}


def ladder_set(L: int) -> list[int]:
    return sorted({k for k in LADDER_REQ if k < L} | {L})


# ---------------------------------------------------------------------------
# one regime
# ---------------------------------------------------------------------------

def run_regime(model: HAGI, ids: Tensor, tgt: Tensor, label: str, seed: int,
               ladder: bool) -> dict:
    L = len(model.blocks)
    t0 = time.perf_counter()
    credits, ctx = assemble(model, ids, tgt, seed)
    t_build = time.perf_counter() - t0
    g = ctx["g"]
    with no_grad():
        base = float(model(ids, tgt).ce.detach())

    print(f"--- {label}")
    print(f"    base CE = {base:.6f}   autograd-pass CE = {ctx['ce']:.6f}   "
          f"head-error scale (logit_scale/N) = {ctx['scale']:.3e}   "
          f"rows N = {g[0].shape[0]}, d = {g[0].shape[1]}")

    idx = buckets(L)
    print(f"    L = {L}; depth buckets "
          + ", ".join(f"{k}={v}" for k, v in idx.items())
          + f"; ladder k in {ladder_set(L)} (requested {list(LADDER_REQ)} + L, clamped)")

    # ---- bridge diagnostics -------------------------------------------------
    dg = ctx["diag"]
    arr = lambda k: np.array([x[k] for x in dg])  # noqa: E731
    gd, idv = arr("gram_dev"), arr("identity")
    rx, rl, ra = arr("rel_x"), arr("rel_lin"), arr("rel_aff")
    rxp, rlp = arr("rel_x_pre"), arr("rel_lin_pre")
    print("    BRIDGE DIAGNOSTICS -- can the bridge predict the remaining depth at all?")
    print(f"      max ||A^T A - I_r||_max = {gd.max():.3e}  ->  A A^T = I_r "
          f"{'HOLDS' if gd.max() < 1e-4 else 'FAILS'} (fp32 round-off is ~1e-7)")
    print(f"      max identity rel.err (a@A vs (e@A)(I+W)) = {idv.max():.3e}  ->  brief algebra "
          f"{'CONFIRMED' if idv.max() < 1e-4 else 'BROKEN'}"
          f"   (affine form {arr('identity_aff').max():.3e})")
    dl, dw = arr("dev_lin_direct"), arr("dev_lin_w")
    da, daw = arr("dev_aff_direct"), arr("dev_aff_w")
    print(f"      ||A W A^T||_F/sqrt(d) vs ||W||_F/sqrt(d) (the identity that ||A^T A=I|| buys): "
          f"max rel diff lin {np.abs(dl / dw - 1).max():.2e}, aff {np.abs(da / daw - 1).max():.2e}")
    print(f"      ridge = {REG:g} * mean(diag(G)); blk0 mean-diag {dg[0]['ridge'] / REG:.3e} "
          f"-> ridge {dg[0]['ridge']:.3e}; affine ridge {dg[0]['ridge_aff']:.3e}")
    hdr = (f"      {'bucket':>8} {'relerr X-only':>14} {'relerr lin':>11} {'relerr aff':>11} "
           f"{'aff/X':>8} {'lin/X':>8} {'<1?':>5}")
    print(hdr)
    bx, bl, ba = by_bucket(rx, idx), by_bucket(rl, idx), by_bucket(ra, idx)
    for k in idx:
        print(f"      {k:>8} {bx[k]:>14.4f} {bl[k]:>11.4f} {ba[k]:>11.4f} "
              f"{ba[k] / bx[k]:>8.4f} {bl[k] / bx[k]:>8.4f} {str(ba[k] < bx[k]):>5}")
    print(f"      mean     {rx.mean():>14.4f} {rl.mean():>11.4f} {ra.mean():>11.4f} "
          f"{ra.mean() / rx.mean():>8.4f} {rl.mean() / rx.mean():>8.4f}     -")
    print("      (relerr < 1 vs the X-only baseline is the survival condition; >= 1 means the "
          "bridge cannot predict the remaining depth and the idea is dead)")
    print(f"      pre-out_norm target (diagnostic only): relerr X-only {rxp.mean():.4f}, "
          f"lin {rlp.mean():.4f}, lin/X {rlp.mean() / rxp.mean():.4f}")
    for key, nm in (("bias_rel", "lifted intercept ||s@A^T||/||e||"),
                    ("lift_rel", "lifted linear   ||A^T W A e||/||e||")):
        b = by_bucket(arr(key), idx)
        print(f"      {nm} by bucket: " + ", ".join(f"{k}={b[k]:.3e}" for k in idx))

    # ---- norm ratios --------------------------------------------------------
    gn = np.array([x.norm() for x in g])
    print("    CREDIT NORM  ||a_l|| / ||g_l||  (pooled over rows; CONVENTION-DEPENDENT: the "
          "credit scale is fixed at logit_scale/N, not optimal, so a raw ratio measures the "
          "convention, not the producer -- only the equal-norm rows below adjudicate quality)")
    print(f"      {'producer':>16}" + "".join(f" {k:>10}" for k in idx) + f" {'mean':>10}")
    for name in ALL:
        v = np.array([a.norm() / max(gn[l], 1e-30) for l, a in enumerate(credits[name])])
        b = by_bucket(v, idx)
        print(f"      {name:>16}" + "".join(f" {b[k]:>10.3e}" for k in idx) + f" {v.mean():>10.3e}")

    # ---- headline: all L blocks, every eta ---------------------------------
    # The loss is an fp32 scalar, so the measurement floor is the fp32 ulp of CE,
    # not the double ulp of the Python float it is cast to. Every dLoss below is a
    # whole multiple of this; a producer whose effect is under it reads as 0.0.
    floor = float(np.spacing(np.float32(base)))
    print(f"    NUDGING dLoss = CE(nudged) - CE(base), all {L} blocks in ONE step "
          f"(NEGATIVE helps). Ranking read at eta = {PRIMARY_ETA:g}")
    print(f"    fp32 loss ulp = {floor:.4e} (CE = {base:.6f} -> exponent {int(np.floor(np.log2(base)))}): "
          f"this is the measurement floor. An effect under it is NOT a small effect, it is "
          f"unrepresentable in the loss and reads as exactly 0.0. \"meas.\" = |dLoss| > ulp below.")
    hdr = f"      {'producer':>16}" + "".join(f" {'eta=' + format(x, 'g'):>13}" for x in ETAS)
    print(hdr)
    print("      " + "-" * (len(hdr) - 6))
    head: dict[str, dict[float, float]] = {}
    t1 = time.perf_counter()
    for name in ALL:
        cd = credit_dict(credits[name], L, L)
        row = {eta: nudge_ce(model, ids, tgt, cd, eta) - base for eta in ETAS}
        head[name] = row
        print(f"      {name:>16}" + "".join(f" {row[x]:>+13.4e}" for x in ETAS))
    print("      " + "-" * (len(hdr) - 6))
    print("      same table in loss ulps (resolution view; <1 ulp = the producer is invisible "
          "to this metric at this eta)")
    print(hdr)
    print("      " + "-" * (len(hdr) - 6))
    for name in ALL:
        row = head[name]
        print(f"      {name:>16}" + "".join(f" {row[x] / floor:>+13.1f}" for x in ETAS))
    print("      " + "-" * (len(hdr) - 6))
    print(f"      zero-row check: {head['zero'][PRIMARY_ETA]:+.3e} "
          f"({'harness exact' if head['zero'][PRIMARY_ETA] == 0.0 else 'NON-ZERO, harness suspect'})")

    out = {"label": label, "base": base, "head": head, "L": L, "idx": idx,
           "diag": dg, "gnorm": gn, "build_s": t_build,
           "dev": arr("dev_aff_direct"), "dev_lin": arr("dev_lin_direct"),
           "norms": {n: np.array([a.norm() for a in credits[n]]) for n in ALL}}

    # ---- SHAPE TEST: per-block improvement of bridge over bcast -------------
    # Pre-registered: improvement must DECREASE with block index and vanish at
    # the last block, because the deficit it repairs is the remaining depth.
    # Two eta lanes: the loss is fp32, so a 0.01 step's per-block effect can land
    # within a few ulps of the loss and the curve would be reading quantisation
    # noise; 0.1 buys 10x resolution on the same directions.
    print("    SHAPE TEST (pre-registered): per-block single injection. improve = dLoss(bcast "
          "only) - dLoss(bridge only); POSITIVE = bridge better")
    print("      bcast and bridge_aff are within a few % of the same norm by construction (both are "
          "e plus a lifted correction), so that pair is already near norm-matched; the _eqe column "
          "is bridge rescaled to exactly ||e||, i.e. direction with the norm argument removed")
    lanes: dict[float, dict[str, np.ndarray]] = {}
    for eta, cols in ((PRIMARY_ETA, ("auto_eqe", "bcast", "bridge_aff", "bridge_aff_eqe")),
                      (0.1, ("bcast", "bridge_aff", "bridge_aff_eqe"))):
        note = "primary" if eta == PRIMARY_ETA else "10x-resolution lane (same directions, bigger step)"
        print(f"      -- eta = {eta:g} [{note}]; all-L bcast effect there = "
              f"{abs(head['bcast'][eta]) / floor:.0f} loss ulps")
        hdr = (f"      {'blk':>4}" + "".join(f" {c:>13}" for c in cols)
               + f" {'impr(raw)':>12} {'ulps':>7} {'impr(eqN)':>12} {'ulps':>7} {'||G-I||/||I||':>14}")
        print(hdr)
        print("      " + "-" * (len(hdr) - 6))
        vals = {c: np.zeros(L) for c in cols}
        for l in range(L):
            for c in cols:
                vals[c][l] = nudge_ce(model, ids, tgt, {l: credits[c][l]}, eta) - base
        imp_raw = vals["bcast"] - vals["bridge_aff"]
        imp_eq = vals["bcast"] - vals["bridge_aff_eqe"]
        for l in range(L):
            print(f"      {l:>4}" + "".join(f" {vals[c][l]:>+13.4e}" for c in cols)
                  + f" {imp_raw[l]:>+12.4e} {imp_raw[l] / floor:>7.1f}"
                  + f" {imp_eq[l]:>+12.4e} {imp_eq[l] / floor:>7.1f} {out['dev'][l]:>14.4e}")
        print("      " + "-" * (len(hdr) - 6))
        lanes[eta] = {"raw": imp_raw, "eq": imp_eq}
        print(f"      spearman(impr raw, depth) = {spearman(imp_raw):+.3f} | "
              f"spearman(impr equal-norm, depth) = {spearman(imp_eq):+.3f}   "
              f"prediction: NEGATIVE for both (largest at shallow blocks)")
    sp_dev = spearman(out["dev"])
    print(f"      spearman(||G-I||/||I||, depth) = {sp_dev:+.3f}   prediction: NEGATIVE "
          f"(G at the last block ~ I, since no depth remains to bridge)")

    # Adjudicate on the lane with resolution; state which one that was.
    imp_eq_hi = lanes[0.1]["eq"]
    imp_eq_lo = lanes[PRIMARY_ETA]["eq"]
    use_hi = bool(np.abs(imp_eq_hi).max() >= np.abs(imp_eq_lo).max())
    best = imp_eq_hi if use_hi else imp_eq_lo
    best_eta = 0.1 if use_hi else PRIMARY_ETA
    admissible = bool(np.abs(best).max() > 8.0 * floor)
    shallow = best[: max(1, L // 2)].mean()
    sp_best = spearman(best)
    print(f"      adjudication lane: eta = {best_eta:g} (max |impr| = {np.abs(best).max():.4e} = "
          f"{np.abs(best).max() / floor:.1f} ulps; 8*ulp gate = {8 * floor:.4e}) -> "
          f"{'ADJUDICABLE' if admissible else 'NOT ADJUDICABLE: the bridge correction is under the loss resolution here'}")
    print(f"      last-block improvement = {best[-1]:+.4e} vs shallow-half mean = {shallow:+.4e} "
          f"-> vanishing at the top: {abs(best[-1]) < 0.25 * abs(shallow)}")
    flat = bool(abs(sp_best) < 0.3 and best.std() < 0.5 * abs(best).mean())
    if not admissible:
        shape = ("NOT ADJUDICABLE at this regime -- the bridge correction is under the fp32 loss "
                 "ulp, so its depth shape cannot be told from quantisation")
    elif sp_best < -0.5 and abs(best[-1]) < 0.25 * abs(shallow):
        shape = ("MECHANISM CONFIRMED -- improvement is largest at shallow blocks and vanishes at "
                 "the last, which is the shape a remaining-depth correction must have")
    elif sp_best > 0.5:
        shape = ("DEFECT, not a win -- improvement is largest at DEEP blocks: degenerate fit or "
                 "intercept dominance")
    elif flat:
        shape = ("better number, unconfirmed mechanism -- improvement is roughly FLAT across depth, "
                 "so the bridge is fixing something other than remaining depth")
    else:
        shape = "INCONCLUSIVE shape -- neither monotone decreasing nor cleanly flat; see the curves above"
    print(f"      SHAPE VERDICT (pre-registered reading): {shape}")
    out["imp_raw"] = lanes[PRIMARY_ETA]["raw"]
    out["imp_eq"] = lanes[PRIMARY_ETA]["eq"]
    out["imp_eq_hi"] = imp_eq_hi
    out["sp_imp"] = spearman(lanes[PRIMARY_ETA]["raw"])
    out["sp_imp_eqe"] = spearman(imp_eq_lo)
    out["sp_best"] = sp_best
    out["best_eta"] = best_eta
    out["admissible"] = admissible
    out["sp_dev"] = sp_dev
    out["shape"] = shape
    out["floor"] = floor

    # ---- depth ladder -------------------------------------------------------
    if ladder:
        ks = ladder_set(L)
        print("    DEPTH-UTILITY LADDER (inject only the last k blocks; dLoss, NEGATIVE helps)")
        print("      natural norms = what the server would actually do; _eqe lanes are all "
              "rescaled to ||e|| so the comparison is direction-only")
        for eta in ETAS:
            tag = "PRIMARY" if eta == PRIMARY_ETA else "secondary"
            names = list(MAIN) + ([f"{n}_eqe" for n in EQE_SRC] if eta == PRIMARY_ETA else [])
            print(f"      eta = {eta:g} [{tag}]"
                  + ("" if eta == PRIMARY_ETA else
                     " (natural norms only; the eqe lane is reported at the primary eta - "
                     "cut for runtime, stated)"))
            hdr = f"      {'producer':>16}" + "".join(f" {'k=' + str(k):>12}" for k in ks)
            print(hdr)
            print("      " + "-" * (len(hdr) - 6))
            for name in names:
                row = [nudge_ce(model, ids, tgt, credit_dict(credits[name], k, L), eta) - base
                       for k in ks]
                out.setdefault("ladder", {})[(name, eta)] = row
                print(f"      {name:>16}" + "".join(f" {v:>+12.4e}" for v in row))
            print("      " + "-" * (len(hdr) - 6))
    print(f"    [producers {t_build:.1f}s, nudging {time.perf_counter() - t1:.1f}s]\n")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()          # e6's parser, so defaults match e6 exactly
    cfg = build_cfg(args)
    measured, training, prov = load_windows(args, cfg.model.vocab_size)
    ids, tgt = measured[0]
    torch.manual_seed(args.seed)
    model = HAGI(cfg).eval()
    L = cfg.model.num_layers

    print(f"e7 producer nudging | {L} blocks, H={cfg.model.hidden_size}, "
          f"V={cfg.model.vocab_size}, rank={args.rank}, T={ids.shape[1]}, "
          f"fp32 cpu, loop_depth=1, window=0 (all relays)")
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"head: W_out {tuple(model.head.weight.shape)}, tied={model.head.is_tied}, "
          f"logit_scale={float(model.head.logit_scale.detach()):.4f}")
    print(f"tokens: {prov}")
    print("\nPRE-REGISTERED (fixed before any number in this session):")
    print(f"  primary eta = {PRIMARY_ETA:g}   sweep = {list(ETAS)}   ridge reg = {REG:g}")
    print("  affine form = (b) centre-and-keep-shift; (a) augmented constant column REJECTED "
          "(a ones-row has norm sqrt(d) and is not orthogonal to span(A), which breaks the "
          "A A^T = I_r reduction the r x r solve rests on)")
    print("  h_L = post-out_norm final hidden (the head input); deep = detached-tail autograd "
          "per block (not a retained stack graph)")
    print("  falsifiable prediction: deep LOSES to bridge_aff (tuned-lens precedent: "
          "intermediate-layer logits are worse predictors than the final layer's)")
    print("  SHAPE prediction: (bridge - bcast) improvement DECREASES with block index and vanishes "
          "at the last block; ||G_l - I||/||I|| largest at shallow l. Flat => unconfirmed "
          "mechanism; reversed => defect (degenerate fit / intercept dominates)")
    print(f"  ladder reported at train-steps {list(LADDER_REGIMES)}; 20 steps is e6's TROUGH "
          f"(cos/ctl 0.995 init -> 0.506 @20 -> 0.755 @50 -> 0.865 @120), adjudication reads 120")
    print("  equal-norm lane: every producer's credit rescaled per block to ||e||, because at "
          "natural norms the logit_scale/N convention spreads credit norms by ~3 orders of "
          "magnitude and a fixed-eta dLoss then measures step SIZE, not direction")
    print("  ranking read at k = L (all blocks) and primary eta\n")

    results = []
    prev = 0
    for k in list(args.train_steps):
        if k < prev:
            raise SystemExit(f"--train-steps must be cumulative ascending, got {list(args.train_steps)}")
        t0 = time.perf_counter()
        ce = train_base(model, training, k - prev, args.lr) if k > prev else float("nan")
        tt = time.perf_counter() - t0
        label = ("INIT (random weights)" if k == 0
                 else f"trained {k} Adam steps on {len(training)} real windows "
                      f"of {args.train_seq} tokens (train CE {ce:.4f})")
        r = run_regime(model, ids, tgt, label, args.seed, ladder=(k in LADDER_REGIMES))
        r["steps"] = k
        r["train_s"] = tt
        results.append(r)
        prev = k

    # ---- ranking ------------------------------------------------------------
    print("=" * 100)
    for r in results:
        h = r["head"]
        print(f"RANKING @ {r['label']}   dLoss at primary eta {PRIMARY_ETA:g}, k = L = {r['L']}")
        print(f"      fp32 loss ulp = {r['floor']:.3e}; entries below it are unrepresentable, "
              f"not merely small -> the NATURAL-norm ranking is a step-size ranking, and the "
              f"EQUAL-NORM block is the direction ranking")
        for lane, names, rand_name in (
                ("natural norms (operational: what the server step size actually does)",
                 list(MAIN) + ["bridge_aff_eqg", "auto_eqb"], "rand"),
                ("equal norms ||a_l|| = ||e|| (direction only)",
                 [f"{n}_eqe" for n in EQE_SRC] + ["bcast"], "rand_eqe")):
            print(f"      -- {lane}")
            order = sorted(names, key=lambda n: h[n][PRIMARY_ETA])
            lane_rand = h[rand_name][PRIMARY_ETA]
            print(f"      {'rank':>4} {'producer':>16} {'dLoss':>13} {'vs rand(lane)':>14} {'measurable':>11}")
            for i, n in enumerate(order):
                v = h[n][PRIMARY_ETA]
                print(f"      {i:>4} {n:>16} {v:>+13.4e} {v - lane_rand:>+14.4e} "
                      f"{str(abs(v) > r['floor']):>11}")
        print()

    print("=" * 100)
    print("VERDICT INPUTS (all measured this session, primary eta, k = L)")
    for r in results:
        h = r["head"]
        fl = r["floor"]
        a, b = h["auto"][PRIMARY_ETA], h["bridge_aff"][PRIMARY_ETA]
        d, rn = h["deep"][PRIMARY_ETA], h["rand"][PRIMARY_ETA]
        bc, z = h["bcast"][PRIMARY_ETA], h["zero"][PRIMARY_ETA]
        print(f"  @ {r['label']}   (loss ulp {fl:.3e}; 'meas.' = |dLoss| above the ulp)")
        print(f"      NATURAL: auto {a:+.4e} [{abs(a) > fl}] | bridge_aff {b:+.4e} [{abs(b) > fl}] | "
              f"bridge_lin {h['bridge_lin'][PRIMARY_ETA]:+.4e} | deep {d:+.4e} [{abs(d) > fl}] | "
              f"bcast {bc:+.4e} | rand {rn:+.4e} [{abs(rn) > fl}] | zero {z:+.2e}")
        eq = {n: h[f"{n}_eqe"][PRIMARY_ETA] for n in EQE_SRC}
        print(f"      EQUAL-NORM(||e||): auto {eq['auto']:+.4e} | bridge_aff {eq['bridge_aff']:+.4e} | "
              f"bridge_lin {eq['bridge_lin']:+.4e} | deep {eq['deep']:+.4e} | rand {eq['rand']:+.4e} | "
              f"bcast {bc:+.4e} [= eqe by identity]")
        best_eq = min(list(eq.values()) + [bc])
        print(f"      equal-norm adjudication: bridge_aff beats rand: "
              f"{eq['bridge_aff'] < eq['rand']} | bridge_aff beats auto: "
              f"{eq['bridge_aff'] < eq['auto']} | bridge_aff is best producer: "
              f"{eq['bridge_aff'] <= best_eq + 1e-12} | gap to auto {eq['bridge_aff'] - eq['auto']:+.4e}")
        print(f"      deep vs bridge (falsifiable pred: deep LOSES): deep {eq['deep']:+.4e} vs "
              f"bridge {eq['bridge_aff']:+.4e} -> prediction survived: {eq['bridge_aff'] < eq['deep']}")
        print(f"      brief's literal control: bridge_aff_eqg {h['bridge_aff_eqg'][PRIMARY_ETA]:+.4e} "
              f"vs auto_eqb {h['auto_eqb'][PRIMARY_ETA]:+.4e} (both at the OTHER's norm)")
        print(f"      shape: spearman(impr, depth) {r['sp_imp']:+.3f} raw / {r['sp_imp_eqe']:+.3f} "
              f"equal-norm at eta {PRIMARY_ETA:g}; {spearman(r['imp_eq_hi']):+.3f} equal-norm at "
              f"eta 0.1; spearman(||G-I||/||I||, depth) {r['sp_dev']:+.3f}; adjudication lane eta "
              f"{r['best_eta']:g} "
              f"{'admissible' if r['admissible'] else 'NOT admissible (under the loss ulp)'}")
        print(f"      -> {r['shape']}")

    # ---- FINAL: ranking + one-line verdict ----------------------------------
    # Regime discipline (pre-registered): 20 steps is e6's trough, so the
    # adjudication reads the deepest trained regime available (120 steps).
    adj = max(results, key=lambda x: x["steps"])
    h = adj["head"]
    fl = adj["floor"]
    print("=" * 100)
    print(f"FINAL RANKING @ {adj['label']}   eta = {PRIMARY_ETA:g}, k = L = {adj['L']}, "
          f"EQUAL CREDIT NORM ||a_l|| = ||e|| (the only lane where the step size is not the "
          f"variable; loss ulp {fl:.4e})")
    eq_names = [f"{n}_eqe" for n in EQE_SRC] + ["bcast"]
    order = sorted(eq_names, key=lambda n: h[n][PRIMARY_ETA])
    print(f"      {'rank':>4} {'producer':>16} {'dLoss':>13} {'loss ulps':>11} {'backprop-free':>14}")
    for i, n in enumerate(order):
        v = h[n][PRIMARY_ETA]
        bp = "no" if n == "auto_eqe" else "yes"
        print(f"      {i:>4} {n:>16} {v:>+13.4e} {v / fl:>11.1f} {bp:>14}")
    auto_v = h["auto_eqe"][PRIMARY_ETA]
    free = {n: h[n][PRIMARY_ETA] for n in order if n != "auto_eqe"}
    best_name = min(free, key=lambda n: free[n])
    best_v = free[best_name]
    rand_v = h["rand_eqe"][PRIMARY_ETA]
    premise = auto_v <= min(free.values())
    if not premise:
        print("\n      !!! PREMISE WRONG: autograd (auto_eqe) is NOT the best producer on nudging at "
              "equal credit norm. The upper bound is not above the backprop-free candidates, so the "
              "experiment's reference point is invalid as a bound.")
    gap = (best_v - auto_v) / abs(auto_v) if auto_v else float("nan")
    print(f"      best backprop-free = {best_name} {best_v:+.4e}; autograd = {auto_v:+.4e}; "
          f"rand = {rand_v:+.4e}; bridge_aff = {h['bridge_aff_eqe'][PRIMARY_ETA]:+.4e}")
    print(f"      best backprop-free beats rand: {best_v < rand_v} | approaches autograd "
          f"(within 10% of |auto|): {best_v <= 0.9 * auto_v} | shortfall vs autograd = {gap:+.1%}")
    if best_v < rand_v and best_v <= 0.9 * auto_v:
        verdict = (f"YES -- a backprop-free producer ({best_name}) matches autograd functionally at "
                   f"equal credit norm: {best_v:+.4e} vs auto {auto_v:+.4e}, both far below rand "
                   f"{rand_v:+.4e}")
    elif best_v < rand_v:
        verdict = (f"PARTIAL -- {best_name} beats the random control ({best_v:+.4e} vs "
                   f"{rand_v:+.4e}) so it carries real credit signal, but does not approach autograd "
                   f"({auto_v:+.4e}; shortfall {gap:+.1%})")
    else:
        verdict = (f"NO -- no backprop-free producer beats the random control at equal credit norm "
                   f"(best {best_name} {best_v:+.4e} vs rand {rand_v:+.4e}); dead")
    print(f"\nVERDICT: {verdict}")
    print(f"         mechanism: {adj['shape']}")
    print(f"\npeak RSS: {peak_rss_bytes() / MIB:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
