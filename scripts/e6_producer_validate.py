"""E6: is the closed-form head error an acceptable substitute for autograd targets?

Why
---
The Python self-evolution contour (:mod:`hagi.train.ttt`) gets its per-block
learning target from a genuine backward pass: ``_harvest`` returns
``dCE/d(block-i output)`` and ``_rows_for`` rescales it to
``-stream_frac * (rms_stream / rms_grad) * grad / lora.scaling``. The proposed
server-side replacement has no backward pass and instead broadcasts one
gradient-free quantity to every block::

    e = (softmax(logits) - onehot(target)) @ W_out        # dCE/d(head input)
    y_i := e   for every block i                          # "along the skip path"

Nobody has measured that ``e ~= dCE/d(h_i)``. This script measures it. The
verdict gates whether the contour is worth migrating at all, so the reference is
the real thing: ``g_i`` is produced by ``TttRls._harvest`` itself -- the shipped
contour's own ``autograd.grad`` call -- not a reimplementation. Reimplementing
the backward here would make the comparison circular.

What separates the two
----------------------
``e`` is identical for every block by construction, so the only question is how
much of the head error survives the trip down the residual stream. Two effects
do the separating, and both are the subject of the measurement, not bugs:

* ``out_norm`` sits between the last block output and the head, so ``e`` is the
  head error of the *normalized* stream; the RMSNorm Jacobian removes the
  component along the stream.
* ``dCE/d(h_i)`` also collects the paths from ``h_i`` into every *later* block,
  which the additive-skip argument ignores. How big that term is depends on how
  much each block actually contributes -- which is why this script measures a
  regime sweep (init, then progressively trained) instead of one snapshot.

Two positive scalars, recorded because they move every norm ratio and no cosine:
``logit_scale`` (``LMHead.loss`` differentiates ``hidden * logit_scale``) and
``1/N`` (``_ChunkedCrossEntropy`` reduces CE as a *mean* over scored rows, so its
backward multiplies by ``grad_loss / n``). The brief's ``e`` is therefore ``N``
times the autograd scale -- the same ~1/N effect ``train/ttt.py`` documents as
the reason ``stream_frac`` exists at all. ``--check`` asserts the identity
``dCE/d(head input) == logit_scale * e / N`` to fp32 round-off before trusting
any depth number.

Controls
--------
* ``cos(g_i, g_j)``, ``i != j`` -- how correlated the *true* per-block gradients
  are with each other. A headline of ~0.2 against a control of ~0.2 is the
  intrinsic noise floor of comparing any two blocks, not a failure of ``e``.
* ``cos(g_i, g_{L-1})`` -- the ceiling for any top-down broadcast that never
  touches the head.
* ``cos_r`` -- cosine after projecting both sides onto the rank-``r`` column
  space of the frozen ``lora_A``. The contour can only ever fit inside that
  subspace (``delta = scaling * (feat @ A) @ B.T``, columns of ``A``), so this is
  the column that decides whether the solved ``B`` differs between the two
  producers. Full-space agreement can hide a subspace disagreement and vice
  versa.
* ``rel_resid`` -- ``||g_i - logit_scale*e/N|| / ||g_i||``, the fraction of the
  true target the broadcast misses in norm. Near 1 the cosines are uninformative
  because everything is correlated with everything; this is the sharp version.

Deviations, recorded
--------------------
* **Tokens.** Real corpus text: a window of ``data/edu.compact.bin`` (uint32,
  compact vocab), read as ``PackedStream`` reads it (``seq+1`` tokens,
  ``input_ids = window[:-1]``, ``targets = window[1:]``, so every scored position
  carries a real next token and nothing wraps), then densely re-indexed into the
  tiny id space so the repeat structure is the corpus's. ``--tokens random``
  re-runs the same geometry on iid ids: that arm can support a GEOMETRY claim
  only, never a quality one.
* **No ``doc_ids``.** ``TttRls._harvest`` calls ``model(input_ids, targets,
  loss_mask=...)`` and never passes ``doc_ids``, so a window attends across
  document boundaries. Mirrored here -- the measurement must describe the contour
  that exists, not an idealised one.
* **z-loss is not neutralised.** ``_harvest`` differentiates ``output.ce``, and
  the closed form is the CE head error, so neither ``model.head.z_loss_weight``
  nor ``train.z_loss_weight`` enters either side. Nothing had to be zeroed.
* **fp32, CPU, ``loop_depth=1``, ``sliding.window=0``** (every layer is a
  full-attention relay) -- the repo's own tiny-config geometry, so that the
  sweep stays in seconds. The real stack is bf16 with a 1:3 relay pattern; a
  windowed layer sees fewer downstream readers, which can only *help* the skip
  approximation, so the all-relay config is the conservative choice.
* **The trained arms are tiny-model trained**, not the 27B: Adam overfitting one
  real window, which inflates block contributions and is the stress the skip
  path cannot survive if it fails anywhere. It is a regime probe, not a
  measurement of the shipped model.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, autograd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.config import Config, validate_config  # noqa: E402
from hagi.data.dataset import dataset_path  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.ttt import TttRls  # noqa: E402

MIB = 2**20


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------

def peak_rss_bytes() -> int:
    """Peak resident set of this process, from the OS (not estimated).

    ``ru_maxrss`` is kilobytes on Linux and bytes on Windows, and Windows has no
    ``resource`` module at all, so the NT path reads ``PeakWorkingSetSize``
    through psapi. The value is a snapshot of the whole process lifetime, so one
    call at the end covers every regime.
    """
    if os.name == "nt":
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = PMC()
        counters.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


# --------------------------------------------------------------------------
# model + tokens
# --------------------------------------------------------------------------

def build_cfg(args: argparse.Namespace) -> Config:
    """Tiny HAGI config with the ttt_lora contour attached, through the real validator."""
    cfg = Config()
    m = cfg.model
    m.hidden_size = args.hidden
    m.num_layers = args.layers
    m.vocab_size = args.vocab
    m.attention.num_query_heads = args.hidden // args.head_dim
    m.attention.num_kv_heads = max(1, args.hidden // args.head_dim // 4)
    m.attention.head_dim = args.head_dim
    m.attention.max_seq_len = args.seq + 8
    m.ffn.intermediate_size = 4 * args.hidden
    m.sliding.window = 0  # every layer is a full-attention relay
    m.head.unigram_prior = False
    m.head.unigram_path = ""
    m.head.sampled_softmax_k = 0  # full CE: the head error must be over the real alphabet
    m.loop_depth = 1
    m.adapters.enabled = True
    m.adapters.pyramid.enabled = False
    m.adapters.ttt_lora.enabled = True
    m.adapters.ttt_lora.rank = args.rank
    cfg.train.grad_checkpointing = False
    cfg.train.batch_size = 1
    cfg.train.data.seq_len = args.seq
    validate_config(cfg)
    return cfg


def load_windows(args: argparse.Namespace, vocab: int) -> tuple[list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]], str]:
    """Return ``(measured, training, provenance)`` windows, all in one id space.

    The measured window is ``[--offset, --offset + seq)``; training windows are
    placed *after* it so the regime stress never sees the tokens it is scored on.
    ``--train-seq`` is shorter than ``--seq`` on purpose: a train step at the full
    window costs ~13s on this host, so a 120-step sweep would run 43 minutes, while
    the stress only needs the base to move off init. The dense re-indexing is
    built from the union of every window, so a model trained on several real
    windows is not memorising the token identities of one.
    """
    specs = [(args.offset, args.seq)]
    specs += [(args.offset + args.seq + k * args.train_seq, args.train_seq)
              for k in range(max(1, args.train_windows))]

    if args.tokens == "random":
        g = torch.Generator().manual_seed(args.seed + 1)
        wins = []
        for _, n in specs:
            raw = torch.randint(1, vocab, (n + 1,), generator=g)
            wins.append((raw[:-1].unsqueeze(0), raw[1:].unsqueeze(0)))
        prov = (f"RANDOM iid ids in [1,{vocab}) (seed {args.seed + 1}); measured T={args.seq}, "
                f"{len(wins) - 1} training windows T={args.train_seq}; supports a GEOMETRY claim "
                f"only, never a quality claim")
        return [wins[0]], wins[1:], prov

    need = specs[-1][0] + specs[-1][1] + 1
    path = dataset_path(args.data_dir, args.dataset)
    stream = np.memmap(path, dtype=np.uint32, mode="r")
    if int(stream.shape[0]) < need:
        raise SystemExit(f"{path}: {int(stream.shape[0])} tokens, need {need}")
    raw = np.concatenate([np.asarray(stream[o : o + n + 1], dtype=np.int64) for o, n in specs])
    _, inv = np.unique(raw, return_inverse=True)
    live = int(inv.max()) + 1
    if live > vocab:
        raise SystemExit(f"{live} distinct ids across {len(specs)} windows exceed vocab {vocab}; "
                         f"raise --vocab, lower --seq/--train-seq, or lower --train-windows")
    wins, cur = [], 0
    for _, n in specs:
        w = inv[cur : cur + n + 1]
        cur += n + 1
        wins.append((torch.from_numpy(w[:-1]).unsqueeze(0), torch.from_numpy(w[1:]).unsqueeze(0)))
    prov = (f"REAL text: {path.name}; measured window @ offset {args.offset} x {args.seq} tokens, "
            f"plus {len(wins) - 1} training windows of {args.train_seq} tokens placed after it "
            f"(no overlap with the measured one); all read as n+1 and shifted as "
            f"hagi.data.dataset does; {live}/{vocab} ids live across the union "
            f"(union-dense re-index, so repeat structure is the corpus's)")
    return [wins[0]], wins[1:], prov


# --------------------------------------------------------------------------
# the two producers
# --------------------------------------------------------------------------

def true_targets(model: HAGI, ids: Tensor, tgt: Tensor) -> tuple[list[Tensor], float]:
    """``g_i = dCE/d(block-i output)`` from the shipped contour's own harvest.

    ``TttRls._harvest`` registers forward hooks on every block, runs one
    teacher-forced pass under ``torch.enable_grad()``, and calls
    ``autograd.grad(ce, flat)``. With ``loop_depth=1`` each block fires once.
    """
    ttt = TttRls(model)
    ce, rows = ttt._harvest(ids, tgt, None)
    out = []
    for i in sorted(rows):
        firings = rows[i]
        if len(firings) != 1:
            raise SystemExit(f"block {i} fired {len(firings)}x; keep loop_depth=1")
        grad = firings[0][1].detach()
        out.append(grad.reshape(-1, grad.shape[-1]).float())
    return out, float(ce)


def head_error(model: HAGI, ids: Tensor, tgt: Tensor) -> tuple[Tensor, float, float, float]:
    """Closed form ``e = (p - onehot(t)) @ W_out`` plus its autograd validation.

    Returns ``(e, rel_err, ce, scale)`` where ``scale = logit_scale / N`` is the
    single positive factor that turns ``e`` into ``dCE/d(head input)``. The
    relative error is the proof that the closed form is the head error and not a
    lookalike: ``LMHead.loss`` builds ``hidden * logit_scale`` internally, so the
    graph node visible from outside is the head input, and the check has to land
    on fp32 round-off or the whole table is noise.
    """
    head = model.head
    with torch.enable_grad():
        out = model(ids, tgt, return_logits=True)
        (g_input,) = autograd.grad(out.ce, out.hidden)

    logits = out.logits.detach().float().reshape(-1, out.logits.shape[-1])
    err = logits.softmax(dim=-1)
    err[torch.arange(err.shape[0]), tgt.reshape(-1)] -= 1.0  # p - onehot(t)
    e = err @ head.weight.detach().float()

    scale = float(head.logit_scale.detach()) / e.shape[0]
    rel = float((g_input.detach().float().reshape(e.shape) - scale * e).norm()
                / (scale * e).norm().clamp_min(1e-30))
    return e, rel, float(out.ce.detach()), scale


def block_streams(model: HAGI, ids: Tensor) -> tuple[list[Tensor], list[Tensor], float]:
    """Detached per-block ``(input, output)`` streams plus ``max|h_i - in_{i+1}|``.

    One no-grad pass. The gap is the evidence for the claim in
    :func:`local_deltas`: ``HAGI._run_blocks`` feeds each block's output straight
    into the next block, so the "block-i output vs block-(i+1) input"
    discrepancy the brief suggested is identically zero here, not merely small.
    """
    ins: dict[int, Tensor] = {}
    outs: dict[int, Tensor] = {}
    blocks = list(model.blocks)

    def pre(i: int):
        def hook(_m: object, a: tuple[Tensor, ...]) -> None:
            ins[i] = a[0].detach()

        return hook

    def post(i: int):
        def hook(_m: object, _a: tuple, o: Tensor) -> None:
            outs[i] = o.detach()

        return hook

    handles = [b.register_forward_pre_hook(pre(i)) for i, b in enumerate(blocks)]
    handles += [b.register_forward_hook(post(i)) for i, b in enumerate(blocks)]
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()

    gap = 0.0
    for i in range(len(blocks) - 1):
        gap = max(gap, float((outs[i] - ins[i + 1]).abs().max()))
    return [ins[i] for i in range(len(blocks))], [outs[i] for i in range(len(blocks))], gap


def local_deltas(model: HAGI, ids: Tensor) -> tuple[list[Tensor], float]:
    """One gradient-free local producer: ``a_i = -(h_i - h_{i-1})``, the block's own undo.

    The brief's example was "block-i output vs block-(i+1) input discrepancy", which
    :func:`block_streams` shows is exactly zero in this architecture; the nearest
    non-degenerate local signal is the block's own residual update. Recorded, not
    silently swapped.
    """
    ins, outs, gap = block_streams(model, ids)
    acts = [(-(o - i)).reshape(-1, o.shape[-1]).float() for i, o in zip(ins, outs, strict=True)]
    return acts, gap


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def rowwise_cos(a: Tensor, b: Tensor) -> Tensor:
    """Per-position cosine between two ``[N, H]`` stacks."""
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    return num / den.clamp_min(1e-30)


def frobenius_cos(a: Tensor, b: Tensor) -> float:
    """Pooled cosine over all rows -- magnitude-weighted, not per-row."""
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def rms_ratio(a: Tensor, b: Tensor) -> float:
    """``rms(a) / rms(b)``, the scale a broadcast target would need."""
    return float(a.pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30))


def project(x: Tensor, a: Tensor) -> Tensor:
    """Project ``[N, H]`` rows onto the orthonormal column space of ``lora_A``.

    ``delta = scaling * (feat @ A) @ B.T`` has its columns in ``span(A)``, so
    ``x @ A`` is the coordinate vector of the reachable part of ``x`` and the
    cosine of the coordinates is the cosine of the projections.
    """
    return x @ a


def block_basis(model: HAGI) -> list[Tensor]:
    return [b.adapters.ttt_lora.lora_A.detach().float() for b in model.blocks]


def spearman(y: np.ndarray) -> float:
    """Rank correlation of ``y`` against its own index (depth)."""
    if len(y) < 3:
        return float("nan")
    rank = np.argsort(np.argsort(y)).astype(np.float64)
    idx = np.arange(len(y), dtype=np.float64)
    return float(np.corrcoef(rank, idx)[0, 1])


def control_matrix(g: list[Tensor]) -> np.ndarray:
    """``[L, L]`` rowwise-mean cosines between the TRUE block gradients."""
    n = len(g)
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = float(rowwise_cos(g[i], g[j]).mean())
            m[i, j] = m[j, i] = v
    return m


def usable_top_blocks(cos: np.ndarray, thresh: float) -> int:
    """Largest k such that the k topmost (last) blocks all clear ``thresh``."""
    k = 0
    for v in cos[::-1]:
        if v < thresh:
            break
        k += 1
    return k


# --------------------------------------------------------------------------
# one regime snapshot
# --------------------------------------------------------------------------

def measure(model: HAGI, ids: Tensor, tgt: Tensor) -> dict:
    """Full per-depth measurement at the model's current weights."""
    g, ce_true = true_targets(model, ids, tgt)
    e, rel, ce_head, scale = head_error(model, ids, tgt)
    basis = block_basis(model)
    L = len(g)

    cos = np.array([float(rowwise_cos(gi, e).mean()) for gi in g])
    cos_r = np.array([float(rowwise_cos(project(gi, a), project(e, a)).mean()) for gi, a in zip(g, basis, strict=True)])
    cos_f = np.array([frobenius_cos(gi, e) for gi in g])
    ratio = np.array([rms_ratio(gi, e) for gi in g])
    resid = np.array([float((gi - scale * e).norm() / gi.norm().clamp_min(1e-30)) for gi in g])
    med = np.array([float(rowwise_cos(gi, e).median()) for gi in g])
    p10 = np.array([float(rowwise_cos(gi, e).quantile(0.1)) for gi in g])

    cm = control_matrix(g)
    off = ~np.eye(L, dtype=bool)
    ceil = np.array([float(rowwise_cos(gi, g[-1]).mean()) for gi in g])

    # regime diagnostic: how much each block actually adds to the stream. The skip
    # approximation is only as good as this share is small.
    ins, outs, _ = block_streams(model, ids)
    share = np.array([float((o - i).pow(2).mean().sqrt() / o.pow(2).mean().sqrt().clamp_min(1e-30))
                      for i, o in zip(ins, outs, strict=True)])
    return {
        "L": L, "g": g, "e": e, "scale": scale,
        "cos": cos, "cos_r": cos_r, "cos_f": cos_f, "ratio": ratio,
        "resid": resid, "med": med, "p10": p10,
        "cm": cm, "ctl_mean": float(cm[off].mean()) if L > 1 else float("nan"),
        "ctl_min": float(cm[off].min()) if L > 1 else float("nan"),
        "ctl_max": float(cm[off].max()) if L > 1 else float("nan"),
        "ctl_argmin": tuple(int(v) for v in np.unravel_index(np.argmin(np.where(off, cm, 9e9)), cm.shape)),
        "ctl_argmax": tuple(int(v) for v in np.unravel_index(np.argmax(np.where(off, cm, -9e9)), cm.shape)),
        "ceil": ceil, "share": share,
        "ce": ce_true, "ce_head": ce_head, "rel": rel,
    }


def print_table(r: dict, label: str) -> None:
    L = r["L"]
    print(f"--- {label}   CE={r['ce']:.4f}   closed-form check rel={r['rel']:.2e}   "
          f"scale(logit_scale/N)={r['scale']:.3e}")
    hdr = (f"{'blk':>4} {'cos(g_i,e)':>11} {'med':>8} {'p10':>8} {'cos_r':>8} {'cos_F':>8} "
           f"{'rel_resid':>10} {'||g_i||/||e||':>14} {'rms(d_i)/rms(h_i)':>18} {'ctl cos(g_i,g_L-1)':>19}")
    print(hdr)
    print("-" * len(hdr))
    for i in range(L):
        print(f"{i:>4} {r['cos'][i]:>11.4f} {r['med'][i]:>8.4f} {r['p10'][i]:>8.4f} "
              f"{r['cos_r'][i]:>8.4f} {r['cos_f'][i]:>8.4f} {r['resid'][i]:>10.4f} "
              f"{r['ratio'][i]:>14.3e} {r['share'][i]:>18.4f} {r['ceil'][i]:>19.4f}")
    print("-" * len(hdr))
    print(f"     mean cos(g_i,e) = {r['cos'].mean():.4f} (range {r['cos'].min():.4f}..{r['cos'].max():.4f})   "
          f"mean cos_r = {r['cos_r'].mean():.4f}")
    print(f"     CONTROL mean cos(g_i,g_j), i!=j = {r['ctl_mean']:.4f}   "
          f"extremes: min {r['ctl_min']:.4f} (blk {r['ctl_argmin'][0]},{r['ctl_argmin'][1]})  "
          f"max {r['ctl_max']:.4f} (blk {r['ctl_argmax'][0]},{r['ctl_argmax'][1]})")
    print(f"     headline/control = {r['cos'].mean() / max(r['ctl_mean'], 1e-9):.3f}x   "
          f"(1.00x = e is no better than some *other* block's true gradient)")
    print(f"     depth: spearman = {spearman(r['cos']):+.3f}   monotone non-increasing: "
          f"{bool(np.all(np.diff(r['cos']) <= 1e-9))}   top-bottom = {r['cos'][-1] - r['cos'][0]:+.4f}")
    spread = r["ratio"].max() / r["ratio"].min()
    print(f"     scale: ||g_i||/||e|| spread {r['ratio'].min():.3e}..{r['ratio'].max():.3e} = {spread:.2f}x -> "
          f"{'ONE global scale is enough' if spread < 3 else 'PER-DEPTH scale needed'}")
    for th in (0.9, 0.5):
        print(f"     usable top-k with cos(g_i,e) >= {th}: {usable_top_blocks(r['cos'], th)} of {L}   "
              f"| cos_r >= {th}: {usable_top_blocks(r['cos_r'], th)} of {L}")
    print()


def train_base(model: HAGI, windows: list[tuple[Tensor, Tensor]], steps: int, lr: float) -> float:
    """Move the base off init so the skip path is stressed; return the last CE seen.

    This trains the frozen-in-production weights on purpose, over *several* real
    windows: the question is what happens when blocks contribute real signal, and
    at init with ``residual_scale = 1/sqrt(2L)`` they contribute almost none.
    Several windows keep the trained regimes from being one-window memorisation.
    """
    params = [p for n, p in model.named_parameters() if "adapters" not in n and p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    ce = float("nan")
    for s in range(steps):
        ids, tgt = windows[s % len(windows)]
        model.train()
        out = model(ids, tgt)
        ce = float(out.ce.detach())
        opt.zero_grad(set_to_none=True)
        out.ce.backward()
        opt.step()
    model.eval()
    return ce


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="closed-form head error vs autograd per-block targets")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024, help="teacher-forced window length (tokens)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", choices=("corpus", "random"), default="corpus")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dataset", default="edu")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--train-windows", type=int, default=4,
                    help="distinct real windows the trained regimes see (measured window is separate)")
    ap.add_argument("--train-seq", type=int, default=256,
                    help="window length for the training steps only (see load_windows)")
    ap.add_argument("--train-steps", type=int, nargs="*", default=[0, 20, 50, 120],
                    help="cumulative Adam steps on the base before each measured regime")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alt", choices=("auto", "on", "off"), default="auto",
                    help="local gradient-free producer; auto = run only if the headline is low")
    ap.add_argument("--alt-if", type=float, default=0.5,
                    help="worst headline mean cos(g_i,e) below which the alternative is worth testing")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = build_cfg(args)
    measured, training, prov = load_windows(args, cfg.model.vocab_size)
    ids, tgt = measured[0]
    torch.manual_seed(args.seed)
    model = HAGI(cfg).eval()
    params = sum(p.numel() for p in model.parameters())

    print(f"e6 producer validation | {cfg.model.num_layers} blocks, H={cfg.model.hidden_size}, "
          f"V={cfg.model.vocab_size}, inter={4 * cfg.model.hidden_size}, rank={args.rank}, "
          f"T={ids.shape[1]}, fp32 cpu, loop_depth=1, window=0 (all relays)")
    print(f"params: {params / 1e6:.2f}M ({params * 4 / MIB:.0f} MiB fp32)")
    print(f"head: W_out = model.head.weight {tuple(model.head.weight.shape)}, "
          f"tied_to_encoder={model.head.is_tied}, "
          f"logit_scale={float(model.head.logit_scale.detach()):.4f}")
    print(f"tokens: {prov}")
    probe = TttRls(model)
    print(f"contour: TttRls defaults (stream_frac={probe.stream_frac}, "
          f"refit_rows={probe.refit_rows}, prior={probe.prior})\n")

    steps = list(args.train_steps)
    results: list[dict] = []
    prev = 0
    alt: dict | None = None
    alt_worst = float("inf")
    for k in steps:
        if k < prev:
            raise SystemExit(f"--train-steps must be cumulative ascending, got {steps}")
        t0 = time.perf_counter()
        ce = train_base(model, training, k - prev, args.lr) if k > prev else float("nan")
        r = measure(model, ids, tgt)
        r["steps"] = k
        r["train_ce"] = ce
        results.append(r)
        if r["rel"] > 1e-4:
            raise SystemExit("closed-form head error does not match autograd at the head input")
        label = ("INIT (random weights)" if k == 0
                 else f"trained {k} Adam steps on {len(training)} real windows of {args.train_seq} tokens (CE {ce:.4f})")
        print_table(r, label)

        # The alternative is measured at the SAME weights as the gradients it is
        # compared against, and only for the worst regime seen so far -- otherwise
        # a later regime would leave it pointing at stale streams.
        if args.alt != "off" and float(r["cos"].mean()) < min(args.alt_if, alt_worst):
            alt_worst = float(r["cos"].mean())
            a, gap = local_deltas(model, ids)
            alt = {
                "steps": k,
                "gap": gap,
                "cos": np.array([float(rowwise_cos(ai, gi).mean()) for ai, gi in zip(a, r["g"], strict=True)]),
                "ratio": np.array([rms_ratio(gi, ai) for gi, ai in zip(r["g"], a, strict=True)]),
                "ctl": r["ctl_mean"],
                "head": float(r["cos"].mean()),
            }
        prev = k
        print(f"     [{time.perf_counter() - t0:.1f}s]\n")

    # ---- verdict across regimes -------------------------------------------------
    print("=" * 100)
    hdr = (f"{'steps':>7} {'CE':>8} {'mean cos':>9} {'mean cos_r':>11} {'mean resid':>11} "
           f"{'ctl mean':>9} {'cos/ctl':>8} {'spread ratio':>13}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['steps']:>7} {r['ce']:>8.3f} {r['cos'].mean():>9.4f} {r['cos_r'].mean():>11.4f} "
              f"{r['resid'].mean():>11.4f} {r['ctl_mean']:>9.4f} "
              f"{r['cos'].mean() / max(r['ctl_mean'], 1e-9):>8.3f} "
              f"{r['ratio'].max() / r['ratio'].min():>13.2f}x")
    print("-" * len(hdr))

    worst_r_idx = int(np.argmin([float(np.mean(r["cos"])) for r in results]))
    worst = results[worst_r_idx]
    init = results[0]
    print(f"\nheadline (init)       mean cos(g_i,e) = {init['cos'].mean():.4f}, "
          f"cos_r = {init['cos_r'].mean():.4f}, control = {init['ctl_mean']:.4f}")
    print(f"headline (worst @{worst['steps']} steps) mean cos(g_i,e) = {worst['cos'].mean():.4f}, "
          f"cos_r = {worst['cos_r'].mean():.4f}, control = {worst['ctl_mean']:.4f}")
    print(f"noise floor         cos/ctl at worst regime = "
          f"{worst['cos'].mean() / max(worst['ctl_mean'], 1e-9):.3f} "
          f"(1.00 = e is as good as some *other* block's true gradient)")
    print(f"miss fraction       mean ||g_i - scale*e|| / ||g_i|| = {worst['resid'].mean():.4f} "
          f"at {worst['steps']} steps (>1 = the error exceeds the target)")

    # ---- the one alternative ----------------------------------------------------
    if alt is not None:
        print(f"\nalternative producer (gradient-free, local) -- measured at {alt['steps']} steps, "
              f"the worst headline regime:")
        print(f"  literal 'block-i output vs block-(i+1) input' discrepancy: max|h_i - in_(i+1)| = "
              f"{alt['gap']:.1e} -> identically zero in this architecture, so the substitute measured "
              f"is a_i = -(h_i - h_(i-1)) (undo the block's own residual update)")
        print(f"  {'blk':>4} {'cos(g_i,a_i)':>13} {'||g_i||/||a_i||':>16}")
        for i in range(len(alt["cos"])):
            print(f"  {i:>4} {alt['cos'][i]:>13.4f} {alt['ratio'][i]:>16.3e}")
        print(f"  mean cos(g_i,a_i) = {alt['cos'].mean():.4f}  (head error {alt['head']:.4f}, "
              f"control floor {alt['ctl']:.4f})")
    else:
        print(f"\nalternative producer skipped: no regime fell below --alt-if {args.alt_if} "
              f"(the local signal is only worth testing when the head error fails)")

    # ---- B_true vs B_approx -----------------------------------------------------
    print("\n" + "=" * 100)
    print("B_true    = autograd dCE/d(block-i output), rescaled by _rows_for -> the Python contour")
    print("B_approx  = (softmax(logits) - onehot(t)) @ W_out, broadcast unchanged to every block")
    print("            -> the server contour, which has no backward pass")
    w = float(worst["cos"].mean())
    floor = float(worst["ctl_mean"])
    ks = usable_top_blocks(worst["cos"], 0.5)
    if w >= 0.9:
        verdict = f"YES -- no depth limit needed: worst-regime mean cos = {w:.4f} across all regimes"
    elif w >= 0.5:
        verdict = (f"YES-WITH-A-DEPTH-LIMIT -- worst-regime mean cos = {w:.4f}; "
                   f"restrict to the top {ks} of {worst['L']} blocks")
    elif w <= floor + 0.05 and w < 0.5:
        verdict = (f"NO -- worst-regime mean cos = {w:.4f} against that regime's own control floor "
                   f"{floor:.4f} (cos/ctl = {w / max(floor, 1e-9):.3f}), and the broadcast misses "
                   f"{worst['resid'].mean():.2f}x the target's own norm: the server cannot get a "
                   f"usable target without backprop")
    else:
        verdict = (f"YES-WITH-A-DEPTH-LIMIT -- worst-regime mean cos = {w:.4f} clears the control "
                   f"floor {floor:.4f} but not 0.5; restrict to the top {ks} of {worst['L']} blocks")
    print(f"\nVERDICT: {verdict}")
    if float(init["cos"].mean()) >= 0.9 and w < 0.5:
        print(f"NOTE: the init regime looks perfect (cos = {init['cos'].mean():.4f}) only because its "
              f"control floor is {init['ctl_mean']:.4f} -- at init every block gradient is nearly "
              f"every other block gradient, so a broadcast is accurate exactly where the target "
              f"carries no per-block information. The {worst['steps']}-step regime is the one that "
              f"matters.")
    print(f"peak RSS: {peak_rss_bytes() / MIB:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
