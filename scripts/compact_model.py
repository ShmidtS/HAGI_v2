"""Apply the no-quantization compaction pipeline to a trained model.

Pipeline (per 2D weight matrix):
  1. Rotation — orthonormal transform (Hadamard/DFT-3) that concentrates
     per-channel energy into fewer coordinates (transform coding).
  2. Sorting — reorder output channels by their rotated energy (norm).
  3. Sparsify — drop the smallest coordinates in the rotated basis; only the
     survivors are kept (lossy compaction WITHOUT int quantization: the
     survivors stay in fp32/bf16, dropped ones become zero).
  4. Packing — after rotation+sorting+sparsify, store the matrix compactly
     (rows are sorted by energy, so a run-length / row-cut encoding packs the
     zeros; ternary body already stores 1.585 bit/weight).
  5. Convolution — a 1D conv over the channel axis can merge neighbouring
     coordinates; here we report the theoretical gain of a conv-2 grouping.

Reports per-tensor: original bytes, rotated-sparsified bytes, and the relative
error (Frobenius) introduced by the sparsification at a given keep ratio.

Usage:
    python scripts/compact_model.py --ckpt checkpoints_l0_merged/step-0002000.pt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.train.checkpoint import load_payload  # noqa: E402


# --- orthonormal rotation bases ---------------------------------------------


def _hadamard_matrix(n: int) -> torch.Tensor:
    if n == 1:
        return torch.ones(1, 1)
    half = _hadamard_matrix(n // 2)
    top = torch.cat([half, half], dim=1)
    bot = torch.cat([half, -half], dim=1)
    return torch.cat([top, bot], dim=0)


def _hadamard_orthonormal(n: int) -> torch.Tensor:
    if n & (n - 1):
        m = 1
        while m < n:
            m <<= 1
        h = _hadamard_matrix(m)[:n].float()
        q, _ = torch.linalg.qr(h)
        return q
    return (_hadamard_matrix(n) / math.sqrt(n)).float()


def _dft3_pair(z: torch.Tensor, k: int) -> torch.Tensor:
    a = z[..., 0]
    b = z[..., 1]
    s = math.sqrt(3) / 2.0
    if k == 1:
        return torch.stack((-0.5 * a - s * b, s * a - 0.5 * b), dim=-1)
    return torch.stack((-0.5 * a + s * b, -s * a - 0.5 * b), dim=-1)


def _dft3_blocks3(x: torch.Tensor, bd: int) -> torch.Tensor:
    xb = x.reshape(x.shape[:-1] + (3, bd // 2, 2))
    z0, z1, z2 = xb[..., 0, :, :], xb[..., 1, :, :], xb[..., 2, :, :]
    s = 1.0 / math.sqrt(3)
    y0 = s * (z0 + z1 + z2)
    y1 = s * (z0 + _dft3_pair(z1, 1) + _dft3_pair(z2, 2))
    y2 = s * (z0 + _dft3_pair(z1, 2) + _dft3_pair(z2, 1))
    return torch.stack([y0, y1, y2], dim=-3).reshape(x.shape)


def rotate(x: torch.Tensor, axis: int, kind: str) -> torch.Tensor:
    """Apply an orthonormal rotation along ``axis`` of a 2D weight."""
    n = x.shape[axis]
    if kind == "hadamard" and n <= 512:
        Q = _hadamard_orthonormal(n)
    elif kind == "dft3" and n % 2 == 0 and _is_pow3(n):
        # DFT-3 on complex pairs along this axis: x is [a, b]; pairs on the
        # axis dim -> reshape [a, n/2, 2] ... but we handle the generic case
        # by transposing the axis to the last dim.
        xr = x.movedim(axis, -1)
        shape = xr.shape
        flat = xr.reshape(-1, n)
        out = _dft3_blocks3(flat, n // 3).reshape(shape) if n == 3 else _dft3_apply_flat(flat, n)
        return out.movedim(-1, axis)
    else:
        return x  # no rotation for this size/kind
    if axis == 0:
        return Q @ x
    return x @ Q.t()


def _is_pow3(n: int) -> bool:
    while n % 3 == 0:
        n //= 3
    return n == 1


def _dft3_apply_flat(x: torch.Tensor, n_blocks: int) -> torch.Tensor:
    bd = x.shape[-1] // n_blocks
    if n_blocks == 3:
        return _dft3_blocks3(x, bd)
    xb = x.reshape(x.shape[:-1] + (n_blocks // 3, 3, bd))
    xc = xb.reshape(xb.shape[:-2] + (3 * bd,))
    yc = _dft3_blocks3(xc, bd)
    y2 = yc.reshape(x.shape)
    return _dft3_apply_flat(y2, n_blocks // 3)


def compact_2d(w: torch.Tensor, keep: float, kind: str, conv_group: int) -> dict:
    """Compact a [out, in] weight: rotate, sort, sparsify, pack.

    Returns a stats dict. ``keep`` is the fraction of rotated coordinates kept
    per output row; ``conv_group`` reports the extra gain of merging
    ``conv_group`` adjacent coordinates (theoretical, no-op here).
    """
    out, inn = w.shape
    orig_bytes = w.numel() * 2  # bf16
    wf = w.float()
    # 1. rotation over the input axis (concentrates energy per output row)
    wr = rotate(wf, 1, kind)
    # 2. GLOBAL sorting of the rotated input coordinates by their total energy
    #    across all output rows (one permutation for the whole matrix, so it is
    #    invertible and keep=1.0 is lossless).
    energy = wr.pow(2).sum(0)  # [in]
    idx = torch.argsort(energy, descending=True)
    wr_sorted = wr[:, idx]
    k = max(1, int(inn * keep))
    # 3. sparsify: keep the top-k most important coordinates, zero the rest.
    ws = torch.zeros_like(wr_sorted)
    ws[:, :k] = wr_sorted[:, :k]
    # 4. packing estimate: store [out, k] survivors (bf16) + [in] permutation
    #    + a trailing-cut flag (zeros are at the end, so no per-row mask needed).
    packed_bytes = out * k * 2 + inn * 2
    # 5. convolution grouping: merging conv_group adjacent coords halves/...
    conv_bytes = packed_bytes // conv_group if conv_group > 1 else packed_bytes
    err = (ws - wr_sorted).norm() / wr_sorted.norm().clamp_min(1e-12)
    return {
        "shape": (out, inn),
        "orig_bytes": orig_bytes,
        "packed_bytes": packed_bytes,
        "conv_bytes": conv_bytes,
        "err": float(err),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_l0_merged/step-0002000.pt")
    ap.add_argument("--keep", type=float, default=0.5, help="fraction of rotated coords kept")
    ap.add_argument("--kind", default="hadamard", choices=["hadamard", "dft3", "none"])
    ap.add_argument("--conv-group", type=int, default=2)
    args = ap.parse_args()

    payload = load_payload(args.ckpt, "cpu")
    sd = payload["model"]
    print(f"model: {args.ckpt}")
    print(f"rotation={args.kind} keep={args.keep} conv_group={args.conv_group}\n")
    print(f"{'tensor':46s} {'orig MB':>8s} {'packed MB':>10s} {'conv MB':>8s} {'err':>8s}")

    tot_orig = tot_packed = tot_conv = 0.0
    for k in sorted(sd):
        v = sd[k]
        if v.ndim != 2:
            continue
        stats = compact_2d(v, args.keep, args.kind, args.conv_group)
        if stats["orig_bytes"] < 1_000_000:
            continue  # skip tiny tensors in the report
        tot_orig += stats["orig_bytes"]
        tot_packed += stats["packed_bytes"]
        tot_conv += stats["conv_bytes"]
        print(
            f"{k[:46]:46s} {stats['orig_bytes']/1e6:8.2f} {stats['packed_bytes']/1e6:10.2f} "
            f"{stats['conv_bytes']/1e6:8.2f} {stats['err']:8.4f}"
        )
    print("\n" + "-" * 90)
    print(
        f"TOTAL  {tot_orig/1e6:8.2f} MB -> packed {tot_packed/1e6:.2f} MB "
        f"({100*tot_packed/tot_orig:.1f}%) | conv {tot_conv/1e6:.2f} MB "
        f"({100*tot_conv/tot_orig:.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
