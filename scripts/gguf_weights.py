"""Stream Ternary-Bonsai-2-27B GGUF tensors into torch, without touching host RAM.

Why this exists: the TTT contour (scripts/e5_ttt_full_depth.py) measures the
features -> delta mechanics on real *dimensions*, but on random initialisation.
Evolving the model the daemon actually serves needs the real weights, and the
only copy on this host is the PQ2_0 GGUF -- the HF caches under
~/.cache/huggingface/hub are 1.0K stubs, and donor/ holds just model_mtp
(849 MB, block 64 alone).

Memory contract, and why it is a *stream*: the full bf16 state dict is 41.8 GiB
against 31.6 GiB of host RAM. Building it in a dict is exactly the build order
that paged this machine to death (see the 5421394 note). So this module yields
one tensor at a time; the caller moves each to the gpu before taking the next.

Codec, from the reference in ggml (prism-llama/ggml/src/ggml-quants.c:494 and
ggml-common.h:202):

    QK_PQ2_0 = 128, block = {half d; uint8 qs[32]}  -> 34 bytes per 128 weights
    q = (qs[j // 4] >> ((j % 4) * 2)) & 3           # 2 bits, low pair first
    w = (q - 1) * d                                  # 00=-1 01=0 10=+1 11=+2

The header says PQ2_0 is "identical 2-bit codec to Q2_0, one fp16 scale per 128
weights" -- unlike Q2_0 there is no min term, so the four values are exact
multiples of a single scale. That is the invariant checked here: every dequantised
weight lies in {-d, 0, d, 2d}. Byte-exactness of the whole file was verified
separately: summing the computed row sizes over all 866 tensors reproduces
`last_end` and the reader's own data_offset (11,121,984) with no remainder.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

GGUF_PATH = Path(
    r"C:\HAGI_v2\models\ternary-bonsai-2-27b-mtp"
    r"\Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf")

PQ2_0 = 142          # GGML_TYPE_PQ2_0
Q8_0 = 8
F32, F16, BF16 = 0, 1, 30
QK_PQ2_0 = 128
BLOCK_PQ2_0 = 34     # 2 (fp16 scale) + 32 (2-bit codes)
QK_Q8_0 = 32
BLOCK_Q8_0 = 34      # 4 (fp32 scale) + 32 (int8)

_DT_SIZE = {F32: 4, F16: 2, BF16: 2}


def _load_tolerant_reader():
    """Reuse scripts/bonsai_gguf_load.py's reader rather than re-parse the header.

    The stock ``gguf.GGUFReader`` rejects type 142 outright (it validates the
    dtype against its enum), which is why the E0 loader carries a tolerant
    subclass. Importing it keeps one parser in the tree.
    """
    ref = Path(__file__).with_name("bonsai_gguf_load.py")
    spec = importlib.util.spec_from_file_location("bonsai_gguf_load", ref)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PQ20TolerantReader


def row_size(dtype: int, shape: tuple[int, ...]) -> int:
    """Bytes one tensor occupies in the data section, for the dtypes this file uses."""
    n = math.prod(shape)
    if dtype in _DT_SIZE:
        return n * _DT_SIZE[dtype]
    rows, cols = (shape[0], shape[1]) if len(shape) > 1 else (1, shape[0])
    if dtype == PQ2_0:
        if cols % QK_PQ2_0:
            raise ValueError(f"PQ2_0 columns not divisible by {QK_PQ2_0}: {shape}")
        return rows * (cols // QK_PQ2_0) * BLOCK_PQ2_0
    if dtype == Q8_0:
        if cols % QK_Q8_0:
            raise ValueError(f"Q8_0 columns not divisible by {QK_Q8_0}: {shape}")
        return rows * (cols // QK_Q8_0) * BLOCK_Q8_0
    raise ValueError(f"unsupported ggml type {dtype} for shape {shape}")


def dequant_pq2_0(raw: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Decode a PQ2_0 blob to float32, vectorised (the reference loops per element)."""
    rows, cols = shape
    nb = cols // QK_PQ2_0
    blk = raw.reshape(rows, nb, BLOCK_PQ2_0)
    d = blk[..., :2].copy().view(np.float16).reshape(rows, nb).astype(np.float32)
    qs = np.ascontiguousarray(blk[..., 2:]).astype(np.uint8)          # (rows, nb, 32)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    codes = ((qs[..., None] >> shifts) & 3).astype(np.float32)        # (rows, nb, 32, 4)
    codes = codes.reshape(rows, nb, QK_PQ2_0)
    return ((codes - 1.0) * d[..., None]).reshape(rows, cols)


def _check_pq2_0_invariant(w: np.ndarray, raw: np.ndarray, shape) -> None:
    """Every weight must be an exact multiple in {-1, 0, 1, 2} of its block scale.

    This is what makes the check a proof and not a smoke test: a wrong stride, a
    swapped byte order or an off-by-one in the bit packing produces values that
    are still finite but no longer lie on that four-point lattice.
    """
    rows, cols = shape
    nb = cols // QK_PQ2_0
    blk = raw.reshape(rows, nb, BLOCK_PQ2_0)
    d = blk[..., :2].copy().view(np.float16).reshape(rows, nb).astype(np.float32)
    w4 = w.reshape(rows, nb, QK_PQ2_0)
    safe = np.where(np.abs(d) > 0, d, 1.0)
    mult = w4 / safe[..., None]
    rounded = np.rint(mult)
    if not np.allclose(mult, rounded, atol=1e-4):
        raise AssertionError("PQ2_0 decode left the {-1,0,1,2} lattice")
    off = np.setdiff1d(np.unique(rounded.astype(np.int8)), np.array([-1, 0, 1, 2]))
    if off.size:
        raise AssertionError(f"PQ2_0 multipliers outside {-1, 0, 1, 2}: {off.tolist()}")


_READERS: dict[Path, object] = {}


def _reader(path: Path):
    """One parsed header per path.

    Building the tolerant reader decodes every F32/Q8_0 tensor, which costs
    seconds on an 8 GiB file; the tests and any caller that lists names before
    streaming would otherwise pay it twice.
    """
    r = _READERS.get(path)
    if r is None:
        r = _load_tolerant_reader()(path)
        _READERS[path] = r
    return r


def tensor_names(path: Path = GGUF_PATH) -> list[tuple[str, tuple[int, ...], int]]:
    return [(e["name"], tuple(e["shape"]), e["dtype"]) for e in _reader(path).custom_tensors]


def stream_tensors(path: Path = GGUF_PATH, verify: bool = False,
                   only: set[str] | None = None):
    """Yield ``(name, np.ndarray float32)`` one tensor at a time.

    ``only`` stops the walk once those names have been produced -- tensors are
    laid out by offset, so a caller wanting one late tensor would otherwise
    decode everything before it (measured: 14s to reach ``blk.0.ssm_alpha``).
    ``verify`` turns on the PQ2_0 lattice check. It costs a second pass over the
    block bytes, so it is off by default and on in the tests.
    """
    reader = _reader(path)
    import gguf  # noqa: PLC0415  (only needed for the Q8_0 reference decoder)
    from gguf.constants import GGMLQuantizationType  # noqa: PLC0415

    data = np.memmap(path, dtype=np.uint8, mode="r")
    base = reader.data_offset
    todo = set(only) if only is not None else None
    for e in sorted(reader.custom_tensors, key=lambda x: x["offset"]):
        name, shape, dt = e["name"], tuple(e["shape"]), e["dtype"]
        if todo is not None and name not in todo:
            continue
        n = row_size(dt, shape)
        if todo is not None:
            # Discard here, not per branch: a name claimed by only one dtype
            # branch used to leave the set non-empty at the end, so asking for a
            # bf16 or f32 tensor raised a spurious KeyError.
            todo.discard(name)
        raw = np.asarray(data[base + e["offset"]: base + e["offset"] + n])
        if dt in (F32, F16, BF16):
            if dt == BF16:
                # bf16 is a truncated float32: widen through the upper half of a
                # float32 word rather than reading it as fp16.
                u16 = raw.copy().view(np.uint16).astype(np.uint32)
                yield name, (u16 << 16).view(np.float32).reshape(shape)
                continue
            view = raw.copy().view(np.dtype("<f4" if dt == F32 else "<f2"))
            yield name, view.astype(np.float32).reshape(shape)
            continue
        if dt == PQ2_0:
            w = dequant_pq2_0(raw, shape)
            if verify:
                _check_pq2_0_invariant(w, raw, shape)
            yield name, w
            continue
        if dt == Q8_0:
            rows, cols = shape
            byte_shape = (rows, cols // QK_Q8_0 * BLOCK_Q8_0)
            w = gguf.dequantize(raw.reshape(byte_shape),
                                GGMLQuantizationType.Q8_0)
            yield name, np.asarray(w, dtype=np.float32).reshape(rows, cols)
            continue
        raise ValueError(f"unsupported ggml type {dt} for {name}")
    if todo:
        raise KeyError(f"tensors not present in {path.name}: {sorted(todo)}")


def gpu_footprint_gib(names_shapes, dtype_bytes: int = 2) -> float:
    """Projected device footprint, for the caller's abort guard."""
    return sum(math.prod(s) for _, s, _ in names_shapes) * dtype_bytes / 2**30


if __name__ == "__main__":
    import sys

    only = sys.argv[1] if len(sys.argv) > 1 else None
    infos = tensor_names()
    print(f"{len(infos)} tensors")
    print(f"full bf16 footprint: {gpu_footprint_gib(infos):.1f} GiB "
          f"(host RAM is 31.6 GiB, so this streams to the gpu)")
    for name, shape, dt in infos:
        if only and only not in name:
            continue
        for tname, arr in stream_tensors(verify=True):
            if tname == name:
                print(f"  {name:26s} {str(shape):16s} dtype={dt} "
                      f"rms={float(np.sqrt((arr ** 2).mean())):.5f} "
                      f"min={arr.min():.4f} max={arr.max():.4f}")
                break
        break
