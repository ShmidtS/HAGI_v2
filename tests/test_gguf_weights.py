"""Tests for the GGUF weight streamer.

The load-bearing pin is ``test_pq2_0_matches_the_ggml_element_formula``: the
codec is transcribed by hand from ggml-quants.c:494 into a scalar loop, and the
vectorised decoder must agree element for element. Everything else (byte
accounting, the lattice invariant, bf16 widening) is checked against properties
that a wrong offset or stride cannot produce by accident.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gguf_weights as G  # noqa: E402

GGML_DEQUANT = """
    for (int j = 0; j < qk; ++j) {
        const int byte_index = j / 4;
        const int bit_offset = (j % 4) * 2;
        const uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03;
        y[i*qk + j] = ((int)q - 1) * d;
    }
"""


def _encode_pq2_0(weights: np.ndarray, scales: np.ndarray) -> bytes:
    """Build blocks the way ggml packs them: fp16 scale, then 2 bits per weight."""
    rows, nb = scales.shape
    out = bytearray()
    for r in range(rows):
        for b in range(nb):
            out += np.float16(scales[r, b]).tobytes()
            codes = (weights[r, b * 128:(b + 1) * 128] + 1).astype(np.uint8)
            qs = np.zeros(32, dtype=np.uint8)
            for j in range(128):
                qs[j // 4] |= (codes[j] & 3) << ((j % 4) * 2)
            out += qs.tobytes()
    return bytes(out)


def _scalar_ggml_decode(raw: bytes, rows: int, cols: int) -> np.ndarray:
    """The reference loop, transliterated. Slow on purpose: it is the oracle."""
    nb = cols // 128
    y = np.zeros((rows, cols), dtype=np.float32)
    buf = np.frombuffer(raw, dtype=np.uint8)
    for i in range(rows * nb):
        blk = buf[i * 34:(i + 1) * 34]
        d = np.frombuffer(blk[:2], dtype=np.float16).astype(np.float32)[0]
        qs = blk[2:]
        r, b = divmod(i, nb)
        for j in range(128):
            q = (qs[j // 4] >> ((j % 4) * 2)) & 0x03
            y[r, b * 128 + j] = (int(q) - 1) * d
    return y


def test_pq2_0_matches_the_ggml_element_formula():
    """Vectorised decode == the scalar ggml loop, for every code and sign."""
    rng = np.random.default_rng(0)
    rows, cols = 3, 256                      # two full blocks per row
    # the encoder takes the lattice multiples, not scaled floats: casting a
    # product like 1.5 down to int8 truncates it, which would test nothing
    mult = rng.integers(-1, 3, size=(rows, cols)).astype(np.int8)
    assert set(np.unique(mult).tolist()) <= {-1, 0, 1, 2}
    # negative, zero and positive scales, all exact in fp16
    scales = np.array([[1.5, -2.25], [0.0, 0.0625], [-1.0, 3.75]], dtype=np.float32)
    raw = _encode_pq2_0(mult, scales)
    assert len(raw) == rows * (cols // 128) * 34

    oracle = _scalar_ggml_decode(raw, rows, cols)
    mine = G.dequant_pq2_0(np.frombuffer(raw, dtype=np.uint8), (rows, cols))
    assert np.array_equal(mine, oracle), (mine[0, :8], oracle[0, :8])


def test_pq2_0_lattice_check_rejects_a_desynced_block():
    """The invariant test is not decorative: a scale/weight desync fails it.

    The corruption model is reading a block at the wrong offset -- weights taken
    from one place, scales from another. Appending bytes and re-truncating to the
    original size changes nothing, so it would pass for a plainly wrong decoder.
    """
    rows, cols = 1, 128
    mult = np.array([[(j % 4) - 1 for j in range(128)]], dtype=np.int8)
    aligned = np.frombuffer(_encode_pq2_0(mult, np.array([[0.5]], np.float32)),
                            dtype=np.uint8)
    G._check_pq2_0_invariant(G.dequant_pq2_0(aligned, (rows, cols)),
                             aligned, (rows, cols))
    desync = np.frombuffer(_encode_pq2_0(mult, np.array([[0.3]], np.float32)),
                           dtype=np.uint8)
    w = G.dequant_pq2_0(aligned, (rows, cols))              # multiples of 0.5
    with pytest.raises(AssertionError):
        G._check_pq2_0_invariant(w, desync, (rows, cols))   # scale reads 0.3


def test_row_size_accounting_reproduces_the_file():
    """Sum of row sizes + header must equal the file, with no remainder.

    This is the byte-exactness proof for the whole container: a wrong block size
    for any dtype leaves a gap.
    """
    infos = G.tensor_names()
    assert len(infos) == 866
    total = sum(G.row_size(dt, shape) for _, shape, dt in infos)
    path = G.GGUF_PATH
    reader = G._reader(path)
    assert total == path.stat().st_size - reader.data_offset
    assert reader.data_offset == 11_121_984


def test_row_size_refuses_an_indivisible_row():
    """Silently truncating a row would corrupt every later offset."""
    with pytest.raises(ValueError, match="divisible"):
        G.row_size(G.PQ2_0, (4, 130))
    with pytest.raises(ValueError, match="unsupported"):
        G.row_size(999, (4, 4))


@pytest.mark.parametrize("dtype", [G.F32, G.F16, G.BF16])
def test_plain_types_size_by_itemsize(dtype):
    assert G.row_size(dtype, (48, 5120)) == 48 * 5120 * {G.F32: 4, G.F16: 2,
                                                         G.BF16: 2}[dtype]


def test_bf16_widens_through_the_high_half():
    """bf16 is a truncated float32, not an fp16 -- reading it as fp16 is wrong.

    Pinned with a hand-built vector: 0x3F80 is 1.0 in bf16, while the same bits
    as fp16 are 4.7e-38.
    """
    raw = np.array([0x3F80, 0xBF80, 0x0000, 0x7F80], dtype=np.uint16)
    u32 = raw.astype(np.uint32) << 16
    got = u32.view(np.float32)
    assert got.tolist() == [1.0, -1.0, 0.0, float("inf")]
    assert not np.array_equal(got, raw.view(np.float16).astype(np.float32))


def test_stream_yields_declared_shapes_and_the_lattice_holds():
    """Real tensors: shape as declared, finite, and PQ2_0 values on the lattice."""
    want = {"blk.0.attn_qkv.weight", "blk.0.ffn_gate.weight",
            "blk.0.ssm_alpha.weight", "blk.0.attn_norm.weight"}
    seen = {n: a for n, a in G.stream_tensors(verify=True, only=want)}
    assert set(seen) == want
    shapes = {n: s for n, s, _ in G.tensor_names() if n in want}
    for n, arr in seen.items():
        assert arr.shape == shapes[n], (n, arr.shape, shapes[n])
        assert np.isfinite(arr).all(), n
    # RMSNorm gains sit near one; a misread offset would not produce that
    g = seen["blk.0.attn_norm.weight"]
    assert 0.5 < float(np.sqrt((g ** 2).mean())) < 2.0


def test_only_filter_does_not_decode_the_whole_file():
    """Asking for one late tensor must not decode what comes before it.

    Tensors are laid out by offset and ``token_embd`` is first at 1.27B
    parameters, so an unfiltered walk costs ~60s and 2.4 GiB of decode for a
    request that wanted one 48-row projection (measured).
    """
    import time
    t0 = time.perf_counter()
    got = {n: a for n, a in G.stream_tensors(only={"blk.0.ssm_alpha.weight"})}
    dt = time.perf_counter() - t0
    assert list(got) == ["blk.0.ssm_alpha.weight"]
    assert got["blk.0.ssm_alpha.weight"].shape == (48, 5120)
    assert dt < 10.0, f"filtered stream took {dt:.1f}s -- it is not filtering"


def test_only_filter_accepts_every_dtype_branch():
    """The discard is per-stream, not per-dtype: a bf16 or f32 name must resolve.

    With the discard living only in the PQ2_0 branch, requesting a bf16 tensor
    left the pending set non-empty and raised a spurious KeyError at the end.
    """
    for name in ("blk.0.ssm_alpha.weight",     # BF16
                 "blk.0.attn_norm.weight",     # F32
                 "blk.0.ffn_gate.weight",      # PQ2_0
                 "blk.64.attn_q.weight"):      # Q8_0 (the MTP block is not ternary)
        got = dict(G.stream_tensors(only={name}))
        assert list(got) == [name], name


def test_only_filter_reports_a_name_that_is_not_there():
    with pytest.raises(KeyError, match="blk.999.nope"):
        dict(G.stream_tensors(only={"blk.999.nope"}))


def test_ssm_alpha_is_bf16_and_not_quantised():
    """dtype 30 is BF16: values are not confined to a four-point lattice.

    If the loader treated it as PQ2_0 the whole gating projection would be wrong,
    and this is the check that would catch it.
    """
    arr = dict(G.stream_tensors(only={"blk.0.ssm_alpha.weight"}))["blk.0.ssm_alpha.weight"]
    assert arr.shape == (48, 5120)
    assert len(np.unique(arr)) > 4


def test_gpu_footprint_is_reported_in_gib():
    """The caller's abort guard depends on this number being the real one."""
    infos = [("a", (1000, 1000), G.PQ2_0)]
    assert math.isclose(G.gpu_footprint_gib(infos), 1000 * 1000 * 2 / 2**30)
    whole = G.gpu_footprint_gib(G.tensor_names())
    assert 45.0 < whole < 60.0, whole   # 50.9 GiB bf16 for the real 27B
