# Exact-engine construction kit — ported from voidash/qwenbend
# bendvm/engine_circuit.py (Apache-2.0): the SwiGLU exact-AND gadget,
# one-hot nibble word encoding (decode_word), latch-head discipline and
# energy ballast; DeltaNet-state-as-RAM technique credited by qwenbend to
# divagr18/qwenputer (MIT), also used by vixhal-baraiya/qwenGPU (MIT).
# Verified in real torch (fp32 and bf16): silu(128) == 128 exactly, the
# gadget output is exactly the weights the builder wrote.

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.model.exact_engine import (  # noqa: E402
    ExactAND,
    NibbleWord,
    energy_ballast,
    latch_head_config,
)

H, INTER, ONE = 256, 64, 255


def _fresh():
    gw = torch.zeros(INTER, H)
    uw = torch.zeros(INTER, H)
    dw = torch.zeros(H, INTER)
    return ExactAND(gw, uw, dw, ONE), gw, uw, dw


def _run(gw, uw, dw, x):
    return dw @ (torch.nn.functional.silu(gw @ x) * (uw @ x))


def _x_with(dims: set[int]) -> torch.Tensor:
    x = torch.zeros(H)
    x[ONE] = 1.0
    for d in dims:
        x[d] = 1.0
    return x


def test_and_gadget_conjunction_and_or_group():
    m, gw, uw, dw = _fresh()
    m.unit([3, [7, 8]], {10: 1.0, 11: -1.0})  # fires iff d3 AND (d7 OR d8)
    for dims, want in [({3, 7}, 1.0), ({3, 8}, 1.0), ({3}, 0.0), ({7}, 0.0), ({3, 7, 20}, 1.0)]:
        got = float(_run(gw, uw, dw, _x_with(dims))[10])
        assert got == want, (dims, got, want)


def test_and_gadget_negative_features():
    m, gw, uw, dw = _fresh()
    m.unit([3], [12], negs=[5])  # fires iff d3 == 1 and d5 == 0
    assert float(_run(gw, uw, dw, _x_with({3}))[12]) == 1.0
    assert float(_run(gw, uw, dw, _x_with({3, 5}))[12]) == 0.0


def test_and_gadget_exact_in_bf16():
    m, gw, uw, dw = _fresh()
    m.unit([3, [7, 8]], {10: 1.0, 11: -1.0})
    o = _run(gw.bfloat16(), uw.bfloat16(), dw.bfloat16(), _x_with({3, 7}).bfloat16())
    assert float(o[10]) == 1.0
    assert float(o[11]) == -1.0


def test_gated_up_scaling_form():
    m, gw, uw, dw = _fresh()
    # up override: gated scaling form — output = silu(256) * w = 256 * w exactly
    m.unit([3], {10: 1.0}, up={ONE: 0.5})
    assert float(_run(gw, uw, dw, _x_with({3}))[10]) == 128.0


def test_and_gadget_noise_robust_margins():
    """qwenputer half-unit margins: a false AND is exactly 0 even when the
    literals carry fp32 noise up to ~0.25 per dim (gate stays <= -256)."""
    m, gw, uw, dw = _fresh()
    m.unit([3, [7, 8]], {10: 1.0, 11: -1.0})
    x = _x_with({3})
    x[7] = 0.2  # noisy false literal
    o = _run(gw, uw, dw, x)
    assert float(o[10]) == 0.0
    # fired side tolerates noise too: gate 256 - 512*0.01 = 254.9, output ~0.996
    x2 = _x_with({3, 7})
    x2[3] = 0.99
    o2 = _run(gw, uw, dw, x2)
    assert 0.95 < float(o2[10]) <= 1.0


def test_unit_capacity_and_shape_validation():
    m, gw, uw, dw = _fresh()
    small = ExactAND(torch.zeros(2, H), torch.zeros(2, H), torch.zeros(H, 2), ONE)
    small.unit([3], [0])
    small.unit([4], [1])
    with pytest.raises(ValueError):
        small.unit([5], [0])
    with pytest.raises(ValueError):
        ExactAND(torch.zeros(4, H), torch.zeros(8, H), torch.zeros(H, 4), ONE)


# --- NibbleWord -------------------------------------------------------------


def test_nibble_word_roundtrip_and_garbage_detection():
    for v in (0, 1, 0x8000_0000, 0xDEAD_BEEF, 2**32 - 1):
        row = torch.zeros(NibbleWord.DIMS)
        for d in NibbleWord.encode(v):
            row[d] = 1.0
        assert NibbleWord.decode(row) == v
    row = torch.zeros(NibbleWord.DIMS)
    for d in NibbleWord.encode(12345):
        row[d] = 1.0
    assert NibbleWord.decode(row.index_fill_(0, torch.tensor([0]), 0.5)) is None
    assert NibbleWord.decode(row * 0.0) is None
    with pytest.raises(ValueError):
        NibbleWord.encode(-1)
    with pytest.raises(ValueError):
        NibbleWord.encode(2**32)


def test_nibbles_helper():
    assert NibbleWord.nibbles(0x1234, 3) == [4, 3, 2]
    assert NibbleWord.nibbles(0, 4) == [0, 0, 0, 0]


# --- latch head -------------------------------------------------------------


def test_latch_head_config_contract():
    cfg = latch_head_config(4, list(range(4)), list(range(4)), list(range(128)), write_dim=200)
    assert cfg["k"].shape == (4, 4)
    assert cfg["q"].shape == (4, 4)
    assert cfg["v"].shape == (128, 4)
    assert float(cfg["beta"][200]) == 1.0
    assert float(cfg["beta"].sum()) == 1.0  # opens ONLY on the write token
    with pytest.raises(ValueError):
        latch_head_config(4, [0, 1], list(range(4)), list(range(128)), 0)


# --- ballast ----------------------------------------------------------------


def test_energy_ballast_hits_target_with_powers_of_two():
    e = torch.zeros(64)
    energy_ballast(e, 100.0)
    assert abs(float((e**2).sum()) - 100.0) < 1e-6
    with pytest.raises(ValueError):
        energy_ballast(torch.ones(4), 1.0)
