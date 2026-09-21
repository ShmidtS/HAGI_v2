"""Tests for the bootstrap LoRA GGUF (``scripts/bonsai_lora_bootstrap.py``).

The load-bearing pin here is ``test_name_set_matches_an_independent_enumeration``: the adapter
covers exactly the tensors the real GGUF says are adaptable, so a topology change in the model
file fails the suite instead of silently shipping a smaller adapter. The mutation tests at the
bottom exist because a verifier that cannot fail is decoration
(pi-rule ``evidence``: reproduce the breach before claiming the guard holds).

Format expectations are quoted from the fork, not from upstream memory — see the module
docstring of the script for the line-by-line provenance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFReader, GGUFType, GGUFValueType, GGUFWriter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bonsai_lora_bootstrap as B  # noqa: E402
import gguf_weights as G  # noqa: E402

#: Hand-written on purpose, from the task's coverage list, so the enumeration below is a real
#: second source rather than a call into the code under test.
LINEAR = ("attn_qkv", "attn_gate", "ssm_out", "ssm_alpha", "ssm_beta", "ffn_gate", "ffn_up", "ffn_down")
FULL = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")

N_LINEAR_BLOCKS = 48
N_FULL_BLOCKS = 16
EXPECTED_PAIRS = N_LINEAR_BLOCKS * len(LINEAR) + N_FULL_BLOCKS * len(FULL)  # 384 + 112 = 496

#: 1-D and non-linear tensors that must never carry a pair.
ONE_D_TAILS = (
    "attn_norm.weight",
    "post_attention_norm.weight",
    "ssm_norm.weight",
    "ssm_dt.bias",
    "ssm_a",
)


@pytest.fixture(scope="session")
def bases() -> list[tuple[str, tuple[int, int]]]:
    return B.adaptable_bases()


@pytest.fixture(scope="session")
def adapter(tmp_path_factory: pytest.TempPathFactory, bases) -> tuple[Path, dict]:
    """Write the real adapter once per session (~223 MiB, ~3 s) and reuse it."""
    out = tmp_path_factory.mktemp("lora") / "bonsai_lora_rank8.gguf"
    summary = B.write_adapter(out, bases=bases)
    return out, summary


def _independent_enumeration() -> dict[str, tuple[int, int]]:
    """Re-derive the expected base list straight from the GGUF table + donor config."""
    types = json.loads(B.CONFIG_PATH.read_text(encoding="utf-8"))["text_config"]["layer_types"]
    want: dict[str, tuple[int, int]] = {}
    for name, shape, _dtype in G.tensor_names():
        if not name.startswith("blk.") or G.gguf_to_hf_name(name) is None or len(shape) != 2:
            continue
        il, tail = int(name.split(".")[1]), name.split(".", 2)[2]
        family = LINEAR if types[il] == "linear_attention" else FULL
        if tail in {t + ".weight" for t in family}:
            want[name] = (int(shape[0]), int(shape[1]))
    return want


def test_counts(adapter, bases):
    _out, summary = adapter
    assert summary["pairs"] == EXPECTED_PAIRS == len(bases)
    assert summary["tensors"] == 2 * EXPECTED_PAIRS == 992
    assert summary["rank"] == B.RANK == 8


def test_file_loads_with_the_stock_reader(adapter):
    """The file is plain F32, so ``gguf.GGUFReader`` — not a tolerant subclass — accepts it."""
    out, _summary = adapter
    reader = GGUFReader(str(out))
    assert len(reader.tensors) == 992
    assert all(t.tensor_type.name == "F32" for t in reader.tensors)


def test_metadata_keys_and_value_types(adapter):
    """``ggml/src/gguf.cpp:194`` asserts the *type* of each getter, so types are the contract."""
    out, _summary = adapter
    reader = GGUFReader(str(out))
    assert reader.fields["general.type"].types[0] is GGUFValueType.STRING
    assert bytes(reader.fields["general.type"].parts[-1].tobytes()).decode() == GGUFType.ADAPTER
    assert bytes(reader.fields["general.architecture"].parts[-1].tobytes()).decode() == "qwen35"
    assert bytes(reader.fields["adapter.type"].parts[-1].tobytes()).decode() == "lora"
    assert reader.fields["adapter.lora.alpha"].types[0] is GGUFValueType.FLOAT32
    assert float(reader.fields["adapter.lora.alpha"].parts[-1][0]) == pytest.approx(1.0)


def test_no_version_or_rank_key(adapter):
    """The fork reads neither; inventing one would be Hyrum surface with no consumer.

    Asserted as the exact KV set, so a future key has to be justified here too —
    ``GGUF.*`` entries are the reader's own structural pseudo-fields (version, counts),
    not metadata we wrote.
    """
    out, _summary = adapter
    reader = GGUFReader(str(out))
    keys = {k for k in reader.fields if not k.startswith("GGUF.")}
    assert keys == {
        "general.architecture",
        "general.type",
        "adapter.type",
        "adapter.lora.alpha",
        "adapter.lora.task_name",
    }


def test_every_tensor_carries_a_loader_suffix(adapter):
    """``llama-adapter.cpp:273-292`` throws on any other suffix, so none may appear."""
    out, _summary = adapter
    reader = GGUFReader(str(out))
    for t in reader.tensors:
        name = str(t.name)
        assert name.endswith((".lora_a", ".lora_b")), name


def test_shapes_compose_against_the_base_table(adapter, bases):
    """The fork's own test: ``a->ne[0] == W->ne[0]``, ``b->ne[1] == W->ne[1]``, ``a->ne[1] == b->ne[0]``."""
    out, _summary = adapter
    reader = GGUFReader(str(out))
    expected = dict(bases)
    for t in reader.tensors:
        name = str(t.name)
        base, _, suffix = name.rpartition(".")
        suffix = "." + suffix
        assert suffix in (".lora_a", ".lora_b")
        out_dim, in_dim = expected[base]
        ne = [int(v) for v in t.shape]  # stored order: [ne[0], ne[1]]
        if suffix == ".lora_a":
            assert ne == [in_dim, B.RANK], f"{name}: ne={ne}, want {[(in_dim, B.RANK)]}"
            assert tuple(t.data.shape) == (B.RANK, in_dim)
        else:
            assert ne == [B.RANK, out_dim], f"{name}: ne={ne}"
            assert tuple(t.data.shape) == (out_dim, B.RANK)
        assert t.data.dtype == np.float32


def test_zero_filled(adapter):
    """The C++ slice owns the orthonormal ``lora_a`` draw; this file only reserves space."""
    out, _summary = adapter
    reader = GGUFReader(str(out))
    nonzero = [str(t.name) for t in reader.tensors if t.data.any()]
    assert not nonzero, f"bootstrap must be zero-filled, first offender: {nonzero[0]}"


def test_one_d_tensors_excluded(bases):
    tails = {name.split(".", 2)[2] for name, _ in bases}
    for tail in ONE_D_TAILS:
        assert tail not in tails, f"{tail} is 1-D and cannot carry a LoRA pair"
    assert "ssm_conv1d.weight" not in tails, "depthwise conv, never through build_lora_mm"
    # ...and they really are 1-D in the file, so the exclusion is not a naming accident.
    shapes = {n: s for n, s, _ in G.tensor_names()}
    for name, shape in shapes.items():
        if name.startswith("blk.") and name.split(".", 2)[2] in ONE_D_TAILS:
            assert len(shape) == 1, f"{name} is {shape}"


def test_both_families_covered(bases):
    per_block: dict[int, set[str]] = {}
    for name, _shape in bases:
        il, tail = name.split(".", 2)[1], name.split(".", 2)[2]
        per_block.setdefault(int(il), set()).add(tail)
    types = B.layer_types()
    assert len(per_block) == N_LINEAR_BLOCKS + N_FULL_BLOCKS == 64
    for il, tails in per_block.items():
        want = {t + ".weight" for t in (LINEAR if types[il] == "linear_attention" else FULL)}
        assert tails == want, f"blk.{il} ({types[il]}): {sorted(tails ^ want)}"
    assert sum(1 for i, t in enumerate(types) if t == "linear_attention") == N_LINEAR_BLOCKS
    assert sum(1 for i, t in enumerate(types) if t == "full_attention") == N_FULL_BLOCKS


def test_mtp_block_64_excluded(bases):
    assert not [n for n, _ in bases if n.startswith("blk.64.")], "blk.64 is the MTP draft head"
    assert not [n for n, _ in bases if n in ("token_embd.weight", "output.weight")], (
        "the embedding path has a flipped pair (llama-adapter.cpp:356-360) and is out of scope"
    )


def test_name_set_matches_an_independent_enumeration(bases):
    assert dict(bases) == _independent_enumeration()


def test_pair_shapes_orientation():
    a, b = B.pair_shapes((10240, 5120), rank=8)
    assert a == (8, 5120) and b == (10240, 8)  # numpy order; ne is the reverse
    with pytest.raises(ValueError, match="2-D"):
        B.pair_shapes((5120,), rank=8)
    with pytest.raises(ValueError, match="exceeds"):
        B.pair_shapes((4, 5120), rank=8)


def test_verify_accepts_the_file_it_wrote(adapter, bases):
    got = B.verify_adapter(adapter[0], bases=bases)
    assert got["pairs"] == EXPECTED_PAIRS
    assert got["linear_pairs"] == N_LINEAR_BLOCKS * len(LINEAR) == 384
    assert got["full_pairs"] == N_FULL_BLOCKS * len(FULL) == 112
    assert got["zero_filled"] is True


# --- mutation tests: the verifier must be capable of failing -------------------------


def _write_raw(out: Path, tensors: dict[str, np.ndarray], arch: str = "qwen35") -> None:
    writer = GGUFWriter(str(out), arch)
    writer.add_type(GGUFType.ADAPTER)
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", 1.0)
    for name, data in tensors.items():
        writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


BASE = ("blk.0.attn_qkv.weight", (10240, 5120))
IN, OUT, RANK = 5120, 10240, 8


def test_verify_rejects_transposed_lora_a(tmp_path):
    """The failure ``llama-adapter.cpp:366`` names: 'lora_a tensor is not transposed'."""
    path = tmp_path / "transposed.gguf"
    _write_raw(
        path,
        {
            f"{BASE[0]}.lora_a": np.zeros((IN, RANK), np.float32),  # swapped
            f"{BASE[0]}.lora_b": np.zeros((OUT, RANK), np.float32),
        },
    )
    with pytest.raises(AssertionError, match="lora_a"):
        B.verify_adapter(path, bases=[BASE])


def test_verify_rejects_half_pair(tmp_path):
    path = tmp_path / "half.gguf"
    _write_raw(path, {f"{BASE[0]}.lora_a": np.zeros((RANK, IN), np.float32)})
    with pytest.raises(AssertionError, match="missing one component"):
        B.verify_adapter(path, bases=[BASE])


def test_verify_rejects_bad_suffix(tmp_path):
    """``llama-adapter.cpp:287-292`` throws on any suffix but ``.lora_a``/``.lora_b``/``_norm.weight``."""
    path = tmp_path / "suffix.gguf"
    _write_raw(
        path,
        {
            f"{BASE[0]}.lora_a": np.zeros((RANK, IN), np.float32),
            f"{BASE[0]}.lora_b": np.zeros((OUT, RANK), np.float32),
            "blk.0.attn_qkv.weight.lora_c": np.zeros((RANK, IN), np.float32),
        },
    )
    with pytest.raises(AssertionError, match="unexpected suffix"):
        B.verify_adapter(path, bases=[BASE])


def test_verify_rejects_nonzero_fill(tmp_path):
    path = tmp_path / "filled.gguf"
    _write_raw(
        path,
        {
            f"{BASE[0]}.lora_a": np.ones((RANK, IN), np.float32),
            f"{BASE[0]}.lora_b": np.zeros((OUT, RANK), np.float32),
        },
    )
    with pytest.raises(AssertionError, match="zero-filled"):
        B.verify_adapter(path, bases=[BASE])


def test_verify_rejects_wrong_arch(tmp_path):
    path = tmp_path / "arch.gguf"
    _write_raw(
        path,
        {
            f"{BASE[0]}.lora_a": np.zeros((RANK, IN), np.float32),
            f"{BASE[0]}.lora_b": np.zeros((OUT, RANK), np.float32),
        },
        arch="qwen3",
    )
    with pytest.raises(AssertionError, match="general.architecture"):
        B.verify_adapter(path, bases=[BASE])
