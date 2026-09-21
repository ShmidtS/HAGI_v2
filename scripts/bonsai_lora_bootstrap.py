#!/usr/bin/env python3
"""Write a zero-filled LoRA GGUF sized for Ternary-Bonsai-2-27B, shaped for *this* runtime.

The runtime is the llama.cpp fork at ``C:\\\\Users\\\\shmid\\\\AppData\\\\Local\\\\Temp\\\\prism-llama``.
Every format choice below was read out of that tree on 2026-09-21 rather than taken from
upstream llama.cpp memory; the line numbers are quoted so a reader can re-check them.
Upstream conventions differ in two places that matter (``lora.version``,
``adapter.lora.rank``) — see FORK FACTS.

What this file is for
---------------------
The in-server online LoRA fitter starts from this container: a later C++ slice loads it into
the running model, overwrites ``lora_a`` with its own orthonormal draw (deliberately *not*
generated here — cross-language RNG agreement is a hazard this design avoids), harvests
per-block ``(mixer input, block output)`` rows, solves ``B`` in closed form, and writes the
result into the live adapter buffers. So the file only has to *reserve* the space: every
tensor is zero-filled F32 with the shapes the loader validates.

FORK FACTS (each one decides code below)
----------------------------------------
``src/llama-adapter.cpp:202-204`` — ``general.type`` must be ``"adapter"``, else
``throw "expect general.type to be 'adapter'"``.

``src/llama-adapter.cpp:206-210`` — ``general.architecture`` is resolved with
``llm_arch_from_string`` and compared to ``model.arch``; mismatch throws ``"model arch and
LoRA arch mismatch"``. ``src/llama-arch.cpp:41`` maps ``LLM_ARCH_QWEN35 -> "qwen35"``, and
the base GGUF's own ``general.architecture`` reads back ``qwen35`` — so the value is read
from the base file (:func:`gguf_weights.general_architecture`), never hardcoded.

``src/llama-adapter.cpp:213-215`` — ``adapter.type`` must be ``"lora"``.

``src/llama-adapter.cpp:218`` — ``adapter.alpha = get_kv_f32(llm_kv(LLM_KV_ADAPTER_LORA_ALPHA))``,
i.e. ``adapter.lora.alpha`` (``src/llama-arch.cpp:394``). ``get_kv_f32`` returns ``0.0f`` for a
missing key, and ``src/llama-adapter.h:53-55`` then falls back to ``scale = adapter_scale``
(no ``alpha/rank``); ``ggml/src/gguf.cpp:194`` asserts ``type_to_gguf_type<T>::value == type``,
so the value **must be written as GGUF FLOAT32** — a uint32 alpha aborts the load, it does not
degrade. Written as ``1.0`` so the runtime scale is ``alpha/rank = 1/8``, which is exactly
``TttLoraAdapter.scaling`` (``src/hagi/config.py`` ``TttLoraConfig``: ``rank=8, alpha=1.0``);
a ``B`` solved by the Python contour therefore transfers into this adapter unscaled.

``lora.version`` — **does not exist in this fork.** ``rg`` over ``src/`` finds only
``adapter.lora.{alpha,task_name,prompt_prefix}`` and ``adapter.alora.invocation_tokens``
(``src/llama-arch.cpp:393-397``). Nothing reads a version key, so none is invented.

Rank is **not metadata either**: ``src/llama-adapter.h:54`` computes it as
``const float rank = (float) b->ne[0];``. The bundled ``gguf-py`` offers
``add_adapter_lora_rank`` (``%s.adapter.lora.rank``) and upstream converters use it, but no
line of this fork reads that key — writing it would be decorative.

``src/llama-adapter.cpp:273-292`` — pairing: a tensor whose name *ends* with ``.lora_a`` or
``.lora_b`` is bundled with its sibling under the name with the suffix stripped;
``_norm.weight`` is skipped ("TODO: add support for norm vector"); **anything else throws**
``LoRA tensor '<name>' has unexpected suffix``. Hence this file contains only ``.lora_a`` /
``.lora_b`` tensors.

``src/llama-adapter.cpp:330-332`` — the stripped name is looked up with
``model.get_tensor(name)`` (exact string compare, ``src/llama-model.cpp:2453-2463``), and a
miss throws ``does not exist in base model``. Hence base names come from the real tensor
table (``gguf_weights.tensor_names()``), not from a typed-out list.

``src/llama-adapter.cpp:362-367`` — shape validation for every non-embedding pair::

    if (model_tensor->ne[0] != w.a->ne[0] || model_tensor->ne[1] != w.b->ne[1])
        throw "tensor '<name>' has incorrect shape";
    if (w.a->ne[1] != w.b->ne[0])
        throw "lora_a tensor is not transposed";

With a base weight ``W`` of logical shape ``(out, in)`` — ``gguf_weights`` reports the
reversed file order, i.e. ``ne = (in, out)`` — that forces ``lora_a`` to ``ne = (in, rank)``
and ``lora_b`` to ``ne = (rank, out)``, i.e. **numpy shapes ``(rank, in)`` and ``(out,
rank)``**, since ``gguf/gguf_writer.py:268`` stores ``shape[n_dims-1-j]`` at position ``j``.
``src/llama-graph.cpp:1565-1567`` confirms the orientation by consuming them that way::

    ggml_tensor * ab_cur = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur));

``ggml/src/ggml.c:3295-3311`` then requires ``a->ne[0] == cur->ne[0]`` (``in``) and produces
``F32`` results, so the intermediate is ``rank`` wide.

dtype — ``src/llama-adapter.cpp`` never inspects a tensor type (it only ``ggml_dup_tensor``s
what ``gguf_init_from_file`` built), so nothing rejects an F16 pair at load. It blows up
later, at graph build, because ``build_lora_mm`` multiplies the pair against the F32
activation. **F32 is therefore required in practice, not by a check** — written as F32.

Disagreement with ``scripts/qwen_ttt_lora.py`` (the repo's other LoRA contour): that module
stores ``A`` as ``[Fin, r]`` and applies ``delta = scaling * B @ A.T`` with ``B:[Fout, r]``
(``scripts/qwen_ttt_lora.py:237-245``, mirrored by ``src/hagi/model/adapters.py:213-221``).
The GGUF ``lora_a`` is the **transpose** of that buffer: ``(rank, in)`` vs ``(in, rank)``.
``lora_b`` matches ``B`` exactly. Only the ``A`` axis order differs; ``scaling`` agrees by
construction (``alpha=1.0``, ``rank=8``).

Coverage
--------
Per ``models/ternary-bonsai-2-27b-mtp/donor/config.json`` ``text_config.layer_types``
(64 entries, 48 ``linear_attention`` / 16 ``full_attention``; ``blk.64`` is the MTP draft
head and is excluded — ``gguf_to_hf_name`` returns ``None`` for it):

* linear block (8 tensors): ``attn_qkv``, ``attn_gate``, ``ssm_out``, ``ssm_alpha``,
  ``ssm_beta``, ``ffn_gate``, ``ffn_up``, ``ffn_down`` — all eight reach the loader through
  ``build_lora_mm`` (``src/models/qwen35.cpp:262,266,387,396,537`` and ``build_ffn`` at
  ``:550-553``).
* full block (7 tensors): ``attn_q``, ``attn_k``, ``attn_v``, ``attn_output``, ``ffn_gate``,
  ``ffn_up``, ``ffn_down`` (``src/models/qwen35.cpp:295,307,310,358`` + ``build_ffn``).

Excluded: every 1-D tensor (``attn_norm``, ``post_attention_norm``, ``ssm_norm``,
``attn_q_norm``, ``attn_k_norm``, ``ssm_dt.bias``, ``ssm_a``) — a LoRA pair on a vector is
meaningless — and ``ssm_conv1d.weight``, which is 2-D but is read as a depthwise conv kernel
(``conv_kernel->ne[0]`` = kernel size, ``src/models/qwen35.cpp:435-437``) and never passed to
``build_lora_mm``, so a pair on it could never be applied.

Memory contract: 496 rank-8 pairs = 233,523,904 bytes (222.7 MiB) of F32 zeros, assembled one
tensor at a time (largest single tensor 557 KiB) and written streaming; the 27B state is never
materialised — ``tensor_names()`` reads only the 866-entry table, not the data section.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFType, GGUFValueType, GGUFWriter
from gguf.constants import GGMLQuantizationType, Keys

if __package__ in (None, ""):  # scripts/ is not a package; tests import it by path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_weights import GGUF_PATH, general_architecture, gguf_to_hf_name, tensor_names  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "models" / "ternary-bonsai-2-27b-mtp" / "donor" / "config.json"

RANK = 8
#: alpha=1.0 with rank=8 -> runtime scale 1/8 == TttLoraAdapter.scaling (see docstring).
ALPHA = 1.0
TASK_NAME = "hagi-ttt-bootstrap"

LINEAR_TAILS = ("attn_qkv", "attn_gate", "ssm_out", "ssm_alpha", "ssm_beta", "ffn_gate", "ffn_up", "ffn_down")
FULL_TAILS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
FAMILY_TAILS = {"linear_attention": LINEAR_TAILS, "full_attention": FULL_TAILS}

#: 2-D tensors that carry no LoRA pair, with the reason each is dropped.
UNADAPTABLE_2D = {"ssm_conv1d.weight": "depthwise conv kernel, never through build_lora_mm"}

WEIGHT_SUFFIX = ".weight"
LORA_A_SUFFIX = ".lora_a"
LORA_B_SUFFIX = ".lora_b"


def _block_index_and_tail(name: str) -> tuple[int, str]:
    parts = name.split(".", 2)
    if len(parts) != 3 or parts[0] != "blk":
        raise ValueError(f"not a block tensor name: {name!r}")
    return int(parts[1]), parts[2]


def layer_types(config_path: Path = CONFIG_PATH) -> list[str]:
    """The 64-entry per-block family list from the donor config."""
    types = json.loads(Path(config_path).read_text(encoding="utf-8"))["text_config"]["layer_types"]
    if len(types) != 64:
        raise ValueError(f"expected 64 layer_types, got {len(types)}")
    return types


def pair_shapes(base_shape: tuple[int, ...], rank: int = RANK) -> tuple[tuple[int, int], tuple[int, int]]:
    """``(out, in)`` -> ``(lora_a shape, lora_b shape)``, numpy order.

    Derived from ``llama-adapter.cpp:362-367``: ``a`` must open with the base's ``ne[0]``
    (``in``) and ``b`` must end with its ``ne[1]`` (``out``), with ``a->ne[1] == b->ne[0]``.
    """
    if len(base_shape) != 2:
        raise ValueError(f"base weight must be 2-D, got {base_shape}")
    out, in_ = base_shape
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if rank > min(out, in_):
        raise ValueError(f"rank {rank} exceeds base dims {(out, in_)}")
    return (rank, in_), (out, rank)


def adaptable_bases(
    gguf_path: Path = GGUF_PATH,
    config_path: Path = CONFIG_PATH,
) -> list[tuple[str, tuple[int, int]]]:
    """Stream the real GGUF tensor table and return ``[(base_name, (out, in)), ...]``.

    The list is *derived*, not typed out: names come from ``tensor_names()``, the family of
    each block from ``config.json``, and ``blk.64``/non-block tensors fall out of
    ``gguf_to_hf_name`` returning ``None``. The tail sets are the one hand-written input, and
    they are checked in both directions — a tail the fork adapts but this list forgot raises,
    and so does a 2-D tail nobody accounted for. That turns a topology change in the GGUF
    into a loud failure instead of a quietly smaller adapter.
    """
    types = layer_types(config_path)
    found: dict[str, tuple[int, int]] = {}
    seen_2d: dict[str, set[str]] = defaultdict(set)
    blocks_seen: dict[str, set[int]] = defaultdict(set)

    for name, shape, _dtype in tensor_names(Path(gguf_path)):
        if not name.startswith("blk."):
            # token_embd / output / output_norm. The task's coverage list is per-block, and
            # the embedding path is special-cased in the fork with a *flipped* pair
            # (llama-adapter.cpp:356-360: ``ne[0] != w.b->ne[1]``), so it needs its own
            # decision rather than a silent inclusion here.
            continue
        if gguf_to_hf_name(name) is None:
            continue  # blk.64, the MTP draft head
        il, tail = _block_index_and_tail(name)
        if len(shape) != 2:
            continue  # 1-D: norms, biases, ssm_dt.bias, ssm_a
        family = types[il]
        seen_2d[family].add(tail)
        if tail in {t + WEIGHT_SUFFIX for t in FAMILY_TAILS[family]}:
            found[name] = (int(shape[0]), int(shape[1]))
            blocks_seen[family].add(il)

    for family, expected in FAMILY_TAILS.items():
        want = {t + WEIGHT_SUFFIX for t in expected}
        missing = want - seen_2d[family]
        if missing:
            raise AssertionError(f"{family} blocks lack expected 2-D tails: {sorted(missing)}")
        unknown = seen_2d[family] - want - set(UNADAPTABLE_2D)
        if unknown:
            raise AssertionError(f"{family} blocks carry unaccounted 2-D tails: {sorted(unknown)}")
        n_blocks = types.count(family)
        if len(blocks_seen[family]) != n_blocks:
            raise AssertionError(
                f"{family}: expected {n_blocks} blocks, adapter covers {len(blocks_seen[family])}"
            )

    return sorted(found.items(), key=lambda kv: (_block_index_and_tail(kv[0])[0], kv[0]))


def write_adapter(
    out: Path,
    rank: int = RANK,
    alpha: float = ALPHA,
    bases: list[tuple[str, tuple[int, int]]] | None = None,
    gguf_path: Path = GGUF_PATH,
) -> dict:
    """Write the zero-filled adapter GGUF; returns a summary dict."""
    bases = adaptable_bases(gguf_path) if bases is None else bases
    arch = general_architecture(Path(gguf_path))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = GGUFWriter(str(out), arch)
    writer.add_type(GGUFType.ADAPTER)  # general.type — llama-adapter.cpp:202-204
    writer.add_string(Keys.Adapter.TYPE, "lora")  # llama-adapter.cpp:213-215
    writer.add_float32(Keys.Adapter.LORA_ALPHA, float(alpha))  # llama-adapter.cpp:218, FLOAT32 required
    writer.add_string(Keys.Adapter.LORA_TASK_NAME, TASK_NAME)  # read for logging: common/common.cpp:1353

    n_bytes = 0
    for name, base in bases:
        a_shape, b_shape = pair_shapes(base, rank)
        for suffix, shape in ((LORA_A_SUFFIX, a_shape), (LORA_B_SUFFIX, b_shape)):
            data = np.zeros(shape, dtype=np.float32)
            n_bytes += data.nbytes
            writer.add_tensor(name + suffix, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    # write_tensors_to_file emits the tensor-info block itself (gguf_writer.py:437 ->
    # write_ti_data_to_file); calling write_ti_data_to_file here would double-advance the
    # writer state and raise on the second call.
    writer.write_tensors_to_file()
    writer.flush()
    writer.close()

    return {
        "path": str(out),
        "arch": arch,
        "rank": rank,
        "alpha": float(alpha),
        "pairs": len(bases),
        "tensors": len(bases) * 2,
        "payload_bytes": n_bytes,
        "file_bytes": out.stat().st_size,
    }


def verify_adapter(
    path: Path,
    rank: int = RANK,
    bases: list[tuple[str, tuple[int, int]]] | None = None,
    gguf_path: Path = GGUF_PATH,
) -> dict:
    """Reopen the written GGUF and re-run the fork's own acceptance rules on it.

    Mirrors ``src/llama-adapter.cpp``: the suffix rule (``:281-299``), the pair completeness
    check (``:315-317``), the shape validation (``:345-351``) and the value types the ggml
    getters assert (``ggml/src/gguf.cpp:194``). Returns counts; raises on any disagreement.
    """
    bases = adaptable_bases(gguf_path) if bases is None else bases
    expected = dict(bases)
    reader = GGUFReader(str(path))

    # --- metadata: value *types* as well as values, because gguf.cpp:194 asserts the type.
    for key, want_type, want in (
        ("general.type", GGUFValueType.STRING, GGUFType.ADAPTER),
        (Keys.Adapter.TYPE, GGUFValueType.STRING, "lora"),
        (Keys.Adapter.LORA_ALPHA, GGUFValueType.FLOAT32, None),
    ):
        field = reader.fields.get(key)
        assert field is not None, f"adapter is missing required key {key!r}"
        assert field.types[0] == want_type, f"{key} must be {want_type}, got {field.types[0]}"
        if want is not None:
            got = bytes(field.parts[-1].tobytes()).decode()
            assert got == want, f"{key} must be {want!r}, got {got!r}"
    arch = bytes(reader.fields["general.architecture"].parts[-1].tobytes()).decode()
    assert arch == general_architecture(Path(gguf_path)), f"general.architecture {arch!r} != base"

    # --- tensors: the fork's suffix rule, and nothing else in the file.
    pairs: dict[str, dict[str, object]] = {}
    n_tensors = 0
    for tensor in reader.tensors:
        n_tensors += 1
        name = str(tensor.name)
        assert tensor.tensor_type == GGMLQuantizationType.F32, (
            f"{name} is not F32 (ggml type {tensor.tensor_type})"
        )
        for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX):
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                slot = pairs.setdefault(base, {})
                assert suffix not in slot, f"duplicate {suffix} for {base}"
                slot[suffix] = tensor
                break
        else:
            # llama-adapter.cpp:287-292 skips *_norm.weight and throws on anything else.
            assert name.endswith("_norm.weight"), f"LoRA tensor {name!r} has unexpected suffix"

    # --- pair count and completeness (llama-adapter.cpp:325-326).
    assert len(pairs) == len(expected), f"pair count {len(pairs)} != expected {len(expected)}"
    assert set(pairs) == set(expected), (
        f"base-name mismatch; missing={sorted(set(expected) - set(pairs))[:5]} "
        f"extra={sorted(set(pairs) - set(expected))[:5]}"
    )

    # --- shapes compose, and match the base tensor (llama-adapter.cpp:362-367).
    zero_ok = True
    for base, slot in pairs.items():
        a = slot.get(LORA_A_SUFFIX)
        b = slot.get(LORA_B_SUFFIX)
        assert a is not None and b is not None, f"LoRA pair for {base!r} is missing one component"
        out, in_ = expected[base]
        a_shape, b_shape = pair_shapes((out, in_), rank)
        assert tuple(a.data.shape) == a_shape, f"{base}.lora_a is {tuple(a.data.shape)}, want {a_shape}"
        assert tuple(b.data.shape) == b_shape, f"{base}.lora_b is {tuple(b.data.shape)}, want {b_shape}"
        # The fork's literal test, in stored (ne) order — ``tensor.shape`` is ``[ne[0], ne[1]]``:
        #   model->ne[0] == a->ne[0] && model->ne[1] == b->ne[1]   (llama-adapter.cpp:362)
        #   a->ne[1] == b->ne[0] == rank                            (llama-adapter.cpp:366)
        a_ne = [int(v) for v in a.shape]
        b_ne = [int(v) for v in b.shape]
        assert a_ne[0] == in_ and b_ne[1] == out, f"{base}: pair does not match base ({(in_, out)})"
        assert a_ne[1] == b_ne[0] == rank, f"{base}: rank does not compose (a ne={a_ne}, b ne={b_ne})"
        zero_ok = zero_ok and not a.data.any() and not b.data.any()

    assert zero_ok, "bootstrap adapter must be zero-filled (the C++ slice owns the lora_a draw)"

    per_family = defaultdict(int)
    for base in pairs:
        per_family[layer_types()[_block_index_and_tail(base)[0]]] += 1

    summary = {
        "path": str(path),
        "pairs": len(pairs),
        "tensors": n_tensors,
        "rank": rank,
        "linear_pairs": per_family["linear_attention"],
        "full_pairs": per_family["full_attention"],
        "payload_bytes": sum(t.n_bytes for t in reader.tensors),
        "file_bytes": Path(path).stat().st_size,
        "zero_filled": zero_ok,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, help="path of the adapter GGUF to write")
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--base", type=Path, default=GGUF_PATH, help="base model GGUF (names + arch source)")
    ap.add_argument("--list", action="store_true", help="print the derived base list and exit")
    args = ap.parse_args(argv)

    bases = adaptable_bases(args.base)
    if args.list:
        for name, shape in bases:
            a, b = pair_shapes(shape, args.rank)
            print(f"  {name:28s} base={shape} lora_a={a} lora_b={b}")
        print(f"{len(bases)} pairs")
        return 0
    if args.out is None:
        ap.error("--out is required unless --list is given")

    summary = write_adapter(args.out, rank=args.rank, alpha=args.alpha, bases=bases, gguf_path=args.base)
    print(
        f"wrote {summary['path']}: {summary['pairs']} pairs "
        f"({summary['tensors']} tensors), arch={summary['arch']}, "
        f"rank={summary['rank']}, alpha={summary['alpha']}, "
        f"payload={summary['payload_bytes'] / 2**20:.1f} MiB, "
        f"file={summary['file_bytes'] / 2**20:.1f} MiB"
    )
    got = verify_adapter(args.out, rank=args.rank, bases=bases, gguf_path=args.base)
    print(
        f"verified: {got['pairs']} pairs, {got['tensors']} tensors "
        f"(linear {got['linear_pairs']}, full {got['full_pairs']}), "
        f"rank composes on every pair, zero_filled={got['zero_filled']}, "
        f"file={got['file_bytes'] / 2**20:.1f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
