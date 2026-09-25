#!/usr/bin/env python3
"""E0: read a frozen PQ2_0 GGUF and verify its MTP tensor contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import gguf
import numpy as np
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES

PQ2_0_TYPE = 142

MTP_EXPECTED_DTYPE = {
    "mtp.fc.weight": GGMLQuantizationType.Q8_0,
    "mtp.norm.weight": GGMLQuantizationType.F32,
    "mtp.pre_fc_norm_embedding.weight": GGMLQuantizationType.F32,
    "mtp.pre_fc_norm_hidden.weight": GGMLQuantizationType.F32,
    "mtp.layers.0.input_layernorm.weight": GGMLQuantizationType.F32,
    "mtp.layers.0.post_attention_layernorm.weight": GGMLQuantizationType.F32,
    "mtp.layers.0.self_attn.q_norm.weight": GGMLQuantizationType.F32,
    "mtp.layers.0.self_attn.k_norm.weight": GGMLQuantizationType.F32,
    "mtp.layers.0.mlp.gate_proj.weight": GGMLQuantizationType.Q8_0,
    "mtp.layers.0.mlp.up_proj.weight": GGMLQuantizationType.Q8_0,
    "mtp.layers.0.mlp.down_proj.weight": GGMLQuantizationType.Q8_0,
    "mtp.layers.0.self_attn.q_proj.weight": GGMLQuantizationType.Q8_0,
    "mtp.layers.0.self_attn.k_proj.weight": GGMLQuantizationType.Q8_0,
    "mtp.layers.0.self_attn.v_proj.weight": GGMLQuantizationType.Q8_0,
    "mtp.layers.0.self_attn.o_proj.weight": GGMLQuantizationType.Q8_0,
}

MTP_TO_GGUF = {
    "mtp.fc.weight": "blk.64.nextn.eh_proj.weight",
    "mtp.norm.weight": "blk.64.nextn.shared_head_norm.weight",
    "mtp.pre_fc_norm_embedding.weight": "blk.64.nextn.enorm.weight",
    "mtp.pre_fc_norm_hidden.weight": "blk.64.nextn.hnorm.weight",
    "mtp.layers.0.input_layernorm.weight": "blk.64.attn_norm.weight",
    "mtp.layers.0.post_attention_layernorm.weight": "blk.64.post_attention_norm.weight",
    "mtp.layers.0.self_attn.q_norm.weight": "blk.64.attn_q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight": "blk.64.attn_k_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight": "blk.64.attn_q.weight",
    "mtp.layers.0.self_attn.k_proj.weight": "blk.64.attn_k.weight",
    "mtp.layers.0.self_attn.v_proj.weight": "blk.64.attn_v.weight",
    "mtp.layers.0.self_attn.o_proj.weight": "blk.64.attn_output.weight",
    "mtp.layers.0.mlp.gate_proj.weight": "blk.64.ffn_gate.weight",
    "mtp.layers.0.mlp.up_proj.weight": "blk.64.ffn_up.weight",
    "mtp.layers.0.mlp.down_proj.weight": "blk.64.ffn_down.weight",
}

MTP_EFFECTIVE_MULTIPLIERS = {
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
}

MTP_TENSOR_NAMES = tuple(MTP_EXPECTED_DTYPE)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest_entry(manifest: dict, suffix: str) -> dict:
    matches = [item for item in manifest.get("files", []) if item.get("path", "").endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest entry ending in {suffix!r}, found {len(matches)}")
    return matches[0]


class PQ20TolerantReader(gguf.GGUFReader):
    """Official GGUF parser with count-only handling for custom type 142."""

    def __init__(self, path: Path):
        self.custom_tensors: list[dict] = []
        super().__init__(path, mode="r")

    def _build_tensors(self, start_offs: int, fields) -> None:
        for field in fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
            stored_shape = [int(value) for value in dims.tolist()]
            logical_shape = list(reversed(stored_shape))
            entry = {
                "name": str(bytes(name_data), encoding="utf-8"),
                "stored_shape": stored_shape,
                "shape": logical_shape,
                "dtype": int(raw_dtype[0]),
                "offset": int(offset_tensor[0]),
            }
            self.custom_tensors.append(entry)
            if entry["dtype"] == int(GGMLQuantizationType.Q8_0):
                self._decode_q80(entry, start_offs)
            elif entry["dtype"] == int(GGMLQuantizationType.F32):
                self._decode_f32(entry, start_offs)

    def _decode_q80(self, entry: dict, start_offs: int) -> None:
        shape = tuple(entry["shape"])
        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q8_0]
        if shape[-1] % block_size:
            raise ValueError(f"Q8_0 row size is not divisible by {block_size}: {entry['name']}")
        byte_shape = shape[:-1] + (shape[-1] // block_size * type_size,)
        n_bytes = math.prod(byte_shape)
        data_offset = start_offs + entry["offset"]
        raw = np.frombuffer(
            self.data, dtype=np.uint8, count=n_bytes, offset=data_offset
        )
        entry["data"] = gguf.dequantize(
            raw.reshape(byte_shape), GGMLQuantizationType.Q8_0
        )
        entry["n_bytes"] = n_bytes

    def _decode_f32(self, entry: dict, start_offs: int) -> None:
        shape = tuple(entry["shape"])
        n_bytes = math.prod(shape) * np.dtype(np.float32).itemsize
        data_offset = start_offs + entry["offset"]
        raw = np.frombuffer(
            self.data,
            dtype=np.dtype(np.float32).newbyteorder("<"),
            count=math.prod(shape),
            offset=data_offset,
        )
        entry["data"] = raw.reshape(shape)
        entry["n_bytes"] = n_bytes


def compare_tensor(
    source_name: str,
    tensor: dict,
    reference: dict,
    expected_shape: tuple[int, ...],
    atol: float,
    rtol: float,
) -> dict:
    expected_dtype = int(MTP_EXPECTED_DTYPE[source_name])
    actual_dtype = int(tensor["dtype"])
    decoded = tensor["data"]
    target = reference[source_name].float().numpy()
    if source_name in MTP_EFFECTIVE_MULTIPLIERS:
        target = target + 1.0

    dtype_ok = actual_dtype == expected_dtype
    shape_ok = tuple(decoded.shape) == expected_shape
    stored_shape_ok = tuple(tensor["stored_shape"]) == tuple(reversed(expected_shape))
    finite = bool(np.isfinite(decoded).all())
    diff = np.abs(decoded - target)
    relative = diff / np.maximum(np.abs(target), 1e-6)
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float(relative.max()) if relative.size else 0.0
    parity_ok = bool(np.all(diff <= atol + rtol * np.abs(target)))
    return {
        "source_tensor": source_name,
        "gguf_tensor": tensor["name"],
        "expected_dtype": expected_dtype,
        "actual_dtype": actual_dtype,
        "expected_shape": list(expected_shape),
        "gguf_stored_shape": tensor["stored_shape"],
        "decoded_shape": list(decoded.shape),
        "dtype_ok": dtype_ok,
        "shape_ok": shape_ok,
        "stored_shape_ok": stored_shape_ok,
        "finite": finite,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "tolerance": {"atol": atol, "rtol": rtol},
        "ok": dtype_ok and shape_ok and stored_shape_ok and finite and parity_ok,
    }


def run_smoke(
    gguf_path: Path,
    manifest_path: Path,
    config_path: Path,
    safetensors_path: Path,
    atol: float,
    rtol: float,
    sha256_override: str | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_shapes = {name: tuple(shape) for name, shape in config["tensor_shapes"].items()}

    gguf_info = manifest_entry(manifest, ".gguf")
    reference_info = manifest_entry(manifest, ".safetensors")
    config_info = manifest_entry(manifest, "mtp_config.json")
    expected_sha256 = sha256_override or gguf_info["sha256"]
    actual_sha256 = sha256_file(gguf_path)
    reference_sha256 = sha256_file(safetensors_path)
    config_sha256 = sha256_file(config_path)

    reader = PQ20TolerantReader(gguf_path)
    by_name = {tensor["name"]: tensor for tensor in reader.custom_tensors}
    counts = Counter(tensor["dtype"] for tensor in reader.custom_tensors)
    duplicate_names = sorted(
        name
        for name, count in Counter(tensor["name"] for tensor in reader.custom_tensors).items()
        if count > 1
    )

    mapped_tensors = {
        source_name: by_name.get(gguf_name)
        for source_name, gguf_name in MTP_TO_GGUF.items()
    }
    present = all(tensor is not None for tensor in mapped_tensors.values())
    dtype_ok = present and all(
        tensor is not None
        and int(tensor["dtype"]) == int(MTP_EXPECTED_DTYPE[name])
        for name, tensor in mapped_tensors.items()
    )

    import safetensors.torch as safetensors

    reference = safetensors.load_file(str(safetensors_path), device="cpu")
    parity_results = []
    if present:
        for source_name, tensor in mapped_tensors.items():
            if tensor is None:
                continue
            expected_shape = config_shapes[source_name]
            reference_shape = tuple(reference[source_name].shape)
            if reference_shape != expected_shape:
                raise ValueError(
                    f"reference/config shape mismatch for {source_name}: "
                    f"{reference_shape} != {expected_shape}"
                )
            parity_results.append(
                compare_tensor(
                    source_name,
                    tensor,
                    reference,
                    expected_shape,
                    atol,
                    rtol,
                )
            )

    payload_bounds_ok = all(
        tensor is not None
        and int(tensor["offset"]) >= 0
        and int(tensor["offset"]) + int(tensor.get("n_bytes", 0)) <= len(reader.data)
        for tensor in mapped_tensors.values()
    )
    finite_payloads = bool(parity_results) and all(
        result["finite"] for result in parity_results
    )
    parity_ok = len(parity_results) == len(MTP_TENSOR_NAMES) and all(
        result["ok"] for result in parity_results
    )
    shape_ok = len(parity_results) == len(MTP_TENSOR_NAMES) and all(
        result["shape_ok"] and result["stored_shape_ok"] for result in parity_results
    )
    config_reference_shapes_ok = all(
        tuple(reference[name].shape) == config_shapes[name]
        for name in MTP_TENSOR_NAMES
    )
    artifact_checks_ok = (
        actual_sha256 == expected_sha256
        and gguf_path.stat().st_size == int(gguf_info["bytes"])
        and reference_sha256 == reference_info["sha256"]
        and safetensors_path.stat().st_size == int(reference_info["bytes"])
        and config_sha256 == config_info["sha256"]
        and config_path.stat().st_size == int(config_info["bytes"])
    )
    no_duplicate_tensors = not duplicate_names
    gate_ok = all(
        (
            artifact_checks_ok,
            no_duplicate_tensors,
            present,
            dtype_ok,
            shape_ok,
            config_reference_shapes_ok,
            payload_bounds_ok,
            finite_payloads,
            parity_ok,
        )
    )

    return {
        "gate": "PASS" if gate_ok else "FAIL",
        "artifact_checks_ok": artifact_checks_ok,
        "artifacts": {
            "gguf": {
                "path": str(gguf_path),
                "bytes": gguf_path.stat().st_size,
                "expected_bytes": int(gguf_info["bytes"]),
                "sha256": actual_sha256,
                "expected_sha256": expected_sha256,
            },
            "safetensors": {
                "path": str(safetensors_path),
                "bytes": safetensors_path.stat().st_size,
                "expected_bytes": int(reference_info["bytes"]),
                "sha256": reference_sha256,
                "expected_sha256": reference_info["sha256"],
            },
            "mtp_config": {
                "path": str(config_path),
                "bytes": config_path.stat().st_size,
                "expected_bytes": int(config_info["bytes"]),
                "sha256": config_sha256,
                "expected_sha256": config_info["sha256"],
            },
        },
        "gguf": {
            "version": int(reader.fields["GGUF.version"].contents()),
            "tensor_count_field": int(reader.fields["GGUF.tensor_count"].contents()),
            "kv_count_field": int(reader.fields["GGUF.kv_count"].contents()),
            "tensor_info_count": len(reader.custom_tensors),
            "data_offset": int(reader.data_offset),
            "dtype_counts": {str(key): value for key, value in sorted(counts.items())},
            "duplicate_tensor_names": duplicate_names,
            "no_duplicate_tensors": no_duplicate_tensors,
        },
        "base": {
            "pq2_0_type": PQ2_0_TYPE,
            "pq2_0_tensor_count": int(counts[PQ2_0_TYPE]),
            "payload_policy": "count_only_no_decode_no_write",
            "payloads_written": False,
        },
        "mtp": {
            "expected_count": len(MTP_TENSOR_NAMES),
            "present_count": sum(tensor is not None for tensor in mapped_tensors.values()),
            "all_present": present,
            "q8_0_count": sum(
                tensor is not None
                and int(tensor["dtype"]) == int(GGMLQuantizationType.Q8_0)
                for tensor in mapped_tensors.values()
            ),
            "f32_count": sum(
                tensor is not None
                and int(tensor["dtype"]) == int(GGMLQuantizationType.F32)
                for tensor in mapped_tensors.values()
            ),
            "dtype_contract_ok": dtype_ok,
            "shape_contract_ok": shape_ok,
            "config_reference_shapes_ok": config_reference_shapes_ok,
            "payload_bounds_ok": payload_bounds_ok,
            "payloads_finite": finite_payloads,
            "parity_ok": parity_ok,
            "results": parity_results,
        },
        "tolerances": {"atol": atol, "rtol": rtol},
    }


def write_report(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gguf",
        type=Path,
        default=Path("models/ternary-bonsai-2-27b-mtp/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("models/ternary-bonsai-2-27b-mtp/manifest.json"),
    )
    parser.add_argument("--sha256", help="override the GGUF SHA256 expected by the manifest")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("models/ternary-bonsai-2-27b-mtp/mtp_config.json"),
    )
    parser.add_argument(
        "--safetensors",
        type=Path,
        default=Path("models/ternary-bonsai-2-27b-mtp/model_mtp.safetensors"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/e0_gguf_mtp_parity.json"))
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="compatibility flag; E0 always performs the MTP parity smoke",
    )
    args = parser.parse_args()

    required = (args.gguf, args.manifest, args.config, args.safetensors)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(f"missing required file(s): {', '.join(missing)}")
        return 2

    result = run_smoke(
        args.gguf,
        args.manifest,
        args.config,
        args.safetensors,
        args.atol,
        args.rtol,
        args.sha256,
    )
    write_report(args.output, result)
    print(
        json.dumps(
            {
                "gate": result["gate"],
                "sha256_ok": result["artifact_checks_ok"],
                "tensors": result["gguf"]["tensor_info_count"],
                "kv": result["gguf"]["kv_count_field"],
                "pq2_0_tensors": result["base"]["pq2_0_tensor_count"],
                "mtp_present": result["mtp"]["present_count"],
                "mtp_q8_0": result["mtp"]["q8_0_count"],
                "mtp_f32": result["mtp"]["f32_count"],
                "parity_ok": result["mtp"]["parity_ok"],
                "report": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if result["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
