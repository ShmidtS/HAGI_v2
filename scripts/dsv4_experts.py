"""DeepSeek-V4 expert decode / split / F3-merge utilities.

Exact formats (verified against the official ``inference/convert.py``):

Routed experts (expert_dtype == "fp4"):
    w1.weight : int8  [inter, dim//2]   -> 2 fp4 (e2m1fn) per byte, logical [inter, dim]
    w1.scale  : float8_e8m0fnu [inter, dim//32]  (1 scale per 32 elements along dim)
    w2.weight : int8  [dim, inter//2]  -> logical [dim, inter]
    w2.scale  : float8_e8m0fnu [dim, inter//32]
    w3.weight : int8  [inter, dim//2]
    w3.scale  : float8_e8m0fnu [inter, dim//32]

Shared experts (fp8 e4m3fn, block 128):
    w.weight   : float8_e4m3fn [out, in]
    w.scale    : float8_e8m0fnu [out//128, in//128]

Gate (router): bfloat16 [n_routed_experts, dim] (+ optional bias, tid2eid).

Bit order (packed float4_e2m1fn_x2): low nibble = even index, high = odd.
"""

from __future__ import annotations

import json
import os
import struct

import torch

# e2m1fn decode table (sign-magnitude, 1s 2e 1m, bias 1). Matches convert.py.
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

FP4_BLOCK = 32
FP8_BLOCK = 128

# DeepSeek-V4-Flash-0731 geometry.
DIM = 4096
INTER = 2048
N_ROUTED = 256
N_SHARED = 1


def dequant_fp4(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode packed fp4 (e2m1fn) weights into float32 (GPU-fast).

    ``w`` is int8 ``[out, in//2]``; ``scale`` is float8_e8m0fnu ``[out, in//32]``.
    Returns float32 ``[out, in]``."""
    table = FP4_TABLE.to(w.device)
    u = w.to(torch.uint8)
    low = table[(u & 0x0F).long()]          # [out, in//2]
    high = table[((u >> 4) & 0x0F).long()]  # [out, in//2]
    out, in2 = u.shape
    v = torch.empty((out, in2 * 2), dtype=torch.float32, device=w.device)
    v[:, 0::2] = low
    v[:, 1::2] = high
    s = scale.to(torch.float32).repeat_interleave(FP4_BLOCK, dim=1)  # [out, in]
    return v * s


def dequant_fp8(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode blockwise fp8 (e4m3fn) weights into float32.

    ``w`` is float8_e4m3fn ``[out, in]``; ``scale`` is float8_e8m0fnu
    ``[out//128, in//128]``. Returns float32 ``[out, in]``.
    """
    v = w.to(torch.float32)
    s = scale.to(torch.float32)
    s = s.repeat_interleave(FP8_BLOCK, dim=0).repeat_interleave(FP8_BLOCK, dim=1)
    return v * s


def _decode(name: str, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dispatch on dtype: int8 -> fp4 path, float8_e4m3fn -> fp8 path."""
    if w.dtype == torch.int8:
        return dequant_fp4(w, scale)
    if w.dtype == torch.float8_e4m3fn:
        return dequant_fp8(w, scale)
    # bf16/fp32 pass-through (should not happen for experts, but be safe).
    return w.to(torch.float32)


def shard_for(name: str, weight_map: dict) -> str:
    return weight_map[name]


def read_tensor(shard_dir: str, weight_map: dict, name: str, device: str = "cpu") -> torch.Tensor:
    """Read a single tensor by name from the correct shard (streaming, no full load).

    ``.clone()`` detaches the tensor from the safetensors mmap so the tensor
    stays valid after the file handle closes (required when accumulating many
    tensors in memory)."""
    from safetensors import safe_open

    shard = os.path.join(shard_dir, weight_map[name])
    with safe_open(shard, framework="pt", device=device) as f:
        return f.get_tensor(name).clone()


def _expert_keys(layer_prefix: str, expert_idx: int) -> list[str]:
    p = f"{layer_prefix}.experts.{expert_idx}"
    return [f"{p}.w1.weight", f"{p}.w1.scale",
            f"{p}.w2.weight", f"{p}.w2.scale",
            f"{p}.w3.weight", f"{p}.w3.scale"]


def load_expert(shard_dir: str, weight_map: dict, layer_prefix: str, expert_idx: int) -> dict[str, torch.Tensor]:
    """Load + decode one routed expert into float32.

    Returns ``{"w1": [inter, dim], "w2": [dim, inter], "w3": [inter, dim]}``.
    """
    names = _expert_keys(layer_prefix, expert_idx)
    tensors = {n: read_tensor(shard_dir, weight_map, n) for n in names}
    base = f"{layer_prefix}.experts.{expert_idx}"
    return {
        "w1": _decode(base, tensors[f"{base}.w1.weight"], tensors[f"{base}.w1.scale"]),
        "w2": _decode(base, tensors[f"{base}.w2.weight"], tensors[f"{base}.w2.scale"]),
        "w3": _decode(base, tensors[f"{base}.w3.weight"], tensors[f"{base}.w3.scale"]),
    }


def load_shared(shard_dir: str, weight_map: dict, layer_prefix: str) -> dict[str, torch.Tensor]:
    """Load + decode the shared expert into float32 (same key layout)."""
    base = f"{layer_prefix}.shared_experts"
    out = {}
    for proj in ("w1", "w2", "w3"):
        w = read_tensor(shard_dir, weight_map, f"{base}.{proj}.weight")
        s = read_tensor(shard_dir, weight_map, f"{base}.{proj}.scale")
        out[proj] = _decode(base, w, s)
    return out


def load_gate(shard_dir: str, weight_map: dict, layer_prefix: str) -> dict[str, torch.Tensor]:
    """Load the router gate (bfloat16)."""
    base = f"{layer_prefix}.gate"
    out = {"weight": read_tensor(shard_dir, weight_map, f"{base}.weight").to(torch.float32)}
    if f"{base}.bias" in weight_map:
        out["bias"] = read_tensor(shard_dir, weight_map, f"{base}.bias").to(torch.float32)
    return out


def read_tensor_from_file(file_path: str, name: str, device: str = "cpu") -> torch.Tensor:
    """Read one tensor by name from a single lossless per-layer safetensors file."""
    from safetensors import safe_open

    with safe_open(file_path, framework="pt", device=device) as f:
        return f.get_tensor(name)


def load_expert_file(file_path: str, layer_prefix: str, expert_idx: int, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load + decode one routed expert from a lossless per-layer file into float32."""
    base = f"{layer_prefix}.experts.{expert_idx}"
    names = _expert_keys(layer_prefix, expert_idx)
    tensors = {n: read_tensor_from_file(file_path, n, device=device) for n in names}
    return {
        "w1": _decode(base, tensors[f"{base}.w1.weight"], tensors[f"{base}.w1.scale"]),
        "w2": _decode(base, tensors[f"{base}.w2.weight"], tensors[f"{base}.w2.scale"]),
        "w3": _decode(base, tensors[f"{base}.w3.weight"], tensors[f"{base}.w3.scale"]),
    }


def load_shared_file(file_path: str, layer_prefix: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load + decode the shared expert from a lossless per-layer file into float32."""
    base = f"{layer_prefix}.shared_experts"
    out = {}
    for proj in ("w1", "w2", "w3"):
        w = read_tensor_from_file(file_path, f"{base}.{proj}.weight", device=device)
        s = read_tensor_from_file(file_path, f"{base}.{proj}.scale", device=device)
        out[proj] = _decode(base, w, s)
    return out


def iter_layer_prefixes(weight_map: dict) -> list[str]:
    """All ffn prefixes that carry experts (layers.* and mtp.*)."""
    prefixes = set()
    for name in weight_map:
        if ".ffn." not in name:
            continue
        # keep up to the .ffn part: e.g. "layers.3.ffn" or "mtp.0.ffn"
        prefixes.add(name.split(".ffn.")[0] + ".ffn")
    return sorted(prefixes)


def iter_expert_ids(weight_map: dict, layer_prefix: str) -> list[int]:
    """Sorted routed expert ids present for a layer."""
    ids = set()
    for name in weight_map:
        if name.startswith(f"{layer_prefix}.experts."):
            rest = name[len(f"{layer_prefix}.experts."):]
            ids.add(int(rest.split(".")[0]))
    return sorted(ids)


def load_index(snapshot_dir: str) -> dict:
    with open(os.path.join(snapshot_dir, "model.safetensors.index.json")) as fh:
        return json.load(fh)


def default_snapshot() -> str:
    cache = os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots")
    snap = sorted(os.listdir(cache))[0]
    return os.path.join(cache, snap)
