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
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

# e2m1fn decode table (sign-magnitude, 1s 2e 1m, bias 1). Matches convert.py.
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
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
    low = table[(u & 0x0F).long()]  # [out, in//2]
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
    return [f"{p}.w1.weight", f"{p}.w1.scale", f"{p}.w2.weight", f"{p}.w2.scale", f"{p}.w3.weight", f"{p}.w3.scale"]


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


def load_expert_file(
    file_path: str, layer_prefix: str, expert_idx: int, device: str = "cpu"
) -> dict[str, torch.Tensor]:
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
            rest = name[len(f"{layer_prefix}.experts.") :]
            try:
                ids.add(int(rest.split(".")[0]))
            except ValueError:
                continue
    return sorted(ids)


def load_index(snapshot_dir: str) -> dict:
    index_path = Path(snapshot_dir, "model.safetensors.index.json").resolve()
    try:
        return json.loads(index_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing model index: {index_path}") from exc


def default_snapshot() -> str:
    cache = os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots")
    try:
        snaps = sorted(os.listdir(cache))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Snapshot cache not found: {cache}") from exc
    if not snaps:
        raise FileNotFoundError(f"No snapshots under: {cache}")
    return os.path.join(cache, snaps[0])


# --- Router / MoE geometry + forward helpers (shared across collect/refit) ---

N_LAYERS = 43
HASH_LAYERS = {0, 1, 2}
TOP_K = 6
ROUTED_SCALE = 1.5
SWIGLU_LIMIT = 10.0
LOSSLESS = "C:/HAGI_v2/lossless_layers"

# Per-layer router weights/biases, populated by ``load_router``.
ROUTER_W: dict[int, torch.Tensor] = {}
ROUTER_BIAS: dict[int, torch.Tensor] = {}
ROUTER_TID: dict[int, torch.Tensor] = {}


def load_router(snap: str, wm: dict) -> None:
    """Load per-layer router gate weights (and bias / tid2eid for hash layers)."""
    for li in range(N_LAYERS):
        p = f"layers.{li}.ffn.gate"
        ROUTER_W[li] = read_tensor(snap, wm, f"{p}.weight", device="cuda").to(torch.float32)
        if li in HASH_LAYERS:
            ROUTER_TID[li] = read_tensor(snap, wm, f"{p}.tid2eid", device="cuda").to(torch.long)
        else:
            ROUTER_BIAS[li] = read_tensor(snap, wm, f"{p}.bias", device="cuda").to(torch.float32)


def dequant_fp4_batch(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Batched fp4 decode: w [B, out, in//2] int8, scale [B, out, in//32] -> [B, out, in] fp32."""
    table = FP4_TABLE.to(w.device)
    u = w.to(torch.uint8)
    low = table[(u & 0x0F).long()]
    high = table[((u >> 4) & 0x0F).long()]
    b, out, in2 = u.shape
    v = torch.empty((b, out, in2 * 2), dtype=torch.float32, device=w.device)
    v[:, :, 0::2] = low
    v[:, :, 1::2] = high
    s = scale.to(torch.float32).repeat_interleave(FP4_BLOCK, dim=2)
    return v * s


def load_selected_experts(li: int, ids: list[int]) -> dict[int, tuple]:
    """Load + batched-dequant ONLY the routed experts (ids) -> dict k:(w1,w2,w3)."""
    fp = os.path.join(LOSSLESS, f"layers_{li}_ffn.safetensors")
    ids = sorted(ids)
    experts: dict[int, dict] = {}
    with safe_open(fp, framework="pt", device="cpu") as f:
        for proj in ("w1", "w2", "w3"):
            w_list = [f.get_tensor(f"layers.{li}.ffn.experts.{k}.{proj}.weight") for k in ids]
            s_list = [f.get_tensor(f"layers.{li}.ffn.experts.{k}.{proj}.scale") for k in ids]
            w_stack = torch.stack(w_list).to("cuda")
            s_stack = torch.stack(s_list).to("cuda")
            decoded = dequant_fp4_batch(w_stack, s_stack)
            for i, k in enumerate(ids):
                experts.setdefault(k, {})[proj] = decoded[i]
    return {k: (v["w1"], v["w2"], v["w3"]) for k, v in experts.items()}


def ffn(xin: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """One expert's SwiGLU forward: (silu(x@w1^T) * (x@w3^T)) @ w2^T."""
    g = (xin @ w1.T).clamp(max=SWIGLU_LIMIT)
    u = (xin @ w3.T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return (F.silu(g) * u) @ w2.T


# --- Ternary quantization (pack/unpack) ---


def ternarize(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched ternary quantize -> (q in {-1,0,1}, scale)."""
    s = W.abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
    q = (W / s).clamp(-1, 1).round()
    return q, s.squeeze(2)


def pack_ternary(q: torch.Tensor) -> torch.Tensor:
    """Pack ternary int8 {-1,0,1} [out, in] -> uint8 [out, ceil(in/5)] (5 trits/byte, base-3)."""
    t = (q + 1).to(torch.int32)  # {0,1,2}
    out, in_ = t.shape
    n = (in_ + 4) // 5
    padded = torch.zeros(out, n * 5, dtype=torch.int32)
    padded[:, :in_] = t
    packed = torch.zeros(out, n, dtype=torch.int32)
    for i in range(5):
        packed += padded[:, i::5] * (3**i)
    return packed.to(torch.uint8)


def unpack_ternary(q: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 [out, ceil(in/5)] -> int8 [out, in] ternary {-1,0,1}."""
    t = q.to(torch.int32)
    out, n = t.shape
    trits = torch.zeros(out, n * 5, dtype=torch.int32, device=t.device)
    for i in range(5):
        trits[:, i::5] = (t // (3**i)) % 3
    return trits - 1


def pack_binary(q: torch.Tensor) -> torch.Tensor:
    """Pack binary int8 {-1,+1} [out, in] -> uint8 [out, ceil(in/8)] (1 bit/weight).
    sign>0 -> bit 1, else bit 0."""
    b = (q > 0).to(torch.uint8)  # {0,1}
    out, in_ = b.shape
    n = (in_ + 7) // 8
    padded = torch.zeros(out, n * 8, dtype=torch.uint8, device=b.device)
    padded[:, :in_] = b
    packed = torch.zeros(out, n, dtype=torch.uint8, device=b.device)
    for i in range(8):
        packed |= padded[:, i::8] << i
    return packed


def unpack_binary(p: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 [out, ceil(in/8)] -> int8 [out, in] binary {-1,+1}."""
    t = p.to(torch.int64)
    out, n = t.shape
    bits = torch.zeros(out, n * 8, dtype=torch.int8, device=t.device)
    for i in range(8):
        bits[:, i::8] = ((t >> i) & 1).to(torch.int8)
    return bits * 2 - 1


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack int4 {-7..7} [out, in] -> uint8 [out, ceil(in/2)] (4 bits/weight).
    Two nibbles per byte: even-index weights low, odd-index high; offset +8."""
    v = (q.round().clamp(-7, 7).to(torch.int64) + 8).clamp(0, 15)  # 0..15
    out, in_ = v.shape
    if in_ % 2:
        v = torch.cat([v, torch.zeros(out, 1, dtype=v.dtype, device=v.device)], dim=1)
        in_ += 1
    packed = (v[:, 0::2] | (v[:, 1::2] << 4)).to(torch.uint8)
    assert packed.shape == (out, in_ // 2)
    return packed


def pack_2bit(q: torch.Tensor) -> torch.Tensor:
    """Pack 4-level int2 {-3,-1,1,3} [out, in] -> uint8 [out, ceil(in/4)].
    Values map to 2-bit codes: -3->0, -1->1, 1->2, 3->3; 4 values per byte."""
    lvl = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=q.device)
    idx = torch.argmin((q.unsqueeze(-1) - lvl).abs(), dim=-1).to(torch.int64)  # [out, in]
    out, in_ = idx.shape
    n = (in_ + 3) // 4
    padded = torch.zeros(out, n * 4, dtype=torch.int64, device=q.device)
    padded[:, :in_] = idx
    packed = torch.zeros(out, n, dtype=torch.uint8, device=q.device)
    for i in range(4):
        packed |= (padded[:, i::4] << (2 * i)).to(torch.uint8)
    return packed


def unpack_2bit(p: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 [out, ceil(in/4)] -> int2 {-3,-1,1,3} [out, in]."""
    lvl = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=p.device)
    t = p.to(torch.int64)
    out = torch.empty(t.shape[0], t.shape[1] * 4, dtype=torch.float32, device=p.device)
    for i in range(4):
        out[:, i::4] = lvl[(t >> (2 * i)) & 3]
    return out


LEVELS = {
    2: [-3.0, -1.0, 1.0, 3.0],
    3: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
}


def pack_nbit(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack small-grid values (LEVELS[bits]) [out, in] -> uint8 bitstream.
    bits=2: 4 values/byte; bits=3: 8 values per 3 bytes (24-bit groups)."""
    lv = torch.tensor(LEVELS[bits], device=q.device)
    idx = torch.argmin((q.unsqueeze(-1) - lv).abs(), dim=-1).to(torch.int64)
    out_, in_ = idx.shape
    if bits == 2:
        n = (in_ + 3) // 4
        padded = torch.zeros(out_, n * 4, dtype=torch.int64, device=q.device)
        padded[:, :in_] = idx
        packed = torch.zeros(out_, n, dtype=torch.uint8, device=q.device)
        for i in range(4):
            packed |= (padded[:, i::4] << (2 * i)).to(torch.uint8)
        return packed
    # bits == 3: 8 values -> 24 bits -> 3 bytes
    n = (in_ + 7) // 8
    padded = torch.zeros(out_, n * 8, dtype=torch.int64, device=q.device)
    padded[:, :in_] = idx
    g = padded.view(out_, n, 8)
    v = torch.zeros(out_, n, dtype=torch.int64, device=q.device)
    for k in range(8):
        v |= g[:, :, k] << (3 * k)
    b0 = (v & 255).to(torch.uint8)
    b1 = ((v >> 8) & 255).to(torch.uint8)
    b2 = ((v >> 16) & 255).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=-1).view(out_, n * 3)


def unpack_nbit(p: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of pack_nbit: uint8 bitstream -> value grid LEVELS[bits]."""
    lv = torch.tensor(LEVELS[bits], device=p.device)
    out_, nb = p.shape
    t = p.to(torch.int64)
    if bits == 2:
        res = torch.empty(out_, nb * 4, dtype=torch.float32, device=p.device)
        for i in range(4):
            res[:, i::4] = lv[(t >> (2 * i)) & 3]
        return res
    n = nb // 3
    b0 = t[:, 0::3]
    b1 = t[:, 1::3]
    b2 = t[:, 2::3]
    v = b0 | (b1 << 8) | (b2 << 16)  # [out, n]
    res = torch.empty(out_, n * 8, dtype=torch.float32, device=p.device)
    for k in range(8):
        res[:, k::8] = lv[(v >> (3 * k)) & 7]
    return res


def pack_int6(q: torch.Tensor) -> torch.Tensor:
    """Pack int6 {-31..31} [out, in] -> uint8 [out, ceil(in/4)*3] (6 bits/weight, 4 values per 3 bytes)."""
    v = q.round().clamp(-31, 31).to(torch.int64) + 31  # 0..62
    out_, in_ = v.shape
    n4 = (in_ + 3) // 4
    if in_ % 4:
        v = torch.cat([v, torch.zeros(out_, n4 * 4 - in_, dtype=v.dtype, device=v.device)], dim=1)
        in_ = n4 * 4
    g = v.view(out_, n4, 4)
    w24 = g[:, :, 0] | (g[:, :, 1] << 6) | (g[:, :, 2] << 12) | (g[:, :, 3] << 18)  # [out, n4] 24-bit
    b0 = (w24 & 255).to(torch.uint8)
    b1 = ((w24 >> 8) & 255).to(torch.uint8)
    b2 = ((w24 >> 16) & 255).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=-1).view(out_, n4 * 3)


def unpack_int6(p: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 [out, n4*3] -> int6 {-31..31} [out, n4*4]."""
    t = p.to(torch.int64)
    out_, nb = t.shape
    n4 = nb // 3
    w24 = t[:, 0::3] | (t[:, 1::3] << 8) | (t[:, 2::3] << 16)
    res = torch.empty(out_, n4 * 4, dtype=torch.int64, device=p.device)
    for i in range(4):
        res[:, i::4] = ((w24 >> (6 * i)) & 63) - 31
    return res


def unpack_int4(p: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 [out, ceil(in/2)] -> int [-7..7] [out, in] (offset binary)."""
    t = p.to(torch.int64)
    lo = (t & 15) - 8
    hi = (t >> 4) - 8
    out = torch.empty(t.shape[0], t.shape[1] * 2, dtype=torch.int64, device=t.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out
