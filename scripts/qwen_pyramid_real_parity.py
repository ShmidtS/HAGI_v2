"""E1: real-weight parity of the dense-Qwen pyramid FFN adapter.

Loads the real MTP layer-0 MLP weights from
``models/ternary-bonsai-2-27b-mtp/model_mtp.safetensors`` into
:mod:`scripts.qwen_pyramid.FrozenFFNLayer` and compares ``mlp_forward`` against
the native ``transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5MLP`` with
bit-for-bit identity on a BF16 input. The frozen base tensors are never
mutated or written; their hashes are recorded before/after to prove
immutability, and the source safetensors file is opened read-only.

Run:
    .venv/Scripts/python.exe scripts/qwen_pyramid_real_parity.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
)

from qwen_pyramid import FrozenFFNLayer  # noqa: E402

DEFAULT_GGUF = "models/ternary-bonsai-2-27b-mtp/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf"
DEFAULT_SAFETENSORS = "models/ternary-bonsai-2-27b-mtp/model_mtp.safetensors"
DEFAULT_DONOR_CONFIG = "models/ternary-bonsai-2-27b-mtp/donor"
DEFAULT_OUTPUT = "reports/e1_pyramid_real_parity.json"

WEIGHT_KEYS = [
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
]


def sha256_of_tensor(tensor: torch.Tensor) -> str:
    payload = (
        tensor.detach().cpu().contiguous().view(-1).to(torch.float32).numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safetensors", default=DEFAULT_SAFETENSORS)
    parser.add_argument("--config", default=DEFAULT_DONOR_CONFIG)
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S", type=int, default=1)
    parser.add_argument("--H", type=int, default=5120)
    parser.add_argument("--I", type=int, default=17408)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def run_real_parity(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    state = load_file(args.safetensors, device="cpu")
    config = AutoConfig.from_pretrained(args.config, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    hidden_size = getattr(text_config, "hidden_size", args.H)
    intermediate_size = getattr(text_config, "intermediate_size", args.I)
    torch.manual_seed(args.seed + 1)
    layer = FrozenFFNLayer(hidden_size, intermediate_size)
    with torch.no_grad():
        layer.load_base_weights(state, "mtp.layers.0.")
        for p in layer.parameters():
            p.requires_grad_(False)
        layer.gate_proj.weight.copy_(
            layer.gate_proj.weight.to(dtype=torch.bfloat16)
        )
        layer.up_proj.weight.copy_(
            layer.up_proj.weight.to(dtype=torch.bfloat16)
        )
        layer.down_proj.weight.copy_(
            layer.down_proj.weight.to(dtype=torch.bfloat16)
        )

    native = Qwen3_5MLP(
        text_config, intermediate_size
    ).to(dtype=torch.bfloat16)
    with torch.no_grad():
        native.gate_proj.weight.copy_(
            state["mtp.layers.0.mlp.gate_proj.weight"].to(torch.bfloat16)
        )
        native.up_proj.weight.copy_(
            state["mtp.layers.0.mlp.up_proj.weight"].to(torch.bfloat16)
        )
        native.down_proj.weight.copy_(
            state["mtp.layers.0.mlp.down_proj.weight"].to(torch.bfloat16)
        )

    file_hash_before = sha256_of_file(args.safetensors)
    weight_hashes_before = {key: sha256_of_tensor(state[key]) for key in WEIGHT_KEYS}

    native_norm = Qwen3_5RMSNorm(args.H).to(dtype=torch.bfloat16)
    with torch.no_grad():
        native_norm.weight.copy_(
            state["mtp.layers.0.post_attention_layernorm.weight"].to(torch.bfloat16)
        )

    torch.manual_seed(args.seed + 2)
    x = torch.randn(args.B, args.S, args.H, dtype=torch.bfloat16)
    xn_native = native_norm(x)
    xn_custom = layer.norm(x)
    norm_equal = torch.equal(xn_native, xn_custom)

    adapter_out = layer.mlp_forward(xn_native)
    native_out = native(xn_native)

    file_hash_after = sha256_of_file(args.safetensors)
    weight_hashes_after = {key: sha256_of_tensor(state[key]) for key in WEIGHT_KEYS}

    diff = (adapter_out.float() - native_out.float()).abs()
    max_abs_err = float(diff.max().item()) if diff.numel() else 0.0
    rel_err = float(
        diff.max().item() / (native_out.float().abs().max().clamp_min(1e-6))
    )

    report = {
        "gate": "e1_pyramid_real_parity",
        "config": {
            "B": args.B,
            "S": args.S,
            "H": args.H,
            "I": args.I,
            "seed": args.seed,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
        },
        "passed": bool(
            torch.equal(adapter_out, native_out)
            and norm_equal
            and adapter_out.dtype == torch.bfloat16
            and torch.isfinite(adapter_out).all()
            and weight_hashes_before == weight_hashes_after
            and file_hash_before == file_hash_after
            and all(p.grad is None for p in layer.parameters())
        ),
        "exact_equal": bool(torch.equal(adapter_out, native_out)),
        "allclose": bool(
            torch.allclose(
                adapter_out.float(), native_out.float(), atol=1e-2, rtol=1e-2
            )
        ),
        "finite": bool(torch.isfinite(adapter_out).all().item()),
        "dtype": str(adapter_out.dtype),
        "norm_equal": bool(norm_equal),
        "base_unchanged": weight_hashes_before == weight_hashes_after
        and file_hash_before == file_hash_after,
        "base_grad_absent": all(p.grad is None for p in layer.parameters()),
        "weight_hashes": {
            key: {"before": weight_hashes_before[key], "after": weight_hashes_after[key]}
            for key in WEIGHT_KEYS
        },
        "safetensors_sha256": {"before": file_hash_before, "after": file_hash_after},
        "tolerances": {"atol": 1e-2, "rtol": 1e-2},
        "max_abs_diff": max_abs_err,
        "max_rel_diff": rel_err,
        "shapes": {
            "input": list(x.shape),
            "adapter_out": list(adapter_out.shape),
            "native_out": list(native_out.shape),
        },
    }
    return report


def main() -> int:
    args = parse_args()
    report = run_real_parity(args)
    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    temporary = args.output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
