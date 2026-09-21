"""E1 identity-parity smoke runner for the Qwen pyramid FFN adapter.

Tests topology identity, not value correctness. Verifies that a single-branch
pyramid path is EXACTLY equal to the flat reference path on the same frozen
layer (mean of one branch == identity), with finite outputs and an immutable
frozen base. Synthetic inputs only — no real model files are loaded or written.
"""
import argparse
import hashlib
import json
import os
import sys

# --- PATH: make scripts/ and project root discoverable -----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for p in (PROJECT_ROOT, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

# --- IMPORT: from qwen_pyramid (see E1 plan, QWEN_PYRAMID_MTP_TTT_PLAN.md) ---
try:
    from qwen_pyramid import FrozenFFNLayer, flat_forward, pyramid_forward
except ImportError as e:
    sys.stderr.write(
        "ERROR: cannot import from qwen_pyramid. "
        "The module scripts/qwen_pyramid.py must exist with "
        "FrozenFFNLayer, flat_forward, pyramid_forward.\n"
        f"Underlying: {e}\n"
    )
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E1 identity-parity smoke runner for qwen_pyramid."
    )
    parser.add_argument("--B", type=int, default=8, help="batch size")
    parser.add_argument("--S", type=int, default=1, help="sequence length")
    parser.add_argument("--H", type=int, default=5120, help="hidden size")
    parser.add_argument("--I", type=int, default=17408, help="intermediate size")
    parser.add_argument(
        "--mix",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="branch mix mode (E1 gate uses mean; sum kept for completeness)",
    )
    parser.add_argument("--output", type=str, default="reports/e1_pyramid_identity.json")
    parser.add_argument("--base-seed", type=int, default=0, help="torch seed")
    return parser.parse_args()


def sha256_of_tensor_bytes(t: torch.Tensor) -> str:
    """Hash a (frozen) base tensor's raw bytes to prove mutation-immutability."""
    buf = t.detach().cpu().contiguous().view(-1).to(torch.float32).numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()


def main() -> int:
    args = parse_args()

    # --- torch deterministic seed -------------------------------------------
    torch.manual_seed(args.base_seed)
    torch.set_grad_enabled(False)

    # --- synthetic data -----------------------------------------------------
    x = torch.randn(args.B, args.S, args.H, dtype=torch.float32)
    # Synthetic gate tensor matching [B, S, I] shape (mean over branches == 1).
    gate = torch.ones(args.B, args.S, args.I, dtype=torch.float32) * (1.0 / args.I)
    gate = gate.view(args.B, args.S, args.I)

    # --- build ONE frozen layer (random init is acceptable for a topology
    # identity gate — we test flat vs pyramid parity, not value correctness) --
    layer = FrozenFFNLayer(hidden_size=args.H, intermediate_size=args.I)
    layer.requires_grad_(False)

    # --- prove no mutation: hash a frozen base tensor before/after ----------
    base_hash_before = sha256_of_tensor_bytes(layer.gate_proj.weight)

    # --- run both paths -----------------------------------------------------
    flat = flat_forward(layer, x)
    pyr = pyramid_forward(layer, x, levels=[1], n_branches_per_level=[1], mix=args.mix)

    # --- checks -------------------------------------------------------------
    close = torch.allclose(flat, pyr, atol=1e-2, rtol=1e-2)
    eq = torch.equal(flat, pyr)  # n_branches=1, mean: exact identity
    finite = bool(torch.isfinite(flat).all().item() and torch.isfinite(pyr).all().item())

    base_hash_after = sha256_of_tensor_bytes(layer.gate_proj.weight)
    base_unchanged = base_hash_before == base_hash_after

    max_abs_err = float((flat - pyr).abs().max().item()) if flat.numel() else 0.0
    rel_err = float((flat - pyr).norm() / (flat.norm() + 1e-12))

    # --- build report -------------------------------------------------------
    report = {
        "gate": "e1_pyramid_identity",
        "config": {
            "B": args.B, "S": args.S, "H": args.H, "I": args.I,
            "mix": args.mix, "seed": args.base_seed,
        },
        "passed": bool(close and eq and finite and base_unchanged),
        "max_abs_err": max_abs_err,
        "rel_err": rel_err,
        "allclose": bool(close),
        "exact_equal": bool(eq),
        "finite": finite,
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "base_unchanged": base_unchanged,
        "shapes": {
            "input": list(x.shape),
            "flat": list(flat.shape),
            "pyramid": list(pyr.shape),
            "gate_proj_weight": list(layer.gate_proj.weight.shape),
        },
        "dtype": str(flat.dtype),
    }

    # --- write JSON ---------------------------------------------------------
    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    # --- print + flush ------------------------------------------------------
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    # --- exit code ----------------------------------------------------------
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
