#!/usr/bin/env python
"""Merge level-1 experts into a single wide model for recursive growth.

Level-1 experts are MergedHAGI (H=2304) trained from the level-0 merged
model.  Their inner CrossMixer weights are dropped during merge and replaced
with fresh level-1 mixers, exactly like the level-0→level-1 transition.

Usage:
    python scripts/merge_level1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

# --- configuration ----------------------------------------------------------

# Expert checkpoint directories (one per domain group).
EXPERT_DIRS = [
    ROOT / "checkpoints_l1" / "ru_general",
    ROOT / "checkpoints_l1" / "en_general",
    ROOT / "checkpoints_l1" / "math_code",
    ROOT / "checkpoints_l1" / "instruct",
]

# Expert geometry (from level-0 merged config).
EXPERT_HIDDEN = 2304
EXPERT_QH = 36
EXPERT_KV = 18
EXPERT_HEAD_DIM = 64
EXPERT_FFN = 1152  # 18 * 64
N_LAYERS = 3
VOCAB = 32768

MERGED_DIR = ROOT / "checkpoints_l1_merged"
MERGED_CKPT = MERGED_DIR / "step-0000000.pt"

# ---------------------------------------------------------------------------


def _latest_ckpt(d: Path) -> Path | None:
    """Find the latest step-*.pt in a checkpoint directory."""
    if not d.exists():
        return None
    ckpts = sorted(d.glob("step-*.pt"))
    return ckpts[-1] if ckpts else None


def make_merged_config(ckpts: list[Path], n: int) -> Path:
    """Generate merged config for N L1 experts at H=2304."""
    import yaml

    base = yaml.safe_load((ROOT / "configs" / "level0_merged.yaml").read_text())
    eh = EXPERT_HIDDEN
    merged = {
        "model": {
            "vocab_size": VOCAB,
            "hidden_size": eh * n,
            "num_layers": N_LAYERS,
            "loop_depth": base["model"]["loop_depth"],
            "norm_eps": base["model"]["norm_eps"],
            "init_orthogonal": base["model"]["init_orthogonal"],
            "attention": {
                "num_query_heads": EXPERT_QH * n,
                "num_kv_heads": EXPERT_KV * n,
                "head_dim": EXPERT_HEAD_DIM,
                "rope_theta": base["model"]["attention"]["rope_theta"],
                "max_seq_len": base["model"]["attention"]["max_seq_len"],
                "qk_norm": base["model"]["attention"]["qk_norm"],
            },
            "sliding": base["model"]["sliding"],
            "embedding": base["model"]["embedding"],
            "ffn": {
                **base["model"]["ffn"],
                "intermediate_size": EXPERT_FFN * n,
            },
            "ternary": base["model"]["ternary"],
            "head": base["model"]["head"],
            "multimodal": base["model"]["multimodal"],
        },
        "merge": {
            "enabled": True,
            "n_experts": n,
            "expert_hidden": eh,
            "expert_checkpoints": [str(p) for p in ckpts],
            "freeze_experts": False,
            "mixer_init_scale": 0.0,
            "mixer_type": "hadamard",
            "mixer_hadamard_groups": [2, 2],
        },
        "train": {
            "max_steps": 4000,
            "batch_size": 64,
            "grad_accum_steps": 4,
            "grad_checkpointing": True,
            "use_muon": False,
            "ternary_step_cache": True,
            "ce_keep_rate": 0.25,
            "ce_keep_mode": "stride",
            "learning_rate": 3e-5,
            "max_grad_norm": 1.0,
            "precision": "bf16",
            "compile_model": True,
            "adam": {"body_lr_scale": 8.0, "beta1": 0.9, "beta2": 0.95, "weight_decay": 0.1},
            "schedule": {"warmup_steps": 200, "decay_fraction": 0.2, "min_lr_ratio": 0.02},
            "data": {
                **base["train"]["data"],
                "weights": {
                    "edu": 1.0, "slimpajama": 1.0, "wikipedia_ru": 1.0,
                    "oscar_ru": 1.0, "openwebmath": 1.0, "smoltalk": 1.0,
                    "tinystories": 1.0, "wikipedia_en": 1.0, "python_instruct": 1.0,
                },
            },
            "logging": base["train"]["logging"],
            "z_loss_weight": base["train"]["z_loss_weight"],
            "checkpoint_dir": str(MERGED_DIR),
            "checkpoint_interval": 500,
            "checkpoint_keep_last": 2,
            "tokenizer": base["train"]["tokenizer"],
        },
        "inference": base["inference"],
    }
    cfg_path = ROOT / "configs" / "level1_merged.yaml"
    cfg_path.parent.mkdir(exist_ok=True)
    yaml.safe_dump(merged, cfg_path.open("w"), sort_keys=False, width=100)
    print(f"wrote {cfg_path}")
    return cfg_path


def main() -> int:
    import subprocess

    # Collect expert checkpoints.
    ckpts: list[Path] = []
    for d in EXPERT_DIRS:
        ck = _latest_ckpt(d)
        if ck is None:
            print(f"WARNING: no checkpoint in {d}, skipping")
            continue
        ckpts.append(ck)
        print(f"  expert: {ck}")

    if len(ckpts) < 2:
        print(f"ERROR: need >=2 experts to merge, found {len(ckpts)}", file=sys.stderr)
        return 1

    n = len(ckpts)
    Hm = EXPERT_HIDDEN * n
    embed = VOCAB * Hm
    print(f"\n{n} experts x H={EXPERT_HIDDEN} -> H_merged={Hm}")
    print(f"  estimated embed+head = {2*embed/1e6:.0f}M")
    print(f"  estimated body = {n*71.7:.0f}M")
    print(f"  estimated total ~ {(2*embed + n*71.7e6)/1e6:.0f}M")

    cfg_path = make_merged_config(ckpts, n)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    # Merge with --drop-expert-mixers (L1 experts have inner mixers).
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "merge_experts.py"),
        "--config", str(cfg_path),
        "--experts", *[str(c) for c in ckpts],
        "--out", str(MERGED_CKPT),
        "--drop-expert-mixers",
    ]
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"ERROR: merge failed (exit {result.returncode})", file=sys.stderr)
        return result.returncode

    # Report params.
    from hagi.train.checkpoint import load_payload
    payload = load_payload(str(MERGED_CKPT), "cpu")
    total = sum(t.numel() for t in payload["model"].values())
    print(f"\nLevel-1 merged: {total/1e6:.1f}M params -> {MERGED_CKPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
