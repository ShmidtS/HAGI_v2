"""Generate the F3 (recursive ternary tree) training + eval configs.

The three leaves MUST be provably identical except for ``train.data.seed``
and ``train.checkpoint_dir`` -- ``merge_recursive_f3`` fingerprints each
child's whole config and only those two keys may differ
(``merge.py:_canonical``). So they are built from one literal dict here and
written out three times, never hand-copied.

Usage:  python scripts/make_f3_configs.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

CONFIGS = _ROOT / "configs"

# Held-out domains in scripts/eval_holdout.py:DOMAIN_FILES, taken from
# data/mix.json minus python_instruct (3.1M tokens -> ~10 epochs at this
# budget, and no untouched region left to score).
WEIGHTS = {
    "slimpajama": 0.25,
    "edu": 0.40,
    "wikipedia_en": 0.10,
    "wikipedia_ru": 0.08,
    "oscar_ru": 0.07,
    "openwebmath": 0.10,
}

LEAF_SEEDS = (1001, 1002, 1003)


def leaf_config(seed: int) -> dict:
    """One depth-1 F3 leaf. The F3 child fences (merge.py:1390-1410) are
    satisfied by construction, not by luck."""
    return {
        "model": {
            "vocab_size": 32768,
            "hidden_size": 384,  # = parent_hidden // 3
            "num_layers": 3,
            # ternary_f3 requires loop_depth=2 (config.py:1485)
            "loop_depth": 2,
            # Master init seed: all three leaves start from the SAME random
            # init, which is the method's core requirement. On the record.
            "init_seed": 0,
            "attention": {
                "num_query_heads": 6,  # 6 * 64 == 384
                "num_kv_heads": 2,
                "head_dim": 64,
                "max_seq_len": 4096,
            },
            "embedding": {
                "tie_lm_head": False,  # fence
                "conv_kernel": 1,  # fence
            },
            # Leaves are effective-sparse, not ternary (merge.py:1393)
            "ternary": {"enabled": False},
            # unigram prior adds once, not per child (config.py:1482);
            # 'prior' proposal would require it, so proposal is uniform.
            "head": {"unigram_prior": False, "sampled_proposal": "uniform"},
            # The bark is born in the lift; children must carry no adaptive
            # state (merge.py:1400, 1354)
            "cortex": {"enabled": False},
            "adapters": {"enabled": False},
            "decision": {"enabled": False},
        },
        "train": {
            "max_steps": 3000,
            "checkpoint_interval": 1000,
            "precision": "bf16",
            "ternary_fp32_master": False,  # fence (config.py:1490)
            "batch_size": 8,
            "grad_accum_steps": 4,
            "learning_rate": 3e-4,
            "checkpoint_dir": f"checkpoints/f3_leaf_s{seed}",
            "data": {
                "data_dir": "data",
                "seq_len": 1024,
                "seed": seed,  # the ONLY per-leaf difference
                "weights": dict(WEIGHTS),
            },
        },
    }


def parent_config(with_cortex: bool) -> dict:
    """Depth-1 parent: three leaves lifted into hidden_size=1152."""
    return {
        "model": {
            "vocab_size": 32768,
            "hidden_size": 1152,  # = 3 * 384
            "num_layers": 3,
            "loop_depth": 2,  # fence, same as the children
            "init_seed": 0,
            "attention": {
                "num_query_heads": 18,  # 18 * 64 == 1152, 18 % n_experts == 0
                "num_kv_heads": 6,  # 2 * 3
                "head_dim": 64,
                "max_seq_len": 4096,
            },
            "embedding": {"tie_lm_head": False, "conv_kernel": 1},
            "ternary": {"enabled": False},
            "head": {"unigram_prior": False, "sampled_proposal": "uniform"},
            "cortex": {
                "enabled": with_cortex,
                # REQUIRED: validate_config rejects any other mode for
                # ternary_f3 (config.py:1495). The nocortex arm keeps the key
                # so the two arms differ in exactly one boolean.
                "mode": "root",
                "num_levels": 3,
                "rank": 64,
                "link_strides": [1, 2],
                "residual_scale": 0.1,
            },
            "adapters": {"enabled": False},
            "decision": {"enabled": False},
        },
        "merge": {
            "enabled": True,
            "mixer_type": "ternary_f3",
            "n_experts": 3,
            "expert_hidden": 384,  # the CHILD width (config.py:1516)
            "ternary_depth": 1,
            "ternary_lift_mode": "parent_preserving",
            "ternary_tree_schema_version": 1,
            "freeze_experts": False,
            "expert_checkpoints": [
                f"checkpoints/f3_leaf_s{s}/step-0003000.pt" for s in LEAF_SEEDS
            ],
        },
        "train": {
            "max_steps": 9000,
            "checkpoint_interval": 1000,
            "precision": "bf16",
            "ternary_fp32_master": False,
            "batch_size": 8,
            "grad_accum_steps": 4,
            "learning_rate": 3e-4,
            "checkpoint_dir": "checkpoints/f3_parent_d1",
            "data": {
                "data_dir": "data",
                "seq_len": 1024,
                "seed": 1001,
                "weights": dict(WEIGHTS),
            },
        },
    }


def eval_config(base: dict, *, ckpt: str) -> dict:
    """Minimal eval-side config: the architecture the checkpoint must be
    rebuilt from, plus the data dir eval_holdout.py reads. No training keys
    beyond the defaults eval does not use."""
    cfg = copy.deepcopy(base)
    cfg["train"].pop("max_steps", None)
    cfg["train"]["checkpoint_dir"] = ckpt.rsplit("/", 1)[0]
    cfg["train"]["data"]["seed"] = 1001
    return cfg


def _dump(path: Path, cfg: dict) -> None:
    header = (
        f"# GENERATED by scripts/make_f3_configs.py -- do not hand-edit.\n"
        f"# Loads clean: python -c \"import sys;sys.path.insert(0,'src');"
        f"from hagi.config import load_config;load_config('{path.as_posix()}')\"\n"
    )
    path.write_text(
        header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {path.relative_to(_ROOT)}")


def main() -> None:
    CONFIGS.mkdir(exist_ok=True)
    for seed in LEAF_SEEDS:
        _dump(CONFIGS / f"f3_leaf_s{seed}.yaml", leaf_config(seed))
    with_cortex = parent_config(True)
    no_cortex = parent_config(False)
    no_cortex["train"]["checkpoint_dir"] = "checkpoints/f3_parent_d1_nocortex"
    _dump(CONFIGS / "f3_parent_d1.yaml", with_cortex)
    _dump(CONFIGS / "f3_parent_d1_nocortex.yaml", no_cortex)
    _dump(
        CONFIGS / "f3_eval_leaf.yaml",
        eval_config(
            leaf_config(1001), ckpt="checkpoints/f3_leaf_s1001/step-0003000.pt"
        ),
    )
    _dump(
        CONFIGS / "f3_eval_parent.yaml",
        eval_config(
            with_cortex, ckpt="checkpoints/f3_parent_d1/step-0009000.pt"
        ),
    )


if __name__ == "__main__":
    main()
