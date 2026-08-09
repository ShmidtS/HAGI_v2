"""Recursive-growth pipeline: train per-corpus experts -> merge -> grow to 1B.

Level 0:
  For each corpus, train experts (H=128) sequentially until the data is
  exhausted. Each expert trains on a contiguous slice (start_offset = previous
  expert's consumed tokens) to saturation OR the end of the slice. All level-0
  experts are block-diagonally merged into H=N*128.

Level 1+ (recursive):
  The merged model becomes the shared prior (--init-from). Level-N experts are
  copies of the merged level-(N-1) model, each specializing on one corpus, then
  merged again. Repeat until the model reaches ~1B parameters.

Usage:
    python scripts/run_corpus_pipeline.py --level 0
    python scripts/run_corpus_pipeline.py --level 0 --dry-run
    python scripts/run_corpus_pipeline.py --level 0 --only-merge
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CONFIGS = ROOT / "configs"
EXPERTS = CONFIGS / "corpus_experts"

# Each corpus is a separate source; experts are trained from it until exhausted.
CORPORA: list[str] = [
    "edu",
    "slimpajama",
    "wikipedia_ru",
    "oscar_ru",
    "openwebmath",
    "smoltalk",
    "tinystories",
    "wikipedia_en",
    "python_instruct",
]

DATA_DIR = ROOT / "data"
TOKEN_DTYPE = np.uint32


def _run(cmd: list[str], dry: bool) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    if dry:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def _corpus_tokens(corpus: str) -> int:
    """Total tokens in a corpus .bin (preferring the compacted stream)."""
    base = DATA_DIR / f"{corpus}.bin"
    for suffix in (".compact2.bin", ".compact.bin", ".bin"):
        p = base.with_suffix(suffix) if suffix != ".bin" else base
        if p.exists():
            return p.stat().st_size // 4
    raise FileNotFoundError(f"no corpus file for {corpus}")


def _consumed(corpus: str) -> int:
    """Cumulative tokens consumed for a corpus (from consumed.json, else 0).

    ``consumed.json`` now holds the *cumulative* offset (start_offset + this
    slice's consumption), so it is directly usable as the next slice's
    start_offset. Clamped to the corpus size by the caller.
    """
    p = ROOT / f"checkpoints_corpus/{corpus}/consumed.json"
    if p.exists():
        return int(json.loads(p.read_text())["consumed_tokens"])
    return 0


def _slice_ckpt(corpus: str, slice_idx: int) -> Path | None:
    """Highest-numbered step-*.pt in a corpus slice's checkpoint dir."""
    from hagi.train.checkpoint import latest_checkpoint

    return latest_checkpoint(ROOT / f"checkpoints_corpus/{corpus}/slice{slice_idx}")


def _train_expert(corpus: str, offset: int, slice_idx: int, init_from: str | None, dry: bool) -> None:
    """Train one expert on a corpus slice starting at ``offset``.

    Each slice writes to its own checkpoint dir (``slice{slice_idx}``) so the
    ``checkpoint_keep_last`` pruning (which sorts by step number) never deletes
    another slice's checkpoints — every slice restarts at step 0.
    """
    cmd = [
        sys.executable,
        str(SCRIPTS / "train.py"),
        "--config", str(EXPERTS / f"expert_{corpus}.yaml"),
        "--start-offset", str(offset),
        "--checkpoint-dir", str(ROOT / f"checkpoints_corpus/{corpus}/slice{slice_idx}"),
    ]
    if init_from is not None:
        cmd += ["--init-from", init_from]
    _run(cmd, dry)


def _train_corpus_experts(corpus: str, init_from: str | None, dry: bool) -> list[Path]:
    """Train experts from a corpus until the data is exhausted.

    Returns the list of expert checkpoints produced (one per slice). Existing
    slices are collected first (so a corpus whose data is already exhausted
    still contributes its trained experts to the merge), then new slices are
    trained from the cumulative offset until the data runs out.
    """
    total = _corpus_tokens(corpus)
    ckpts: list[Path] = []
    # Collect all already-trained slices.
    slice_idx = 0
    while True:
        existing = _slice_ckpt(corpus, slice_idx)
        if existing is None:
            break
        ckpts.append(existing)
        slice_idx += 1
    offset = min(_consumed(corpus), total)
    print(f"\n=== corpus {corpus}: {total/1e9:.3f}B tokens, {slice_idx} slices trained, offset {offset} ===")
    while offset < total:
        # Skip this slice if it already has a trained expert.
        existing = _slice_ckpt(corpus, slice_idx)
        if existing is not None:
            print(f"[skip] corpus {corpus} slice{slice_idx} already trained: {existing}")
            ckpts.append(existing)
            offset = min(_consumed(corpus), total)
            slice_idx += 1
            continue
        _train_expert(corpus, offset, slice_idx, init_from, dry)
        ckpt = _slice_ckpt(corpus, slice_idx)
        if ckpt is not None:
            ckpts.append(ckpt)
        new_offset = min(_consumed(corpus), total)
        if new_offset <= offset:
            print(f"[warn] corpus {corpus}: no progress (offset {offset} -> {new_offset}); stopping")
            break
        offset = new_offset
        slice_idx += 1
        if dry:
            # In dry-run, simulate one expert per corpus.
            break
    return ckpts


def _make_merged_config(ckpts: list[Path], n_experts: int, expert_hidden: int, level: int) -> Path:
    """Generate a merged config for N experts block-diagonally merged into
    H = N * expert_hidden. The merged model's geometry (hidden_size, heads) is
    derived from the expert geometry and the expert count.
    """
    import yaml

    # Base: the first expert config (H=128 geometry) as the expert template.
    expert_cfg = yaml.safe_load((EXPERTS / "expert_edu.yaml").read_text())
    eh = expert_hidden
    n = n_experts
    merged = {
        "model": {
            "vocab_size": expert_cfg["model"]["vocab_size"],
            "hidden_size": eh * n,
            "num_layers": expert_cfg["model"]["num_layers"],
            "loop_depth": expert_cfg["model"]["loop_depth"],
            "norm_eps": expert_cfg["model"]["norm_eps"],
            "init_orthogonal": expert_cfg["model"]["init_orthogonal"],
            "attention": {
                "num_query_heads": expert_cfg["model"]["attention"]["num_query_heads"] * n,
                "num_kv_heads": expert_cfg["model"]["attention"]["num_kv_heads"] * n,
                "head_dim": expert_cfg["model"]["attention"]["head_dim"],
                "rope_theta": expert_cfg["model"]["attention"]["rope_theta"],
                "max_seq_len": expert_cfg["model"]["attention"]["max_seq_len"],
                "qk_norm": expert_cfg["model"]["attention"]["qk_norm"],
            },
            "sliding": expert_cfg["model"]["sliding"],
            "embedding": expert_cfg["model"]["embedding"],
            "ffn": expert_cfg["model"]["ffn"],
            "ternary": expert_cfg["model"]["ternary"],
            "head": expert_cfg["model"]["head"],
            "multimodal": expert_cfg["model"]["multimodal"],
        },
        "merge": {
            "enabled": True,
            "n_experts": n,
            "expert_hidden": eh,
            "expert_checkpoints": [str(p) for p in ckpts],
            "freeze_experts": False,
            "mixer_init_scale": 0.0,
        },
        "train": {
            "max_steps": 2000,
            "batch_size": 256,
            "grad_accum_steps": 1,
            "grad_checkpointing": False,
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
                **expert_cfg["train"]["data"],
                # Joint training must see all corpora, not just the first
                # expert's single source. Equal weights: the merged model should
                # learn to route across all domains.
                "weights": {c: 1.0 for c in CORPORA},
            },
            "logging": expert_cfg["train"]["logging"],
            "z_loss_weight": expert_cfg["train"]["z_loss_weight"],
            "checkpoint_dir": "",
            "checkpoint_interval": 500,
            "checkpoint_keep_last": 2,
            "tokenizer": expert_cfg["train"]["tokenizer"],
        },
        "inference": expert_cfg["inference"],
    }
    path = CONFIGS / f"level{level}_merged.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(merged, sort_keys=False, width=100))
    return path


def _merge_experts(ckpts: list[Path], merged_cfg: Path, out: Path, dry: bool) -> None:
    """Block-diagonally merge N expert checkpoints into one wide model."""
    if out.exists():
        print(f"[skip] merged checkpoint already exists: {out}")
        return
    _run(
        [
            sys.executable,
            str(SCRIPTS / "merge_experts.py"),
            "--config", str(merged_cfg),
            "--experts", *[str(p) for p in ckpts],
            "--out", str(out),
        ],
        dry,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the recursive-growth pipeline")
    parser.add_argument("--level", type=int, default=0, help="growth level (0 = first experts)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-merge", action="store_true")
    args = parser.parse_args()

    # Level 0: experts start from random init (init_seed=1234), not a merged prior.
    init_from = None
    if args.level > 0:
        # Level N: experts start from the merged level-(N-1) model.
        prev_merged = ROOT / f"checkpoints_level{args.level-1}_merged/step-0000000.pt"
        if not prev_merged.exists():
            print(f"ERROR: level-{args.level-1} merged checkpoint not found: {prev_merged}", file=sys.stderr)
            return 1
        init_from = str(prev_merged)

    # 0. Generate configs.
    _run([sys.executable, str(SCRIPTS / "make_corpus_expert_configs.py")], args.dry_run)

    # 1. Train experts from each corpus until data is exhausted.
    all_ckpts: list[Path] = []
    if not args.only_merge:
        for corpus in CORPORA:
            ckpts = _train_corpus_experts(corpus, init_from, args.dry_run)
            all_ckpts.extend(ckpts)

    # 2. Merge all level-0 experts.
    merged_dir = ROOT / f"checkpoints_level{args.level}_merged"
    merged = merged_dir / "step-0000000.pt"
    if args.only_merge:
        # Collect all expert checkpoints from disk (one per slice).
        for corpus in CORPORA:
            slice_idx = 0
            while True:
                ckpt = _slice_ckpt(corpus, slice_idx)
                if ckpt is None:
                    break
                all_ckpts.append(ckpt)
                slice_idx += 1
    if not all_ckpts:
        print("ERROR: no expert checkpoints to merge", file=sys.stderr)
        return 1
    # Expert geometry: H=128, qh=2, kv=1 (from the expert base config).
    expert_hidden = 128
    merged_cfg = _make_merged_config(all_ckpts, len(all_ckpts), expert_hidden, args.level)
    _merge_experts(all_ckpts, merged_cfg, merged, args.dry_run)

    print(f"\nLevel-{args.level} complete: {len(all_ckpts)} experts merged -> {merged}")
    print(f"  total params: {_merged_params(merged) if merged.exists() else 'n/a'}")
    return 0


def _merged_params(path: Path) -> str:
    """Report total params of a merged checkpoint (best-effort)."""
    try:
        from hagi.train.checkpoint import load_payload
        payload = load_payload(str(path), "cpu")
        model = payload["model"]
        total = sum(t.numel() for t in model.values())
        return f"{total/1e6:.1f}M"
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


if __name__ == "__main__":
    raise SystemExit(main())
