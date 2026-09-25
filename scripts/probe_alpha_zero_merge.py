#!/usr/bin/env python3
"""Read-only diagnostic: per-source Banking77 holdout CE at alpha = 0.

Rebuilds the same three F3 children and the merged recursive-F3 candidate that
``run_banking77_cycle`` produces for a given seed, but passes
``prompt_ids=None`` to the *existing* ``_build_candidate`` helper. That
argument shape is the documented alpha = 0 state: the fresh "pyramid contour"
adapter scalars are zeroed and the ``self_improve`` gradient step is never run,
so the measured checkpoint is exactly the merge output with no bounded update
applied.

The question this decides: if a per-source regression already exceeds the
0.01 nats budget at alpha = 0, the defect is the non-function-preserving
three-way F3 self-merge itself and no trust region / step multiplier can fix
it. If alpha = 0 is clean, the regression is attributable to the bounded
self_improve update instead.

Read-only with respect to ``src/hagi/**`` and ``tests/**``: this script only
imports and calls their public (or already-imported) functions. It writes
solely into its own ``--output`` root.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from hagi.orchestrator import real_cycle as rc  # noqa: E402
from hagi.orchestrator.recursive import (  # noqa: E402
    CHILD_NAMES,
    ChildContext,
    GenerationRequest,
    SourceSpan,
)
from hagi.orchestrator.state import GrowthRunStore  # noqa: E402
from hagi.orchestrator.real_cycle import build_deterministic_parent  # noqa: E402

SOURCE_NAMES = ("A", "B", "C")
# The real gate: _verdict requires macro delta <= 0.0 and worst per-source
# delta <= 0.01 (recursive.py _verdict).
PER_SOURCE_BUDGET_NATS = 0.01
MACRO_BUDGET_NATS = 0.0

DEFAULT_ARTIFACT = _ROOT / "artifacts" / "datasets" / "banking77" / "57ec275d8078af65b7731c2a98be812d844a6d6b"
# Pinned manifest used by the existing runs (BANKING77_MANIFEST_SHA256).
PINNED_MANIFEST_SHA256 = rc.BANKING77_MANIFEST_SHA256
# Recorded in reports/banking77_pinned_multiseed_20260925.json as the
# tokenizer/artifact manifest digest for the published Banking77 artifact.
REPORTED_ARTIFACT_MANIFEST_SHA256 = "44d50edd994e7c32f30066a26052e15f902c74b79a7498ba60f475c76b453f3b"
DEFAULT_SEEDS = (416114, 416115, 416116)
# Diagnostic switch: the outer ternary lift used to build the candidate.
# "f3_tree" is the legacy staged transform (default, unchanged);
# "parent_preserving" is the orthogonal lift that fixes (1,1,1) so a self-merge
# of three identical parents is identity.
DEFAULT_LIFT_MODE = "f3_tree"


def _force_lift_mode(mode: str):
    """Pin merge.ternary_lift_mode for every target config built in-process.

    Monkeypatches the module-level helper so no file under src/ is touched.
    """
    original = rc._target_config

    def patched(parent_cfg):
        cfg = original(parent_cfg)
        cfg.merge.ternary_lift_mode = mode
        return cfg

    rc._target_config = patched
    return original


def _assemble(root: Path, artifact: str | Path, seed: int, max_steps: int):
    """Replicate run_banking77_cycle's data assembly, without training."""
    train_ids, test_ids, manifest, actual_manifest_sha, compact_vocab = rc._banking_data(artifact)
    if actual_manifest_sha != PINNED_MANIFEST_SHA256:
        raise ValueError("Banking77 manifest digest mismatch")
    store = GrowthRunStore(root / "store")
    pointer = store.read_parent()
    if pointer is None:
        pointer, parent_checkpoint = build_deterministic_parent(
            root, seed=seed, max_steps=max_steps, vocab_size=compact_vocab
        )
        store.bootstrap_parent(
            pointer,
            owner_id=rc.OWNER_ID,
            trusted_parent_token=rc.TrustedParentToken(rc.OWNER_ID, pointer.digest()),
        )
    else:
        parent_checkpoint = rc._load_committed_parent(store, pointer)
    train_parts = [
        [int(value) for value in part]
        for part in np.array_split(np.asarray(train_ids, dtype=np.int64), 3)
    ]
    if any(len(part) < 2 for part in train_parts):
        raise ValueError("Banking77 training partition is too small")
    holdout_parts = [
        [int(value) for value in part[:256]]
        for part in np.array_split(np.asarray(test_ids, dtype=np.int64), 3)
    ]
    if any(len(part) < 2 for part in holdout_parts):
        raise ValueError("Banking77 holdout partition is too small")
    row_ids = {
        source: rc._token_bytes_digest(part)
        for source, part in zip(SOURCE_NAMES, holdout_parts, strict=True)
    }
    holdout = rc._holdout_contract(
        root / "banking77-holdout.json",
        vocab_size=compact_vocab,
        streams=dict(zip(SOURCE_NAMES, holdout_parts, strict=True)),
        row_ids=row_ids,
        data_manifest_sha256=actual_manifest_sha,
        tokenizer_name=str(manifest["tokenizer_name"]),
    )
    spans = tuple(
        SourceSpan(
            source,
            sum(len(previous) for previous in train_parts[:index]),
            sum(len(previous) for previous in train_parts[: index + 1]),
            rc._token_bytes_digest(part),
        )
        for index, (source, part) in enumerate(zip(SOURCE_NAMES, train_parts, strict=True))
    )
    request = GenerationRequest(
        rc._next_generation_id(pointer),
        seed,
        tuple(rc.ChildPlan(source, span) for source, span in zip(SOURCE_NAMES, spans, strict=True)),
        pointer,
        str(parent_checkpoint),
        holdout,
    )
    return request, train_parts, holdout, manifest


def _probe_seed(
    root: Path,
    artifact: str | Path,
    seed: int,
    max_steps: int,
    device: str,
    cross_parent_transform: str = "f3_tree",
) -> dict[str, Any]:
    request, train_parts, holdout, manifest = _assemble(root, artifact, seed, max_steps)
    build_root = root / "build" / request.generation_id
    tokens = dict(zip(SOURCE_NAMES, train_parts, strict=True))

    children = [
        rc._build_child(
            ChildContext(
                generation_id=request.generation_id,
                parent_checkpoint_path=str(request.parent_checkpoint_path),
                parent_checkpoint_sha256=request.parent.checkpoint_sha256,
                parent_manifest_sha256=request.parent.manifest_sha256,
                name=CHILD_NAMES[index],
                source_id=source,
                seed=request.child_seeds[index],
                span=plan.span,
                protocol_sha256=request.holdout.protocol_sha256,
            ),
            build_root,
            tokens,
            max_steps,
        )
        for index, (source, plan) in enumerate(zip(SOURCE_NAMES, request.child_plans, strict=True))
    ]

    # alpha = 0: prompt_ids=None => contour scalars zeroed, no self_improve step.
    candidate = rc._build_candidate(
        rc.CandidateContext(
            generation_id=request.generation_id,
            parent_checkpoint_path=str(request.parent_checkpoint_path),
            parent_checkpoint_sha256=request.parent.checkpoint_sha256,
            parent_manifest_sha256=request.parent.manifest_sha256,
            children=tuple(children),
            protocol_sha256=request.holdout.protocol_sha256,
            base_seed=request.base_seed,
            self_improve_seed=request.base_seed + rc._SELF_IMPROVE_OFFSET,
        ),
        build_root,
        None,
        cross_parent_transform=cross_parent_transform,
    )

    holdout_payload = json.loads(Path(holdout.path).read_text(encoding="utf-8"))
    parent_metrics = rc._score_checkpoint_bytes(
        Path(request.parent_checkpoint_path).read_bytes(), holdout_payload, device
    )
    candidate_metrics = rc._score_checkpoint_bytes(
        Path(candidate.checkpoint_path).read_bytes(), holdout_payload, device
    )

    state = rc.load_payload(candidate.checkpoint_path)["model"]
    contour_keys = sorted(
        key for key in state if key.endswith(".adapters.pyramid.scale")
    )
    contour_values = [float(state[key].abs().max()) for key in contour_keys]
    per_source = {}
    for source in SOURCE_NAMES:
        parent_ce = float(parent_metrics[source].exact_ce)
        candidate_ce = float(candidate_metrics[source].exact_ce)
        delta = candidate_ce - parent_ce
        per_source[source] = {
            "parent_exact_ce": parent_ce,
            "alpha0_candidate_exact_ce": candidate_ce,
            "delta": delta,
            "exceeds_budget": bool(delta > PER_SOURCE_BUDGET_NATS),
            "scored_rows": int(parent_metrics[source].scored_rows),
        }
    macro_parent = sum(per_source[s]["parent_exact_ce"] for s in SOURCE_NAMES) / 3.0
    macro_candidate = sum(per_source[s]["alpha0_candidate_exact_ce"] for s in SOURCE_NAMES) / 3.0
    return {
        "seed": seed,
        "max_steps": max_steps,
        "device": device,
        "generation_id": request.generation_id,
        "parent_checkpoint_sha256": request.parent.checkpoint_sha256,
        "alpha0_candidate_checkpoint_sha256": candidate.checkpoint_sha256,
        "alpha0_applied_updates": int(candidate.self_improve.accepted_updates),
        "data_manifest_sha256": holdout.data_manifest_sha256,
        "tokenizer_name": holdout.tokenizer_name,
        "vocab_size": int(holdout_payload["vocab_size"]),
        "artifact_manifest_sha256_reported": REPORTED_ARTIFACT_MANIFEST_SHA256,
        "per_source": per_source,
        "macro_parent_exact_ce": macro_parent,
        "macro_alpha0_candidate_exact_ce": macro_candidate,
        "macro_delta": macro_candidate - macro_parent,
        "macro_exceeds_budget": bool(macro_candidate - macro_parent > MACRO_BUDGET_NATS),
        "worst_source_delta": max(per_source[s]["delta"] for s in SOURCE_NAMES),
        "any_source_exceeds_budget": any(per_source[s]["exceeds_budget"] for s in SOURCE_NAMES),
        "contour_keys": contour_keys,
        "contour_max_abs_values": contour_values,
        "contour_all_zero": all(value == 0.0 for value in contour_values),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh run root")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--manifest-sha256", default=PINNED_MANIFEST_SHA256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--lift-mode",
        choices=("f3_tree", "parent_preserving"),
        default=DEFAULT_LIFT_MODE,
        help="outer ternary lift used to build the candidate",
    )
    args = parser.parse_args(argv)

    if args.device != "cpu":
        print("probe is CPU-only; --device must be cpu", file=sys.stderr)
        return 2
    if not 1 <= args.max_steps <= 8:
        print("--max-steps must be in [1, 8]", file=sys.stderr)
        return 2
    if args.manifest_sha256 != PINNED_MANIFEST_SHA256:
        print("--manifest-sha256 must match the pinned Banking77 manifest", file=sys.stderr)
        return 2
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    _force_lift_mode(args.lift_mode)
    print(f"outer ternary lift mode: {args.lift_mode}")

    try:
        per_seed = []
        for seed in seeds:
            root = Path(args.output).resolve() / f"seed-{seed}"
            root.mkdir(parents=True, exist_ok=True)
            record = _probe_seed(
                root, args.artifact, seed, args.max_steps, args.device,
                cross_parent_transform=args.lift_mode,
            )
            per_seed.append(record)
            print(f"--- seed {seed} ---")
            for source in SOURCE_NAMES:
                item = record["per_source"][source]
                print(
                    f"{source}: parent={item['parent_exact_ce']:.6f} "
                    f"alpha0={item['alpha0_candidate_exact_ce']:.6f} "
                    f"delta={item['delta']:+.6f} "
                    f"exceeds_0.01={item['exceeds_budget']}"
                )
            print(
                f"macro: parent={record['macro_parent_exact_ce']:.6f} "
                f"alpha0={record['macro_alpha0_candidate_exact_ce']:.6f} "
                f"delta={record['macro_delta']:+.6f} "
                f"worst_source={record['worst_source_delta']:+.6f} "
                f"contour_all_zero={record['contour_all_zero']}"
            )
        payload = {
            "schema": "hagi_alpha_zero_merge_probe_v1",
            "alpha": 0.0,
            "lift_mode": args.lift_mode,
            "definition": (
                "alpha=0 is _build_candidate(..., prompt_ids=None): merged recursive-F3 "
                "state with every '.adapters.pyramid.scale' contour scalar zeroed and "
                "the self_improve gradient step never executed."
            ),
            "per_source_budget_nats": PER_SOURCE_BUDGET_NATS,
            "macro_budget_nats": MACRO_BUDGET_NATS,
            "pinned_manifest_sha256": args.manifest_sha256,
            "artifact_dir": str(Path(args.artifact).resolve()),
            "max_steps": args.max_steps,
            "device": args.device,
            "seeds": per_seed,
            "any_seed_source_exceeds_budget": any(
                item["any_source_exceeds_budget"] for item in per_seed
            ),
            "any_seed_macro_exceeds_budget": any(
                item["macro_exceeds_budget"] for item in per_seed
            ),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"alpha-zero probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
