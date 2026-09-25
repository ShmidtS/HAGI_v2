#!/usr/bin/env python3
"""Run one bounded deterministic recursive-generation cycle."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.orchestrator.real_cycle import (  # noqa: E402
    SYNTHETIC_V3_SEED,
    result_data_provenance,
    run_banking77_cycle,
    run_bounded_cycle,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args(argv)

    try:
        if (args.artifact is None) != (args.manifest_sha256 is None):
            raise ValueError(
                "--artifact and --manifest-sha256 must be supplied together"
            )
        if not 1 <= args.max_steps <= 8:
            raise ValueError("--max-steps must be in [1, 8]")
        if args.device != "cpu":
            raise ValueError(
                "--device must be cpu in the bounded production runner"
            )
        if args.artifact is not None:
            # Banking77 keeps its own explicit, reproducible seed and its frozen
            # manifest pin; the synthetic preregistered seed is never applied
            # to real-data training.
            if args.seed is None:
                raise ValueError("Banking77 execution requires an explicit --seed")
            result = run_banking77_cycle(
                args.output,
                args.artifact,
                seed=args.seed,
                max_steps=args.max_steps,
                manifest_sha256=args.manifest_sha256,
                device=args.device,
            )
            data_kind = "banking77"
            data_manifest_sha256, tokenizer_name = result_data_provenance(result)
            if data_manifest_sha256 != args.manifest_sha256:
                raise ValueError("terminal evidence manifest provenance mismatch")
        else:
            seed = SYNTHETIC_V3_SEED if args.seed is None else args.seed
            if seed != SYNTHETIC_V3_SEED:
                raise ValueError(
                    f"synthetic execution requires preregistered seed {SYNTHETIC_V3_SEED}"
                )
            result = run_bounded_cycle(
                args.output,
                seed=seed,
                max_steps=args.max_steps,
                device=args.device,
            )
            data_kind = "synthetic"
            data_manifest_sha256 = None
            tokenizer_name = "synthetic-packed-v1"
        payload = {
            "data_kind": data_kind,
            "tokenizer_name": tokenizer_name,
            "decision": result.decision,
            "generation_id": result.generation_id,
            "report_path": result.report_path,
            "manifest_path": result.manifest_path,
            "candidate_checkpoint_path": result.candidate_checkpoint_path,
            "holdout_evidence_path": result.holdout_evidence_path,
            "incumbent_macro_ce": result.incumbent_macro_ce,
            "candidate_macro_ce": result.candidate_macro_ce,
            "ce_regression": result.ce_regression,
            "worst_source_regression": result.worst_source_regression,
            "mechanism_supported": result.mechanism_supported,
            "quality_supported": result.quality_supported,
            "security_supported": result.security_supported,
            "production_promotion": result.production_promotion,
            "pareto_improvement": result.pareto_improvement,
        }
        if data_manifest_sha256 is not None:
            payload["data_manifest_sha256"] = data_manifest_sha256
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"recursive growth failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
