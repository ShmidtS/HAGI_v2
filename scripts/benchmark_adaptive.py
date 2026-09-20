"""Benchmark adaptive inference components and optional HAGI generation.

Usage examples:
  python scripts/benchmark_adaptive.py --router-only --repeats 1000
  python scripts/benchmark_adaptive.py --checkpoint ... --prompts-file prompts.txt

The model benchmark compares the same generation workload with adaptive mode
disabled/enabled. A secondary checkpoint can be supplied for specialist routing.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from hagi.inference.adaptive import (
    AdaptiveInferenceController, InferenceResult, ParameterMap, RouteCandidate, Route,
)
from hagi.inference.router import HeuristicRouter
from hagi.inference.verifier import GenerationVerifier


def bench_router(repeats: int) -> None:
    router = HeuristicRouter()
    prompts = [
        "write a python function that parses JSON",
        "prove the derivative of x^2",
        "переведи этот текст на английский",
        "объясни работу алгоритма",
    ]
    samples = []
    for i in range(repeats):
        t0 = time.perf_counter()
        router.predict(prompts[i % len(prompts)])
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    p50 = statistics.median(samples)
    p95 = samples[min(len(samples) - 1, int(0.95 * len(samples)))]
    print(f"router_us_p50={p50:.2f} router_us_p95={p95:.2f} repeats={repeats}")


def bench_controller(repeats: int) -> None:
    router = HeuristicRouter()
    candidates = ParameterMap((
        RouteCandidate(Route.SPECIALIST, "CODE", 1.0, 0.25),
        RouteCandidate(Route.MAIN, "*", 10.0, 1.0),
    ))
    verifier = GenerationVerifier()

    def execute(req, candidate):
        return InferenceResult("ok", candidate.route)

    controller = AdaptiveInferenceController(router, candidates, execute, verifier)
    samples = []
    for i in range(repeats):
        t0 = time.perf_counter()
        controller.generate("python function for parsing JSON")
        samples.append((time.perf_counter() - t0) * 1e6)
    p95 = sorted(samples)[min(len(samples) - 1, int(0.95 * len(samples)))]
    print(
        f"controller_us_median={statistics.median(samples):.2f} "
        f"controller_us_p95={p95:.2f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-only", action="store_true")
    ap.add_argument("--repeats", type=int, default=1000)
    args = ap.parse_args()
    bench_router(args.repeats)
    if not args.router_only:
        bench_controller(max(100, args.repeats // 10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
