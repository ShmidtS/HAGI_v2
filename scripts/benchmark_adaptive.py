"""End-to-end adaptive inference benchmark for native HAGI checkpoints.

The benchmark uses the same prompts and generation settings for:
  B0 baseline      : main checkpoint directly
  B1 router-only   : lightweight route decision only
  B2 adaptive      : router + route registry + verifier + fallback

A specialist checkpoint can be supplied with --specialist_code/--specialist_math/...
No speed or quality claim is made unless this script is run on the target hardware.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gigatoken as gt
import torch

from hagi.inference import (
    AdaptiveInferenceController, InferenceResult, ParameterMap, Route,
    RouteCandidate,
)
from hagi.inference.generate import generate
from hagi.inference.model_pool import CheckpointModelPool
from hagi.inference.router import HeuristicRouter
from hagi.inference.verifier import GenerationVerifier
from infer_adaptive import load_main


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def read_prompts(path: str | None, inline: list[str] | None) -> list[str]:
    if inline:
        return inline
    if path:
        lines = [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines()]
        return [x for x in lines if x]
    return [
        "write a python function that parses JSON safely",
        "prove the derivative of x^2",
        "объясни принцип работы attention",
        "напиши SQL запрос с группировкой",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prompts_file")
    ap.add_argument("--prompt", action="append")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--specialist_code")
    ap.add_argument("--specialist_math")
    ap.add_argument("--specialist_ru")
    ap.add_argument("--specialist_en")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    main_model, main_cfg = load_main(Path(args.checkpoint), device)
    pool = CheckpointModelPool(device, max_resident=2)
    pool.register_loaded("main", main_model, main_cfg)

    candidates = [RouteCandidate(Route.MAIN, "*", float("inf"), 1.0, target_id="main")]
    for domain, path in (
        ("CODE", args.specialist_code),
        ("MATH", args.specialist_math),
        ("RU", args.specialist_ru),
        ("EN", args.specialist_en),
    ):
        if path:
            key = domain.lower()
            pool.register(key, path)
            candidates.insert(0, RouteCandidate(
                Route.SPECIALIST, domain, float("inf"), 0.75, target_id=key,
            ))

    router = HeuristicRouter()
    verifier = GenerationVerifier()
    controller = AdaptiveInferenceController(
        router,
        ParameterMap(candidates),
        lambda req, cand: generate_result(req, cand, pool, main_cfg, args),
        verifier,
        fallback=lambda req, prev: generate_result(
            req,
            RouteCandidate(Route.MAIN, "*", float("inf"), 1.0, target_id="main"),
            pool,
            main_cfg,
            args,
        ),
    )

    prompts = read_prompts(args.prompts_file, args.prompt)
    tokenizer = gt.Tokenizer(main_cfg.train.tokenizer)
    vocab_map = None
    map_path = Path(main_cfg.train.data.data_dir) / "vocab_map.npz"
    if map_path.exists() and main_cfg.model.vocab_size < 262144:
        from hagi.data.vocab_map import VocabMap
        vocab_map = VocabMap(map_path)
    prompt_ids_cache = {}
    for p in prompts:
        raw = tokenizer.encode(p)
        ids = list(raw) if not isinstance(raw, list) else raw
        if vocab_map is not None:
            ids = vocab_map.to_compact(ids).tolist()
        prompt_ids_cache[p] = torch.tensor([ids], dtype=torch.long, device=device)

    # Router-only measurement.
    route_samples = []
    for _ in range(max(1, args.repeats) * len(prompts)):
        for p in prompts:
            t0 = time.perf_counter()
            router.predict(p)
            route_samples.append((time.perf_counter() - t0) * 1e6)
    print(f"B1 router_us_p50={statistics.median(route_samples):.2f} router_us_p95={percentile(route_samples, .95):.2f}")

    def run_baseline() -> list[tuple[float, int]]:
        rows = []
        for _ in range(max(1, args.repeats)):
            for p in prompts:
                t0 = time.perf_counter()
                out = generate(
                    main_model,
                    prompt_ids_cache[p],
                    max_new_tokens=args.max_tokens,
                    eos_token_id=main_cfg.train.data.eos_token_id,
                    pad_token_id=main_cfg.train.data.pad_token_id,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
                elapsed = (time.perf_counter() - t0) * 1000.0
                rows.append((elapsed, int(out.lengths[0])))
        return rows

    def run_adaptive() -> list[tuple[float, int, str]]:
        rows = []
        for _ in range(max(1, args.repeats)):
            for p in prompts:
                result, trace = controller.generate(
                    {"prompt": p, "input_ids": prompt_ids_cache[p]},
                )
                rows.append((trace.elapsed_ms, len(result.output), trace.selected_route.value))
        return rows

    baseline = run_baseline()
    adaptive = run_adaptive()
    b_times = [x[0] for x in baseline]
    a_times = [x[0] for x in adaptive]
    b_tok = sum(x[1] for x in baseline) / max(sum(x[0] for x in baseline) / 1000.0, 1e-9)
    a_tok = sum(x[1] for x in adaptive) / max(sum(x[0] for x in adaptive) / 1000.0, 1e-9)
    routes = {}
    for _, _, route in adaptive:
        routes[route] = routes.get(route, 0) + 1

    print(
        f"B0 baseline_ms_p50={statistics.median(b_times):.2f} "
        f"baseline_ms_p95={percentile(b_times, .95):.2f} "
        f"baseline_gen_tok_s={b_tok:.2f}"
    )
    print(
        f"B2 adaptive_ms_p50={statistics.median(a_times):.2f} "
        f"adaptive_ms_p95={percentile(a_times, .95):.2f} "
        f"adaptive_gen_tok_s={a_tok:.2f} routes={routes}"
    )
    print("Note: this benchmark measures latency/throughput, not semantic quality. Pair it with eval_domains.py/golden tests.")
    return 0


def generate_result(req, candidate, pool, fallback_cfg, args):
    key = candidate.target_id or "main"
    model = pool.get(key)
    cfg = pool.config(key)
    if cfg.model.vocab_size != fallback_cfg.model.vocab_size:
        raise ValueError(f"route {key!r} has incompatible vocab size")
    out = generate(
        model,
        req["input_ids"],
        max_new_tokens=args.max_tokens,
        eos_token_id=cfg.train.data.eos_token_id,
        pad_token_id=cfg.train.data.pad_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    generated = out.token_ids[0, req["input_ids"].shape[1]:].tolist()
    return InferenceResult(generated, candidate.route, metadata={"route_target": key})


if __name__ == "__main__":
    raise SystemExit(main())
