"""Adaptive interactive inference entry point for HAGI_v2.

It preserves the existing tokenizer/checkpoint conventions and adds a lightweight
router plus optional domain specialist checkpoints. The default route remains the
main model. Specialist checkpoints are explicitly supplied by the user.

Unlike the DeepSeek-V4 TTT scripts, this path operates on native HAGI checkpoints.
Online learning is intentionally a queue hook; no self-generated sample is treated
as trusted supervision automatically.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import gigatoken as gt

from hagi.inference import (
    AdaptiveInferenceController, InferenceResult, ParameterMap, Route,
    RouteCandidate,
)
from hagi.inference.generate import generate
from hagi.inference.model_pool import CheckpointModelPool
from hagi.inference.router import HeuristicRouter
from hagi.inference.verifier import GenerationVerifier
from hagi.inference.lora import HeadLoRA, HeadLoRATrainer
from hagi.model.model import HAGI
from hagi.train.checkpoint import config_from_dict, load_payload
from hagi.train.loop import cast_model

try:
    from infer import decode_text
except ImportError:
    def decode_text(tokenizer, ids, vocab_map=None):
        if vocab_map is not None:
            ids = vocab_map.to_old(torch.tensor(ids)).tolist()
        raw = tokenizer.decode(ids)
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)


def load_main(checkpoint: Path, device: torch.device):
    payload = load_payload(checkpoint, device)
    cfg = config_from_dict(payload["config"])
    model: torch.nn.Module = HAGI(cfg).to(device)
    if cfg.merge.enabled:
        from hagi.model.merge import MergedHAGI
        model = MergedHAGI(cfg, n_mixers=1, mixer_init_scale=cfg.merge.mixer_init_scale).to(device)
    state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    cast_model(model, cfg.train.precision)
    model.eval()
    return model, cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="HAGI adaptive inference")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--adaptive_shadow", action="store_true")
    ap.add_argument("--adaptive_code_checkpoint")
    ap.add_argument("--adaptive_math_checkpoint")
    ap.add_argument("--adaptive_ru_checkpoint")
    ap.add_argument("--adaptive_en_checkpoint")
    ap.add_argument("--adaptive_max_resident", type=int, default=2)
    ap.add_argument("--adaptive_budget_ms", type=float, default=float("inf"))
    ap.add_argument("--adaptive_log", action="store_true")
    ap.add_argument("--adaptive_lora_rank", type=int, default=0)
    ap.add_argument("--adaptive_lora_path")
    ap.add_argument("--adaptive_lora_save_dir", default="adaptive_lora")
    ap.add_argument("--adaptive_self_train", action="store_true", help="opt-in pseudo-label head-LoRA update after each turn")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    main_path = Path(args.checkpoint)
    main_model, cfg = load_main(main_path, device)

    tokenizer = gt.Tokenizer(cfg.train.tokenizer)
    vocab_map = None
    map_path = Path(cfg.train.data.data_dir) / "vocab_map.npz"
    if map_path.exists() and cfg.model.vocab_size < 262144:
        from hagi.data.vocab_map import VocabMap
        vocab_map = VocabMap(map_path)

    pool = CheckpointModelPool(device, max_resident=args.adaptive_max_resident)
    pool.register_loaded("main", main_model, cfg)

    lora_trainer = None
    main_lora = None
    if args.adaptive_lora_path:
        main_lora = HeadLoRA.load(args.adaptive_lora_path, cfg.model.hidden_size, cfg.model.vocab_size, device)
        main_model.head.attach_lora(main_lora)
        if args.adaptive_self_train:
            lora_trainer = HeadLoRATrainer()
    elif args.adaptive_lora_rank > 0:
        lora_trainer = HeadLoRATrainer()
        main_lora = lora_trainer.attach(main_model, rank=args.adaptive_lora_rank)

    candidates = [
        RouteCandidate(Route.MAIN, "*", float("inf"), 1.0, target_id="main"),
    ]
    for domain, path in (
        ("CODE", args.adaptive_code_checkpoint),
        ("MATH", args.adaptive_math_checkpoint),
        ("RU", args.adaptive_ru_checkpoint),
        ("EN", args.adaptive_en_checkpoint),
    ):
        if path:
            key = domain.lower()
            pool.register(key, path)
            candidates.insert(0, RouteCandidate(
                Route.SPECIALIST, domain, float("inf"), 0.75, target_id=key,
            ))

    router = HeuristicRouter()
    verifier = GenerationVerifier()
    route_map = ParameterMap(candidates)

    def execute(req, candidate):
        model_key = candidate.target_id or "main"
        model = pool.get(model_key)
        model_cfg = pool.config(model_key)
        prompt_ids = req["input_ids"]
        if model_cfg.model.vocab_size != cfg.model.vocab_size:
            raise ValueError(f"route {model_key!r} has incompatible vocab size")
        out = generate(
            model,
            prompt_ids,
            max_new_tokens=args.max_tokens,
            eos_token_id=model_cfg.train.data.eos_token_id,
            pad_token_id=model_cfg.train.data.pad_token_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        generated = out.token_ids[0, prompt_ids.shape[1]:].tolist()
        return InferenceResult(
            generated,
            candidate.route,
            metadata={"route_target": candidate.target_id},
        )

    def fallback(req, previous):
        return execute(req, RouteCandidate(Route.MAIN, "*", float("inf"), 1.0, target_id="main"))

    controller = AdaptiveInferenceController(
        router,
        route_map,
        execute,
        verifier,
        fallback=fallback,
        shadow=args.adaptive_shadow,
    )

    max_seq_len = cfg.model.attention.max_seq_len
    budget = max_seq_len - args.max_tokens
    if budget < 8:
        raise ValueError("max_tokens leaves too little prompt context")

    history: list[int] = []

    def encode(text: str) -> list[int]:
        raw = tokenizer.encode(text)
        ids = list(raw) if not isinstance(raw, list) else raw
        if vocab_map is not None:
            ids = vocab_map.to_compact(ids).tolist()
        return ids

    def turn(text: str) -> tuple[list[int], object]:
        user_ids = encode(text)
        context = (history + user_ids)[-budget:]
        prompt = torch.tensor([context], dtype=torch.long, device=device)
        result, trace = controller.generate(
            {"prompt": text, "input_ids": prompt},
            latency_budget_ms=args.adaptive_budget_ms,
        )
        history.extend(user_ids + result.output)
        return result.output, trace

    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip() or prompt.strip().lower() in {"q", "quit", "exit"}:
            break
        output, trace = turn(prompt)
        print(decode_text(tokenizer, output, vocab_map))
        if args.adaptive_self_train:
            if lora_trainer is None or main_lora is None:
                raise RuntimeError("--adaptive_self_train requires --adaptive_lora_rank > 0 or --adaptive_lora_path")
            seq = encode(prompt) + output
            if len(seq) >= 2:
                max_train = min(len(seq) - 1, 64)
                train_seq = torch.tensor([seq[-(max_train + 1):]], dtype=torch.long, device=device)
                loss = lora_trainer.step(main_model, train_seq[:, :-1], train_seq[:, 1:])
                path = lora_trainer.save_version(main_lora, args.adaptive_lora_save_dir, int(time.time() * 1000))
                if args.adaptive_log:
                    print(f"[adaptive-lora] pseudo_loss={loss:.4f} saved={path}")
        if args.adaptive_log:
            d = trace.decision
            print(
                f"[adaptive] route={trace.selected_route.value} domain={d.domain} "
                f"confidence={d.confidence:.3f} risk={d.risk:.3f} "
                f"reason={d.reason} fallback={trace.fallback_used} "
                f"latency_ms={trace.elapsed_ms:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
