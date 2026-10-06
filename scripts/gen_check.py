"""Batch generation-quality check across checkpoints.

Loads each checkpoint ONCE, runs a fixed prompt battery (greedy +
sampled), writes a markdown report so checkpoints can be compared
side by side. The inference stack is the same as scripts/infer.py
(banned ids, compact vocab map, KV-cache generate).

Usage:
    python scripts/gen_check.py --out logs/gen_check_report.md \
        checkpoints/dbridge_gen7_joint/best.pt \
        checkpoints/dbridge_gen6_joint/step-0001300.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import os

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import numpy as np
import torch

from hagi.inference.generate import generate
from hagi.train.checkpoint import config_from_dict, load_payload
from hagi.train.loop import cast_model
from hagi.model.merge import build_model_from_payload

try:
    import gigatoken as gt
except ImportError:
    print("Требуется gigatoken: pip install gigatoken")
    sys.exit(1)

# Домены: EN chat, RU chat, factual, math, code, story.
PROMPTS = [
    ("chat-en", "Hello! How are you today?"),
    ("chat-ru", "Привет! Расскажи, что ты умеешь."),
    ("fact", "The capital of France is"),
    ("math", "2 + 2 = "),
    ("code", "def fibonacci(n):"),
    ("story", "Once upon a time"),
]


def decode_text(tokenizer, ids, vocab_map) -> str:
    if vocab_map is not None:
        ids = vocab_map.to_old(ids).tolist()
    raw = tokenizer.decode(ids)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+", help="ckpt paths, in report column order")
    ap.add_argument("--out", type=Path, default=Path("logs/gen_check_report.md"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")

    ckpt_paths = [Path(c) for c in args.checkpoints]
    reports: dict[str, dict[tuple[str, str], str]] = {}

    for ckpt_path in ckpt_paths:
        print(f"\n=== loading {ckpt_path} ===")
        payload = load_payload(ckpt_path, device)
        cfg = config_from_dict(payload["config"])
        model = build_model_from_payload(
            cfg, payload["model"], n_mixers=1,
            mixer_init_scale=cfg.merge.mixer_init_scale, device=device,
        )
        model.load_state_dict(payload["model"], strict=True)
        cast_model(
            model, cfg.train.precision,
            ternary_fp32_master=cfg.train.ternary_fp32_master,
        )
        model.eval()
        vocab = cfg.model.vocab_size

        banned_ids: set[int] = set()
        if cfg.model.head.unigram_prior and cfg.model.head.unigram_path:
            unigram = np.load(cfg.model.head.unigram_path)
            for cid, count in enumerate(unigram):
                if count == 0:
                    banned_ids.add(cid)
        banned_ids.update(cid for cid in (0, 2, 3, 4, 5) if cid < vocab)
        if vocab >= 256:
            banned_ids.update(range(6, min(256, vocab)))

        vocab_map = None
        map_path = Path(cfg.train.data.data_dir) / "vocab_map.npz"
        if map_path.exists() and vocab < 262144:
            from hagi.data.vocab_map import VocabMap
            vocab_map = VocabMap(map_path)

        tokenizer = gt.Tokenizer(cfg.train.tokenizer)
        max_seq_len = cfg.model.attention.max_seq_len
        budget = max_seq_len - args.max_tokens

        def encode(text: str) -> list[int]:
            raw = tokenizer.encode(text)
            old = list(raw) if not isinstance(raw, list) else raw
            if vocab_map is not None:
                return vocab_map.to_compact(old).tolist()
            return old

        def run(prompt_ids: list[int], temperature: float) -> str:
            context = prompt_ids[-budget:]
            t = torch.tensor([context], dtype=torch.long, device=device)
            out = generate(
                model, t,
                max_new_tokens=args.max_tokens,
                eos_token_id=cfg.train.data.eos_token_id,
                pad_token_id=cfg.train.data.pad_token_id,
                temperature=temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                banned=tuple(sorted(banned_ids)),
            )
            full = out.token_ids[0].tolist()
            return decode_text(tokenizer, full[len(context):], vocab_map)

        label = f"{ckpt_path.parent.name}/{ckpt_path.name}"
        reports[label] = {}
        for tag, prompt in PROMPTS:
            ids = encode(prompt)
            for mode, temp in (("greedy", 0.0), ("t0.8", args.temperature)):
                text = run(ids, temp)
                reports[label][(tag, mode)] = text
                print(f"  [{tag}/{mode}] {text[:100]!r}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # markdown report
    lines = ["# Generation quality check", "",
             f"- max_tokens={args.max_tokens}, top_k={args.top_k}, "
             f"top_p={args.top_p}, rep={args.repetition_penalty}",
             f"- checkpoints: {len(reports)}", ""]
    labels = list(reports)
    for tag, prompt in PROMPTS:
        lines += [f"## {tag}", "", f"**prompt:** `{prompt}`", ""]
        for mode in ("greedy", "t0.8"):
            lines += [f"### {mode}", ""]
            for label in labels:
                text = reports[label][(tag, mode)]
                lines += [f"**{label}**", "", "```", text, "```", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
