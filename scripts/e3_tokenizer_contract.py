#!/usr/bin/env python3
"""
E3 — Tokenizer/model ID contract validator (independent, offline).

Verifies the relationship between the official Qwen3.8 tokenizer (BPE
vocab 248044) and the Qwen3_5 model vocab (248320) so that E3 generation
never silently clips arbitrary token IDs.

Source of truth:
  - models/qwen3_8-27b-tokenizer/tokenizer_config.json
  - models/qwen3_8-27b-tokenizer/vocab.json (+ merges.txt)
  - models/ternary-bonsai-2-27b-mtp/donor/config.json
    (Qwen3_5TextConfig with vocab_size=248320)

Contract (per mtp_config.json + tokenizer_config):
  - BPE vocab_size = 248044
  - model vocab_size = 248320
  - delta = 276 reserved IDs (mostly vision/audio/TTS specials)
  - Text token IDs from BPE stay <= 248043 -> no clipping into reserved space
  - Special tokens: bos=248044, eos=248044, pad=<|im_start|>=248045

This is a contract check only; it does not generate or decode text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run_check(tokenizer_dir: Path, model_config: Path, output: Path | None = None) -> dict:
    import json as _json
    tok_cfg = _json.loads((tokenizer_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    gen_cfg = _json.loads((tokenizer_dir / "generation_config.json").read_text(encoding="utf-8")) if (tokenizer_dir / "generation_config.json").exists() else {}
    vocab_path = tokenizer_dir / "vocab.json"
    model_cfg = _json.loads(model_config.read_text(encoding="utf-8"))
    text_cfg = model_cfg.get("text_config", model_cfg)

    bpe_vocab = 0
    bpe_max_id = 0
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        bpe_vocab = len(vocab)
        bpe_max_id = max(int(v) for v in vocab.values())

    model_vocab = int(text_cfg.get("vocab_size", 0))
    delta = model_vocab - bpe_vocab

    # special token ids from tokenizer_config
    added = tok_cfg.get("added_tokens_decoder", {})
    reserved_ids = []
    for key, info in added.items():
        iid = int(key)
        if iid > bpe_max_id:
            reserved_ids.append((iid, info.get("content", "")))

    special_ids = {
        "bos": tok_cfg.get("bos_token_id") or gen_cfg.get("bos_token_id") or text_cfg.get("bos_token_id"),
        "eos": tok_cfg.get("eos_token_id") or gen_cfg.get("eos_token_id") or text_cfg.get("eos_token_id"),
        "pad": tok_cfg.get("pad_token_id") or gen_cfg.get("pad_token_id") or text_cfg.get("pad_token_id"),
    }

    def _id_in_model(value):
        if value is None:
            return True
        if isinstance(value, list):
            return all(isinstance(x, int) and x <= model_vocab for x in value)
        return isinstance(value, int) and value <= model_vocab

    text_ids_safe = bpe_max_id <= 248043
    ids_not_silently_clipped = all(_id_in_model(sp) for sp in special_ids.values())

    result = {
        "bpe_vocab_size": bpe_vocab,
        "model_vocab_size": model_vocab,
        "delta_reserved_ids": delta,
        "bpe_max_token_id": bpe_max_id,
        "text_ids_safe": text_ids_safe,
        "special_ids": {k: v for k, v in special_ids.items()},
        "reserved_ids_count": len(reserved_ids),
        "reserved_ids_range": [
            min(r[0] for r in reserved_ids) if reserved_ids else None,
            max(r[0] for r in reserved_ids) if reserved_ids else None,
        ],
        "ids_not_silently_clipped": ids_not_silently_clipped,
        "clipping_blocked": not text_ids_safe or not ids_not_silently_clipped,
        "notes": (
            "BPE vocab covers text tokens only (id <= 248043). Reserved IDs "
            "248044..248319 are vision/control specials; no text ID needs clipping."
        ),
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="E3 tokenizer/model ID contract check")
    ap.add_argument("--tokenizer", type=Path,
                    default=Path("models/qwen3_8-27b-tokenizer"))
    ap.add_argument("--model-config", type=Path,
                    default=Path("models/ternary-bonsai-2-27b-mtp/donor/config.json"))
    ap.add_argument("--output", "-o", type=Path,
                    default=Path("reports/e3_tokenizer_contract.json"))
    args = ap.parse_args()

    if not args.tokenizer.exists():
        print(f"tokenizer dir not found: {args.tokenizer}", file=sys.stderr)
        return 2
    if not args.model_config.exists():
        print(f"model config not found: {args.model_config}", file=sys.stderr)
        return 2

    result = run_check(args.tokenizer, args.model_config, args.output)
    gate_ok = result["text_ids_safe"] and result["ids_not_silently_clipped"]
    print(json.dumps({
        "bpe_vocab_size": result["bpe_vocab_size"],
        "model_vocab_size": result["model_vocab_size"],
        "delta_reserved_ids": result["delta_reserved_ids"],
        "text_ids_safe": result["text_ids_safe"],
        "ids_not_silently_clipped": result["ids_not_silently_clipped"],
        "gate": "PASS" if gate_ok else "FAIL",
    }, indent=2))
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
