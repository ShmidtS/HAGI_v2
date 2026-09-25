#!/usr/bin/env python3
"""Evaluate a HAGI checkpoint on identical decoded UTF-8 text.

Native-token cross-entropy is not comparable across tokenizers. This runner
therefore keeps the exact source text as the common reference, scores it with
each lane's native tokenization, and aggregates total negative log-likelihood
by the source UTF-8 byte count:

    native_token_ce = total_nll / token_count
    byte_perplexity = exp(total_nll / byte_count)
    bits_per_byte = total_nll / (byte_count * ln(2))

The scoring contract is adapted from maintained lm-evaluation-harness rolling
windows and Lighteval's byte-weighted corpus metric. Long text is never
truncated: retained-prefix windows score every native content token exactly
once. No EOS is appended. A declared prefix token conditions the first native
content token and contributes zero source bytes.

This is an opt-in research evaluator. It never changes the production model or
packed-data loader, and it always reports ``quality_supported=false``. A
quality verdict additionally requires a validated immutable text artifact,
matched tokenizer/model training recipes, fixed held-outs, multiple seeds and
the pre-registered experiment gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from importlib import metadata
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import Config  # noqa: E402
from hagi.data.artifacts import write_json_atomic  # noqa: E402
from hagi.data.vocab_map import VocabMap  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.checkpoint import (  # noqa: E402
    config_from_dict,
    config_to_dict,
    load_model,
    load_payload,
)
from hagi.train.loop import cast_model  # noqa: E402

DEFAULT_MAX_SOURCE_BYTES = 64 * 1024 * 1024
REFERENCE_URLS = {
    "lighteval": "https://github.com/huggingface/lighteval/blob/main/src/lighteval/metrics/metrics_corpus.py",
    "lm_eval_harness": "https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/utils.py",
}


@dataclass(frozen=True)
class TokenizerLane:
    """Minimal native-tokenizer seam used by the evaluator."""

    name: str
    version: str
    vocab_size: int
    encode: Any
    decode: Any
    fingerprint: str | None = None


@dataclass(frozen=True)
class ScoredWindow:
    """One retained-prefix window and the exact native targets it scores."""

    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    scored_positions: tuple[int, ...]


def _validate_nonnegative_ids(ids: list[int]) -> None:
    if not ids:
        raise ValueError("content tokens must be non-empty")
    for value in ids:
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
            raise ValueError("token ids must be non-negative integers")


def build_scoring_windows(
    token_ids: list[int],
    prefix_token_id: int | None,
    max_seq_len: int,
    context_tokens: int,
) -> list[ScoredWindow]:
    """Build retained-prefix windows that score each target exactly once.

    The first window scores from an explicit prefix when one is supplied. Later
    windows retain ``context_tokens`` already-scored native tokens and predict
    the next ``max_seq_len - context_tokens + 1`` tokens. Inputs therefore stay
    at or below ``max_seq_len`` and every target has a causal input position.
    """
    _validate_nonnegative_ids(token_ids)
    if type(max_seq_len) is not int or max_seq_len < 2:
        raise ValueError("max_seq_len must be an integer >= 2")
    if type(context_tokens) is not int or not 1 <= context_tokens < max_seq_len:
        raise ValueError("context_tokens must be an integer in [1, max_seq_len)")
    if prefix_token_id is not None and (type(prefix_token_id) is not int or prefix_token_id < 0):
        raise ValueError("prefix_token_id must be a non-negative integer or None")

    windows: list[ScoredWindow] = []
    predicted = 0
    first_pred_len = max_seq_len
    first_pred_len = min(first_pred_len, len(token_ids))
    if first_pred_len < (1 if prefix_token_id is None else 1):
        raise ValueError("content tokens are too short for the scoring policy")

    if prefix_token_id is None:
        if len(token_ids) < 2:
            raise ValueError("content tokens are too short for prefix-free scoring")
        start_target = 1
        first_inputs = token_ids[:first_pred_len]
        first_targets = token_ids[1 : first_pred_len + 1]
    else:
        start_target = 0
        first_inputs = [prefix_token_id, *token_ids[: first_pred_len - 1]]
        first_targets = token_ids[:first_pred_len]
    if first_targets:
        windows.append(
            ScoredWindow(
                input_ids=tuple(first_inputs),
                target_ids=tuple(first_targets),
                scored_positions=tuple(range(len(first_targets))),
            )
        )
    predicted = start_target + len(first_targets)

    predicted_per_window = max_seq_len - context_tokens + 1
    while predicted < len(token_ids):
        target_end = min(len(token_ids), predicted + predicted_per_window)
        target_ids = token_ids[predicted:target_end]
        context_start = max(0, predicted - context_tokens)
        context = token_ids[context_start:predicted]
        input_ids = [*context, *target_ids[:-1]]
        scored_start = len(context) - 1
        windows.append(
            ScoredWindow(
                input_ids=tuple(input_ids),
                target_ids=tuple(target_ids),
                scored_positions=tuple(range(scored_start, len(input_ids))),
            )
        )
        predicted = target_end
    return windows


def make_metrics(*, total_nll: float, token_count: int, text: str) -> dict[str, float | int]:
    """Aggregate summed native NLL using one exact UTF-8 byte denominator."""
    if not math.isfinite(total_nll) or total_nll < 0:
        raise ValueError("total_nll must be finite and non-negative")
    if type(token_count) is not int or token_count < 1:
        raise ValueError("token_count must be a positive integer")
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    byte_count = len(text.encode("utf-8"))
    if byte_count < 1:
        raise ValueError("text must contain at least one UTF-8 byte")
    return {
        "total_nll": float(total_nll),
        "token_count": token_count,
        "byte_count": byte_count,
        "character_count": len(text),
        "native_token_ce": float(total_nll) / token_count,
        "byte_perplexity": math.exp(float(total_nll) / byte_count),
        "bits_per_byte": float(total_nll) / (byte_count * math.log(2.0)),
        "tokens_per_byte": token_count / byte_count,
    }


def _decode_native(tokenizer: TokenizerLane, ids: list[int]) -> str:
    raw = tokenizer.decode(ids)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="strict")
    if not isinstance(raw, str):
        raise ValueError("tokenizer.decode must return bytes or str")
    return raw


def _model_ids(
    native_ids: list[int],
    cfg: Config,
    vocab_map: VocabMap | None,
) -> tuple[list[int], dict[str, int | bool | None]]:
    if vocab_map is not None and vocab_map.new_vocab != cfg.model.vocab_size:
        raise ValueError(
            f"vocab map size {vocab_map.new_vocab} does not match model vocabulary {cfg.model.vocab_size}"
        )
    if vocab_map is None:
        if min(native_ids) < 0 or max(native_ids) >= cfg.model.vocab_size:
            raise ValueError("native token id is outside the model vocabulary")
        return list(native_ids), {
            "applied": False,
            "old_vocab_size": None,
            "new_vocab_size": None,
        }

    mapped = vocab_map.to_compact(native_ids).tolist()
    recovered = vocab_map.to_old(mapped).tolist()
    if recovered != native_ids:
        raise ValueError("vocab map would replace at least one source token; strict round-trip is required")
    if min(mapped) < 0 or max(mapped) >= cfg.model.vocab_size:
        raise ValueError("mapped token id is outside the model vocabulary")
    return mapped, {
        "applied": True,
        "old_vocab_size": vocab_map.old_vocab,
        "new_vocab_size": vocab_map.new_vocab,
    }


def evaluate_text(
    model: nn.Module,
    cfg: Config,
    text: str,
    tokenizer: TokenizerLane,
    *,
    context_tokens: int | None = None,
    device: str | torch.device = "cpu",
    vocab_map: VocabMap | None = None,
) -> dict[str, Any]:
    """Score one exact text through HAGI's full-alphabet receiver."""
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    if not isinstance(tokenizer, TokenizerLane):
        raise TypeError("tokenizer must be a TokenizerLane")
    model.eval()
    target_device = torch.device(device)
    native_raw = tokenizer.encode(text)
    native_ids = list(native_raw)
    _validate_nonnegative_ids(native_ids)
    native_ids = [int(value) for value in native_ids]
    decoded = _decode_native(tokenizer, native_ids)
    if decoded != text:
        raise ValueError("tokenizer round-trip is not exact")
    if tokenizer.vocab_size < 1:
        raise ValueError("tokenizer vocab_size must be positive")
    if any(value >= tokenizer.vocab_size for value in native_ids):
        raise ValueError("native token id is outside tokenizer vocab_size")
    if tokenizer.name != cfg.train.tokenizer:
        raise ValueError(
            f"tokenizer {tokenizer.name!r} does not match checkpoint tokenizer {cfg.train.tokenizer!r}"
        )

    model_ids, map_metadata = _model_ids(native_ids, cfg, vocab_map)
    prefix_native: int | None = cfg.train.data.eos_token_id
    prefix_model = _model_ids([prefix_native], cfg, vocab_map)[0][0] if prefix_native is not None else None
    max_seq_len = int(cfg.model.attention.max_seq_len)
    if context_tokens is not None and type(context_tokens) is not int:
        raise ValueError("context_tokens must be an integer or None")
    resolved_context = max_seq_len // 2 if context_tokens is None else context_tokens
    windows = build_scoring_windows(model_ids, prefix_model, max_seq_len, resolved_context)
    scored_token_count = sum(len(window.target_ids) for window in windows)
    if scored_token_count < 1:
        raise ValueError("scoring policy selected no content tokens")

    total_nll = 0.0
    with torch.no_grad():
        for window in windows:
            input_ids = torch.tensor([window.input_ids], dtype=torch.long, device=target_device)
            targets = torch.tensor(window.target_ids, dtype=torch.long, device=target_device)
            if len(window.scored_positions) != len(window.target_ids):
                raise ValueError("scoring positions and target count differ")
            output = model(input_ids)
            selected = output.hidden[0, list(window.scored_positions)]
            row_nll = model.head.exact_loss(selected, targets)
            value = float(row_nll)
            if not math.isfinite(value):
                raise ValueError("model produced non-finite exact NLL")
            total_nll += value * len(window.target_ids)

    metrics = make_metrics(total_nll=total_nll, token_count=scored_token_count, text=text)
    return {
        **metrics,
        "scored_token_count": scored_token_count,
        "unscored_leading_token_count": 1 if prefix_native is None else 0,
        "round_trip_exact": True,
        "tokenizer": {
            "name": tokenizer.name,
            "version": tokenizer.version,
            "vocab_size": tokenizer.vocab_size,
            "fingerprint": tokenizer.fingerprint,
        },
        "vocab_map": map_metadata,
        "scoring_policy": {
            "max_seq_len": max_seq_len,
            "context_tokens": resolved_context,
            "window_count": len(windows),
            "prefix_token_id": prefix_model,
            "prefix_kind": "checkpoint_eos" if prefix_model is not None else "none",
            "bos_added": False,
            "appended_eos": False,
            "truncation": False,
            "exact_full_alphabet_head": True,
            "sampled_softmax_forbidden": True,
        },
    }


def create_gigatoken_adapter(
    name: str,
    *,
    module=None,
    version: str | None = None,
) -> TokenizerLane:
    """Build a lossless native Gigatoken lane using the verified 0.10 API."""
    if not isinstance(name, str) or not name:
        raise ValueError("tokenizer name must be a non-empty string")
    if module is None:
        try:
            import gigatoken as module
        except ImportError as exc:
            raise RuntimeError("gigatoken is required for the production tokenizer lane") from exc
    if version is None:
        try:
            version = metadata.version("gigatoken")
        except metadata.PackageNotFoundError:
            version = "unknown"

    native = module.Tokenizer(name)
    vocab_size = int(native.vocab_size)
    if vocab_size < 1:
        raise ValueError("tokenizer vocab_size must be positive")

    def encode(text: str) -> list[int]:
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")
        rows = native.encode_batch_list([text])
        if len(rows) != 1:
            raise ValueError("tokenizer returned a different number of rows")
        return [int(value) for value in rows[0]]

    def decode(ids: list[int]) -> str:
        raw = native.decode(ids)
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="strict")
        if not isinstance(raw, str):
            raise ValueError("gigatoken.decode returned neither bytes nor str")
        return raw

    return TokenizerLane(
        name=name,
        version=str(version),
        vocab_size=vocab_size,
        encode=encode,
        decode=decode,
        fingerprint=None,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_sha256(cfg: Config) -> str:
    payload = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_model(cfg: Config) -> nn.Module:
    if cfg.merge.enabled:
        from hagi.model.merge import MergedHAGI

        return MergedHAGI(cfg, n_mixers=1, mixer_init_scale=cfg.merge.mixer_init_scale)
    return HAGI(cfg)


def evaluate_checkpoint(
    checkpoint: str | Path,
    text_file: str | Path,
    tokenizer: TokenizerLane,
    *,
    expected_source_sha256: str,
    device: str | torch.device = "cpu",
    context_tokens: int | None = None,
    vocab_map_path: str | Path | None = None,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    artifact_manifest_validated: bool = False,
) -> dict[str, Any]:
    """Load one checkpoint from its own config and emit a fail-closed report."""
    checkpoint_path = Path(checkpoint)
    source_path = Path(text_file)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if type(max_source_bytes) is not int or max_source_bytes < 1:
        raise ValueError("max_source_bytes must be a positive integer")
    if type(artifact_manifest_validated) is not bool:
        raise ValueError("artifact_manifest_validated must be a bool")
    if not isinstance(expected_source_sha256, str) or len(expected_source_sha256) != 64:
        raise ValueError("expected_source_sha256 must be a 64-character hexadecimal digest")
    if any(char not in "0123456789abcdefABCDEF" for char in expected_source_sha256):
        raise ValueError("expected_source_sha256 must be a 64-character hexadecimal digest")
    if source_path.stat().st_size > max_source_bytes:
        raise ValueError("source text exceeds max_source_bytes")
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if expected_source_sha256.lower() != source_sha256:
        raise ValueError("source text sha256 mismatch")
    try:
        text = source_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("source text is not strict UTF-8") from exc
    if not text:
        raise ValueError("source text must be non-empty")

    payload = load_payload(checkpoint_path, "cpu")
    cfg = config_from_dict(payload["config"])
    model = _build_model(cfg)
    completed_steps, stored_cfg = load_model(checkpoint_path, model, "cpu")
    if _config_sha256(stored_cfg) != _config_sha256(cfg):
        raise ValueError("loaded checkpoint config differs from validated payload config")
    target_device = torch.device(device)
    model = model.to(target_device)
    cast_model(
        model,
        cfg.train.precision,
        ternary_fp32_master=cfg.train.ternary_fp32_master,
    )
    vocab_map = VocabMap(vocab_map_path) if vocab_map_path is not None else None

    result = evaluate_text(
        model,
        cfg,
        text,
        tokenizer,
        context_tokens=context_tokens,
        device=target_device,
        vocab_map=vocab_map,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "execution_completed": True,
        "quality_supported": False,
        "promotion_status": "research-only",
        "source": {
            "path": str(source_path.resolve()),
            "sha256": source_sha256,
            "byte_count": len(source_bytes),
            "strict_utf8": True,
            "artifact_manifest_validated": artifact_manifest_validated,
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256_file(checkpoint_path),
            "format_version": int(payload["format_version"]),
            "completed_steps": completed_steps,
            "config_sha256": _config_sha256(cfg),
            "precision": cfg.train.precision,
            "ternary_fp32_master": cfg.train.ternary_fp32_master,
            "model_vocab_size": cfg.model.vocab_size,
            "model_max_seq_len": cfg.model.attention.max_seq_len,
            "merged": cfg.merge.enabled,
        },
        "method": {
            "common_reference": "exact_decoded_utf8_text",
            "formula": {
                "native_token_ce": "total_nll / token_count",
                "byte_perplexity": "exp(total_nll / byte_count)",
                "bits_per_byte": "total_nll / (byte_count * ln(2))",
            },
            "references": REFERENCE_URLS,
            "limitations": [
                "this evaluator alone does not establish tokenizer or model quality",
                "different tokenizers use different native conditioning contexts",
                "a quality verdict requires matched training and pre-registered multi-seed gates",
            ],
        },
        **result,
    }
    json.dumps(report, sort_keys=True, allow_nan=False)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-version")
    parser.add_argument("--vocab-map")
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if len(args.expected_source_sha256) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in args.expected_source_sha256
    ):
        parser.error("--expected-source-sha256 must be a 64-character hexadecimal digest")
    if args.max_source_bytes < 1:
        parser.error("--max-source-bytes must be positive")
    if args.context_tokens is not None and args.context_tokens < 1:
        parser.error("--context-tokens must be positive")
    args.quality_supported = False
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tokenizer = create_gigatoken_adapter(args.tokenizer, version=args.tokenizer_version)
    report = evaluate_checkpoint(
        args.checkpoint,
        args.text_file,
        tokenizer,
        expected_source_sha256=args.expected_source_sha256,
        device=args.device,
        context_tokens=args.context_tokens,
        vocab_map_path=args.vocab_map,
        max_source_bytes=args.max_source_bytes,
    )
    if args.output:
        write_json_atomic(args.output, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
