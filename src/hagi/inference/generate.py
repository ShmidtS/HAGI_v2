"""Causal autoregressive generation with an incremental KV-cache.

Standard GPT-style left-to-right token generation matching the causal training
mode. The model sees the REAL context at every step — nothing is erased.

V27 uses the per-layer KV-cache (rehabilitated from V23's never-wired
``kv_cache.py``): the prompt is prefetched once (KV cached), then each new
token is a single-position forward pass that reuses the frozen prefix. This is
O(T) generation instead of V25's O(T^2) full-recompute-per-step (the
convolutional-code decoder-state analogy). The cache is allocated, used, and
always detached.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GenerationOutput:
    """Padded generated sequences and their per-row semantic lengths."""

    token_ids: torch.LongTensor
    generated_lengths: torch.LongTensor
    finished: torch.BoolTensor


def _validate_token_ids(name: str, token_ids: tuple[int, ...], vocab_size: int) -> None:
    if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
        raise ValueError(f"{name} must contain only in-vocabulary IDs")


def process_generation_logits(
    logits: torch.Tensor,
    context_ids: torch.Tensor,
    *,
    generated_lengths: torch.Tensor,
    forbidden_token_ids: tuple[int, ...],
    eos_token_id: int,
    min_new_tokens: int,
    repetition_penalty: float,
    repetition_window: int,
    no_repeat_ngram_size: int,
    temperature: float,
    top_k: int,
    generator: torch.Generator | None = None,
) -> torch.LongTensor:
    """Apply generation constraints in one ordered path and select one token per row."""
    if logits.ndim != 2 or context_ids.ndim != 2 or logits.shape[0] != context_ids.shape[0]:
        raise ValueError("logits and context_ids must be row-aligned rank-2 tensors")
    vocab_size = logits.shape[-1]
    if context_ids.dtype != torch.long:
        raise ValueError("context_ids must be a LongTensor")
    _validate_token_ids("context_ids", tuple(context_ids.flatten().tolist()), vocab_size)
    _validate_token_ids("forbidden_token_ids", forbidden_token_ids, vocab_size)
    _validate_token_ids("eos_token_id", (eos_token_id,), vocab_size)
    if generated_lengths.shape != (logits.shape[0],):
        raise ValueError("generated_lengths must contain one value per logits row")
    if generated_lengths.dtype != torch.long:
        raise ValueError("generated_lengths must be a LongTensor")
    if (generated_lengths < 0).any():
        raise ValueError("generated_lengths must be non-negative")
    if not isinstance(min_new_tokens, int) or isinstance(min_new_tokens, bool) or min_new_tokens < 0:
        raise ValueError("min_new_tokens must be a non-negative integer")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if repetition_window < 1:
        raise ValueError("repetition_window must be positive")
    if no_repeat_ngram_size < 0 or temperature < 0 or top_k < 0:
        raise ValueError("no_repeat_ngram_size, temperature, and top_k must be non-negative")

    processed = logits.float().clone()
    if forbidden_token_ids:
        processed[:, list(forbidden_token_ids)] = float("-inf")
    processed[generated_lengths + 1 < min_new_tokens, eos_token_id] = float("-inf")

    if repetition_penalty != 1.0 and context_ids.numel() > 0:
        window = context_ids[:, -repetition_window:]
        for row in range(processed.shape[0]):
            repeated = window[row].unique()
            scores = processed[row, repeated]
            processed[row, repeated] = torch.where(
                scores < 0, scores * repetition_penalty, scores / repetition_penalty,
            )

    n = no_repeat_ngram_size
    if n > 0 and context_ids.shape[1] + 1 >= n:
        for row in range(processed.shape[0]):
            tokens = context_ids[row].tolist()
            prefix = tuple(tokens[-(n - 1):]) if n > 1 else ()
            banned = {
                tokens[index + n - 1]
                for index in range(len(tokens) - n + 1)
                if tuple(tokens[index: index + n - 1]) == prefix
            }
            if banned:
                processed[row, list(banned)] = float("-inf")

    if temperature > 0:
        processed = processed / temperature
    if top_k > 0:
        threshold = torch.topk(processed, min(top_k, vocab_size), dim=-1).values[:, -1:]
        processed = torch.where(processed < threshold, float("-inf"), processed)
    if torch.isneginf(processed).all(dim=-1).any():
        raise ValueError("generation constraints banned every token for at least one row")
    if temperature == 0:
        return processed.argmax(dim=-1)
    return torch.multinomial(F.softmax(processed, dim=-1), 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
    forbidden_token_ids: tuple[int, ...] = (),
    min_new_tokens: int = 2,
    temperature: float = 0.8,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    repetition_window: int = 64,
    no_repeat_ngram_size: int = 2,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> GenerationOutput:
    """Pure causal autoregressive generation with an incremental KV-cache.

    Prefetch: the prompt is run once with the cache attached (prefilling all
    prompt positions). Then each new token is a single-position forward pass
    that appends its KV to the cache and attends to the full frozen prefix.

    Set ``use_cache=False`` to fall back to full-recompute-per-step (O(T^2)).
    """
    if prompt_ids.ndim != 2 or prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must be a rank-2 LongTensor")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
        raise ValueError("0 <= min_new_tokens <= max_new_tokens required")

    model.eval()
    batch_size, prompt_len = prompt_ids.shape
    device = prompt_ids.device
    vocab_size = model.cfg.model.vocab_size
    _validate_token_ids("prompt_ids", tuple(prompt_ids.flatten().tolist()), vocab_size)
    _validate_token_ids("eos_token_id", (eos_token_id,), vocab_size)
    _validate_token_ids("pad_token_id", (pad_token_id,), vocab_size)
    _validate_token_ids("forbidden_token_ids", forbidden_token_ids, vocab_size)
    if eos_token_id == pad_token_id:
        raise ValueError("eos_token_id and pad_token_id must be distinct")

    invalid_token_ids = tuple(dict.fromkeys((*forbidden_token_ids, pad_token_id)))

    sequence = torch.cat(
        (prompt_ids, torch.full((batch_size, max_new_tokens), pad_token_id, dtype=torch.long, device=device)),
        dim=1,
    )
    generated_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    caches = None
    if use_cache and hasattr(model, "allocate_for_cache"):
        caches = model.allocate_for_cache(batch_size, next(model.parameters()).dtype, device)

    try:
        # Prefetch the prompt once: cache holds KV for all prompt positions.
        valid_target_mask = torch.ones((batch_size, prompt_len), dtype=torch.bool, device=device)
        output = model(
            prompt_ids, targets=None, prediction_mask=valid_target_mask,
            valid_target_mask=valid_target_mask, attention_mode="causal",
        )
        if output.logits is None:
            raise ValueError("model output must include logits")
        next_logits = output.logits.view(batch_size, prompt_len, -1)[:, -1, :]

        for r in range(batch_size):
            if finished[r]:
                next_logits[r] = float("-inf")
                next_logits[r, pad_token_id] = 0.0

        for step in range(max_new_tokens):
            if finished.all():
                break

            context = sequence[:, : prompt_len + step]
            next_tokens = process_generation_logits(
                next_logits, context,
                generated_lengths=generated_lengths, forbidden_token_ids=invalid_token_ids,
                eos_token_id=eos_token_id, min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty, repetition_window=repetition_window,
                no_repeat_ngram_size=no_repeat_ngram_size, temperature=temperature,
                top_k=top_k, generator=generator,
            )

            for r in range(batch_size):
                if finished[r]:
                    continue
                token = int(next_tokens[r].item())
                sequence[r, prompt_len + step] = token
                generated_lengths[r] += 1
                if token == eos_token_id:
                    finished[r] = True

            if step + 1 >= max_new_tokens or finished.all():
                break

            # Single-position forward pass that reuses the cached prefix.
            new_pos = prompt_len + step
            new_token = sequence[:, new_pos:new_pos + 1]
            if use_cache and caches is not None:
                pos_tensor = torch.tensor([new_pos], device=device)
                output = model(
                    new_token, targets=None, attention_mode="causal", positions=pos_tensor,
                )
            else:
                full_input = sequence[:, : new_pos + 1]
                vtm = torch.ones((batch_size, full_input.shape[1]), dtype=torch.bool, device=device)
                output = model(
                    full_input, targets=None, prediction_mask=vtm, valid_target_mask=vtm,
                    attention_mode="causal",
                )
            if output.logits is None:
                raise ValueError("model output must include logits")
            next_logits = output.logits.view(batch_size, -1, vocab_size)[:, -1, :]
            for r in range(batch_size):
                if finished[r]:
                    next_logits[r] = float("-inf")
                    next_logits[r, pad_token_id] = 0.0
    finally:
        if caches is not None and hasattr(model, "reset_cache"):
            model.reset_cache()

    token_ids = sequence[:, : prompt_len + max_new_tokens]
    return GenerationOutput(token_ids=token_ids, generated_lengths=generated_lengths, finished=finished)
