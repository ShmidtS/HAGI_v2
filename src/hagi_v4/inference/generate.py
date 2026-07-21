"""V20: Pure causal autoregressive generation.

Replaces the V12 LLaDA-style mask-predict loop with standard GPT-style
left-to-right token generation. The model is now trained with UniLM-style
mixed attention (40% bidir, 30% prefix-LM, 30% causal), so the inference
path uses the causal mode that was explicitly trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hagi_v4.model.codec_contracts import InferenceShapeConfig


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
                scores < 0,
                scores * repetition_penalty,
                scores / repetition_penalty,
            )

    n = no_repeat_ngram_size
    if n > 0 and context_ids.shape[1] + 1 >= n:
        for row in range(processed.shape[0]):
            tokens = context_ids[row].tolist()
            prefix = tuple(tokens[-(n - 1) :]) if n > 1 else ()
            banned = {
                tokens[index + n - 1]
                for index in range(len(tokens) - n + 1)
                if tuple(tokens[index : index + n - 1]) == prefix
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
def _generate(
    model: torch.nn.Module,
    prompt_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    max_iterations: int,
    eos_token_id: int,
    pad_token_id: int,
    forbidden_token_ids: tuple[int, ...] = (),
    min_new_tokens: int = 2,
    temperature: float = 0.8,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    repetition_window: int = 64,
    no_repeat_ngram_size: int = 3,
    physical_corruption_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> GenerationOutput:
    """V20: Pure causal autoregressive generation.

    Feeds the prompt, takes logits at the last position, samples the next
    token, appends it, and repeats. This is the standard GPT-style AR loop
    that matches the causal training mode (30% of V20 batches).

    Unlike the V12 mask-predict loop, this does NOT use iterative
    refinement or parallel commit — each token is generated sequentially,
    conditioned on all previously generated tokens.
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
    inference_config = getattr(model, "inference_config", None)
    vocab_size = (
        inference_config.vocab_size
        if inference_config is not None
        else InferenceShapeConfig.from_hagi_config(model.cfg).vocab_size
    )
    _validate_token_ids("prompt_ids", tuple(prompt_ids.flatten().tolist()), vocab_size)
    _validate_token_ids("eos_token_id", (eos_token_id,), vocab_size)
    _validate_token_ids("pad_token_id", (pad_token_id,), vocab_size)
    _validate_token_ids("forbidden_token_ids", forbidden_token_ids, vocab_size)
    if eos_token_id == pad_token_id:
        raise ValueError("eos_token_id and pad_token_id must be distinct")

    invalid_token_ids = tuple(dict.fromkeys((*forbidden_token_ids, pad_token_id)))

    # Build the sequence buffer: prompt + space for new tokens (filled with pad).
    sequence = torch.cat(
        (prompt_ids, torch.full((batch_size, max_new_tokens), pad_token_id, dtype=torch.long, device=device)),
        dim=1,
    )
    generated_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # V20: physical_corruption_mask is not used in causal AR mode (no
    # channel noise during generation). If provided, ignored.
    _ = physical_corruption_mask

    for step in range(max_new_tokens):
        if finished.all():
            break

        # Forward pass with causal attention over the current sequence.
        current_len = prompt_len + step
        input_ids = sequence[:, :current_len]
        # V20: prediction_mask selects ONLY the last position for each row
        # (next-token prediction). semantic_unknown_mask must be a superset
        # of prediction_mask (model contract), so we mark the last position
        # as unknown too. The model returns logits as [n_selected, V] where
        # n_selected = prediction_mask.sum() = batch_size.
        prediction_mask = torch.zeros((batch_size, current_len), dtype=torch.bool, device=device)
        prediction_mask[:, -1] = True
        semantic_unknown_mask = prediction_mask.clone()
        valid_target_mask = torch.ones((batch_size, current_len), dtype=torch.bool, device=device)
        physical_mask = torch.zeros((batch_size, current_len), dtype=torch.bool, device=device)
        output = model(
            input_ids,
            targets=None,
            semantic_unknown_mask=semantic_unknown_mask,
            prediction_mask=prediction_mask,
            valid_target_mask=valid_target_mask,
            physical_corruption_mask=physical_mask,
            attention_mode="causal",
            prefix_len=None,
        )
        if output.logits is None:
            raise ValueError("model output must include logits")

        # output.logits is [B, V] (one logit per row for the last position).
        next_logits = output.logits  # [B, V]

        # Zero out finished rows so they don't produce tokens.
        for r in range(batch_size):
            if finished[r]:
                next_logits[r] = float("-inf")
                next_logits[r, pad_token_id] = 0.0

        # Sample next token for each row.
        context = sequence[:, :current_len]
        next_tokens = process_generation_logits(
            next_logits,
            context,
            generated_lengths=generated_lengths,
            forbidden_token_ids=invalid_token_ids,
            eos_token_id=eos_token_id,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )

        # Write tokens and update state.
        for r in range(batch_size):
            if finished[r]:
                continue
            token = int(next_tokens[r].item())
            sequence[r, current_len] = token
            generated_lengths[r] += 1
            if token == eos_token_id:
                finished[r] = True

    token_ids = sequence[:, : prompt_len + max_new_tokens]
    return GenerationOutput(token_ids=token_ids, generated_lengths=generated_lengths, finished=finished)


def generate(
    model: torch.nn.Module,
    prompt_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    max_iterations: int,
    eos_token_id: int,
    pad_token_id: int,
    forbidden_token_ids: tuple[int, ...] = (),
    min_new_tokens: int = 2,
    temperature: float = 0.8,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    repetition_window: int = 64,
    no_repeat_ngram_size: int = 3,
    physical_corruption_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> GenerationOutput:
    training_states = tuple((module, module.training) for module in model.modules())
    try:
        return _generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            max_iterations=max_iterations,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            forbidden_token_ids=forbidden_token_ids,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            no_repeat_ngram_size=no_repeat_ngram_size,
            physical_corruption_mask=physical_corruption_mask,
            generator=generator,
        )
    finally:
        model.train(training_states[0][1])
        for module, was_training in training_states:
            module.training = was_training
