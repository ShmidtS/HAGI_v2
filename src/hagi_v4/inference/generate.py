"""Masked refinement followed by adaptive left-to-right generation."""

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
    """Refine a fully unknown suffix, then sample it once from left to right."""
    if prompt_ids.ndim != 2 or prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must be a rank-2 LongTensor")
    if max_new_tokens < 2 or not 2 <= min_new_tokens <= max_new_tokens:
        raise ValueError("generation requires 2 <= min_new_tokens <= max_new_tokens")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    model.eval()
    batch_size, original_prompt_len = prompt_ids.shape
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

    internal_prompt_len = max(original_prompt_len, 4)
    prompt_padding = torch.full(
        (batch_size, internal_prompt_len - original_prompt_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    internal_prompt = torch.cat((prompt_ids, prompt_padding), dim=1)
    suffix = torch.full((batch_size, max_new_tokens), pad_token_id, dtype=torch.long, device=device)
    refinement_ids = torch.cat((internal_prompt, suffix), dim=1)
    semantic_unknown_mask = torch.zeros_like(refinement_ids, dtype=torch.bool)
    semantic_unknown_mask[:, internal_prompt_len:] = True
    if physical_corruption_mask is None:
        physical_mask = torch.zeros_like(refinement_ids, dtype=torch.bool)
    else:
        if physical_corruption_mask.shape != refinement_ids.shape:
            raise ValueError("physical_corruption_mask must match the internal generation sequence")
        physical_mask = physical_corruption_mask.to(device=device, dtype=torch.bool)

    decoder_n_iters = getattr(getattr(model, "decoder", None), "n_iters", max_iterations)
    effective_iterations = min(max(1, max_iterations), decoder_n_iters)

    posterior = None
    output = model(
        refinement_ids,
        targets=None,
        semantic_unknown_mask=semantic_unknown_mask,
        prediction_mask=semantic_unknown_mask,
        valid_target_mask=torch.ones_like(semantic_unknown_mask),
        physical_corruption_mask=physical_mask,
        refinement_iterations=effective_iterations,
    )
    if output.logits is None or output.prediction_indices is None:
        raise ValueError("model output must include logits and prediction_indices")
    if output.logits.shape != (batch_size * max_new_tokens, vocab_size):
        raise ValueError("model must return one logits row per unknown suffix position")
    posterior = output.logits.new_empty((batch_size, max_new_tokens, vocab_size), dtype=torch.float32)
    logits_rows = torch.full((refinement_ids.numel(),), -1, dtype=torch.long, device=device)
    logits_rows[output.prediction_indices.to(device)] = torch.arange(output.logits.shape[0], device=device)
    for row in range(batch_size):
        positions = torch.arange(
            row * refinement_ids.shape[1] + internal_prompt_len,
            row * refinement_ids.shape[1] + internal_prompt_len + max_new_tokens,
            device=device,
        )
        selected_rows = logits_rows.index_select(0, positions)
        if (selected_rows < 0).any():
            raise ValueError("prediction_indices omitted an unknown suffix position")
        posterior[row] = output.logits.index_select(0, selected_rows.to(output.logits.device)).float()

    if posterior is None:
        raise RuntimeError("generation did not produce a posterior")
    generated = suffix.clone()
    generated_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    invalid_token_ids = tuple(dict.fromkeys((*forbidden_token_ids, pad_token_id)))
    for position in range(max_new_tokens):
        active_rows = (~finished).nonzero(as_tuple=False).squeeze(-1)
        if active_rows.numel() == 0:
            break
        context = torch.cat(
            (prompt_ids.index_select(0, active_rows), generated.index_select(0, active_rows)[:, :position]),
            dim=1,
        ).to(posterior.device)
        posterior_rows = active_rows.to(posterior.device)
        tokens = process_generation_logits(
            posterior.index_select(0, posterior_rows)[:, position],
            context,
            generated_lengths=generated_lengths.index_select(0, active_rows).to(posterior.device),
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
        tokens = tokens.to(device)
        generated[active_rows, position] = tokens
        generated_lengths[active_rows] += 1
        finished[active_rows[tokens == eos_token_id]] = True

    token_ids = torch.cat((prompt_ids, generated), dim=1)
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
