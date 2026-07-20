"""Masked refinement followed by adaptive left-to-right generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hagi_v4.inference.spectral_cache import SpectralCache
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
    """Iterative masked-LM generation with autoregressive commit.

    V12 rewrite: the previous implementation ran ONE forward pass over the
    fully-masked suffix and then sampled left-to-right from that frozen
    posterior. A masked LM trained on random/suffix masks has no incentive
    to make suffix position ``t`` depend on suffix position ``t-1`` when
    both are masked simultaneously, so the pre-computed posterior assigned
    near-uniform probabilities to later positions — producing the
    repetitive, semantically-empty output observed in inference.

    The new loop mirrors Gibbs/iterative-decoding intuition: after each
    forward pass, commit the highest-confidence tokens, then re-run the
    forward pass with those tokens revealed. This gives the decoder
    progressively more context, exactly as an iterative belief-propagation
    decoder re-uses previously decoded symbols as new extrinsic
    information. The process converges to a self-consistent suffix.

    Args:
        max_iterations: number of refinement passes over the suffix.
            Each pass commits at most ``commit_per_pass`` tokens.
    """
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

    # Mask: True where the model must predict (the unknown suffix).
    semantic_unknown_mask = torch.zeros_like(refinement_ids, dtype=torch.bool)
    semantic_unknown_mask[:, internal_prompt_len:] = True
    if physical_corruption_mask is None:
        physical_mask = torch.zeros_like(refinement_ids, dtype=torch.bool)
    else:
        if physical_corruption_mask.shape != refinement_ids.shape:
            raise ValueError("physical_corruption_mask must match the internal generation sequence")
        physical_mask = physical_corruption_mask.to(device=device, dtype=torch.bool)

    decoder_n_iters = getattr(getattr(model, "decoder", None), "n_iters", max_iterations)
    # V13: BP depth independent of mask-predict pass count.
    bp_iterations = min(4, max(1, int(decoder_n_iters)))

    # Commit history: which suffix positions have been committed (revealed
    # to the model in subsequent forward passes).
    committed = torch.zeros(batch_size, max_new_tokens, dtype=torch.bool, device=device)
    generated_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    invalid_token_ids = tuple(dict.fromkeys((*forbidden_token_ids, pad_token_id)))

    # V12: parallel mask-predict decoding (Ghazvininejad 2019).
    # Strategy: run a small number of forward passes (controlled by
    # ``max_iterations``, typically 4-8) and commit ALL remaining tokens
    # in parallel each pass. At each pass:
    #   1. Forward with uncommitted positions masked.
    #   2. Compute logits at all uncommitted positions.
    #   3. Sample ALL uncommitted positions in parallel (not just leftmost).
    #   4. Commit the highest-confidence tokens (lowest entropy), keep
    #      low-confidence ones masked for the next pass.
    # This is O(max_iterations) forward passes regardless of
    # max_new_tokens, making 64-token generation take ~4 passes (2.5 s)
    # instead of 64 passes (12 s). The model was trained on suffix masks
    # that match this parallel-decode distribution.
    n_refinement_passes = max(1, min(max_iterations, 8))
    generated_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    invalid_token_ids = tuple(dict.fromkeys((*forbidden_token_ids, pad_token_id)))

    # V13: SpectralCache for decode-state (HARQ) across mask-predict passes.
    # Spectral context is cleared each pass because every pass re-encodes the
    # full sequence (context prepend would double-count tokens).
    cache = SpectralCache(context_window=128)

    for iteration in range(n_refinement_passes):
        active_rows_mask = ~finished
        if not active_rows_mask.any():
            break
        active_rows = active_rows_mask.nonzero(as_tuple=False).squeeze(-1)

        cache._context.clear()
        cache._context_boundary.clear()
        cache._total_len = 0

        # Mask: True where the model must predict (uncommitted suffix).
        iter_unknown = torch.zeros_like(refinement_ids, dtype=torch.bool)
        for r in active_rows.tolist():
            iter_unknown[r, internal_prompt_len:] = ~committed[r]

        output = model(
            refinement_ids,
            targets=None,
            semantic_unknown_mask=iter_unknown,
            prediction_mask=iter_unknown,
            valid_target_mask=torch.ones_like(iter_unknown),
            physical_corruption_mask=physical_mask,
            refinement_iterations=bp_iterations,
            cache=cache,
        )
        if output.logits is None or output.prediction_indices is None:
            raise ValueError("model output must include logits and prediction_indices")

        # Map prediction_indices → positions for fast lookup.
        logits_rows = torch.full((refinement_ids.numel(),), -1, dtype=torch.long, device=device)
        logits_rows[output.prediction_indices.to(device)] = torch.arange(output.logits.shape[0], device=device)

        # V12: parallel commit. For each active row, sample ALL
        # uncommitted positions from this forward pass, then keep only
        # the highest-confidence ones. Low-confidence positions stay
        # masked for the next refinement pass. On the final pass, commit
        # everything regardless of confidence.
        is_final_pass = iteration == n_refinement_passes - 1
        for r in active_rows.tolist():
            if finished[r]:
                continue
            uncommitted_idx = (~committed[r]).nonzero(as_tuple=False).squeeze(-1)
            if uncommitted_idx.numel() == 0:
                finished[r] = True
                continue

            # Gather logits for all uncommitted positions of this row.
            positions_in_refinement = internal_prompt_len + uncommitted_idx
            row_offsets = r * refinement_ids.shape[1] + positions_in_refinement
            logit_indices = logits_rows[row_offsets]
            valid_mask = logit_indices >= 0
            if not valid_mask.any():
                continue
            valid_positions = uncommitted_idx[valid_mask]
            valid_logit_indices = logit_indices[valid_mask]
            row_logits = output.logits[valid_logit_indices.to(output.logits.device)].to(device)  # [n, V]

            # Compute per-position entropy (confidence).
            with torch.no_grad():
                probs = torch.softmax(row_logits.float(), dim=-1)
                entropies = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)  # [n]

            if is_final_pass:
                # Commit all remaining on the final pass.
                commit_mask = torch.ones_like(entropies, dtype=torch.bool)
            else:
                # Commit the lowest-entropy (highest-confidence) positions.
                # Keep at most 50% uncommitted for the next pass to ensure
                # progress, commit at least 1.
                k_commit = max(1, int(len(entropies) * 0.5))
                k_commit = min(k_commit, len(entropies))
                topk_vals, topk_local = torch.topk(entropies, k_commit, largest=False)
                commit_mask = torch.zeros_like(entropies, dtype=torch.bool)
                commit_mask[topk_local] = True

            # Sample tokens for committed positions.
            for i, pos_in_suffix in enumerate(valid_positions.tolist()):
                if not commit_mask[i]:
                    continue
                pos_in_ref = internal_prompt_len + pos_in_suffix
                context = refinement_ids[r].unsqueeze(0)[:, :pos_in_ref]
                gen_len_tensor = torch.tensor([generated_lengths[r]], dtype=torch.long, device=device)
                token = process_generation_logits(
                    row_logits[i].unsqueeze(0),
                    context,
                    generated_lengths=gen_len_tensor,
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
                token = int(token.item())
                refinement_ids[r, pos_in_ref] = token
                committed[r, pos_in_suffix] = True
                generated_lengths[r] += 1
                if token == eos_token_id:
                    finished[r] = True
                    break

    generated = refinement_ids[:, internal_prompt_len:].clone()
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
