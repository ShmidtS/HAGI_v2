# V23: fixed model.hrm references and model forward signature.

"""Speculative Block Generation — predict next block while refining current.

Shannon analogy: In polar coding, successive cancellation decoding
processes bits in order, but can speculatively decode later bits
before earlier bits are fully confirmed. If speculation is correct,
latency is reduced; if wrong, rollback and re-decode.

V5's generation: sequential blocks, each with 2 refine passes.
  Block 1: rough -> refined -> commit
  Block 2: rough -> refined -> commit  (starts after block 1 fully done)

V6's speculative generation: overlap block N+1 prediction with block N refinement.
  Block 1: rough -> [speculatively predict block 2] -> refined -> commit block 1
  Block 2: [use speculative prediction as starting point] -> refined -> commit

If the speculative prediction is close to the refined result, we save
one forward pass (the rough pass for block 2). This is analogous to
speculative execution in CPUs: predict the branch, execute ahead,
verify later.

The key insight: block-parallel generation with bidirectional attention
means the rough prediction for block N+1 is available after block N's
rough pass. We can use block N's refined pass to simultaneously
provide context for block N+1's refinement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_speculative(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    mask_token_id: int = 49153,
    eos_token_id: int | None = None,
    temperature: float = 0.8,
    top_k: int = 50,
    min_tokens: int = 2,
    block_size: int = 16,
    refine_passes: int = 2,
    repetition_penalty: float = 0.8,
    repetition_window: int = 32,
    speculation_confidence_threshold: float = 0.5,
) -> torch.Tensor:
    """Generate text with speculative block prediction.

    While refining block N, speculatively predict block N+1 using the
    rough tokens of block N as additional context. If the speculative
    prediction has high confidence, skip the rough pass for block N+1.

    Args:
        model: HAGI model (eval mode).
        prompt_ids: [B, T_prompt] prompt token IDs.
        max_new_tokens: hard cap on generated tokens.
        mask_token_id: token ID for masked positions.
        eos_token_id: token ID for EOS.
        temperature: 0 for greedy, >0 for sampling.
        top_k: sample from top-k logits.
        min_tokens: minimum tokens before EOS.
        block_size: tokens generated per forward pass.
        refine_passes: forward passes per block (Turbo iterative).
        repetition_penalty: divide logit of recently used tokens.
        repetition_window: recent tokens to penalize.
        speculation_confidence_threshold: if speculative block mean
            confidence exceeds this, skip rough pass for next block.

    Returns:
        [B, T_prompt + generated] token IDs.
    """
    model.eval()
    B = prompt_ids.shape[0]
    T_prompt = prompt_ids.shape[1]
    device = prompt_ids.device
    V = model.cfg.model.vocab_size

    min_len = 4
    if T_prompt < min_len:
        pad = torch.full((B, min_len - T_prompt), mask_token_id, dtype=torch.long, device=device)
        full_ids = torch.cat([prompt_ids, pad], dim=1)
    else:
        full_ids = prompt_ids.clone()

    def apply_echo_cancellation(logits: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
        if repetition_penalty <= 0.0 or context_ids.numel() == 0:
            return logits
        window = context_ids[:, -repetition_window:] if context_ids.shape[1] >= repetition_window else context_ids
        for b in range(B):
            unique, counts = window[b].unique(return_counts=True)
            unique = unique[unique < V]
            counts = counts[: len(unique)].to(logits.dtype)
            logits[b, :, unique] = logits[b, :, unique] - counts * repetition_penalty
        return logits

    def sample_tokens(logits: torch.Tensor, n_block: int) -> tuple[torch.Tensor, float]:
        if temperature > 0:
            lt = logits / temperature
        else:
            lt = logits

        if repetition_penalty > 0.0 and full_ids.shape[1] > 0:
            lt = apply_echo_cancellation(lt.clone(), full_ids)

        if top_k > 0:
            v, _ = torch.topk(lt, min(top_k, V), dim=-1)
            lt = torch.where(lt < v[..., -1:], float("-inf"), lt)

        if temperature > 0:
            probs = F.softmax(lt, dim=-1)
            tokens = torch.multinomial(probs.reshape(-1, V), 1).reshape(B, n_block)
        else:
            tokens = lt.argmax(dim=-1)

        confidence = F.softmax(lt, dim=-1).max(dim=-1).values.mean().item()
        return tokens, confidence

    generated = 0
    speculative_tokens: torch.Tensor | None = None
    speculative_confidence: float = 0.0

    while generated < max_new_tokens:
        n_block = min(block_size, max_new_tokens - generated)

        if speculative_tokens is not None and speculative_confidence > speculation_confidence_threshold:
            new_tokens = speculative_tokens[:, :n_block]
            rough_logits = None
        else:
            mask_tokens = torch.full((B, n_block), mask_token_id, dtype=torch.long, device=device)
            seq = torch.cat([full_ids, mask_tokens], dim=1)
            T = seq.shape[1]

            semantic_unknown_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            semantic_unknown_mask[:, T - n_block :] = True
            prediction_mask = semantic_unknown_mask.clone()
            valid_target_mask = torch.ones(B, T, dtype=torch.bool, device=device)
            physical_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

            output = model(
                seq,
                targets=None,
                semantic_unknown_mask=semantic_unknown_mask,
                prediction_mask=prediction_mask,
                valid_target_mask=valid_target_mask,
                physical_corruption_mask=physical_mask,
                attention_mode="bidir",
                refinement_iterations=1,
            )
            if output.logits is None:
                break

            rough_logits = output.logits.view(B, n_block, V)
            new_tokens, _ = sample_tokens(rough_logits, n_block)

        for pass_idx in range(1, refine_passes):
            seq = torch.cat([full_ids, new_tokens], dim=1)
            T = seq.shape[1]

            semantic_unknown_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            prediction_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            prediction_mask[:, T - n_block :] = True
            valid_target_mask = torch.ones(B, T, dtype=torch.bool, device=device)
            physical_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

            output = model(
                seq,
                targets=None,
                semantic_unknown_mask=semantic_unknown_mask,
                prediction_mask=prediction_mask,
                valid_target_mask=valid_target_mask,
                physical_corruption_mask=physical_mask,
                attention_mode="bidir",
            )
            if output.logits is None:
                break
            refined_logits = output.logits.view(B, n_block, V)

            if rough_logits is not None:
                combined_logits = 0.5 * rough_logits + 0.5 * refined_logits
            else:
                combined_logits = refined_logits
                rough_logits = refined_logits

            new_tokens, _ = sample_tokens(combined_logits, n_block)

        if eos_token_id is not None and generated < min_tokens:
            new_tokens[:, 0] = torch.where(
                new_tokens[:, 0] == eos_token_id,
                torch.full_like(new_tokens[:, 0], mask_token_id),
                new_tokens[:, 0],
            )

        full_ids = torch.cat([full_ids, new_tokens], dim=1)
        generated += n_block

        if n_block == block_size and generated < max_new_tokens:
            next_n = min(block_size, max_new_tokens - generated)
            next_mask_tokens = torch.full((B, next_n), mask_token_id, dtype=torch.long, device=device)
            spec_seq = torch.cat([full_ids, next_mask_tokens], dim=1)
            T_spec = spec_seq.shape[1]
            spec_semantic = torch.zeros(B, T_spec, dtype=torch.bool, device=device)
            spec_semantic[:, T_spec - next_n :] = True
            spec_pred = spec_semantic.clone()
            spec_valid = torch.ones(B, T_spec, dtype=torch.bool, device=device)
            spec_physical = torch.zeros(B, T_spec, dtype=torch.bool, device=device)

            spec_output = model(
                spec_seq,
                targets=None,
                semantic_unknown_mask=spec_semantic,
                prediction_mask=spec_pred,
                valid_target_mask=spec_valid,
                physical_corruption_mask=spec_physical,
                attention_mode="bidir",
                refinement_iterations=1,
            )
            if spec_output.logits is not None:
                spec_logits = spec_output.logits.view(B, next_n, V)
                speculative_tokens, speculative_confidence = sample_tokens(spec_logits, next_n)

        if eos_token_id is not None and (new_tokens == eos_token_id).any():
            break

    return full_ids[:, : T_prompt + generated]
