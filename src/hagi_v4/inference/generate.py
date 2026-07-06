"""Progressive unmasking inference for HAGI V4.

V4 generates text through iterative mask-predict with progressive
unmasking:

1. Start with prompt + mask tokens for max_new_tokens
2. Run model -> get predictions for all masked positions
3. Unmask highest-confidence predictions (or sample with temperature)
4. Repeat until all unmasked or EOS detected
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 256,
    max_iterations: int = 4,
    confidence_schedule: tuple[float, ...] = (0.9, 0.7, 0.5, 0.1),
    mask_token_id: int = 49153,
    eos_token_id: int = 49154,
    temperature: float = 0.0,
    top_k: int = 0,
) -> torch.Tensor:
    """Generate text through progressive unmasking.

    Args:
        model: HAGIv4 model (eval mode).
        prompt_ids: [B, T_prompt] prompt token IDs.
        max_new_tokens: maximum tokens to generate.
        max_iterations: max mask-predict rounds.
        confidence_schedule: decreasing thresholds for each round.
        mask_token_id: token ID for masked positions.
        eos_token_id: token ID for end-of-sequence.
        temperature: 0 for argmax, >0 for sampling.
        top_k: if >0, sample from top-k logits.

    Returns:
        [B, T_prompt + generated] token IDs.
    """
    model.eval()
    B = prompt_ids.shape[0]
    T_prompt = prompt_ids.shape[1]
    device = prompt_ids.device
    V = model.cfg.model.vocab_size

    mask_tokens = torch.full((B, max_new_tokens), mask_token_id, dtype=torch.long, device=device)
    full_ids = torch.cat([prompt_ids, mask_tokens], dim=1)

    schedule = confidence_schedule[:max_iterations]
    if len(schedule) < max_iterations:
        schedule = schedule + (0.1,) * (max_iterations - len(schedule))

    for round_idx, conf_threshold in enumerate(schedule):
        mask = full_ids == mask_token_id
        if not mask.any():
            break

        output = model(full_ids)
        logits = output.logits
        probs = F.softmax(logits.float(), dim=-1)
        confidence, predicted = probs.max(dim=-1)

        if temperature > 0:
            for b in range(B):
                masked_positions = mask[b].nonzero(as_tuple=True)[0]
                if len(masked_positions) == 0:
                    continue
                for pos in masked_positions:
                    logit = logits[b, pos] / temperature
                    if top_k > 0:
                        v, _ = torch.topk(logit, min(top_k, V))
                        logit = logit.clone()
                        logit[logit < v[-1]] = float("-inf")
                    sampled = torch.multinomial(torch.softmax(logit, dim=-1), 1)
                    full_ids[b, pos] = sampled
                    confidence[b, pos] = probs[b, pos, sampled]
        else:
            fill_mask = mask & (confidence > conf_threshold)
            if fill_mask.any():
                full_ids[fill_mask] = predicted[fill_mask]

        if eos_token_id is not None:
            eos_mask = full_ids[:, T_prompt:] == eos_token_id
            if eos_mask.any():
                min_eos = eos_mask.nonzero(as_tuple=False)[:, 1].min().item() + T_prompt
                full_ids = full_ids[:, :min_eos]
                return full_ids

        remaining = (full_ids == mask_token_id).any()
        if not remaining:
            break

    mask = full_ids == mask_token_id
    if mask.any():
        output = model(full_ids)
        logits = output.logits
        predicted = logits.argmax(dim=-1)
        full_ids[mask] = predicted[mask]

    return full_ids
