"""Progressive unmasking inference for HAGI V4.

Dynamic-length generation: the model decides when to stop by emitting EOS.
The mask buffer grows as needed instead of being fixed upfront.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 512,
    max_iterations: int = 4,
    mask_token_id: int = 49153,
    eos_token_id: int = 49154,
    temperature: float = 0.0,
    top_k: int = 0,
    init_tokens: int = 32,
    extend_tokens: int = 32,
    min_tokens: int = 2,
) -> torch.Tensor:
    """Generate text through progressive unmasking with dynamic length.

    Starts with init_tokens mask positions. After each round of iterations,
    if no EOS was emitted, extends the buffer by extend_tokens and continues.
    Stops when EOS appears or max_new_tokens is reached.

    Args:
        model: HAGIv4 model (eval mode).
        prompt_ids: [B, T_prompt] prompt token IDs.
        max_new_tokens: hard cap on generated tokens.
        max_iterations: mask-predict rounds per chunk.
        mask_token_id: token ID for masked positions.
        eos_token_id: token ID for end-of-sequence.
        temperature: 0 for argmax, >0 for sampling.
        top_k: if >0, sample from top-k logits.
        init_tokens: initial mask buffer size.
        extend_tokens: buffer growth per extension.
        min_tokens: minimum generated tokens before EOS is accepted.

    Returns:
        [B, T_prompt + generated] token IDs (truncated at EOS).
    """
    model.eval()
    B = prompt_ids.shape[0]
    T_prompt = prompt_ids.shape[1]
    device = prompt_ids.device
    V = model.cfg.model.vocab_size

    cur_len = min(init_tokens, max_new_tokens)
    mask_tokens = torch.full((B, cur_len), mask_token_id, dtype=torch.long, device=device)
    full_ids = torch.cat([prompt_ids, mask_tokens], dim=1)

    while True:
        for round_idx in range(max_iterations):
            mask = full_ids == mask_token_id
            if not mask.any():
                break

            output = model(full_ids, mask=mask)
            logits = output.logits
            probs = F.softmax(logits.float(), dim=-1)

            for b in range(B):
                masked_positions = mask[b].nonzero(as_tuple=True)[0]
                if len(masked_positions) == 0:
                    continue
                for pos in masked_positions:
                    logit = logits[b, pos]
                    block_eos = eos_token_id is not None and (pos - T_prompt) < min_tokens
                    if temperature > 0:
                        logit = logit / temperature
                        if top_k > 0:
                            v, _ = torch.topk(logit, min(top_k, V))
                            logit = logit.clone()
                            logit[logit < v[-1]] = float("-inf")
                        if block_eos:
                            logit = logit.clone()
                            logit[eos_token_id] = float("-inf")
                        sampled = torch.multinomial(torch.softmax(logit, dim=-1), 1)
                    else:
                        if block_eos:
                            logit = logit.clone()
                            logit[eos_token_id] = float("-inf")
                        sampled = logit.argmax(dim=-1, keepdim=True)
                    full_ids[b, pos] = sampled
                    if eos_token_id is not None and sampled.item() == eos_token_id:
                        full_ids[b, pos + 1 :] = mask_token_id
                        break

            if eos_token_id is not None:
                eos_mask = full_ids[:, T_prompt:] == eos_token_id
                if eos_mask.any():
                    min_eos = eos_mask.nonzero(as_tuple=False)[:, 1].min().item() + T_prompt
                    return full_ids[:, :min_eos]

            if round_idx < max_iterations - 1:
                mask = full_ids == mask_token_id
                if mask.any():
                    output = model(full_ids, mask=mask)
                    logits = output.logits
                    probs = F.softmax(logits.float(), dim=-1)
                    confidence = probs.max(dim=-1).values
                    for b in range(B):
                        masked_positions = mask[b].nonzero(as_tuple=True)[0]
                        if len(masked_positions) <= 1:
                            continue
                        confs = confidence[b, masked_positions]
                        n_remask = len(masked_positions) // 2
                        if n_remask > 0:
                            _, low_conf_idx = torch.topk(confs, n_remask, largest=False)
                            for idx in low_conf_idx:
                                pos = masked_positions[idx]
                                full_ids[b, pos] = mask_token_id

        if cur_len >= max_new_tokens:
            break

        add = min(extend_tokens, max_new_tokens - cur_len)
        ext = torch.full((B, add), mask_token_id, dtype=torch.long, device=device)
        full_ids = torch.cat([full_ids, ext], dim=1)
        cur_len += add

    return full_ids
