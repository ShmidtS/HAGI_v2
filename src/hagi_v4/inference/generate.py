"""Autoregressive generation for HAGI V4.

The model uses bidirectional attention with next-token targets (shifted by 1).
At inference, no masking is needed — the model predicts token[t+1] at position t
given the full visible context. Standard next-token decoding loop.
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
    min_tokens: int = 2,
) -> torch.Tensor:
    """Generate text autoregressively (next-token prediction).

    Args:
        model: HAGIv4 model (eval mode).
        prompt_ids: [B, T_prompt] prompt token IDs.
        max_new_tokens: hard cap on generated tokens.
        max_iterations: unused (kept for API compatibility).
        mask_token_id: unused (kept for API compatibility).
        eos_token_id: token ID for end-of-sequence.
        temperature: 0 for argmax, >0 for sampling.
        top_k: if >0, sample from top-k logits.
        min_tokens: minimum generated tokens before EOS is accepted.

    Returns:
        [B, T_prompt + generated] token IDs (truncated at EOS).
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

    for step in range(max_new_tokens):
        output = model(full_ids, targets=None, mask=None)
        logits = output.logits[:, -1, :]

        block_eos = eos_token_id is not None and step < min_tokens
        if block_eos:
            logits = logits.clone()
            logits[:, eos_token_id] = float("-inf")

        if temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, V), dim=-1)
                logits = torch.where(logits < v[:, -1:], float("-inf"), logits)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
        else:
            next_tok = logits.argmax(dim=-1, keepdim=True)

        full_ids = torch.cat([full_ids, next_tok], dim=1)

        if eos_token_id is not None and next_tok[0].item() == eos_token_id:
            return full_ids[:, : T_prompt + step]

    return full_ids[:, : T_prompt + max_new_tokens]
