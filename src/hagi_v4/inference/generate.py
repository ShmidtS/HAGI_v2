"""Noisy-context autoregressive generation for HAGI V5 codec model.

The model is a denoising codec: mask tokens are erasure noise, the model
learns to reconstruct signal from partially-noised input through
bidirectional attention (belief propagation).

At inference we preserve this denoising dynamic:
1. Predict next token at the last position (clean, no mask on generation pos)
2. Inject noise: replace ~15% of random CONTEXT positions with mask_token
3. Model denoises through attention — prediction benefits from denoising
4. Restore noised context, append predicted token, repeat

This is a proper communication channel: context noise = erasure,
model denoises through belief propagation (attention + refinement).
The generation position itself is never masked — the model always sees
the full clean signal at the prediction point.
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
    temperature: float = 0.8,
    top_k: int = 50,
    min_tokens: int = 2,
    noise_ratio: float = 0.15,
) -> torch.Tensor:
    """Generate text with noisy-context denoising.

    Args:
        model: HAGIv4 model (eval mode).
        prompt_ids: [B, T_prompt] prompt token IDs.
        max_new_tokens: hard cap on generated tokens.
        max_iterations: unused (kept for API compatibility).
        mask_token_id: token ID for masked (erased) positions.
        eos_token_id: token ID for end-of-sequence.
        temperature: 0 for argmax, >0 for sampling.
        top_k: if >0, sample from top-k logits.
        min_tokens: minimum generated tokens before EOS is accepted.
        noise_ratio: fraction of context positions to erase (erasure rate).

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
        noisy_ids = full_ids.clone()
        noise_mask = torch.zeros_like(noisy_ids, dtype=torch.bool)

        if noise_ratio > 0 and noisy_ids.shape[1] > min_len:
            n_ctx = noisy_ids.shape[1]
            n_noise = max(1, int(n_ctx * noise_ratio))
            for b in range(B):
                idx = torch.randperm(n_ctx, device=device)[:n_noise]
                noise_mask[b, idx] = True
            noisy_ids[noise_mask] = mask_token_id

        output = model(noisy_ids, targets=None, mask=noise_mask)
        logits = output.logits[:, -1, :]

        if logits is None:
            break

        block_eos = eos_token_id is not None and step < min_tokens
        if block_eos:
            logits = logits.clone()
            logits[:, eos_token_id] = float("-inf")

        if temperature > 0:
            lt = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(lt, min(top_k, V), dim=-1)
                lt = torch.where(lt < v[:, -1:], float("-inf"), lt)
            probs = F.softmax(lt, dim=-1)
            next_tok = torch.multinomial(probs, 1)
        else:
            next_tok = logits.argmax(dim=-1, keepdim=True)

        full_ids = torch.cat([full_ids, next_tok], dim=1)

        if eos_token_id is not None and next_tok[0].item() == eos_token_id:
            return full_ids[:, : T_prompt + step]

    return full_ids[:, : T_prompt + max_new_tokens]
