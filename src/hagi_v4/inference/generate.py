"""Autoregressive generation for HAGI V5 codec model.

Bidirectional model: no KV-cache, each token recomputes the full sequence.
The refinement loop (7 layers × N iterations) is the dominant cost.

Optimisations:
- noise_ratio=0 by default: skip erasure injection at inference (the model
  already learned to predict; noise is a training regulariser, not needed
  for greedy/sampled decoding)
- Greedy mode (temperature=0) for fast deterministic generation
- Progress logging every 25 tokens for interactive feedback
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    max_iterations: int = 4,
    mask_token_id: int = 49153,
    eos_token_id: int | None = None,
    temperature: float = 0.8,
    top_k: int = 50,
    min_tokens: int = 2,
    noise_ratio: float = 0.0,
) -> torch.Tensor:
    """Generate text autoregressively.

    Args:
        model: HAGIv4 model (eval mode).
        prompt_ids: [B, T_prompt] prompt token IDs.
        max_new_tokens: hard cap on generated tokens.
        max_iterations: unused (kept for API compatibility).
        mask_token_id: token ID for masked positions.
        eos_token_id: token ID for end-of-sequence (None = no EOS).
        temperature: 0 for greedy, >0 for sampling.
        top_k: if >0, sample from top-k logits.
        min_tokens: minimum generated tokens before EOS accepted.
        noise_ratio: fraction of context to erase (0 = clean inference).
        verbose: print progress every 25 tokens.

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

    for step in range(max_new_tokens):
        if noise_ratio > 0 and full_ids.shape[1] > min_len:
            noisy_ids = full_ids.clone()
            noise_mask = torch.zeros_like(noisy_ids, dtype=torch.bool)
            n_ctx = noisy_ids.shape[1]
            n_noise = max(1, int(n_ctx * noise_ratio))
            for b in range(B):
                idx = torch.randperm(n_ctx, device=device)[:n_noise]
                noise_mask[b, idx] = True
            noisy_ids[noise_mask] = mask_token_id
            output = model(noisy_ids, targets=None, mask=noise_mask)
        else:
            output = model(full_ids, targets=None, mask=None)

        logits = output.logits[:, -1, :]

        if logits is None:
            break

        if eos_token_id is not None and step < min_tokens:
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
