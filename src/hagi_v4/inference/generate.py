"""Block-parallel generation for HAGI V5 codec model.

5G analogies:
- Block coding: 5G LDPC decodes an entire codeword in parallel, not 1 bit
  at a time. We generate block_size tokens per forward pass using the
  model's native masking capability.
- Fixed iterations at inference: 5G LDPC hardware uses a fixed small
  iteration count (not adaptive). We force 1 refinement iteration at
  inference — multiple iterations are a training regulariser.
- Successive blocks: each block's predictions become frozen context for
  the next block (polar code successive cancellation analog).

Speed: block_size=8 with 1 iteration gives ~8x fewer forward passes
and ~4-6x fewer iterations per pass vs autoregressive with 4-6 iters.
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
    block_size: int = 8,
) -> torch.Tensor:
    """Generate text block-parallel with fixed 1-iteration inference.

    Args:
        model: HAGIv4 model (eval mode).
        prompt_ids: [B, T_prompt] prompt token IDs.
        max_new_tokens: hard cap on generated tokens.
        max_iterations: unused (kept for API compatibility).
        mask_token_id: token ID for masked positions.
        eos_token_id: token ID for EOS (None = no EOS check).
        temperature: 0 for greedy, >0 for sampling.
        top_k: if >0, sample from top-k logits.
        min_tokens: minimum tokens before EOS accepted.
        noise_ratio: unused (kept for API compatibility).
        block_size: tokens generated per forward pass (5G block coding).

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

    orig_n_iters = model.hrm.entropy_scheduler.n_iterations
    orig_adaptive = model.hrm.entropy_scheduler.use_entropy_adaptive
    model.hrm.entropy_scheduler.use_entropy_adaptive = False
    model.hrm.entropy_scheduler.n_iterations = 1

    generated = 0
    try:
        while generated < max_new_tokens:
            n_block = min(block_size, max_new_tokens - generated)
            mask_tokens = torch.full((B, n_block), mask_token_id, dtype=torch.long, device=device)
            seq = torch.cat([full_ids, mask_tokens], dim=1)
            T = seq.shape[1]

            mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            mask[:, T - n_block :] = True

            output = model(seq, targets=None, mask=mask)
            logits = output.logits

            if logits is None:
                break

            block_logits = logits[:, T - n_block :, :]

            if temperature > 0:
                lt = block_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(lt, min(top_k, V), dim=-1)
                    lt = torch.where(lt < v[..., -1:], float("-inf"), lt)
                probs = F.softmax(lt, dim=-1)
                new_tokens = torch.multinomial(probs.reshape(-1, V), 1).reshape(B, n_block)
            else:
                new_tokens = block_logits.argmax(dim=-1)

            if eos_token_id is not None and generated < min_tokens:
                new_tokens[:, 0] = torch.where(
                    new_tokens[:, 0] == eos_token_id,
                    torch.full_like(new_tokens[:, 0], mask_token_id),
                    new_tokens[:, 0],
                )

            full_ids = torch.cat([full_ids, new_tokens], dim=1)
            generated += n_block

            if eos_token_id is not None and (new_tokens == eos_token_id).any():
                break
    finally:
        model.hrm.entropy_scheduler.n_iterations = orig_n_iters
        model.hrm.entropy_scheduler.use_entropy_adaptive = orig_adaptive

    return full_ids[:, : T_prompt + generated]
