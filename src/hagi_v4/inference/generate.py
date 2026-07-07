"""Block-parallel generation with iterative refinement for HAGI V5.

5G analogies:
- Block coding (LDPC): generate block_size tokens per forward pass using
  the model's native masking, not 1 token at a time.
- Turbo iterative refinement: after generating tokens with masks (rough
  decode), run a second pass with tokens unmasked so they attend to each
  other (refined decode). Two component passes exchange information.
- Fixed 1-iteration inference: 5G LDPC hardware uses fixed small iteration
  counts. We force 1 refinement iteration at inference.
- Repetition penalty: 5G rate matching avoids repeating coded bits.
  Penalize recently used tokens to prevent degenerate loops.
- Successive blocks: each block becomes frozen context for the next
  (polar code successive cancellation analog).
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
    block_size: int = 16,
    refine_passes: int = 2,
    repetition_penalty: float = 1.5,
    repetition_window: int = 16,
) -> torch.Tensor:
    """Generate text block-parallel with Turbo-style iterative refinement.

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
        refine_passes: total forward passes per block (Turbo iterative).
        repetition_penalty: divide logit of recently used tokens by this.
        repetition_window: number of recent tokens to penalize.

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

    def apply_repetition_penalty(logits: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
        if repetition_penalty <= 1.0 or context_ids.numel() == 0:
            return logits
        window = context_ids[:, -repetition_window:] if context_ids.shape[1] >= repetition_window else context_ids
        for b in range(B):
            used = window[b].unique()
            used = used[used < V]
            logits[b, :, used] = logits[b, :, used] / repetition_penalty
        return logits

    def sample_tokens(logits: torch.Tensor, n_block: int, gen_count: int) -> torch.Tensor:
        if temperature > 0:
            lt = logits / temperature
        else:
            lt = logits

        if repetition_penalty > 1.0 and full_ids.shape[1] > 0:
            lt = apply_repetition_penalty(lt.clone(), full_ids)

        if top_k > 0:
            v, _ = torch.topk(lt, min(top_k, V), dim=-1)
            lt = torch.where(lt < v[..., -1:], float("-inf"), lt)

        if temperature > 0:
            probs = F.softmax(lt, dim=-1)
            return torch.multinomial(probs.reshape(-1, V), 1).reshape(B, n_block)
        else:
            return lt.argmax(dim=-1)

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
            if output.logits is None:
                break

            block_logits = output.logits[:, T - n_block :, :]

            new_tokens = sample_tokens(block_logits, n_block, generated)

            for pass_idx in range(1, refine_passes):
                seq[:, T - n_block :] = new_tokens
                mask_refined = torch.zeros_like(mask)
                output = model(seq, targets=None, mask=mask_refined)
                if output.logits is None:
                    break
                refined_logits = output.logits[:, T - n_block :, :]
                block_logits = 0.5 * block_logits + 0.5 * refined_logits
                new_tokens = sample_tokens(block_logits, n_block, generated)

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
