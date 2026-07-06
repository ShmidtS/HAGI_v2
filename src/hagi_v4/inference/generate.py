"""Masked autoregressive generation for HAGI V4.

The model is a denoising codec: mask tokens are noise, the model learns to
reconstruct the signal (next-token) from partially-masked input. At inference
we preserve this denoising dynamic instead of bypassing it.

Each new token goes through a mask-predict cycle:
1. Append mask token at the end
2. Run model with mask on that position
3. Model denoises: predicts the next token using bidirectional context
4. Replace mask with prediction, repeat

Mask ratio during generation matches training (~30%) to stay in-distribution:
we inject extra mask tokens into the visible context as noise.
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
    noise_ratio: float = 0.15,
) -> torch.Tensor:
    """Generate text through masked autoregressive denoising.

    Args:
        model: HAGIv4 model (eval mode).
        prompt_ids: [B, T_prompt] prompt token IDs.
        max_new_tokens: hard cap on generated tokens.
        max_iterations: denoise iterations per token (refine prediction).
        mask_token_id: token ID for masked positions.
        eos_token_id: token ID for end-of-sequence.
        temperature: 0 for argmax, >0 for sampling.
        top_k: if >0, sample from top-k logits.
        min_tokens: minimum generated tokens before EOS is accepted.
        noise_ratio: fraction of visible context to mask as noise (matches training).

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
        gen_mask = torch.zeros_like(full_ids, dtype=torch.bool)
        gen_mask[:, -1:] = True

        noise_positions = None
        saved = None
        if noise_ratio > 0 and full_ids.shape[1] > min_len:
            visible = (~gen_mask).nonzero(as_tuple=False)
            n_visible = visible.shape[0]
            n_noise = max(1, int(n_visible * noise_ratio))
            noise_idx = torch.randperm(n_visible, device=device)[:n_noise]
            noise_positions = visible[noise_idx]
            saved = full_ids[noise_positions[:, 0], noise_positions[:, 1]].clone()
            full_ids[noise_positions[:, 0], noise_positions[:, 1]] = mask_token_id
            gen_mask[noise_positions[:, 0], noise_positions[:, 1]] = True

        next_tok = None
        for it in range(max_iterations):
            output = model(full_ids, targets=None, mask=gen_mask)
            logits = output.logits[:, -1, :]

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

            full_ids[:, -1:] = next_tok
            gen_mask[:, -1:] = False

            if noise_ratio > 0 and it < max_iterations - 1:
                output2 = model(full_ids, targets=None, mask=None)
                conf = F.softmax(output2.logits[:, -1, :].float(), dim=-1).max(dim=-1).values
                if conf.item() > 0.5:
                    break

        if noise_ratio > 0 and noise_positions is not None:
            full_ids[noise_positions[:, 0], noise_positions[:, 1]] = saved

        if eos_token_id is not None and next_tok[0].item() == eos_token_id:
            return full_ids[:, : T_prompt + step]

        next_mask = torch.full((B, 1), mask_token_id, dtype=torch.long, device=device)
        full_ids = torch.cat([full_ids, next_mask], dim=1)

    return full_ids[:, : T_prompt + max_new_tokens]
