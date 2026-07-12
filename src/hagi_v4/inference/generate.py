"""Block-parallel generation with iterative refinement for HAGI V7.1.

5G analogies:
  - Block coding (LDPC): generate block_size tokens per forward pass
  - Turbo iterative refinement: rough decode then refined decode
  - Spectral cache: OFDM cyclic prefix analog — cache hidden states
    at layer boundaries + Kalman P state across blocks
  - Repetition penalty: echo cancellation (G.168 analog)
  - Successive blocks: polar code successive cancellation analog
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hagi_v4.inference.spectral_cache import SpectralCache


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
    repetition_penalty: float = 0.8,
    repetition_window: int = 32,
    no_repeat_ngram_size: int = 3,
    use_cache: bool = True,
    cache_window: int = 128,
) -> torch.Tensor:
    """Generate text block-parallel with Turbo-style iterative refinement.

    Spectral cache (OFDM cyclic prefix analog):
      First call processes the full prompt and caches perception output.
      Subsequent calls process only (cached_window + new_block) instead of
      (full_prompt + all_generated + new_block). FFT complexity drops from
      O(T_total * log(T_total)) to O((W + block) * log(W + block)).

    Args:
        model: HAGIv4 model (eval mode).
        prompt_ids: [B, T_prompt] prompt token IDs.
        max_new_tokens: hard cap on generated tokens.
        max_iterations: turbo iterations at inference.
        mask_token_id: token ID for masked positions.
        eos_token_id: token ID for EOS (None = no EOS check).
        temperature: 0 for greedy, >0 for sampling.
        top_k: if >0, sample from top-k logits.
        min_tokens: minimum tokens before EOS accepted.
        block_size: tokens generated per forward pass (5G block coding).
        refine_passes: total forward passes per block (Turbo iterative).
        repetition_penalty: multiplicative penalty for recently used tokens.
        repetition_window: number of recent tokens to penalize.
        no_repeat_ngram_size: ban tokens that would create repeated n-grams.
        use_cache: enable spectral cache for efficient inference.

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

    cache = SpectralCache(context_window=cache_window) if use_cache else None

    turbo = getattr(model, "turbo", None)
    orig_n_iters_backup = None
    if turbo is not None:
        orig_n_iters_backup = turbo.n_iters
        turbo.n_iters = max(max_iterations, 2)

    def apply_echo_cancellation(logits: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
        if repetition_penalty <= 0.0 or context_ids.numel() == 0:
            return logits
        window = context_ids[:, -repetition_window:] if context_ids.shape[1] >= repetition_window else context_ids
        for b in range(B):
            unique, counts = window[b].unique(return_counts=True)
            unique = unique[unique < V]
            counts = counts[: len(unique)].to(logits.dtype)
            factor = repetition_penalty**counts
            score = logits[b, :, unique]
            score = torch.where(score > 0, score / factor, score * factor)
            logits[b, :, unique] = score
        return logits

    def ban_repeated_ngrams(logits: torch.Tensor, context_ids: torch.Tensor, n: int) -> torch.Tensor:
        if n <= 0 or context_ids.numel() == 0 or context_ids.shape[1] < n:
            return logits
        for b in range(B):
            seq = context_ids[b].tolist()
            ngram_set = set()
            for i in range(len(seq) - n + 1):
                ngram_set.add(tuple(seq[i : i + n]))
            prefix = tuple(seq[-(n - 1) :]) if len(seq) >= n - 1 else None
            if prefix is None:
                continue
            banned = set()
            for ngram in ngram_set:
                if ngram[:-1] == prefix:
                    banned.add(ngram[-1])
            if banned:
                ban_idx = torch.tensor(list(banned), dtype=torch.long, device=logits.device)
                logits[b, :, ban_idx] = float("-inf")
        return logits

    def sample_tokens(logits: torch.Tensor, n_block: int) -> torch.Tensor:
        if temperature > 0:
            lt = logits / temperature
        else:
            lt = logits

        if repetition_penalty > 0.0 and full_ids.shape[1] > 0:
            lt = apply_echo_cancellation(lt.clone(), full_ids)

        if no_repeat_ngram_size > 0 and full_ids.shape[1] >= no_repeat_ngram_size:
            lt = ban_repeated_ngrams(lt, full_ids, no_repeat_ngram_size)

        if top_k > 0:
            v, _ = torch.topk(lt, min(top_k, V), dim=-1)
            lt = torch.where(lt < v[..., -1:], float("-inf"), lt)

        if temperature > 0:
            probs = F.softmax(lt, dim=-1)
            return torch.multinomial(probs.reshape(-1, V), 1).reshape(B, n_block)
        else:
            return lt.argmax(dim=-1)

    if cache is not None:
        _ = model(full_ids, targets=None, mask=None, cache=cache)

    generated = 0
    try:
        while generated < max_new_tokens:
            n_block = min(block_size, max_new_tokens - generated)

            if cache is not None:
                mask_tokens = torch.full((B, n_block), mask_token_id, dtype=torch.long, device=device)
                block_mask = torch.ones(B, n_block, dtype=torch.bool, device=device)
                output = model(mask_tokens, targets=None, mask=block_mask, cache=cache)
            else:
                mask_tokens = torch.full((B, n_block), mask_token_id, dtype=torch.long, device=device)
                seq = torch.cat([full_ids, mask_tokens], dim=1)
                T = seq.shape[1]
                mask = torch.zeros(B, T, dtype=torch.bool, device=device)
                mask[:, T - n_block :] = True
                output = model(seq, targets=None, mask=mask)

            if output.logits is None:
                break

            block_logits = output.logits[:, -n_block:, :]
            new_tokens = sample_tokens(block_logits, n_block)

            for pass_idx in range(1, refine_passes):
                if cache is not None:
                    output = model(new_tokens, targets=None, mask=None, cache=cache)
                else:
                    seq[:, -n_block:] = new_tokens
                    mask_refined = torch.zeros(B, seq.shape[1], dtype=torch.bool, device=device)
                    output = model(seq, targets=None, mask=mask_refined)
                if output.logits is None:
                    break
                refined_logits = output.logits[:, -n_block:, :]
                block_logits = 0.5 * block_logits + 0.5 * refined_logits
                new_tokens = sample_tokens(block_logits, n_block)

            if eos_token_id is not None and generated < min_tokens:
                new_tokens = torch.where(
                    new_tokens == eos_token_id,
                    torch.full_like(new_tokens, mask_token_id),
                    new_tokens,
                )

            full_ids = torch.cat([full_ids, new_tokens], dim=1)
            generated += n_block

            if eos_token_id is not None and (new_tokens == eos_token_id).any():
                break
    finally:
        if turbo is not None and orig_n_iters_backup is not None:
            turbo.n_iters = orig_n_iters_backup

    return full_ids[:, : T_prompt + generated]
