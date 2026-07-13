"""Generation for HAGI V7.1 masked LM (LLaDA-style iterative decoding).

5G analogies:
  - Block coding: generate max_new_tokens mask tokens, fill iteratively
  - Turbo iterative refinement: rough decode then refined decode
  - Spectral cache: OFDM cyclic prefix analog — cache hidden states
  - Repetition penalty: echo cancellation (G.168 analog)
  - Iterative decoding: LDPC belief propagation — fill confident positions first
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hagi_v4.inference.spectral_cache import SpectralCache
from hagi_v4.model.codec_contracts import InferenceShapeConfig


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    max_iterations: int = 4,
    mask_token_id: int = 49153,
    eos_token_id: int | None = None,
    temperature: float = 0.4,
    top_k: int = 20,
    min_tokens: int = 2,
    noise_ratio: float = 0.0,
    block_size: int = 8,
    refine_passes: int = 3,
    repetition_penalty: float = 1.1,
    repetition_window: int = 64,
    no_repeat_ngram_size: int = 0,
    use_cache: bool = False,
    cache_window: int = 128,
) -> torch.Tensor:
    """Generate text via LLaDA-style iterative masked decoding.

    Instead of generating block-by-block (which creates 100% mask ratio OOD),
    this creates prompt + max_new_tokens mask tokens and iteratively fills
    them based on model confidence — like LDPC belief propagation.

    Args:
        model: HAGIv4 model (eval mode).
        prompt_ids: [B, T_prompt] prompt token IDs.
        max_new_tokens: number of mask tokens to fill.
        max_iterations: forward passes for iterative decoding.
        mask_token_id: token ID for masked positions.
        eos_token_id: token ID for EOS (None = no EOS check).
        temperature: 0 for greedy, >0 for sampling.
        top_k: if >0, sample from top-k logits.
        min_tokens: minimum tokens before EOS accepted.
        block_size: unused (kept for API compat).
        refine_passes: unused (kept for API compat).
        repetition_penalty: multiplicative penalty for recently used tokens.
        repetition_window: number of recent tokens to penalize.
        no_repeat_ngram_size: ban tokens that would create repeated n-grams.
        use_cache: unused (kept for API compat).

    Returns:
        [B, T_prompt + max_new_tokens] token IDs.
    """
    model.eval()
    B = prompt_ids.shape[0]
    T_prompt = prompt_ids.shape[1]
    device = prompt_ids.device
    inference_config = getattr(model, "inference_config", None)
    V = (
        inference_config.vocab_size
        if inference_config is not None
        else InferenceShapeConfig.from_hagi_config(model.cfg).vocab_size
    )

    min_len = 4
    if T_prompt < min_len:
        pad = torch.full((B, min_len - T_prompt), mask_token_id, dtype=torch.long, device=device)
        prompt_ids = torch.cat([prompt_ids, pad], dim=1)
        T_prompt = prompt_ids.shape[1]

    mask_block = torch.full((B, max_new_tokens), mask_token_id, dtype=torch.long, device=device)
    seq = torch.cat([prompt_ids, mask_block], dim=1)
    mask = torch.zeros_like(seq, dtype=torch.bool)
    mask[:, T_prompt:] = True

    def apply_echo_cancellation(logits: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
        if repetition_penalty <= 1.0 or context_ids.numel() == 0:
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
            s = context_ids[b].tolist()
            ngram_set = set()
            for i in range(len(s) - n + 1):
                ngram_set.add(tuple(s[i : i + n]))
            prefix = tuple(s[-(n - 1) :]) if len(s) >= n - 1 else None
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

    def sample_token(logits: torch.Tensor) -> torch.Tensor:
        logits[:, mask_token_id] = float("-inf")
        logits[:, 0] = float("-inf")

        if repetition_penalty > 1.0 and seq.shape[1] > 0:
            visible = seq[~mask].reshape(B, -1)
            if visible.numel() > 0:
                logits = apply_echo_cancellation(logits.clone(), visible)

        if no_repeat_ngram_size > 0:
            visible = seq[~mask].reshape(B, -1)
            if visible.shape[1] >= no_repeat_ngram_size:
                logits = ban_repeated_ngrams(logits, visible, no_repeat_ngram_size)

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, V), dim=-1)
            logits = torch.where(logits < v[..., -1:], float("-inf"), logits)

        if temperature > 0:
            probs = F.softmax(logits.float(), dim=-1)
            return torch.multinomial(probs.reshape(B, -1), 1).squeeze(-1)
        else:
            return logits.argmax(dim=-1)

    turbo = getattr(model, "turbo", None)
    orig_n_iters = None
    if turbo is not None:
        orig_n_iters = turbo.n_iters
        turbo.n_iters = max(max_iterations, 2)

    try:
        for iteration in range(max_iterations):
            output = model(seq, targets=None, mask=mask)
            if output.logits is None:
                break

            logits = output.logits  # [B, T, V]
            mask_positions = mask[0].nonzero(as_tuple=True)[0]

            if len(mask_positions) == 0:
                break

            mask_logits = logits[0, mask_positions]  # [n_mask, V]

            if iteration < max_iterations - 1:
                confidence = mask_logits.float().max(dim=-1).values
                if temperature > 0:
                    confidence = confidence / temperature
                n_fill = max(1, int(len(mask_positions) * 0.5))
                top_confident = confidence.topk(n_fill).indices
                fill_positions = mask_positions[top_confident]
                fill_logits = mask_logits[top_confident]
            else:
                fill_positions = mask_positions
                fill_logits = mask_logits

            for i, pos in enumerate(fill_positions):
                single_logits = fill_logits[i : i + 1]
                if temperature > 0:
                    lt = single_logits / temperature
                else:
                    lt = single_logits
                if top_k > 0:
                    v, _ = torch.topk(lt, min(top_k, V), dim=-1)
                    lt = torch.where(lt < v[..., -1:], float("-inf"), lt)
                if temperature > 0:
                    probs = F.softmax(lt.float(), dim=-1)
                    tok = torch.multinomial(probs.reshape(1, -1), 1).squeeze(-1)
                else:
                    tok = lt.argmax(dim=-1)
                seq[0, pos] = tok
                mask[0, pos] = False

            if eos_token_id is not None and (seq[0, T_prompt:] == eos_token_id).any():
                break

    finally:
        if turbo is not None and orig_n_iters is not None:
            turbo.n_iters = orig_n_iters

    result = seq[:, : T_prompt + max_new_tokens]
    if eos_token_id is not None:
        eos_mask = result[0, T_prompt:] == eos_token_id
        if eos_mask.any():
            first_eos = T_prompt + eos_mask.nonzero()[0].item()
            result = result[:, : first_eos + 1]

    return result
