"""Autoregressive generation with an incremental KV-cache.

Two phases, matching the training path exactly:

* **Prefill** — the prompt runs once, filling every layer's KV-cache. Cost is one
  forward over the prompt.
* **Decode** — each new token is a single-position forward that reuses the frozen
  prefix. Cost per token is O(cached length) for attention and O(1) for
  everything else, so a T-token completion is O(T) rather than O(T^2).

Sampling operates on the same logits the training objective scored, including the
unigram prior. The three constraints applied here — repetition penalty, top-k,
top-p — are shaping of the output distribution, not corrections to it: a model
whose samples are only usable with heavy repetition penalties has a problem the
sampler cannot fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GenerationOutput:
    """Generated sequences and their termination state.

    Attributes:
        token_ids: ``[B, prompt_len + generated]`` including the prompt.
        lengths: ``[B]`` tokens generated per row.
        finished: ``[B]`` whether the row emitted EOS.
    """

    token_ids: torch.Tensor
    lengths: torch.Tensor
    finished: torch.Tensor


def filter_logits(
    logits: torch.Tensor,
    context: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    repetition_window: int,
    banned: tuple[int, ...] = (),
) -> torch.Tensor:
    """Shape ``[B, V]`` logits into the distribution to sample from.

    Order matters: penalties act on raw logits, then temperature scales, then
    truncation removes tail mass. Applying temperature before the penalty would
    make the penalty's strength depend on the temperature.

    Args:
        logits: ``[B, V]`` raw logits.
        context: ``[B, T]`` tokens generated so far (for the repetition penalty).
        temperature: 0 selects greedy decoding downstream.
        top_k: keep the k highest logits; 0 disables.
        top_p: keep the smallest prefix whose cumulative probability reaches p;
            1.0 disables.
        repetition_penalty: >1 discourages recently emitted tokens.
        repetition_window: how far back the penalty looks.
        banned: token ids to forbid outright.

    Returns:
        ``[B, V]`` float32 logits with removed entries at ``-inf``.
    """
    out = logits.float().clone()
    if banned:
        out[:, list(banned)] = float("-inf")

    if repetition_penalty != 1.0 and context.shape[1] > 0:
        window = context[:, -repetition_window:]
        gathered = out.gather(1, window)
        # Divide positive logits, multiply negative ones: both move the token
        # toward lower probability, which a single operation would not do.
        adjusted = torch.where(gathered > 0, gathered / repetition_penalty, gathered * repetition_penalty)
        out.scatter_(1, window, adjusted)

    if temperature > 0:
        out = out / temperature

    if top_k > 0:
        k = min(top_k, out.shape[-1])
        threshold = out.topk(k, dim=-1).values[:, -1:]
        out = out.masked_fill(out < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = out.sort(dim=-1, descending=True)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Keep everything up to and including the token that crosses p, so the
        # kept set is never empty even when one token holds more than p.
        remove = cumulative - sorted_logits.softmax(dim=-1) > top_p
        remove[:, 0] = False
        out = out.masked_fill(remove.scatter(1, sorted_idx, remove), float("-inf"))

    return out


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    repetition_window: int = 128,
    min_new_tokens: int = 1,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
    banned: tuple[int, ...] = (),
) -> GenerationOutput:
    """Sample a continuation for each prompt row.

    Args:
        model: a :class:`~hagi.model.model.HAGI`.
        prompt_ids: ``[B, T]`` int64 prompt.
        max_new_tokens: generation budget.
        eos_token_id: termination token.
        pad_token_id: fill for finished rows.
        temperature: 0 for greedy.
        top_k, top_p: truncation.
        repetition_penalty, repetition_window: repetition control.
        min_new_tokens: EOS is suppressed below this many tokens.
        use_cache: incremental decoding; False recomputes the full prefix each
            step (same output, O(T^2) cost) and is useful for verifying the cache.
        generator: seeded RNG for reproducible sampling.
        banned: token ids to forbid at every step (e.g. UNK / unused fillers the
            model learned as ordinary tokens from a corrupt compaction).

    Returns:
        :class:`GenerationOutput`.
    """
    if prompt_ids.ndim != 2 or prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must be a 2D int64 tensor")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError("min_new_tokens must be in [0, max_new_tokens]")

    model.eval()
    batch, prompt_len = prompt_ids.shape
    device = prompt_ids.device
    vocab = model.cfg.model.vocab_size
    if int(prompt_ids.max()) >= vocab or int(prompt_ids.min()) < 0:
        raise ValueError("prompt_ids contains out-of-vocabulary ids")

    sequence = torch.cat(
        [prompt_ids, torch.full((batch, max_new_tokens), pad_token_id, dtype=torch.long, device=device)],
        dim=1,
    )
    lengths = torch.zeros(batch, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    model.reset_cache()
    caches = model.allocate_cache(next(model.parameters()).dtype, device) if use_cache else None
    try:
        output = model(prompt_ids, return_logits=True, use_cache=use_cache)
        next_logits = output.logits[:, -1, :]

        for step in range(max_new_tokens):
            context = sequence[:, : prompt_len + step]
            banned = tuple(set(banned) | {pad_token_id})
            logits = filter_logits(
                next_logits,
                context,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                banned=banned,
            )
            if step + 1 < min_new_tokens:
                logits[:, eos_token_id] = float("-inf")

            if temperature == 0:
                chosen = logits.argmax(dim=-1)
            else:
                chosen = torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).squeeze(-1)

            # Finished rows emit padding and stop advancing their length.
            chosen = torch.where(finished, torch.full_like(chosen, pad_token_id), chosen)
            sequence[:, prompt_len + step] = chosen
            lengths += (~finished).long()
            finished |= chosen == eos_token_id

            if bool(finished.all()) or step + 1 >= max_new_tokens:
                break

            position = prompt_len + step
            if use_cache:
                output = model(
                    sequence[:, position : position + 1],
                    positions=torch.arange(position, position + 1, device=device),
                    use_cache=True,
                    return_logits=True,
                )
            else:
                output = model(sequence[:, : position + 1], return_logits=True)
            next_logits = output.logits[:, -1, :]
    finally:
        model.reset_cache()
        del caches

    return GenerationOutput(
        token_ids=sequence[:, : prompt_len + max_new_tokens], lengths=lengths, finished=finished
    )
