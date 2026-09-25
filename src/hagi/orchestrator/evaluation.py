"""Fail-closed exact evaluation helpers for packed HAGI holdouts."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from hagi.config import Config


def evaluate_packed_tokens(
    model: nn.Module,
    cfg: Config,
    token_ids: Sequence[int],
    *,
    context_tokens: int | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, float | int]:
    """Score packed token IDs with exact full-alphabet CE.

    Every content token is scored once. Windows use a declared prefix token
    when configured; no text decoding or truncation occurs.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise ValueError("token_ids must be a sequence of integers")
    if not token_ids:
        raise ValueError("token_ids must be non-empty")
    ids = [int(value) for value in token_ids]
    if any(value < 0 or value >= cfg.model.vocab_size for value in ids):
        raise ValueError("token id is outside model vocabulary")
    max_seq_len = int(cfg.model.attention.max_seq_len)
    if max_seq_len < 2:
        raise ValueError("model max_seq_len must be at least 2")
    context = max_seq_len // 2 if context_tokens is None else int(context_tokens)
    if context < 1 or context >= max_seq_len:
        raise ValueError("context_tokens must be in [1, max_seq_len)")

    model.eval()
    target_device = torch.device(device)
    prefix = cfg.train.data.eos_token_id
    total_nll = 0.0
    total_tokens = 0
    position = 0
    while position < len(ids):
        end = min(len(ids), position + context)
        if prefix is None:
            if position == 0:
                # The first content token has no preceding context.
                position = 1
                continue
            input_ids = ids[position - 1:end]
            targets = ids[position:end]
        else:
            input_ids = [int(prefix), *ids[max(0, position - 1):end - 1]]
            targets = ids[position:end]
        if not targets:
            break
        with torch.no_grad():
            output = model(torch.tensor([input_ids], dtype=torch.long, device=target_device))
            selected = output.hidden[0, -len(targets):]
            value = float(model.head.exact_loss(selected, torch.tensor(targets, dtype=torch.long, device=target_device)))
        if not math.isfinite(value):
            raise ValueError("model produced non-finite exact CE")
        total_nll += value * len(targets)
        total_tokens += len(targets)
        position = end
    if total_tokens < 1:
        raise ValueError("scoring policy selected no tokens")
    return {"exact_ce": total_nll / total_tokens, "scored_token_count": total_tokens}


__all__ = ["evaluate_packed_tokens"]
