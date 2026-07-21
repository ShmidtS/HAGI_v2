"""Shared helpers for the HAGI V21 codec package."""

from __future__ import annotations

import torch
import torch.nn as nn


def _block_call(
    block: nn.Module,
    x: torch.Tensor,
    attention_mode: str,
    prefix_len: torch.Tensor | int | None,
    soft_beta: float | None = None,
) -> torch.Tensor:
    """Helper for checkpoint-friendly invocation of AttentionBlock.

    torch.utils.checkpoint needs a callable with positional tensor args
    for reliable grad accumulation. This wraps the block call so the
    attention_mode/prefix_len/soft_beta kwargs are passed through.
    """
    return block(x, attention_mode=attention_mode, prefix_len=prefix_len, soft_beta=soft_beta)
