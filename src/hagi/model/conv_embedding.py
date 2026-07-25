"""ConvEmbedding — factorized source encoder + causal pulse-shaping filter.

Replaces the monolithic ``nn.Embedding(V, H)`` table. In source-channel
separation terms the embedding is the memoryless source encoder: it maps
discrete symbols to a continuous transmit signal. Two principles:

  1. Low-rank source coding. ``token_compress`` (V x r) + ``token_expand``
     (r x H) is a rank-r factorization of the V x H lookup table. Natural
     language is low-rank: tokens share semantic structure, so a compact
     latent code r (64..256) suffices.
  2. Pulse-shaping filter. A CAUSAL depthwise Conv1d (left-pad only) locally
     mixes neighbouring transmitted symbols without future leak. The V25
     symmetric-pad conv leaked future tokens into ``hidden[t]`` (the
     future-leak root cause of prompt-independent garbage at inference) — the
     left-pad conv is the fix and is MANDATORY for a generative LM.

KV-cache awareness: the causal conv has a receptive field of ``kernel_size``,
so a single-token decode step would otherwise see only left_pad zeros instead
of the real prior tokens. :meth:`set_conv_cache` / :meth:`forward` cooperate
so a decode step is exact-equivalent to a full forward (the last ``left_pad``
token IDs are retained and prepended to the new token before the conv, with
only the new position's output returned).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hagi.model.norms import RMSNorm


class ConvEmbedding(nn.Module):
    """Factorized token embedding with a causal depthwise Conv1d pulse-shaping filter.

    Args:
        vocab_size: vocabulary size V.
        hidden_size: channel dimension H.
        factor_rank: inner rank r of the low-rank embedding.
        kernel_size: depthwise Conv1d kernel size (pulse-shaping filter).
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        factor_rank: int = 128,
        kernel_size: int = 5,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.factor_rank = factor_rank
        self.kernel_size = kernel_size

        self.token_compress = nn.Embedding(vocab_size, factor_rank)
        self.token_expand = nn.Linear(factor_rank, hidden_size, bias=False)
        nn.init.normal_(self.token_compress.weight, mean=0.0, std=1.0 / (factor_rank**0.5))
        nn.init.normal_(self.token_expand.weight, mean=0.0, std=1.0 / (factor_rank**0.5))

        self.left_pad = kernel_size - 1
        self.local_conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=kernel_size, padding=0, groups=hidden_size, bias=True
        )
        nn.init.normal_(self.local_conv.weight, mean=0.0, std=1.0 / (kernel_size**0.5))
        if self.local_conv.bias is not None:
            nn.init.zeros_(self.local_conv.bias)
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self._conv_cache: torch.Tensor | None = None  # [B, <=left_pad] last token IDs

    @property
    def weight(self) -> torch.Tensor:
        """Materialize the effective V x H table (diagnostics / weight-tying)."""
        with torch.no_grad():
            return self.token_expand.weight @ self.token_compress.weight.t()

    def reset_conv_cache(self) -> None:
        """Drop the conv-history cache."""
        self._conv_cache = None

    def _conv(self, h: torch.Tensor) -> torch.Tensor:
        """Causal depthwise conv + norm on ``[B, T, H]`` (left-pad only)."""
        h = h.transpose(1, 2)  # [B, H, T]
        h = F.pad(h, (self.left_pad, 0))
        h = self.local_conv(h)
        h = h.transpose(1, 2)
        return self.norm(h)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to the channel-space hidden state.

        When the conv-history cache is set (incremental decode), ``input_ids``
        is treated as the NEW tokens: the retained prior token IDs are
        prepended so the causal conv sees the real local context, and only the
        outputs for the new positions are returned.

        Args:
            input_ids: ``[B, T]`` long tensor.

        Returns:
            ``[B, T, H]`` continuous hidden state after causal local mixing.
        """
        use_cache = (not self.training) and self._conv_cache is not None
        if use_cache:
            cached = self._conv_cache.to(input_ids.device)
            full_ids = torch.cat([cached, input_ids], dim=1)
            t_new = input_ids.shape[1]
        else:
            full_ids = input_ids
            t_new = input_ids.shape[1]

        compressed = self.token_compress(full_ids)
        h = self.token_expand(compressed)
        h_out = self._conv(h)

        # Update the conv-history cache ONLY in eval (incremental decode). In
        # training the cache stays None — every batch is a fresh full sequence
        # and prepending stale IDs would corrupt the input.
        if not self.training:
            keep = min(self.left_pad, full_ids.shape[1])
            self._conv_cache = full_ids[:, -keep:].detach().clone()

        if use_cache and t_new < h_out.shape[1]:
            return h_out[:, -t_new:]
        return h_out
