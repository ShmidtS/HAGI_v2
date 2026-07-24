"""ConvEmbedding — factorized source encoder + pulse-shaping filter.

Replaces the monolithic ``nn.Embedding(V, H)`` table, which for a 49k
vocabulary dominated the parameter budget (91% in the V8 17.5M run produced a
97M checkpoint). In Source-Channel Separation terms the embedding is the
memoryless source encoder: it maps discrete symbols to a continuous transmit
signal. Two communication-theory principles drive the design:

  1. **Low-rank source coding.** ``token_compress`` (``V x r``) +
     ``token_expand`` (``r x H``) is a rank-``r`` factorization of the
     ``V x H`` lookup table. Natural language is low-rank: tokens share
     semantic structure, so a compact latent code ``r`` (64..256) suffices.
     Cost: ``V*r + r*H`` instead of ``V*H`` (e.g. 6.3M vs 44M for V=49154,
     H=512, r=128). The compressed code is the true source message; the
     expand layer is the modulation that lifts it into the channel space.

  2. **Pulse-shaping filter.** A causal depthwise Conv1d (kernel ``k``)
     locally mixes neighbouring transmitted symbols. In OFDM systems the
     pulse-shaping filter confines the transmit spectrum and reduces
     inter-symbol interference; here it gives the source encoder a local
     temporal context (FIR filter) before frequency-domain processing.
     Depthwise keeps the cost at ``O(H * k)`` — one filter per channel.

The semantic-erasure indicator (``semantic_unknown_embed``) is applied by the
caller before this module, so it sees the same continuous tensor regardless of
which tokens were erased; erasure is never represented by a token ID.

Weight tying with ``lm_head`` is handled by the model: ``lm_head`` is an
independent ``Linear(H, V)`` and is *not* tied to the factorized embedding,
because a low-rank source encoder does not have a natural inverse of the same
rank. A separate dense output projection keeps the decoder well-conditioned
and is a small fraction of the budget thanks to the smaller ``H``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hagi_v4.model.norms import RMSNorm


class ConvEmbedding(nn.Module):
    """Factorized token embedding with a depthwise Conv1d pulse-shaping filter.

    Args:
        vocab_size: vocabulary size ``V``.
        hidden_size: channel dimension ``H``.
        factor_rank: inner rank ``r`` of the low-rank embedding.
        kernel_size: depthwise Conv1d kernel size (pulse-shaping filter).
        norm_eps: RMSNorm epsilon for the post-conv normalization.
        init: unused; kept for config compatibility with the V8 embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        factor_rank: int = 128,
        kernel_size: int = 5,
        norm_eps: float = 1e-6,
        init: str = "normal",
    ) -> None:
        super().__init__()
        del init
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.factor_rank = factor_rank
        self.kernel_size = kernel_size

        self.token_compress = nn.Embedding(vocab_size, factor_rank)
        self.token_expand = nn.Linear(factor_rank, hidden_size, bias=False)
        nn.init.normal_(self.token_compress.weight, mean=0.0, std=1.0 / (factor_rank**0.5))
        nn.init.normal_(self.token_expand.weight, mean=0.0, std=1.0 / (factor_rank**0.5))

        # Causal depthwise Conv1d (left-pad only): hidden[t] depends on
        # input[0..t] and NOTHING from the future. This is MANDATORY for a
        # generative LM. The old symmetric padding (kernel_size // 2, centred,
        # receptive field [-k//2, +k//2]) leaked future tokens into the
        # embedding — so the attention never learned to condition the
        # next-token prediction on the real context (proven by the leak probe:
        # perturbing input[1] changed hidden[0] by 4.16). At inference the
        # last position has no future to read, so its prediction collapsed to
        # the marginal distribution of frequent tokens, independent of the
        # prompt — the prompt-independent word-salad symptom.
        left_pad = kernel_size - 1
        self.left_pad = left_pad
        self.local_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=0,
            groups=hidden_size,
            bias=True,
        )
        nn.init.normal_(self.local_conv.weight, mean=0.0, std=1.0 / (kernel_size**0.5))
        if self.local_conv.bias is not None:
            nn.init.zeros_(self.local_conv.bias)

        self.norm = RMSNorm(hidden_size, eps=norm_eps)

    @property
    def weight(self) -> torch.Tensor:
        """Materialize the effective ``V x H`` table for compatibility helpers.

        This is *not* used by the forward path — it exists so that callers that
        only need the effective embedding (e.g. for diagnostics or a tied
        ``lm_head`` when weight tying is explicitly enabled) can access a
        dense projection. The factorized path is always cheaper.
        """
        with torch.no_grad():
            return self.token_expand.weight @ self.token_compress.weight.t()

    def forward_with_erasure(
        self,
        input_ids: torch.Tensor,
        unknown_embed: torch.Tensor,
        semantic_unknown_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Channel-correct embedding with semantic erasure.

        Erasure is applied to the *compressed* code ``r`` (the true source
        message) before the expand and pulse-shaping steps, so the
        pulse-shaping filter never mixes an erased symbol with its
        neighbours. The erased positions receive the learned
        ``unknown_embed`` projected into the compressed space; non-erased
        positions keep their token code.

        Args:
            input_ids: ``[B, T]`` long tensor of token IDs.
            unknown_embed: ``[H]`` learned erasure indicator (already in the
                model's parameter space). Projected to rank ``r`` via
                ``token_compress`` pseudo-inverse is avoided: instead we
                expand it to ``H`` after erasure, matching the forward path.
            semantic_unknown_mask: ``[B, T]`` boolean erasure mask. ``None``
                is equivalent to "no erasure".

        Returns:
            ``[B, T, H]`` continuous hidden state after local temporal mixing.
        """
        compressed = self.token_compress(input_ids)  # [B, T, r]
        if semantic_unknown_mask is not None:
            mask = semantic_unknown_mask.to(compressed.device).unsqueeze(-1)
            # Project the learned erasure indicator (in H-space) into the
            # rank-``r`` compressed source code via the adjoint of the
            # expand layer. token_expand is Linear(r, H), so its weight has
            # shape [H, r] and the adjoint projection is unknown @ weight.
            unknown_h = unknown_embed.to(compressed.dtype).to(compressed.device)
            unknown_code = unknown_h @ self.token_expand.weight  # [r]
            compressed = torch.where(mask, unknown_code, compressed)
        h = self.token_expand(compressed)  # [B, T, H]
        h = h.transpose(1, 2)
        # Left-pad then causal conv: output[t] uses only input[0..t].
        h = torch.nn.functional.pad(h, (self.left_pad, 0))
        h = self.local_conv(h)
        h = h.transpose(1, 2)
        return self.norm(h)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to the channel-space hidden state.

        Args:
            input_ids: ``[B, T]`` long tensor of token IDs.

        Returns:
            ``[B, T, H]`` continuous hidden state after local temporal mixing.
        """
        compressed = self.token_compress(input_ids)  # [B, T, r]
        h = self.token_expand(compressed)  # [B, T, H]
        # Conv1d expects [B, C, T]; depthwise groups=H keeps it O(H*k).
        h = h.transpose(1, 2)
        # Left-pad then causal conv: output[t] uses only input[0..t].
        h = torch.nn.functional.pad(h, (self.left_pad, 0))
        h = self.local_conv(h)
        h = h.transpose(1, 2)
        return self.norm(h)
