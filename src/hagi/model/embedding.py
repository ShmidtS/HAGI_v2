"""Source coder: token codebook plus a causal transmit filter.

Two components, both with a direct channel interpretation.

**Codebook.** A full-rank ``[V, H]`` table. V31 does *not* factor it. A rank-r
factorization ``V x r -> r x H`` is a lossy compression of the code assignment,
and when the same table is tied to the LM head that loss becomes a hard
information floor: fitting a known full-rank conditional distribution left
1.42 nats of irreducible KL at r=128 against 0.92 at r>=512 (measured). Paying
``V*H`` once and tying it to the receiver is strictly better than paying
``V*r + r*H`` twice and capping the achievable cross-entropy.

**Transmit filter.** A causal depthwise Conv1d over the sequence axis, which is
pulse shaping: it mixes each symbol with the ``k-1`` symbols before it, giving
the first attention layer a locally-smoothed signal instead of isolated
impulses. Left-padded only. A symmetric pad puts token ``t+1`` inside the
representation of position ``t``, and then next-token prediction is trivially
solvable at training time and impossible at inference — the failure looks like
"training works, generation is garbage".

The filter carries decode state: at a single-token decode step the receptive
field extends ``k-1`` positions into the past, so the module keeps the last
``k-1`` projected vectors and prepends them. That makes incremental decoding
bit-exact against a full forward pass, which :func:`tests` asserts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hagi.model.norms import RMSNorm


class SourceEncoder(nn.Module):
    """Token codebook + causal depthwise pulse-shaping filter.

    Args:
        vocab_size: alphabet size V.
        hidden_size: channel width H.
        conv_kernel: filter width; 1 disables the filter.
        norm_eps: RMSNorm epsilon.
        init_std: codebook initialization standard deviation.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        conv_kernel: int = 4,
        norm_eps: float = 1e-5,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if conv_kernel < 1:
            raise ValueError("conv_kernel must be >= 1")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.conv_kernel = conv_kernel
        self.left_pad = conv_kernel - 1

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

        if conv_kernel > 1:
            self.conv = nn.Conv1d(
                hidden_size, hidden_size, kernel_size=conv_kernel, padding=0, groups=hidden_size, bias=True
            )
            # Identity-dominant init: the filter starts as a near-passthrough of
            # the current symbol, so it cannot corrupt the code before it has
            # learned anything useful about the local context.
            with torch.no_grad():
                self.conv.weight.zero_()
                self.conv.weight[:, 0, -1] = 1.0
                self.conv.weight.add_(torch.randn_like(self.conv.weight) * 0.02)
                self.conv.bias.zero_()
            self.norm = RMSNorm(hidden_size, eps=norm_eps)
        else:
            self.conv = None
            self.norm = None

        self._state: torch.Tensor | None = None  # [B, left_pad, H] decode history

    @property
    def weight(self) -> torch.Tensor:
        """The ``[V, H]`` codebook (tied to the LM head when configured)."""
        return self.embedding.weight

    def reset_state(self) -> None:
        """Drop the filter's decode history."""
        self._state = None

    def forward(self, input_ids: torch.Tensor, use_state: bool = False) -> torch.Tensor:
        """Encode token IDs into the channel space.

        Args:
            input_ids: ``[B, T]`` token IDs.
            use_state: incremental decode. The retained history is prepended
                before the filter and only the new positions are returned; the
                history is then updated. Requires ``eval`` semantics from the
                caller (the state is not part of the autograd graph).

        Returns:
            ``[B, T, H]``.
        """
        h = self.embedding(input_ids)
        if self.conv is None:
            return h

        t_new = h.shape[1]
        if use_state and self._state is not None:
            h_in = torch.cat([self._state.to(h.dtype).to(h.device), h], dim=1)
        else:
            h_in = h

        x = h_in.transpose(1, 2)
        x = F.pad(x, (self.left_pad, 0))
        x = self.conv(x).transpose(1, 2)
        out = self.norm(x)

        if use_state:
            keep = min(self.left_pad, h_in.shape[1])
            self._state = h_in[:, h_in.shape[1] - keep :].detach()
            if out.shape[1] > t_new:
                out = out[:, -t_new:]
        return out

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, conv_kernel={self.conv_kernel}"
