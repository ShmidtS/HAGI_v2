"""Wave block — non-transformer layer with frequency routing (V7).

Replaces TransformerBlock with:
  1. WaveRouter (frequency resonance instead of QKV attention)
  2. AllToAllConnection (every layer reads from all layers)
  3. Linear FFN (90% linear, 10% nonlinear gate)
  4. RouteSelector (selective layer activation)

No attention. No softmax. No causal mask. No RoPE.
Instead: frequency-domain resonance + sparse routing.

Shannon mapping:
  - WaveRouter = matched filter receiver (frequency-domain)
  - AllToAllConnection = full parity check matrix (LDPC)
  - Linear FFN = systematic code (linear encoding)
  - RouteSelector = IRA selective decoding (only high-info nodes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.wave_routing import WaveRouter, AllToAllConnection, RouteSelector
from hagi_v4.model.norms import RMSNorm


class WaveBlock(nn.Module):
    """Single wave routing layer.

    Architecture:
      x → WaveRouter (frequency resonance) → +residual
        → LinearFFN (90% linear, 10% gated GELU) → +residual
        → AllToAllConnection (reads from all layers)

    No attention. No RoPE. No positional encoding needed —
    frequency phases encode relative positions naturally.
    """

    def __init__(
        self,
        hidden_size: int,
        n_frequencies: int = 32,
        top_k_ratio: float = 0.25,
        n_kv_heads: int = 4,
        head_dim: int = 72,
        ffn_intermediate: int = 384,
        max_layers: int = 12,
        layer_idx: int = 0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.wave_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.wave_router = WaveRouter(
            hidden_size=hidden_size,
            n_frequencies=n_frequencies,
            top_k_ratio=top_k_ratio,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )

        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_up = nn.Linear(hidden_size, ffn_intermediate, bias=False)
        self.ffn_down = nn.Linear(ffn_intermediate, hidden_size, bias=False)
        self.ffn_gate = nn.Parameter(torch.tensor(-2.0))

        self.all_to_all = AllToAllConnection(hidden_size, max_layers, layer_idx)

        nn.init.normal_(self.ffn_up.weight, std=0.02)
        nn.init.normal_(self.ffn_down.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
        all_outputs: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        wave_out, energy = self.wave_router(self.wave_norm(x))
        x = x + wave_out

        ffn_in = self.ffn_norm(x)
        ffn_hidden = self.ffn_up(ffn_in)
        gate = torch.sigmoid(self.ffn_gate)
        ffn_out = self.ffn_down(gate * ffn_hidden + (1 - gate) * ffn_hidden.detach())
        x = x + ffn_out

        if all_outputs is not None:
            x = self.all_to_all(x, all_outputs)

        rp = energy.unsqueeze(-1) if energy.dim() == 2 else energy
        return x, torch.tensor(0.0, device=x.device), rp


class WaveStack(nn.Module):
    """Stack of WaveBlocks with selective route activation.

    All blocks share all-to-all connections. RouteSelector determines
    which blocks to compute for each input (sparse activation).

    SSD analog: non-activated blocks "sleep on SSD" — not computed,
    not loaded to RAM. Only the computed route is activated.

    Usage:
        wave_stack = WaveStack(hidden_size=288, n_layers=7, ...)
        z = wave_stack(z)  # automatically selects routes
    """

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        n_frequencies: int = 32,
        top_k_ratio: float = 0.25,
        n_kv_heads: int = 4,
        head_dim: int = 72,
        ffn_intermediate: int = 384,
        route_top_k: int | None = None,
        route_threshold: float = 0.5,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.route_threshold = route_threshold

        self.blocks = nn.ModuleList(
            [
                WaveBlock(
                    hidden_size=hidden_size,
                    n_frequencies=n_frequencies,
                    top_k_ratio=top_k_ratio,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    ffn_intermediate=ffn_intermediate,
                    max_layers=n_layers,
                    layer_idx=i,
                    norm_eps=norm_eps,
                )
                for i in range(n_layers)
            ]
        )

        self.route_selector = RouteSelector(hidden_size, n_layers, top_k=route_top_k)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        B = x.shape[0]
        route_probs = self.route_selector(x)

        all_outputs: list[torch.Tensor | None] = [None] * self.n_layers
        total_aux = x.new_zeros(())
        all_router_probs: list = []

        if self.training:
            for i, block in enumerate(self.blocks):
                all_outputs[i] = x
                x, aux, rp = block(x, cos, sin, modality_ids, all_outputs)
                total_aux = total_aux + aux
                all_router_probs.append(route_probs[:, i])
        else:
            for i, block in enumerate(self.blocks):
                if route_probs[:, i].max().item() > self.route_threshold:
                    all_outputs[i] = x
                    x, aux, rp = block(x, cos, sin, modality_ids, all_outputs)
                    total_aux = total_aux + aux
                    all_router_probs.append(route_probs[:, i])
                else:
                    all_outputs[i] = x.detach()

        return x, total_aux, all_router_probs
