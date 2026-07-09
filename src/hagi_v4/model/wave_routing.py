"""Wave routing — frequency-domain resonance architecture (V7).

Replaces transformer attention with sound-physics-based routing:
  - Frequency decomposition (OFDM subcarriers, Hz analog)
  - Resonance / sympathy (matched filter, constructive interference)
  - Meaning extraction (decoded signal from resonance patterns)
  - 90% linear, 10% nonlinear (linear block codes analog)

No QKV attention. No softmax. Instead:
  1. Hidden state → (amplitude, phase) per frequency band [linear]
  2. Resonance(i,j) = sum_k amp_i[k] * amp_j[k] * cos(phase_i - phase_j) [bilinear]
  3. meaning[i] = sum_j resonance(i,j) * value_j [linear]
  4. Sparse top-k resonance routing (selective activation)

Shannon mapping:
  - Hz frequencies = OFDM subcarriers (different frequency bands)
  - Sympathy = matched filter (max SNR when frequencies align)
  - Meaning = decoded information from resonance patterns
  - Sparse routing = IRA selective decoding (only high-info check nodes)
  - Linear (90%) = linear block codes (efficient encoding)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.norms import RMSNorm


class WaveRouter(nn.Module):
    """Frequency-domain resonance router — replaces attention.

    Maps hidden states to (amplitude, phase) frequency representation.
    Resonance between tokens = phase alignment weighted by amplitude.
    Sparse top-k routing = selective activation (only high-resonance connections).

    Complexity: O(T * k * n_freqs) where k << T (sparse).
    Attention is O(T^2 * H). WaveRouter is faster for long sequences.
    """

    def __init__(
        self,
        hidden_size: int,
        n_frequencies: int = 32,
        top_k_ratio: float = 0.25,
        n_kv_heads: int = 4,
        head_dim: int = 72,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_freqs = n_frequencies
        self.top_k_ratio = top_k_ratio

        self.n_heads = n_kv_heads
        self.head_dim = head_dim
        self.freq_dim = n_kv_heads * n_frequencies

        self.amp_proj = nn.Linear(hidden_size, self.freq_dim, bias=False)
        self.phase_proj = nn.Linear(hidden_size, self.freq_dim, bias=False)
        self.value_proj = nn.Linear(hidden_size, self.freq_dim, bias=False)
        self.out_proj = nn.Linear(self.freq_dim, hidden_size, bias=False)

        self.norm = RMSNorm(hidden_size, eps=1e-6)

        nn.init.normal_(self.amp_proj.weight, std=0.02)
        nn.init.normal_(self.phase_proj.weight, std=0.02)
        nn.init.normal_(self.value_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        x_n = self.norm(x)

        amps = torch.sigmoid(self.amp_proj(x_n))
        phases = torch.tanh(self.phase_proj(x_n)) * math.pi
        values = self.value_proj(x_n)

        amps = amps.view(B, T, self.n_heads, self.n_freqs)
        phases = phases.view(B, T, self.n_heads, self.n_freqs)
        values = values.view(B, T, self.n_heads, self.n_freqs)

        energy = amps.sum(dim=-1)

        k = max(1, min(int(T * self.top_k_ratio), T))

        _topk_vals, topk_idx = energy.topk(k, dim=1)
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.n_freqs)

        amps_sel = amps.gather(1, idx_exp)
        phases_sel = phases.gather(1, idx_exp)
        values_sel = values.gather(1, idx_exp)

        phase_diff = phases.unsqueeze(2) - phases_sel.unsqueeze(1)
        resonance = torch.cos(phase_diff) * amps.unsqueeze(2) * amps_sel.unsqueeze(1)
        resonance = resonance.sum(dim=-1)
        resonance = resonance / (resonance.sum(dim=2, keepdim=True) + 1e-8)

        out = torch.einsum("btkh,bkhf->bthf", resonance, values_sel)
        out = out.reshape(B, T, self.freq_dim)
        return self.out_proj(out), energy.mean(dim=-1)


class AllToAllConnection(nn.Module):
    """All-to-all layer connections with learned sparse gates.

    Every layer reads from all other layers. Gates control information flow.
    Most gates start near 0 (sparse), learned during training.

    Shannon mapping: full parity check matrix (LDPC with sparse activation).
    Only high-gate connections are "active" (selective decoding).

    Unlike DenseNet (sequential), this supports arbitrary layer ordering
    and selective activation (only compute high-gate connections).
    """

    def __init__(self, hidden_size: int, max_layers: int, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.max_layers = max_layers

        self.gates = nn.Parameter(torch.zeros(max_layers))
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(max_layers)])
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.nl_gate = nn.Parameter(torch.tensor(-5.0))

        for t in self.transforms:
            nn.init.normal_(t.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        all_outputs: list[torch.Tensor | None],
    ) -> torch.Tensor:
        agg = x.new_zeros_like(x)

        for j, (h_j, gate) in enumerate(zip(all_outputs, self.gates)):
            if h_j is not None and j != self.layer_idx:
                g = torch.sigmoid(gate)
                agg = agg + g * self.transforms[j](h_j)

        agg = agg + x

        nl = torch.sigmoid(self.nl_gate) * F.gelu(self.norm(agg))
        return agg + nl


class RouteSelector(nn.Module):
    """Selective route activation — SSD-style lazy loading.

    Computes which layers to activate based on input frequency signature.
    Only activated layers are computed (sparse activation).
    Non-activated layers are skipped (saved on SSD, not loaded to RAM).

    Shannon mapping: IRA (irregular repeat-accumulate) selective decoding.
    Only decode check nodes with high information content.

    Usage:
        probs = route_selector(x.mean(dim=1))  # [B, n_layers]
        for i, layer in enumerate(layers):
            if probs[:, i].max() > threshold:
                x = layer(x, ...)
    """

    def __init__(self, hidden_size: int, n_layers: int, top_k: int | None = None) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, n_layers, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        scores = self.router(pooled)
        probs = torch.sigmoid(scores)

        if self.top_k is not None:
            k = min(self.top_k, self.n_layers)
            topk_vals, topk_idx = probs.topk(k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, topk_idx, 1.0)
            probs = probs * mask

        return probs
