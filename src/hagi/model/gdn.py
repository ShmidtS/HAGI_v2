"""Gated DeltaNet — full-matrix delta-rule mixer (target-architecture core).

Reference-first:
- Yang et al., "Parallelizing Linear Transformers with the Delta Rule over
  Sequence Length" (Gated DeltaNet, arXiv:2412.06464) — canonical gated form
  S' = g_t·S + β_t·(v_t − (g_t·S k_t)ᵀk_t)... per-head matrix form used here:
      S ← g_t ⊙ S + β_t · (v_t − (g_t·S) k_t) k_tᵀ
- Lean SSOT (primes): Hagi/Audit/... ``gdnUpdate`` (GDNBridge): canonical
  gated form keeps the residual against the DECAYED state (as in the paper);
  ``gdn_onehot_is_colUpdate`` — g=1, β=1, k=e_i is an EXACT column write:
  S[i,:] ← v (Qwenputer RAM mode); ``gdn_transpose_is_ramWrite`` — bridge to
  GDNRAM abstract RAM semantics.

Two execution modes:
- neural (default): gates g_t = exp(-softplus(nu_t)) ∈ (0,1], β_t = σ(β-logit)
  learned per token; sequential chunked scan (training) + single-step decode.
- ram (exact): g forced to 1, β forced to 1, keys one-hot addresses → the
  state is a physical addressable memory; forward is deterministic.

Interface mirrors hagi.model.recurrent.Recurrent: forward(x[B,T,H]) ->
[B,T,H]; pre-norm in the module; out_proj zero-init + BranchScale so the
block starts as an exact identity (function-preserving insertion); named 2D
weights (in_proj/out_proj) compatible with per-tensor latent merge;
attach_state/detach_state incremental decoding like KVCache.

Training pass is a plain sequential loop over T (state is full-matrix, the
chunkwise WY representation of the paper is a future optimization; at the
HAGI scales here T ≤ 2k and the loop stays within budget on iGPU).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from hagi.model.ffn import BranchScale, linear
from hagi.model.norms import RMSNorm


@dataclass
class GDNConfig:
    """Per-layer GDN geometry (аналог RecurrentConfig).

    key_dim / value_dim default to hidden_size / hidden_size per head.
    n_heads state blocks are independent: S is [B, n_heads, key_dim, value_dim].
    """

    n_heads: int = 1
    key_dim: int = 0  # 0 = hidden_size
    value_dim: int = 0  # 0 = hidden_size
    gate_bias: float = 1.0  # init bias: softplus(1)≈1.31 → g≈e^-1.31 (paper-like)


class GatedDeltaNet(nn.Module):
    """Gated delta-rule mixer with exact RAM mode.

    State per head: S [B, K, V]. Per token t (per head):
        k_t = W_k x_t  [K],  v_t = W_v x_t  [V]
        g_t = exp(-softplus(nu_t))  [K]  (per-key decay, neural mode)
        beta_t = sigmoid(beta_t logit)  [K]
        S ← g_t ⊙ S + β_t (v_t − (g_t ⊙ S) k_t) k_tᵀ
        y_t = W_o (S k_t)   [reads the memory before update? — paper reads
        after update via v − (gSk)k... canonical GDN output uses the post-
        update retrieval; we follow qwenputer convention: y_t = S_t k_t^read
        where the read key is a separate projection]
    """

    def __init__(
        self,
        hidden_size: int,
        cfg: GDNConfig | None = None,
        norm_eps: float = 1e-5,
        use_ternary: bool = True,
        residual_scale: float = 1.0,
        clamp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        cfg = cfg or GDNConfig()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        if cfg.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {cfg.n_heads}")
        self.hidden_size = hidden_size
        self.n_heads = int(cfg.n_heads)
        self.key_dim = int(cfg.key_dim) or hidden_size
        self.value_dim = int(cfg.value_dim) or hidden_size
        if self.key_dim % self.n_heads or self.value_dim % self.n_heads:
            raise ValueError("key_dim and value_dim must divide by n_heads")
        self.kh = self.key_dim // self.n_heads
        self.vh = self.value_dim // self.n_heads
        self.gate_bias = float(cfg.gate_bias)

        self.rec_norm = RMSNorm(hidden_size, eps=norm_eps)
        # Fused in_proj: [x_k, x_v, nu_gate, beta_gate] over hidden_size ->
        # kh*Hn + vh*Hn + kh + kh (gates per key-channel per head).
        self.gate_out = self.kh * self.n_heads  # per-key decay channels
        self.in_proj = linear(
            hidden_size,
            self.key_dim + self.value_dim + 2 * self.gate_out,
            use_ternary,
        )
        self.out_norm = RMSNorm(self.value_dim, eps=norm_eps)
        self.out_proj = linear(self.value_dim, hidden_size, use_ternary)
        nn.init.zeros_(self.out_proj.weight)  # function-preserving start
        # NOTE: retrieval is post-update S_t k_t inside the scan (read-after-
        # write; qwenputer load semantics). No separate read projection — a
        # dead parameter would violate the Lean gdnUpdate SSOT shape.
        self.branch_scale = BranchScale(residual_scale, clamp_ratio=clamp_ratio)

        self.mode: str = "neural"  # "neural" | "ram"
        self.wy_chunk: int = 32  # WY chunk length: decay budget keeps fp32 exact
        # (at 64, strong-decay chunks corrupt via clamp artifacts — measured
        # 0.02-0.18 vs sequential; 32 stays at fp32 eps on real + random gates)
        self._state: torch.Tensor | None = None
        self._state_enabled = False
        self._state_len = 0

    # -- mode control ---------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch neural <-> ram. ram = exact addressable memory (α=1, β=1)."""
        if mode not in ("neural", "ram"):
            raise ValueError(f"mode must be 'neural' or 'ram', got {mode!r}")
        self.mode = mode

    # -- incremental state (KVCache-compatible interface) ---------------------

    def attach_state(self) -> None:
        self._state_enabled = True
        self._state = None
        self._state_len = 0

    def detach_state(self) -> None:
        self._state_enabled = False
        self._state = None
        self._state_len = 0

    @property
    def length(self) -> int:
        return self._state_len if self._state_enabled else 0

    def reset_state(self) -> None:
        """Hard zero of the memory (ram semantics: power-on)."""
        self._state = None
        self._state_len = 0

    # -- core -----------------------------------------------------------------

    def _split(self, z: torch.Tensor, B: int, T: int):
        """Split fused in_proj into per-head k, v, nu, beta logits."""
        dk, dv, g = self.key_dim, self.value_dim, self.gate_out
        k = z[..., :dk].view(B, T, self.n_heads, self.kh)
        v = z[..., dk : dk + dv].view(B, T, self.n_heads, self.vh)
        nu = z[..., dk + dv : dk + dv + g].view(B, T, self.n_heads, self.kh)
        beta = z[..., dk + dv + g :].view(B, T, self.n_heads, self.kh)
        return k, v, nu, beta

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B,T,H] -> [B,T,H]; causality is structural (mask ignored)."""
        B, T, _ = x.shape
        z = self.in_proj(self.rec_norm(x))
        k, v, nu, beta = self._split(z, B, T)

        if self.mode == "ram":
            g = torch.ones_like(torch.sigmoid(nu))
            b = torch.ones_like(torch.sigmoid(beta))
            # one-hot address: argmax over key channels, exact per head
            idx = k.argmax(dim=-1)  # [B,T,Hn]
            k_ram = torch.zeros_like(k)
            k_ram.scatter_(-1, idx.unsqueeze(-1), 1.0)
            k_eff, v_eff, g_eff, b_eff = k_ram, v, g, b
        else:
            g_eff = torch.exp(-torch.nn.functional.softplus(nu + math.log(math.e - 1)))
            b_eff = torch.sigmoid(beta)
            # Canonical DeltaNet (arXiv:2412.06464, as in qwenputer/q35):
            # keys are l2-normalized per head. Without it the projection term
            # (gS)k kT scales the state by ~beta*||k||^2 per step (Kh=128 ->
            # ~115x) and the recurrence diverges exponentially (measured NaN
            # by step ~50). With one-hot keys (ram mode) the norm is already
            # 1 — normalization is a no-op there, RAM writes stay exact.
            k_eff = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)
            v_eff = v

        h = self._state if self._state_enabled else None
        if self.mode == "ram":
            # RAM: exact one-hot column semantics, sequential path
            outs, h = gdn_scan(k_eff, v_eff, g_eff, b_eff, h)
        else:
            # neural: chunkwise WY (exact vs sequential, verified) — matmul
            # shaped per chunk instead of a python loop over every step
            outs, h = wy_scan(k_eff, v_eff, g_eff, b_eff, self.wy_chunk, h)
        if self._state_enabled:
            self._state = h
            self._state_len += T
        y = outs.reshape(B, T, self.value_dim)
        return self.branch_scale(self.out_proj(self.out_norm(y.to(x.dtype))))

    def ram_write(self, address: int, vector: torch.Tensor) -> None:
        """Exact write S[address] ← vector in ram mode (gdn_onehot_is_colUpdate).

        Requires state attached and mode == 'ram'; vector [n_heads, vh].
        """
        if self.mode != "ram":
            raise ValueError("ram_write requires mode='ram'")
        if not self._state_enabled or self._state is None:
            raise ValueError("ram_write requires an attached, materialized state")
        if vector.shape != (self.n_heads, self.vh):
            raise ValueError(f"vector must be [{self.n_heads}, {self.vh}]")
        with torch.no_grad():
            onehot = torch.zeros(
                self.n_heads, self.kh, device=self._state.device, dtype=self._state.dtype
            )
            onehot[:, address % self.kh] = 1.0
            # S' = S + (v − S k) kᵀ with g=1, β=1 → exact column overwrite
            kv = (self._state * onehot[None, :, :, None]).sum(dim=2)  # [B,Hn,vh]
            self._state = self._state + (vector.to(self._state.dtype) - kv)[
                :, :, None, :
            ] * onehot[:, :, None]

    def ram_read(self, address: int) -> torch.Tensor:
        """Exact read: S[address] per head, [n_heads, vh]."""
        if self.mode != "ram" or self._state is None:
            raise ValueError("ram_read requires ram mode with materialized state")
        return self._state[0 if self._state.dim() == 4 else slice(None), :, address % self.kh, :].squeeze(0).clone()


def wy_scan(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk: int = 64,
    h0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunkwise WY form of the gated delta rule (Yang et al.
    arXiv:2412.06464) — exact vs :func:`gdn_scan` (verified to fp32 eps
    in tests), with per-chunk state updates instead of a per-step python
    loop over the full sequence. Keys must be pre-normalized (the runtime
    normalizes in :meth:`GatedDeltaNet.forward` / :func:`gdn_scan`).

    Args:
        k, v: [B, T, Hn, Kh]/[B, T, Hn, Vh]; g, beta: [B, T, Hn, Kh].
        h0: [B, Hn, Kh, Vh] or None.

    Returns:
        (outputs [B, T, Hn, Vh], final state [B, Hn, Kh, Vh]).
    """
    fd = torch.float32
    # normalize here (matches gdn_scan, which normalizes per step): callers may
    # pass raw keys; unnormalized keys made A entries explode (measured NaN at
    # t=68 with randn keys, norm ~16 per key).
    k = torch.nn.functional.normalize(k.to(fd), dim=-1, eps=1e-6)
    v = v.to(fd)
    g = g.to(fd)
    beta = beta.to(fd)
    B, T, Hn, Kh = k.shape
    Vh = v.shape[-1]
    h = (
        torch.zeros(B, Hn, Kh, Vh, device=k.device)
        if h0 is None
        else h0.to(fd).clone()
    )
    out_chunks: list[torch.Tensor] = []
    # cumulative log-decay: pair ratios become exp differences (Yang alg. 1);
    # batched pair matrix replaces per-pair outer products (OOM fix: the
    # naive double loop saved ~L^2/2 rank-1 graphs per chunk — at L=64,
    # D=384, B=4 that is ~5 GB per block per chunk kept for backward).
    # cumulative log-decay via cumsum of logs (Yang alg. 1): cumprod underflows
    # to 0 at long T (strong decay: measured 0.0 by step ~256), destroying the
    # chunk-relative differences; log-space cumsum is exact.
    cg = torch.cumsum(torch.log(g.clamp_min(1e-9)), dim=1)
    for s0 in range(0, T, chunk):
        s = s0
        e = min(s + chunk, T)
        L = e - s
        kb = k[:, s:e]                      # [B,L,Hn,Kh]
        vb = v[:, s:e]                      # [B,L,Hn,Vh]
        bb = beta[:, s:e]
        ktilde = bb * kb                    # write keys (Lean bt ⊙ k̂)
        # CHUNK-RELATIVE cumulative log-decay: the pair matrix and base terms
        # combine S0 (the chunk-start state) with in-chunk decays only — the
        # sequence-start decay is already folded into S0. Using the global cg
        # double-counted the pre-chunk decay (measured off-by-decay^chunk
        # error in chunk 2+; test_gdn_wy_scan_exact_vs_sequential catches it).
        cgL = cg[:, s:e]
        if s > 0:
            cgL = cgL - cg[:, s - 1 : s]
        # Numerics (Yang alg. 1 / FLA convention): the pair matrix factors
        # exp(cg') and exp(−cg') with a SHIFTED log-decay (shift cancels in
        # the difference) so neither factor overflows; base/outputs/state
        # use the TRUE chunk-relative exp(cgL) — bounded in (0, 1].
        cg_shift = cgL - cgL.max(dim=1, keepdim=True).values
        # fp32 overflow guard: a chunk of strong decay accumulates cg' as low
        # as −210 (64 steps × ln 0.038), making e^{−cg'} = inf. Clamp the
        # shifted decay at −70 (e^{70} ≈ 2.5e30, safely inside fp32): the
        # affected pair terms are true-ratio ≤ e^{−140} ≈ 0, so the clamped
        # value (≤ e^{−40} absolute) is negligible against O(1) outputs.
        # Exactness vs the sequential scan is preserved wherever no clamp is
        # active (verified by the exactness test with moderate decay).
        cg_shift = cg_shift.clamp_min(-70.0)
        # A[b,h,i,j] = Σ_k k̂_i,k k̃_j,k exp(cg_i,k − cg_j,k)
        A = torch.einsum(
            "blhk,bmhk->bhlm", kb * torch.exp(cg_shift), ktilde * torch.exp(-cg_shift)
        )                                    # [B,Hn,L,L]
        hstart = h
        hstart = h
        w_rows: list[torch.Tensor] = []
        for i in range(L):
            # dec_i·k̂_i = (e^{cg_i} ⊙ S0)·k̂_i
            base = (
                (torch.exp(cgL[:, i]).unsqueeze(-1) * h) * kb[:, i].unsqueeze(-1)
            ).sum(dim=2)                     # [B,Hn,Vh]
            if i:
                corr = torch.einsum(
                    "bhij,bhjv->bhiv",
                    A[:, :, i, :i].unsqueeze(2),
                    torch.stack(w_rows, dim=1).transpose(1, 2),
                ).squeeze(2)
            else:
                corr = torch.zeros(B, Hn, Vh, device=k.device)
            w_rows.append(vb[:, i] - base - corr)
        W = torch.stack(w_rows, dim=1)      # [B,L,Hn,Vh]
        chunk_out: list[torch.Tensor] = []
        for i in range(L):
            # y_i = (G_i⊙S0)·k̂_i + Σ_{j≤i} A[i,j] w_j
            base_y = torch.einsum(
                "bhk,bhkv->bhv", torch.exp(cgL[:, i]) * kb[:, i], hstart
            )
            acc_out = torch.einsum(
                "bhij,bhjv->bhiv",
                A[:, :, i, : i + 1].unsqueeze(2),
                W[:, : i + 1].transpose(1, 2),
            ).squeeze(2)
            chunk_out.append(base_y + acc_out)
        out_chunks.append(torch.stack(chunk_out, dim=1))
        # end state per-channel decay: (G_end/G_j)_k = exp(cg_end,k − cg_j,k)
        ratio_end = torch.exp(cgL[:, -1].unsqueeze(1) - cgL)  # [B,L,Hn,Kh]
        # S_end[k,v] = G_end,k S0[k,v] + Σ_j w_j,v k̃_j,k ratio_end[j,k]
        outer = torch.einsum(
            "bhjv,bjhk->bhkv", W.transpose(1, 2), ratio_end * ktilde
        )
        h = torch.exp(cgL[:, -1]).unsqueeze(-1) * hstart + outer
    outs = (
        torch.cat(out_chunks, dim=1)
        if out_chunks
        else torch.zeros(B, 0, Hn, Vh, device=k.device)
    )
    return outs, h


def gdn_scan(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential delta-rule scan.

    Args:
        k, v: [B, T, Hn, Kh]/[B, T, Hn, Vh]; g, beta: [B, T, Hn, Kh].
        h0: [B, Hn, Kh, Vh] or None.

    Returns:
        (outputs [B, T, Hn, Vh], final state [B, Hn, Kh, Vh]).
        Output at step t is the post-update retrieval S_t k_t (read-after-
        write; in ram mode k is one-hot so the output is exactly the
        addressed cell — the qwenputer load semantics).
    """
    floatd = torch.float32
    k = k.to(floatd)
    v = v.to(floatd)
    g = g.to(floatd)
    beta = beta.to(floatd)
    B, T, Hn, Kh = k.shape
    Vh = v.shape[-1]
    if h0 is None:
        h = k.new_zeros(B, Hn, Kh, Vh)
    else:
        h = h0.to(floatd).clone()
    outs = k.new_empty(B, T, Hn, Vh)
    for t in range(T):
        kt = torch.nn.functional.normalize(k[:, t], dim=-1, eps=1e-6)  # [B,Hn,Kh]
        vt = v[:, t]  # [B,Hn,Vh]
        gt = g[:, t].unsqueeze(-1)  # [B,Hn,Kh,1] decay per key channel
        bt = beta[:, t].unsqueeze(-1)
        dec = h * gt
        kv = (dec * kt.unsqueeze(-1)).sum(dim=2, keepdim=True)  # [B,Hn,1,Vh]
        h = dec + bt * (vt.unsqueeze(2) - kv) * kt.unsqueeze(-1)
        # post-update retrieval: (S_t k_t) — in ram mode reads the written cell
        outs[:, t] = (h * kt.unsqueeze(-1)).sum(dim=2)
    return outs, h
