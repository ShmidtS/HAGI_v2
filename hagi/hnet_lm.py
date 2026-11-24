"""HNetByteLM — torch port of the "HAGI v2 Optimal" byte LM
(checkpoints/Надя.txt, class HagiOptimalByteLM; numpy, another agent).

Why: measured on 4 REAL disk files (4000 steps, same data, correct
metric target = byte right after the window):
    this architecture, d=24 h=48, 28k params   whole-file CE 1.531
    RopeAttentionByteLM, d=128 h=256, 132k     whole-file CE 2.18
Ablation (same budget): bigram skip -0.038, H-Net pooling -0.033,
2 heads vs 1 = 0.000, k in {32,64,128} +-0.02. HybridMuon vs Adam
with cosine: 1.6605 vs 1.6551 (noise) — the gain is the ARCHITECTURE,
so the optimizer is NOT ported (wave247 reached the same verdict
about Muon on our models). Only the forward is ported; autograd is
torch's.

Same interface as RopeAttentionByteLM (k, dim, hidden, params,
logits, full_logits, ce, train_guarded/train_certified via the mixin
below), so Unified can switch models by name.

Input convention (same as training): ctx (B, k) OLDEST-first.
"""
from __future__ import annotations

import math


class HNetByteLM:
    ARCH = "hnet"
    k = 64
    adapter_dim = None   # set in __init__ = dim (Rope: hidden)
    _TRAIN_MIXIN = None   # set below

    def __init__(self, dim: int = 24, hidden: int = 48, pool_r: int = 4,
                 n_heads: int = 1, bi_buckets: int = 256, d_bi: int = 12,
                 theta0: float = 10000.0, lam: float = 0.15, seed: int = 0,
                 **_ignored):
        import torch as t
        self.t = t
        self.dim, self.hidden, self.r, self.nh = dim, hidden, pool_r, n_heads
        self.hd = dim // n_heads
        self.adapter_dim = dim
        self.theta0, self.lam = theta0, lam
        self.bi_buckets, self.d_bi = bi_buckets, d_bi
        g = t.Generator().manual_seed(seed)
        def rn(*sh, s=1.0):
            return t.nn.Parameter(t.randn(*sh, generator=g) * s)
        d = dim
        self.emb = rn(256, d, s=0.08)
        self.loc_idx = (-8, -6, -4, -3, -2, -1)
        self.W_loc = rn(len(self.loc_idx) * d, d, s=1 / math.sqrt(len(self.loc_idx) * d))
        self.p_pool = t.nn.Parameter(t.full((pool_r, d), 1.0 / pool_r))
        self.W_q = rn(d, d, s=1 / math.sqrt(d)); self.W_k = rn(d, d, s=1 / math.sqrt(d))
        self.W_v = rn(d, d, s=1 / math.sqrt(d)); self.W_o = rn(d, d, s=1 / math.sqrt(d))
        self.rms1 = t.nn.Parameter(t.ones(d)); self.rms2 = t.nn.Parameter(t.ones(d))
        self.W_g = rn(d, hidden, s=1 / math.sqrt(d)); self.W_u = rn(d, hidden, s=1 / math.sqrt(d))
        self.W_d = rn(hidden, d, s=1 / math.sqrt(hidden))
        self.b2 = t.nn.Parameter(t.zeros(256))
        self.W_out = rn(d, 256, s=1 / math.sqrt(d))
        self._extra = []
        self.rms_eps = 1e-5
        if bi_buckets > 0:
            self.bi_emb = rn(bi_buckets, d_bi, s=0.05)
            self.W_bi = rn(d_bi, 256, s=1 / math.sqrt(d_bi))
            self._extra = [self.bi_emb, self.W_bi]
        M = self.k // pool_r
        half = self.hd // 2
        freq = theta0 ** (-2.0 * t.arange(half, dtype=t.float32) / max(1, self.hd))
        ang = t.arange(M, dtype=t.float32)[:, None] * freq[None, :]
        self.register_buffer = None
        self.cos_m = t.cos(ang)[None, :, None, :]
        self.sin_m = t.sin(ang)[None, :, None, :]
        self.alibi = (-lam * (M - 1 - t.arange(M, dtype=t.float32)))[None, None, :]
        self.blocks: list[dict] = []
        self.frozen_blocks: set = set()
        self.domain_adapter = None
        self.device = "cpu"
        self._resolve_device()

    # -- plumbing --------------------------------------------------------
    def _resolve_device(self):
        import os
        from hagi.lm import _resolve_device
        self.device = _resolve_device(self.t, self.dim, self.hidden)
        if self.device == "cuda":
            for p in self.params():
                p.data = p.data.to(self.device)
            self.cos_m, self.sin_m, self.alibi = (x.to(self.device) for x in (self.cos_m, self.sin_m, self.alibi))

    _NAMES = ("emb", "W_loc", "p_pool", "W_q", "W_k", "W_v", "W_o", "rms1", "rms2", "W_g", "W_u", "W_d", "b2", "W_out")

    def param_names(self):
        return list(self._NAMES) + (["bi_emb", "W_bi"] if self._extra else [])

    def params(self):
        return [self.emb, self.W_loc, self.p_pool, self.W_q, self.W_k, self.W_v, self.W_o,
                self.rms1, self.rms2, self.W_g, self.W_u, self.W_d, self.b2, self.W_out] + self._extra

    def _rope_tables(self, M: int, half: int):
        t = self.t
        freq = self.theta0 ** (-2.0 * t.arange(half, dtype=t.float32) / max(1, self.hd))
        ang = t.arange(M, dtype=t.float32)[:, None] * freq[None, :]
        return t.cos(ang)[None, :, None, :], t.sin(ang)[None, :, None, :]

    def grow_width(self, d_new: int, h_new: int, seed: int = 0) -> "HNetByteLM":
        """WAVE 263 — WIDTH growth that preserves the logits EXACTLY
        (Hagi/Theory/NadyaOptimal.lean, `wlogits_width_preserved`, n_heads = 1):
          rms_eps *= d0/d1;  rms1, rms2 *= sqrt(d0/d1), new coords 0;
          emb/pool/W_loc/W_k/W_v/W_o/W_g/W_u/W_d/W_out: zero-padded blocks;
          W_q: old block *= sqrt(hd1/hd0);
          RoPE: old frequencies kept, new pairs cos=sin=0 (padTo);
          b2, alibi unchanged.
        Not a heuristic: the theorem says logits_new(x) = logits_old(x)."""
        assert self.nh == 1, "width growth is proven for n_heads = 1 only"
        t = self.t
        d0, h0 = self.dim, self.hidden
        assert d_new >= d0 and h_new >= h0 and d_new % 2 == 0
        m = HNetByteLM(dim=d_new, hidden=h_new, pool_r=self.r, n_heads=1,
                       bi_buckets=self.bi_buckets, d_bi=self.d_bi,
                       theta0=self.theta0, lam=self.lam, seed=seed)
        rs = math.sqrt(d0 / d_new)
        with t.no_grad():
            m.emb.zero_(); m.emb[:, :d0] = self.emb
            m.p_pool.fill_(1.0 / self.r); m.p_pool[:, :d0] = self.p_pool
            m.rms1.zero_(); m.rms1[:d0] = self.rms1 * rs
            m.rms2.zero_(); m.rms2[:d0] = self.rms2 * rs
            m.b2.copy_(self.b2)
            n_loc = len(self.loc_idx)
            m.W_loc.zero_()
            for si in range(n_loc):
                m.W_loc[si * d_new:si * d_new + d0, :d0] = self.W_loc[si * d0:(si + 1) * d0, :]
            qsc = math.sqrt(d_new / d0)
            for name, sc in (("W_q", qsc), ("W_k", 1.0), ("W_v", 1.0)):
                getattr(m, name).zero_(); getattr(m, name)[:d0, :d0] = getattr(self, name) * sc
            m.W_o.zero_(); m.W_o[:d0, :d0] = self.W_o
            m.W_g.zero_(); m.W_g[:d0, :h0] = self.W_g
            m.W_u.zero_(); m.W_u[:d0, :h0] = self.W_u
            m.W_d.zero_(); m.W_d[:h0, :d0] = self.W_d
            m.W_out.zero_(); m.W_out[:d0, :] = self.W_out
            if self._extra:
                m.bi_emb.copy_(self.bi_emb); m.W_bi.copy_(self.W_bi)
        # RoPE: old frequencies exactly as before, new pairs zero (padTo)
        half0, half1 = d0 // 2, d_new // 2
        c0, s0 = self._rope_tables(m.k // m.r, half0)
        M = m.k // m.r
        zeros = t.zeros((1, M, 1, half1 - half0), device=c0.device)
        m.cos_m = t.cat([c0, zeros], dim=-1); m.sin_m = t.cat([s0, zeros], dim=-1)
        m.alibi = self.alibi.clone()
        m.rms_eps = self.rms_eps * (d0 / d_new)
        m.device = self.device
        if self.device == "cuda":
            for p_ in m.params():
                p_.data = p_.data.to(self.device)
            m.cos_m, m.sin_m, m.alibi = (x.to(self.device) for x in (m.cos_m, m.sin_m, m.alibi))
        return m

    def grow(self, seed: int = 0) -> int:      # interface compat; no-op identity
        return len(self.blocks)

    def _block_add(self, a, h):
        return h

    # -- model -----------------------------------------------------------
    def _rms(self, x, g, eps=None):
        eps = self.rms_eps if eps is None else eps
        return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt() * g

    def _rot(self, x, cos, sin):
        half = self.hd // 2
        if half == 0:
            return x
        x0, x1 = x[..., 0:2 * half:2], x[..., 1:2 * half:2]
        out = x.clone()
        out[..., 0:2 * half:2] = x0 * cos - x1 * sin
        out[..., 1:2 * half:2] = x0 * sin + x1 * cos
        return out

    def logits(self, ctx):
        t = self.t
        if ctx.device != self.device:
            ctx = ctx.to(self.device)
        B, k = ctx.shape
        d, r, nh, hd = self.dim, self.r, self.nh, self.hd
        M = k // r
        E = self.emb[ctx]
        x0 = E[:, list(self.loc_idx), :].reshape(B, len(self.loc_idx) * d) @ self.W_loc
        C = E[:, -M * r:, :].reshape(B, M, r, d).mul(self.p_pool[None, None]).sum(2)
        C_n = self._rms(C, self.rms1)
        x0_n = self._rms(x0, self.rms1)
        Q = (x0_n @ self.W_q).reshape(B, 1, nh, hd)
        K = (C_n @ self.W_k).reshape(B, M, nh, hd)
        V = (C_n @ self.W_v).reshape(B, M, nh, hd)
        Qr = self._rot(Q, self.cos_m[:, -1:], self.sin_m[:, -1:])
        Kr = self._rot(K, self.cos_m, self.sin_m)
        sc = t.einsum("bmhd,bnhd->bhm", K, Qr) / math.sqrt(hd) + self.alibi      # (B,nh,M)
        w = t.softmax(sc, dim=-1)
        attn = t.einsum("bhm,bmhd->bhd", w, V).reshape(B, d)
        h1 = x0 + attn @ self.W_o
        h1_n = self._rms(h1, self.rms2)
        gp = h1_n @ self.W_g
        h2 = h1 + ((gp * t.sigmoid(gp)) * (h1_n @ self.W_u)) @ self.W_d
        ad = getattr(self, "domain_adapter", None)
        if ad is not None:                       # same residual form as Rope: h + (h@A^T)@B^T
            At = t.as_tensor(ad.A, device=h2.device, dtype=h2.dtype)
            Bt = t.as_tensor(ad.B, device=h2.device, dtype=h2.dtype)
            h2 = h2 + (h2 @ At.T) @ Bt.T
        self._last_h2 = h2
        lg = h2 @ self.W_out + self.b2
        if self._extra:
            idx = (ctx[:, -2].long() * 131 + ctx[:, -1].long()) % self.bi_buckets
            lg = lg + self.bi_emb[idx] @ self.W_bi
        return lg

    def features(self, ctx):
        """Residual stream h2 (B, dim) of the last forward — the features
        the output head reads (semantic-distill head regresses them)."""
        if ctx.device != self.device:
            ctx = ctx.to(self.device)
        with self.t.no_grad():
            self.logits(ctx)
        return self._last_h2

    def full_logits(self, ctx):
        """Last-position logits for every prefix (B, L) — via L forward
        calls on prefixes (generation only; training uses logits)."""
        return t_stack_logits(self, ctx)

    def train_certified(self, data: bytes, epochs: int = 60,
                        holdout: float = 0.1, batch: int = 64,
                        log_every: int = 0, holdout_seed: int | None = None,
                        lr: float | None = None) -> dict:
        """Unified-compatible training. No certified-refine phase: the
        Rope refine (_certified_finite_step) is a JVP of THAT forward;
        for HNet the step is accepted by the measured holdout gate only
        (stated honestly: selection, not a certificate)."""
        r = self.train_guarded(data, epochs=epochs, holdout=holdout, batch=batch,
                               holdout_seed=holdout_seed, **({} if lr is None else {"lr": lr}))
        r["refine"] = "none (hnet: selection gate, no certificate)"
        return r

    def adapter_head_step(self, ctx, nxt, lr: float = 1e-2):
        """Train ONLY the domain adapter (A, B) on this batch; trunk
        and head frozen (the HNet counterpart of Rope's method)."""
        ad = getattr(self, "domain_adapter", None)
        if ad is None:
            return
        t = self.t
        At = t.as_tensor(ad.A, device=self.device, dtype=t.float32).requires_grad_(True)
        Bt = t.as_tensor(ad.B, device=self.device, dtype=t.float32).requires_grad_(True)
        opt = t.optim.Adam([At, Bt], lr=lr)
        opt.zero_grad()
        t.nn.functional.cross_entropy(self.logits_with(ctx, At, Bt), nxt.to(self.device)).backward()
        opt.step()
        ad.A, ad.B = At.detach().cpu().numpy(), Bt.detach().cpu().numpy()

    def logits_with(self, ctx, At, Bt):
        """logits() with an explicit adapter pair (keeps domain_adapter untouched)."""
        prev = self.domain_adapter
        self.domain_adapter = _AB(At, Bt)
        try:
            return self.logits(ctx)
        finally:
            self.domain_adapter = prev

    def ce(self, ctx, nxt):
        nxt = nxt.to(ctx.device if ctx.device == self.device else self.device)
        return self.t.nn.functional.cross_entropy(self.logits(ctx), nxt)


def t_stack_logits(m, ctx):
    """(B, L, 256): logits of the next byte after every prefix. HNet
    needs exactly k columns (pooling by r), so short prefixes are
    LEFT-padded with byte 0 — the same construction train_guarded uses
    for the first k positions of a file."""
    t = m.t
    B, L = ctx.shape
    outs = []
    for i in range(1, L + 1):
        if i >= m.k:
            outs.append(m.logits(ctx[:, i - m.k:i]))
        else:
            pad = t.zeros((B, m.k - i), dtype=ctx.dtype, device=ctx.device)
            outs.append(m.logits(t.cat([pad, ctx[:, :i]], dim=1)))
    return t.stack(outs, dim=1)


# -- training mixin: reuse RopeAttentionByteLM's train loop ---------------
def _install():
    from hagi import lm as _lm
    R = _lm.RopeAttentionByteLM
    for name in ("train_guarded", "eval_holdout_ce", "_read_all_windows",
                 "generate", "generate_auto"):
        if hasattr(R, name) and not hasattr(HNetByteLM, name):
            setattr(HNetByteLM, name, getattr(R, name))
    HNetByteLM._TRAIN_MIXIN = R


_install()


class _AB:
    def __init__(self, A, B):
        self.A, self.B = A, B
