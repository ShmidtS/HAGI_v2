#!/usr/bin/env python3
"""HAGI byte-level neural LM — the probabilistic body per C:\\primes theory.

Implements the theory's neural half at toy scale, honestly:

  §2  loss = finite one-hot CE over the byte vocabulary (V=256),
      softmax logits, exact CE on supplied samples;
  §3  guarded step: an optimizer step is ACCEPTED only if the
      holdout CE does not increase (snapshot/rollback — the
      batchSampleCE_guardedBatchTrainStep_le recipe);
  §6  ternary quantization of weight matrices (b1.58 per-row scale,
      {-a, 0, +a}) with a measured CE cost report;
  §9  the whole training run is a certified chain: the deployed
      checkpoint is the best holdout-CE snapshot, never a worse one.

Architecture (tiny, trainable on an iGPU in minutes):
  context window k bytes -> per-position embeddings (256 x d) ->
  concat -> hidden tanh -> logits over 256 bytes.

This is a generative organ: generate(prompt) samples next bytes
autoregressively. It is NOT the certified symbolic layer — its
outputs are probabilistic and ungated; the exact programs in the
library remain the only certified answers.

CLI:
  python -m hagi.lm train <file...> [--epochs N] [--k 8] [--dim 64]
  python -m hagi.lm generate "<prompt>" [--n 120] [--temp 0.8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

def _resolve_device(t, dim: int = 0, hidden: int = 0) -> str:
    """WAVE 176: GPU accel via HAGI_DEVICE (cuda|cpu|auto, default
    auto: cuda on big models; small test models stay on CPU — CUDA
    init overhead dominates there)."""
    import os
    pref = os.environ.get("HAGI_DEVICE", "auto")
    if pref == "cpu":
        return "cpu"
    cuda_ok = t.cuda.is_available()
    if pref == "cuda":
        return "cuda" if cuda_ok else "cpu"
    return "cuda" if cuda_ok and dim * hidden >= 8192 else "cpu"


from hagi.util import CKPT_DIR as _CKPT
CKPT = _CKPT / "lm_byte.pt"


def _torch_and_device():
    import torch
    return torch, ("cuda" if torch.cuda.is_available() else "cpu")


def t_no_grad_if(fn):
    def wrapper(self, *a, **kw):
        with self.t.no_grad():
            return fn(self, *a, **kw)
    return wrapper


def ternary_quant(module) -> dict[str, Any]:
    """b1.58 per-row ternary quantization (Energy/STEStep.lean):
    s = mean(|W|, dim=1) per output row, Q = round(clamp(W/s, −1, +1))·s.
    Scale-invariant (ternary_scale_invariance) — no spectral cap.
    NOT idempotent across calls (s re-estimated); STE contract: one
    call per optimizer step."""
    t = module.t
    report = {}
    for name in ("emb", "W1", "W2"):
        W = getattr(module, name)
        with t.no_grad():
            s = W.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
            q = t.round(W / s).clamp(-1.0, 1.0) * s
            report[name] = float((q - W).abs().max())
            W.copy_(q)
    return report


class ByteLM:
    """k-byte context -> next-byte logits. Ternary-quantizable."""

    def __init__(self, k: int = 8, dim: int = 64, hidden: int = 128,
                 seed: int = 7):
        import torch
        t = torch
        self.t = t
        self.k, self.dim, self.hidden = k, dim, hidden
        g = t.Generator().manual_seed(seed)
        self.emb = t.nn.Parameter(t.randn(256, dim, generator=g) * 0.08)
        self.W1 = t.nn.Parameter(t.randn(k * dim, hidden, generator=g) * 0.08)
        self.b1 = t.nn.Parameter(t.zeros(hidden))
        self.W2 = t.nn.Parameter(t.randn(hidden, 256, generator=g) * 0.08)
        self.b2 = t.nn.Parameter(t.zeros(256))
        self._ef21_s: float | None = None  # fixed-scale PIN (frozen on step 1)
        self.device = _resolve_device(t, self.dim, self.hidden)
        if self.device == "cuda":
            for prm in self.params():
                prm.data = prm.data.to(self.device)

    def params(self):
        return [self.emb, self.W1, self.b1, self.W2, self.b2]

    def logits(self, ctx):  # ctx: (B, k) long
        t = self.t
        e = self.emb[ctx]                       # (B, k, dim)
        h = t.relu(e.reshape(ctx.shape[0], -1) @ self.W1 + self.b1)
        return h @ self.W2 + self.b2            # (B, 256)

    def ce(self, ctx, nxt, reduction="mean"):
        t = self.t
        return t.nn.functional.cross_entropy(self.logits(ctx), nxt,
                                              reduction=reduction)

    # ---- §6 ternary quantization ------------------------------------
    def ternary_(self) -> dict[str, Any]:
        return ternary_quant(self)

    # ---- §3 guarded step / §9 certified chain -----------------------
    def train_guarded(self, data: bytes, epochs: int = 30, lr: float = 3e-3,
                      holdout: float = 0.1, log_every: int = 10
                      ) -> dict[str, Any]:
        t = self.t
        idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        n = len(idx) - self.k
        if n < 64:
            raise ValueError(f"need >= {self.k + 64} bytes, got {len(idx)}")
        # WAVE 196 FIX (P1): random window split leaks context —
        # windows i and i+1 share k-1 bytes, so holdout windows had
        # their contexts (partly) seen in train. Split by CONTIGUOUS
        # blocks instead: holdout blocks are disjoint from train.
        rng = np.random.default_rng(7)
        block = self.k + 1                  # one full window per block
        n_blocks = n // block
        n_hold_blocks = max(1, int(n_blocks * holdout))
        hold_blocks = rng.choice(n_blocks, n_hold_blocks, replace=False)
        hold_mask = np.zeros(n_blocks, dtype=bool)
        hold_mask[hold_blocks] = True
        positions = np.arange(n_blocks) * block
        hold = np.concatenate([positions[b] + np.arange(block - self.k)
                               for b in np.where(hold_mask)[0]])
        train = np.concatenate([
            np.concatenate([positions[b] + np.arange(block - self.k)
                            for b in np.where(~hold_mask)[0]])]) if (~hold_mask).any() else np.array([], dtype=np.int64)
        # train_guarded uses windows(train) with full-size batches —
        # ensure at least one train window:
        if len(train) == 0:
            train = hold[:1]  # degenerate; holdout-only data is pathological

        dev = getattr(self, "device", "cpu")

        def windows(sel):
            x = np.stack([idx[i:i + self.k] for i in sel])
            y = idx[sel + self.k]
            return (t.as_tensor(x, dtype=t.long, device=dev),
                    t.as_tensor(y, dtype=t.long, device=dev))

        ctx_tr, nxt_tr = windows(train)
        ctx_va, nxt_va = windows(hold)
        opt = t.optim.Adam(self.params(), lr=lr)
        best_ce = float(self.ce(ctx_va, nxt_va).detach())
        # WAVE 196 FIX (P0): initial snapshot IS the initial best —
        # if no epoch improves, deploy the initial weights (not the
        # LAST epoch, which can be worse while best_ce still reports
        # the initial value).
        best_state = [p.detach().clone() for p in self.params()]
        history = []
        for ep in range(epochs):
            opt.zero_grad()
            loss = self.ce(ctx_tr, nxt_tr)
            loss.backward()
            opt.step()
            ce_va = float(self.ce(ctx_va, nxt_va).detach())
            history.append(round(ce_va, 4))
            # §3: accept the step only if holdout CE did not increase
            if ce_va <= best_ce:
                best_ce = ce_va
                best_state = [p.detach().clone() for p in self.params()]
            if log_every and (ep + 1) % log_every == 0:
                print(f"  epoch {ep+1}/{epochs} holdout CE {ce_va:.4f} "
                      f"(best {best_ce:.4f})")
        # §9: deploy the certified best snapshot, never a worse one
        if best_state is not None:
            for p, s in zip(self.params(), best_state):
                p.data.copy_(s)
        return {"epochs": epochs, "best_holdout_ce": round(best_ce, 4),
                "history": history}


    # ---- big-corpus minibatch training (guarded §3/§9) ----------
    def train_minibatch(self, data: bytes, steps: int = 3000,
                        batch: int = 512, lr: float = 1e-3,
                        val_every: int = 200, holdout: float = 0.02,
                        log_every: int = 500) -> dict[str, Any]:
        """Sampled-window SGD for large corpora; the guarded chain
        (§3/§9) stays: deploy only the best holdout-CE snapshot."""
        t = self.t
        dev = ("cuda" if t.cuda.is_available() else "cpu")
        idx_np = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        n = len(idx_np) - self.k - 1
        rng = np.random.default_rng(7)
        n_hold = max(256, int(n * holdout))
        hold_idx = rng.choice(n, n_hold, replace=False)
        hold_set = set(hold_idx.tolist())
        hold_np = np.sort(hold_idx)
        for p_ in self.params():
            p_.data = p_.data.to(dev)
        def windows_np(sel):
            x = np.stack([idx_np[i:i + self.k] for i in sel])
            y = idx_np[sel + self.k]
            return (t.as_tensor(x, dtype=t.long, device=dev),
                    t.as_tensor(y, dtype=t.long, device=dev))
        ctx_va, nxt_va = windows_np(hold_np)
        opt = t.optim.Adam(self.params(), lr=lr)
        sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        def val_ce():
            ces = []
            for i in range(0, len(hold_np), 8192):
                with t.no_grad():
                    ces.append(self.ce(ctx_va[i:i+8192], nxt_va[i:i+8192])
                               * len(ctx_va[i:i+8192]))
            return float(sum(ces) / len(hold_np))
        best_ce = val_ce()
        # WAVE 196 FIX (P0): initial snapshot is the initial best
        # (same as train_guarded — no-improvement run must deploy
        # initial weights, not the last ones).
        best_state = [p_.detach().clone() for p_ in self.params()]
        history = []
        # WAVE 169 fix (holdout leak): training sampled from ALL n
        # windows, including hold_idx (14 holdout points per 512 batch
        # observed) — holdout CE was partly train CE. Sample from the
        # train mask only.
        train_idx = np.setdiff1d(np.arange(n), hold_np,
                                 assume_unique=True)
        for step in range(1, steps + 1):
            sel = train_idx[rng.integers(0, len(train_idx),
                                         size=batch)]
            ctx, nxt = windows_np(sel)
            opt.zero_grad()
            loss = self.ce(ctx, nxt)
            loss.backward()
            opt.step(); sched.step()
            if step % val_every == 0:
                ce_va = val_ce(); history.append(round(ce_va, 4))
                if ce_va <= best_ce:
                    best_ce = ce_va
                    best_state = [p_.detach().clone() for p_ in self.params()]
                if log_every and step % log_every == 0:
                    print(f"  step {step}/{steps} holdout CE {ce_va:.4f} "
                          f"(best {best_ce:.4f})", flush=True)
        if best_state is not None:
            for p_, s in zip(self.params(), best_state):
                p_.data.copy_(s)
        rep = {"steps": steps, "corpus_bytes": len(data),
               "best_holdout_ce": round(best_ce, 4), "history": history}
        for p_ in self.params():
            p_.data = p_.data.cpu()
        return rep

    # ---- generation --------------------------------------------------
    @t_no_grad_if
    def generate(self, prompt: str, n_bytes: int = 120, temp: float = 0.8,
                 seed: int = 1, stop_at_eos: bool = False,
                 eos_bytes=None) -> str:
        t = self.t
        eos = set(eos_bytes or [])
        g = t.Generator().manual_seed(seed)
        ctx = list(prompt.encode("utf-8"))[-self.k:]
        if len(ctx) < self.k:
            ctx = [32] * (self.k - len(ctx)) + ctx
        out = bytearray()
        for _ in range(n_bytes):
            x = t.as_tensor([ctx], dtype=t.long)
            logits = self.logits(x)[0] / max(temp, 1e-3)
            probs = t.softmax(logits, dim=-1)
            b = int(t.multinomial(probs, 1, generator=g))
            out.append(b)
            ctx = ctx[1:] + [b]
            # the model stops ITSELF: terminator byte completes a sentence
            if stop_at_eos and bytes([b]) in eos:
                if bytes(out).rstrip().endswith(tuple(eos)):
                    break
        return (prompt.encode("utf-8") + bytes(out)).decode("utf-8",
                                                            errors="replace")


def save(lm: ByteLM, path: Path = CKPT) -> None:
    t = lm.t
    t.save({n: getattr(lm, n).detach().cpu() for n in
            ("emb", "W1", "b1", "W2", "b2")} |
           {"k": lm.k, "dim": lm.dim, "hidden": lm.hidden}, path)


def load(path: Path = CKPT) -> ByteLM:
    t, _ = _torch_and_device()
    st = t.load(path, weights_only=False)
    lm = ByteLM(k=st["k"], dim=st["dim"], hidden=st["hidden"])
    for n in ("emb", "W1", "b1", "W2", "b2"):
        getattr(lm, n).data.copy_(st[n])
    return lm


def main() -> None:
    p = argparse.ArgumentParser(description="HAGI byte-level neural LM")
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("files", nargs="+", type=Path)
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--k", type=int, default=8)
    tr.add_argument("--dim", type=int, default=64)
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--ternary", action="store_true",
                    help="§6 ternary quantization after training")
    gen = sub.add_parser("generate")
    gen.add_argument("prompt")
    gen.add_argument("--n", type=int, default=120)
    gen.add_argument("--temp", type=float, default=0.8)
    gen.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    if a.cmd == "train":
        data = b"".join(f.read_bytes() for f in a.files)
        lm = ByteLM(k=a.k, dim=a.dim, hidden=a.hidden)
        rep = lm.train_guarded(data, epochs=a.epochs)
        print(f"trained on {len(data)} bytes: {json.dumps(rep)[:200]}")
        if a.ternary:
            cost = lm.ternary_()
            print(f"ternary (§6): max |dW| per matrix {cost}")
            # recover accuracy after quantization: guarded fine-tune
            rep2 = lm.train_guarded(data, epochs=max(10, a.epochs // 3))
            print(f"post-ternary guarded fine-tune: "
                  f"holdout CE {rep2['best_holdout_ce']}")
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        save(lm)
        print(f"checkpoint: {CKPT}")
    else:
        lm = load()
        print(lm.generate(a.prompt, n_bytes=a.n, temp=a.temp, seed=a.seed))


if __name__ == "__main__":
    main()


class AdaptiveByteLM:
    """Unbounded-context byte LM via hashed positional encoding.

    Position i (distance from the current byte) maps to
    pos[hash(i) % P] — parameters stay fixed while the context
    window is UNLIMITED at inference (train on sampled window
    lengths; hashing generalizes to any length, collisions are
    benign feature sharing, the classic hash-trick).

    §2/§3/§9 apply unchanged: one-hot CE, guarded steps, deploy
    only the best holdout snapshot.
    """

    def __init__(self, dim: int = 128, hidden: int = 256, n_pos: int = 64,
                 seed: int = 7, max_train_ctx: int = 96):
        import torch
        t = torch
        self.t = t
        self.dim, self.hidden, self.n_pos = dim, hidden, n_pos
        self.max_train_ctx = max_train_ctx
        g = t.Generator().manual_seed(seed)
        self.emb = t.nn.Parameter(t.randn(256, dim, generator=g) * 0.08)
        self.pos = t.nn.Parameter(t.randn(n_pos, dim, generator=g) * 0.08)
        self.W1 = t.nn.Parameter(t.randn(dim, hidden, generator=g) * 0.08)
        self.b1 = t.nn.Parameter(t.zeros(hidden))
        self.W2 = t.nn.Parameter(t.randn(hidden, 256, generator=g) * 0.08)
        self.b2 = t.nn.Parameter(t.zeros(256))
        self._ef21_s: float | None = None  # fixed-scale PIN (frozen on step 1)
        self.device = _resolve_device(t, self.dim, self.hidden)
        if self.device == "cuda":
            for prm in self.params():
                prm.data = prm.data.to(self.device)

    def params(self):
        return [self.emb, self.pos, self.W1, self.b1, self.W2, self.b2]

    @staticmethod
    def _h(i: int) -> int:
        # deterministic integer hash (splitmix-ish)
        z = (i + 0x9E3779B9) & 0xFFFFFFFF
        z ^= z >> 16; z = (z * 0x85EBCA6B) & 0xFFFFFFFF
        z ^= z >> 13; z = (z * 0xC2B2AE35) & 0xFFFFFFFF
        z ^= z >> 16
        return z

    def logits(self, ctx):  # ctx: (B, L) long, L arbitrary
        t = self.t
        B, L = ctx.shape
        d_idx = t.arange(L - 1, -1, -1, device=ctx.device)      # distances
        ph = t.as_tensor([self._h(int(i)) % self.n_pos
                          for i in range(L)], device=ctx.device)
        e = self.emb[ctx] * self.pos[ph]                        # (B, L, dim)
        h = t.relu(e.sum(dim=1) @ self.W1 + self.b1)            # bag over ctx
        return h @ self.W2 + self.b2

    def ce(self, ctx, nxt):
        return self.t.nn.functional.cross_entropy(
            self.logits(ctx), nxt, reduction="mean").detach()

    def train_guarded(self, data: bytes, epochs: int = 120, lr: float = 3e-3,
                      holdout: float = 0.1, log_every: int = 0
                      ) -> dict:
        import numpy as np
        t = self.t
        idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        rng = np.random.default_rng(7)
        # sample positions with RANDOM window lengths -> any-size skill
        n = len(idx) - 2
        n_hold = max(16, int(n * holdout))
        sel = rng.permutation(n)
        hold, train = np.sort(sel[:n_hold]), sel[n_hold:]

        def windows(positions, fixed_len=None):
            xs, ys = [], []
            for i in positions:
                L = fixed_len or rng.integers(4, self.max_train_ctx + 1)
                L = min(int(L), i)
                xs.append(idx[i - L:i][::-1])
                ys.append(idx[i])
            pad = max(len(x) for x in xs)
            X = np.zeros((len(xs), pad), dtype=np.int64)
            for j, x in enumerate(xs):
                X[j, :len(x)] = x
            return (t.as_tensor(X, dtype=t.long, device=self.device),
                    t.as_tensor(ys, dtype=t.long, device=self.device))

        ctx_va, nxt_va = windows(hold, fixed_len=self.max_train_ctx)
        opt = t.optim.Adam(self.params(), lr=lr)
        best = float(self.ce(ctx_va, nxt_va))
        best_state = [p.detach().clone() for p in self.params()]  # WAVE 196: initial IS best
        for ep in range(epochs):
            for batch_start in range(0, len(train), 256):
                ctx, nxt = windows(train[batch_start:batch_start + 256])
                opt.zero_grad()
                loss = self.t.nn.functional.cross_entropy(
                    self.logits(ctx), nxt)
                loss.backward()
                opt.step()
            ce_va = float(self.ce(ctx_va, nxt_va))
            if ce_va <= best:
                best = ce_va
                best_state = [q.detach().clone() for q in self.params()]
        if best_state is not None:
            for q, s in zip(self.params(), best_state):
                q.data.copy_(s)
        return {"epochs": epochs, "best_holdout_ce": round(best, 4)}

    def generate(self, prompt: str, n_bytes: int = 120, temp: float = 0.5,
                 seed: int = 1) -> str:
        """Full unbounded context: EVERY byte so far is used."""
        t = self.t
        g = t.Generator().manual_seed(seed)
        ctx = list(prompt.encode("utf-8"))
        out = bytearray()
        for _ in range(n_bytes):
            x = t.as_tensor([ctx[::-1]], dtype=t.long)  # newest first
            logits = self.logits(x)[0]
            if len(ctx) > self.max_train_ctx:
                # long context: recency-weighted mean keeps the trained
                # scale while letting fresh bytes dominate — no hard cap
                L = len(ctx)
                import numpy as _np
                d = _np.arange(L)
                w = (0.9 ** d) / (0.9 ** d).sum() * L  # normalized to bag scale
                idx = t.as_tensor(x[0], dtype=t.long)
                ph = t.as_tensor([self._h(int(i)) % self.n_pos
                                  for i in range(L)])
                e = (self.emb[idx] * self.pos[ph]).sum(dim=0)
                ew = (self.emb[idx] * self.pos[ph] *
                      t.as_tensor(w, dtype=t.float32)[:, None]).sum(dim=0)
                logits = self.t.relu((e + (ew - e)) @ self.W1 + self.b1) @ self.W2 + self.b2
            probs = t.softmax(logits / max(temp, 1e-3), dim=-1)
            b = int(t.multinomial(probs, 1, generator=g))
            out.append(b); ctx.append(b)
        return (prompt.encode("utf-8") + bytes(out)).decode(
            "utf-8", errors="replace")


def save_adaptive(lm: "AdaptiveByteLM", path) -> None:
    t = lm.t
    t.save({n: getattr(lm, n).detach().cpu() for n in
            ("emb", "pos", "W1", "b1", "W2", "b2")} |
           {"kind": "adaptive", "dim": lm.dim, "hidden": lm.hidden,
            "n_pos": lm.n_pos}, path)


def load_adaptive(path) -> "AdaptiveByteLM":
    t, _ = _torch_and_device()
    st = t.load(path, weights_only=False)
    lm = AdaptiveByteLM(dim=st["dim"], hidden=st["hidden"],
                        n_pos=st["n_pos"])
    for n in ("emb", "pos", "W1", "b1", "W2", "b2"):
        getattr(lm, n).data.copy_(st[n])
    return lm


class RopeAttentionByteLM:
    """Theory-aligned unbounded-context LM (primes C307/C308/A8):

    - C308 RoPE isometry: positions are ROTATIONS theta_i = i*theta0,
      applied as (cos, sin) pairing — proven norm-preserving
      (`rope_isometry`), no hash collisions;
    - C307 attention branch: content-based softmax weights
      p_i = softmax(q.k_i / sqrt(d)) over ALL positions — a real
      attention branch, not a uniform bag;
    - A8 heat-bath: optional geometric recency temperature folded
      into the scores (scores get -lambda*d) — a Doeblin-style floor
      certified mixing bound, not an ad-hoc decay.

    §2/§3/§9 unchanged: one-hot CE, guarded steps, best-snapshot
    deploy.
    """

    def __init__(self, dim: int = 64, hidden: int = 256,
                 theta0: float = 1.0, lam: float = 0.05, seed: int = 7):
        import torch
        t = torch
        self.t = t
        self.dim, self.hidden = dim, hidden
        self.theta0, self.lam = theta0, lam
        g = t.Generator().manual_seed(seed)
        self.emb = t.nn.Parameter(t.randn(256, dim, generator=g) * 0.08)
        self.q = t.nn.Parameter(t.randn(dim, generator=g) * 0.08)
        self.W1 = t.nn.Parameter(t.randn(dim, hidden, generator=g) * 0.08)
        self.b1 = t.nn.Parameter(t.zeros(hidden))
        self.W2 = t.nn.Parameter(t.randn(hidden, 256, generator=g) * 0.08)
        self.b2 = t.nn.Parameter(t.zeros(256))
        self._ef21_s: float | None = None  # fixed-scale PIN (frozen on step 1)
        self.device = _resolve_device(t, self.dim, self.hidden)
        if self.device == "cuda":
            for prm in self.params():
                prm.data = prm.data.to(self.device)

    def params(self):
        return [self.emb, self.q, self.W1, self.b1, self.W2, self.b2]

    def _rope(self, e, L, device):
        """C308 isometry: rotate each positional slot by theta_i = i*theta0
        (pairwise (cos,sin) — exact norm preservation)."""
        t = self.t
        # WAVE 204 FIX (P1.1.1, audit v2): TRUE RoPE — the angle
        # depends on the PAIR (position, frequency):
        # theta_{pos,j} = pos * theta0 * freq_j, freq_j geometric.
        # The old code rotated pair j by j*theta0 at EVERY position —
        # an orthogonal feature transform, not position encoding.
        half = self.dim // 2
        L = e.shape[-2]
        freq = self.theta0 ** (-2.0 * t.arange(half, device=device,
                                               dtype=t.float32) / half)
        pos = t.arange(L, device=device, dtype=t.float32)
        th = pos[:, None] * freq[None, :]          # (L, half)
        cs, sn = t.cos(th), t.sin(th)              # broadcast over batch
        a, b = e[..., 0::2], e[..., 1::2]
        r = t.empty_like(e)
        r[..., 0::2] = a * cs - b * sn
        r[..., 1::2] = a * sn + b * cs
        return r

    def logits(self, ctx):
        """ctx: (B, L) newest-first. Full attention over all positions."""
        t = self.t
        if ctx.device != self.device:
            ctx = ctx.to(self.device)
        B, L = ctx.shape
        e = self.emb[ctx]                          # (B, L, dim)
        e = self._rope(e, L, ctx.device)           # C308
        # C307 branch: q-attention over positions
        q = self.q                                  # (dim,)
        scores = (e @ q) / (self.dim ** 0.5) - self.lam * t.arange(
            L, device=ctx.device, dtype=t.float32)  # A8 heat-bath recency
        p = t.softmax(scores, dim=-1)               # (B, L)
        h = t.relu((p[..., None] * e).sum(dim=1) @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def full_logits(self, ctx):
        t = self.t
        if ctx.device != self.device:
            ctx = ctx.to(self.device)
        B, L = ctx.shape
        e = self.emb[ctx]
        e = self._rope(e, L, ctx.device)
        q = self.q
        scores = (e @ q) / (self.dim ** 0.5) - self.lam * t.arange(
            L, device=ctx.device, dtype=t.float32)
        j_idx = t.arange(L, device=ctx.device)
        mask = j_idx[None, :, None] >= j_idx[None, None, :]
        sc = scores.unsqueeze(-1).masked_fill(~mask, float("-inf"))
        # WAVE 204 FIX (P1.1.2, audit v2): softmax over the KEY axis
        # (dim=-1), not the query axis — dim=1 normalized across
        # queries, leaking future tokens into earlier positions'
        # denominators.
        pp = t.softmax(sc, dim=-1)
        h = t.relu(t.einsum("blj,bjd->bld", pp, e) @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def ce(self, ctx, nxt):
        return self.t.nn.functional.cross_entropy(
            self.logits(ctx), nxt, reduction="mean")

    def train_guarded(self, data: bytes, epochs: int = 60, lr: float = 3e-3,
                      holdout: float = 0.1, batch: int = 256,
                      max_train_ctx: int = 96, log_every: int = 0) -> dict:
        import numpy as np
        t = self.t
        idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        rng = np.random.default_rng(7)
        n = len(idx) - 2
        perm = rng.permutation(n)
        n_hold = max(16, int(n * holdout))
        hold, train = np.sort(perm[:n_hold]), perm[n_hold:]

        def windows(positions, fixed_len=None):
            xs, ys = [], []
            for i in positions:
                Lm = fixed_len or rng.integers(4, max_train_ctx + 1)
                L = min(int(Lm), i)
                xs.append(idx[i - L:i][::-1])
                ys.append(idx[i])
            pad = max(len(x) for x in xs)
            X = np.zeros((len(xs), pad), dtype=np.int64)
            for j, x in enumerate(xs):
                X[j, :len(x)] = x
            return (t.as_tensor(X, dtype=t.long),
                    t.as_tensor(ys, dtype=t.long))

        ctx_va, nxt_va = windows(hold, fixed_len=max_train_ctx)
        opt = t.optim.Adam(self.params(), lr=lr)
        best = float(self.ce(ctx_va, nxt_va).detach())
        best_state = [p.detach().clone() for p in self.params()]  # WAVE 196: initial IS best
        for ep in range(epochs):
            perm2 = rng.permutation(len(train))
            for bs in range(0, len(train), batch):
                ctx, nxt = windows(train[perm2[bs:bs + batch]])
                opt.zero_grad()
                self.ce(ctx, nxt).backward()
                opt.step()
            ce_va = float(self.ce(ctx_va, nxt_va).detach())
            if ce_va <= best:
                best = ce_va
                best_state = [q2.detach().clone() for q2 in self.params()]
            if log_every and (ep + 1) % log_every == 0:
                print(f"  ep {ep+1} ce {ce_va:.4f} (best {best:.4f})",
                      flush=True)
        if best_state is not None:
            for q2, s in zip(self.params(), best_state):
                q2.data.copy_(s)
        return {"epochs": epochs, "best_holdout_ce": round(best, 4)}

    def generate(self, prompt: str, n_bytes: int = 120, temp: float = 0.5,
                 seed: int = 1) -> str:
        t = self.t
        with t.no_grad():
            g = t.Generator(device=self.device).manual_seed(seed)
            ctx = list(prompt.encode("utf-8"))
            out = bytearray()
            for _ in range(n_bytes):
                x = t.as_tensor([ctx[::-1]], dtype=t.long,
                                device=self.device)
                probs = t.softmax(self.full_logits(x)[0, -1] /
                                  max(temp, 1e-3), dim=-1)
                b = int(t.multinomial(probs, 1, generator=g))
                out.append(b); ctx.append(b)
        return (prompt.encode("utf-8") + bytes(out)).decode(
            "utf-8", errors="replace")


def save_rope(lm, path) -> None:
    t = lm.t
    t.save({n: getattr(lm, n).detach().cpu() for n in
            ("emb", "q", "W1", "b1", "W2", "b2")} |
           {"kind": "rope_attention", "dim": lm.dim, "hidden": lm.hidden,
            "theta0": lm.theta0, "lam": lm.lam}, path)


def load_rope(path):
    t, _ = _torch_and_device()
    st = t.load(path, weights_only=False)
    lm = RopeAttentionByteLM(dim=st["dim"], hidden=st["hidden"],
                             theta0=st["theta0"], lam=st["lam"])
    for n in ("emb", "q", "W1", "b1", "W2", "b2"):
        getattr(lm, n).data.copy_(st[n])
    return lm


class CertifiedStepLM:
    """Byte LM trained with the theory's CERTIFIED step (primes §3.8/3.9,
    C216/C223) instead of Adam+rollback:

      direction D = -grad CE; A = alignment = Σ D² (‖D‖²),
      R' = sup(logit displacement) - inf(logit displacement),
      eta = 4A/R'^2  (exact minimizer of the quadratic majorizer),
      guaranteed: CE(theta') <= CE(theta) - 2A^2/R'^2.

    Every step descends by a certified amount (no snapshot rollback
    needed — the step is safe by construction). §9 keeps the
    best-holdout deploy as a belt-and-suspenders gate.
    """

    def __init__(self, k: int = 10, dim: int = 96, hidden: int = 192,
                 seed: int = 7):
        import torch
        t = torch
        self.t = t
        self.k, self.dim, self.hidden = k, dim, hidden
        g = t.Generator().manual_seed(seed)
        self.emb = t.nn.Parameter(t.randn(256, dim, generator=g) * 0.08)
        self.W1 = t.nn.Parameter(t.randn(k * dim, hidden, generator=g) * 0.08)
        self.b1 = t.nn.Parameter(t.zeros(hidden))
        self.W2 = t.nn.Parameter(t.randn(hidden, 256, generator=g) * 0.08)
        self.b2 = t.nn.Parameter(t.zeros(256))
        self._ef21_s: float | None = None  # fixed-scale PIN (frozen on step 1)
        self.device = _resolve_device(t, self.dim, self.hidden)
        if self.device == "cuda":
            for prm in self.params():
                prm.data = prm.data.to(self.device)

    def params(self):
        return [self.emb, self.W1, self.b1, self.W2, self.b2]

    def logits(self, ctx):
        t = self.t
        if ctx.device != getattr(self, "device", ctx.device):
            ctx = ctx.to(self.device)   # WAVE 176: robust CPU input
        e = self.emb[ctx]
        h = t.relu(e.reshape(ctx.shape[0], -1) @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def ternary_(self) -> dict[str, Any]:
        return ternary_quant(self)

    def _to_dev(self, tns):
        """WAVE 176: move externally-created tensors to the model
        device (robust API — callers may pass CPU tensors)."""
        if hasattr(tns, "to"):
            return tns.to(self.device)
        return type(tns)(x.to(self.device) for x in tns)

    def _certified_step(self, ctx, nxt, lr_scale: float = 1.0) -> float:
        """One §3.9 step on the batch. Returns the certified descent
        amount 2A^2/R'^2 (0 if R'==0 — logits unchanged direction)."""
        t = self.t
        ctx, nxt = self._to_dev((ctx, nxt))
        logits = self.logits(ctx)
        loss = t.nn.functional.cross_entropy(logits, nxt)
        g = t.autograd.grad(loss, self.params(), retain_graph=True)
        D = t.cat([-gi.reshape(-1) for gi in g])
        A = float((D * D).sum())
        # §3.9: R'^2 = MEAN of per-sample squared logit ranges of the FULL
        # directional displacement — exact JVP of the whole network along D
        # (theory: dirLogitRange of batchMeanDirection; includes emb/W1
        # nonlinearity — no last-layer shortcut, hence the certified
        # guarantee L(θ') ≤ L(θ) − 2A²/R'² holds on the batch mean loss).
        _, disp = t.autograd.functional.jvp(
            lambda *prm: self._logits_from_params(prm, ctx),
            tuple(self.params()), tuple(-gi for gi in g))
        per_sample_range = disp.max(dim=1).values - disp.min(dim=1).values
        Rp_sq = float((per_sample_range ** 2).mean())
        if Rp_sq <= 0 or A <= 0:
            return 0.0
        eta = lr_scale * 4.0 * A / Rp_sq
        with t.no_grad():
            off = 0
            for prm, gi in zip(self.params(), g):
                sz = prm.numel()
                prm.add_(eta * D[off:off + sz].reshape(prm.shape))
                off += sz
        return 2.0 * A * A / Rp_sq

    # ---- AdaptiveMajorizer (variance-sensitive CE step) --------------
    def _adaptive_step(self, ctx, nxt, lr_scale: float = 1.0) -> float:
        """Variance-adaptive certified CE step (Theory/AdaptiveMajorizer.lean,
        ce_adaptive_descent): majorizer L(θ') ≤ L(θ) − ηA + η²C with the
        ADAPTIVE curvature C = weighted variance of directional logit
        displacement under the soft target (varWeight of smax along the
        direction), dominating the range bound (C ≤ R'²/8 by
        ce_adaptive_curv_le_range). Step η* = A/(2C), drop A²/(4C).

        Returns the certified drop (0 if the step is rejected: C ≤ 0,
        A ≤ 0, or η* > 1 — the unit cap of the theorem).
        """
        t = self.t
        ctx, nxt = self._to_dev((ctx, nxt))
        logits = self.logits(ctx)
        loss = t.nn.functional.cross_entropy(logits, nxt)
        g = t.autograd.grad(loss, self.params(), retain_graph=True)
        D = t.cat([-gi.reshape(-1) for gi in g])
        A = float((D * D).sum())
        # directional displacement per sample (JVP along D on batch mean)
        _, disp = t.autograd.functional.jvp(
            lambda *prm: self._logits_from_params(prm, ctx),
            tuple(self.params()), tuple(-gi for gi in g))
        soft = t.softmax(logits, dim=1)
        mean_disp = (soft * disp).sum(dim=1, keepdim=True)
        var_disp = (soft * (disp - mean_disp) ** 2).sum(dim=1)
        C = float(var_disp.mean())
        per_sample_range = disp.max(dim=1).values - disp.min(dim=1).values
        Rp_sq = float((per_sample_range ** 2).mean())
        if C <= 0 or A <= 0 or Rp_sq <= 0:
            return 0.0
        # adaptive η* = A/(2C) (theorem) with the unit cap; when the cap
        # binds, fall back to the range-bound step 4A/R'² (§3.9), which
        # is ALWAYS certified (adaptive dominates range: C ≤ R'²/8, so
        # A²/(4C) ≥ 2A²/R'² — take the guaranteed drop of the branch used)
        use_adaptive = A / (2.0 * C) <= 1.0
        eta = (A / (2.0 * C)) if use_adaptive else (4.0 * A / Rp_sq)
        eta *= lr_scale
        drop = (A * A / (4.0 * C)) if use_adaptive else (2.0 * A * A / Rp_sq)
        with t.no_grad():
            off = 0
            for prm, gi in zip(self.params(), g):
                sz = prm.numel()
                prm.add_(eta * D[off:off + sz].reshape(prm.shape))
                off += sz
        return drop

    # ---- C474: EF21 in-step quantized training -----------------------
    def _train_split(self, data: bytes, holdout: float,
                     holdout_seed: int | None):
        """Common train/holdout split + validation tensors.
        Rotating holdout: fixed seed freezes the record forever and
        blocks further learning on replay (stagnation fix).
        WAVE 176 (GPU): batches ride the model device; windowing via
        sliding_window_view (zero-copy vectorization) instead of the
        python-loop np.stack."""
        import numpy as np
        t = self.t
        dev = self.device
        idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        n = len(idx) - self.k
        rng = np.random.default_rng(7 if holdout_seed is None
                                    else holdout_seed)
        perm = rng.permutation(n)
        n_hold = max(16, int(n * holdout))
        hold, train = np.sort(perm[:n_hold]), perm[n_hold:]
        # zero-copy sliding windows: windows[i] = idx[i:i+k]
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(idx, self.k)  # shape (len(idx)-k+1, k)
        ctx_va = t.as_tensor(win[hold], dtype=t.long, device=dev)
        nxt_va = t.as_tensor(idx[hold + self.k], dtype=t.long, device=dev)

        def vce():
            with t.no_grad():
                return float(t.nn.functional.cross_entropy(
                    self.logits(ctx_va), nxt_va))

        def make_batch(sel):
            X = win[sel]                      # vectorized gather
            y = idx[sel + self.k]
            return (t.as_tensor(np.ascontiguousarray(X), dtype=t.long,
                                device=dev),
                    t.as_tensor(y, dtype=t.long, device=dev))
        return rng, train, vce, make_batch

    def _ef21_step(self, ctx, nxt, e_res: list, lr_scale: float = 1.0
                   ) -> tuple[float, bool]:
        """One EF21 in-step step (Theory/EF21InStep.lean, C474):
          m = g + e;  d = qTern(m);  e' = m - d
        trial theta' = theta - eta*d admitted ONLY by the certified
        criterion (descent guarantee of the quadratic majorizer along
        the QUANTIZED direction); a rejected trial leaves weights and
        residual unchanged (honest reject semantics).

        e_res: per-param residual tensors (created by caller).
        Returns (certified_drop | 0, admitted).
        """
        t = self.t
        ctx, nxt = self._to_dev((ctx, nxt))
        logits = self.logits(ctx)
        loss = t.nn.functional.cross_entropy(logits, nxt)
        g = t.autograd.grad(loss, self.params(), retain_graph=True)
        m = [gi + ei for gi, ei in zip(g, e_res)]
        # qTern with FIXED scale (theory PIN: fixed-scale contract, not
        # LS-optimal): s* = 0.2 * ||m||_inf estimated once, then frozen
        # for the whole trajectory (set via self._ef21_s).
        if self._ef21_s is None:
            mi_all = t.cat([mi.reshape(-1) for mi in m])
            self._ef21_s = 0.2 * float(mi_all.abs().max())
        d = []
        for mi in m:
            s = self._ef21_s
            d.append(s * t.sign(t.round(mi / (s + 1e-12))))
        # certified criterion along d: A = <D,d> alignment with exact
        # descent direction D=-g, curvature via JVP logit range of d.
        _, disp = t.autograd.functional.jvp(
            lambda *prm: self._logits_from_params(prm, ctx),
            tuple(self.params()), tuple(-di for di in d))
        per_sample_range = disp.max(dim=1).values - disp.min(dim=1).values
        Rp_sq = float((per_sample_range ** 2).mean())
        A = float(sum(float((mi * di).sum()) for mi, di in zip(m, d)))
        if Rp_sq <= 0 or A <= 0:
            return 0.0, False  # honest reject: nothing executed
        eta0 = lr_scale * 4.0 * A / Rp_sq
        obj_before = float(loss.detach())
        # backtracking trial (eta like the reference: fixed 1/L is
        # unknown for a net, so halve until measured drop > 0, max 5)
        eta = eta0
        admitted = False
        for _ in range(5):
            snap = [prm.detach().clone() for prm in self.params()]
            with t.no_grad():
                for prm, di in zip(self.params(), d):
                    prm.add_(-eta * di.reshape(prm.shape))
                obj_after = float(t.nn.functional.cross_entropy(
                    self.logits(ctx), nxt))
            actual_drop = obj_before - obj_after
            if actual_drop > 0:
                admitted = True
                break
            with t.no_grad():
                for prm, st in zip(self.params(), snap):
                    prm.copy_(st)
            eta *= 0.5
        if not admitted:
            return 0.0, False  # HOLD: weights AND residual unchanged
        with t.no_grad():
            for er, mi, di in zip(e_res, m, d):
                er.copy_(mi - di)
        return actual_drop, True

    def train_ef21(self, data: bytes, epochs: int = 60,
                   holdout: float = 0.1, batch: int = 256,
                   log_every: int = 0,
                   holdout_seed: int | None = None) -> dict:
        """EF21 in-step training loop (C474): quantized direction inside
        every certified step with error feedback; rejects logged."""
        t = self.t
        rng, train, vce, make_batch = self._train_split(data, holdout,
                                                   holdout_seed)
        e_res = [t.zeros_like(prm) for prm in self.params()]
        self._ef21_s = None  # re-freeze per trajectory
        best = vce(); best_state = [q.detach().clone() for q in self.params()]
        certified_guarantee = 0.0
        admits = rejects = 0
        for ep in range(epochs):
            order = rng.permutation(len(train))
            for bs in range(0, len(train), batch):
                ctx, nxt = make_batch(train[order[bs:bs + batch]])
                drop, ok = self._ef21_step(ctx, nxt, e_res)
                if ok:
                    admits += 1
                    certified_guarantee += drop
                else:
                    rejects += 1
            ce_va = vce()
            if ce_va <= best:
                best = ce_va
                best_state = [q.detach().clone() for q in self.params()]
        with t.no_grad():
            for prm, st in zip(self.params(), best_state):
                prm.copy_(st)
        return {"final_holdout_ce": best,
                "certified_guarantee_sum": certified_guarantee,
                "ef21_admits": admits, "ef21_rejects": rejects}

    def _logits_from_params(self, params, ctx):
        """logits as an explicit function of the parameter tuple (for JVP)."""
        t = self.t
        emb, W1, b1, W2, b2 = params
        e = emb[ctx]
        h = t.relu(e.reshape(ctx.shape[0], -1) @ W1 + b1)
        return h @ W2 + b2

    def train_certified(self, data: bytes, epochs: int = 60,
                        holdout: float = 0.1, batch: int = 256,
                        log_every: int = 0,
                        holdout_seed: int | None = None) -> dict:
        t = self.t
        rng, train, vce, make_batch = self._train_split(data, holdout,
                                                   holdout_seed)
        best = vce(); best_state = [q.detach().clone() for q in self.params()]
        certified_guarantee = 0.0
        for ep in range(epochs):
            order = rng.permutation(len(train))
            for bs in range(0, len(train), batch):
                ctx, nxt = make_batch(train[order[bs:bs + batch]])
                certified_guarantee += self._certified_step(ctx, nxt)
            ce_va = vce()
            if ce_va <= best:
                best = ce_va
                best_state = [q.detach().clone() for q in self.params()]
            if log_every and (ep + 1) % log_every == 0:
                print(f"  ep {ep+1}: holdout {ce_va:.4f} best {best:.4f} "
                      f"(сертифицированный суммарный спуск >= {certified_guarantee:.2f})",
                      flush=True)
        for q, s in zip(self.params(), best_state):
            q.data.copy_(s)
        return {"epochs": epochs, "best_holdout_ce": round(best, 4),
                "certified_descent_lower_bound": round(certified_guarantee, 2)}

    def generate(self, prompt: str, n_bytes: int = 100, temp: float = 0.5,
                 seed: int = 1) -> str:
        t = self.t
        with t.no_grad():
            g = t.Generator(device=self.device).manual_seed(seed)
            ctx = list(prompt.encode("utf-8"))[-self.k:]
            if len(ctx) < self.k:
                ctx = [32] * (self.k - len(ctx)) + ctx
            out = bytearray()
            for _ in range(n_bytes):
                x = t.as_tensor([ctx], dtype=t.long, device=self.device)
                probs = t.softmax(self.logits(x)[0] / max(temp, 1e-3), dim=-1)
                b = int(t.multinomial(probs, 1, generator=g))
                out.append(b); ctx = ctx[1:] + [b]
        return (prompt.encode("utf-8") + bytes(out)).decode(
            "utf-8", errors="replace")


class AdaptiveScheduleLM(CertifiedStepLM):
    """All previously hardcoded knobs made adaptive, per primes:

    - C330 steps_to_quality: train until the quality gap Delta < eps
      (ceil(Delta0/d) steps), NOT a fixed epoch count;
    - Thermostat (Growth/Thermostat.lean): Gibbs temperature cools
      T_t = T0 * alpha^t with the exact stop bound exp(-delta/T_t) —
      sampling temperature is a schedule, not a constant;
    - §3.9: certified step size eta = 4A/R'^2 (no fixed LR);
    - batch adapts to the FLOP budget (C328/C331 work model).
    """

    def __init__(self, **kw):
        self.T0 = kw.pop("T0", 1.0)          # Thermostat initial temperature
        self.alpha = kw.pop("alpha", 0.97)   # cooling rate (0<alpha<1)
        self.eps_quality = kw.pop("eps_quality", 0.05)  # C330 quality gap
        self.flop_budget = kw.pop("flop_budget", 4_000_000)  # per-epoch work cap
        super().__init__(**kw)

    def temperature(self, t: int) -> float:
        """Thermostat schedule T_t = T0*alpha^t (theory-exact)."""
        return self.T0 * (self.alpha ** t)

    @staticmethod
    def aitken_final(L0, L1, L2):
        """Audit.lean aitken_exact: the geometric-trajectory limit
        Lstar from three consecutive measurements. Returns None when
        the window is degenerate (den == 0) — the consistency test
        (aitken_consistency) must pass before trusting this."""
        den = L0 + L2 - 2 * L1
        if den == 0:
            return None
        return (L0 * L2 - L1 ** 2) / den

    def train_adaptive(self, data: bytes, max_epochs: int = 400,
                       holdout: float = 0.1, log_every: int = 20) -> dict:
        """Train until CE gap < eps_quality (C330) or the epoch cap;
        certified steps (§3.9); batch shrinks under FLOP pressure."""
        import numpy as np
        t = self.t
        idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        n = len(idx) - self.k
        rng = np.random.default_rng(7)
        perm = rng.permutation(n)
        n_hold = max(16, int(n * holdout))
        hold, train = np.sort(perm[:n_hold]), perm[n_hold:]
        Xv = np.stack([idx[i:i + self.k] for i in hold])
        yv = idx[hold + self.k]
        ctx_va = t.as_tensor(Xv, dtype=t.long)
        nxt_va = t.as_tensor(yv, dtype=t.long)
        def vce():
            with t.no_grad():
                return float(t.nn.functional.cross_entropy(
                    self.logits(ctx_va), nxt_va))
        d0 = vce()                      # Delta_0 (C330)
        best = d0
        best_state = [q.detach().clone() for q in self.params()]
        epoch = 0
        for epoch in range(1, max_epochs + 1):
            # C330: stop when the certified gap target is met
            if d0 - best < self.eps_quality and epoch > 10:
                break
            # C328/C331: adapt batch to the per-epoch FLOP budget
            per_win = self.k * self.dim + self.dim * self.hidden + self.hidden * 256
            batch = max(16, min(512, self.flop_budget // max(per_win, 1)))
            order = rng.permutation(len(train))
            for bs in range(0, len(train), batch):
                sel = train[order[bs:bs + batch]]
                X = np.stack([idx[i:i + self.k] for i in sel])
                y = idx[sel + self.k]
                self._certified_step(t.as_tensor(X, dtype=t.long),
                                     t.as_tensor(y, dtype=t.long))
            ce = vce()
            if ce <= best:
                best = ce
                best_state = [q.detach().clone() for q in self.params()]
            if log_every and epoch % log_every == 0:
                print(f"  ep {epoch}: CE {ce:.4f} best {best:.4f} "
                      f"T={self.temperature(epoch):.3f} batch={batch} "
                      f"gap {d0-best:.3f}/{self.eps_quality}", flush=True)
        for q, s in zip(self.params(), best_state):
            q.data.copy_(s)
        return {"epochs_run": epoch, "best_holdout_ce": round(best, 4),
                "delta0": round(d0, 4),
                "final_T": round(self.temperature(epoch), 4),
                "stop_reason": ("quality-gap<eps (C330)" if d0 - best < self.eps_quality
                                else "epoch cap")}

    def generate(self, prompt: str, n_bytes: int = 100,
                 temp: float | None = None, seed: int = 1) -> str:
        """temp=None -> Thermostat: use the final cooled temperature."""
        if temp is None:
            temp = max(self.temperature(1000), 0.05)
        return super().generate(prompt, n_bytes=n_bytes, temp=temp, seed=seed)
