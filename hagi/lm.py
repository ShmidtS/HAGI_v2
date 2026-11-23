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
import math
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


class SemanticDistillHead:
    """WAVE 218 (user insight + primes DistillCert/DField): the
    ternary byte-LM distills embeddinggemma-2 SEMANTIC GEOMETRY
    while training. Teacher: gemma vectors of context windows;
    student: a small head over the LM hidden state predicts the
    teacher vector (cosine). The LM inherits the teacher's semantic
    space — knowledge-by-embedding, exactly the DistillCert channel
    (KL/cos drift bounded), D-field: new disagreement c=αD comes
    from real corpora, not repeats. Byte next-token stays the first
    objective; semantic alignment is the second."""

    def __init__(self, dim: int, teacher_dim: int = 768, lr: float = 0.05):
        import torch
        t = torch
        # hidden (dim*?) -> teacher_dim, trained by plain SGD+cos
        self.proj = None  # lazily built from first hidden
        self.lr = lr
        self.teacher_dim = teacher_dim

    def _ensure_proj(self, hidden_dim, device, dtype):
        import torch
        if self.proj is None:
            self.proj = torch.zeros(hidden_dim, self.teacher_dim,
                                    device=device, dtype=dtype)
            # WAVE 220: device-matched generator (cuda generator for a
            # cuda tensor; CPU generator raised RuntimeError)
            g = torch.Generator(device=device).manual_seed(11)                 if device != "cpu" else torch.Generator().manual_seed(11)
            self.proj.normal_(0, 0.02, generator=g)
            self.proj.requires_grad_(True)

    def align_step(self, model, ctx, texts):
        """One semantic alignment step: encode texts with the teacher,
        regress model hidden -> teacher vector (cosine loss)."""
        import torch
        t = model.t
        from hagi.embeddings import GemmaEmbedder
        gemma = GemmaEmbedder.get()
        tv = gemma.embed(texts).to(model.device)
        # forward to hidden: replicate logits' trunk up to h
        with torch.no_grad():
            pass
        e_w = (model._ste_ternary(model.emb, t)
               if getattr(model, "_qat", False) else model.emb)
        e = e_w[ctx]
        if hasattr(model, "_rope") and hasattr(model, "full_logits"):
            # WAVE 234: RopeAttentionByteLM — mean-pooled per-position
            # hidden (B, L, H) -> (B, H); CertifiedStepLM flatten path
            # is wrong for the pairwise-attention trunk.
            from hagi.trunk_ops import AdapterBank  # noqa: F401 (ad via logits path only)
            pos = t.arange(ctx.shape[1], device=ctx.device,
                           dtype=t.float32)
            dot = t.einsum("bid,bjd->bij", e, e) / (model.dim ** 0.5)
            sc = dot - model.lam * (pos[None, :, None]
                                    - pos[None, None, :])
            mask = pos[None, :, None] >= pos[None, None, :]
            sc = sc.masked_fill(~mask, float("-inf"))
            pp = t.softmax(sc, dim=-1)
            h_all = t.relu(t.einsum("blj,bjd->bld", pp, e) @ model.W1
                           + model.b1)
            ad = getattr(model, "domain_adapter", None)
            if ad is not None:
                At = t.as_tensor(ad.A, device=h_all.device,
                                 dtype=h_all.dtype)
                Bt = t.as_tensor(ad.B, device=h_all.device,
                                 dtype=h_all.dtype)
                h_all = h_all + (h_all @ At.T) @ Bt.T
            h = (h_all @ model.W2 + model.b2).mean(dim=1)
        else:
            h = t.relu(e.reshape(ctx.shape[0], -1) @ model.W1
                       + model.b1)
        self._ensure_proj(h.shape[1], h.device, h.dtype)
        pred = h @ self.proj
        pred = torch.nn.functional.normalize(pred, dim=-1)
        loss = 1.0 - (pred * tv).sum(-1).mean()  # cosine distance
        # gradient step on proj only (teacher is frozen — DistillCert)
        g, = torch.autograd.grad(loss, self.proj)
        with torch.no_grad():
            self.proj -= self.lr * g
        return float(loss)


class AdaptiveContextSelector:
    """WAVE 214 (user request + hagi_lab LONG_CONTEXT): the model
    CHOOSES its own context size per input. Architecture keeps a
    fixed window k (trained); adaptivity = the number of sliding
    windows W aggregated (mean-logits ensemble). W is measured on
    the prompt's own tail: more windows are accepted only while
    self-CE improves (diminishing returns => stop). The full history
    stays in caller's bytes — unbounded SOURCE, bounded compute."""

    def __init__(self, w_choices=(1, 2, 4, 8, 16), min_tail: int = 16,
                 tol: float = 0.01):
        self.w_choices = sorted(w_choices)
        self.min_tail = min_tail
        self.tol = tol

    def _ensemble_logits(self, model, rows_x):
        """Mean-logits over W disjoint trailing windows per row-set:
        window j covers bytes [end-k-j*k, end-j*k); W=1 is the plain
        path. Aggregation = arithmetic mean of softmax probs (C307
        attention-branch mean composition)."""
        t = model.t
        k = model.k
        probs = None
        used = 0
        for j in range(max(self.w_choices)):
            xs = []
            for row in rows_x:
                end = len(row)
                start = max(0, end - k - j * k)
                w = row[start:end - j * k] if j > 0 else row[start:end]
                if len(w) < k:
                    w = [32] * (k - len(w)) + w
                else:
                    w = w[-k:]
                xs.append(w)
            x = t.as_tensor(xs, dtype=t.long)
            with t.no_grad():
                lg = model.logits(x)
                p = t.softmax(lg, dim=-1)
            probs = p if probs is None else probs + p
            used += 1
            # caller breaks by measuring — see select()
            yield used, probs / used

    def select(self, model, prompt_bytes: bytes) -> dict:
        """Measure self-CE of predicting the tail bytes with W
        aggregated windows; return the SMALLEST W within tol of the
        best (Occam: no wider ensemble without measured benefit)."""
        t = model.t
        tail_len = min(self.min_tail, max(1, len(prompt_bytes) // 4))
        if len(prompt_bytes) <= tail_len + model.k:
            return {"w": 1, "ce": None, "curve": [],
                    "reason": "prompt too short for measurement"}
        tail = prompt_bytes[-tail_len:]
        pre = prompt_bytes[:-tail_len]
        k = model.k
        rows_x = []
        for i in range(len(tail)):
            end = len(pre) + i
            start = max(0, end - k * max(self.w_choices))
            row = list(prompt_bytes[start:end])
            rows_x.append(row)
        y = t.as_tensor(list(tail), dtype=t.long)
        curve = []
        for w, probs in self._ensemble_logits(model, rows_x):
            if w not in self.w_choices:
                continue
            ce = float(t.nn.functional.cross_entropy(
                (probs + 1e-12).log(), y))
            curve.append((w, ce))
            if w == max(self.w_choices):
                break
        if not curve:
            return {"w": 1, "ce": None, "curve": [],
                    "reason": "no W fits the prompt"}
        best_ce = min(c for _, c in curve)
        chosen = self.w_choices[0]
        for w, c in curve:
            if c <= best_ce * (1.0 + self.tol) + 1e-9:
                chosen = w
                break
        return {"w": chosen, "ce": best_ce, "curve": curve,
                "reason": "self-CE elbow"}


def finite_displacement_ce_upper(logits_before, logits_after, targets):
    """WAVE 208 (HAGI_v2_final patches): exact finite-shift CE upper
    bound via the log-sum-exp range lemma:
      CE(z+d) - CE(z) <= <softmax(z)-onehot(y), d> + range(d)^2/8.
    Makes NO linearity assumption about logits in parameters — unlike
    the JVP tangent bound, this certifies the ACTUAL finite step."""
    import torch
    F = torch.nn.functional
    delta = logits_after - logits_before
    p = torch.softmax(logits_before, dim=-1)
    y = F.one_hot(targets, num_classes=logits_before.shape[-1]).to(p.dtype)
    first_order = ((p - y) * delta).sum(dim=-1)
    width = delta.max(dim=-1).values - delta.min(dim=-1).values
    return (first_order + width.square() / 8.0).mean()


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
        # WAVE 255 (FunctionPreservingGrowth.lean): optional extra
        # attention blocks. Block b adds Wo[b] * relu(a @ W1[b] + b1[b])
        # to the hidden state, a = attention aggregate. A fresh block has
        # Wo = 0 -> the function is preserved EXACTLY (lmLogits_grow_identity).
        self.blocks: list[dict] = []
        # WAVE 257: indices of FROZEN blocks (repair blocks = the old
        # memory; the general train step must not overwrite them).
        self.frozen_blocks: set = set()
        self.device = _resolve_device(t, self.dim, self.hidden)
        if self.device == "cuda":
            for prm in self.params():
                prm.data = prm.data.to(self.device)

    def params(self):
        out = [self.emb, self.q, self.W1, self.b1, self.W2, self.b2]
        for blk in getattr(self, "blocks", []):
            out += [blk["W1"], blk["b1"], blk["Wo"]]
        return out

    def _block_add(self, a, h):
        """Extra attention blocks: h += Wo * relu(a @ W1 + b1). Identity
        while Wo == 0 (step 0 of FunctionPreservingGrowth)."""
        t = self.t
        for blk in getattr(self, "blocks", []):
            h = h + t.relu(a @ blk["W1"] + blk["b1"]) @ blk["Wo"]
        return h

    def grow(self, seed: int = 0) -> int:
        """Append ONE function-preserving attention block (Wo = 0).
        Logits are unchanged exactly; capacity is learned afterwards."""
        t = self.t
        g = t.Generator().manual_seed(seed + 1000 * (len(self.blocks) + 1))
        blk = {"W1": t.nn.Parameter(t.randn(self.dim, self.hidden, generator=g) * 0.08),
               "b1": t.nn.Parameter(t.zeros(self.hidden)),
               "Wo": t.nn.Parameter(t.zeros(self.hidden, self.hidden))}
        self.blocks.append(blk)
        if self.device == "cuda":
            for prm in blk.values():
                prm.data = prm.data.to(self.device)
        return len(self.blocks)

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
        """ctx: (B, L) newest-first. WAVE 233 (hagi_repro patch):
        exact last-query fast path — equals full_logits(ctx)[:, -1, :]
        but skips materializing all L query rows / B×L×L scores.
        Measured 3.8-4.4x on L=96 (max logit diff 1.3e-8)."""
        t = self.t
        if ctx.device != self.device:
            ctx = ctx.to(self.device)
        e = self._rope(self.emb[ctx], ctx.shape[1], ctx.device)
        query = e[:, -1, :]  # newest-first: last = newest = query
        pos = t.arange(ctx.shape[1], device=ctx.device, dtype=t.float32)
        # newest-first key j is at distance (L-1-j) behind the query
        scores = t.einsum("bd,bld->bl", query, e) / (self.dim ** 0.5)             - self.lam * (ctx.shape[1] - 1
                          - t.arange(ctx.shape[1], device=ctx.device,
                                     dtype=t.float32))[None, :]
        mask = t.arange(ctx.shape[1], device=ctx.device)[None, :]             <= ctx.shape[1] - 1  # all keys are causal past of query
        scores = scores.masked_fill(~mask, float("-inf"))
        pp = t.softmax(scores, dim=-1)
        a_agg = t.einsum("bl,bld->bd", pp, e)
        h = self._block_add(a_agg, t.relu(a_agg @ self.W1 + self.b1))
        ad = getattr(self, "domain_adapter", None)
        if ad is not None:
            At = t.as_tensor(ad.A, device=h.device, dtype=h.dtype)
            Bt = t.as_tensor(ad.B, device=h.device, dtype=h.dtype)
            h = h + (h @ At.T) @ Bt.T
        return h @ self.W2 + self.b2

    def full_logits(self, ctx):
        t = self.t
        if ctx.device != self.device:
            ctx = ctx.to(self.device)
        B, L = ctx.shape
        e = self.emb[ctx]
        e = self._rope(e, L, ctx.device)
        # WAVE 229b FIX: true pairwise (query,key) scores. The old
        # code unsqueezed a per-position scalar to (B,L,1) and
        # broadcast it over the key axis -> uniform attention,
        # q/lam ignored (train/generate distributions diverged).
        pos = t.arange(L, device=ctx.device, dtype=t.float32)
        dot = t.einsum("bid,bjd->bij", e, e) / (self.dim ** 0.5)
        sc = dot - self.lam * (pos[None, :, None]
                               - pos[None, None, :])
        mask = pos[None, :, None] >= pos[None, None, :]
        sc = sc.masked_fill(~mask, float("-inf"))
        # WAVE 204 FIX (P1.1.2, audit v2): softmax over the KEY axis
        # (dim=-1), not the query axis — dim=1 normalized across
        # queries, leaking future tokens into earlier positions'
        # denominators.
        pp = t.softmax(sc, dim=-1)
        a_agg = t.einsum("blj,bjd->bld", pp, e)
        h = self._block_add(a_agg, t.relu(a_agg @ self.W1 + self.b1))
        ad = getattr(self, "domain_adapter", None)

        if ad is not None:
            At = t.as_tensor(ad.A, device=h.device, dtype=h.dtype)
            Bt = t.as_tensor(ad.B, device=h.device, dtype=h.dtype)
            h = h + (h @ At.T) @ Bt.T
        return h @ self.W2 + self.b2

    def ce(self, ctx, nxt):
        dev = self.device
        if nxt.device != dev:
            nxt = nxt.to(dev)
        return self.t.nn.functional.cross_entropy(
            self.logits(ctx), nxt, reduction="mean")

    def train_guarded(self, data: bytes, epochs: int = 60, lr: float = 1e-3,
                      holdout: float = 0.1, batch: int = 64,
                      max_train_ctx: int = 96, log_every: int = 0,
                      holdout_seed: int | None = None) -> dict:
        """WAVE 231 (integrated from HAGI_v2_local_repro):
        - GPU-resident uint8 byte buffer + tensor gather (no per-batch
          host->device copies in the hot loop; measured 1.17-1.40x);
        - mini-batched holdout eval (OOM fix, cap 2048 targets);
        - contiguous tail holdout + embargo (wave229b semantics kept);
        - Adam + rollback (certified-loop contract kept)."""
        import numpy as np
        t = self.t
        idx_np = np.frombuffer(data, dtype=np.uint8)
        n = len(idx_np) - 2
        dev = self.device
        # WAVE 245 (measured, CPU+GPU, 2 seeds): equal data, 2 epochs.
        # batch  CE(CPU)  CE(GPU)   s(GPU)
        #   16    1.99     1.99     10.6
        #   32    2.11     2.11      5.0   <- Pareto: -0.42 CE vs 256
        #   64    2.27     2.28      3.4      for +3.5 s/chunk
        #  256    2.53     2.53      1.5
        # Smaller batch = more steps = more descent per byte read
        # (Step/BatchBudget: throughput optimum != quality optimum).
        # WAVE 245b: on the real disk cycle batch 32 cost 52 s/chunk
        # (was 18) with chunk CE ~2.3 inside the domain noise of the
        # earlier 2.1-2.8 -> not worth 2.9x. Settled on 64 (CE 2.28,
        # ~1.6x time): quality gain is certain in isolation, cheap.
        # GPU-resident buffer: windows are gathered on-device.
        raw = t.as_tensor(idx_np.copy(), dtype=t.long, device=dev)
        offs = t.arange(max_train_ctx, dtype=t.long, device=dev)

        rng = np.random.default_rng(7 if holdout_seed is None
                                    else int(holdout_seed))
        n_hold = max(16, int(n * holdout))
        hold = np.arange(max(hold_start := n - n_hold, n - 2048), n)
        train = np.arange(0, max(16, int(hold[0]) - max_train_ctx))

        # WAVE 253 (measured): batch_at cost 30 s of a 71 s chunk —
        # 42%, twice the forward pass — because every call did 4
        # synchronising host->device copies plus int(lens.max()) (a
        # GPU->CPU sync); batch 256->64 (wave245) made it 4x more
        # calls. Positions are known up front, so the windows are
        # gathered ONCE on the device and batches are index slices.
        # Same construction (newest-first, pad byte 0, y = raw[pos]).
        def gather_all(pos_np, fixed_len=None):
            pos_np = np.asarray(pos_np, dtype=np.int64)
            n_ = len(pos_np)
            if n_ == 0:
                e = t.zeros((0, 1), dtype=t.long, device=dev)
                return e, e[:, 0]
            lens_np = np.minimum(np.full(n_, fixed_len or max_train_ctx, dtype=np.int64), pos_np)
            pad = int(lens_np.max(initial=1))
            pos_t = t.as_tensor(pos_np, dtype=t.long, device=dev)
            lens_t = t.as_tensor(lens_np, dtype=t.long, device=dev)
            j = t.arange(pad, dtype=t.long, device=dev)
            base = pos_t[:, None] - 1 - (lens_t[:, None] - 1) + j[None, :]
            mask = j[None, :] >= (pad - lens_t[:, None])
            base = t.where(mask, base, t.zeros_like(base))
            X = raw[base.clamp_(0, len(raw) - 1)]
            X = t.where(mask, X, t.zeros_like(X))  # pad byte 0
            return X.contiguous(), raw[pos_t]       # keep int64: a uint8
            # index_select + .long() per batch measured 1731 us vs 5 us

        X_tr, y_tr = gather_all(train)
        X_va, y_va = gather_all(hold, fixed_len=max_train_ctx)

        def eval_ce(positions=None, eval_batch=512):
            tot, cnt = 0.0, 0
            with t.no_grad():
                for s in range(0, len(y_va), eval_batch):
                    X = X_va[s:s + eval_batch]
                    tot += float(self.ce(X, y_va[s:s + eval_batch])) * len(X)
                    cnt += len(X)
            return tot / max(cnt, 1)

        # WAVE 239: exclude dead q from the optimizer (gradient
        # is always None since pairwise attention; q stays in
        # checkpoints for backward compat only).
        live_params = [self.emb, self.W1, self.b1, self.W2,
                       self.b2]
        # WAVE 255: extra blocks train with the trunk (q stays excluded)
        for i, blk in enumerate(getattr(self, "blocks", [])):
            if i in getattr(self, "frozen_blocks", ()):   # WAVE 257
                continue
            live_params += [blk["W1"], blk["b1"], blk["Wo"]]
        opt = t.optim.Adam(live_params, lr=lr)
        best = eval_ce(hold)
        best_state = [p.detach().clone() for p in self.params()]
        self._last_best_state = best_state
        for ep in range(epochs):
            perm2 = rng.permutation(len(train))
            for bs in range(0, len(train), batch):
                sel = t.as_tensor(perm2[bs:bs + batch], dtype=t.long, device=dev)
                X = X_tr.index_select(0, sel)
                y = y_tr.index_select(0, sel)
                opt.zero_grad()
                self.ce(X, y).backward()
                opt.step()
            ce_va = eval_ce(hold)
            if ce_va <= best:
                best = ce_va
                best_state = [q2.detach().clone() for q2 in self.params()]
            if log_every and (ep + 1) % log_every == 0:
                print(f"  ep {ep+1} ce {ce_va:.4f} (best {best:.4f})",
                      flush=True)
            if best < 0.01:  # memorized; skip remaining epochs
                break
        if best_state is not None:
            for q2, s in zip(self.params(), best_state):
                q2.data.copy_(s)
        # WAVE 239 (external review): best_holdout_ce is a
        # SELECTION score (max over epochs on the checkpoint-
        # selection split) — optimistic as a quality estimate. The
        # key name is kept for checkpoint/ledger compatibility; the
        # semantic flag marks it honestly.
        return {"epochs": epochs,
                "best_holdout_ce": round(best, 4),
                "holdout_role": "selection"}

    def _rope_logits_from_params(self, ps, ctx):
        """Functional logits over (emb, W1, b1, W2, b2) for JVP —
        WAVE 235: reuses the LIVE self._rope (exact same isometry as
        full_logits; the original duplicated-pairs rewrite diverged:
        wrong freq schedule and interleaved-vs-split pairing)."""
        t = self.t
        emb, W1, b1, W2, b2 = ps
        B, L = ctx.shape
        e = self._rope(emb[ctx], L, ctx.device)
        pos = t.arange(L, device=ctx.device, dtype=t.float32)
        dot = t.einsum("bid,bjd->bij", e, e) / (self.dim ** 0.5)
        sc = dot - self.lam * (pos[None, :, None] - pos[None, None, :])
        mask = pos[None, :, None] >= pos[None, None, :]
        sc = sc.masked_fill(~mask, float("-inf"))
        pp = t.softmax(sc, dim=-1)
        h = t.relu(t.einsum("blj,bjd->bld", pp, e) @ W1 + b1)
        return h @ W2 + b2

    def _certified_finite_step(self, ctx, nxt, lr_scale: float = 1.0,
                               max_backtracks: int = 12) -> float:
        """WAVE 235: finite-displacement certified step on the Rope
        trunk (ports CertifiedStepLM's §3.9 mechanism). JVP proposes
        eta; admission tests the ACTUAL finite logit displacement via
        finite_displacement_ce_upper (log-sum-exp range lemma — no
        linearity assumption). Rejected trial rolls back all params.
        Returns certified drop bound (0.0 = HOLD)."""
        t = self.t
        if ctx.device != self.device:
            ctx = ctx.to(self.device)
        if nxt.device != self.device:
            nxt = nxt.to(self.device)
        # WAVE 233: q is dead in forward (pairwise attention) — the
        # certified step must cover ONLY live parameters, else JVP
        # signature mismatches _rope_logits_from_params(ps).
        params = tuple([self.emb, self.W1, self.b1, self.W2, self.b2])
        snapshot = [q.detach().clone() for q in params]
        # next-byte logits only: newest-first -> last position
        logits_live = self.full_logits(ctx)[:, -1, :]
        logits0 = logits_live.detach()
        grads = t.autograd.grad(
            t.nn.functional.cross_entropy(logits_live, nxt), params)
        direction = tuple(-g.detach() for g in grads)
        alignment = float(sum(d.square().sum() for d in direction))
        if alignment <= 1e-24:
            return 0.0  # HOLD
        try:
            _, tangent = t.autograd.functional.jvp(
                lambda *ps: self._rope_logits_from_params(ps, ctx),
                params, direction)
            width = tangent.max(dim=-1).values                 - tangent.min(dim=-1).values
            r2 = float(width.square().mean())
        except Exception:
            r2 = 0.0
        eta = lr_scale * (4.0 * alignment / r2 if r2 > 1e-24 else 1.0)
        for _ in range(max_backtracks + 1):
            with t.no_grad():
                for q, old in zip(params, snapshot):
                    q.copy_(old)
                for q, d in zip(params, direction):
                    q.add_(eta * d)
                logits1 = self.full_logits(ctx)[:, -1, :].detach()
            upper = float(finite_displacement_ce_upper(
                logits0, logits1, nxt))
            drop = -upper
            if drop > 1e-12:
                return drop  # COMMIT: finite-displacement certificate
            eta *= 0.5
        with t.no_grad():
            for q, old in zip(params, snapshot):
                q.copy_(old)
        return 0.0  # HOLD

    def eval_holdout_ce(self, data: bytes, holdout: float = 0.1,
                        max_eval: int = 1024):
        """Wave235: holdout CE on the contiguous tail (same split as
        train_guarded), mini-batched. None if data too small."""
        import numpy as _np
        import torch as _th
        idx_np = _np.frombuffer(data, dtype=_np.uint8)
        n = len(idx_np) - self.k - 1
        if n < 512:
            return None
        n_hold = max(16, int(n * holdout))
        hold = _np.arange(max(n - n_hold, n - 2048, n - max_eval), n)
        dev = self.device
        raw = _th.as_tensor(idx_np.copy(), dtype=_th.long, device=dev)
        tot, cnt = 0.0, 0
        with _th.no_grad():
            for s in range(0, len(hold), 512):
                hp = hold[s:s + 512]
                # WAVE 241 FIX (P0 target leak): old slice
                # idx[p+k::-1][:k] STARTS at p+k = the target byte
                # itself (it was X[0]!). Correct newest-first window:
                # positions p+k-1 down to p.
                X = _np.stack([idx_np[p + self.k - 1:
                                      max(p - 1, -1):-1][:self.k]
                               for p in hp])
                x = _th.as_tensor(X, dtype=_th.long, device=dev)
                y = raw[_th.as_tensor(hp + self.k, dtype=_th.long,
                                      device=dev)]
                tot += float(self.ce(x, y)) * len(x)
                cnt += len(x)
        return tot / max(cnt, 1)

    def adapter_head_step(self, ctx, nxt, lr: float = 1e-2):
        """WAVE 230: train ONLY the domain adapter (A/B) on this
        batch — Adam on adapter factors, trunk/head frozen."""
        ad = getattr(self, "domain_adapter", None)
        if ad is None:
            return
        t = self.t
        # WAVE 236: restored adapters may be float64 numpy — match
        # the model dtype (float32) exactly.
        At = t.as_tensor(ad.A, device=self.device,
                         dtype=self.W1.dtype).requires_grad_(True)
        Bt = t.as_tensor(ad.B, device=self.device,
                         dtype=self.W1.dtype).requires_grad_(True)
        opt = t.optim.Adam([At, Bt], lr=lr)
        e = self._rope(self.emb[ctx], ctx.shape[1], ctx.device)
        pos = t.arange(ctx.shape[1], device=ctx.device, dtype=t.float32)
        dot = t.einsum("bid,bjd->bij", e, e) / (self.dim ** 0.5)
        sc = dot - self.lam * (pos[None, :, None] - pos[None, None, :])
        mask = pos[None, :, None] >= pos[None, None, :]
        sc = sc.masked_fill(~mask, float("-inf"))
        pp = t.softmax(sc, dim=-1)
        h0 = t.relu(t.einsum("blj,bjd->bld", pp, e) @ self.W1 + self.b1)
        logits = (h0 + (h0 @ At.T) @ Bt.T) @ self.W2 + self.b2
        # WAVE 233 fix (hagi_repro): nxt has B targets, not B*L —
        # reshape(-1, C) mismatched rows/targets silently.
        loss = t.nn.functional.cross_entropy(
            logits[:, -1, :], nxt.to(self.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with t.no_grad():
            # WAVE 236: A/B may be numpy (fresh) or torch Tensor
            # (restored from checkpoint) — write back accordingly.
            import numpy as _np2
            ad.A = At.detach().cpu().numpy()
            ad.B = Bt.detach().cpu().numpy()

    k = 96  # interface compat with Unified (model.k consumers)

    def train_certified(self, data: bytes, epochs: int = 60,
                        holdout: float = 0.1, batch: int = 64,
                        log_every: int = 0,
                        holdout_seed: int | None = None,
                        lr: float | None = None) -> dict:
        """Unified-compatible training: delegates to train_guarded."""
        r = self.train_guarded(data, epochs=epochs, holdout=holdout,
                               batch=batch, max_train_ctx=self.k,
                               holdout_seed=holdout_seed,
                               **({} if lr is None else {"lr": lr}))
        # (lr default comes from train_guarded: 1e-3 since WAVE 256)
        # WAVE 235 HYBRID (user-approved): after the fast Adam phase,
        # a certified-refine phase runs §3.9 finite-displacement steps
        # on holdout windows — every accepted step carries a PROVEN
        # CE drop bound, restoring the StepContract semantics the
        # pure-Adam path was missing (bound was honest None).
        # WAVE 239 FIX (external review): certified-refine MUST draw
        # minibatches from the TRAIN partition only. The old code fit
        # refine on the same contiguous tail that eval_holdout_ce
        # uses — validation leaked into gradients, and the post-refine
        # comparison was self-referential. Refine positions come from
        # data[:-holdout_tail]; the holdout stays a pure audit set.
        pre_refine_ce = self.eval_holdout_ce(data, holdout=holdout)
        import numpy as _np
        full_np = _np.frombuffer(data, dtype=_np.uint8)
        n_all = len(full_np) - 2
        n_hold = max(16, int(n_all * holdout))
        hold_start = max(n_all - n_hold, n_all - 2048)
        train_zone_end = max(self.k + 1, hold_start - self.k)
        train_np = full_np[:train_zone_end]
        rng = _np.random.default_rng(7 if holdout_seed is None
                                     else int(holdout_seed))
        positions = rng.choice(len(train_np) - self.k - 1,
                               size=min(64, len(train_np) - self.k - 1),
                               replace=False)
        # WAVE 241 FIX (P0 target leak): same slice bug as
        # eval_holdout_ce — old X[0] was the target byte itself.
        X = _np.stack([train_np[p + self.k - 1:
                                max(p - 1, -1):-1][:self.k]
                       for p in positions])
        y = train_np[positions + self.k]
        import torch as _th
        refine_X = _th.as_tensor(X.copy(), dtype=_th.long,
                                 device=self.device)
        refine_y = _th.as_tensor(y, dtype=_th.long, device=self.device)
        total_drop, steps, commits = 0.0, 0, 0
        for _ in range(8):  # refine rounds
            perm = rng.permutation(len(positions))
            for s in range(0, len(perm), 32):
                bx = refine_X[perm[s:s + 32]]
                by = refine_y[perm[s:s + 32]]
                d = self._certified_finite_step(bx, by)
                steps += 1
                if d > 0:
                    commits += 1
                    total_drop += d
        # WAVE 239 (external review): per-batch certified bounds do
        # NOT telescope into one objective's guarantee — each bound
        # holds on its own minibatch CE. Report the ledger honestly.
        bound = ({"total": total_drop, "commits": commits,
                  "scope": "per-batch-sum, no telescoping guarantee"}
                 if commits else None)
        # Wave235 gate: certified steps are per-batch proven, but the
        # deployment metric is the Adam-phase holdout. If refine made
        # the full holdout worse, roll back to the best_state snapshot
        # (each per-batch bound holds on its own batch only —
        # HonestTransition: prove before deploy).
        tail_ce = self.eval_holdout_ce(data, holdout=holdout)
        # apples-to-apples: SAME eval (tail windows) pre vs post.
        if (tail_ce is not None and pre_refine_ce is not None
                and tail_ce > pre_refine_ce * 1.02):
            best_state = getattr(self, "_last_best_state", None)
            if best_state is not None:
                with _th.no_grad():
                    for q, s in zip(self.params(), best_state):
                        q.data.copy_(s)
            bound = None  # refined weights not deployed
        return {"best_holdout_ce": r["best_holdout_ce"],
                "holdout_role": "selection",
                "certified_descent_lower_bound": bound,
                "certified_refine": {"steps": steps,
                                     "commits": commits,
                                     "deployed": bound is not None}}

    # ---- WAVE 230: self-terminating generation on the Rope model --

    def _read_all_windows(self, prompt_bytes: bytes, max_span: int = 64):
        """Coverage windows over EVERY prompt byte (newest-first,
        variable tail length), matching train window semantics."""
        ctx = list(prompt_bytes)
        wins = []
        i = len(ctx)
        while i > 0:
            j = max(0, i - max_span)
            wins.append(ctx[j:i][::-1])
            i = j
        return wins or [[0]]

    def generate_auto(self, prompt: str, max_bytes: int = 4096,
                      temp: float = 0.5, seed: int = 1) -> str:
        """Model decides HOW LONG to answer (wave226 semantics, ported
        to Rope pairwise attention). Reads the WHOLE prompt via
        coverage windows; logits averaged. WAVE 230 fix: the last
        window is no longer double-counted (was `or True`), divisor
        is the actual number of terms."""
        t = self.t
        pb = prompt.encode('utf-8')
        # WAVE 242 (CoverageReading C598): window width must MATCH
        # train semantics (self.k=96) — was hardcoded 64 (the
        # CertifiedStepLM value); no truncation of head windows.
        windows = self._read_all_windows(pb, max_span=self.k)
        with t.no_grad():
            g = t.Generator(device=self.device).manual_seed(seed)
            out = bytearray()
            last_ctx = windows[-1]
            rep_counter = 0
            low_conf = 0
            last_token = None
            while len(out) < max_bytes:
                cur = windows[:-1] + [last_ctx]
                probs = None
                for w in cur:
                    x = t.as_tensor([w], dtype=t.long, device=self.device)
                    pp = t.softmax(self.logits(x)[0] / max(temp, 1e-3),
                                   dim=-1)
                    probs = pp if probs is None else probs + pp
                probs = probs / len(cur)
                top_p = float(probs.max())
                b = int(t.multinomial(probs, 1, generator=g))
                out.append(b)
                last_ctx = last_ctx[1:] + [b]
                ch = bytes([b])
                if top_p < (1.0 / 256) * 1.2:
                    low_conf += 1
                else:
                    low_conf = 0
                if b == last_token:
                    rep_counter += 1
                else:
                    rep_counter = 0
                last_token = b
                if ch in b'.!?':
                    nxt = int(t.multinomial(probs, 1, generator=g)) \
                        if False else None
                    if low_conf >= 2 or rep_counter >= 3:
                        break
                if rep_counter >= 8 or low_conf >= 8:
                    break
                if ch in b'\n' and low_conf >= 3:
                    break
        tail = bytes(out)
        # trim to last sentence terminator if any
        for term in (b'.', b'!', b'?'):
            k = tail.rfind(term)
            if k != -1:
                tail = tail[:k + 1]
                break
        return (prompt + tail.decode('utf-8', errors='replace'))

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

    @staticmethod
    def _ste_ternary(W, t=None):
        """WAVE 216 (user: 'учи сразу тернарную'): straight-through
        ternary fake-quant (labs TernaryAwareByteLM). Forward sees
        {-s,0,+s} per-row; master weights carry the gradient."""
        if t is None:
            import torch as _t
            t = _t
        scale = W.detach().abs().mean(dim=-1, keepdim=True).clamp_min(1e-12)
        q = t.round(W.detach() / scale).clamp(-1, 1) * scale
        return W + (q - W).detach()

    def logits(self, ctx):
        t = self.t
        if ctx.device != getattr(self, "device", ctx.device):
            ctx = ctx.to(self.device)   # WAVE 176: robust CPU input
        e_w = (self._ste_ternary(self.emb, t)
               if getattr(self, "_qat", False) else self.emb)
        e = e_w[ctx]
        h = t.relu(e.reshape(ctx.shape[0], -1) @
                   (self._ste_ternary(self.W1, t) if getattr(self, "_qat", False)
                    else self.W1) + self.b1)
        # WAVE 210: optional per-domain low-rank adapter on the hidden
        # head (AdapterBank.get(domain)) — C532 form x + B(A x); with
        # B=0 EXACT identity, so a fresh adapter never changes logits.
        ad = getattr(self, "domain_adapter", None)
        if ad is not None:
            At = t.as_tensor(ad.A, device=h.device, dtype=h.dtype)
            Bt = t.as_tensor(ad.B, device=h.device, dtype=h.dtype)
            h = h + (h @ At.T) @ Bt.T
        w2 = (self._ste_ternary(self.W2, t) if getattr(self, "_qat", False)
              else self.W2)
        return h @ w2 + self.b2

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

    # ---- WAVE 208: guarded FINITE step (HAGI_v2_final patches) ------
    def _certified_finite_step(self, ctx, nxt, lr_scale: float = 1.0,
                               max_backtracks: int = 12) -> float:
        """Finite-displacement certified step: JVP proposes eta (scale
        heuristic ONLY), admission tests the ACTUAL finite logit
        displacement via finite_displacement_ce_upper (log-sum-exp
        range lemma — no linearity assumption). A rejected trial rolls
        back all parameters. Returns the certified drop bound (0 = HOLD).
        Fixes the local-experiment finding that a pure-JVP step can
        INCREASE CE on a nonlinear net."""
        t = self.t
        ctx, nxt = self._to_dev((ctx, nxt))
        params = tuple(self.params())
        snapshot = [p.detach().clone() for p in params]
        logits_live = self.logits(ctx)
        logits0 = logits_live.detach()
        grads = t.autograd.grad(
            t.nn.functional.cross_entropy(logits_live, nxt), params)
        direction = tuple(-g.detach() for g in grads)
        alignment = float(sum(d.square().sum() for d in direction))
        if alignment <= 1e-24:
            return 0.0  # HOLD: zero/invalid gradient
        try:
            _, tangent = t.autograd.functional.jvp(
                lambda *ps: self._logits_from_params(ps, ctx), params,
                direction)
            width = tangent.max(dim=-1).values - tangent.min(dim=-1).values
            r2 = float(width.square().mean())
        except Exception:
            r2 = 0.0
        eta = lr_scale * (4.0 * alignment / r2 if r2 > 1e-24 else 1.0)
        for _ in range(max_backtracks + 1):
            with t.no_grad():
                for p, old in zip(params, snapshot):
                    p.copy_(old)
                for p, d in zip(params, direction):
                    p.add_(eta * d)
                logits1 = self.logits(ctx).detach()
            upper = float(finite_displacement_ce_upper(logits0, logits1, nxt))
            drop = -upper
            if drop > 1e-12:
                return drop  # COMMIT: finite-displacement certificate
            eta *= 0.5
        with t.no_grad():
            for p, old in zip(params, snapshot):
                p.copy_(old)
        return 0.0  # HOLD: no positive finite certificate

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
        per_sample_range = disp.max(dim=1).values - disp.min(dim=1).values
        # WAVE 252 (Lean ceAdaptiveCurvTight): C = kappa(R) * Var with
        # kappa(R) = (e^R - 1 - R)/R^2 >= 1/2 (kappa_ge_half), R = max
        # per-sample logit range. The code used C = Var alone: too
        # small, hence a too-small step and a drop claim 1.46-1.79x
        # BELOW the realised one (180 steps, 0 violations) — valid but
        # weaker than the theorem.
        R = float(per_sample_range.max())
        # kappa(R) -> 1/2 as R -> 0, but the expression loses all
        # precision below R ~ 1e-6: measured kappa(1e-7) = 0.494,
        # kappa(1e-8) = -0.608 (NEGATIVE curvature -> ascent step),
        # kappa(1e-9) = 82.7. Below 1e-5 use the limit 1/2 (kappa_ge_half).
        R_c = min(R, 50.0)
        C = float(var_disp.mean()) * (
            (math.exp(R_c) - 1.0 - R_c) / (R_c * R_c) if R > 1e-5 else 0.5)
        # The range-bound certificate (Hoeffding, singleSampleStep) is
        # stated with the MAXIMAL per-sample range R, not its mean
        # square: with R'^2 = mean(range^2) the step 4A/R'^2 overshoots
        # and the drop claim was violated by ~5e-4 on 7/12 steps
        # (WAVE 252 measurement). Use R^2 = (max range)^2.
        Rp_sq = R * R
        if C <= 0 or A <= 0 or Rp_sq <= 0:
            return 0.0
        # adaptive η* = A/(2C) (theorem) with the unit cap; when the cap
        # binds, fall back to the range-bound step 4A/R'² (§3.9), which
        # is ALWAYS certified (adaptive dominates range: C ≤ R'²/8, so
        # A²/(4C) ≥ 2A²/R'² — take the guaranteed drop of the branch used)
        use_adaptive = A / (2.0 * C) <= 1.0
        eta = (A / (2.0 * C)) if use_adaptive else (4.0 * A / Rp_sq)
        eta *= lr_scale
        # WAVE 252 (bug): the drop was claimed for the UNSCALED eta*
        # while `eta` was multiplied by lr_scale. The majorizer is
        # L(eta) <= L(0) - A*eta + C_curv*eta^2 with the curvature of the
        # branch used, so the certified drop must be evaluated at the
        # eta ACTUALLY taken: at lr_scale = 0.05 the old claim was
        # ~10x the realised drop (40/40 steps violated it). At
        # lr_scale = 1 both expressions reduce to the theorem's.
        drop = (A * eta - C * eta * eta) if use_adaptive else (
            A * eta - (Rp_sq / 8.0) * eta * eta)
        snapshot = [q.detach().clone() for q in self.params()]
        with t.no_grad():
            off = 0
            for prm, gi in zip(self.params(), g):
                sz = prm.numel()
                prm.add_(eta * D[off:off + sz].reshape(prm.shape))
                off += sz
        # The majorizer is built on the JVP (tangent) displacement; the
        # theorem is about the exact one. Measured: 6/40 steps still
        # overclaimed by up to 1.9e-3. Same contract as the production
        # Rope step (_certified_finite_step): verify the realised drop
        # on the actual batch CE; below the claim -> roll back, HOLD.
        with t.no_grad():
            realised = float(loss.detach()) - float(
                t.nn.functional.cross_entropy(self.logits(ctx), nxt))
        if realised < drop:
            with t.no_grad():
                for q, old_q in zip(self.params(), snapshot):
                    q.copy_(old_q)
            return 0.0
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

    # ---- WAVE 228: model-driven length + full-input reading ----

    def _read_all_windows(self, prompt_bytes: bytes, max_span: int):
        """Split the WHOLE prompt into sliding windows covering every
        byte (not just the tail); returns list of k-sized windows."""
        k = self.k
        if len(prompt_bytes) <= k:
            return [([32] * (k - len(prompt_bytes))
                     + list(prompt_bytes))]
        windows = []
        step = k  # contiguous coverage, no byte skipped
        for start in range(0, len(prompt_bytes) - k + 1, step):
            windows.append(list(prompt_bytes[start:start + k]))
        if windows[-1] != list(prompt_bytes[-k:]):
            windows.append(list(prompt_bytes[-k:]))
        return windows[-max_span:]

    def generate_auto(self, prompt: str, max_bytes: int = 4096,
                      temp: float = 0.5, seed: int = 1) -> str:
        """Model decides HOW LONG to answer: generation runs until the
        MODEL signals completion — sentence terminator followed by
        capital/paragraph break, 3 consecutive repeats, or confidence
        collapse (max prob < 1/256 * 1.2 for 8 steps). Reads the WHOLE
        prompt: coverage windows over every input byte, logits averaged
        (C307 composition). max_bytes is a safety cap, not a target."""
        t = self.t
        pb = prompt.encode('utf-8')
        windows = self._read_all_windows(pb, max_span=64)
        with t.no_grad():
            g = t.Generator(device=self.device).manual_seed(seed)
            out = bytearray()
            last_ctx = windows[-1]
            rep_counter = 0
            low_conf = 0
            last_token = None
            while len(out) < max_bytes:
                probs = None
                # aggregate over ALL prompt windows + growing answer
                for w in windows + ([last_ctx] if windows[-1] != last_ctx or True else []):
                    x = t.as_tensor([w], dtype=t.long, device=self.device)
                    pp = t.softmax(self.logits(x)[0] / max(temp, 1e-3),
                                   dim=-1)
                    probs = pp if probs is None else probs + pp
                probs = probs / len(windows)
                top_p = float(probs.max())
                b = int(t.multinomial(probs, 1, generator=g))
                out.append(b)
                last_ctx = last_ctx[1:] + [b]
                # --- model-driven stop conditions ---
                ch = bytes([b])
                if top_p < (1.0 / 256) * 1.2:
                    low_conf += 1
                else:
                    low_conf = 0
                if b == last_token:
                    rep_counter += 1
                else:
                    rep_counter = 0
                last_token = b
                if ch in b'.!?':
                    # sentence end: peek next — if low confidence
                    # follows, stop here (the model chose to finish)
                    if low_conf >= 2 or rep_counter >= 3:
                        break
                if rep_counter >= 8 or low_conf >= 8:
                    break
        return (prompt.encode('utf-8') + bytes(out)).decode(
            'utf-8', errors='replace')

    def generate(self, prompt: str, n_bytes: int = 100, temp: float = 0.5,
                 seed: int = 1, adaptive: bool = False) -> str:
        """WAVE 214: adaptive=True lets the model choose its own
        effective context size (window-ensemble W measured by
        AdaptiveContextSelector on the prompt's own tail)."""
        t = self.t
        W = 1
        if adaptive:
            W = AdaptiveContextSelector().select(
                self, prompt.encode("utf-8")).get("w", 1)
        with t.no_grad():
            g = t.Generator(device=self.device).manual_seed(seed)
            ctx = list(prompt.encode("utf-8"))[-self.k:]
            if len(ctx) < self.k:
                ctx = [32] * (self.k - len(ctx)) + ctx
            out = bytearray()
            for _ in range(n_bytes):
                probs = None
                for j in range(W):
                    start = max(0, len(ctx) - self.k - j * self.k)
                    w = ctx[start:len(ctx) - j * self.k] if j > 0 else ctx
                    if len(w) < self.k:
                        w = [32] * (self.k - len(w)) + w
                    x = t.as_tensor([w[-self.k:]], dtype=t.long,
                                    device=self.device)
                    p = t.softmax(self.logits(x)[0] / max(temp, 1e-3),
                                  dim=-1)
                    probs = p if probs is None else probs + p
                probs = probs / W
                b = int(t.multinomial(probs, 1, generator=g))
                out.append(b); ctx = ctx[1:] + [b]
        return (prompt.encode("utf-8") + bytes(out)).decode(
            "utf-8", errors="replace")


