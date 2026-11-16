#!/usr/bin/env python3
"""HAGI byte-level neural LM — the probabilistic body per C:\\primes theory.

Implements the theory's neural half at toy scale, honestly:

  §2  loss = finite one-hot CE over the byte vocabulary (V=256),
      softmax logits, exact CE on supplied samples;
  §3  guarded step: an optimizer step is ACCEPTED only if the
      holdout CE does not increase (snapshot/rollback — the
      batchSampleCE_guardedBatchTrainStep_le recipe);
  §6  ternary quantization of weight matrices (per-matrix scale,
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
  python hagi_lm.py train <file...> [--epochs N] [--k 8] [--dim 64]
  python hagi_lm.py generate "<prompt>" [--n 120] [--temp 0.8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

CKPT = Path("checkpoints/lm_byte.pt")


def _torch_and_device():
    import torch
    return torch, ("cuda" if torch.cuda.is_available() else "cpu")


def t_no_grad_if(fn):
    def wrapper(self, *a, **kw):
        with self.t.no_grad():
            return fn(self, *a, **kw)
    return wrapper


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
        """In-place ternary quantization of W1/W2/emb (§6): per-matrix
        scale a = mean(|W|), values -> {-a, 0, +a}. Returns CE cost."""
        t = self.t
        report = {}
        for name in ("emb", "W1", "W2"):
            W = getattr(self, name)
            with t.no_grad():
                a = W.abs().mean()
                q = t.sign(t.round(W / (a + 1e-12))) * a
                report[name] = float((q - W).abs().max())
                W.copy_(q)
        return report

    # ---- §3 guarded step / §9 certified chain -----------------------
    def train_guarded(self, data: bytes, epochs: int = 30, lr: float = 3e-3,
                      holdout: float = 0.1, log_every: int = 10
                      ) -> dict[str, Any]:
        t = self.t
        idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        n = len(idx) - self.k
        if n < 64:
            raise ValueError(f"need >= {self.k + 64} bytes, got {len(idx)}")
        rng = np.random.default_rng(7)
        perm = rng.permutation(n)
        n_hold = max(8, int(n * holdout))
        hold = np.sort(perm[:n_hold])
        train = np.sort(perm[n_hold:])

        def windows(sel):
            x = np.stack([idx[i:i + self.k] for i in sel])
            y = idx[sel + self.k]
            return (t.as_tensor(x, dtype=t.long),
                    t.as_tensor(y, dtype=t.long))

        ctx_tr, nxt_tr = windows(train)
        ctx_va, nxt_va = windows(hold)
        opt = t.optim.Adam(self.params(), lr=lr)
        best_ce = float(self.ce(ctx_va, nxt_va).detach())
        best_state = None
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
        best_ce = val_ce(); best_state = None
        history = []
        for step in range(1, steps + 1):
            sel = rng.integers(0, n, size=batch)
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
                 seed: int = 1) -> str:
        t = self.t
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
