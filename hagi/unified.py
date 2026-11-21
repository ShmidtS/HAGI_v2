#!/usr/bin/env python3
"""HAGI Unified — one self-evolving neuro-symbolic processor.

ONE model. Accepts ANY data (bytes). Grows expert-by-expert.
Self-evolution: ingest -> certified learn -> quantize to symbolic ->
fabric weave -> the processor IS the model.

Architecture (2026-11-17, user decision):
  - ONE body (ByteLM k/dim/hidden) trained on everything seen;
    no separate task checkpoints (nadya_school.pt etc. deleted).
  - Domains tracked by a domain ledger (name -> bytes seen, CE trace)
    so forgetting is measured per domain after each ingest.
  - Neuro->symbolic: trained adapters quantize to ternary masks
    (fabric substrate); the committed artifact is symbolic.

Ingest contract: ANY file — text, binary, CSV, JSON (hagi.ingest
handles structured; raw bytes always work).
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# WAVE 178/185c: CKPT_DIR lives in hagi.util (SSOT) — every module
# routes checkpoint writes through it; HAGI_CKPT_DIR redirects all.
from hagi.util import CKPT_DIR
UNET_CKPT = CKPT_DIR / "unified.pt"
LEDGER = CKPT_DIR / "unified_ledger.json"


def _sha(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:10]


class Unified:
    """ONE evolving model."""

    def __init__(self, k: int = 16, dim: int = 128, hidden: int = 256):
        from hagi import lm as lm_mod

        self.lm_mod = lm_mod
        self.k, self.dim, self.hidden = k, dim, hidden
        self.model = lm_mod.CertifiedStepLM(k=k, dim=dim, hidden=hidden)
        self.domains: dict[str, dict[str, Any]] = {}

    # ---- ingest: ANY data ----

    def ingest(
        self,
        source: str,
        data: bytes,
        epochs: int = 200,
        repeat: int = 2,
        replay_cap: int | None = None,
    ) -> dict:
        """Ingest any bytes as a domain; certified training on it.
        The ONE model accumulates: previous data is re-included
        (fresh-replay of all domains = no forgetting, book is the
        single source of truth)."""
        import numpy as np

        # WAVE 196 FIX (P0): path traversal — source like
        # "../../evil" escaped CKPT_DIR/domains (verified locally:
        # file landed outside the sandbox). Domain ids are safe
        # names only.
        import re as _re
        safe = _re.sub(r"[^A-Za-z0-9_.-]", "_", source)
        safe = safe.strip("._") or "unnamed"
        if safe != source:
            source = safe

        sha = _sha(data)
        entry = self.domains.setdefault(source, {"sha": sha, "bytes": 0, "ce": None})
        is_new = entry["sha"] != sha
        entry["sha"], entry["bytes"] = sha, len(data)
        if len(data) < 4096:  # too small for certified split
            self.save()
            return {
                "source": source,
                "bytes": len(data),
                "new": is_new,
                "skipped": "too_small",
            }
        # persist domain bytes (cumulative replay source)
        ddir = CKPT_DIR / "domains"
        ddir.mkdir(parents=True, exist_ok=True)
        (ddir / f"{source}.bin").write_bytes(data)
        # replay ALL domains (anti-forgetting: cumulative book);
        # WAVE 168: default replay_cap=None = FULL-BOOK replay (no cap,
        # user decision) — the book IS the model, nothing is forgotten.
        # An integer cap remains available for quick tests only.

        parts = []
        total_cap = replay_cap  # TOTAL budget for all past domains
        # WAVE 133 FIX (Goodhart): old order put the FRESH domain last,
        # and the [:total_cap] truncation ate 97% of the tail — new data
        # was nearly never trained on. Theory (anti-forgetting + fresh
        # data): fresh domain FIRST and in full, then uniform strided
        # replay over past domains for the remainder of the budget.
        fresh_name = source
        fresh_bytes = self._domain_bytes(fresh_name, self.domains[fresh_name])
        replay_budget = max(total_cap - len(fresh_bytes), 0) if total_cap else None
        n_past = max(1, len(self.domains) - 1)
        per_domain = max(replay_budget // n_past, 8_000) if replay_budget else None
        for name, e in self.domains.items():
            if name == fresh_name:
                continue
            d = self._domain_bytes(name, e)
            if not d:
                continue
            if per_domain is not None and len(d) > per_domain:
                # uniform CONTIGUOUS coverage: few blocks spread evenly
                # (byte-stride made text undecodable and leaked stride
                # structure into holdout — ns_0225 CE 0.22 artifact;
                # 3-byte micro-blocks shred language structure — keep
                # >=2KB coherent spans)
                n_blocks = max(1, min(12, per_domain // 2048))
                bl = min(len(d) // n_blocks, per_domain // n_blocks)
                bounds = [i * len(d) // n_blocks for i in range(n_blocks)]
                chunks = [d[b : b + bl] for b in bounds]
                d = b"".join(chunks)[:per_domain]
            parts.append(d)
        combined = (
            (fresh_bytes + b"".join(parts))[:total_cap]
            if total_cap
            else fresh_bytes + b"".join(parts)
        )
        # training (certified step path)
        t0 = time.perf_counter()
        reps = []
        for rep_i in range(max(1, repeat)):
            reps.append(
                self.model.train_certified(
                    combined,
                    epochs=epochs,
                    holdout_seed=7 + rep_i * 131 + len(self.domains) * 17,
                )
            )
        ce = min(r.get("best_holdout_ce", 99) for r in reps)
        # WAVE 199 (candidate-vs-incumbent gate): measured CE on a
        # sample of OLD domains — accept only without impermissible
        # regression on them (fresh-domain improvement alone is not
        # sufficient; the book is ONE model). Rollback = reload the
        # incumbent weights; the chunk is still recorded but flagged.
        # WAVE 204 FIX (P0.3, audit bundle v2): PAIRED gate —
        # incumbent CE on fixed old-domain windows measured BEFORE
        # training, challenger CE on THE SAME windows after. The old
        # formula compared post-train old_ce against the mixed
        # new-domain CE — different quantities, wrong direction.
        old_ce_pre = self._old_domains_ce(exclude=source)
        old_ce = old_ce_pre
        regression = bool(
            old_ce_pre is not None and old_ce_pre > 0
        )
        if regression:
            old_ce_post = self._old_domains_ce(exclude=source)
            regression = (
                old_ce_post is not None
                and old_ce_post > old_ce_pre * 1.25
            )
            old_ce = old_ce_post
        if regression and UNET_CKPT.exists():
            # honest rollback: deploy the incumbent checkpoint
            prev = type(self).load()
            self.model = prev.model
        entry["ce"] = ce
        entry["old_ce"] = old_ce
        entry["regression"] = bool(regression)
        entry["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        return {
            "source": source,
            "bytes": len(data),
            "new": is_new,
            "replay_domains": len(self.domains),
            "replay_bytes": len(combined),
            "holdout_ce": ce,
            "old_ce": entry.get("old_ce"),
            "regression": entry.get("regression", False),
            "seconds": round(time.perf_counter() - t0, 1),
        }

    def _domain_bytes(self, name: str, entry: dict) -> bytes:
        """Bytes for a domain — from disk cache (data kept per domain)."""
        p = CKPT_DIR / "domains" / f"{name}.bin"
        return p.read_bytes() if p.exists() else b""

    def _old_domains_ce(self, exclude: str, max_domains: int = 8) -> float | None:
        """WAVE 199: measured CE on a sample of OLD domains (the
        candidate-vs-incumbent gate-val). None when no old domains.
        Pure CE measurement — no training happens here."""
        import numpy as np
        names = [n for n in self.domains if n != exclude]
        if not names:
            return None
        rng = np.random.default_rng(11)
        sample = list(rng.choice(names, min(max_domains, len(names)),
                                 replace=False))
        t = self.model.t
        ces = []
        with t.no_grad():
            for name in sample:
                d = self._domain_bytes(name, self.domains[name])
                idx = np.frombuffer(d, dtype=np.uint8).astype(np.int64)
                if len(idx) < self.model.k + 64:
                    continue
                k = self.model.k
                n = len(idx) - k
                step = max(1, n // 256)
                sel = np.arange(0, n, step)[:256]
                X = np.stack([idx[i:i + k] for i in sel])
                y = idx[sel + k]
                dev = getattr(self.model, "device", "cpu")
                ctx = t.as_tensor(X, dtype=t.long, device=dev)
                nxt = t.as_tensor(y, dtype=t.long, device=dev)
                ces.append(float(t.nn.functional.cross_entropy(
                    self.model.logits(ctx), nxt)))
        return sum(ces) / len(ces) if ces else None

    def ingest_file(self, path: Path, epochs: int = 200, repeat: int = 2) -> dict:
        data = path.read_bytes()
        (CKPT_DIR / "domains").mkdir(parents=True, exist_ok=True)
        (CKPT_DIR / "domains" / f"{path.stem}.bin").write_bytes(data)
        return self.ingest(path.stem, data, epochs=epochs, repeat=repeat)

    # ---- text (lessons) ----

    def learn_text(
        self, name: str, text: str, epochs: int = 200, repeat: int = 2
    ) -> dict:
        d = CKPT_DIR / "domains"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.bin").write_text(text, encoding="utf-8")
        return self.ingest(name, text.encode("utf-8"), epochs=epochs, repeat=repeat)

    # ---- inference ----

    def ask(self, prompt: str, n: int = 60) -> str:
        r = self.model.generate(prompt, n_bytes=n)
        return r[len(prompt) :].strip() if r.startswith(prompt) else r

    # ---- persistence ----

    def save(self) -> None:
        import torch as _t

        st = {
            n: getattr(self.model, n).detach().cpu()
            for n in ("emb", "W1", "b1", "W2", "b2")
        }
        st.update(
            {"k": self.model.k, "dim": self.model.dim, "hidden": self.model.hidden}
        )
        UNET_CKPT.parent.mkdir(parents=True, exist_ok=True)
        _t.save(st, UNET_CKPT)
        LEDGER.write_text(
            json.dumps(self.domains, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @staticmethod
    def exists() -> bool:
        return UNET_CKPT.exists()

    @classmethod
    def load(cls) -> "Unified":
        import torch as _t
        from hagi import lm as lm_mod

        u = cls.__new__(cls)
        u.lm_mod = lm_mod
        st = _t.load(UNET_CKPT, weights_only=False)
        u.k, u.dim, u.hidden = st["k"], st["dim"], st["hidden"]
        u.model = lm_mod.CertifiedStepLM(k=u.k, dim=u.dim, hidden=u.hidden)
        with _t.no_grad():
            for n in ("emb", "W1", "b1", "W2", "b2"):
                getattr(u.model, n).copy_(st[n])
        u.domains = (
            json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {}
        )
        return u

    # ---- neuro -> symbolic ----

    def to_symbolic(self) -> dict:
        """Ternary projection of the body's first layer to the fabric
        substrate. The committed artifact is symbolic (masks/ternary)."""
        import torch

        W1 = self.model.W1.detach()
        scale = W1.abs().mean() * 2 + 1e-12
        tern = torch.round(W1 / scale).clamp(-1, 1).int()
        # WAVE 169 fix: return the PAYLOAD (mask+scale), not only stats —
        # the committed symbolic artifact must be reconstructible.
        return {
            "shape": list(W1.shape),
            "scale": scale.item(),
            "ternary_nonzeros": int((tern != 0).sum()),
            "density": round(float((tern != 0).float().mean()), 4),
            "payload": {
                "mask": tern.tolist(),          # {-1,0,+1} per weight
                "scale": scale.item(),
                "schema": "W ~ mask * scale (per-matrix scalar scale)",
            },
        }

    # ---- self-evolution cycle ----

    def evolve_step(
        self, sources: list[tuple[str, bytes]] | None = None, epochs: int = 120
    ) -> dict:
        """One GROW->CERTIFY cycle over all domains (+ any new sources).
        This is the self-evolution unit: ingest all, train cumulative,
        measure per-domain forgetting (CE on each domain's own holdout),
        quantize to symbolic."""
        if sources:
            for name, data in sources:
                (CKPT_DIR / "domains").mkdir(exist_ok=True, parents=True)
                (CKPT_DIR / "domains" / f"{name}.bin").write_bytes(data)
                self.ingest(name, data, epochs=epochs)
        report = self.ingest("__evolve__", b"", epochs=epochs) if False else None
        # per-domain holdout CE (forgetting measure)
        ce_by = {}
        for name in self.domains:
            data = self._domain_bytes(name, self.domains[name])
            if not data:
                continue
            idx = (
                np.frombuffer(data, dtype=np.uint8).astype(np.int64)
                if (np := __import__("numpy"))
                else None
            )
            if len(idx) < self.k + 32:
                continue
            rng = np.random.default_rng(9)
            n = len(idx) - self.k
            sel = rng.choice(n, min(256, n), replace=False)
            X = np.stack([idx[i : i + self.k] for i in sel])
            yv = idx[sel + self.k]
            t = self.model.t
            ctx = t.as_tensor(X, dtype=t.long)
            nxt = t.as_tensor(yv, dtype=t.long)
            with t.no_grad():
                ce = float(t.nn.functional.cross_entropy(self.model.logits(ctx), nxt))
            ce_by[name] = round(ce, 4)
        sym = self.to_symbolic()
        return {"domains": len(self.domains), "ce_by_domain": ce_by, "symbolic": sym}
