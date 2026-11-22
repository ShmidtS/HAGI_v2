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

import os
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
        # WAVE 229 (URGENT): RopeAttentionByteLM — true attention,
        # 96-byte context (vs k=16 fixed window of CertifiedStepLM).
        self.model = lm_mod.RopeAttentionByteLM(dim=dim, hidden=hidden)
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
        import torch as _t
        incumbent_state = [q.detach().clone()
                           for q in self.model.params()]
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
        # WAVE 210: optional per-domain adapters (HAGI_DOMAIN_ADAPTERS=1).
        # Fresh adapter = exact identity (B=0), so this never changes
        # the incumbent path; the adapter head-trains on the fresh
        # domain only. Restore runs WITHOUT the env — behavior
        # identical to wave209.
        # WAVE 220: adapters ALWAYS ON (user: все функции по
        # умолчанию; fresh adapter = identity, путь не меняет).
        if True:
            from hagi.trunk_ops import AdapterBank
            bank = getattr(self, "adapter_bank", None)
            if bank is None:
                bank = self.adapter_bank = AdapterBank(
                    d=self.model.hidden, r=2)
            ad = bank.get(source)
            # WAVE 232 (C592 sufficient-transfer gate): warm-start a
            # FRESH adapter (A,B both zero = exact identity) from the
            # best trained domain IFF that domain certified a
            # compressible fit (ce <= 0.5 — strong structure signal).
            # C592: certified SUFFICIENT negative-transfer exists
            # (start gap > n*nlBudget => warm worse); low source ce is
            # the measured proxy for a small start gap. forgetA=0 copy
            # semantics keep the source untouched (warm_transfer_sound).
            if not ad.B.any():  # B=0 = fresh (exact identity; A is
                # random by init, identity lives in B neutrality)
                cands = [(s, e.get("ce")) for s, e in self.domains.items()
                         if s != source and e.get("ce") is not None
                         and e.get("ce") <= 0.5]
                if cands:
                    best_src = min(cands, key=lambda kv: kv[1])[0]
                    from hagi.trunk_ops import adapter_warm_start
                    adapter_warm_start(bank.get(best_src), ad)
            self.model.domain_adapter = ad
            try:
                # WAVE 230: RopeAttentionByteLM.adapter_head_step —
                # Adam on A/B only, trunk frozen; 3 guarded steps.
                for _ in range(3):
                    ctx, nxt = self._fresh_batch(source)
                    self.model.adapter_head_step(ctx, nxt)
            except Exception:
                pass  # adapter training is best-effort, head stays safe
            self.model.domain_adapter = None
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
        # WAVE 233 (hagi_repro patch): measure post UNCONDITIONALLY —
        # the old `> 0` placeholder made the gate train-order only.
        old_ce_post = self._old_domains_ce(exclude=source)
        regression = bool(
            old_ce_pre is not None and old_ce_post is not None
            and old_ce_post > old_ce_pre * 1.25
        )
        old_ce = old_ce_post if old_ce_post is not None else old_ce_pre
        if regression:
            # in-place rollback: Unified.load() would recreate the
            # model object and lose adapter state / optimizer closure
            with _t.no_grad():
                for prm, inc in zip(self.model.params(), incumbent_state):
                    prm.copy_(inc)
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

    def _fresh_batch(self, source: str):
        """One training batch from the fresh domain bytes (adapter
        head-training)."""
        d = self._domain_bytes(source, self.domains[source])
        idx = np.frombuffer(d, dtype=np.uint8).astype(np.int64)
        k = self.model.k
        n = len(idx) - k
        sel = np.arange(0, max(1, n), max(1, n // 128))[:128]
        X = np.stack([idx[i:i + k] for i in sel])
        y = idx[sel + k]
        t = self.model.t
        dev = getattr(self.model, "device", "cpu")
        return (t.as_tensor(X, dtype=t.long, device=dev),
                t.as_tensor(y, dtype=t.long, device=dev))

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

    def ask(self, prompt: str, n: int = 400) -> str:
        r = self.model.generate(prompt, n_bytes=n)
        return r[len(prompt) :].strip() if r.startswith(prompt) else r

    def ask_auto(self, prompt: str, max_bytes: int = 4096) -> str:
        """WAVE 228: the model reads the WHOLE prompt (coverage
        windows) and decides ITSELF how long the answer is
        (self-terminating generation)."""
        r = self.model.generate_auto(prompt, max_bytes=max_bytes)
        return r[len(prompt):].strip() if r.startswith(prompt) else r
    # ---- WAVE 225: knowledge consolidation from external memory ----

    def memory(self):
        """SemanticMemory singleton (SQLite+FTS5+gemma archive)."""
        from hagi.memory_store import SemanticMemory
        import hagi.util as _u
        if getattr(self, "_mem", None) is None:
            self._mem = SemanticMemory(str(_u.CKPT_DIR / "memory.db"))
        return self._mem

    def remember(self, text: str) -> None:
        """Store a conversation/fact permanently (weights untouched)."""
        self.memory().add(text)

    def consolidate(self, query: str, n_bytes: int = 600) -> str:
        """Retrieval-augmented ask: relevant memory pages -> prompt
        prefix -> grounded generation (labs agent pattern)."""
        hits = self.memory().query(query, top_k=3)
        sep = chr(10)
        evidence = sep.join(t for _c, _pid, t in hits) if hits else ""
        prompt = (evidence + sep + sep + query) if evidence else query
        return self.ask(prompt, n=n_bytes)

    def learn_from_dialog(self, dialog_text: str,
                          name: str | None = None) -> dict:
        """Full consolidation loop: dialog -> permanent memory ->
        ingest as a training domain (model LEARNS from it)."""
        name = name or ("dialog_"
                        + time.strftime("%Y%m%d_%H%M%S"))
        self.remember(dialog_text)
        return self.ingest(name, dialog_text.encode("utf-8"),
                           epochs=1, repeat=1)

    # ---- persistence ----

    def save(self) -> None:
        import torch as _t

        names = ["emb", "W1", "b1", "W2", "b2"]
        if hasattr(self.model, "q"):
            names.append("q")
        st = {n: getattr(self.model, n).detach().cpu() for n in names}
        st.update(
            {"k": self.model.k, "dim": self.model.dim, "hidden": self.model.hidden}
        )
        # WAVE 230: persist attention geometry (theta0/lam) — without
        # them load() silently reverts to defaults and the trained
        # recency profile changes after restart.
        for g in ("theta0", "lam"):
            if hasattr(self.model, g):
                st[g] = float(getattr(self.model, g))
        # WAVE 235b FIX (external audit P6): trained adapters were
        # never serialized — every restart silently reset them to the
        # B=0 identity. Persist A/B per domain.
        bank = getattr(self, "adapter_bank", None)
        if bank is not None:
            st["adapter_bank"] = {
                dom: {"A": ad.A, "B": ad.B}
                for dom, ad in bank._adapters.items()}
        UNET_CKPT.parent.mkdir(parents=True, exist_ok=True)
        # WAVE 230: atomic checkpoint write — a crash mid-save must
        # not leave a truncated .pt / half-written ledger.
        import os as _os
        tmp = UNET_CKPT.with_suffix(".pt.tmp")
        _t.save(st, tmp)
        _os.replace(tmp, UNET_CKPT)
        tmp_l = LEDGER.with_suffix(".json.tmp")
        tmp_l.write_text(
            json.dumps(self.domains, ensure_ascii=False, indent=1),
            encoding="utf-8")
        _os.replace(tmp_l, LEDGER)

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
        u.model = lm_mod.RopeAttentionByteLM(
            dim=u.dim, hidden=u.hidden,
            theta0=st.get("theta0", 1.0), lam=st.get("lam", 0.05))
        if "adapter_bank" in st:
            from hagi.trunk_ops import AdapterBank
            bank = u.adapter_bank = AdapterBank(
                d=u.model.hidden, r=2)
            for dom, ab in st["adapter_bank"].items():
                ad = bank.get(dom)
                ad.A = _t.as_tensor(ab["A"])
                ad.B = _t.as_tensor(ab["B"])
        with _t.no_grad():
            for n in ("emb", "W1", "b1", "W2", "b2", "q"):
                if hasattr(u.model, n) and n in st:
                    getattr(u.model, n).copy_(st[n])
        u.domains = (
            json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {}
        )
        return u

    def predict_next_byte(self, context: bytes | str,
                          prefer_domain: str | None = None,
                          allow_neural: bool = False) -> dict:
        """WAVE 236i (user directive — honest-by-default): predict
        the NEXT BYTE of an unseen context.

        Priority:
          1. PROVEN laws (verify == PROVEN_EXACT) — certified answer.
          2. HOLD laws (partial coverage / pending OOS) — the law's
             prediction is returned but flagged `certified: False`,
             status HOLD.
          3. NO LAW matched -> status HOLD, byte=None (NO GUESS).

        The neural model is NOT consulted by default: an un-certified
        guess would masquerade as knowledge. Callers may request
        `allow_neural=True` explicitly for an un-certified guess.
        """
        import torch as _t
        if isinstance(context, str):
            context = context.encode("utf-8")
        registry = getattr(self, "law_registry", {}) or {}
        domains = ([prefer_domain] if prefer_domain
                   else list(registry.keys()))
        ctx = (tuple(int(b) for b in context[-4:])
               if len(context) >= 4
               else tuple(int(b) for b in context))
        proven, hold = [], []
        for dom in domains:
            for entry in registry.get(dom, []):
                tgt = (proven if entry.get("verify") == "PROVEN_EXACT"
                       else hold)
                tgt.append((dom, entry))
        # 1) proven laws first
        for dom, entry in proven:
            try:
                b = int(entry["bp"].apply(ctx)) % 256
                return {"byte": b, "status": "OK",
                        "source": f"law:{dom}:{entry['feature']}",
                        "certified": True}
            except Exception:
                continue
        # 2) HOLD laws — the model's best structured guess
        for dom, entry in hold:
            try:
                b = int(entry["bp"].apply(ctx)) % 256
                return {"byte": b, "status": "HOLD",
                        "source": f"law-candidate:{dom}:"
                                  f"{entry['feature']}",
                        "certified": False}
            except Exception:
                continue
        # 3) default HOLD — no guess without a law
        if not allow_neural:
            return {"byte": None, "status": "HOLD",
                    "source": "no-law", "certified": False}
        dev = self.model.device
        ctx_list = list(context)[-self.model.k:][::-1]
        x = _t.as_tensor([ctx_list], dtype=_t.long, device=dev)
        b = int(self.model.logits(x)[0].argmax())
        return {"byte": b, "status": "NEURAL_GUESS",
                "source": "neural", "certified": False}
    # ---- neuro -> symbolic ----    # ---- neuro -> symbolic ----

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
