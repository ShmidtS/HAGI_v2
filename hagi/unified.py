#!/usr/bin/env python3
"""HAGI Unified — one self-evolving neuro-symbolic processor.

ONE model. Accepts ANY data (bytes). Grows expert-by-expert.
Self-evolution: ingest -> certified learn -> quantize to symbolic ->
fabric weave -> the processor IS the model.

Architecture (2026-11-17, user decision):
  - ONE body (RopeAttentionByteLM, context k=96) trained on all seen;
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
# WAVE 259 (DATA-LOSS FIX): these were module constants computed at
# import time. Tests monkeypatch CKPT_DIR (and experiments override
# it) — but save()/load()/exists() kept writing to the PRODUCTION
# files. That erased the production checkpoint (770 domains, 283
# adapters) on 2026-11-24 via test_live_adapter's u.save(). Now every
# path is derived from the CURRENT CKPT_DIR, and save() refuses to
# touch the real checkpoint from under pytest.
_PROD_CKPT = CKPT_DIR


def _unified_pt():
    return CKPT_DIR / "unified.pt"


def _ledger_js():
    return CKPT_DIR / "unified_ledger.json"


def _guard_prod_write():
    import os as _o
    if _o.environ.get("PYTEST_CURRENT_TEST") and CKPT_DIR == _PROD_CKPT:
        raise RuntimeError(
            "refusing to save into the production checkpoint from a "
            "test: set HAGI_CKPT_DIR / monkeypatch U.CKPT_DIR to a sandbox")


def _sha(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:10]


ADAPTER_MIN_GAIN = 0.005  # paired-validation gain an adapter must clear (WAVE 246)

class Unified:
    """ONE evolving model."""
    # WAVE 261: HAGI_ARCH=hnet -> HNetByteLM (measured: whole-file CE
    # 1.54 vs Rope 1.81 on 4 real files, 4.7x fewer params). Default
    # stays rope until the cycle has run on hnet end-to-end.
    ARCH = os.environ.get("HAGI_ARCH", "rope")

    def __init__(self, k: int = 16, dim: int = 128, hidden: int = 256):
        from hagi import lm as lm_mod

        self.lm_mod = lm_mod
        self.k, self.dim, self.hidden = k, dim, hidden
        # WAVE 229 (URGENT): RopeAttentionByteLM — true attention,
        # 96-byte context (vs k=16 fixed window of CertifiedStepLM).
        self.model = lm_mod.make_lm(self.ARCH, dim, hidden)
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
        # WAVE 243 (user: replay ratio measured, not assumed): the
        # full-book replay (wave168) grew quadratically — 94 domains
        # = 19.7 MB per ingest, the cycle effectively stalled. Sweep
        # on OUR Rope model, conflicting A->B domains, 5 seeds,
        # balanced test CE:
        #   0.20 2.056 | 0.25 1.990 | 0.30 1.986 | 1/e 2.039 |
        #   0.42 2.100 | 0.50 2.213 | 0.75 2.729 | 0 6.682
        # Dense sweep (7 seeds, ±SEM 0.06): 0.26..0.368 all within
        # noise (2.016..2.066) — a PLATEAU 0.25-0.35, not a point
        # optimum. 1/e and 1/pi are NOT distinguished from 0.30.
        # Mid-plateau 0.30 chosen for distance from both edges
        # (<0.2 forgets A, >0.4 starves B). Old domains get 30% of
        # the fresh bytes, stratified over ALL past domains. An
        # explicit replay_cap still overrides (tests / restore).
        if replay_cap is None:
            _fresh_len = len(self._domain_bytes(
                source, self.domains[source]))
            replay_cap = len(data) + int(_fresh_len * 0.30)
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
        # WAVE 243 fix: the old 8KB floor summed to MORE than the
        # whole budget once domains > budget/8KB, and the final
        # [:total_cap] cut then kept only the FIRST domains —
        # stratification silently collapsed to "replay domain #1".
        # Share the budget evenly; the floor never exceeds it.
        per_domain = (max(replay_budget // n_past, 0)
                      if replay_budget else None)
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
        # WAVE 255: capacity growth BEFORE the incumbent snapshot, so the
        # grown block is part of both sides of the paired gate.
        growth = self._maybe_grow(source, data)
        incumbent_state = [q.detach().clone()
                           for q in self.model.params()]
        # WAVE 241 FIX (P0 anti-forgetting gate): incumbent CE must
        # be measured BEFORE any weight update. It was measured
        # after training — both gate sides saw the same trained
        # model, so forgetting was structurally undetectable.
        old_ce_pre_early = self._old_domains_ce(exclude=source)
        reps = []
        for rep_i in range(max(1, repeat)):
            reps.append(
                self.model.train_certified(
                    combined,
                    epochs=epochs,
                    holdout_seed=7 + rep_i * 131 + len(self.domains) * 17,
                )
            )
        # WAVE 241 FIX (external review): the deployed weights are
        # the LAST repeat's — log THAT repeat's CE, not the min over
        # reps (the old min could belong to a different model than
        # the one saved). selection_ce kept for reference.
        ce = reps[-1].get("best_holdout_ce", 99)
        selection_ce = min(r.get("best_holdout_ce", 99)
                           for r in reps)
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
                    d=getattr(self.model, "adapter_dim", self.model.hidden), r=2)
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
            self.model.domain_adapter = None
            # WAVE 246 (HAGI_v2_local_reproduction §2/§7/§8): the
            # adapter is CONDITIONAL, not automatic. Ranks 1/2/4 are
            # trained from the identity (trunk frozen — old domains
            # execute an unchanged path, regression 0 by construction)
            # and accepted only on a PAIRED per-example validation CE
            # gain over the rank-0 baseline (normal-approx lower
            # bound > ADAPTER_MIN_GAIN). HOLD = the adapter stays at
            # B = 0 (identity).
            # Empirical gate, not a Lean certificate.
            self._fit_domain_adapter(source)
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
        # WAVE 241 FIX: use the TRUE pre-training incumbent CE
        # (measured before train_certified above), not a post-hoc
        # re-measurement on already-updated weights.
        old_ce_pre = old_ce_pre_early
        old_ce = old_ce_pre
        # WAVE 233 (hagi_repro patch): measure post UNCONDITIONALLY —
        # the old `> 0` placeholder made the gate train-order only.
        old_ce_post = self._old_domains_ce(exclude=source)
        regression = bool(
            old_ce_pre is not None and old_ce_post is not None
            and old_ce_post > old_ce_pre * 1.25
        )
        repair = self._repair_forgetting(source, old_ce_pre, incumbent_state)
        if repair["rounds"]:
            old_ce_post = self._old_domains_ce(exclude=source)
            regression = bool(
                old_ce_pre is not None and old_ce_post is not None
                and old_ce_post > old_ce_pre * 1.25)
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
        if growth.get("status") == "GROW":
            entry["growth"] = growth
        if repair["rounds"]:
            entry["repair"] = repair
        self.save()
        return {
            "source": source,
            "bytes": len(data),
            "new": is_new,
            "replay_domains": len(self.domains),
            "replay_bytes": len(combined),
            "holdout_ce": ce,
            "old_ce": entry.get("old_ce"),
            "growth": growth,
            "repair": repair,
            "regression": entry.get("regression", False),
            "seconds": round(time.perf_counter() - t0, 1),
        }

    def _domain_bytes(self, name: str, entry: dict) -> bytes:
        """Bytes for a domain — from disk cache (data kept per domain)."""
        p = CKPT_DIR / "domains" / f"{name}.bin"
        return p.read_bytes() if p.exists() else b""

    def _windows(self, source: str, pos):
        """(X, y) windows — the SAME construction as train_guarded's
        gather_all (idx[p-k .. p-1], OLDEST-first, slot L-1 = newest
        byte = query); target = idx[p]. WAVE 258 FIX: the window was
        reversed (newest-first) by a wrong comment in lm.py ("Same
        construction (newest-first)") — gather_all builds oldest-first.
        Measured on the production trunk, 3 chunks: CE oldest-first
        3.51/4.36/3.67 vs newest-first 6.03/6.41/6.09 nats. Every gate
        built on _windows (adapter wave246/249, growth, repair) was
        scoring the model on reversed input."""
        d = self._domain_bytes(source, self.domains[source])
        idx = np.frombuffer(d, dtype=np.uint8).astype(np.int64)
        k = self.model.k
        pos = np.asarray(pos, dtype=np.int64)
        lens = np.minimum(k, pos)                      # short tails: pad byte 0
        X = np.zeros((len(pos), k), dtype=np.int64)
        for i, (p, n_) in enumerate(zip(pos, lens)):
            X[i, k - n_:] = idx[p - n_:p]
        y = idx[pos + k]
        t = self.model.t
        dev = getattr(self.model, "device", "cpu")
        return (t.as_tensor(X, dtype=t.long, device=dev),
                t.as_tensor(y, dtype=t.long, device=dev))

    def _adapter_val_ce(self, ad, X, y) -> "np.ndarray":
        """Per-example CE of the frozen trunk + adapter `ad` (rank 0
        = identity adapter gives the baseline)."""
        t = self.model.t
        self.model.domain_adapter = ad
        out = []
        with t.no_grad():
            for i in range(0, len(X), 128):
                lg = self.model.logits(X[i:i + 128])
                out.append(t.nn.functional.cross_entropy(
                    lg, y[i:i + 128].to(lg.device),
                    reduction="none").cpu().numpy())
        self.model.domain_adapter = None
        return np.concatenate(out)

    def _fit_domain_adapter(self, source: str) -> dict:
        from hagi.trunk_ops import LowRankAdapter, paired_gain
        bank = getattr(self, "adapter_bank", None)
        if bank is None:
            from hagi.trunk_ops import AdapterBank
            bank = self.adapter_bank = AdapterBank(d=self.model.hidden, r=2)
        d = self._domain_bytes(source, self.domains[source])
        k = self.model.k
        n = len(d) - k - 1
        if n < 64:
            return {"status": "HOLD", "reason": "too_short"}
        cut = int(n * 0.85)                      # train | validation
        rng = np.random.default_rng(7)
        base = bank.get(source)
        if np.asarray(base.B).any():             # re-ingest: continue
            cands = [LowRankAdapter(bank.d, base.r, 0)]
            cands[0].A, cands[0].B = np.array(base.A), np.array(base.B)
        else:                                    # fresh: grow 1 -> 2 -> 4
            cands = []
            for r in (1, 2, 4):
                c = LowRankAdapter(bank.d, r, seed=0)
                if base.r == r:
                    c.A, c.B = np.array(base.A), np.array(base.B)
                cands.append(c)
        val = self._windows(source, np.arange(cut, n))
        before = self._adapter_val_ce(LowRankAdapter(bank.d, 1), *val)
        best, report = None, {"status": "HOLD", "baseline_ce": float(before.mean()),
                              "attempts": []}
        for c in cands:
            self.model.domain_adapter = c
            for _ in range(3):
                self.model.adapter_head_step(*self._windows(
                    source, rng.integers(0, cut, 128)))
            self.model.domain_adapter = None
            after = self._adapter_val_ce(c, *val)
            gain, lower = paired_gain(before, after)
            report["attempts"].append({"rank": c.r, "ce": float(after.mean()),
                                       "gain": gain, "lower_bound": lower})
            # lower bound must clear MIN_GAIN (repro default 0.005):
            # on a uniform trunk every paired diff is float noise ~1e-7
            # and a bare "> 0" accepted a no-op (measured, seed 1).
            if lower > ADAPTER_MIN_GAIN and (
                    best is None or after.mean() < best[1]):
                best = (c, float(after.mean()))
        if best is not None:
            bank._adapters[source] = best[0]
            report.update(status="GROW", rank=best[0].r, ce=best[1])
        else:
            bank._adapters[source] = base
        return report

    # WAVE 255 (FunctionPreservingGrowth.lean, GrowthGate): the trunk
    # was capacity-limited — chunk CE crept 2.4 -> 4.9 over 500 chunks
    # and CE on OLD domains 2.5 -> 4.8 while the GrowthGate signal
    # n*G^2/(2L) was 427..1788 (huge gradient at a near-uniform CE).
    # Measured A/B from the SAME checkpoint, 6 real chunks through
    # ingest: 0 blocks forget +0.498, 1 block +0.123 (4x less) AND
    # learns the new chunks better (-0.122 vs -0.054 CE).
    # A fresh block is an exact identity (Wo = 0). It is warmed up
    # with the trunk FROZEN on the fresh domain (so the rollback-by-
    # selection of train_guarded cannot discard it: at epochs=2 that
    # rollback silently zeroed the block — measured) and kept only if
    # the holdout CE does not get worse.
    # WAVE 257 (user: iterativnyj rost — posle rosta pereuchivat' starye
    # domeny, poka zabyvanie ne upadet v nol'). Forgetting is measured on
    # the gate's FIXED old-domain windows (_old_domains_ce, paired pre/post).
    # While it is > FORGET_TOL and the block budget allows: grow one exact-
    # identity block, train ONLY that block (trunk frozen -> the fresh chunk
    # executes an unchanged path) on windows of the old domains, keep the
    # best state seen, re-measure. No improvement -> pop (exact rollback)
    # and stop; improvement but still > 0 -> another block.
    MAX_BLOCKS = 4          # total blocks (repair reserve included)
    GROW_BLOCKS = 0         # WAVE 257: OFF by default — measured (60 chunks,
                            # OLD first-10 CE): blocks +2.07 vs no blocks +0.55
    FORGET_TOL = 0.0
    # WAVE 259: round acceptance = paired gain > empirical-Bernstein margin
    # (RepairForgetting.lean tol; HoldoutGateTwoWay.lean union bound).
    REPAIR_MODE = "replay"   # "replay" (retrain old bytes) | "block"
    REPAIR_ROUNDS = 6       # measured: rounds=3/lr=1e-3 -> +0.343;
                            # rounds=6/lr=3e-4 -> +0.036 (no repair: +0.553)
    REPAIR_LR = 3e-4
    REPAIR_BYTES = 400_000

    def _repair_forgetting(self, exclude: str, pre, incumbent_state) -> dict:
        import torch as _t
        import numpy as _np
        m = self.model
        names = [n for n in self.domains if n != exclude]
        if pre is None or not names:
            return {"rounds": [], "why": "no_old"}
        rng = _np.random.default_rng(23)
        sample = list(rng.choice(names, min(8, len(names)), replace=False))
        sets = []
        for name in sample:
            d = self._domain_bytes(name, self.domains[name])
            if len(d) < m.k + 64:
                continue
            X, y = self._windows(name, rng.integers(0, len(d) - m.k - 1, 256))
            sets.append((X, y))
        if not sets:
            return {"rounds": [], "why": "no_bytes"}
        post = self._old_domains_ce(exclude)
        post_vec = self._old_domains_ce_vec(exclude)[1]
        rounds = []
        if self.REPAIR_MODE == "replay":
            # WAVE 257 (measured, fresh production-size model, 60 chunks,
            # OLD first-10 CE after 60): no blocks +0.55, repair blocks
            # +2.62, frozen repair blocks +3.48, growth blocks +2.07.
            # Blocks overfit the 8 sampled domains and do not carry old
            # memory. What the user asked for literally — RETRAIN the old
            # domains — is a plain training step on old bytes, rolled back
            # unless the paired gate improves. Bounded: REPAIR_ROUNDS.
            for _ in range(self.REPAIR_ROUNDS):
                if post <= pre + self.FORGET_TOL:
                    break
                snap = [q.detach().clone() for q in m.params()]
                old_bytes = b"".join(self._domain_bytes(n, self.domains[n])
                                     for n in sample if n in self.domains)[: self.REPAIR_BYTES]
                if len(old_bytes) < m.k + 64:
                    break
                m.train_certified(old_bytes, epochs=2, holdout_seed=99,
                                    lr=self.REPAIR_LR)
                new_vec = self._old_domains_ce_vec(exclude)[1]
                new = None if new_vec is None else float(new_vec.mean())
                # WAVE 259 (RepairForgetting.lean): a round is accepted
                # only if the paired gain clears the empirical-Bernstein
                # margin (the old 1e-4 tolerance made repair_drop_bound
                # vacuous). n_states = rounds tried so far + this one.
                from hagi.trunk_ops import paired_margin_eb
                margin = (paired_margin_eb(post_vec - new_vec, n_states=len(rounds) + 1)
                          if (new_vec is not None and post_vec is not None) else float("inf"))
                gain = None if new is None else post - new
                if gain is None or not (gain > margin):
                    with _t.no_grad():
                        for q, s0 in zip(m.params(), snap):
                            q.copy_(s0)
                    rounds.append({"ce": post, "after": new, "gain": gain,
                                   "margin": None if margin == float("inf") else round(margin, 4),
                                   "status": "ROLLBACK"})
                    break
                rounds.append({"ce": post, "after": new, "gain": round(gain, 4),
                               "margin": round(margin, 4), "status": "RETRAIN"})
                post, post_vec = new, new_vec
            return {"rounds": rounds, "mode": "replay", "why": "ran" if rounds else "none",
                    "forgot": None if post is None else round(post - pre, 4)}
        why = ("no_post" if post is None else
               "cap" if len(getattr(m, "blocks", [])) >= self.MAX_BLOCKS else
               "none" if post <= pre + self.FORGET_TOL else "ran")
        while post is not None and post > pre + self.FORGET_TOL \
                and len(getattr(m, "blocks", [])) < self.MAX_BLOCKS:
            m.grow(seed=len(m.blocks) + 1)
            blk = m.blocks[-1]
            opt = _t.optim.Adam([blk["W1"], blk["b1"], blk["Wo"]], lr=1e-3)
            best_ce, best_blk = post, {k: v.detach().clone() for k, v in blk.items()}
            for step in range(600):
                X, y = sets[int(rng.integers(0, len(sets)))]
                sel = _t.as_tensor(rng.integers(0, len(y), 64), device=y.device)
                opt.zero_grad()
                _t.nn.functional.cross_entropy(m.logits(X[sel]), y[sel]).backward()
                opt.step()
                if step % 50 == 49:
                    c = self._old_domains_ce(exclude)
                    if c is not None and c < best_ce:
                        best_ce, best_blk = c, {k: v.detach().clone() for k, v in blk.items()}
            with _t.no_grad():
                for k_ in blk:
                    blk[k_].copy_(best_blk[k_])
            new = self._old_domains_ce(exclude)
            if new is None or new >= post - 1e-4:
                m.blocks.pop()                       # not earned
                m.frozen_blocks.discard(len(m.blocks))
                rounds.append({"ce": post, "after": new, "status": "ROLLBACK"})
                break
            incumbent_state.extend(q.detach().clone() for q in blk.values())
            m.frozen_blocks.add(len(m.blocks) - 1)   # measured: if the general train_certified
            # keeps moving a repair block, OLD ends at +2.62 (worse than
            # no blocks at all, +0.55). The block is the old memory: freeze.
            rounds.append({"ce": post, "after": new, "status": "GROW"})
            post = new
        return {"rounds": rounds, "why": why,
                "forgot": None if post is None else round(post - pre, 4)}


    def _maybe_grow(self, source: str, data: bytes) -> dict:
        import torch as _t
        import numpy as _np
        m = self.model
        # WAVE 257: slots are split — growth for the fresh domain may
        # use GROW_BLOCKS; the rest is the reserve of _repair_forgetting
        # (measured: with a shared budget _maybe_grow took all 4 slots in
        # 4 chunks and the repair was capped for the next 55).
        if len(getattr(m, "blocks", [])) >= self.GROW_BLOCKS or len(data) < m.k + 4096:
            return {"status": "SKIP"}
        idx = _np.frombuffer(data, dtype=_np.uint8).astype(_np.int64)
        n = len(idx) - m.k - 1
        # GrowthGate signal on a probe batch of the fresh domain
        pos = _np.random.default_rng(11).integers(m.k, n, 128)
        X, y = self._windows(source, pos) if source in self.domains else (None, None)
        if X is None:
            raw = _t.as_tensor(idx, dtype=_t.long, device=m.device)
            p = _t.as_tensor(pos, dtype=_t.long, device=m.device)
            X = raw[p[:, None] + _t.arange(m.k, device=m.device)[None, :]]
            y = raw[p + m.k]
        params = [q for q in m.params() if q is not getattr(m, "q", None)]
        loss = _t.nn.functional.cross_entropy(m.logits(X), y)
        g = _t.autograd.grad(loss, params, allow_unused=True)
        G2 = float(sum((gi ** 2).sum() for gi in g if gi is not None))
        L = float(loss)
        c_add = (m.dim * m.hidden + m.hidden * m.hidden) / max(1, sum(q.numel() for q in params)) * 10.0
        signal = 128 * G2 / (2 * max(L, 1e-6))
        if signal <= c_add:
            return {"status": "HOLD", "signal": round(signal, 1), "c_add": round(c_add, 1)}
        n_before = len(m.blocks)
        m.grow(seed=n_before + 1)
        blk = m.blocks[-1]
        cut = int(n * 0.9)
        raw = _t.as_tensor(idx, dtype=_t.long, device=m.device)
        k = m.k
        rng = _np.random.default_rng(7)
        def batch(p):
            pt = _t.as_tensor(p, dtype=_t.long, device=m.device)
            return raw[pt[:, None] + _t.arange(k, device=m.device)[None, :]], raw[pt + k]
        Xv, yv = batch(_np.arange(max(cut, n - 2048), n))
        self.model.domain_adapter = None

        def hold_vec():
            with _t.no_grad():
                return _t.nn.functional.cross_entropy(m.logits(Xv), yv, reduction="none").double().cpu().numpy()

        def hold_ce():
            return float(hold_vec().mean())
        before = hold_ce()
        base_before = hold_vec()            # block is an exact identity here
        opt = _t.optim.Adam([blk["W1"], blk["b1"], blk["Wo"]], lr=1e-3)   # trunk frozen
        tr = _np.arange(m.k, cut)
        # WAVE 255b: the block's holdout CE is NOT monotone in the warm-up
        # (measured on the production model: 4.11 -> 6.56 at step 100 ->
        # 2.45 at step 400 with lr=1e-3). Keep the BEST block state seen
        # (best_state contract of train_guarded) instead of the last one —
        # the first version returned the last state and rolled back a
        # block that was already winning.
        best_ce, best_blk = before, {k: v.detach().clone() for k, v in blk.items()}
        n_eval = 1
        for _ in range(2):
            for s in range(0, len(tr), 64):
                sel = rng.choice(tr, min(64, len(tr)), replace=False)
                Xb, yb = batch(sel)
                opt.zero_grad()
                _t.nn.functional.cross_entropy(m.logits(Xb), yb).backward()
                opt.step()
                if s % (64 * 50) == 0:
                    n_eval += 1
                    c = hold_ce()
                    if c < best_ce:
                        best_ce = c
                        best_blk = {k: v.detach().clone() for k, v in blk.items()}
                if s >= 64 * 300:
                    break                      # bounded warm-up (~300 steps/epoch)
        with _t.no_grad():
            for k_ in blk:
                blk[k_].copy_(best_blk[k_])
        after = hold_ce()
        # WAVE 257: strict — measured on the production model a block with
        # ce_before == ce_after (3.4673 -> 3.4673) was accepted as GROW.
        # WAVE 259 (HoldoutGateTwoWay.lean): the gate may accept only if
        # the paired gain clears the empirical-Bernstein margin over the
        # `n_states` block states that were scored on this holdout
        # (best-of-K selection = union bound). Measured on 6 fresh
        # domains: gain 0.50..2.11 vs margin 0.29..0.62 (k=512),
        # 0.12..0.26 (k=2048).
        from hagi.trunk_ops import paired_margin_eb
        diff = base_before - hold_vec()                # old - new, >0 = block helps
        gain = float(diff.mean())
        margin = paired_margin_eb(diff, n_states=n_eval)
        if not (gain > margin):
            m.blocks.pop()                    # not earned -> exact rollback
            return {"status": "HOLD_UNCERTAIN", "ce_before": before, "ce_after": after,
                    "gain": round(gain, 4), "margin": round(margin, 4)}
        return {"status": "GROW", "blocks": len(m.blocks), "signal": round(signal, 1),
                "ce_before": round(before, 4), "ce_after": round(after, 4),
                "gain": round(gain, 4), "margin": round(margin, 4),
                "params": sum(q.numel() for q in m.params())}

    def _live_adapter(self, domain: str | None):
        """WAVE 249: adapter of `domain` for inference, or None.

        Trained adapters were never applied outside ingest (283 of
        them sat unused in the checkpoint). Wiring them in blindly
        would be WRONG: the acceptance gate (wave246) certified each
        adapter against the trunk AS IT WAS at ingest, and the trunk
        has moved since — measured on the current trunk, 1/3 of
        trained adapters HURT their own domain (worst -0.103). So the
        same paired gate is re-run live on the current trunk (4 ms,
        256 probe windows): lower bound > ADAPTER_MIN_GAIN, else
        identity. Measured: every kept adapter helps on an
        independent holdout (3/3, +0.098..+0.255); the gate is
        conservative and also drops some useful ones — the safe side.
        No domain given -> no adapter (neutral path, unchanged)."""
        if not domain:
            return None
        bank = getattr(self, "adapter_bank", None)
        ad = bank._adapters.get(domain) if bank else None
        if ad is None or not np.asarray(ad.B).any():
            return None
        data = self._domain_bytes(domain, self.domains.get(domain, {}))
        k = self.model.k
        if len(data) < k + 600:
            return None
        n = len(data) - k - 1
        pos = np.random.default_rng(7).integers(0, int(n * 0.85), 256)
        X, y = self._windows(domain, pos)
        self.model.domain_adapter = None
        _t = self.model.t
        with _t.no_grad():
            base = _t.nn.functional.cross_entropy(
                self.model.logits(X), y, reduction="none").cpu().numpy()
        after = self._adapter_val_ce(ad, X, y)
        from hagi.trunk_ops import paired_gain
        return ad if paired_gain(base, after)[1] > ADAPTER_MIN_GAIN else None

    class _DomainScope:
        """Applies the live-gated adapter of `domain` for the duration
        of a neural call; always restores the previous state."""
        def __init__(self, u, domain):
            self.u, self.domain = u, domain
        def __enter__(self):
            self.ad = self.u._live_adapter(self.domain)
            self.prev = getattr(self.u.model, "domain_adapter", None)
            self.u.model.domain_adapter = self.ad        # None on gate failure = identity
            return self.ad
        def __exit__(self, *exc):
            self.u.model.domain_adapter = self.prev
            return False

    def _old_domains_ce_vec(self, exclude: str, max_domains: int = 8):
        """WAVE 259: per-window CE on FIXED old-domain windows (the
        same sample/windows as the old scalar gate: rng(11) domains,
        <=256 strided windows each). Returns (names, vector) or
        (None, None). The vector is what a PAIRED gate needs: the
        empirical-Bernstein margin of EmpiricalBernstein.lean applies
        to per-window differences, not to a mean of means."""
        names = [n for n in self.domains if n != exclude]
        if not names:
            return None, None
        rng = np.random.default_rng(11)
        sample = list(rng.choice(names, min(max_domains, len(names)),
                                 replace=False))
        t = self.model.t
        vec = []
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
                ce = t.nn.functional.cross_entropy(
                    self.model.logits(t.as_tensor(X, dtype=t.long, device=dev)),
                    t.as_tensor(y, dtype=t.long, device=dev), reduction="none")
                vec.append(ce.double().cpu().numpy())
        if not vec:
            return None, None
        return sample, np.concatenate(vec)

    def _old_domains_ce(self, exclude: str, max_domains: int = 8) -> float | None:
        """WAVE 199: measured CE on a sample of OLD domains (the
        candidate-vs-incumbent gate-val). None when no old domains.
        Pure CE measurement — no training happens here. WAVE 259: the
        mean of the per-window vector (same number as before)."""
        _, v = self._old_domains_ce_vec(exclude, max_domains)
        return None if v is None else float(v.mean())

    def ingest_file(self, path: Path, epochs: int = 200, repeat: int = 2) -> dict:
        data = path.read_bytes()
        (CKPT_DIR / "domains").mkdir(parents=True, exist_ok=True)
        (CKPT_DIR / "domains" / f"{path.stem}.bin").write_bytes(data)
        return self.ingest(path.stem, data, epochs=epochs, repeat=repeat)

    # ---- book recovery ----

    def reindex_domains(self) -> int:
        """Rebuild the domain book from CKPT_DIR/domains/*.bin (WAVE
        259: the ledger was erased together with the checkpoint).
        sha/bytes are recomputed from the bytes; entries that already
        carry a measured `ce` for the CURRENT model are kept, every
        other one gets ce=None (not measured — never a guessed number).
        Returns the number of added domains."""
        ddir = CKPT_DIR / "domains"
        added = 0
        for pth in sorted(ddir.glob("*.bin")):
            name = pth.stem
            if name in self.domains and self.domains[name].get("ce") is not None:
                continue
            b = pth.read_bytes()
            if len(b) < 4096:
                continue
            self.domains[name] = {"sha": _sha(b), "bytes": len(b), "ce": None}
            added += 1
        return added

    # ---- text (lessons) ----

    def learn_text(
        self, name: str, text: str, epochs: int = 200, repeat: int = 2
    ) -> dict:
        d = CKPT_DIR / "domains"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.bin").write_text(text, encoding="utf-8")
        return self.ingest(name, text.encode("utf-8"), epochs=epochs, repeat=repeat)

    # ---- inference ----

    def ask(self, prompt: str, n: int = 400,
            domain: str | None = None) -> str:
        with self._DomainScope(self, domain):
            r = self.model.generate(prompt, n_bytes=n)
        return r[len(prompt) :].strip() if r.startswith(prompt) else r

    def ask_auto(self, prompt: str, max_bytes: int = 4096,
                 domain: str | None = None) -> str:
        """WAVE 228: the model reads the WHOLE prompt (coverage
        windows) and decides ITSELF how long the answer is
        (self-terminating generation). `domain` (WAVE 249): apply that
        domain's adapter iff it passes the live paired gate."""
        with self._DomainScope(self, domain):
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

        # WAVE 261: named by the model itself (Rope keeps its legacy names:
        # emb,q,W1,b1,W2,b2 + blk{i}.* ; HNet has its own set)
        st = {n: p_.detach().cpu() for n, p_ in zip(self.model.param_names(), self.model.params())}
        # WAVE 255: function-preserving attention blocks
        if getattr(self.model, "ARCH", "rope") == "rope":
            st["blocks"] = [
                {k: v.detach().cpu() for k, v in blk.items()}
                for blk in getattr(self.model, "blocks", [])]
        st.update(
            {"k": self.model.k, "dim": self.model.dim, "hidden": self.model.hidden}
        )
        st["arch"] = getattr(self.model, "ARCH", "rope")
        st["frozen_blocks"] = sorted(getattr(self.model, "frozen_blocks", ()))
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
        # WAVE 236m: persist the law registry — laws survive restart.
        reg = getattr(self, "law_registry", None)
        if reg:
            st["law_registry"] = {
                dom: [{"feature": e["feature"],
                       "verify": e["verify"],
                       "support": e.get("support"),
                       "coefs": e["bp"].q_param[0],
                       "bias": e["bp"].q_param[1],
                       "basis": e["bp"].basis,
                       "payload": e["bp"].payload,
                       "accepted_to_registry": e.get(
                           "accepted_to_registry", True),
                       "commit_id": e.get("commit_id"),
                       "shift": e.get("shift"),
                       "n_shifts": e.get("n_shifts"),
                       "evidence": e.get("evidence", [])}
                      for e in v]
                for dom, v in reg.items()}
        _guard_prod_write()
        _unified_pt().parent.mkdir(parents=True, exist_ok=True)
        # WAVE 230: atomic checkpoint write — a crash mid-save must
        # not leave a truncated .pt / half-written ledger.
        import os as _os
        tmp = _unified_pt().with_suffix(".pt.tmp")
        _t.save(st, tmp)
        _os.replace(tmp, _unified_pt())
        tmp_l = _ledger_js().with_suffix(".json.tmp")
        tmp_l.write_text(
            json.dumps(self.domains, ensure_ascii=False, indent=1),
            encoding="utf-8")
        _os.replace(tmp_l, _ledger_js())

    @staticmethod
    def exists() -> bool:
        return _unified_pt().exists()

    @classmethod
    def load(cls) -> "Unified":
        import torch as _t
        from hagi import lm as lm_mod

        u = cls.__new__(cls)
        u.lm_mod = lm_mod
        st = _t.load(_unified_pt(), weights_only=False)
        u.k, u.dim, u.hidden = st["k"], st["dim"], st["hidden"]
        u.model = lm_mod.make_lm(st.get("arch", "rope"), u.dim, u.hidden,
                                 theta0=st.get("theta0", 1.0), lam=st.get("lam", 0.05))
        if "adapter_bank" in st:
            from hagi.trunk_ops import AdapterBank
            bank = u.adapter_bank = AdapterBank(
                d=getattr(u.model, "adapter_dim", u.model.hidden), r=2)
            for dom, ab in st["adapter_bank"].items():
                ad = bank.get(dom)
                ad.r = _t.as_tensor(ab["A"]).shape[0]  # WAVE 246: rank varies
                ad.A = _t.as_tensor(ab["A"])
                ad.B = _t.as_tensor(ab["B"])
        if "law_registry" in st:
            import numpy as _np
            from hagi.invention_multi import _affine_k
            from hagi.invention import ConstructBlueprint
            u.law_registry = {}
            for dom, laws in st["law_registry"].items():
                entries = []
                for e in laws:
                    ffn = _affine_k(tuple(e["coefs"]), e["bias"])
                    bp = ConstructBlueprint(
                        e["feature"], (tuple(e["coefs"]), e["bias"]),
                        ffn, e["basis"], e["payload"], 9)
                    entries.append({"feature": e["feature"],
                                    "bp": bp,
                                    "verify": e["verify"],
                                    "support": e.get("support"),
                                    "accepted_to_registry": e.get(
                                        "accepted_to_registry", True),
                                    "commit_id": e.get("commit_id"),
                                    "shift": e.get("shift"),
                                    "n_shifts": e.get("n_shifts"),
                                    "evidence": e.get("evidence", [])})
                if entries:
                    u.law_registry[dom] = entries
        with _t.no_grad():
            for n, p_ in zip(u.model.param_names(), u.model.params()):
                if n in st:
                    p_.copy_(st[n].to(p_.device))
        # legacy Rope checkpoints store grown blocks separately
        for blk in st.get("blocks", []) or []:
            u.model.grow()
            live = u.model.blocks[-1]
            with _t.no_grad():
                for k in live:
                    live[k].copy_(blk[k].to(live[k].device))
        u.model.frozen_blocks = set(st.get("frozen_blocks", []))
        u.domains = (
            json.loads(_ledger_js().read_text(encoding="utf-8")) if _ledger_js().exists() else {}
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
        with self._DomainScope(self, prefer_domain):
            b = int(self.model.logits(x)[0].argmax())
            used = self.model.domain_adapter is not None
        return {"byte": b, "status": "NEURAL_GUESS", "adapter": used,
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
