#!/usr/bin/env python3
"""Neuro-symbolic self-evolution cycle — ONE model, fabric inside.

Loop (per iteration):
  1. GROW (neural):   sweep C:/ → sample files → certified training
                      chunk on the cumulative corpus (ONE body).
  2. DERIVE (symbolic): from the trained body, derive boolean fabric
                      tasks from what was just learned — e.g. sign
                      structure of W1 rows / next-bit predicates on
                      high-energy bytes. Each task = truth table.
  3. WEAVE (fabric):  certified-first evolution (GROW correctness +
                      neutral drift, then OPTIMIZE cost hard-gated) —
                      the fabric SELF-ASSEMBLES a circuit per task.
  4. COMMIT (tape-out): certified circuits appended to the processor
                      netlist (fabric.json); neuro checkpoint saved.

Every commit carries a certificate (exhaustive truth-table check).
No commit without one — the processor is built only from proven
circuits. This wires fabric evolution INTO the training loop.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hagi import fabric, unified  # noqa: E402
from hagi.cycle import (compose_adder2, commit, derive_tasks,  # noqa: E402
                        fusion_fastpath, weave_task)
from hagi.util import CKPT_DIR
from scripts.feed_all_c import (CHUNK_TARGET, SKIP_DIRS, iter_files,
                                load_state, save_state, sample_bytes)  # noqa: E402

from hagi.util import CKPT_DIR
TAPEOUT = CKPT_DIR / "processor_netlist.json"

# ---------- DERIVE: fabric tasks FROM the trained body ----------

def main() -> None:
    max_minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 480.0
    deadline = time.time() + max_minutes * 60
    u = unified.Unified.load() if unified.Unified.exists() \
        else unified.Unified()
    st = load_state()
    seen = st["seen"]
    chunk, chunk_bytes = [], 0
    files_now = 0
    it = 0
    (CKPT_DIR / 'inbox').mkdir(parents=True, exist_ok=True)
    print(f"neuro-symbolic cycle start: chunks={st['chunks']}", flush=True)
    while time.time() < deadline:
        # ---- 1. GROW: collect a chunk from C:/ (persistent cursor) ----
        last_path = None
        for path in iter_files("C:/", start_after=st.get("cursor")):
            last_path = path
            if chunk_bytes >= CHUNK_TARGET or time.time() > deadline:
                break
            st["visited"] += 1
            try:
                size = __import__("os").path.getsize(path)
            except OSError:
                continue
            if size < 16:
                continue
            h = __import__("hashlib").sha1(
                path.encode("utf-8", errors="replace")).hexdigest()[:12]
            if h in seen:
                continue
            data = sample_bytes(path, size)
            if data is None:
                seen[h] = "locked"
                continue
            seen[h] = len(data)
            st["sampled"] += 1
            chunk.append(data)
            chunk_bytes += len(data)
            st["bytes_ingested"] += len(data)
            files_now += 1
        if last_path:
            st["cursor"] = last_path     # resume here next iteration
        if not chunk:
            print("no more files — disk fully swept", flush=True)
            break
        blob = b"\n\x00FILE\x00\n".join(chunk)
        cid = f"ns_{st['chunks']:04d}"
        (CKPT_DIR / "domains").mkdir(parents=True, exist_ok=True)
        (CKPT_DIR / "domains" / f"{cid}.bin").write_bytes(blob)
        # ---- neural training on cumulative corpus ----
        # WAVE 168 (user decision): replay cap REMOVED — full-book
        # replay on every ingest (no forgetting at all; time cost is
        # accepted by the user)
        # WAVE 185: unified pipeline — neural + symbolic halves.
        # HAGI_SYMBOLIC=1 adds diagnose->invent->gate->reify after
        # the certified LM step; default keeps the replay loop fast.
        import os as _os
        if _os.environ.get("HAGI_SYMBOLIC") == "1":
            from hagi.pipeline import unified_ingest
            r = unified_ingest(u, cid, blob, epochs=2, repeat=1,
                               replay_cap=None)
        else:
            r = u.ingest(cid, blob, epochs=2, repeat=1, replay_cap=None)
            # WAVE 241: re-check accepted laws even without symbolic
            # invention on this chunk.
            try:
                from hagi.law_critique import run_law_self_critique
                r["self_critique"] = run_law_self_critique(u)
            except Exception as _ce:
                print("    [self-critique] skipped: " + repr(_ce),
                      flush=True)
        st["chunks"] += 1
        ce = r.get("holdout_ce", -1)
        # WAVE 238 (user directive): LIVE CHAT during training.
        # Drop a .chat file into checkpoints/inbox/ — the TRAINING
        # PROCESS itself ingests it (single-owner weight writes,
        # no checkpoint races); reply lands as .reply.
        try:
            inbox = CKPT_DIR / "inbox"
            if inbox.exists():
                for chat_file in sorted(inbox.glob("*.chat")):
                    reply = []
                    try:
                        msg = chat_file.read_text(
                            encoding="utf-8", errors="ignore")
                        dom = "chat_" + chat_file.stem
                        if "АВТОСАМОКРИТИКА" in msg:
                            # WAVE 242: critique chats carry hex pairs
                            # (context -> target). Teach the RAW byte
                            # stream, not hex text — and skip the
                            # gemma consolidate (minutes per chat,
                            # no semantic value in hex dumps).
                            import re as _re
                            import binascii
                            raw = bytearray()
                            for m in _re.finditer(
                                    r"^([0-9a-f]{8,})\s*->\s*"
                                    r"([0-9a-f]{2})", msg, _re.M):
                                raw += binascii.unhexlify(m.group(1))
                                raw += bytes([int(m.group(2), 16)])
                            if len(raw) >= 4096:
                                u.ingest(dom, bytes(raw), epochs=1,
                                         repeat=1)
                            else:
                                u.remember(msg)
                            reply.append("learned: raw critique "
                                         "stream " + str(len(raw))
                                         + "B")
                            print("    [chat] " + chat_file.stem
                                  + ": critique stream "
                                  + str(len(raw)) + "B", flush=True)
                            (inbox / (chat_file.stem
                                      + ".reply")).write_text(
                                chr(10).join(reply), encoding="utf-8")
                            chat_file.unlink()
                            continue
                        u.learn_from_dialog(msg, name=dom)
                        dom_ce = u.domains.get(dom, {}).get("ce")
                        reply.append("learned: domain " + dom
                                     + " CE=" + str(dom_ce))
                        ans = u.consolidate(msg[:400])
                        reply.append("answer: " + ans[:600])
                        print("    [chat] " + chat_file.stem
                              + ": learned+replied", flush=True)
                    except Exception as chat_err:
                        reply.append("error: " + repr(chat_err))
                        print("    [chat] " + chat_file.stem
                              + " error: " + repr(chat_err),
                              flush=True)
                    reply_path = inbox / (chat_file.stem + ".reply")
                    reply_path.write_text(chr(10).join(reply),
                                          encoding="utf-8")
                    chat_file.unlink()
        except Exception:
            pass  # inbox optional
        # WAVE 236: law synthesizer verdict for this chunk.
        sym = r.get("symbolic")
        if sym:
            for note in sym.get("notes", []):
                if "multi-byte" in note or "law" in note.lower():
                    print("    " + note, flush=True)
        # WAVE 234 (user directive): semantic distill ALWAYS ON in the
        # full-disk cycle — same protocol as restore_domains (gemma
        # teacher -> LM hidden cosine distill on chunk fragments).
        try:
            from hagi.lm import SemanticDistillHead
            bank = getattr(u, "sem_head", None)
            if bank is None:
                bank = u.sem_head = SemanticDistillHead(dim=0)
            blob_u = blob.decode("utf-8", errors="ignore")
            frags = [blob_u[i:i + 200]
                     for i in range(0, min(len(blob_u), 2000), 200)][:8]
            frags = [f.ljust(200) for f in frags]  # uniform length
            if frags:
                import torch as _th
                dev = getattr(u.model, "device", "cpu")
                idx = _th.tensor(
                    [[(ord(c) % 256) for c in f[:u.model.k]]
                     for f in frags], dtype=_th.long, device=dev)
                if idx.shape[1] < u.model.k:
                    pad = _th.full(
                        (idx.shape[0], u.model.k - idx.shape[1]), 32,
                        dtype=_th.long)
                    idx = _th.cat([pad, idx], dim=1)
                u.model.domain_adapter = None
                sl = bank.align_step(u.model, idx, frags)
                print(f"    semantic loss {sl:.3f}", flush=True)
        except Exception as _e:
            print(f"    semantic skipped: {_e!r}", flush=True)
        print(f"[{cid}] neural: +{chunk_bytes/1e6:.1f}MB "
              f"({files_now} files) → CE {ce} "
              f"({r.get('seconds', '?')}s)", flush=True)
        gr = r.get("growth") or {}
        if gr.get("status") in ("GROW", "ROLLBACK"):
            print(f"    growth: {gr.get('status')} blocks={gr.get('blocks')} "
                  f"signal={gr.get('signal')} CE {gr.get('ce_before')}->{gr.get('ce_after')}", flush=True)
        chunk, chunk_bytes, files_now = [], 0, 0
        # ---- 2+3. DERIVE + WEAVE: fabric self-assembly ----
        model = u.model
        tasks = derive_tasks(model, k=u.k,
                             seed=13 + st['chunks'] * 7)
        # dedup: truth-table hash — same boolean problem = same circuit;
        # honest "reused" instead of wasting search budget
        solved = {}
        netlist_path = CKPT_DIR / "processor_netlist.json"
        if netlist_path.exists():
            for tape in json.loads(netlist_path.read_text(encoding="utf-8")):
                for c in tape["circuits"]:
                    solved[c["name"]] = c
        circuits, reused = [], 0
        seen_tables = {}
        # index: truth-hash → certified circuit from past tape-outs
        solved_tables = {}
        netlist_path = CKPT_DIR / "processor_netlist.json"
        if netlist_path.exists():
            for tape in json.loads(
                    netlist_path.read_text(encoding="utf-8")):
                for c in tape["circuits"]:
                    th = c.get("truth_hash")
                    if th:
                        solved_tables[th] = c
        for i, task in enumerate(tasks):
            thash = __import__("hashlib").sha1(
                json.dumps(sorted(task["truth"].items()),
                           default=str).encode()).hexdigest()[:10]
            if task.get("meta", {}).get("stage", 0) == 1:
                res = compose_adder2()
                if res:
                    res["truth_hash"] = thash
                    res["profile"] = f"{task['name']}_{thash}"
                    circuits.append(res)
                    # ladder advance: stage 1 composed ONCE → stage 2
                    sf = CKPT_DIR / "ladder_stage.json"
                    if json.loads(sf.read_text(encoding="utf-8")).get("stage", 0) == 1:
                        sf.write_text(json.dumps({"stage": 2}),
                                      encoding="utf-8")
                        print(f"  LADDER: stage 1 composed → stage 2 "
                              f"(MUX) unlocked", flush=True)
                    print(f"  fabric [{task['name']}]: COMPOSED from "
                          f"certified cells — CERTIFIED 32/32",
                          flush=True)
                else:
                    print(f"  fabric [{task['name']}]: composition "
                          f"failed (honest: no commit)", flush=True)
                continue
            if thash in seen_tables:
                print(f"  fabric [{task['name']}]: DUPLICATE of "
                      f"[{seen_tables[thash]}] — reused, no re-search",
                      flush=True)
                reused += 1
                continue
            seen_tables[thash] = task["name"]
            if thash in solved_tables:
                # already solved in an earlier tape-out: honest reuse
                prev = solved_tables[thash]
                print(f"  fabric [{task['name']}]: SOLVED earlier "
                      f"[{prev['name']}] — reusing certified circuit",
                      flush=True)
                circuits.append({"profile": f"{task['name']}_{thash}",
                                 "seed": prev.get("seed", 0),
                                 "certified_gen": prev.get("certified_gen"),
                                 "best": None, "bits": prev.get("bits"),
                                 "_reuse_of": prev})
                continue
            res = weave_task(task, seed=5 + i)
            if res:
                res["truth_hash"] = thash
                # ladder: certified stage task → advance complexity
                stage = task.get("meta", {}).get("stage", 0)
                sf = CKPT_DIR / "ladder_stage.json"
                sf.write_text(json.dumps({"stage": stage + 1}),
                              encoding="utf-8")
                print(f"  LADDER: stage {stage} solved → stage "
                      f"{stage+1} unlocked", flush=True)
            if res:
                res["profile"] = f"{task['name']}_{thash}"
                circuits.append(res)
                nb = res["bits"]
                ng = len(res["best"]["gates"])
                print(f"  fabric [{task['name']}]: CERTIFIED "
                      f"{ng} gates, {nb} bits, gen "
                      f"{res['certified_gen']}", flush=True)
            else:
                print(f"  fabric [{task['name']}]: no certified circuit "
                      f"(honest: no commit)", flush=True)
        # ---- 4. COMMIT tape-out ----
        if circuits:
            commit(circuits, cid, ce)
            print(f"  TAPE-OUT: {len(circuits)} certified circuits "
                  f"→ processor_netlist.json", flush=True)
        # ---- persist one-model state ----
        u.save()
        save_state(st)
        it += 1
    print(f"cycle done: iterations={it} chunks={st['chunks']} "
          f"ingested={st['bytes_ingested']/1e6:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
