#!/usr/bin/env python3
"""HAGI mind — hybrid interface over both organs (theory §17 Venn head).

ask(query) routes between the two organs, honestly labelled:

  EXACT   the certified symbolic layer answers: query matches a
          known program invocation (e.g. "gcd 1071 462",
          "xor привет 42") -> exact, verified answer;
  NEURAL  no exact program matches -> the byte LM answers
          probabilistically (clearly marked as a guess, never
          certified).

The LM trains on the model's OWN artifacts (its written programs,
docs, state) — self-referential neural growth. Guarded training
(§3/§9) means the deployed LM is always the best holdout snapshot.

CLI:
  python hagi_mind.py ask "gcd 12 18"
  python hagi_mind.py ask "xor hello 42"
  python hagi_mind.py ask "расскажи о себе"
  python hagi_mind.py retrain        # LM on own artifacts
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from hagi import hagi_execute as hx  # noqa: E402
from hagi import hagi_lm  # noqa: E402

SELF_SOURCES = ["generated_programs.py", "hagi_lm.py", "hagi_execute.py",
                "hagi_ingest.py", "hagi_selfloop.py", "hagi_selfrewrite.py"]

# exact-route patterns: name -> (program, arg parser)
_ROUTES: dict[str, tuple[str, callable]] = {
    r"^(?:чему равен )?gcd (\d+) (\d+)$": ("GCD",
        lambda m: (int(m[1]), int(m[2]))),
    r"^(?:чему равен )?нок (\d+) (\d+)$": ("LCM",
        lambda m: (int(m[1]), int(m[2]))),
    r"^abs (-?\d+)$": ("ABS", lambda m: (int(m[1]),)),
    r"^square (-?\d+)$": ("SQUARE", lambda m: (int(m[1]),)),
    r"^cube (-?\d+)$": ("CUBE", lambda m: (int(m[1]),)),
    r"^mul (-?\d+) (-?\d+)$": ("MUL", lambda m: (int(m[1]), int(m[2]))),
    r"^add (-?\d+) (-?\d+)$": ("ADD", lambda m: (int(m[1]), int(m[2]))),
    r"^xor (.+) (-?\d+)$": ("LANG8_XOR_CIPHER",
        lambda m: (m[1], int(m[2]))),
    r"^caesar (.+)$": ("LANG8_CAESAR1", lambda m: (m[1],)),
}


def _exact_answer(query: str) -> tuple[str, str] | None:
    q = query.strip().lower()
    from hagi.hagi_selfrewrite import _fn_hash
    for pat, (prog, parse) in _ROUTES.items():
        m = re.match(pat, q)
        if not m:
            continue
        args = parse(m)
        prog_h = _fn_hash(prog)   # programs are exported under hash names
        if prog in ("LANG8_XOR_CIPHER", "LANG8_CAESAR1", "LANG_ATBASH",
                    "LANG8_SHIFT"):
            text, key = args if len(args) == 2 else (args[0], None)
            out = hx.execute_text(prog_h, text, key=key) if key is not None \
                else hx.execute_text(prog_h, text)
        else:
            out = hx.execute(prog_h, *args)
        return prog, str(out)
    return None


def ask(query: str) -> str:
    exact = _exact_answer(query)
    if exact:
        prog, out = exact
        return f"[EXACT · {prog}] {out}"
    ckpt = hagi_lm.CKPT
    if not ckpt.exists():
        return ("[NEURAL · нет чекпойнта] сначала: "
                "python hagi_mind.py retrain")
    lm = hagi_lm.load()
    return "[NEURAL · вероятностная догадка, не сертифицировано] " + \
        lm.generate(query, n_bytes=100, temp=0.6)


def retrain() -> str:
    root = Path(__file__).resolve().parent.parent   # repo root
    base = root / "checkpoints" / "self"
    files = [base / "generated_programs.py"] + \
            [root / f for f in SELF_SOURCES[1:]] + \
            [root / "README.md", root / "THEORY_ALIGNMENT.md"]
    files = [f for f in files if f.exists()]
    data = b"".join(f.read_bytes() for f in files)
    lm = hagi_lm.ByteLM(k=10, dim=96, hidden=192)
    rep = lm.train_guarded(data, epochs=90, log_every=30)
    hagi_lm.CKPT.parent.mkdir(parents=True, exist_ok=True)
    hagi_lm.save(lm)
    return (f"LM перетренирована на {len(data)} байтах собственных "
            f"артефактов ({len(files)} файлов): holdout CE "
            f"{rep['best_holdout_ce']}")


def main() -> None:
    p = argparse.ArgumentParser(description="HAGI hybrid mind")
    sub = p.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("ask"); a1.add_argument("query")
    sub.add_parser("retrain")
    a = p.parse_args()
    print(retrain() if a.cmd == "retrain" else ask(a.query))


if __name__ == "__main__":
    main()
