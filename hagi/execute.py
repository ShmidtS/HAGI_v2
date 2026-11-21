#!/usr/bin/env python3
"""HAGI program executor — the model RUNS the programs it wrote.

Loads the model's own generated_programs.py (standalone functions,
validated for exact equality against certified experts) and executes
them:

  - execute(name, *args)          — call one program
  - execute_text(name, text, ...) — stream a UTF-8 string through a
    byte-level program (any unicode text: cyrillic, emoji, CJK) and
    return the decoded result; binary key programs take a key
  - compose(f, g, *args)          — run g, feed into f (DAG chaining
    by hand; the library itself composes via `call` nodes)

CLI:
  python -m hagi.execute --list
  python -m hagi.execute --task LANG8_SHIFT --text "привет" --key 3
  python -m hagi.execute --task LANG8_XOR_CIPHER --text "мир" --key 42
  python -m hagi.execute --task LANG8_SHIFT --text <shifted> --key -3   (decrypt)
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from hagi.util import CKPT_DIR as _CKPT
DEFAULT_PROGRAMS = _CKPT / "self" / "generated_programs.py"

_mod = None


def _load(path: Path | None = None):
    global _mod
    if _mod is not None:
        return _mod
    p = path or DEFAULT_PROGRAMS
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run hagi_selfrewrite first so the model "
            "writes its programs")
    spec = importlib.util.spec_from_file_location("hagi_generated", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _mod = mod
    return mod


def available() -> list[str]:
    return [n for n in dir(_load()) if not n.startswith("_") and n != "hagi_generated"]


def execute(name: str, *args: int) -> int:
    fn = getattr(_load(), name, None)
    if fn is None:
        raise KeyError(f"no program named {name!r} (model has {len(available())})")
    return int(fn(*args))


def execute_text(name: str, text: str, key: int | None = None) -> str:
    """Stream UTF-8 text byte-by-byte through a byte-level program."""
    data = text.encode("utf-8")
    arity = 2 if key is not None else 1
    out = bytearray()
    for b in data:
        out.append(execute(name, b, int(key or 0)) & 0xFF if arity == 2
                   else execute(name, b) & 0xFF)
    # decode back, lossy so any byte stream is viewable
    return out.decode("utf-8", errors="replace")


def main() -> None:
    p = argparse.ArgumentParser(description="Execute programs the HAGI model wrote")
    p.add_argument("--programs", type=Path, default=DEFAULT_PROGRAMS)
    p.add_argument("--list", action="store_true")
    p.add_argument("--task")
    p.add_argument("--text")
    p.add_argument("--key", type=int, default=None)
    p.add_argument("values", nargs="*", type=int,
                   help="direct integer arguments, e.g. --task GCD 12 18")
    a = p.parse_args()
    if a.list:
        names = available()
        print(f"{len(names)} programs:")
        for n in sorted(names):
            print(" ", n)
        return
    if not a.task:
        p.error("--task or --list required")
    if a.text is not None:
        print(execute_text(a.task, a.text, a.key))
    else:
        print(execute(a.task, *a.values))


if __name__ == "__main__":
    main()
