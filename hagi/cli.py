"""Unified CLI (WAVE 190): one entry point for every HAGI tool.

    python -m hagi.cli <command> [args...]

Commands map to the module CLIs (thin dispatch, no logic here):
  core       self-test / grow / benchmark (canonical kernel)
  lm         train / generate (neural body)
  ingest     learn from any data file
  execute    run a written program
  loop       self-development loop
  rewrite    self-rewrite loop
  genesis    architecture genesis
  certificates   portable certificate suite
"""
from __future__ import annotations

import sys

COMMANDS = {
    "core": ("hagi.core", "_cli"),
    "lm": ("hagi.lm", "main"),
    "ingest": ("hagi.ingest", "main"),
    "execute": ("hagi.execute", "main"),
    "loop": ("hagi.loop", "main"),
    "rewrite": ("hagi.rewrite", "main"),
    "genesis": ("hagi.genesis", "main"),
    "certificates": ("hagi.certificates", "main"),
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("commands:", ", ".join(sorted(COMMANDS)))
        return
    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print("commands:", ", ".join(sorted(COMMANDS)), file=sys.stderr)
        raise SystemExit(2)
    mod_name, fn_name = COMMANDS[cmd]
    module = __import__(mod_name, fromlist=[fn_name])
    fn = getattr(module, fn_name)
    sys.argv = [f"hagi {cmd}"] + argv[1:]
    fn()


if __name__ == "__main__":
    main()
