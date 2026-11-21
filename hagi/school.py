"""School: lesson book, curriculum accounting, mind interface."""
from __future__ import annotations


# ============================================================
# LESSON BOOK (cumulative, full replay)
# ============================================================

import json
from pathlib import Path
from hagi.util import CKPT_DIR as _CKPT
BOOK = _CKPT / 'nadya_lessons.json'
def load_book() -> dict[str, str]:
    if BOOK.exists():
        return json.loads(BOOK.read_text(encoding='utf-8'))
    return {}
def add_lesson(name: str, text: str, repeat: int=10) -> dict[str, str]:
    book = load_book()
    book[name] = ' '.join([text.strip()] * repeat)
    BOOK.parent.mkdir(parents=True, exist_ok=True)
    BOOK.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding='utf-8')
    return book
def corpus() -> bytes:
    return ' '.join(load_book().values()).encode('utf-8')
SCHOOL_CKPT = _CKPT / 'nadya_school.pt'
def teach_school(epochs: int=120, k: int=10, dim: int=96, hidden: int=192) -> dict:
    """Train the dedicated school model FROM SCRATCH on the whole
    book. Fresh training on a small book never forgets: the book is
    the single source of truth, re-learned every time."""
    from hagi import lm as lm_mod
    data = corpus()
    model = lm_mod.ByteLM(k=k, dim=dim, hidden=hidden)
    rep = model.train_guarded(data, epochs=epochs, log_every=0)
    SCHOOL_CKPT.parent.mkdir(parents=True, exist_ok=True)
    lm_mod.save(model, SCHOOL_CKPT)
    return rep
def ask_school(prompt: str, n: int=40, temp: float=0.3) -> str:
    from hagi import lm as lm_mod
    model = lm_mod.load(SCHOOL_CKPT)
    return model.generate(prompt, n_bytes=n, temp=temp).split('\n')[0][len(prompt):].strip()

# ============================================================
# CURRICULUM PHASE ACCOUNTING (C330 port)
# ============================================================

from dataclasses import dataclass, asdict
from typing import Any
PHASES = ('GROW', 'DERIVE', 'OPTIMIZE', 'GATE')
@dataclass
class PhaseRecord:
    phase: str
    delta: float
    d_cert: float
    steps: int = 1

    @property
    def contribution(self) -> float:
        return self.delta / self.d_cert if self.d_cert > 0 else 0.0
def curriculum_total(records: list[PhaseRecord]) -> float:
    """Order-free certificate: sum of per-phase delta/d (theory
    curriculumTotal; commutativity is exact by construction here)."""
    return round(sum((r.contribution for r in records)), 6)
def certificate(state: dict[str, Any]) -> dict[str, Any]:
    """Attach/refresh the curriculum certificate to a self_state dict."""
    recs = [PhaseRecord(**r) for r in state.get('curriculum', [])]
    state['curriculum_certificate'] = {'phases': [asdict(r) for r in recs], 'total': curriculum_total(recs), 'order_free': True, 'theorem': 'curriculum_phase_accounting (CurriculumCert.lean)'}
    return state

# ============================================================
# MIND: hybrid exact/symbolic interface
# ============================================================

import argparse
import re
from typing import Any, Callable
from pathlib import Path
from hagi.io import execute, execute_text
from hagi import lm
SELF_SOURCES = ['generated_programs.py', 'lm.py', 'execute.py', 'ingest.py', 'loop.py', 'rewrite.py']
_ROUTES: dict[str, tuple[str, Callable[..., tuple]]] = {'^(?:чему равен )?gcd (\\d+) (\\d+)$': ('GCD', lambda m: (int(m[1]), int(m[2]))), '^(?:чему равен )?нок (\\d+) (\\d+)$': ('LCM', lambda m: (int(m[1]), int(m[2]))), '^abs (-?\\d+)$': ('ABS', lambda m: (int(m[1]),)), '^square (-?\\d+)$': ('SQUARE', lambda m: (int(m[1]),)), '^cube (-?\\d+)$': ('CUBE', lambda m: (int(m[1]),)), '^mul (-?\\d+) (-?\\d+)$': ('MUL', lambda m: (int(m[1]), int(m[2]))), '^add (-?\\d+) (-?\\d+)$': ('ADD', lambda m: (int(m[1]), int(m[2]))), '^xor (.+) (-?\\d+)$': ('LANG8_XOR_CIPHER', lambda m: (m[1], int(m[2]))), '^caesar (.+)$': ('LANG8_CAESAR1', lambda m: (m[1],))}
def _exact_answer(query: str) -> tuple[str, str] | None:
    q = query.strip().lower()
    from hagi.rewrite import _fn_hash
    for pat, (prog, parse) in _ROUTES.items():
        m = re.match(pat, q)
        if not m:
            continue
        args = parse(m)
        prog_h = _fn_hash(prog)
        if prog in ('LANG8_XOR_CIPHER', 'LANG8_CAESAR1', 'LANG_ATBASH', 'LANG8_SHIFT'):
            text, key = args if len(args) == 2 else (args[0], None)
            out = execute_text(prog_h, text, key=key) if key is not None else execute_text(prog_h, text)
        else:
            out = execute(prog_h, *args)
        return (prog, str(out))
    return None
def ask(query: str) -> str:
    exact = _exact_answer(query)
    if exact:
        prog, out = exact
        return f'[EXACT · {prog}] {out}'
    ckpt = lm.CKPT
    if not ckpt.exists():
        return '[NEURAL · нет чекпойнта] сначала: python -m hagi.mind retrain'
    model = lm.load()
    return '[NEURAL · вероятностная догадка, не сертифицировано] ' + model.generate(query, n_bytes=100, temp=0.6)
def retrain() -> str:
    root = Path(__file__).resolve().parent.parent
    pkg = root / 'hagi'
    from hagi.util import CKPT_DIR
    base = CKPT_DIR / 'self'
    files = [base / 'generated_programs.py'] + [root / f for f in SELF_SOURCES[1:]] + [root / 'README.md', root / 'THEORY_ALIGNMENT.md']
    files = [f for f in files if f.exists()]
    data = b''.join((f.read_bytes() for f in files))
    model = lm.ByteLM(k=10, dim=96, hidden=192)
    rep = model.train_guarded(data, epochs=90, log_every=30)
    lm.CKPT.parent.mkdir(parents=True, exist_ok=True)
    lm.save(model)
    return f"LM перетренирована на {len(data)} байтах собственных артефактов ({len(files)} файлов): holdout CE {rep['best_holdout_ce']}"
def main() -> None:
    p = argparse.ArgumentParser(description='HAGI hybrid mind')
    sub = p.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('ask')
    a1.add_argument('query')
    sub.add_parser('retrain')
    a = p.parse_args()
    print(retrain() if a.cmd == 'retrain' else ask(a.query))
if __name__ == '__main__':
    main()