"""Nadya's lesson book — cumulative curriculum.

Each lesson is a short repeated text. The book only GROWS; every
teach() run fine-tunes on the WHOLE book so new knowledge never
erases old (anti catastrophic forgetting via full replay).
"""
from __future__ import annotations

import json
from pathlib import Path

BOOK = Path("checkpoints/nadya_lessons.json")


def load_book() -> dict[str, str]:
    if BOOK.exists():
        return json.loads(BOOK.read_text(encoding="utf-8"))
    return {}


def add_lesson(name: str, text: str, repeat: int = 10) -> dict[str, str]:
    book = load_book()
    book[name] = " ".join([text.strip()] * repeat)
    BOOK.parent.mkdir(parents=True, exist_ok=True)
    BOOK.write_text(json.dumps(book, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return book


def corpus() -> bytes:
    return " ".join(load_book().values()).encode("utf-8")


SCHOOL_CKPT = Path("checkpoints/nadya_school.pt")


def teach_school(epochs: int = 120, k: int = 10, dim: int = 96,
                 hidden: int = 192) -> dict:
    """Train the dedicated school model FROM SCRATCH on the whole
    book. Fresh training on a small book never forgets: the book is
    the single source of truth, re-learned every time."""
    import hagi_lm
    data = corpus()
    lm = hagi_lm.ByteLM(k=k, dim=dim, hidden=hidden)
    rep = lm.train_guarded(data, epochs=epochs, log_every=0)
    SCHOOL_CKPT.parent.mkdir(parents=True, exist_ok=True)
    hagi_lm.save(lm, SCHOOL_CKPT)
    return rep


def ask_school(prompt: str, n: int = 40, temp: float = 0.3) -> str:
    import hagi_lm
    lm = hagi_lm.load(SCHOOL_CKPT)
    return lm.generate(prompt, n_bytes=n, temp=temp).split("\n")[0][len(prompt):].strip()
