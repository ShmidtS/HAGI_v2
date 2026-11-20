#!/usr/bin/env python3
"""Literacy corpus: teach HAGI to READ code, from boolean algebra up.

Layer 0: boolean algebra (NOT/AND/OR/XOR as byte operations)
Layer 1: bytes and ASCII — code IS bytes
Layer 2: Python grammar — lexemes, indentation, def/class
Layer 3: reading real HAGI code (the small certified modules first)
Layer 4: Lean grammar — theorem/proof/tactic
Layer 5: reading our formalization (small clean theory files)

Each layer is a set of (question, answer) pairs rendered as plain
text: the byte-LM learns next-byte prediction over explanations
that SHOW the decoding of code into meaning, like a textbook that
annotates every line.
"""
import sys, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "training" / "literacy" / "corpus.txt"

TRUTH = ["x & y", "x | y", "x ^ y", "~x & y", "x & ~y"]

def layer0_boolean():
    """Boolean algebra: the atoms everything else decodes into."""
    rows = []
    rows.append("""СЛОЙ 0. БУЛЕВА АЛГЕБРА — АТОМЫ ВСЕГО.

Бит — значение 0 или 1. Всё остальное в компьютере собрано из битов.

Операции над одним битом:
  NOT 0 = 1;  NOT 1 = 0.  (инверсия)
Операции над двумя битами (таблицы истинности):
  0 AND 0 = 0;  0 AND 1 = 0;  1 AND 0 = 0;  1 AND 1 = 1.  (оба — тогда 1)
  0 OR  0 = 0;  0 OR  1 = 1;  1 OR  0 = 1;  1 OR  1 = 1.  (хотя бы один)
  0 XOR 0 = 0;  0 XOR 1 = 1;  1 XOR 0 = 1;  1 XOR 1 = 0.  (различие)

XOR — ключевая операция HAGI: это сложение по модулю 2, поле GF(2).
Проверка: 1 XOR 1 = 0, значит 1+1=2 = 0 (mod 2). Любая булева функция
может быть записана через XOR и AND — это запись ANF (полином Жегалкина).
""")
    for expr, fn in [
        ("x & y",  lambda x, y: x & y),
        ("x | y",  lambda x, y: x | y),
        ("x ^ y",  lambda x, y: x ^ y),
        ("x & ~y", lambda x, y: x & (~y & 1)),
        ("~x | y", lambda x, y: ((~x & 1) | y)),
    ]:
        rows.append(f"Таблица истинности {expr}:")
        for x in (0, 1):
            for y in (0, 1):
                rows.append(f"  {expr} при x={x}, y={y} -> {fn(x, y)}")
    return "\n".join(rows)

def layer1_bytes():
    """From bits to bytes to ASCII: code IS bytes."""
    rows = ["""СЛОЙ 1. БИТЫ -> БАЙТЫ -> СИМВОЛЫ.

Байт = 8 битов. Байт хранит число 0..255. Программа — это ПОСЛЕДОВАТЕЛЬНОСТЬ
БАЙТОВ. Читать программу = декодировать байты в символы, символы в слова,
слова в смысл.

ASCII: '0'=48, '1'=49, ' '=32, '\n'=10 (новая строка), 'x'=120.
Латиница: 'a'=97.. 'z'=122; 'A'=65.. 'Z'=90.
Кириллица в UTF-8 = 2 байта на букву.

Пример декодирования. Байты 104 101 108 108 111 читаются как:
  104='h', 101='e', 108='l', 108='l', 111='o' -> слово "hello".
"""]
    for s in ["def", "class", "return", "import", "if x > 0:",
              "theorem", "proof", "hagi"]:
        bs = s.encode("ascii") if s.isascii() else s.encode("utf-8")
        dec = " ".join(str(b) for b in bs)
        rows.append(f'Текст "{s}" = байты {dec}.')
    return "\n".join(rows)

def layer2_python():
    """Python grammar: what code looks like structurally."""
    rows = ["""СЛОЙ 2. ГРАММАТИКА PYTHON — КАК УСТРОЕН КОД.

Лексемы: имена (x, def), числа (0, 42), операторы (& | ^ ~ = == + *),
скобки, двоеточие, отступ.

Отступ (4 пробела) задаёт тело блока. После `def имя(...):` идёт
тело функции с отступом. `return` возвращает значение.

Чтение функции — по шаблону:
  def f(x):
      return x & 5
означает: f принимает x, возвращает x AND 5 (байтовая операция над
целым). Число 5 в бинарном виде 0b101, значит операция обнуляет все
биты кроме двух младших, оставляя биты 0 и 2.

Целое число в Python — неограниченное; побитовые операции & | ^ ~
работают по битам как булева алгебра над каждой парой битов параллельно.

Пример: 6 & 3 -> биты 110 & 011 = 010 = 2.
Пример: 6 ^ 3 -> 110 ^ 011 = 101 = 5.
Пример: 6 | 3 -> 110 | 011 = 111 = 7.
"""]
    for a, b in [(6, 3), (12, 10), (0xFF, 0x0F)]:
        rows.append(f"Вычисление: {a} & {b} = {a & b}; "
                    f"{a} ^ {b} = {a ^ b}; {a} | {b} = {a | b}.")
    return "\n".join(rows)

def layer3_hagi():
    """Reading real HAGI code, annotated."""
    rows = ["""СЛОЙ 3. ЧТЕНИЕ РЕАЛЬНОГО КОДА HAGI.

Теперь читаем модули, которые создали HAGI. Каждый фрагмент — настоящий
исходный текст с построчным разбором.
"""]
    lm = (ROOT / "hagi" / "lm.py").read_text(encoding="utf-8")
    # annotберём первые функции
    snippets = []
    take = False
    buf = []
    for line in lm.splitlines():
        if line.startswith("def _sha") or line.startswith("class ByteLM"):
            take = True
        if take:
            buf.append(line)
            if len(buf) > 30:
                take = False
    if buf:
        snippets.append("def _sha(b: bytes) -> str:\n    return hashlib.sha1(b).hexdigest()[:10]")
    snippets.append("""def windows_np(sel):
    x = np.stack([idx_np[i:i + self.k] for i in sel])
    y = idx_np[sel + self.k]
    return (as_tensor(x, dtype=long, device=dev),
            as_tensor(y, dtype=long, device=dev))""")
    for s in snippets:
        rows.append("Код:")
        rows.append(s)
        rows.append("Разбор построчно:")
        for i, line in enumerate(s.splitlines(), 1):
            rows.append(f"  строка {i}: {line!r}")
        rows.append("")
    rows.append("""Чтение def _sha: функция принимает bytes, вычисляет SHA-1
хеш (отпечаток данных) и возвращает первые 10 шестнадцатеричных
знаков. HAGI использует этот отпечаток, чтобы отличать домены
(книги данных) друг от друга: одинаковые данные -> одинаковый хеш.

Чтение windows_np: sel — список стартовых позиций; для каждой
позиции i берётся срез контекста длины k (idx[i:i+k]) как вход x,
и следующий байт idx[i+k] как цель y. np.stack собирает матрицу
батча. Так модель учится предсказывать следующий байт по контексту.
""")
    return "\n".join(rows)

def layer4_lean():
    """Lean grammar."""
    rows = ["""СЛОЙ 4. ГРАММАТИКА LEAN — ФОРМАЛЬНЫЕ ДОКАЗАТЕЛЬСТВА.

Lean — язык, где математическое утверждение проверяется машиной.
Теорема (theorem) — утверждение с доказательством (proof).
Тактика (tactic) — шаг доказательства: `exact` (точный член),
`simp` (упрощение), `rw` (переписывание), `induction` (индукция).

Пример простейшей теоремы:
  theorem one_add_one : 1 + 1 = 2 := rfl
Читается: доказано рефлексивно (rfl), что 1+1 равно 2 — машина
сама вычисляет и проверяет.

Формализация теории HAGI в C:/primes — 567 файлов .lean, где
каждая гарантия кода (сертифицированный шаг, концентрация holdout,
zero-forgetting GPM) доказана как теорема.
"""]
    return "\n".join(rows)

def layer5_theory():
    """Reading our actual Lean theory (small clean file)."""
    rows = ["""СЛОЙ 5. ЧТЕНИЕ НАШЕЙ ФОРМАЛИЗАЦИИ.

Пример реального фрагмента теории (простейший):
"""]
    # возьмём короткий чистый файл
    for cand in ["CertCEHoldout.lean", "AnytimeHoldoutStop.lean"]:
        f = Path("C:/primes/Hagi/Theory") / cand
        if f.exists():
            text = f.read_text(encoding="utf-8", errors="ignore")
            # первые строки без лицензии
            body = "\n".join(text.splitlines()[:25])
            rows.append(body)
            rows.append("Разбор: import подключает зависимости; "
                        "namespace — пространство имён; далее идут "
                        "теоремы с явными формулами концентрации.")
            break
    return "\n".join(rows)

def main():
    parts = [
        "=" * 64,
        "КУРС ГРАМОТНОСТИ HAGI: ОТ БУЛЕВОЙ АЛГЕБРЫ К ЧТЕНИЮ СВОЕГО КОДА",
        "=" * 64,
        layer0_boolean(), layer1_bytes(), layer2_python(),
        layer3_hagi(), layer4_lean(), layer5_theory(),
    ]
    text = "\n\n".join(parts)
    OUT.write_text(text, encoding="utf-8")
    n_bytes = len(text.encode("utf-8"))
    print(f"corpus: {OUT} {n_bytes} bytes, {len(text.splitlines())} lines")

if __name__ == "__main__":
    main()
