#!/usr/bin/env python3
"""Корпус «грамотности»: научить модель ЧИТАТЬ код — от булевой алгебры вверх.

    Каждый проверяемый факт корпуса ВЫВОДИТСЯ ИЗ РЕАЛЬНОГО ВЫЧИСЛЕНИЯ — никаких
    написанных вручную таблиц истинности и шестнадцатеричных дампов. Если строка
    говорит «6 & 3 = 2», она напечатана как print(f"{a} & {b} = {a & b}").
    Ручной текст ограничен заголовками разделов и постановкой задач.

    Слои:
      0  булева алгебра: таблицы истинности над 4-битными словами (генерируются)
      1  байты <-> ASCII <-> hex (из chr, ord и encode)
      2  поразрядные операции Python над числами (генерируются)
      3  код -> поток байтов (из encode())
      4  следы исполнения маленьких функций (получены их запуском)
      5  сигнатуры реальных модулей hagi (из ast)
      6  токены формального языка как байты (из encode())"""
import ast  # разбор исходников Python
import random  # генератор случайных чисел
from pathlib import Path  # пути

ROOT = Path(__file__).resolve().parents[2]  # корень проекта — на два уровня выше файла
OUT_DIR = ROOT / "training" / "literacy"  # каталог корпуса
OUT = OUT_DIR / "bits_corpus.txt"  # файл корпуса


def layer0_boolean(rows):
    """Слой 0: NOT, AND, XOR, OR над 4-битными словами и проверки однобитовых тождеств.

        rows: список строк, в который дописываются строки слоя."""
    rows.append("L0 BOOLEAN ALGEBRA (generated)")  # строка-заголовок слоя
    # NOT / AND / OR / XOR над 4-битными словами — печатаются из реальных операций
    for x in range(16):  # по всем 4-битным словам x = 0..15
        rows.append(f"NOT {x:04b} = {~x & 0xF:04b}")  # строка: NOT x = (~x) & 0xF — инверсия 4 битов
    for x in range(16):  # по всем 4-битным словам x = 0..15
        for y in range(16):  # по всем 4-битным словам y = 0..15
            rows.append(f"{x:04b} AND {y:04b} = {x & y:04b}")  # строка: x AND y = x & y
            rows.append(f"{x:04b} XOR {y:04b} = {x ^ y:04b}")  # строка: x XOR y = x ^ y
            rows.append(f"{x:04b} OR  {y:04b} = {x | y:04b}")  # строка: x OR y = x | y
    # однобитовые тождества, напечатанные из проверок
    facts = [  # список (название тождества, результат проверки)
        ("XOR self-inverse", all((a ^ a) == 0 for a in range(2))),  # тождество a ^ a = 0
        ("XOR commutes", all((a ^ b) == (b ^ a) for a in range(2) for b in range(2))),  # тождество a ^ b = b ^ a
        ("AND absorbs", all((a & 0) == 0 for a in range(2))),  # тождество a & 0 = 0
        ("OR identity", all((a | 0) == a for a in range(2))),  # тождество a | 0 = a
        ("DeMorgan NOT(a AND b) = NOT a OR NOT b",  # название тождества (продолжение кортежа)
         all((not (a and b)) == ((not a) or (not b))  # закон де Моргана: not (a and b) = (not a) or (not b)
             for a in (0, 1) for b in (0, 1))),  # по a, b из {0, 1}
    ]  # конец списка
    for name, ok in facts:  # по парам (название, результат)
        rows.append(f"CHECK {name} -> {'TRUE' if ok else 'FAIL'}")  # строка «CHECK имя -> TRUE/FALSE»: TRUE, если ok истинно, иначе FAIL


def layer1_bytes(rows):
    """Слой 1: печатный ASCII из chr (номер, hex, двоичный вид, символ) и слова из encode."""
    rows.append("L1 BYTES ASCII HEX (generated)")  # строка-заголовок слоя
    # печатный ASCII — прямо из ord()
    for b in range(0x20, 0x7F):  # по печатным кодам ASCII 0x20..0x7E
        c = chr(b)  # символ с кодом b
        rows.append(f"CHR {b:3d} HEX {b:02x} BIN {b:08b} = '{c}'")  # строка «CHR b HEX xx BIN bbbb = 'c'»: b в десятичном (3 знака), hex и двоичном (8 разрядов)
    # закодированные слова, напечатанные из encode()
    for w in ["def", "class", "return", "import", "if", "for",  # по словам-примерам w (продолжение списка ниже)
              "theorem", "proof", "rfl", "hagi", "xor", "and", "or"]:  # слова w, байты которых печатаются в hex
        bs = w.encode("ascii")  # bs = w.encode('ascii') — байты слова w
        rows.append(f"WORD '{w}' = " + " ".join(f"{b:02x}" for b in bs))  # строка «WORD 'w' = xx xx ..»: байты слова в hex через пробел


def layer2_python_ops(rows):
    """Слой 2: поразрядные операции Python над числами 0..63 с шагом 3 и небольшие функции от них."""
    rows.append("L2 PYTHON BITWISE OPS (generated)")  # строка-заголовок слоя
    for a in range(0, 64, 3):  # по a = 0, 3, 6, …, 63
        for b in (1, 3, 5, 7, 10, 12):  # по вторым операндам b
            rows.append(f"{a} AND {b} = {a & b}")  # строка «a AND b = a & b»
            rows.append(f"{a} XOR {b} = {a ^ b}")  # строка «a XOR b = a ^ b»
            rows.append(f"{a} OR {b} = {a | b}")  # строка «a OR b = a | b»
        rows.append(f"{a} SHL1 = {(a << 1) & 0xFF}")  # строка «a SHL1 = (a << 1) & 0xFF» — сдвиг влево на 1, обрезанный до 8 битов
        rows.append(f"{a} SHR1 = {a >> 1}")  # строка «a SHR1 = a >> 1» — сдвиг вправо на 1
        rows.append(f"{a} MOD7 = {a % 7}")  # строка «a MOD7 = a mod 7»
        rows.append(f"{a} POPCOUNT = {bin(a).count('1')}")  # строка «a POPCOUNT = число единичных битов a»


def layer3_code_bytes(rows):
    """Слой 3: строки кода и их байты в hex (из encode())."""
    rows.append("L3 CODE AS BYTES (generated from encode)")  # строка-заголовок слоя
    snippets = [  # набор строк кода для слоя
        "def f(x):",  # строка кода-примера (данные)
        "    return x & 5",  # строка кода-примера (данные)
        "    return x ^ 3",  # строка кода-примера (данные)
        "def g(a, b):",  # строка кода-примера (данные)
        "    return (a ^ b) & 0xFF",  # строка кода-примера (данные)
        "class H:",  # строка кода-примера (данные)
        "    def __init__(self):",  # строка кода-примера (данные)
        "        self.k = 16",  # строка кода-примера (данные)
        "for i in range(8):",  # строка кода-примера (данные)
        "    r = r ^ i",  # строка кода-примера (данные)
        "if x > 0:",  # строка кода-примера (данные)
        "    return x",  # строка кода-примера (данные)
    ]
    for s in snippets:  # по строкам кода
        rows.append("CODE " + s)  # строка «CODE s» — сама строка кода
        rows.append("HEX " + " ".join(f"{b:02x}" for b in s.encode("ascii")))  # строка «HEX xx xx ..»: байты s в hex через пробел


def layer4_execution(rows):
    """Слой 4: следы исполнения маленьких функций — каждая строка получена реальным вызовом."""
    rows.append("L4 EXECUTION TRACES (generated by running the fns)")  # строка-заголовок слоя
    # функция f: x & 5
    f = lambda x: x & 5  # функция f(x) = x & 5
    rows.append("FN f(x) = x AND 5")  # строка-описание функции f(x) = x AND 5
    for x in range(0, 32, 3):  # по x = 0, 3, 6, …, 30
        rows.append(f"CALL f({x}) -> {f(x)}")  # строка «CALL f(x) -> f(x)» — вызов с результатом
    # функция g: (a ^ b) & 0xFF
    g = lambda a, b: (a ^ b) & 0xFF  # функция g(a, b) = (a ^ b) & 0xFF
    rows.append("FN g(a, b) = (a XOR b) AND 255")  # строка-описание функции g(a, b) = (a XOR b) AND 255
    for a in range(0, 32, 5):  # по a = 0, 5, 10, …, 30
        for b in (1, 4, 9):  # по b из 1, 4, 9
            rows.append(f"CALL g({a}, {b}) -> {g(a, b)}")  # строка «CALL g(a, b) -> g(a, b)» — вызов с результатом
    # след цикла: r = 0, r ^= i для i из 0..7 — по шагам
    rows.append("FN loop8: r=0; for i in 0..7: r ^= i")  # строка-описание цикла loop8
    r = 0  # аккумулятор r = 0
    for i in range(8):  # по i из 0..7
        r ^= i  # присвоить r := r XOR i
        rows.append(f"  i={i} r={r}")  # строка шага цикла: i и текущее r
    # чётность через вес (число единичных битов)
    rows.append("FN parity(x) = popcount(x) mod 2")  # строка-описание функции parity(x)
    for x in range(0, 16):  # по x из 0..15
        rows.append(f"CALL parity({x}) -> {bin(x).count('1') % 2}")  # строка «CALL parity(x) -> popcount(x) mod 2»


def layer5_hagi_signatures(rows):
    """Слой 5: сигнатуры классов и функций реальных модулей hagi (из синтаксического дерева)."""
    rows.append("L5 REAL HAGI MODULE SIGNATURES (generated from ast)")  # строка-заголовок слоя
    mods = ["cegis.py", "representations.py", "diagnosis.py",  # список имён модулей hagi (продолжение ниже)
            "unified.py", "reifier.py", "invention.py", "lm.py",  # имена модулей (продолжение списка)
            "core.py"]  # последнее имя модуля — конец списка
    for m in mods:  # по именам модулей
        p = ROOT / "hagi" / m  # путь к файлу модуля
        if not p.exists():  # модуль отсутствует на диске
            continue  # пропустить его
        tree = ast.parse(p.read_text(encoding="utf-8"))  # разобрать файл в синтаксическое дерево
        defs = []  # список описаний определений
        for node in ast.walk(tree):  # по всем узлам дерева
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):  # узел — функция или класс
                args = ""  # аргументы пока пусты
                if isinstance(node, ast.FunctionDef):  # для функции — разобрать аргументы
                    try:  # попытка
                        args = ast.unparse(node.args)  # текст аргументов функции
                    except Exception:  # при ошибке разбора
                        args = "..."  # аргументы заменить на «...»
                defs.append(f"{'class' if isinstance(node, ast.ClassDef) else 'def'} "  # описание: «class» или «def», имя и аргументы функции (у класса — пусто)
                            f"{node.name}({args if isinstance(node, ast.FunctionDef) else ''})")  # (продолжение строки описания)
        rows.append(f"MODULE hagi/{m}: {len(defs)} defs")  # добавить описание в список
        for d in defs[:40]:  # первые 40 определений модуля
            rows.append("  " + d)  # строка с отступом: описание определения d


def layer6_lean_tokens(rows):
    """Слой 6: токены формального языка как байты в hex (из encode()) и список файлов теории, если он есть."""
    rows.append("L6 LEAN TOKENS AS BYTES (generated from encode)")  # строка-заголовок слоя
    for tok in ["theorem", "proof", "rfl", "exact", "simp", "rw",  # по токенам (продолжение списка ниже)
                "induction", ":=", "->", "∀", "λ", "Nat", "Bool"]:  # список токенов
        tb = tok.encode("utf-8")  # tb = байты токена в utf-8
        rows.append("TOK " + " ".join(f"{b:02x}" for b in tb) + " = " + tok)  # строка «TOK xx xx .. = токен»: байты в hex, затем сам токен
    # имена файлов формальной теории (если каталог существует)
    theory = Path("C:/primes/Hagi/Theory")  # каталог теории формального проекта
    if theory.exists():  # каталог теории существует
        names = sorted(p.stem for p in theory.glob("*.lean"))[:30]  # имена файлов теории (без расширения) по алфавиту, первые 30
        rows.append(f"THEORY FILES: {len(names)} shown")  # строка с числом показанных файлов
        for n in names:  # по именам файлов
            rows.append("  " + n)  # строка с отступом: имя файла теории


def main():
    """Собрать все слои в файл bits_corpus.txt и самопроверить строки вида «A OP B = C»."""
    rows = ["HAGI LITERACY CORPUS — ALL FACTS MACHINE-GENERATED"]  # первая строка списка — заголовок корпуса
    layer0_boolean(rows)  # слой 0 — булева алгебра
    layer1_bytes(rows)  # слой 1 — байты, ASCII, hex
    layer2_python_ops(rows)  # слой 2 — поразрядные операции Python
    layer3_code_bytes(rows)  # слой 3 — код как байты
    layer4_execution(rows)  # слой 4 — следы исполнения
    layer5_hagi_signatures(rows)  # слой 5 — сигнатуры модулей hagi
    layer6_lean_tokens(rows)  # слой 6 — токены формального языка
    rng = random.Random(7)  # генератор с зерном 7 (не используется: перемешивания нет)
  # печать пути, числа байтов и числа строк
  # самопроверка: каждая строка «A OP B = C» обязана быть верной
    text = "\n".join(rows) + "\n"  # text = строки через \n и завершающий \n
    OUT.write_bytes(text.encode("utf-8"))  # записать text в кодировке utf-8 в файл OUT
    print(f"{OUT}: {len(text.encode('utf-8'))} bytes, {len(rows)} rows")  # печать пути, числа байтов и числа строк
    # самопроверка: каждая строка 'A OP B = C' обязана быть верной
    bad = 0  # число неверных строк
    for line in rows:  # по всем строкам корпуса
        parts = line.split()  # разбить строку на слова
        if len(parts) == 5 and parts[3] == "=" and parts[1] in ("AND", "XOR", "OR"):  # формат «A OP B = C» с двоичными числами: 5 слов, четвёртое — «=»
            a, op, b, _, c = parts  # разобрать a, операцию, b и результат c
            try:  # попытка (не числа — строка пропускается)
                av, bv, cv = int(a, 2), int(b, 2), int(c, 2)  # числа читаются как двоичные
                want = {"AND": av & bv, "XOR": av ^ bv, "OR": av | bv}[op]  # ожидаемый результат: a AND b, a XOR b или a OR b
                if want != cv:  # результат не совпал
                    print("WRONG:", line)  # напечатать неверную строку
                    bad += 1  # число неверных строк + 1
            except ValueError:  # строка не числовая — пропустить
                pass  # ничего не делать
        elif len(parts) == 5 and parts[1] in ("AND", "XOR", "OR") and parts[3] == "=":  # формат «A OP B = C» с десятичными числами: 5 слов, четвёртое — «=»
            a, op, b, _, c = parts  # разобрать a, операцию, b и результат c
            try:  # попытка (не числа — строка пропускается)
                av, bv, cv = int(a), int(b), int(c)  # числа читаются как десятичные
                want = {"AND": av & bv, "XOR": av ^ bv, "OR": av | bv}[op]  # ожидаемый результат: a AND b, a XOR b или a OR b
                if want != cv:  # результат не совпал
                    print("WRONG:", line)  # напечатать неверную строку
                    bad += 1  # число неверных строк + 1
            except ValueError:  # строка не числовая — пропустить
                pass  # ничего не делать
    print(f"self-verify: {'ALL TRUE' if bad == 0 else f'{bad} WRONG LINES'}")  # печать итога самопроверки: все верны либо число неверных
  # печать итога самопроверки: все верны либо число неверных

if __name__ == "__main__":  # истинно, когда файл запущен как сценарий
    main()  # вызов основной функции при запуске файла как сценария
