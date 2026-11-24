#!/usr/bin/env python3
"""Аудитор комментариев к строкам кода.

Назначение: проверить, что к КАЖДОЙ строке кода проекта есть комментарий
на русском языке и что в комментариях нет отсылок к внешней документации.

Как считается строка кода:
  * файл разбирается синтаксическим деревом; строка считается строкой кода,
    если она попадает в диапазон (lineno..end_lineno) любого оператора или
    выражения и не является пустой;
  * строка, состоящая только из закрывающих скобок, запятой или точки с
    запятой, считается продолжением и не требует комментария;
  * содержимое строкового литерала (в том числе docstring) — это не код, но
    оно проверяется на отсылки к документации.

Чем считается комментарий:
  * текст после символа # на той же строке, причем # не внутри строкового
    литерала;
  * комментарий считается русским, если содержит хотя бы одну букву
    кириллицы.

Что считается отсылкой к документации (запрещено в комментариях и
строковых литералах-документах; точный шаблон задаёт REF_RE):
  * номера рабочих «волн» с числом; коды вида латинская C + три цифры
    (+ одна строчная буква); особые имена служебных файлов: файл теории,
    файл устройства проекта, файл с кратким описанием, файл состояния;
  * пути в каталог текстов проекта, знак параграфа, слово-указатель
    «посмотри в другом месте», слово «коммит» с шестнадцатеричным хешем,
    даты в формате год-месяц-день, временный каталог другого проекта,
    имена сохранённых состояний модели, чужие служебные каталоги.

Печать: по каждому файлу число строк кода, число строк без комментария,
число нерусских комментариев, число отсылок; затем список до N примеров.
Код возврата 0, если замечаний нет, иначе 1.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import argparse  # разбор аргументов командной строки
import ast  # синтаксическое дерево Python
import io  # чтение исходника как потока для токенизатора
import re  # регулярные выражения
import sys  # код возврата процесса
import tokenize  # поиск комментариев-токенов
from pathlib import Path  # пути к файлам

ROOT = Path(__file__).resolve().parent.parent  # корень проекта — каталог на два уровня выше этого файла
DEFAULT_ROOTS = ("hagi", "scripts", "tests", "training")  # корни проверки по умолчанию, если пути не заданы

# регулярное выражение для отсылок к внешней документации
REF_RE = re.compile(  # шаблон запрещённых отсылок (описание классов — в docstring модуля); начало
    r"\bWAVE\s*\d+|\bwave\s*\d+|\bC\d{3}[a-z]?\b|\bprimes\b|\.lean\b|THEORY_ALIGNMENT"  # номер волны, код вида буква C + три цифры, служебные имена файлов документации
    r"|ARCHITECTURE\.md|\bREADME\b|STATUS\.md|\bdocs/|§|\bсм\.\s|\bsee\s+(Hagi|the\s+doc|doc)"
    r"|\bcommit\b\s*[0-9a-f]{6,}|\b20\d\d-\d\d-\d\d\b|/var/tmp|checkpoints/Надя|checkpoints/HAGI_EXACT"
    r"|\bREAD-ONLY\b|\bupstream\b|\brepro\b|\bsandbox\b|\breport\b|\bbench\b|\bbridge\b|\bHagi/"  # режим чтения, слова-маркеры чужих отчётов и префикс пути чужого проекта; конец
)
# закрывающая строка: только скобки, запятая, точка с запятой, двоеточие
CLOSER_RE = re.compile(r"^[\s)\]}.,:]*$")  # строка только из скобок, запятой, точки с запятой, двоеточия
CYR_RE = re.compile(r"[А-Яа-яЁё]")  # любая буква кириллицы (Ё и ё включены)


def code_line_numbers(tree: ast.AST, src_lines: list[str]) -> set[int]:
    """Номера физических строк, на которых есть код.

    Строки, целиком лежащие внутри строкового литерала (docstring,
    многострочный текст), кодом не считаются — они сами являются
    текстом и проверяются отдельно на отсылки."""
    nums: set[int] = set()  # множество номеров строк с кодом — пока пустое
    literal: set[int] = set()  # множество номеров строк внутри строковых литералов — пока пустое
    for node in ast.walk(tree):  # обойти все узлы дерева
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):  # docstring / строка-описание — целиком не код
            for ln in range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1):  # по строкам docstring или строки-описания: lineno..end_lineno
                literal.add(ln)  # строка ln целиком текст, не код
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):  # узел — строковая или байтовая константа
            # исключаем только ВНУТРЕННИЕ строки многострочного литерала (без кода на них);
            # первая и последняя строки содержат код (присваивание, вызов)
            end = getattr(node, "end_lineno", node.lineno)  # последняя строка литерала
            for ln in range(node.lineno + 1, end):  # внутренние строки литерала: lineno+1..end−1
                literal.add(ln)  # строка ln — текст литерала, не код
        if isinstance(node, (ast.stmt, ast.expr)):  # любой оператор или выражение
            end = getattr(node, "end_lineno", node.lineno)  # последняя строка узла
            for ln in range(node.lineno, end + 1):  # диапазон lineno..end
                nums.add(ln)  # строка ln содержит код
    return {n for n in nums if src_lines[n - 1].strip() and n not in literal}  # оставить только непустые строки, не лежащие в литералах


def comment_map(src: str) -> dict[int, str]:
    """Номер строки -> текст комментария на этой строке."""
    out: dict[int, str] = {}  # результат: номер строки -> комментарий
    try:  # токенизация может упасть на незакрытой скобке
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))  # список всех токенов исходника
    except tokenize.TokenError:  # ошибка токенизации
        return out  # вернуть пустой результат
    prev_end = (1, 0)  # координата конца предыдущего токена (не используется)
    for tok in toks:  # по всем токенам
        if tok.type == tokenize.COMMENT:  # токен — комментарий
            # комментарий относится к строке, где он стоит; если строка
            # началась раньше (многострочный выражение), всё равно берём её
            out[tok.start[0]] = tok.string[1:].strip()  # текст без # и крайних пробелов; ключ — номер строки, где стоит
        prev_end = tok.end  # запомнить конец токена
    return out  # карта комментариев


def docstring_strings(tree: ast.AST) -> list[tuple[int, str]]:
    """(номер строки, текст) всех строковых литералов-документов модуля,
    классов и функций, а также строк-описаний — строковых литералов отдельным
    оператором-выражением. Строковые литералы-код (в присваиваниях, вызовах)
    документацией не считаются."""
    out: list[tuple[int, str]] = []  # список (номер строки, текст)
    for node in ast.walk(tree):  # обойти все узлы дерева
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):  # модуль, класс или функция
            ds = ast.get_docstring(node, clean=False)  # docstring узла без очистки отступов
            if ds is not None:  # docstring есть
                first = node.body[0]  # первый оператор тела — сам литерал
                out.append((first.lineno, ds))  # добавить (строка литерала, текст)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):  # строка-описание отдельным оператором (не первый в теле) — тоже документация
            out.append((node.lineno, node.value.value))  # запомнить её текст
    return out  # все найденные тексты документации


def audit_file(path: Path) -> dict:  # проверить один файл и вернуть отчёт-словарь
    src = path.read_text(encoding="utf-8")  # исходник в UTF-8
    lines = src.splitlines()  # список строк
    tree = ast.parse(src)  # дерево исходника
    code = code_line_numbers(tree, lines)  # номера строк кода
    cmts = comment_map(src)  # карта комментариев
    docs = docstring_strings(tree)  # тексты документации
    missing, foreign, refs = [], [], []  # списки: без комментария, нерусские, отсылки
    # строки def/class, у которых есть docstring, считаются пояснёнными
    documented = {node.lineno for node in ast.walk(tree)  # строки def/class, имеющие docstring
                  if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))  # только классы и функции
                  and ast.get_docstring(node) is not None}  # docstring не пуст
    for n in sorted(code):  # по номерам строк кода по возрастанию
        if n in documented:  # строка с docstring уже пояснена
            continue  # пропустить
        raw = lines[n - 1]  # текст строки n
        if CLOSER_RE.match(raw):  # строка — только закрывающие скобки/знаки
            continue  # не требует комментария
        c = cmts.get(n)  # комментарий строки или None
        if c is None and raw.rstrip().endswith("\\") and n - 1 in cmts:  # после обратного слеша комментарий синтаксически невозможен — пояснением служит комментарий строки выше
            continue  # считается пояснённой
        if c is None:  # комментария нет
            missing.append(n)  # в список пропущенных
        elif not CYR_RE.search(c):  # комментарий без кириллицы
            foreign.append(n)  # в список нерусских
    # отсылки: в комментариях и в docstring/многострочных литералах
    for n, c in cmts.items():  # по всем комментариям
        if REF_RE.search(c):  # комментарий содержит отсылку
            refs.append((n, "комментарий"))  # записать (строка, вид «комментарий»)
    for n, d in docs:  # по всем текстам документации
        for off, line in enumerate(d.splitlines()):  # по строкам текста с их смещением
            if REF_RE.search(line):  # строка текста содержит отсылку
                refs.append((n + off, "docstring"))  # записать (строка n+off, вид «docstring»)
    return {"path": (str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)).replace("\\", "/"),  # путь для отчёта
            "code": len(code), "missing": missing, "foreign": foreign, "refs": refs}  # отчёт: относительный путь с /, число строк кода и три списка замечаний


class _DropDocstrings(ast.NodeTransformer):
    """Убрать docstring из модуля, классов и функций (они не влияют на вычисления)."""

    def _drop(self, node):  # общий обработчик для узлов с телом
        # убрать ВСЕ операторы-выражения из строковых констант: docstring (первый)
        # и строки-описания после операторов — они ничего не вычисляют
        node.body = [st for st in node.body if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)  # оставить операторы, не являющиеся строковой константой-выражением
                                                     and isinstance(st.value.value, str))] or [ast.Pass()]  # пустое тело заменить на pass
        self.generic_visit(node)  # обойти тело узла
        return node  # вернуть узел

    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _drop  # применять ко всем таким узлам


def same_code(old: Path, new: Path) -> bool:
    """True, если old и new — один и тот же код (совпадают деревья без docstring)."""
    a = ast.dump(_DropDocstrings().visit(ast.parse(old.read_text(encoding="utf-8"))))  # дерево старого файла
    b = ast.dump(_DropDocstrings().visit(ast.parse(new.read_text(encoding="utf-8"))))  # дерево нового файла
    if a != b:  # код изменился
        print("КОД ИЗМЕНЁН:", old, "!=", new)  # сообщить, какой файл
    return a == b  # результат сравнения


def main(argv: list[str] | None = None) -> int:  # точка входа: разбор аргументов и проверка файлов; 0 — замечаний нет
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])  # парсер аргументов; описание — первый абзац docstring модуля
    ap.add_argument("paths", nargs="*", help="файлы или каталоги; по умолчанию все корни проекта")  # позиционные: файлы или каталоги
    ap.add_argument("--examples", type=int, default=12, help="сколько примеров печатать на файл")  # число примеров на файл
    ap.add_argument("--quiet", action="store_true", help="печать только итог и файлы с замечаниями")  # краткий вывод
    ap.add_argument("--lines", type=int, nargs=2, default=None, metavar=("A", "B"),  # диапазон строк A..B
                    help="проверять только строки с A по B включительно (для больших файлов по частям)")  # подсказка диапазона
    ap.add_argument("--ast-compare", nargs=2, default=None, metavar=("OLD", "NEW"),  # пара файлов OLD NEW для сравнения кода
                    help="сравнить код OLD и NEW без комментариев и docstring: код обязан совпасть")  # подсказка сравнения
    a = ap.parse_args(argv)  # argv по умолчанию — аргументы процесса
    if a.ast_compare:  # проверка неизменности кода
        return 0 if same_code(Path(a.ast_compare[0]), Path(a.ast_compare[1])) else 1  # режим сравнения: код совпал — 0, иначе 1
    roots = a.paths or list(DEFAULT_ROOTS)  # заданные пути или корни по умолчанию
    files: list[Path] = []  # список файлов для проверки
    for r in roots:  # по корням
        p = ROOT / r if not Path(r).is_absolute() else Path(r)  # относительный путь — от корня проекта
        if p.is_file():  # это файл
            files.append(p)  # добавить его
        elif p.is_dir():  # это каталог
            files += sorted(p.rglob("*.py"))  # все .py внутри, по порядку имён
    tot_code = tot_missing = tot_foreign = tot_refs = 0  # суммы по всем файлам
    bad_files = 0  # число файлов с замечаниями
    for f in files:  # по файлам
        try:  # файл может не разобраться
            r = audit_file(f)  # отчёт по файлу
            src = f.read_text(encoding="utf-8"); lines = src.splitlines(); tree = ast.parse(src)  # строки и дерево для подсчёта кода в диапазоне
        except SyntaxError as e:  # синтаксическая ошибка
            print(f"ОШИБКА СИНТАКСИСА {f}: {e}")  # сообщить файл и ошибку
            bad_files += 1  # файл с замечанием
            continue  # к следующему файлу
        if a.lines:  # оставить только замечания в заданном диапазоне строк
            A, B = a.lines  # границы диапазона
            r["missing"] = [n for n in r["missing"] if A <= n <= B]  # пропущенные только внутри диапазона
            r["foreign"] = [n for n in r["foreign"] if A <= n <= B]  # нерусские только внутри диапазона
            r["refs"] = [(n, k) for n, k in r["refs"] if A <= n <= B]  # отсылки только внутри диапазона
            r["code"] = sum(1 for n in code_line_numbers(tree, lines) if A <= n <= B)  # строк кода внутри диапазона (нумерация исходного файла)
        tot_code += r["code"]; tot_missing += len(r["missing"])  # накопить строки кода и пропуски
        tot_foreign += len(r["foreign"]); tot_refs += len(r["refs"])  # накопить нерусские и отсылки
        if r["missing"] or r["foreign"] or r["refs"]:  # есть замечания
            bad_files += 1  # учесть файл
            if not a.quiet:  # в полном режиме — печать деталей
                print(f"{r['path']}: код {r['code']} | без комментария {len(r['missing'])}"  # строка отчёта: путь, код, без комментария
                      f" | не по-русски {len(r['foreign'])} | отсылки {len(r['refs'])}")  # хвост: не по-русски, отсылки
                lines = f.read_text(encoding="utf-8").splitlines()  # строки файла
                shown = 0  # число напечатанных примеров
                for n in r["missing"][:a.examples]:  # примеры строк без комментария
                    print(f"    нет   {n:5}: {lines[n-1].strip()[:100]}"); shown += 1  # печать строки; shown + 1
                for n in r["foreign"][:max(0, a.examples - shown)]:  # примеры нерусских, не более остатка
                    print(f"    англ  {n:5}: {lines[n-1].strip()[:100]}"); shown += 1  # печать строки; shown + 1
                for n, kind in r["refs"][:max(0, a.examples - shown)]:  # примеры отсылок, не более остатка
                    print(f"    отсыл {n:5} ({kind}): {lines[n-1].strip()[:100]}")  # печать строки и вида отсылки
    print(f"ИТОГ: файлов {len(files)} | строк кода {tot_code} | без комментария {tot_missing}"  # итог: файлы, строки кода, без комментария
          f" | не по-русски {tot_foreign} | отсылок {tot_refs} | файлов с замечаниями {bad_files}")  # хвост итога
    return 0 if (tot_missing == 0 and tot_foreign == 0 and tot_refs == 0 and bad_files == 0) else 1  # 0, если замечаний нет, иначе 1


if __name__ == "__main__":  # запуск как скрипт
    sys.exit(main())  # код возврата main
