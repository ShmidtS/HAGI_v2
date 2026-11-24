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
строковых литералах-документах):
  * номера волн (WAVE 123, wave123), коды теорем (C307, C598), слова
    primes / lean / THEORY_ALIGNMENT / ARCHITECTURE / README / STATUS.md,
    ссылки docs/..., символ параграфа §, слова "см." и "commit" с хешем,
    даты вида 2026-11-24, ссылки на каталоги других проектов.

Печать: по каждому файлу число строк кода, число строк без комментария,
число нерусских комментариев, число отсылок; затем список до N примеров.
Код возврата 0, если замечаний нет, иначе 1.
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = ("hagi", "scripts", "tests", "training")

# регулярное выражение для отсылок к внешней документации
REF_RE = re.compile(
    r"\bWAVE\s*\d+|\bwave\s*\d+|\bC\d{3}[a-z]?\b|\bprimes\b|\.lean\b|THEORY_ALIGNMENT"
    r"|ARCHITECTURE\.md|\bREADME\b|STATUS\.md|\bdocs/|§|\bсм\.\s|\bsee\s+(Hagi|the\s+doc|doc)"
    r"|\bcommit\b\s*[0-9a-f]{6,}|\b20\d\d-\d\d-\d\d\b|/var/tmp|checkpoints/Надя|checkpoints/HAGI_EXACT"
    r"|\bREAD-ONLY\b|\bupstream\b|\brepro\b|\bsandbox\b|\breport\b|\bbench\b|\bbridge\b|\bHagi/"
)
# закрывающая строка: только скобки, запятая, точка с запятой, двоеточие
CLOSER_RE = re.compile(r"^[\s)\]}.,:]*$")
CYR_RE = re.compile(r"[А-Яа-яЁё]")


def code_line_numbers(tree: ast.AST, src_lines: list[str]) -> set[int]:
    """Номера физических строк, на которых есть код.

    Строки, целиком лежащие внутри строкового литерала (docstring,
    многострочный текст), кодом не считаются — они сами являются
    текстом и проверяются отдельно на отсылки."""
    nums: set[int] = set()
    literal: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):  # docstring / строка-описание — целиком не код
            for ln in range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1):
                literal.add(ln)
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            # исключаем только ВНУТРЕННИЕ строки многострочного литерала (без кода на них);
            # первая и последняя строки содержат код (присваивание, вызов)
            end = getattr(node, "end_lineno", node.lineno)
            for ln in range(node.lineno + 1, end):
                literal.add(ln)
        if isinstance(node, (ast.stmt, ast.expr)):
            end = getattr(node, "end_lineno", node.lineno)
            for ln in range(node.lineno, end + 1):
                nums.add(ln)
    return {n for n in nums if src_lines[n - 1].strip() and n not in literal}


def comment_map(src: str) -> dict[int, str]:
    """Номер строки -> текст комментария на этой строке."""
    out: dict[int, str] = {}
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return out
    prev_end = (1, 0)
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            # комментарий относится к строке, где он стоит; если строка
            # началась раньше (многострочный выражение), всё равно берём её
            out[tok.start[0]] = tok.string[1:].strip()
        prev_end = tok.end
    return out


def docstring_strings(tree: ast.AST) -> list[tuple[int, str]]:
    """(номер строки, текст) всех строковых литералов-документов модуля,
    классов и функций, а также строк-описаний — строковых литералов отдельным
    оператором-выражением. Строковые литералы-код (в присваиваниях, вызовах)
    документацией не считаются."""
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            ds = ast.get_docstring(node, clean=False)
            if ds is not None:
                first = node.body[0]
                out.append((first.lineno, ds))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):  # строка-описание отдельным оператором (не первый в теле) — тоже документация
            out.append((node.lineno, node.value.value))  # запомнить её текст
    return out


def audit_file(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    code = code_line_numbers(tree, lines)
    cmts = comment_map(src)
    docs = docstring_strings(tree)
    missing, foreign, refs = [], [], []
    # строки def/class, у которых есть docstring, считаются пояснёнными
    documented = {node.lineno for node in ast.walk(tree)
                  if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and ast.get_docstring(node) is not None}
    for n in sorted(code):
        if n in documented:
            continue
        raw = lines[n - 1]
        if CLOSER_RE.match(raw):
            continue
        c = cmts.get(n)
        if c is None and raw.rstrip().endswith("\\") and n - 1 in cmts:  # после обратного слеша комментарий синтаксически невозможен — пояснением служит комментарий строки выше
            continue
        if c is None:
            missing.append(n)
        elif not CYR_RE.search(c):
            foreign.append(n)
    # отсылки: в комментариях и в docstring/многострочных литералах
    for n, c in cmts.items():
        if REF_RE.search(c):
            refs.append((n, "комментарий"))
    for n, d in docs:
        for off, line in enumerate(d.splitlines()):
            if REF_RE.search(line):
                refs.append((n + off, "docstring"))
    return {"path": (str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)).replace("\\", "/"),  # путь для отчёта
            "code": len(code), "missing": missing, "foreign": foreign, "refs": refs}


class _DropDocstrings(ast.NodeTransformer):
    """Убрать docstring из модуля, классов и функций (они не влияют на вычисления)."""

    def _drop(self, node):  # общий обработчик для узлов с телом
        # убрать ВСЕ операторы-выражения из строковых констант: docstring (первый)
        # и строки-описания после операторов — они ничего не вычисляют
        node.body = [st for st in node.body if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="файлы или каталоги; по умолчанию все корни проекта")
    ap.add_argument("--examples", type=int, default=12, help="сколько примеров печатать на файл")
    ap.add_argument("--quiet", action="store_true", help="печать только итог и файлы с замечаниями")
    ap.add_argument("--lines", type=int, nargs=2, default=None, metavar=("A", "B"),
                    help="проверять только строки с A по B включительно (для больших файлов по частям)")
    ap.add_argument("--ast-compare", nargs=2, default=None, metavar=("OLD", "NEW"),
                    help="сравнить код OLD и NEW без комментариев и docstring: код обязан совпасть")
    a = ap.parse_args(argv)
    if a.ast_compare:  # проверка неизменности кода
        return 0 if same_code(Path(a.ast_compare[0]), Path(a.ast_compare[1])) else 1
    roots = a.paths or list(DEFAULT_ROOTS)
    files: list[Path] = []
    for r in roots:
        p = ROOT / r if not Path(r).is_absolute() else Path(r)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += sorted(p.rglob("*.py"))
    tot_code = tot_missing = tot_foreign = tot_refs = 0
    bad_files = 0
    for f in files:
        try:
            r = audit_file(f)
            src = f.read_text(encoding="utf-8"); lines = src.splitlines(); tree = ast.parse(src)  # строки и дерево для подсчёта кода в диапазоне
        except SyntaxError as e:
            print(f"ОШИБКА СИНТАКСИСА {f}: {e}")
            bad_files += 1
            continue
        if a.lines:  # оставить только замечания в заданном диапазоне строк
            A, B = a.lines  # границы диапазона
            r["missing"] = [n for n in r["missing"] if A <= n <= B]  # пропущенные только внутри диапазона
            r["foreign"] = [n for n in r["foreign"] if A <= n <= B]  # нерусские только внутри диапазона
            r["refs"] = [(n, k) for n, k in r["refs"] if A <= n <= B]  # отсылки только внутри диапазона
            r["code"] = sum(1 for n in code_line_numbers(tree, lines) if A <= n <= B)  # строк кода внутри диапазона (нумерация исходного файла)
        tot_code += r["code"]; tot_missing += len(r["missing"])
        tot_foreign += len(r["foreign"]); tot_refs += len(r["refs"])
        if r["missing"] or r["foreign"] or r["refs"]:
            bad_files += 1
            if not a.quiet:
                print(f"{r['path']}: код {r['code']} | без комментария {len(r['missing'])}"
                      f" | не по-русски {len(r['foreign'])} | отсылки {len(r['refs'])}")
                lines = f.read_text(encoding="utf-8").splitlines()
                shown = 0
                for n in r["missing"][:a.examples]:
                    print(f"    нет   {n:5}: {lines[n-1].strip()[:100]}"); shown += 1
                for n in r["foreign"][:max(0, a.examples - shown)]:
                    print(f"    англ  {n:5}: {lines[n-1].strip()[:100]}"); shown += 1
                for n, kind in r["refs"][:max(0, a.examples - shown)]:
                    print(f"    отсыл {n:5} ({kind}): {lines[n-1].strip()[:100]}")
    print(f"ИТОГ: файлов {len(files)} | строк кода {tot_code} | без комментария {tot_missing}"
          f" | не по-русски {tot_foreign} | отсылок {tot_refs} | файлов с замечаниями {bad_files}")
    return 0 if (tot_missing == 0 and tot_foreign == 0 and tot_refs == 0 and bad_files == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
