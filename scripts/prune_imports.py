"""Убирает из модулей неиспользуемые и повторные импорты (чистка после разбиения монолита).

Правила:
  * импорт считается неиспользуемым, если его локальное имя не встречается в
    модуле ни как имя, ни как корень атрибута, ни в строковой аннотации;
  * из повторных импортов одного и того же (модуль + имя) остаётся первый;
  * `import *` не трогается (переносит неизвестный набор имён);
  * файлы `__init__.py` не трогаются (там импорты ради перепродажи имён).

Запуск: python -X utf8 scripts/prune_imports.py hagi/core hagi/*.py
"""
from __future__ import annotations  # отложенные аннотации

import ast  # разбор исходников
import re  # регулярные выражения
import sys  # аргументы
from pathlib import Path  # пути


def names_used(tree: ast.AST) -> set[str]:
    """Все имена, к которым модуль обращается: имена, корни атрибутов, идентификаторы строковых аннотаций, global."""
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}  # имена (загрузка и связь)
    used |= {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}  # корни атрибутов
    used |= {g for n in ast.walk(tree) if isinstance(n, ast.Global) for g in n.names}  # global-имена
    docs = set()  # узлы docstring
    for node in ast.walk(tree):  # по узлам
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):  # тела
            for st in node.body:  # первые операторы
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):  # строковый литерал (докстринг)
                    docs.add(st.value)  # запомнить
    for n in ast.walk(tree):  # строковые литералы-аннотации
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n not in docs:  # не docstring
            used |= set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", n.value))  # все идентификаторы
    return used  # множество имён


def prune(path: Path, extra_used: set[str] | None = None) -> int:
    """Убрать неиспользуемые/повторные импорты в файле; вернуть число удалённых строк."""
    src = path.read_text(encoding="utf-8")  # текст
    tree = ast.parse(src)  # дерево
    L = src.split("\n")  # строки
    used = names_used(tree) | set(extra_used or ())  # используемые имена + имена, которые другие импортируют отсюда
    drop: set[int] = set()  # номера строк к удалению (1-основанные, диапазоны)
    seen: set[tuple] = set()  # уже виденные (модуль, имена)
    for node in tree.body:  # верхний уровень
        if not isinstance(node, (ast.Import, ast.ImportFrom)):  # не импорт
            continue  # дальше
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):  # звёздочка
            continue  # не трогаем
        key = (getattr(node, "module", None), tuple(sorted((a.name, a.asname) for a in node.names)))  # ключ импорта
        if key in seen:  # повтор
            drop.update(range(node.lineno, node.end_lineno + 1))  # удалить целиком (все строки оператора)
            continue  # дальше
        seen.add(key)  # запомнить
        keep = []  # имена, которые оставляем
        for a in node.names:  # по именам импорта
            local = a.asname or a.name.split(".")[0]  # локальное имя
            if local in used or getattr(node, "module", None) == "__future__":  # используется (либо служебная директива __future__)
                keep.append(a)  # оставить
        if not keep:  # все имена не используются
            drop.update(range(node.lineno, node.end_lineno + 1))  # удалить весь оператор (все строки)
            continue  # дальше
        if len(keep) != len(node.names) and len(L[node.lineno - 1].rstrip()) >= 1:  # часть имён не используется
            # переписать оператор в одну строку, сохранив комментарий исходной строки
            line = L[node.lineno - 1]  # первая строка оператора
            cmt = line[line.find("#"):] if "#" in line and not line.strip().startswith("#") else ""  # комментарий
            if isinstance(node, ast.ImportFrom):  # форма «from X import a, b»
                text = f"from {node.module} import " + ", ".join((a.name if not a.asname else f"{a.name} as {a.asname}") for a in keep)  # текст
            else:  # форма «import X»
                text = "import " + ", ".join((a.name if not a.asname else f"{a.name} as {a.asname}") for a in keep)  # текст
            L[node.lineno - 1] = text + (("  " + cmt) if cmt else "")  # заменить первую строку
            for ln in range(node.lineno + 1, node.end_lineno + 1):  # остальные строки оператора
                drop.add(ln)  # удалить
    for ln in sorted(drop, reverse=True):  # с конца
        del L[ln - 1]  # удалить строку
    out = "\n".join(L)  # новый текст
    ast.parse(out)  # проверка синтаксиса
    if out != src:  # изменилось
        path.write_text(out, encoding="utf-8")  # записать
    return len(drop)  # число удалённых строк


def main(argv: list[str]) -> int:
    """argv[1:] — файлы или каталоги (рекурсивно, *.py, без __init__.py)."""
    files: list[Path] = []  # список файлов
    for a in argv[1:]:  # по аргументам
        p = Path(a)  # путь
        if p.is_file():  # файл
            files.append(p)  # добавить
        elif p.is_dir():  # каталог
            files += sorted(q for q in p.rglob("*.py") if q.name != "__init__.py")  # все модули без __init__
    total = 0  # всего удалено строк
    # имена, которые ДРУГИЕ модули проекта импортируют из наших файлов
    # (перепродажа: часть импортирует имя только ради следующей части)
    demanded: dict[str, set[str]] = {}  # путь файла -> множество имён
    for g in list(Path("hagi").rglob("*.py")) + list(Path("scripts").glob("*.py")) + list(Path("tests").glob("*.py")):  # весь проект
        try:  # разбор
            tree = ast.parse(g.read_text(encoding="utf-8"))  # дерево
        except SyntaxError:  # битый файл
            continue  # пропустить
        for node in ast.walk(tree):  # по узлам
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("hagi"):  # импорт из пакета
                target = Path(*node.module.split(".")).with_suffix(".py")  # путь модуля-источника
                for a_ in node.names:  # по именам
                    if a_.name == "*":  # звёздочка — нужны все публичные
                        demanded.setdefault(str(target), set())  # пометить файл (пустое множество = всё публичное)
                    else:  # конкретное имя
                        demanded.setdefault(str(target), set()).add(a_.name)  # добавить
    for f in files:  # по файлам
        rel = f.as_posix()  # относительный путь
        extra = set()  # дополнительные «используемые» имена
        for k, v in demanded.items():  # по спросу
            if rel == k.replace("\\", "/"):  # спрос на этот файл (точное совпадение пути, не по суффиксу)
                extra |= v  # учесть (при звёздочке множество пусто — публичные имена и так не импорт)
        n = prune(f, extra)  # почистить
        if n:  # было что удалять
            print(f"{f.as_posix()}: −{n}")  # отчёт
            total += n  # суммировать
    print(f"ИТОГ: файлов {len(files)}, удалено строк импорта {total}")  # итог
    return 0  # успех


if __name__ == "__main__":  # запуск из командной строки
    sys.exit(main(sys.argv))  # выполнить
