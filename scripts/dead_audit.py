"""Поиск недостижимых верхнеуровневых имён одного модуля пакета hagi: какие имена никто не использует.

Аргумент: имя модуля (например core). Строится словарь узлов: каждая функция и
класс верхнего уровня, а также каждое имя, присвоенное на уровне модуля (в том
числе внутри блоков if/try/for); операторы верхнего уровня без присваивания
образуют псевдоузел <module code>. Зависимости: узел n зависит от m, если слово m
встречается в тексте узла n. Точка входа: имя, встречающееся в любом другом
файле проекта (кроме самого модуля), плюс main и <module code>. Достижимость —
обход в глубину от точек входа. Печатаются только недостижимые функции и классы.
Недостижимое не равно мёртвому: перед удалением проверить другими средствами.
"""
import ast  # разбор исходника в дерево
import os  # обход каталогов и смена текущего каталога
import re  # поиск слова в тексте
import sys  # аргумент командной строки

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # корень проекта: каталог на два уровня выше файла
os.chdir(ROOT)  # сделать корень проекта текущим каталогом


def read(p):
    """Прочитать файл в текст UTF-8; недопустимые байты отбрасываются."""
    return open(p, encoding="utf-8", errors="ignore").read()  # текст файла p


def sources(exclude):
    """Пути всех исходников проекта (hagi, scripts, tests, training, .github и файлы .py в корне), кроме пути exclude."""
    for base in ("hagi", "scripts", "tests", "training", ".github"):  # по пяти каталогам
        for r, _, fs in os.walk(base):  # обход каталога: r — каталог, fs — имена файлов
            for f in fs:  # по файлам каталога
                p = os.path.join(r, f)  # путь файла
                if (os.path.normpath(p) != os.path.normpath(exclude)  # путь не совпадает с исключённым (нормализованные пути)
                        and f.endswith((".py", ".yml", ".yaml", ".sh", ".ps1"))):  # расширение относится к коду или настройке
                    yield p  # выдать путь
    for f in os.listdir("."):  # по именам в корне проекта
        if f.endswith(".py"):  # только файлы .py
            yield f  # выдать имя файла


def markdown():
    """Пути всех файлов .md проекта вне каталогов checkpoints и .git."""
    for r, _, fs in os.walk("."):  # обход всего проекта
        if "checkpoints" in r or ".git" in r:  # каталог относится к контрольным точкам или к хранилищу git
            continue  # пропустить каталог
        for f in fs:  # по файлам каталога
            if f.endswith(".md"):  # только файлы .md
                yield os.path.join(r, f)  # выдать путь файла


def nodes(tree, lines):
    """Словарь: имя -> исходный текст узла.

        Узлы: каждая функция и класс верхнего уровня и каждое имя, присвоенное на
        уровне модуля, включая блоки if/try/for. Текст присваивающего узла — весь
        оператор верхнего уровня, который его связывает (поэтому
        `QUOTIENTS.append(q_shift(d))` внутри if считается). Операторы верхнего
        уровня без присваивания образуют псевдоузел <module code>."""
    out = {}  # результат: имя -> текст
    MODULE = "<module code>"  # имя псевдоузля для прочего кода уровня модуля

    def text(stmt):
        """Исходный текст оператора stmt: строки с stmt.lineno по stmt.end_lineno, соединённые переводом строки."""
        return "\n".join(lines[stmt.lineno - 1:stmt.end_lineno])  # срез строк [lineno−1, end_lineno)

    for n in tree.body:  # по операторам верхнего уровня модуля
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):  # функция или класс
            out[n.name] = text(n)  # имя -> текст определения
            continue  # следующий оператор
        owner = text(n)  # текст прочего оператора верхнего уровня
        out[MODULE] = out.get(MODULE, "") + "\n" + owner  # добавить его текст к псевдоузлу через перевод строки
        for st in ast.walk(n):  # по всем вложенным узлам оператора
            if isinstance(st, ast.Assign):  # обычное присваивание a = b = ...
                targets = st.targets  # список целей присваивания
            elif isinstance(st, ast.AnnAssign):  # присваивание с аннотацией a: T = ...
                targets = [st.target]  # цель одна
            else:  # не присваивание
                continue  # пропустить узел
            for t_ in targets:  # по целям присваивания
                if isinstance(t_, ast.Name):  # цель — простое имя
                    out.setdefault(t_.id, owner)  # имя -> текст оператора (не перезаписывать, если уже есть)
    return out  # вернуть словарь узлов


def main(argv: list[str]) -> int:  # точка входа: argv — аргументы командной строки
    mod = argv[1]  # имя модуля — первый аргумент командной строки
    path = f"hagi/{mod}.py"  # путь файла модуля
    src = read(path)  # текст модуля
    tree = ast.parse(src)  # синтаксическое дерево модуля
    lines = src.splitlines()  # строки текста модуля
    tops = nodes(tree, lines)  # словарь узлов: имя -> текст
    funcs = {n.name: (n.lineno, n.end_lineno) for n in tree.body  # имя функции или класса -> (номер первой строки, номер последней строки)
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}  # только функции и классы верхнего уровня
    corpus = list(sources(path))  # список всех прочих исходников проекта
    assert os.path.normpath(path) not in {os.path.normpath(p) for p in corpus}, \
        "module leaked into its own corpus"  # проверить, что модуль не попал в собственный корпус
    others = "\n".join(read(p) for p in corpus)  # весь текст прочих исходников, соединённый переводами строки
    docs = "\n".join(read(p) for p in markdown())  # весь текст файлов .md, соединённый переводами строки
    word = lambda n: r"\b%s\b" % re.escape(n)  # регулярное выражение: слово n целиком, спецсимволы экранированы
    deps = {n: {m for m in tops if m != n and re.search(word(m), b)}  # deps[n] = {m : m ≠ n и слово m встречается в тексте b узла n}
            for n, b in tops.items()}  # по всем узлам (n — имя, b — текст)
    entry = {n for n in tops if re.search(word(n), others)} | {"main"}  # точки входа: имена, встречающиеся в прочих исходниках, плюс main
    entry.add("<module code>")          # операторы уровня модуля выполняются при импорте
    reach = {e for e in entry if e in tops}   # достижимые сначала — сами точки входа, имеющиеся среди узлов
    stack = list(reach)  # стек обхода в глубину
    while stack:  # пока стек не пуст
        n = stack.pop()  # снять узел с вершины стека
        for m in deps[n]:  # по зависимостям узла n
            if m not in reach:  # зависимость ещё не достигнута
                reach.add(m)  # добавить её к достижимым
                stack.append(m)  # и положить в стек
    total = 0  # суммарное число строк недостижимых функций и классов
    for n in sorted(tops):  # по именам узлов в порядке сортировки
        if n in reach or n not in funcs:  # узел достижим либо не является функцией или классом
            continue  # не печатать
        a, b = funcs[n]  # a — первая строка, b — последняя строка определения
        total += b - a + 1  # total += число строк b − a + 1
        print(f"{path}:{a}-{b}  {n:34} {b - a + 1:4} lines  "  # печать пути, диапазона строк, имени в 14 позициях и числа строк
              f"code={'yes' if re.search(word(n), others) else 'no '} "  # code = yes, если имя встречается в прочих исходниках, иначе no
              f"docs={'yes' if re.search(word(n), docs) else 'no'}")  # docs = yes, если имя встречается в файлах .md, иначе no
    print(f"{path}: unreachable {total} lines")  # печать суммарного числа недостижимых строк
    return 0  # код завершения: успех



if __name__ == "__main__":  # запуск из командной строки
    sys.exit(main(sys.argv))  # выполнить и вернуть код завершения
