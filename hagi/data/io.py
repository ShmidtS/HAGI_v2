"""Ввод и вывод данных: учиться на произвольных файлах и исполнять написанные моделью программы.

    Две половины файла.
    1) УЧЁБА НА ДАННЫХ. Любой файл превращается в список точных задач TaskSpec
       (функция учителя f: x -> y, заданная таблицей наблюдённых пар):
         * CSV/TSV/TAB: колонка — список целых чисел (нецелые ячейки пропускаются);
           для каждой упорядоченной пары разных колонок (X, Y) задача «X -> Y»;
         * JSON: обход дерева, собираются все целые числа (без логических значений),
           числа с чётными позициями — аргументы, с нечётными — ответы;
         * любой другой файл: по байтам задача «следующий байт»: f(a) = b для пар
           (data[i], data[i+1]); байты, за которыми идут разные байты, исключаются.
       Задача создаётся, только если каждому x соответствует ровно одно y и таких
       пар не меньше 4 (для следующего байта — не меньше 8).
       2) ИСПОЛНЕНИЕ ПРОГРАММ. Файл checkpoints/self/generated_programs.py (его пишет
       модель) загружается как модуль; execute(name, *args) вызывает функцию по имени.

    Экспортирует: ingest_file, learn_from_data, ingest_main, available, execute,
    execute_text, execute_main, DEFAULT_PROGRAMS."""
from __future__ import annotations  # отложенное вычисление аннотаций типов


# ==============================================================================
# ЗАГРУЗКА: учиться на произвольных данных
# ==============================================================================

import argparse  # разбор аргументов командной строки
import csv  # чтение таблиц CSV/TSV
import json  # чтение и запись JSON
from pathlib import Path  # работа с путями файлов
from typing import Any  # аннотации типов
from hagi.core import GrowingHagiProcessor, TaskSpec  # ядро: процессор и описание задачи
def _int_or_none(v: str) -> int | None:  # целое число из текста ячейки или None, если это не целое
    try:  # пробуем превратить текст в число
        return int(v.strip())  # убрать крайние пробелы и превратить в int
    except (ValueError, AttributeError):  # текст не число или это не строка
        return None  # нецелая ячейка
def _numeric_columns(path: Path) -> dict[str, list[int]]:  # числовые колонки таблицы: имя колонки -> список целых значений
    delim = '\t' if path.suffix.lower() in ('.tsv', '.tab') else ','  # разделитель: табуляция для .tsv/.tab, иначе запятая
    cols: dict[str, list[int]] = {}  # накопитель колонок
    with path.open('r', encoding='utf-8', errors='replace', newline='') as fh:  # открыть файл; неисправимые байты заменяются
        reader = csv.reader(fh, delimiter=delim)  # читатель строк таблицы
        for row in reader:  # по строкам таблицы
            if not row:  # пустая строка
                continue  # пропустить
            vals = [_int_or_none(c) for c in row]  # в каждой ячейке: целое число или None
            if all((v is None for v in vals)):  # строка целиком нечисловая — это заголовок
                names = [c.strip() or f'c{i}' for i, c in enumerate(row)]  # имена колонок; пустое имя заменяется на c<i>
                for n in names:  # каждое имя
                    cols.setdefault(n, [])  # завести колонку, если её ещё нет
                continue  # заголовок обработан
            if not cols:  # заголовка не было — колонки ещё не заведены
                cols = {f'c{i}': [] for i in range(len(row))}  # завести колонки c0..c(m−1) по числу ячеек
            keys = list(cols.keys())  # имена колонок в порядке следования
            for i, v in enumerate(vals):  # по ячейкам строки
                if v is not None and i < len(keys):  # ячейка — целое число и её колонка существует
                    cols[keys[i]].append(v)  # добавить число в свою колонку
    return cols  # таблица колонок
def _json_numbers(obj: Any) -> list[int]:  # все целые числа дерева JSON в порядке обхода в глубину
    out: list[int] = []  # накопитель чисел

    def walk(o: Any) -> None:  # обход узла o
        if isinstance(o, bool):  # логическое значение — подкласс int в Python
            return  # его не собираем
        if isinstance(o, int):  # целое число
            out.append(o)  # добавить в накопитель
        elif isinstance(o, float) and o.is_integer():  # дробное с нулевой дробной частью
            out.append(int(o))  # добавить как целое
        elif isinstance(o, dict):  # словарь
            for v in o.values():  # по его значениям
                walk(v)  # обойти каждое
        elif isinstance(o, list):  # список
            for v in o:  # по элементам
                walk(v)  # обойти каждый
    walk(obj)  # обход всего дерева
    return out  # список целых чисел
def _pair_tasks(xs: list[int], ys: list[int], name: str) -> list[TaskSpec]:  # задача f(x) = y по парам (xs[i], ys[i]); [] , если функция неоднозначна или пар мало
    """Тест на точную функцию: каждому x должен соответствовать только один y."""
    mapping: dict[int, int] = {}  # таблица учителя x -> y
    for x, y in zip(xs, ys):  # по парам (x, y)
        if x in mapping and mapping[x] != y:  # x уже встречался с другим ответом
            return []  # функция неоднозначна — задачи нет
        mapping[x] = y  # записать (или подтвердить) ответ y для x
    if len(mapping) < 4:  # различных аргументов меньше 4
        return []  # слишком мало данных
    dom = tuple(sorted(mapping))  # область определения — сортированные ключи таблицы

    def teacher(*args, _m=mapping) -> int:  # функция учителя: ответ по первому аргументу
        v = _m.get(int(args[0]))  # ответ y для x = int(args[0]) или None
        if v is None:  # x вне наблюдённых данных
            raise ValueError('outside ingested domain')  # ошибка: вне загруженной области
        return v  # табличный ответ
    return [TaskSpec(name, 1, (dom,), teacher, family='ingest')]  # одна задача: арность 1, область dom, семейство ingest
def tasks_from_numeric_file(path: Path, max_tasks: int=16) -> list[TaskSpec]:  # не более max_tasks числовых задач из файла
    tasks: list[TaskSpec] = []  # накопитель задач
    if path.suffix.lower() == '.json':  # файл JSON
        nums = _json_numbers(json.loads(path.read_text(encoding='utf-8', errors='replace')))  # все целые числа файла
        if len(nums) >= 8:  # чисел достаточно
            tasks += _pair_tasks(nums[::2][:512], nums[1::2][:512], f'INGEST_{path.stem}_json_pairs')  # задача: чётные позиции (до 512) -> нечётные позиции (до 512)
    else:  # табличный файл
        cols = _numeric_columns(path)  # числовые колонки
        names = [n for n in cols if len(cols[n]) >= 8]  # колонки, где не меньше 8 чисел
        for i in range(len(names)):  # колонка аргумента X
            for j in range(len(names)):  # колонка ответа Y
                if i == j or len(tasks) >= max_tasks:  # та же колонка или лимит задач исчерпан
                    continue  # пропустить
                tasks += _pair_tasks(cols[names[i]], cols[names[j]], f'INGEST_{path.stem}_{names[i]}_to_{names[j]}')  # задача X -> Y; [] , если неоднозначно
    return tasks  # список задач
def tasks_from_text_file(path: Path) -> list[TaskSpec]:  # задача «следующий байт» из файла
    data = path.read_bytes()  # сырые байты файла
    nxt: dict[int, int] = {}  # кандидат таблицы: байт a -> следующий за ним байт b
    conflict: set[int] = set()  # байты, за которыми встречались разные байты
    for a, b in zip(data, data[1:]):  # по соседним парам байтов (data[i], data[i+1])
        if a in nxt and nxt[a] != b:  # за a уже встречался другой байт
            conflict.add(a)  # a — конфликтный: a не определяет b однозначно
        else:  # a новый или b совпадает с прежним
            nxt[a] = b  # записать a -> b
    for a in conflict:  # по конфликтным байтам
        nxt.pop(a, None)  # убрать их из таблицы (нет записи — не ошибка)
    if len(nxt) < 8:  # определённых пар меньше 8
        return []  # задачи нет
    dom = tuple(sorted(nxt))  # область определения — сортированные байты-аргументы

    def teacher(*args, _m=nxt) -> int:  # функция учителя: следующий байт
        v = _m.get(int(args[0]))  # следующий байт для x = int(args[0]) или None
        if v is None:  # x вне определённого подмножества
            raise ValueError('outside deterministic subset')  # ошибка: вне однозначной части
        return v  # следующий байт
    return [TaskSpec(f'INGEST_{path.stem}_nextbyte', 1, (dom,), teacher, family='ingest')]  # одна задача арности 1 с областью dom
def ingest_file(path: Path) -> list[TaskSpec]:  # все задачи, извлекаемые из одного файла
    if path.suffix.lower() == '.json':  # файл JSON
        return tasks_from_numeric_file(path)  # только числовые пары
    if path.suffix.lower() in ('.csv', '.tsv', '.tab'):  # табличный файл
        return tasks_from_numeric_file(path)  # только пары колонок
    tasks = tasks_from_text_file(path)  # прочий файл: задача «следующий байт»
    numeric = tasks_from_numeric_file(path)  # и попытка прочитать его как таблицу
    return tasks + numeric  # объединение обоих списков
def learn_from_data(paths: list[Path], budget: int=200000) -> dict[str, Any]:  # выучить задачи всех файлов; budget — бюджет узлов процессора
    proc = GrowingHagiProcessor(node_budget=budget)  # процессор с заданным бюджетом узлов
    report: dict[str, Any] = {'files': [], 'tasks': 0, 'committed': 0}  # отчёт: список файлов, число задач, число закреплённых
    for path in paths:  # по файлам
        try:  # извлечение может упасть на битом файле
            tasks = ingest_file(path)  # задачи файла
        except Exception as e:  # ошибка извлечения
            report['files'].append({'file': str(path), 'error': repr(e)[:120]})  # записать файл с текстом ошибки (первые 120 знаков)
            continue  # следующий файл
        f_info: dict[str, Any] = {'file': str(path), 'tasks': len(tasks), 'results': []}  # сведения о файле
        for sp in tasks:  # по задачам файла
            ev = proc.learn(sp, seed=10000, train_neural=True)  # выучить задачу; ev — событие результата (action, reason, program)
            f_info['results'].append({'task': sp.name, 'action': ev['action'], 'reason': ev.get('reason', '')[:40], 'program': ev.get('program', '')[:80]})  # строка результата: имя, действие, причина (40 знаков), программа (80 знаков)
            report['tasks'] += 1  # всего задач: +1
            report['committed'] += int(ev['action'] == 'COMMIT')  # закреплённых: +1, если действие COMMIT, иначе +0
        report['files'].append(f_info)  # добавить файл в отчёт
    return report  # итоговый отчёт
def ingest_main() -> None:  # командная строка: выучить точные программы из файлов данных
    p = argparse.ArgumentParser(description='Learn exact programs from any data')  # разборщик аргументов
    p.add_argument('inputs', nargs='+', type=Path)  # обязательные позиции: файлы или каталоги
    p.add_argument('--budget', type=int, default=200000)  # бюджет узлов, по умолчанию 200000
    p.add_argument('--json', action='store_true')  # флаг: напечатать отчёт в JSON
    a = p.parse_args()  # разобрать аргументы командной строки
    paths: list[Path] = []  # список файлов для учёбы
    for i in a.inputs:  # по аргументам-входам
        paths.extend(sorted(i.iterdir()) if i.is_dir() else [i])  # каталог раскрывается в сортированный список его записей, файл добавляется как есть
    rep = learn_from_data(paths, budget=a.budget)  # выучить и получить отчёт
    if a.json:  # запрошен JSON
        print(json.dumps(rep, indent=2, ensure_ascii=False))  # напечатать отчёт с отступами и без экранирования не-латиницы
    else:  # человеческий формат
        for f in rep['files']:  # по файлам отчёта
            print(f"{f['file']}: {f.get('tasks', '?')} tasks")  # имя файла и число задач (или ?, если была ошибка)
            for r in f.get('results', []):  # по результатам файла
                print(f"  {r['task']}: {r['action']} ({r['reason']})")  # задача, действие, причина
        print(f"total: {rep['committed']}/{rep['tasks']} committed")  # итого: закреплено из всех задач

# ==============================================================================
# ИСПОЛНЕНИЕ: запуск написанных программ
# ==============================================================================

import importlib.util  # загрузка модуля Python из файла по пути
from hagi.base.util import CKPT_DIR as _CKPT  # каталог чекпойнтов
DEFAULT_PROGRAMS = _CKPT / 'self' / 'generated_programs.py'  # файл с программами, которые написала модель
_mod = None  # загруженный модуль программ (пока не загружен)
_load_path: Path | None = None  # путь, из которого загружен модуль
def _load(path: Path | None=None):  # загрузить модуль программ с пути path (по умолчанию DEFAULT_PROGRAMS); повторно не читать, если путь тот же
    global _mod, _load_path  # изменяются переменные уровня модуля
    if _mod is not None and (path or DEFAULT_PROGRAMS) == _load_path:  # модуль уже загружен и путь совпадает
        return _mod  # вернуть сохранённый модуль
    p = path or DEFAULT_PROGRAMS  # путь к файлу: заданный или по умолчанию
    _load_path = p  # запомнить путь
    if not p.exists():  # файла нет
        raise FileNotFoundError(f'{p} not found — run hagi_selfrewrite first so the model writes its programs')  # ошибка: модели ещё не написаны программы
    spec = importlib.util.spec_from_file_location('hagi_generated', p)  # описание модуля hagi_generated из файла p
    if spec is None or spec.loader is None:  # спецификацию или загрузчик получить не удалось
        raise ImportError(f'cannot load {p}')  # ошибка загрузки
    mod = importlib.util.module_from_spec(spec)  # создать пустой объект модуля
    spec.loader.exec_module(mod)  # исполнить файл — заполнить модуль
    _mod = mod  # сохранить модуль
    return mod  # вернуть модуль
def available(path: Path | None=None) -> list[str]:  # имена открытых программ модуля (без начинающихся с подчёркивания и без имени hagi_generated)
    return [n for n in dir(_load(path)) if not n.startswith('_') and n != 'hagi_generated']  # фильтр имён: n не начинается с «_» и n ≠ hagi_generated
def execute(name: str, *args: int, programs: Path | None=None) -> int:  # вызвать программу name с целыми аргументами args; результат приводится к int
    fn = getattr(_load(programs), name, None)  # функция name из модуля программ или None
    if fn is None:  # программы с таким именем нет
        raise KeyError(f'no program named {name!r} (model has {len(available())})')  # ошибка: нет программы; в скобках — сколько программ есть
    return int(fn(*args))  # результат программы, приведённый к int
def execute_text(name: str, text: str, key: int | None=None) -> str:  # прогнать UTF-8 текст побайтно через программу name
    """Потоково пропускает UTF-8 текст побайтно через программу байтового уровня."""
    data = text.encode('utf-8')  # байты текста в кодировке UTF-8
    arity = 2 if key is not None else 1  # число аргументов программы: 2, если задан ключ key, иначе 1
    out = bytearray()  # накопитель выходных байтов
    for b in data:  # по байтам текста
        out.append(execute(name, b, int(key or 0)) & 255 if arity == 2 else execute(name, b) & 255)  # выходной байт = f(b, key) & 255 при арности 2, иначе f(b) & 255
    return out.decode('utf-8', errors='replace')  # собрать текст; повреждённые последовательности заменяются знаком �
def execute_main() -> None:  # командная строка: исполнить написанную моделью программу
    p = argparse.ArgumentParser(description='Execute programs the HAGI model wrote')  # разборщик аргументов
    p.add_argument('--programs', type=Path, default=DEFAULT_PROGRAMS)  # файл программ, по умолчанию DEFAULT_PROGRAMS
    p.add_argument('--list', action='store_true')  # флаг: только перечислить программы
    p.add_argument('--task')  # имя программы для запуска
    p.add_argument('--text')  # текст для побайтной обработки
    p.add_argument('--key', type=int, default=None)  # второй аргумент программы (ключ), по умолчанию нет
    p.add_argument('values', nargs='*', type=int, help='direct integer arguments, e.g. --task GCD 12 18')  # позиционные целые аргументы программы
    a = p.parse_args()  # разобрать аргументы командной строки
    if a.list:  # запрошен только список
        names = available(a.programs)  # имена программ
        print(f'{len(names)} programs:')  # заголовок с числом программ
        for n in sorted(names):  # по именам по алфавиту
            print(' ', n)  # напечатать имя
        return  # конец
    if not a.task:  # имя программы не задано
        p.error('--task or --list required')  # ошибка командной строки
    if a.text is not None:  # задан текст
        print(execute_text(a.task, a.text, a.key))  # напечатать обработанный текст
    else:  # заданы целые аргументы
        print(execute(a.task, *a.values, programs=a.programs))  # напечатать результат программы
if __name__ == '__main__':  # файл запущен как скрипт
    execute_main()  # разбор аргументов и исполнение
