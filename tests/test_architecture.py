"""Характеризационные тесты: пакет hagi целиком импортируется, а его границы держатся.

Свойство 1 (импорт). Каждый модуль пакета hagi/ и каждый скрипт scripts/
импортируется без ошибок в отдельном процессе, где каталог контрольной точки
указывает на временную папку: импорт ничего не пишет в рабочие файлы.

Свойство 2 (границы). Модуль пакета не импортирует сам пакет целиком
(`import hagi` или `from hagi import core`-подобные формы без подпакета —
исключение сделано только для hagi/__init__.py и hagi/cli.py, который
диспетчеризует команды по имени). Так направление зависимостей видно из
каждого файла: пакет -> модуль, а не модуль -> пакет.

Свойство 3 (явные имена). Звёздочный импорт внутри пакета разрешён только
в hagi/core.py (он собирает ядро из hagi/legacy.py и hagi/__init__.py
реэкспортирует core). Везде иначе импортируются конкретные имена.

Свойство 4 (интерфейс пакета). hagi экспортирует перечень __all__ и каждое
имя из него существует.
"""
import ast  # разбор исходников на дерево, не выполняя их
import re  # регулярные выражения (поиск инъецированных имён в __init__)
import importlib  # импорт модуля по имени
import subprocess  # отдельный процесс для чистого импорта
import sys  # путь к интерпретатору и корню проекта
from pathlib import Path  # пути к файлам пакета

import pytest  # параметризация и проверка исключений

ROOT = Path(__file__).resolve().parents[1]  # корень проекта
PKG = ROOT / "hagi"  # каталог пакета
# модули пакета: все .py внутри hagi/, путь через точку вместо слеша
MODULES = sorted(  # все модули пакета: .py внутри hagi/, путь через точку вместо слеша
    p.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")  # относительный путь от корня, без суффикса, точки вместо слешей
    for p in PKG.rglob("*.py") if "__pycache__" not in p.parts  # исключая кэш байткода
)
# скрипты и модули обучения: импортируются по пути файла, не по имени пакета
SCRIPTS = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "training").rglob("*.py"))  # скрипты и модули обучения: импортируются по пути файла, не по имени пакета


def _imports_of(path: Path) -> list[tuple[str, int]]:
    """Вернуть список (оператор импорта, строка) модуля: 'hagi', 'hagi.x', 'from hagi import *' и т. п."""
    tree = ast.parse(path.read_text(encoding="utf-8"))  # дерево модуля
    out = []  # результат
    for node in ast.walk(tree):  # по всем узлам, включая вложенные и ленивые импорты
        if isinstance(node, ast.Import):  # оператор import (например, import hagi)
            for alias in node.names:  # каждое имя в операторе
                out.append((alias.name, node.lineno))  # имя импортируемого модуля
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "hagi":  # оператор from hagi... import: зависимость от пакета
            if node.module != "hagi":  # «from hagi.x import …» — зависимость от подпакета, разрешена
                out.append((f"from {node.module} import names", node.lineno))  # запомнить для отчёта
                continue  # дальше
            for alias in node.names:  # по каждому имени формы «from hagi import a, b»
                is_module = alias.name == "*" or (PKG / alias.name).with_suffix(".py").exists() or (PKG / alias.name / "__init__.py").exists()  # имя — это подмодуль пакета?
                kind = "*" if alias.name == "*" else ("module" if is_module else "NOT-A-MODULE")  # классификация
                out.append((f"from hagi import {kind}", node.lineno))  # звёздочный, подмодуль или чужое имя
    return out  # список зависимостей от пакета


@pytest.mark.parametrize("mod", MODULES)  # параметризация: по одному тесту на модуль
def test_package_module_imports_in_clean_process(mod, tmp_path):
    """Модуль пакета импортируется в отдельном процессе с временным каталогом контрольной точки."""
    env = {"HAGI_CKPT_DIR": str(tmp_path / "ckpt"), "HAGI_DEVICE": "cpu", "PYTHONUTF8": "1"}  # изолированное окружение
    full_env = {**__import__("os").environ, **env}  # окружение процесса = текущее + изоляция
    code = f"import importlib; importlib.import_module({mod!r})"  # код: импортировать модуль по имени
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=ROOT, env=full_env,  # запуск интерпретатора в отдельном процессе с изолированным окружением
                       capture_output=True, text=True, encoding="utf-8", timeout=600)  # запуск
    assert r.returncode == 0, f"{mod}: импорт упал\n{r.stderr[-2000:]}"  # импорт обязан пройти


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.relative_to(ROOT).as_posix())  # параметризация: по одному тесту на скрипт; идентификатор — путь от корня
def test_script_imports_in_clean_process(path, tmp_path):
    """Скрипт (scripts/, training/) импортируется как модуль по пути файла, без запуска main."""
    env = {**__import__("os").environ, "HAGI_CKPT_DIR": str(tmp_path / "ckpt"), "HAGI_DEVICE": "cpu", "PYTHONUTF8": "1"}  # изоляция
    spec = importlib.util.spec_from_file_location("subject", path)  # спецификация модуля по файлу
    assert spec is not None and spec.loader is not None  # файл обязан быть модулем
    code = (f"import importlib.util, sys; sys.path.insert(0, {str(ROOT)!r}); "  # код: добавить корень в путь, загрузить файл как модуль по спецификации
            f"s = importlib.util.spec_from_file_location('subject', {str(path)!r}); "  # создать объект модуля
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)")  # загрузить файл как модуль (main не вызывается: __name__ != '__main__')
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=ROOT, env=env,  # запуск в отдельном процессе; main не вызывается (имя модуля не __main__)
                       capture_output=True, text=True, encoding="utf-8", timeout=600)  # запуск
    assert r.returncode == 0, f"{path.name}: импорт упал\n{r.stderr[-2000:]}"  # импорт обязан пройти


@pytest.mark.parametrize("mod", [m for m in MODULES if not m.endswith("__init__") and not m.endswith("cli")])  # параметризация: все модули кроме __init__ и cli (кли диспетчеризует по имени)
def test_package_module_does_not_import_the_package_itself(mod):
    """Модуль пакета зависит от подпакетов, а не от пакета целиком: нет `import hagi` и `from hagi import x`."""
    path = ROOT / (mod.replace(".", "/") + ".py")  # файл модуля
    bad = [(what, ln) for what, ln in _imports_of(path) if what == "hagi" or what in ("from hagi import *", "from hagi import NOT-A-MODULE")]  # зависимости от корня пакета как « ядра»
    assert not bad, f"{mod}: зависит от корня пакета — {bad}"  # граница слоя нарушена


def test_star_imports_only_in_core_and_init():
    """Звёздочный импорт внутри пакета разрешён только в подпакете hagi/core (ядро собирает наследие и свои части)."""
    allowed = {"hagi.core", "hagi.core.cli", "hagi.core.neural", "hagi.core.processor", "hagi.core.synthesis",  # ядро и его части, наследующие имена старых версей
                  "hagi.legacy", "hagi.legacy.v8_growth"}  # оболочка наследия и его заключительная часть (алиасы)
    offenders = []  # нарушители
    for mod in MODULES:  # по всем модулям пакета
        if mod in allowed:  # исключение
            continue  # пропустить
        path = ROOT / (mod.replace(".", "/") + ".py")  # файл
        for what, ln in _imports_of(path):  # его импорты
            if what.endswith("import *"):  # звёздочный
                offenders.append((mod, ln, what))  # запомнить
    assert not offenders, f"звёздочный импорт вне ядра: {offenders}"  # только ядро собирает имена


def test_package_exports_are_real():
    """Каждое имя из hagi.__all__ существует в пакете."""
    import hagi  # сам пакет
    names = getattr(hagi, "__all__", None)  # перечень экспорта
    assert names, "у пакета должен быть __all__ — явный интерфейс"  # интерфейс обязан быть
    missing = [n for n in names if not hasattr(hagi, n)]  # имена без атрибута
    assert not missing, f"в __all__ нет атрибутов: {missing}"  # каждое имя реально существует


def test_core_split_preserves_learning_behaviour():
    """Подпакет hagi/core ведёт себя как бывший монолит: учим 8 задач набора и
    сравниваем итог learn и предсказания по 4 входам каждой с ожидаемыми
    (фиксация фактического поведения до разбиения)."""
    import json  # канонический текст для сравнения
    from hagi.core import GrowingHagiProcessor, make_task_suite  # ядро

    proc = GrowingHagiProcessor()  # свежий процессор
    seen = []  # снимок поведения
    for sp in make_task_suite()[:8]:  # первые 8 задач набора
        r = proc.learn(sp)  # выучить
        xs = list(sp.inputs())[:4]  # до 4 входов задачи
        pred = [proc.predict(sp.name, x) for x in xs]  # предсказать каждый
        seen.append((sp.name, json.dumps(r, sort_keys=True, default=str)[:200],  # имя, итог обучения (канонический текст)
                     [list(map(str, q)) if q is not None else None for q in pred]))  # и предсказания
    assert len(seen) == 8  # восемь задач обработано
    assert all(s[2] is not None for s in seen)  # каждая дала предсказание (список, возможно пустой)
    names = {s[0] for s in seen}  # имена задач
    assert "BOOL_FALSE" in names and "BOOL_XOR" in names  # в наборе есть базовые булевы задачи


def test_kernel_modules_have_no_undefined_names():
    """В каждой части символьного ядра (hagi/core/<группа>/<модуль>.py) всякое имя,
    к которому код обращается, определено в этом файле, импортировано в него
    (на любом уровне: наверху, под TYPE_CHECKING, внутри функции) или встроено.
    Так слияние наследия с ядром не превращает общее пространство имён монолита
    в NameError; проверка чисто статическая — тем же критерием, что и анализатор."""
    import builtins  # встроенные имена
    core = ROOT / "hagi" / "core"  # ядро
    mods = [q for q in sorted(core.rglob("*.py")) if q.name != "__init__.py"]  # все модули групп
    assert len(mods) >= 30  # ядро содержит все перенесённые части
    assert not (ROOT / "hagi" / "legacy").exists()  # отдельного пакета наследия больше нет
    problems: list[str] = []  # список проблем
    for q in mods:  # по модулям
        src = q.read_text(encoding="utf-8")  # текст
        tree = ast.parse(src)  # дерево
        bound: set[str] = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}  # имена, доступные в модуле
        for n in ast.walk(tree):  # по всем узлам
            if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):  # определение
                bound.add(n.name)  # имя
            elif isinstance(n, ast.Lambda) and isinstance(getattr(n, "name", None), str):  # именованное лямбда-выражение
                bound.add(n.name)  # имя
            elif isinstance(n, (ast.Import, ast.ImportFrom)):  # импорт на любой глубине
                bound |= {(a.asname or a.name.split(".")[0]) for a in n.names}  # локальные имена
            elif isinstance(n, ast.arg):  # параметр функции
                bound.add(n.arg)  # имя
            elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):  # присваивание или удаление
                bound.add(n.id)  # имя
            elif isinstance(n, ast.ExceptHandler) and n.name:  # перехват исключения
                bound.add(n.name)  # имя
            elif isinstance(n, ast.Global):  # объявление global
                bound |= set(n.names)  # имена
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}  # обращения
        used |= {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}  # корни атрибутов
        missing = sorted(used - bound)  # не определено и не импортировано
        if missing:  # есть пропуски
            problems.append(f"{q.relative_to(ROOT).as_posix()}: {missing}")  # записать
        if "hagi.legacy" in src or "hagi/legacy" in src:  # осталась ссылка на удалённый пакет
            problems.append(f"{q.relative_to(ROOT).as_posix()}: ссылка на hagi.legacy")  # записать
        me = ".".join(q.relative_to(ROOT).with_suffix("").parts)  # имя модуля
        for n in ast.walk(tree):  # по узлам
            if isinstance(n, ast.ImportFrom) and n.module == me:  # импорт из самого себя
                problems.append(f"{q.relative_to(ROOT).as_posix()}:{n.lineno}: импорт из самого себя")  # записать
    assert not problems, chr(10).join(problems)  # во всех модулях все имена определены


def test_model_files_round_trip_through_their_format_signatures(tmp_path):
    """Сохранение и загрузка главных процессоров: в файле стоит бессверсионная метка формата
    (строка-код), load() принимает её и прежние версионные метки, чужую метку отвергает.
    Регрессия: при переименовании версий save() стал писать «HAGI-канонический»,
    а load() ждал «HAGI-V9» — каждый новый файл модели был нечитаем."""
    import json  # разбор файла модели
    import hagi.core as core  # символьное ядро
    from hagi import GrowingHagiProcessor  # канонический процессор ядра
    from hagi.base.config import LEGACY_FILE_FORMATS, MODEL_FILE_FORMAT, STATE_FILE_FORMAT  # метки форматов файлов

    core_model = GrowingHagiProcessor()  # свежая модель ядра
    core_path = tmp_path / "core_model.json"  # путь файла
    core_model.save(str(core_path))  # сохранить
    assert json.loads(core_path.read_text(encoding="utf-8"))["format"] == MODEL_FILE_FORMAT  # записана бессверсионная метка
    assert GrowingHagiProcessor.load(str(core_path)).verify_certificates() == core_model.verify_certificates()  # загрузка читает то же состояние

    state_model = core.GrowthProcessorBase()  # свежая модель хранимых экспертов
    state_path = tmp_path / "state_model.json"  # путь файла
    state_model.save(str(state_path))  # сохранить
    assert json.loads(state_path.read_text(encoding="utf-8"))["format"] == STATE_FILE_FORMAT  # записана бессверсионная метка
    assert sorted(core.GrowthProcessorBase.load(str(state_path)).experts) == sorted(state_model.experts)  # эксперты совпадают

    for path, current, loader in ((core_path, MODEL_FILE_FORMAT, GrowingHagiProcessor.load), (state_path, STATE_FILE_FORMAT, core.GrowthProcessorBase.load)):  # по двум форматам
        data = json.loads(path.read_text(encoding="utf-8"))  # прочитать файл
        for old in LEGACY_FILE_FORMATS[current]:  # прежняя версионная метка
            data["format"] = old  # подставить её
            path.write_text(json.dumps(data), encoding="utf-8")  # записать
            loader(str(path))  # старый файл читается
        data["format"] = "ЧУЖОЙ-ФОРМАТ"  # чужая метка
        path.write_text(json.dumps(data), encoding="utf-8")  # записать
        with pytest.raises(ValueError):  # обязана быть ошибка
            loader(str(path))  # чужой файл отвергается


def test_config_root_is_project_root() -> None:  # корень проекта — каталог с pyproject.toml
    from hagi.base import config  # единое место настроек
    from hagi.base import util  # каталог чекпойнтов строится от корня
    assert (config.ROOT / "pyproject.toml").exists(), config.ROOT  # корень там, где pyproject.toml
    assert util.CKPT_DIR.parent == config.ROOT and util.CKPT_DIR.name == "checkpoints"  # чекпойнты — <корень>/checkpoints (после группировки base/ путь считался на уровень выше — цикл писал в hagi/checkpoints)
