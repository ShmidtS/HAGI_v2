"""Символьное ядро обучения экспертов: классы StoredExpert, StoredTheoremMiner и GrowthProcessorBase.

Идея: для каждой задачи подбирается конечная символьная программа (дерево
выражения над переменными, константами, полиномами и вызовами уже изученных
экспертов). Программа допускается в библиотеку только после точной проверки на
ВСЕЙ конечной области определения задачи: для каждого входа xs выполняется
program(xs) == teacher(xs).

Основные формулы и величины:
  * разбиение выборки: случайная перестановка perm; обучение — первые n_train
    точек, проверка — остальные; n_train = round(train_fraction·|D|), ограничено
    1..|D|−1 (для булевых задач обучение оставляет одну точку на проверку);
  * длина кода программы: bits = TOKEN_BITS + сумма длин поддеревьев (по правилу
    CallDagProgram.bits); ссылка на эксперта номер j стоит gamma_bits(j);
  * проверка графа вызовов: обход в глубину, цикл найден, когда вершина
    встречается на текущем пути; граф ацикличен ⟺ нет повторной вершины на пути;
  * точность: accuracy = correct / max(1, checked).

Классы:
  StoredExpert      — запись об изученной операции (программа, области аргументов, длина кода);
  StoredTheoremMiner — перебор экспертов на алгебраические свойства (коммутативность,
                  инволютивность, идемпотентность, чётная/нечётная симметрия);
  GrowthProcessorBase — растущая библиотека экспертов: learn, predict, save, load.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов
import itertools, json, os  # модули стандартной библиотеки: аргументы командной строки, абстрактное дерево, копирование, куча, переборы, JSON, математика, ОС, случайность
from dataclasses import dataclass  # декоратор класса-записи и описание поля по умолчанию
from pathlib import Path  # объектный путь к файлу
from typing import Any, Optional, Sequence  # типы аннотаций: произвольный, вызываемый, перечислимый, необязательный, последовательность
import numpy as np  # численные массивы
from typing import Any, Optional, Sequence  # имена из «stdlib:typing» (модуль, где они определены)
from hagi.core.language.call_dag import CallDagExpr, CallDagProgram  # имена из «call_dag» (модуль, где они определены)
from hagi.core.memory.aliases import TOKEN_BITS, gamma_bits  # имена из «aliases» (модуль, где они определены)
from hagi.core.synthesis.venn import anf_to_call_dag, anf_transform  # выражение по коэффициентам Жегалкина и само преобразование
from hagi.core.synthesis.venv_anf import venn_head_eval  # вычисление булевой функции по маске диаграмм (хозяин — модуль ANF)
from hagi.core.neural.certified_ce import (  # импорт из модуля нейронного остатка
    FactorizedAdapter,  # малый нейронный адаптер одного эксперта
    FactorizedSharedCore  # общий нейронный core, разделяемый всеми адаптерами
)  # конец импорта
from hagi.core.neural.chunking import (  # импорт из модуля адаптивной сегментации
    GDNRAM,  # растущая адресная память
    HedgeRouter  # маршрутизатор с экспоненциальными весами
)  # конец импорта
from hagi.core.memory.ledger import (  # импорт из модуля учёта статистического бюджета
    AuditLedger  # журнал распределения вероятности ошибки между проверками
)  # конец импорта
from hagi.core.synthesis.fast_mdl import (  # импорт из модуля перебора программ по принципу минимальной длины
    bounded_fold_hypotheses,  # ограниченные свёртки (кванторы по конечным диапазонам)
    exact_piecewise_poly_fit,  # точная кусочно-полиномиальная подгонка
    synthesize_batched_mdl_stored  # пакетный перебор выражений с вызовами библиотеки
)  # конец импорта
from hagi.base.config import STATE_FILE_FORMAT, LEGACY_FILE_FORMATS, STATE_FORMAT  # единая метка формата состояния и перечень прежних меток
from hagi.core.synthesis.best_first import SynthesisStats  # имена из модуля «best_first» (прямой импорт вместо обращения через псевдоним пакета)
from hagi.core.synthesis.induction import Theorem, _exact_multivariate_poly_fit, _exact_poly_fit  # имена из модуля «induction» (прямой импорт вместо обращения через псевдоним пакета)
from hagi.core.processor.processor import _expert_to_dict, _expert_total_neural_bytes  # импорт общих функций сериализации эксперта и подсчёта размера нейронной части
from hagi.core.processor.experts import GrowthEvent, expert_supports  # импорт GrowthEvent и общей проверки применимости эксперта
from hagi.core.tasks.stored_tasks import TaskSpec  # имена из модуля «stored_tasks» (прямой импорт вместо обращения через псевдоним пакета)
from hagi.core.tasks.candidates import candidate_programs  # имена из модуля «candidates» (прямой импорт вместо обращения через псевдоним пакета)

# ==============================================================================
# Символьные эксперты, поиск теорем и растущий процессор
# ==============================================================================


@dataclass  # класс-запись: конструктор и методы-заглушки генерируются автоматически
class StoredExpert:
    """Изученная операция: символьная программа и условия её применимости.

        Поля:
        name        — имя операции (совпадает с именем задачи);
        family      — семейство задачи (например, "BoolVenn");
        program     — программа CallDagProgram, вычисляющая выход по входам;
        domains     — кортеж допустимых значений каждого аргумента;
        valid_rule  — дополнительное правило области: "all" (нет), "nonzero_y"
                      (второй аргумент не 0), "ordered_yz" (x1 <= x2);
        code_length — длина кода программы в битах;
        adapter     — необязательный нейронный адаптер для приближённого ответа;
        parents     — имена экспертов, из которых эта операция была выращена.

        Применимость: supports(xs) истинно, когда число аргументов совпадает,
        каждый xs[i] лежит в domains[i] и выполнено правило valid_rule.
    """
    name: str  # имя операции
    family: str  # семейство задачи
    program: CallDagProgram  # символьная программа операции
    domains: tuple[tuple[int, ...], ...]  # допустимые значения каждого аргумента
    valid_rule: str  # имя правила дополнительного ограничения области
    code_length: int  # длина кода программы в битах
    adapter: Optional[FactorizedAdapter] = None  # нейронный адаптер (по умолчанию отсутствует)
    parents: tuple[str, ...] = ()  # кортеж имён экспертов-родителей (по умолчанию пуст)

    def supports(self, xs: Sequence[int]) -> bool:  # xs — вход
        """Проверить, применима ли операция к входу xs.

            xs — кортеж целых аргументов. Возвращает True, когда
            len(xs) == len(domains), xs[i] ∈ domains[i] для всех i и выполнено
            правило valid_rule; иначе False."""
        return expert_supports(self, xs)  # проверить применимость эксперта к аргументам

    def to_dict(self) -> dict[str, Any]:
        """Сериализация записи в словарь JSON-совместимых значений (кортежи -> списки)."""
        return _expert_to_dict(self)  # собрать словарь с полями эксперта

    @staticmethod  # метод без доступа к экземпляру — конструктор из словаря
    def from_dict(d: dict[str, Any]) -> "StoredExpert":  # d — словарь из to_dict
        """Восстановить запись из словаря d, полученного to_dict; возвращает StoredExpert."""
        return StoredExpert(  # конструктор записи по полям словаря
            str(d["name"]),  # имя как строка
            str(d["family"]),  # семейство как строка
            CallDagProgram.from_dict(d["program"]),  # восстановленная программа
            tuple(tuple(int(v) for v in row) for row in d["domains"]),  # области: каждый список -> кортеж целых
            str(d.get("valid_rule", "all")),  # правило области, по умолчанию "all"
            int(d["code_length"]),  # длина кода
            FactorizedAdapter.from_dict(d["adapter"]) if d.get("adapter") else None,  # адаптер, если он был сохранён, иначе None
            tuple(str(x) for x in d.get("parents", [])),  # родители кортежем строк
        )


class StoredTheoremMiner:
    """Поиск алгебраических свойств среди всех экспертов библиотеки.

        Перебирается каждый эксперт (без усечения списка). Для бинарных операций
        проверяются: коммутативность f(x,y) = f(y,x); самопоглощение f(x,x) = x;
        правый ноль f(x,0) = x. Для унарных: замкнутость образа, инволютивность
        f(f(x)) = x, идемпотентность f(f(x)) = f(x), чётная f(−x) = f(x) и
        нечётная f(−x) = −f(x) симметрия. Свойство признаётся, если оно выполняется
        на всех допустимых точках. Возвращается список новых теорем."""

    def mine(self, model: "GrowthProcessorBase") -> list[Theorem]:  # model — процессор с экспертами
        """Найти и добавить в model.theorems новые теоремы; возвращает их список.

            Теорема добавляется, только если её имени ещё нет среди известных."""
        found: list[Theorem] = []  # найденные на этом запуске теоремы
        known = {t.name for t in model.theorems}  # имена уже известных теорем
        env = model.experts  # среда для вызовов: имя эксперта -> эксперт

        for e in model.experts.values():  # по всем экспертам библиотеки
            if e.program.arity == 2:  # бинарная операция
                dx, dy = e.domains[0], e.domains[1]  # области первого и второго аргумента
                declared = [(x, y) for x in dx for y in dy if e.supports((x, y))]  # все допустимые пары (x, y) ∈ dx × dy
                swapped = [(x, y) for x, y in declared if e.supports((y, x))]  # пары, у которых переставленная (y, x) тоже допустима
                if swapped and len(swapped) == len(declared):  # перестановка допустима для всех пар
                    nm = f"commutativity({e.name})"  # имя теоремы: коммутативность операции
                    if nm not in known:  # теорема ещё не известна
                        try:  # вычисление программы может завершиться ошибкой
                            if all(  # проверить все пары: f(x,y) == f(y,x)
                                e.program.eval((x, y), env)  # значение f(x, y)
                                == e.program.eval((y, x), env)  # сравнение со значением f(y, x)
                                for x, y in swapped  # по допустимым парам
                            ):  # все сравнения истинны
                                th = Theorem(  # новая теорема
                                    nm,  # имя
                                    f"for all supported pairs: {e.name}(x,y)={e.name}(y,x)",  # текст формулировки
                                    len(swapped),  # число проверенных точек
                                )
                                model.theorems.append(th)  # добавить в список теорем модели
                                found.append(th)  # добавить в результат
                                known.add(nm)  # имя теперь известно
                        except Exception:  # ошибка вычисления — свойство не подтверждено
                            pass  # пропустить
                self_pairs = [x for x in dx if x in dy and e.supports((x, x))]  # значения x, для которых пара (x, x) допустима
                nm = f"self-absorption({e.name})"  # имя теоремы: самопоглощение
                if self_pairs and len(self_pairs) == len(dx) and nm not in known:  # (x, x) допустима для всех x из dx
                    try:  # вычисление может завершиться ошибкой
                        if all(e.program.eval((x, x), env) == x for x in self_pairs):  # сравнение: f(x, x) == x для всех x
                            th = Theorem(  # новая теорема
                                nm,  # имя
                                f"for all overlapping x: {e.name}(x,x)=x",  # текст формулировки
                                len(self_pairs),  # число проверенных точек
                            )
                            model.theorems.append(th)  # добавить в теоремы модели
                            found.append(th)  # добавить в результат
                            known.add(nm)  # имя теперь известно
                    except Exception:  # ошибка вычисления
                        pass  # пропустить
                zero_pairs = [x for x in dx if 0 in dy and e.supports((x, 0))]  # x, для которых пара (x, 0) допустима
                nm = f"right-identity-zero({e.name})"  # имя теоремы: правый нейтральный ноль
                if zero_pairs and len(zero_pairs) == len(dx) and nm not in known:  # (x, 0) допустима для всех x из dx
                    try:  # вычисление может завершиться ошибкой
                        if all(e.program.eval((x, 0), env) == x for x in zero_pairs):  # сравнение: f(x, 0) == x для всех x
                            th = Theorem(  # новая теорема
                                nm, f"for all x: {e.name}(x,0)=x", len(zero_pairs)  # имя, формулировка, число точек
                            )
                            model.theorems.append(th)  # добавить в теоремы модели
                            found.append(th)  # добавить в результат
                            known.add(nm)  # имя теперь известно
                    except Exception:  # ошибка вычисления
                        pass  # пропустить
            elif e.program.arity == 1:  # унарная операция
                dom = tuple(int(x) for x in e.domains[0])  # область определения как кортеж целых
                img = {}  # образ: x -> f(x)
                closed = True  # признак замкнутости: f(x) снова в области
                try:  # вычисление может завершиться ошибкой
                    for x in dom:  # по всем x области
                        y = int(e.program.eval((x,), env))  # значение y = f(x)
                        if not e.supports((y,)):  # f(x) вне области
                            closed = False  # образ не замкнут
                            break  # прекратить перебор
                        img[x] = y  # запомнить f(x)
                except Exception:  # ошибка вычисления
                    closed = False  # образ не замкнут
                if closed:  # f переводит область в себя
                    for prop, check_fn in (  # два свойства и их проверки
                        (  # инволютивность: f(f(x)) == x
                            "involutivity",  # имя свойства
                            lambda: all(  # проверить все x: f(img[x]) == x
                                e.program.eval((img[x],), env) == x for x in dom  # сравнение: f(f(x)) == x
                            ),
                        ),
                        (  # идемпотентность: f(f(x)) == f(x)
                            "idempotence",  # имя свойства
                            lambda: all(  # проверить все x: f(img[x]) == img[x]
                                e.program.eval((img[x],), env) == img[x] for x in dom  # сравнение: f(f(x)) == f(x)
                            ),
                        ),
                    ):  # конец пар (свойство, проверка)
                        nm = f"{prop}({e.name})"  # имя теоремы: свойство(операция)
                        if nm not in known:  # теорема ещё не известна
                            try:  # проверка может завершиться ошибкой
                                if check_fn():  # свойство выполнено на всей области
                                    th = Theorem(  # новая теорема
                                        nm, f"{prop} on certified domain", len(dom)  # имя, формулировка, число точек
                                    )
                                    model.theorems.append(th)  # добавить в теоремы модели
                                    found.append(th)  # добавить в результат
                                    known.add(nm)  # имя теперь известно
                            except Exception:  # ошибка проверки
                                pass  # пропустить
                if all((-x) in dom for x in dom):  # область симметрична: −x ∈ dom для всех x
                    for sym_name, sgn in (("even-symmetry", 1), ("odd-symmetry", -1)):  # чётная: знак +1; нечётная: знак −1
                        nm = f"{sym_name}({e.name})"  # имя теоремы: вид симметрии(операция)
                        if nm not in known:  # теорема ещё не известна
                            try:  # проверка может завершиться ошибкой
                                if all(  # проверить все x: f(−x) == sgn·f(x)
                                    e.program.eval((-x,), env)  # значение f(−x)
                                    == sgn * e.program.eval((x,), env)  # равно sgn·f(x)
                                    for x in dom  # по всем x области
                                ):  # все сравнения истинны
                                    th = Theorem(  # новая теорема
                                        nm,  # имя
                                        f"{e.name}(-x)={'+' if sgn==1 else '-'}{e.name}(x)",  # текст: f(−x) = ±f(x), знак по sgn
                                        len(dom),  # число проверенных точек
                                    )
                                    model.theorems.append(th)  # добавить в теоремы модели
                                    found.append(th)  # добавить в результат
                                    known.add(nm)  # имя теперь известно
                            except Exception:  # ошибка проверки
                                pass  # пропустить
        return found  # список найденных теорем


def _expr_call_names(expr: Optional[CallDagExpr]) -> set[str]:  # expr — корень выражения или None
    """Собрать имена экспертов, вызываемых внутри выражения expr.

        Рекурсивный обход: узел с операцией "call" даёт имя expr.value, остальные
        имена берутся из детей expr.args. Возвращает множество имён."""
    if expr is None:  # выражения нет
        return set()  # пустое множество
    names = {str(expr.value)} if expr.op == "call" else set()  # {expr.value}, если узел — вызов, иначе пусто
    for child in expr.args:  # по дочерним узлам
        names.update(_expr_call_names(child))  # добавить имена, найденные в ребёнке
    return names  # множество имён вызовов


def verify_call_dag(model: "GrowthProcessorBase") -> dict[str, Any]:  # model — процессор с экспертами
    """Проверить граф вызовов между экспертами: все ссылки существуют, циклов нет.

        Вершина — эксперт, ребро a -> b, если программа a содержит вызов b.
        Обход в глубину: вершина на текущем пути означает цикл. Возвращает
        словарь: valid (нет пропавших ссылок и циклов), missing, cycles, nodes,
        edges = Σ|refs|."""
    graph: dict[str, set[str]] = {}  # граф: имя эксперта -> множество вызываемых имён
    missing: list[tuple[str, str]] = []  # пары (эксперт, отсутствующая ссылка)
    for name, expert in model.experts.items():  # по всем экспертам
        refs = _expr_call_names(expert.program.expr)  # имена, вызываемые программой
        for ref in refs:  # по вызываемым именам
            if ref not in model.experts:  # вызываемый эксперт не существует
                missing.append((name, ref))  # записать пропуск
        graph[name] = refs  # рёбра вершины name
    visiting: set[str] = set()  # вершины на текущем пути обхода
    visited: set[str] = set()  # полностью обработанные вершины
    cycles: list[str] = []  # описания найденных циклов

    def dfs(node: str, trail: tuple[str, ...]) -> None:  # node — вершина; trail — путь от корня до неё (не включая её)
        """Обход в глубину от вершины node; найденный цикл записывается в cycles."""
        if node in visited:  # вершина уже полностью обработана
            return  # ничего делать не нужно
        if node in visiting:  # вершина лежит на текущем пути — цикл
            cycles.append(" -> ".join((*trail, node)))  # записать путь с повторной вершиной в конце
            return  # выйти
        visiting.add(node)  # вершина открыта
        for child in graph.get(node, ()):  # по вызываемым именам (нет рёбер — пустой кортеж)
            if child in graph:  # ребёнок существует как вершина
                dfs(child, (*trail, node))  # обходить ребёнка с путём trail + node
        visiting.remove(node)  # вершина закрыта
        visited.add(node)  # вершина обработана полностью

    for node in graph:  # по всем вершинам
        dfs(node, ())  # обход от вершины с пустым путём
    return {  # отчёт о графе
        "valid": not missing and not cycles,  # граф корректен: нет пропусков и циклов
        "missing": missing,  # пропущенные ссылки
        "cycles": cycles,  # найденные циклы
        "nodes": len(graph),  # число вершин
        "edges": sum(len(v) for v in graph.values()),  # число рёбер = Σ|refs|
    }


def verify_growth_lineage(model: "GrowthProcessorBase") -> dict[str, Any]:  # model — процессор с экспертами
    """Проверить происхождение экспертов: родители существуют, циклов порождения нет.

        Вершина — эксперт, ребро a -> b для каждого родителя b эксперта a.
        Возвращает словарь: valid, missing, cycles, nodes_with_parents (число
        экспертов с родителями), edges = Σ|parents|."""
    graph: dict[str, tuple[str, ...]] = {}  # граф: имя эксперта -> кортеж родителей
    missing: list[tuple[str, str]] = []  # пары (эксперт, отсутствующий родитель)
    for name, expert in model.experts.items():  # по всем экспертам
        graph[name] = tuple(expert.parents)  # рёбра — родители эксперта
        for parent in expert.parents:  # по родителям
            if parent not in model.experts:  # родитель не существует
                missing.append((name, parent))  # записать пропуск
    visiting: set[str] = set()  # вершины на текущем пути обхода
    visited: set[str] = set()  # полностью обработанные вершины
    cycles: list[str] = []  # описания найденных циклов

    def dfs(node: str, trail: tuple[str, ...]) -> None:  # node — вершина; trail — пройденный путь до неё
        """Обход в глубину от вершины node; найденный цикл записывается в cycles."""
        if node in visited:  # вершина уже обработана
            return  # выйти
        if node in visiting:  # вершина на текущем пути — цикл
            cycles.append(" -> ".join((*trail, node)))  # записать путь с повторной вершиной
            return  # выйти
        visiting.add(node)  # открыть вершину
        for parent in graph.get(node, ()):  # по родителям (нет — пустой кортеж)
            if parent in graph:  # родитель существует как вершина
                dfs(parent, (*trail, node))  # обходить родителя с путём trail + node
        visiting.remove(node)  # закрыть вершину
        visited.add(node)  # вершина обработана полностью

    for node in graph:  # по всем вершинам
        dfs(node, ())  # обход с пустым путём
    return {  # отчёт о графе порождения
        "valid": not missing and not cycles,  # корректно: нет пропусков и циклов
        "missing": missing,  # пропущенные родители
        "cycles": cycles,  # найденные циклы
        "nodes_with_parents": sum(bool(v) for v in graph.values()),  # число экспертов, у которых есть родители
        "edges": sum(len(v) for v in graph.values()),  # число рёбер = Σ|parents|
    }


class GrowthProcessorBase:
    """Растущая библиотека символьных экспертов с точной проверкой на конечной области.

        learn(spec) подбирает для задачи программу и добавляет эксперта, только
        если программа совпадает с учителем на всей области; иначе событие HOLD.
        predict(name, xs) возвращает точное значение программы или приближение
        адаптером, либо отказ ABSTAIN вне области. save/load — сохранение и
        восстановление состояния в JSON."""

    def __init__(  # конструктор процессора
        self, seed: int = 42, node_budget: int = 4000, delta_total: float = 0.05  # seed — зерно общего нейронного ядра; node_budget — предел проверенных узлов при переборе; delta_total — общий бюджет вероятности ошибки
    ):
        """Создать пустой процессор: ядро, маршрутизатор, память, журнал."""
        self.core = FactorizedSharedCore(seed=seed)  # общее нейронное ядро с весами по зерну seed
        self.node_budget = int(node_budget)  # предел проверок при переборе программ
        self.experts: dict[str, StoredExpert] = {}  # библиотека экспертов: имя -> эксперт (пока пустая)
        self.version = 0  # номер версии модели — число зафиксированных экспертов
        self.total_commits = 0  # счётчик успешных добавлений
        self.total_holds = 0  # счётчик отказов
        self.growth_log: list[GrowthEvent] = []  # журнал событий роста
        self.theorems: list[Theorem] = []  # найденные теоремы
        self.ledger = AuditLedger(delta_total=delta_total)  # журнал статистического бюджета
        self.stats = SynthesisStats()  # счётчики синтеза
        self.router = HedgeRouter(4, eta=0.5)  # маршрутизатор на 4 эксперта с темпом обучения 0.5
        self.ram = GDNRAM(n_slots=32, dim=4)  # память на 32 слота размерности 4

    def _expert_index(self) -> dict[str, int]:
        """Номера экспертов в отсортированном порядке имён: имя -> позиция i от 0."""
        return {nm: i for i, nm in enumerate(sorted(self.experts))}  # словарь «имя -> номер» по отсортированному списку

    def learn(  # изучить одну задачу
        self,  # экземпляр процессора
        spec: TaskSpec,  # описание задачи: имя, семейство, области, функция-учитель
        seed: int = 0,  # зерно разбиения на обучение и проверку
        train_fraction: float = 0.70,  # доля точек на обучение
        poison_one_label: bool = False,  # имитация испорченной метки: задача будет отклонена
        train_neural: bool = True,  # обучать ли нейронный адаптер
        deadline: Optional[float] = None,  # предельное время (секунды) для перебора программ
        parents: Sequence[str] = ()  # имена экспертов-родителей
    ) -> GrowthEvent:  # аннотация возврата — событие роста
        """Подобрать для задачи spec точную программу и добавить эксперта либо отказаться.

            Если эксперт с таким именем есть — событие REUSE; неизвестные родители
            или пустая область — HOLD. Область D = spec.inputs() делится
            перестановкой на train_x (n_train точек) и val_x (остальные). Кандидаты
            перебираются по уровням: готовый эксперт, булева диаграмма, полином
            Жегалкина, шаблоны примитивов, точный полином, кусочный полином,
            ограниченные свёртки, пакетный перебор. Кандидат принимается, если
            program(xs) == teacher(xs) на train_x, val_x и всей D. Возвращает
            событие COMMIT, REUSE или HOLD."""
        if spec.name in self.experts:  # эксперт с таким именем уже изучен
            ev = GrowthEvent(self.version, spec.name, "REUSE", "already learned")  # событие повторного использования
            self.growth_log.append(ev)  # записать в журнал
            return ev  # вернуть событие
        parent_names = tuple(dict.fromkeys(str(x) for x in parents))  # родители строками без повторов (порядок сохранён)
        unknown_parents = [x for x in parent_names if x not in self.experts]  # родители, которых нет в библиотеке
        if unknown_parents:  # есть неизвестные родители
            return self._hold(  # отказ
                spec, f"UNKNOWN_GROWTH_PARENTS:{','.join(unknown_parents)}", 0, 0  # причина — список неизвестных родителей
            )
        domain = spec.inputs()  # конечная область D — все допустимые входы задачи
        if not domain:  # область пуста
            return self._hold(spec, "EMPTY_VALID_DOMAIN", 0, 0)  # отказ: пустая область
        rng = np.random.default_rng(seed)  # генератор случайных чисел по зерну seed
        perm = rng.permutation(len(domain))  # случайная перестановка номеров 0..|D|−1
        if spec.family == "BoolVenn" or (  # булева задача или
            all(set(d) <= {0, 1} for d in spec.domains) and len(domain) <= 16  # все области ⊆ {0,1} и |D| <= 16
        ):  # конец условия
            n_train = max(1, len(domain) - 1)  # на обучение всё, кроме одной точки: |D| − 1 (не меньше 1)
        else:  # иначе
            n_train = max(  # n_train не меньше 1
                1, min(len(domain) - 1, int(round(len(domain) * train_fraction)))  # минимум из |D| − 1 и round(train_fraction·|D|)
            )
        train_x = [domain[int(i)] for i in perm[:n_train]]  # обучающие входы — первые n_train точек перестановки
        val_x = [domain[int(i)] for i in perm[n_train:]]  # проверочные входы — остальные
        all_y = {xs: int(spec.teacher(*xs)) for xs in domain}  # ответы учителя на всей области: teacher(xs)
        train_y = [all_y[xs] for xs in train_x]  # метки обучения
        val_y = [all_y[xs] for xs in val_x]  # метки проверки
        if poison_one_label and train_y:  # включена имитация испорченной метки
            self.ledger.record_reject(spec.name)  # записать отклонённую задачу в журнал бюджета
            return self._hold(  # отказ
                spec, "POISONED_LABEL_INJECTION", len(train_x), len(val_x)  # причина — испорченная метка
            )

        # Конечная исчерпывающая проверка детерминирована и статистический бюджет
        # вероятности ошибки не тратит. Внешние статистические проверки должны
        # резервировать бюджет явно через model.reserve_statistical_test(...).

        adapter: Optional[FactorizedAdapter] = None  # нейронный адаптер пока отсутствует
        if train_neural and val_x and spec.family != "BoolVenn":  # нужен адаптер, есть проверочные точки и задача не булева
            adapter = self.core.fit_residual(train_x, train_y, val_x, val_y)  # обучить адаптер на остатке ядра

        # Уровень 0: повторное использование готового эксперта библиотеки
        ex_idx = self._expert_index()  # номера экспертов для вычисления длины кода
        for idx, (nm, ex) in enumerate(sorted(self.experts.items())):  # по экспертам в порядке имён с номером idx
            if ex.program.arity != spec.arity:  # другое число аргументов
                continue  # не подходит
            if not all(ex.supports(xs) for xs in domain):  # эксперт применим не ко всей области задачи
                continue  # не подходит
            self.stats.evals += len(train_x)  # учесть вычисления на обучающих точках
            try:  # вычисление программы может завершиться ошибкой
                if any(  # найдено хотя бы одно несовпадение на обучении
                    ex.program.eval(xs, self.experts) != yy  # значение эксперта expert(xs) != y
                    for xs, yy in zip(train_x, train_y)  # по обучающим парам
                ):  # есть несовпадение
                    continue  # не подходит
                if any(  # найдено несовпадение на проверке
                    ex.program.eval(xs, self.experts) != yy  # значение эксперта expert(xs) != y
                    for xs, yy in zip(val_x, val_y)  # по проверочным парам
                ):  # есть несовпадение
                    continue  # не подходит
                if any(ex.program.eval(xs, self.experts) != all_y[xs] for xs in domain):  # несовпадение на всей области: expert(xs) != teacher(xs)
                    continue  # не подходит
            except Exception:  # ошибка вычисления
                continue  # не подходит
            ref_bits = TOKEN_BITS + gamma_bits(idx + 1)  # длина кода ссылки: биты токена + код номера idx + 1
            ev = GrowthEvent(  # событие повторного использования
                self.version,  # версия модели
                spec.name,  # имя задачи
                "REUSE",  # тип события
                f"verified-expert-coverage:{nm}",  # причина: покрытие проверенным экспертом nm
                ex.program.pretty(),  # текст программы
                ref_bits,  # длина кода ссылки
                len(train_x),  # число обучающих точек
                len(val_x),  # число проверочных точек
                1.0,  # точность
            )
            self.growth_log.append(ev)  # записать в журнал
            return ev  # вернуть событие

        def try_commit(program: CallDagProgram, reason: str) -> Optional[GrowthEvent]:  # попытка принять кандидата program с причиной reason; None — не принят
            """Проверить кандидата program на train, val и всей области; принять или вернуть None."""
            try:  # вычисление программы может завершиться ошибкой
                self.stats.evals += len(train_x)  # учесть вычисления на обучении
                if any(  # несовпадение на обучении
                    program.eval(xs, self.experts) != yy  # значение программы program(xs) != y
                    for xs, yy in zip(train_x, train_y)  # по обучающим парам
                ):  # есть несовпадение
                    return None  # отклонить
                if val_x and any(  # есть проверочные точки и несовпадение на них
                    program.eval(xs, self.experts) != yy for xs, yy in zip(val_x, val_y)  # значение программы program(xs) != y
                ):  # есть несовпадение
                    return None  # отклонить
                if any(program.eval(xs, self.experts) != all_y[xs] for xs in domain):  # program(xs) != teacher(xs) на всей области
                    return None  # отклонить
            except Exception:  # ошибка вычисления
                return None  # отклонить
            bits = program.bits(ex_idx)  # длина кода программы в битах
            expert = StoredExpert(  # новый эксперт
                spec.name,  # имя
                spec.family,  # семейство
                program,  # программа
                spec.domains,  # области аргументов
                spec.valid_rule,  # правило области
                bits,  # длина кода
                adapter,  # нейронный адаптер
                parent_names,  # родители
            )
            self.experts[spec.name] = expert  # добавить в библиотеку
            self.version += 1  # версия: +1
            self.total_commits += 1  # счётчик добавлений: +1
            ev = GrowthEvent(  # событие добавления
                self.version,  # версия модели
                spec.name,  # имя задачи
                "COMMIT",  # тип события
                reason,  # причина принятия
                program.pretty(),  # текст программы
                bits,  # длина кода
                len(train_x),  # число обучающих точек
                len(val_x),  # число проверочных точек
                1.0,  # точность
            )
            self.growth_log.append(ev)  # записать в журнал
            return ev  # вернуть событие

        # Уровень 1: булевы задачи — точная диаграмма по 4-битной маске
        if spec.family == "BoolVenn":  # булева задача
            for m in range(16):  # по маскам m = 0..15
                ev = try_commit(CallDagProgram(2, venn_mask=m), "boolvenn-exact")  # попытка принять программу с маской m
                if ev is not None:  # программа принята
                    return ev  # вернуть событие

        if all(set(d) <= {0, 1} for d in spec.domains) and 1 <= spec.arity <= 6:  # булевы аргументы и арность от 1 до 6
            if set(train_y) <= {0, 1}:  # метки только 0 и 1
                full_inputs = list(itertools.product((0, 1), repeat=spec.arity))  # все 2^арность булевых входов
                train_map = dict(zip(train_x, train_y))  # вход -> метка по обучающей выборке
                missing = [r for r in full_inputs if r not in train_map]  # входы без метки
                if len(missing) <= 1:  # неизвестна не более одна точка
                    best_anf = None  # лучший кандидат (ключ, программа)
                    for fill in (0,) if not missing else (0, 1):  # нет пропусков — один вариант; иначе оба значения 0 и 1
                        tt = [train_map.get(r, fill) for r in full_inputs]  # полная таблица истинности с заполнением fill
                        coeffs = anf_transform(tt)  # коэффициенты полинома Жегалкина c_s = ⊕_{u ⊆ s} t_u
                        active_s = [s for s, c in enumerate(coeffs) if c]  # номера мономов с коэффициентом 1
                        max_deg = max((bin(s).count("1") for s in active_s), default=0)  # старшая степень = максимум веса |s| активных мономов
                        cand_e = anf_to_call_dag(coeffs, spec.arity)  # выражение по коэффициентам
                        cand_p = CallDagProgram(spec.arity, cand_e)  # программа-кандидат
                        score = (max_deg, len(active_s), cand_p.bits(ex_idx))  # ключ сравнения: (степень, число мономов, длина кода)
                        if best_anf is None or score < best_anf[0]:  # первый кандидат или лексикографически лучше
                            best_anf = (score, cand_p)  # запомнить лучшего
                    if best_anf is not None:  # кандидат построен
                        ev = try_commit(  # попытка принять
                            best_anf[1], "anf-zhegalkin-butterfly+full-support"  # лучшая программа, причина — полином Жегалкина
                        )
                        if ev is not None:  # принята
                            return ev  # вернуть событие

        # Уровень 2: прямые шаблоны примитивных программ
        for p5 in sorted(  # шаблоны, отсортированные по (стоимость, текст)
            candidate_programs(spec.arity), key=lambda p: (p.cost, p.pretty())  # кандидаты заданной арности; ключ сортировки
        ):  # конец сортированного списка
            if p5.expr is not None:  # у шаблона есть выражение
                ev = try_commit(  # попытка принять
                    CallDagProgram(spec.arity, CallDagExpr.from_h5(p5.expr)),  # программа из выражения шаблона
                    "direct-template+full-support",  # причина — прямой шаблон
                )
                if ev is not None:  # принята
                    return ev  # вернуть событие

        # Уровень 3: точный рациональный полином (арность 1..3)
        if 1 <= spec.arity <= 3 and not spec.family.startswith("Bool"):  # арность от 1 до 3, семейство не булево
            if spec.arity == 1:  # один аргумент
                coeffs = _exact_poly_fit(  # коэффициенты p(x) = Σ c_k·x^k, степень k <= 5
                    [r[0] for r in train_x], train_y, max_degree=5  # аргументы x, значения y, предел степени 5
                )
                if coeffs is not None:  # точная подгонка найдена
                    ev = try_commit(  # попытка принять
                        CallDagProgram(1, CallDagExpr("poly", (CallDagExpr.var(0),), coeffs)),  # программа: полином от переменной 0
                        "exact-rational-poly+full-support",  # причина — точный рациональный полином
                    )
                    if ev is not None:  # принята
                        return ev  # вернуть событие
            else:  # два или три аргумента
                terms = _exact_multivariate_poly_fit(  # словарь мономов: показатель -> коэффициент
                    train_x, train_y, spec.arity, max_degree=4  # точки, значения, арность, полная степень <= 4
                )
                if terms is not None:  # точная подгонка найдена
                    ev = try_commit(  # попытка принять
                        CallDagProgram(spec.arity, CallDagExpr("poly_nd", (), terms)),  # программа: многомерный полином
                        "exact-multivar-poly+full-support",  # причина — точный многомерный полином
                    )
                    if ev is not None:  # принята
                        return ev  # вернуть событие

        # Уровень 4: точный кусочный полином и модульная арифметика
        pw = exact_piecewise_poly_fit(train_x, train_y)  # выражение-кандидат: кусочный полином по (train_x, train_y)
        if pw is not None:  # кандидат построен
            ev = try_commit(  # попытка принять
                CallDagProgram(spec.arity, pw), "exact-piecewise-poly+full-support"  # программа из выражения, причина
            )
            if ev is not None:  # принята
                return ev  # вернуть событие

        # Уровень 5: ограниченные арифметические свёртки (кванторы «для всех» и
        # «существует» по конечным целым диапазонам). Это общее семейство гипотез,
        # а не подгонка под имена задач.
        if spec.arity == 1:  # один аргумент
            fold_candidates = sorted(  # свёртки, отсортированные по (длина кода, текст)
                bounded_fold_hypotheses(spec.arity),  # выражения-свёртки данной арности
                key=lambda expr: (expr.bits(ex_idx), expr.pretty()),  # ключ сортировки
            )
            for fold_expr in fold_candidates:  # по кандидатам
                ev = try_commit(  # попытка принять
                    CallDagProgram(spec.arity, fold_expr),  # программа из выражения-свёртки
                    "bounded-arithmetic-fold+full-support",  # причина — ограниченная свёртка
                )
                if ev is not None:  # принята
                    return ev  # вернуть событие

        # Уровень 6: пакетный перебор деревьев выражений по минимальной длине кода
        # с вызовами экспертов библиотеки
        lib_list = [  # библиотека для вызовов: (номер, имя, эксперт)
            (i, nm, self.experts[nm]) for i, nm in enumerate(sorted(self.experts))  # по отсортированным именам с номерами
        ]
        hit_expr, n_chk = synthesize_batched_mdl_stored(  # лучший кандидат и число проверенных узлов
            spec.arity,  # арность
            train_x,  # обучающие входы
            train_y,  # метки обучения
            val_x,  # проверочные входы
            val_y,  # метки проверки
            library_experts=lib_list,  # эксперты, которые можно вызывать
            max_evals=self.node_budget,  # предел числа вычислений
            max_bits=84,  # предел длины кода кандидата в битах
            deadline=deadline,  # предельное время
        )
        self.stats.candidates_generated += n_chk  # счётчик порождённых кандидатов: +n_chk
        if hit_expr is not None:  # кандидат найден
            ev = try_commit(  # попытка принять
                CallDagProgram(spec.arity, hit_expr), "batched-mdl-dag+full-support"  # программа из найденного выражения, причина
            )
            if ev is not None:  # принята
                return ev  # вернуть событие

        return self._hold(spec, "NO_VALIDATED_PROGRAM", len(train_x), len(val_x))  # ни один кандидат не прошёл — отказ

    def _hold(  # зафиксировать отказ
        self, spec: TaskSpec, reason: str, ntrain: int, nval: int  # задача, причина отказа, число точек обучения и проверки
    ) -> GrowthEvent:  # аннотация возврата
        """Записать событие HOLD в журнал и вернуть его."""
        self.total_holds += 1  # счётчик отказов: +1
        ev = GrowthEvent(  # событие отказа
            self.version, spec.name, "HOLD", reason, "", 0, ntrain, nval, 0.0  # версия, имя, тип, причина, пустая программа, длина 0, размеры выборок, точность 0
        )
        self.growth_log.append(ev)  # записать в журнал
        return ev  # вернуть событие

    def compress_library_mdl(self) -> dict[str, Any]:
        """Сократить библиотеку по принципу минимальной длины: поддерево -> вызов эксперта.

            Для поддерева s стоимости >= 2 программы эксперта t подбирается
            библиотечный эксперт l той же арности с вызовом call(x0.., l).
            Замена принимается, если bits(call) < bits(s), новая длина всей
            программы bits' < bits(t), граф вызовов остаётся ациклическим и
            новая программа совпадает со старой на всей области t.
            Возвращает before_bits, after_bits, refactored (число замен)."""
        ex_idx = self._expert_index()  # номера экспертов
        before_bits = sum(e.code_length for e in self.experts.values())  # суммарная длина до: Σ bits(e)
        refactored = 0  # число замен
        for target_name, target_ex in list(self.experts.items()):  # по экспертам-целям (копия списка)
            if target_ex.program.expr is None:  # программы нет
                continue  # пропустить
            cur_expr = target_ex.program.expr  # текущее выражение цели
            subs = [  # поддеревья-кандидаты на замену
                s  # поддерево s
                for s in cur_expr.subtrees()  # по всем поддеревьям
                if s.op not in ("var", "const", "call") and s.cost >= 2  # не лист и не вызов, стоимость >= 2
            ]
            if not subs:  # кандидатов нет
                continue  # пропустить
            domain = [  # область цели
                xs  # вход xs
                for xs in itertools.product(*target_ex.domains)  # по декартову произведению областей
                if target_ex.supports(xs)  # только допустимые входы
            ]
            for sub in subs:  # по поддеревьям-кандидатам
                for lib_name, lib_ex in list(self.experts.items()):  # по библиотечным экспертам (копия списка)
                    if (  # пропуск
                        lib_name == target_name  # сам эксперт
                        or lib_ex.program.arity != target_ex.program.arity  # другая арность
                    ):  # конец условия
                        continue  # не подходит
                    if lib_ex.program.arity == 2:  # бинарный эксперт
                        call_e = CallDagExpr(  # вызов от переменных 0 и 1
                            "call", (CallDagExpr.var(0), CallDagExpr.var(1)), lib_name  # операция, аргументы, имя вызываемого
                        )
                    elif lib_ex.program.arity == 1:  # унарный эксперт
                        call_e = CallDagExpr("call", (CallDagExpr.var(0),), lib_name)  # вызов от переменной 0
                    else:  # другая арность
                        continue  # не подходит
                    if call_e.bits(ex_idx) >= sub.bits(ex_idx):  # вызов не короче поддерева: bits(call) >= bits(s)
                        continue  # замена не экономит
                    cand_expr = cur_expr.replace_subtree(sub, call_e)  # выражение с заменой s на вызов
                    cand_prog = CallDagProgram(target_ex.program.arity, cand_expr)  # программа-кандидат
                    new_bits = cand_prog.bits(ex_idx)  # её длина bits'
                    if new_bits >= target_ex.code_length:  # bits' >= bits(t) — не короче
                        continue  # не экономит
                    # Экономящая биты замена допустима, только если общий граф
                    # вызовов остаётся ациклическим: одно равенство выходов не
                    # мешает рекурсивным циклам через псевдонимы.
                    old_program = target_ex.program  # запомнить старую программу
                    target_ex.program = cand_prog  # временно подставить кандидата
                    dag_check = verify_call_dag(self)  # проверка графа вызовов
                    target_ex.program = old_program  # вернуть старую программу
                    if not dag_check["valid"]:  # граф некорректен
                        continue  # замена отклонена
                    try:  # вычисление может завершиться ошибкой
                        if all(  # выходы совпадают на всей области
                            cand_prog.eval(xs, self.experts)  # новая программа
                            == target_ex.program.eval(xs, self.experts)  # равна старой
                            for xs in domain  # по всем допустимым входам
                        ):  # совпадает всюду
                            target_ex.program = cand_prog  # принять новую программу
                            target_ex.code_length = new_bits  # обновить длину кода
                            cur_expr = cand_expr  # текущее выражение обновлено
                            refactored += 1  # замен: +1
                    except Exception:  # ошибка вычисления
                        pass  # замена не принята
        after_bits = sum(e.code_length for e in self.experts.values())  # суммарная длина после: Σ bits'(e)
        return {  # итог
            "before_bits": before_bits,  # битов до
            "after_bits": after_bits,  # битов после
            "refactored": refactored,  # число замен
        }

    def predict(self, name: str, *xs: int, mode: str = "exact") -> dict[str, Any]:  # ответ эксперта name на входе xs; mode — "exact" или "neural"
        """Выдать ответ эксперта name на входе xs.

            Возвращает словарь со статусом: OK и точное значение программы
            (mode = "exact"); APPROXIMATE и значение адаптера либо булевой
            диаграммы (mode = "neural"); ABSTAIN, если эксперт неизвестен, вход
            вне области или приближение недоступно."""
        ex = self.experts.get(name)  # эксперт по имени или None
        if ex is None:  # эксперт не найден
            return {"status": "ABSTAIN", "reason": "unknown operation"}  # отказ: неизвестная операция
        if not ex.supports(xs):  # вход вне области
            return {"status": "ABSTAIN", "reason": "outside certified finite domain"}  # отказ: вне области
        if mode == "exact":  # точный режим
            return {  # точный ответ
                "status": "OK",  # статус
                "value": ex.program.eval(xs, self.experts),  # значение программы program(xs)
                "source": "verified-symbolic-dag",  # источник — проверенная символьная программа
                "model_version": self.version,  # версия модели
            }
        if mode == "neural":  # приближённый режим
            if ex.adapter is None:  # адаптера нет
                if ex.program.venn_mask is not None:  # программа — булева диаграмма
                    return {  # точный ответ диаграммы, помеченный как приближение
                        "status": "APPROXIMATE",  # статус
                        "value": float(venn_head_eval(ex.program.venn_mask, xs)),  # значение функции по маске
                        "source": "venn-head-exact-ternary",  # источник — булева диаграмма
                        "model_version": self.version,  # версия модели
                    }
                return {"status": "ABSTAIN", "reason": "neural adapter unavailable"}  # отказ: адаптер недоступен
            return {  # ответ адаптера
                "status": "APPROXIMATE",  # статус
                "value": ex.adapter.predict(self.core, xs, quantized=True),  # предсказание адаптера поверх общего ядра, квантованное
                "source": "factorized-core-bitalloc-rvq",  # источник — нейронный адаптер
                "model_version": self.version,  # версия модели
            }
        raise ValueError("mode must be exact or neural")  # неизвестный режим — ошибка

    def verify_all(self, specs: Sequence[TaskSpec]) -> dict[str, Any]:  # specs — задачи
        """Проверить всех изученных экспертов на полной области своих задач.

            Для каждой задачи из specs с изученным экспертом считается число
            совпадений program(xs) == teacher(xs). Возвращает checked, correct,
            accuracy = correct / max(1, checked), committed = |experts|."""
        total = correct = 0  # всего проверено и сколько верно
        for sp in specs:  # по задачам
            ex = self.experts.get(sp.name)  # эксперт задачи или None
            if ex is None:  # задача не изучена
                continue  # пропустить
            rows = sp.inputs()  # полная область задачи
            ok = sum(  # число верных ответов
                int(ex.program.eval(xs, self.experts) == int(sp.teacher(*xs)))  # 1, если program(xs) == teacher(xs), иначе 0
                for xs in rows  # по входам области
            )
            total += len(rows)  # проверено: +|D|
            correct += ok  # верно: +ok
        return {  # итог
            "checked": total,  # всего проверено
            "correct": correct,  # верных
            "accuracy": correct / max(1, total),  # точность = correct / max(1, checked)
            "committed": len(self.experts),  # число экспертов
        }

    def reserve_statistical_test(self, task: str, alpha: float) -> bool:  # зарезервировать долю бюджета alpha для задачи task; True — резерв выделен
        """Зарезервировать долю alpha статистического бюджета для внешней проверки task."""
        return self.ledger.spend_anytime(task, alpha)  # списать alpha из бюджета; True, если удалось

    def verify_lineage(self) -> dict[str, Any]:  # проверить граф вызовов и граф порождения
        """Проверить корректность обоих графов модели; возвращает valid, call_dag, growth_lineage."""
        calls = verify_call_dag(self)  # отчёт по графу вызовов
        growth = verify_growth_lineage(self)  # отчёт по графу порождения
        return {  # общий отчёт
            "valid": calls["valid"] and growth["valid"],  # корректно, когда корректны оба графа
            "call_dag": calls,  # граф вызовов
            "growth_lineage": growth,  # граф порождения
        }

    def save(self, path: str | Path) -> None:  # сохранить состояние в файл path
        """Атомарно сохранить полное состояние модели в JSON-файл path.

            Состояние считается некорректным (и не сохраняется), если граф
            вызовов или граф порождения содержит пропуски или циклы. Файл
            пишется во временный path + ".tmp", затем подменяет цель."""
        audit = self.verify_lineage()  # проверка графов
        if not audit["valid"]:  # модель некорректна
            raise ValueError(f"cannot save invalid model graph: {audit}")  # ошибка сохранения
        data = {  # словарь состояния
            "format": STATE_FILE_FORMAT,  # метка формата файла (строка-код: её читает load(); не переводится)
            "version": STATE_FORMAT,  # версия формата
            "node_budget": self.node_budget,  # предел проверок
            "model_version": self.version,  # версия модели
            "total_commits": self.total_commits,  # число добавлений
            "total_holds": self.total_holds,  # число отказов
            "core": {  # общее нейронное ядро
                "W1": self.core.W1.tolist(),  # веса первого слоя списком
                "b1": self.core.b1.tolist(),  # смещения списком
                "W1_q": self.core.W1_q.tolist(),  # квантованные веса списком
                "b1_q": self.core.b1_q.tolist(),  # квантованные смещения списком
                "core_bytes": self.core.core_bytes,  # размер ядра в байтах
            },
            "experts": {  # эксперты
                name: ex.to_dict() for name, ex in sorted(self.experts.items())  # имя -> словарь эксперта, в порядке имён
            },
            "theorems": [vars(t) for t in self.theorems],  # теоремы словарями атрибутов
            "growth_log": [vars(ev) for ev in self.growth_log],  # журнал роста словарями атрибутов
            "ledger": {  # статистический бюджет
                "delta_total": self.ledger.delta_total,  # общий бюджет
                "spent": dict(self.ledger.spent),  # израсходовано по задачам
                "rejected_tasks": sorted(self.ledger.rejected_tasks),  # отклонённые задачи списком по алфавиту
                "final_test_touched": self.ledger.final_test_touched,  # признак использования финальной проверки
            },
            "router": {  # маршрутизатор
                "K": self.router.K,  # число экспертов маршрутизатора
                "eta": self.router.eta,  # темп обучения
                "weights": self.router.weights.tolist(),  # веса списком
                "cum_loss": self.router.cum_loss.tolist(),  # накопленные потери списком
            },
            "ram": {"mem": self.ram.mem.tolist()},  # память: матрица mem списком
            "stats": {  # счётчики синтеза
                "candidates_generated": self.stats.candidates_generated,  # порождено кандидатов
                "evals": self.stats.evals,  # число вычислений
                "sieve_survivors": self.stats.sieve_survivors,  # выживших после просеивания
            },
            "audit": audit,  # отчёт о графах на момент сохранения
        }
        target = Path(path)  # целевой путь
        target.parent.mkdir(parents=True, exist_ok=True)  # создать каталог цели со всеми родителями
        temp = target.with_suffix(target.suffix + ".tmp")  # временный файл: к расширению добавлено ".tmp"
        temp.write_text(  # записать временный файл
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False),  # JSON с отступом 2, ключи по алфавиту, без NaN
            encoding="utf-8",  # кодировка UTF-8
        )
        os.replace(temp, target)  # атомарно подменить цель временным файлом

    @classmethod  # метод класса — конструктор из файла
    def load(cls, path: str | Path) -> "GrowthProcessorBase":  # загрузить состояние из файла path
        """Загрузить модель из JSON-файла path, проверив формы и значения тензоров.

            Ошибки ValueError: неверная сигнатура формата, несогласованные
            формы тензоров ядра (W1 — 3 строки, b1 — (1, число столбцов W1),
            квантованные тензоры той же формы), неверные векторы маршрутизатора
            (форма (K,), веса конечны, неотрицательны, Σ w = 1), неверная
            память, некорректные графы после загрузки."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))  # прочитать и разобрать JSON
        if data.get("format") != STATE_FILE_FORMAT and data.get("format") not in LEGACY_FILE_FORMATS[STATE_FILE_FORMAT]:  # метка формата не опознана (ни текущая, ни прежняя)
            raise ValueError("не файл состояния модели HAGI")  # ошибка формата
        core_data = data.get("core", {})  # блок ядра (по умолчанию пустой словарь)
        w1 = np.asarray(core_data.get("W1"), dtype=np.float64)  # веса W1 вещественным массивом
        b1 = np.asarray(core_data.get("b1"), dtype=np.float64)  # смещения b1
        w1q = np.asarray(core_data.get("W1_q"), dtype=np.float64)  # квантованные веса
        b1q = np.asarray(core_data.get("b1_q"), dtype=np.float64)  # квантованные смещения
        if w1.ndim != 2 or w1.shape[0] != 3 or b1.shape != (1, w1.shape[1]):  # W1 не матрица 3×n или b1 не 1×n
            raise ValueError("invalid shared-core float tensor shapes")  # ошибка формы ядра
        if w1q.shape != w1.shape or b1q.shape != b1.shape:  # квантованные тензоры другой формы
            raise ValueError("invalid shared-core quantized tensor shapes")  # ошибка формы квантованных тензоров
        model = cls(  # новый процессор
            seed=0,  # зерно (веса будут заменены загруженными)
            node_budget=int(data.get("node_budget", 4000)),  # предел проверок, по умолчанию 4000
            delta_total=float(data.get("ledger", {}).get("delta_total", 0.05)),  # бюджет ошибки, по умолчанию 0.05
        )
        model.core.W1, model.core.b1 = w1, b1  # подставить вещественные веса и смещения
        model.core.W1_q, model.core.b1_q = w1q, b1q  # подставить квантованные
        model.core.core_bytes = int(core_data.get("core_bytes", 0))  # размер ядра в байтах, по умолчанию 0
        model.experts = {  # восстановить экспертов
            name: StoredExpert.from_dict(ex) for name, ex in data.get("experts", {}).items()  # имя -> эксперт из словаря
        }
        model.version = int(data.get("model_version", 0))  # версия модели, по умолчанию 0
        model.total_commits = int(data.get("total_commits", len(model.experts)))  # число добавлений, по умолчанию |experts|
        model.total_holds = int(data.get("total_holds", 0))  # число отказов, по умолчанию 0
        model.theorems = [Theorem(**t) for t in data.get("theorems", [])]  # теоремы из словарей
        model.growth_log = [GrowthEvent(**ev) for ev in data.get("growth_log", [])]  # журнал роста из словарей
        led = data.get("ledger", {})  # блок статистического бюджета
        model.ledger = AuditLedger(  # восстановить журнал бюджета
            float(led.get("delta_total", 0.05)),  # общий бюджет
            {str(k): float(v) for k, v in led.get("spent", {}).items()},  # израсходовано: ключи-строки, значения-float
            set(str(x) for x in led.get("rejected_tasks", [])),  # множество отклонённых задач
            bool(led.get("final_test_touched", False)),  # признак использования финальной проверки
        )
        rd = data.get("router", {})  # блок маршрутизатора
        model.router = HedgeRouter(int(rd.get("K", 4)), float(rd.get("eta", 0.5)))  # маршрутизатор: K по умолчанию 4, eta 0.5
        model.router.weights = np.asarray(  # веса маршрутизатора
            rd.get("weights", model.router.weights), dtype=np.float64  # из файла или текущие
        )
        model.router.cum_loss = np.asarray(  # накопленные потери
            rd.get("cum_loss", model.router.cum_loss), dtype=np.float64  # из файла или текущие
        )
        if model.router.weights.shape != (  # форма весов не (K,)
            model.router.K,  # ожидаемое число весов K
        ) or model.router.cum_loss.shape != (model.router.K,):  # или форма потерь не (K,)
            raise ValueError("invalid serialized router vectors")  # ошибка векторов маршрутизатора
        if (  # проверка весов
            not np.all(np.isfinite(model.router.weights))  # есть не конечный вес
            or np.any(model.router.weights < 0)  # есть отрицательный вес
            or not np.isclose(model.router.weights.sum(), 1.0)  # сумма весов Σ w не близка к 1
        ):  # конец условия
            raise ValueError("invalid serialized router weights")  # ошибка весов маршрутизатора
        mem = np.asarray(  # матрица памяти
            data.get("ram", {}).get("mem", model.ram.mem), dtype=np.float64  # из файла или текущая
        )
        if mem.shape != model.ram.mem.shape or not np.all(np.isfinite(mem)):  # другая форма или есть не конечное значение
            raise ValueError("invalid serialized GDN RAM state")  # ошибка памяти
        model.ram.mem = mem  # подставить память
        st = data.get("stats", {})  # блок счётчиков
        model.stats.candidates_generated = int(st.get("candidates_generated", 0))  # порождено кандидатов, по умолчанию 0
        model.stats.evals = int(st.get("evals", 0))  # число вычислений, по умолчанию 0
        model.stats.sieve_survivors = int(st.get("sieve_survivors", 0))  # выживших после просеивания, по умолчанию 0
        audit = model.verify_lineage()  # проверка графов после загрузки
        if not audit["valid"]:  # графы некорректны
            raise ValueError(f"loaded model graph is invalid: {audit}")  # ошибка загруженной модели
        return model  # вернуть модель

    def total_neural_bytes(self) -> int:  # оценка размера нейронной части в байтах
        """Оценить размер упакованных нейронных данных в байтах.

            total = core_bytes + venn_bytes + Σ adapter.compressed_bytes,
            где venn_bytes = 13, если есть хоть один эксперт семейства
            "BoolVenn", иначе 0."""
        return _expert_total_neural_bytes(self)  # посчитать суммарный размер нейронной части в байтах

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'StoredExpert', 'StoredTheoremMiner', 'verify_call_dag', 'verify_growth_lineage', 'GrowthProcessorBase',  # имя интерфейса
]  # конец перечня
