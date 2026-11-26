"""Процессор символьного ядра HAGI: эксперт, майнер теорем, унификация графа вызовов.

Каждая строка снабжена русским комментарием: что делает и по какой формуле.
"""

import itertools  # импорт стандартной библиотеки, нужный этой части
import json  # импорт стандартной библиотеки, нужный этой части
from dataclasses import dataclass  # импорт стандартной библиотеки, нужный этой части
from pathlib import Path  # импорт стандартной библиотеки, нужный этой части
from typing import Any, Optional, Sequence  # импорт стандартной библиотеки, нужный этой части
import numpy as np  # импорт стандартной библиотеки, нужный этой части
from hagi.core.synthesis.synthesis import _exact_multivariate_poly_fit, _exact_poly_fit  # имена из модуля ядра «synthesis» (как до слияния)
from hagi.core.memory.aliases import TOKEN_BITS  # имена из «aliases» (прежний общий импорт из пакета наследия)
from hagi.core.memory.aliases import gamma_bits  # имена из «aliases» (прежний общий импорт из пакета наследия)
from hagi.core.tasks.tasks import TaskSpec  # имена из модуля ядра «tasks» (как до слияния)
from hagi.core.language.expressions import Expr, Program  # имена из модуля ядра «expressions» (как до слияния)
from hagi.core.language.expressions import candidate_programs  # имена из модуля ядра «expressions» (как до слияния)
from hagi.core.synthesis.venv_anf import anf_to_expr  # имена из «venv_anf» (модуль, где они определены)
from hagi.core.synthesis.venv_anf import anf_transform, venn_head_eval  # имена из модуля ядра «venv_anf» (как до слияния)
from hagi.core.neural.neural import FactorizedAdapter, FactorizedSharedCore  # имена из модуля ядра «neural» (как до слияния)
from hagi.core.neural.segmentation import GDNRAM, boundary_chunks, boundary_heads  # имена из модуля ядра «segmentation» (как до слияния)
from hagi.core.processor.experts import expert_supports  # импорт общей проверки применимости эксперта к аргументам
from hagi.core.memory.audit import AuditLedger  # имена из модуля ядра «audit» (как до слияния)
from hagi.core.memory.audit import AuditStream, CertStepGen  # имена из «audit» (модуль, где они определены)
from hagi.core.memory.audit import HedgeRouter  # имена из модуля ядра «audit» (как до слияния)
from hagi.core.synthesis.synthesis import _CURRENT_FAMILY, _FULL_CAND, _mod_affine_expr, _sparse_membership, synthesize_batched_mdl  # имена из «synthesis» (модуль, где они определены)
from hagi.core.synthesis.synthesis import _exact_multivariate_poly_fit, _exact_poly_fit  # имена из модуля ядра «synthesis» (как до слияния)
from hagi.core.synthesis.synthesis import exact_piecewise_poly_fit  # имена из модуля ядра «synthesis» (как до слияния)
from hagi.base.config import MODEL_FILE_FORMAT, LEGACY_FILE_FORMATS, STATE_FORMAT  # единая метка формата состояния и перечень прежних меток

# ==============================================================================
# 10. Эксперт, майнер теорем, структурная унификация графа вызовов и процессор
# ==============================================================================


@dataclass  # декоратор: конструктор, сравнение и представление генерируются по аннотациям полей
class Expert:
    """Эксперт: именованная программа с конечными областями аргументов.
    
            Поля: name — имя; family — семейство; program — программа (формула);
            domains — кортеж допустимых значений для каждого аргумента; valid_rule —
            дополнительное правило корректности входа ("nonzero_y" — второй аргумент не 0,
            "ordered_yz" — второй не больше третьего, иначе без правила); code_length —
            длина кода; adapter — адаптер общих весов или None; parents — имена родителей."""
    name: str  # имя эксперта
    family: str  # семейство эксперта
    program: Program  # программа — вычисляемая формула
    domains: tuple[tuple[int, ...], ...]  # области аргументов: кортеж кортежей, domains[i] — допустимые значения i-го аргумента
    valid_rule: str  # правило корректности входа
    code_length: int  # длина кода программы
    adapter: Optional[FactorizedAdapter] = None  # адаптер общих весов (по умолчанию нет)
    parents: tuple[str, ...] = ()  # имена родителей (по умолчанию пусто)

    def supports(self, xs: Sequence[int]) -> bool:
        """Проверить, допустим ли вход xs: True, если каждый аргумент в своей области и правило valid_rule выполнено.
        
                    xs — кортеж целых аргументов."""
        return expert_supports(self, xs)  # проверить применимость эксперта к аргументам

    def to_dict(self) -> dict[str, Any]:
        """Словарь с полями эксперта для сериализации: имени, семейства, программы, областей, правила, длины кода, адаптера (словарь или None) и родителей.
        
                    Области и родители записываются списками."""
        return _expert_to_dict(self)  # собрать словарь с полями эксперта

    @staticmethod  # статический метод — создаётся без экземпляра
    def from_dict(d: dict[str, Any], core: "FactorizedSharedCore") -> "Expert":
        """Восстановить эксперта из словаря d (поля — как в to_dict).
        
                    d — словарь с ключами name, family, program, domains, valid_rule,
                    code_length и необязательными adapter, parents; core — общий блок весов,
                    нужный адаптеру для восстановления. Возвращает Expert."""
        return Expert(  # конструктор эксперта
            name=str(d["name"]),  # имя
            family=str(d["family"]),  # семейство
            program=Program.from_dict(d["program"]),  # программа из словаря
            domains=tuple(tuple(int(v) for v in dom) for dom in d["domains"]),  # области: каждый список d превращается в кортеж целых
            valid_rule=str(d["valid_rule"]),  # правило корректности
            code_length=int(d["code_length"]),  # длина кода
            adapter=(  # адаптер
                FactorizedAdapter.from_dict(d["adapter"], core)  # восстановить адаптер по словарю, если он был сохранён
                if d.get("adapter")  # если адаптер сохранён
                else None  # иначе None
            ),  # закрытие аргумента adapter
            parents=tuple(str(p) for p in d.get("parents", [])),  # родители — кортеж строк (по умолчанию пусто)
        )  # закрытие вызова конструктора


@dataclass  # декоратор: конструктор и представление генерируются по аннотациям полей
class Theorem:
    """Теорема о свойствах эксперта на конечной области: имя, текст утверждения, число проверенных точек и признак проверки.
    
            verified по умолчанию True."""
    name: str  # имя теоремы
    statement: str  # текст утверждения
    domain_points: int  # число проверенных точек области
    verified: bool = True  # признак проверки (по умолчанию истина)




def _expert_to_dict(obj) -> dict[str, Any]:  # собрать словарь с полями эксперта: имя, семейство, программа, области, правило, длина кода, адаптер, родители
    """Словарь с полями эксперта для сериализации: имени, семейства, программы, областей, правила, длины кода, адаптера (словарь или None) и родителей.
    
                Области и родители записываются списками."""
    return {  # словарь с полями эксперта
        "name": obj.name,  # имя
        "family": obj.family,  # семейство
        "program": obj.program.to_dict(),  # программа как словарь
        "domains": [list(d) for d in obj.domains],  # области — списки значений: [list(d) для каждого d]
        "valid_rule": obj.valid_rule,  # правило корректности
        "code_length": obj.code_length,  # длина кода
        "adapter": obj.adapter.to_dict() if obj.adapter else None,  # адаптер как словарь, если он есть, иначе None
        "parents": list(obj.parents),  # родители — список
    }


def _expert_total_neural_bytes(obj) -> int:  # посчитать суммарный размер нейронной части: общий блок весов + голова Венна + адаптеры
    """Суммарное число байт нейросетевой части: общий блок весов + голова функции Венна + адаптеры.
    
            Голова функции Венна занимает 13 байт, если есть хоть один эксперт семейства «BoolVenn».
            Возвращает целое число байт."""
    venn_bytes = (  # байты головы функции Венна
        13 if any(e.family == "BoolVenn" for e in obj.experts.values()) else 0  # 13, если есть эксперт «BoolVenn», иначе 0
    )  # закрытие скобки
    adapter_bytes = sum(  # байты всех адаптеров
        e.adapter.compressed_bytes  # размер сжатого адаптера
        for e in obj.experts.values()  # по всем экспертам
        if e.adapter is not None  # у которых есть адаптер
    )
    return obj.core.core_bytes + venn_bytes + adapter_bytes  # общий блок + голова + адаптеры


class TheoremMiner:
    """Майнер теорем: полный перебор конечных областей всех экспертов модели (без усечения).
    
            Для каждого эксперта проверяются свойства: коммутативность, поглощение
            собой (e(x, x) = x), правый ноль (e(x, 0) = x) — для бинарных; замкнутость,
            инволютивность, идемпотентность, чётная и нечётная симметрия — для унарных.
            Свойство объявляется теоремой, только если оно выполнено на ВСЕХ допустимых точках."""
    """Найти теоремы модели: проверить свойства каждого эксперта на всей его конечной области.
    
                model — процессор с экспертами model.experts и списком уже известных
                теорем model.theorems. Возвращает список новых объектов Theorem; найденные
                теоремы добавляются в model.theorems."""

    def mine(self, model: "GrowingHagiProcessor") -> list[Theorem]:  # найти в модели новые теоремы и вернуть их список
        found: list[Theorem] = []  # список новых теорем
        known = {t.name for t in model.theorems}  # имена уже известных теорем
        env = model.experts  # среда вычисления — эксперты модели
        for e in model.experts.values():  # по экспертам модели
            if e.program.arity == 2:  # бинарный эксперт
                dx, dy = e.domains[0], e.domains[1]  # области первого и второго аргументов
                declared = [(x, y) for x in dx for y in dy if e.supports((x, y))]  # все допустимые пары: (x, y), x ∈ dx, y ∈ dy, e.supports((x, y)) = True
                swapped = [(x, y) for x, y in declared if e.supports((y, x))]  # пары, для которых допустим и обратный порядок (y, x)
                if swapped and len(swapped) == len(declared):  # коммутативность имеет смысл, только если допустимы все пары: |swapped| = |declared|
                    nm = f"commutativity({e.name})"  # имя теоремы коммутативности
                    if nm not in known:  # теоремы с таким именем ещё нет
                        try:  # попытка проверки (ошибка вычисления — теорема не объявляется)
                            if all(  # если для всех пар e(x, y) = e(y, x)
                                e.program.eval((x, y), env)  # вычислить e(x, y) в среде экспертов
                                == e.program.eval((y, x), env)  # равно e(y, x)
                                for x, y in swapped  # по допустимым парам
                            ):
                                th = Theorem(  # новая теорема
                                    nm,  # имя
                                    f"for all supported pairs: {e.name}(x,y)={e.name}(y,x)",  # текст теоремы о перестановке аргументов
                                    len(swapped),  # число проверенных пар
                                )
                                model.theorems.append(th)  # добавить в теоремы модели
                                found.append(th)  # добавить в найденные
                                known.add(nm)  # запомнить имя
                        except Exception:  # ошибка вычисления
                            pass  # пропустить свойство
                self_pairs = [x for x in dx if x in dy and e.supports((x, x))]  # пары (x, x), где x есть в обеих областях и e.supports((x, x)) = True
                nm = f"self-absorption({e.name})"  # имя теоремы поглощения собой
                if self_pairs and len(self_pairs) == len(dx) and nm not in known:  # если x пробегает всю первую область и теоремы ещё нет
                    try:  # попытка проверки
                        if all(e.program.eval((x, x), env) == x for x in self_pairs):  # если для всех x: e(x, x) = x
                            th = Theorem(  # новая теорема
                                nm,  # имя
                                f"for all overlapping x: {e.name}(x,x)=x",  # текст теоремы о совпадающих аргументах
                                len(self_pairs),  # число проверенных точек
                            )
                            model.theorems.append(th)  # добавить в теоремы модели
                            found.append(th)  # добавить в найденные
                            known.add(nm)  # запомнить имя
                    except Exception:  # ошибка вычисления
                        pass  # пропустить свойство
                zero_pairs = [x for x in dx if 0 in dy and e.supports((x, 0))]  # значения x из первой области, для которых допустима пара (x, 0): 0 ∈ dy и e.supports((x, 0))
                nm = f"right-identity-zero({e.name})"  # имя теоремы о правом нуле
                if zero_pairs and len(zero_pairs) == len(dx) and nm not in known:  # если x пробегает всю первую область и теоремы ещё нет
                    try:  # попытка проверки
                        if all(e.program.eval((x, 0), env) == x for x in zero_pairs):  # если для всех x: e(x, 0) = x
                            th = Theorem(  # новая теорема
                                nm, f"for all x: {e.name}(x,0)=x", len(zero_pairs)  # текст теоремы о нейтральном нуле, число проверенных пар
                            )
                            model.theorems.append(th)  # добавить в теоремы модели
                            found.append(th)  # добавить в найденные
                            known.add(nm)  # запомнить имя
                    except Exception:  # ошибка вычисления
                        pass  # пропустить свойство
            elif e.program.arity == 1:  # унарный эксперт
                dom = tuple(int(x) for x in e.domains[0])  # область определения — кортеж целых
                img = {}  # образ img: x → y
                closed = True  # признак замкнутости (образ лежит в области)
                try:  # попытка вычислить образ всей области
                    for x in dom:  # по точкам области
                        y = int(e.program.eval((x,), env))  # вычислить y = e(x) как целое
                        if not e.supports((y,)):  # y вне области — не замкнуто
                            closed = False  # прекратить перебор
                            break  # прекратить перебор области
                        img[x] = y  # запомнить y = e(x)
                except Exception:  # ошибка вычисления — не замкнуто
                    closed = False  # считать не замкнутым
                if closed:  # образ вычислен; проверять свойства e∘e и симметрию
                    for prop, fn in (  # по свойствам: имя и функция проверки без аргументов
                        (  # открытие кортежа
                            "involutivity",  # инволютивность
                            lambda: all(  # проверка: для всех x: e(e(x)) = x
                                e.program.eval((img[x],), env) == x for x in dom  # вычислить e(img[x]); равно x
                            ),  # равно x
                        ),
                        (  # идемпотентность
                            "idempotence",  # идемпотентность
                            lambda: all(  # проверка: для всех x: e(e(x)) = e(x)
                                e.program.eval((img[x],), env) == img[x] for x in dom  # вычислить e(img[x]); равно img[x]
                            ),
                        ),  # закрытие
                    ):  # закрытие кортежа свойств
                        nm = f"{prop}({e.name})"  # имя теоремы
                        if nm not in known:  # теоремы с таким именем ещё нет
                            try:  # попытка проверки
                                if fn():  # свойство выполнено на всей области
                                    th = Theorem(  # новая теорема
                                        nm, f"{prop} on certified domain", len(dom)  # текст утверждения
                                    )  # число проверенных точек
                                    model.theorems.append(th)  # добавить в теоремы модели
                                    found.append(th)  # добавить в найденные
                                    known.add(nm)  # запомнить имя
                            except Exception:  # ошибка вычисления
                                pass  # пропустить свойство
                if all((-x) in dom for x in dom):  # область симметрична: для каждого x число −x тоже в области
                    for sym_name, sgn in (("even-symmetry", 1), ("odd-symmetry", -1)):  # по видам симметрии: (название, знак) — чётная со знаком 1, нечётная со знаком −1
                        nm = f"{sym_name}({e.name})"  # имя теоремы симметрии
                        if nm not in known:  # теоремы с таким именем ещё нет
                            try:  # попытка проверки
                                if all(  # если для всех x: e(−x) = sgn·e(x)
                                    e.program.eval((-x,), env)  # вычислить e(−x); равно sgn·e(x)
                                    == sgn * e.program.eval((x,), env)  # равно sgn·e(x)
                                    for x in dom  # по точкам области
                                ):
                                    th = Theorem(  # новая теорема
                                        nm,  # имя
                                        f"{e.name}(-x)={'+' if sgn == 1 else '-'}{e.name}(x)",  # текст теоремы о значении на отрицательном аргументе
                                        len(dom),  # число проверенных точек
                                    )
                                    model.theorems.append(th)  # добавить в теоремы модели
                                    found.append(th)  # добавить в найденные
                                    known.add(nm)  # запомнить имя
                            except Exception:  # ошибка вычисления
                                pass  # пропустить свойство
        return found  # вернуть все найденные теоремы


def _match_pattern(
    pattern: Expr, target: Expr, env_bind: dict[int, Expr]  # аргументы: образец pattern, цель target и словарь связываний env_bind
) -> bool:  # возвращает True, если деревья сопоставились
    """Структурная унификация: сопоставить дерево pattern с поддеревом target.
    
            pattern — выражение-образец, его узлы var(i) — «дыры» с номером i; target —
            проверяемое выражение; env_bind — накопленные связывания номер дыры → выражение
            (изменяется на месте). Возвращает True, если деревья совпадают: операция и
            значение узлов равны, аргументы сочетаются попарно, дыра связывается с любым
            поддеревом, повторная дыра требует равенства с прежним связыванием."""
    """Structural unification: match pattern AST (with var(i) holes) against target subtree."""
    if pattern.op == "var":  # узел-образец — дыра var(i)
        idx = int(pattern.value or 0)  # номер дыры idx (по умолчанию 0)
        if idx in env_bind:  # дыра уже связана
            return env_bind[idx] == target  # True, если прежнее связывание равно target
        env_bind[idx] = target  # связать дыру idx с target
        return True  # сопоставление удалось
    if (  # если узлы разные
        pattern.op != target.op  # разные операции
        or pattern.value != target.value  # или разные значения
        or len(pattern.args) != len(target.args)  # или разное число аргументов
    ):
        return False  # деревья не совпадают
    return all(  # все аргументы сочетаются попарно: _match_pattern(pattern.args[i], target.args[i]) для всех i
        _match_pattern(pa, ta, env_bind) for pa, ta in zip(pattern.args, target.args)  # генератор: сопоставить попарно аргументы pa из pattern.args и ta из target.args с общим env_bind
    )


def _dag_reaches(experts: dict[str, Expert], start: str, target: str) -> bool:
    """Проверить по графу вызовов, достижима ли задача target из задачи start: идёт ли вызов (прямо или через других экспертов) от start к target.
    
            experts — все эксперты по именам; start — имя начального эксперта; target — имя искомого.
            Обход в глубину по ребрам «эксперт → вызываемые им эксперты». Возвращает True, если target достигнут."""
    visited = set()  # множество посещённых имён
    stack = [start]  # стек обхода, начиная со start
    while stack:  # пока стек не пуст
        cur = stack.pop()  # снять имя с вершины стека
        if cur == target:  # достигли цели
            return True  # вернуть «достижима»
        if cur in visited or cur not in experts:  # уже посещено или такого эксперта нет
            continue  # пропустить
        visited.add(cur)  # пометить как посещённое
        ex = experts[cur]  # эксперт с этим именем
        if ex.program.expr is not None:  # у эксперта есть программа
            stack.extend(ex.program.expr.callees())  # положить в стек всех, кого он вызывает: список callees() его выражения
    return False  # стек исчерпан — цель не достигнута


class GrowingHagiProcessor:
    """Саморазвивающийся процессор с управлением: общий блок весов, библиотека экспертов, аудит значимости, маршрутизатор и память.
    
            Хранит экспертов по именам задач (self.experts), теоремы о них, журнал роста
            (growth_log) и сертификаты шагов (cert_log). Новая задача либо решается уже
            готовым экспертом, либо для неё синтезируется и проверяется программа;
            непроверенные изменения не принимаются (удержание). Цель — доля нерешённых
            задач: unsolved_objective уменьшается на 1 за каждый принятый эксперта,
            расходуются бюджет вычислений compute_budget и лимит риска risk_rem."""
    """Unified HAGI канонический Self-Developing Processor with Full Governance."""

    def __init__(  # конструктор (без docstring): каждая строка параметров пояснена комментарием
        self,  # сам экземпляр
        seed: int = 42,  # seed — зерно случайности общего блока весов
        node_budget: int = 8000,  # node_budget — предел числа проверенных выражений при синтезе
        delta_total: float = 0.05,  # delta_total — общий бюджет статистической значимости аудита
        compute_budget: float = 1000.0,  # compute_budget — бюджет вычислений
        risk_limit: float = 10.0,  # risk_limit — лимит риска
    ):
        self.core = FactorizedSharedCore(seed=seed, width=48)  # общий блок весов: 48 ветвей, зерно seed
        self.node_budget = int(node_budget)  # предел проверенных выражений как целое
        self.experts: dict[str, Expert] = {}  # библиотека экспертов: имя задачи → эксперт (изначально пуста)
        self.version = 0  # версия схемы (номер принятого изменения)
        self.total_commits = 0  # счётчик принятых изменений
        self.total_holds = 0  # счётчик удержаний
        self.growth_log: list[dict[str, Any]] = []  # журнал роста: список событий обучения
        self.cert_log: list[CertStepGen] = []  # журнал сертификатов шагов
        # признак полноты сохранённого состояния управления: старое сохранение, потерявшее
        # журнал управления, не должно читаться как успешное пустое доказательство
        # (поэтому после загрузки неполного состояния проверка сертификатов обязана провалиться)
        self.governance_state_complete = True  # управляющее состояние полностью сохранено (истина, пока не доказано обратное)
        self.theorems: list[Theorem] = []  # список теорем о свойствах экспертов
        self.ledger = AuditLedger(delta_total=delta_total)  # журнал аудита значимости с общим бюджетом delta_total
        self.router = HedgeRouter(len(FactorizedSharedCore.BRANCHES), eta=0.5)  # маршрутизатор ветвей с темпом обучения eta = 0.5
        self.ram = GDNRAM(n_slots=64, dim=8)  # ассоциативная память на 64 слота размерности 8
        self.compute_budget = float(compute_budget)  # бюджет вычислений
        self.risk_rem = float(risk_limit)  # остаток риска
        self.unsolved_objective = 100.0  # цель: доля нерешённых задач (изначально 100)
        self.evals = 0  # число вычислений программ
        self.candidates_generated = 0  # число сгенерированных кандидатов

    def _expert_index(self) -> dict[str, int]:
        """Построить индекс экспертов для гамма-кодирования: чем чаще эксперта вызывают, тем меньше его код.
        
                Число вызовов — сколько раз имя эксперта встречается среди вызываемых
                в программах всех экспертов. Сортировка по убыванию числа вызовов, затем по
                длине кода, затем по имени. Возвращает словарь имя → номер позиции."""
        """Zipf frequency-ordered index: most-called experts get smallest gamma codes."""
        counts = {nm: 0 for nm in self.experts}  # счётчики вызовов, изначально нулевые
        for ex in self.experts.values():  # по всем экспертам
            if ex.program.expr is not None:  # у эксперта есть программа
                for c in ex.program.expr.callees():  # по именам, которые он вызывает
                    if c in counts:  # вызываемый — известный эксперт
                        counts[c] += 1  # счётчик вызовов + 1
        ordered = sorted(  # отсортированные имена
            self.experts.keys(),  # ключи — имена экспертов
            key=lambda nm: (-counts[nm], self.experts[nm].code_length, nm),  # ключ сортировки: (−число вызовов, длина кода, имя)
        )  # закрытие вызова sorted
        return {nm: i for i, nm in enumerate(ordered)}  # имя → номер в отсортированном порядке

    def _snapshot_state_with_optimizer(self) -> dict[str, Any]:  # возвращает словарь снимка
        """Полный снимок изменяемого состояния: общего блока весов, маршрутизатора и памяти.
        
                Возвращает словарь с ключами "core", "router", "ram" (значения — их снимки)."""
        """Полный снимок состояния: общий блок весов, маршрутизатор и память."""
        return {  # словарь снимка
            "core": self.core.snapshot(),  # снимок общего блока весов
            "router": self.router.snapshot(),  # снимок маршрутизатора
            "ram": self.ram.snapshot(),  # снимок памяти
        }

    def _restore_state_with_optimizer(self, snap: dict[str, Any]) -> None:  # ничего не возвращает
        """Восстановить состояние из снимка, сделанного _snapshot_state_with_optimizer.
        
                snap — словарь с ключами "core", "router", "ram"."""
        self.core.restore(snap["core"])  # восстановить общий блок весов
        self.router.restore(snap["router"])  # восстановить маршрутизатор
        self.ram.restore(snap["ram"])  # восстановить память

    def _record_ram_fingerprint(  # Записать в память отпечаток таблицы значений задачи: вектор из 8 признаков по первым 32 точкам области.
        self, name: str, domain: list[tuple[int, ...]], ys: dict[tuple[int, ...], int]  # вторая строка параметров
    ) -> None:  # ничего не возвращает
        vals = [ys[r] for r in domain[:32]]  # значения первых 32 точек: vals = [ys[r] для r из domain[:32]]
        chunks = boundary_chunks(lambda v: v == 0 or abs(v) > 4, vals)  # разбиение значений на сегменты: элемент с v = 0 или |v| > 4 начинает новый сегмент
        heads = boundary_heads(lambda v: v == 0 or abs(v) > 4, vals)  # первые элементы сегментов
        sig = np.zeros(8, dtype=np.float64)  # нулевой вектор признаков из 8 компонент
        sig[0] = float(len(domain))  # sig[0] — размер области
        sig[1] = float(len(chunks))  # sig[1] — число сегментов
        sig[2] = float(len(heads))  # sig[2] — число голов
        for k in range(min(5, len(heads))):  # по первым min(5, число голов) головам
            sig[3 + k] = float(heads[k])  # sig[3 + k] = k-я голова
        slot = sum(ord(c) for c in name) % 64  # слот памяти: остаток от суммы кодов символов имени при делении на 64
        self.ram.store(slot, sig)  # записать вектор признаков sig в слот slot

    def learn(  # Выучить задачу spec: проверить готовые эксперты, затем синтезировать программу и принять её только при точном совпадении на train, val и всей области.
        self,  # сам экземпляр
        spec: TaskSpec,  # spec — задача: имя, арность, области, учитель
        seed: int = 0,  # seed — зерно разбиения на train и val
        train_fraction: float = 0.70,  # доля train в области
        poison_one_label: bool = False,  # признак «отравленной» метки: задача немедленно удерживается
        train_neural: bool = True,  # обучать ли нейросетевой адаптер
        deadline: Optional[float] = None,  # предельное время (monotonic) или None
    ) -> dict[str, Any]:  # возвращает словарь-событие
        if spec.name in self.experts:  # эксперт с именем задачи уже есть — переиспользовать
            return {  # словарь события
                "version": self.version,  # версия схемы
                "task": spec.name,  # имя задачи
                "action": "REUSE",  # действие «переиспользование»
                "reason": "already learned",  # причина «уже выучено»
                "code_length": self.experts[spec.name].code_length,  # длина кода готового эксперта
            }
        domain = spec.inputs()  # область — все допустимые входы задачи
        if not domain:  # область пуста
            return self._hold(spec, "EMPTY_VALID_DOMAIN", 0, 0)  # удержать задачу: пустая допустимая область
        # запомнить семейство задачи для языкового критерия синтеза (атрибут family или пустая строка)
        _CURRENT_FAMILY[0] = getattr(spec, "family", "")  # запомнить семейство задачи (атрибут family или пустая строка)
        # кандидат на всю область: значения учителя по ВСЕЙ объявленной области;
        # каждый кандидат всё равно проходит полную проверку на train, val и всей области
        # при принятии
        _FULL_CAND[0] = None  # кандидата на всю область пока нет
        if _CURRENT_FAMILY[0] == "language" and spec.arity == 1:  # языковое семейство с одним аргументом
            try:  # попытка построить кандидата
                yy = np.array([int(spec.teacher(*r)) for r in domain],  # значения учителя yy = [int(spec.teacher(*r))] по всей области
                              dtype=np.int64)  # тип значений — целые 64-битные
                Xd = np.asarray(domain, dtype=np.int64)  # матрица входов Xd: (|domain|, 1)
                if set(np.unique(yy)) <= {0, 1}:  # значения только 0 и 1
                    _FULL_CAND[0] = _sparse_membership(Xd, yy)  # разреженная принадлежность как кандидат на всю область
                else:  # иначе
                    # модульно-аффинный или кусочный кандидат на всей области
                    # (для алфавитного типа выходов)
                    _FULL_CAND[0] = _mod_affine_expr(Xd, yy, 1)  # модульно-аффинный кандидат по арности 1
            except Exception:  # ошибка вычисления
                _FULL_CAND[0] = None  # кандидата нет

        opt_snap = self._snapshot_state_with_optimizer()  # снимок состояния до обучения

        rng = np.random.default_rng(seed)  # генератор случайных чисел с зерном seed
        perm = rng.permutation(len(domain))  # случайная перестановка номеров точек области
        if spec.family == "BoolVenn" or (  # булева задача или (все области из 0 и 1 и не более 16 точек)
            all(set(d) <= {0, 1} for d in spec.domains) and len(domain) <= 16  # условие: каждая область d ⊆ {0, 1} и |domain| ≤ 16
        ):  # закрытие условия
            n_train = max(1, len(domain) - 1)  # на train всё, кроме одной точки: n_train = max(1, |domain| − 1)
        else:  # иначе
            n_train = max(  # n_train = max(1, min(|domain| − 1, …)) — не больше |domain| − 1
                1, min(len(domain) - 1, int(round(len(domain) * train_fraction)))  # округлённая доля: int(round(|domain|·train_fraction))
            )  # закрытие min
        train_x = [domain[int(i)] for i in perm[:n_train]]  # входы train — первые n_train точек перестановки
        val_x = [domain[int(i)] for i in perm[n_train:]]  # входы val — остальные точки перестановки
        all_y = {xs: int(spec.teacher(*xs)) for xs in domain}  # значения учителя на всей области: all_y[xs] = int(spec.teacher(*xs))
        train_y = [all_y[xs] for xs in train_x]  # значения train
        val_y = [all_y[xs] for xs in val_x]  # значения val

        self.ledger.allocate_stream(spec.name, len(train_x) + len(val_x))  # выделить задаче поток аудита на |train| + |val| точек

        if poison_one_label and train_y:  # отравленная метка и train непуст
            self.ledger.record_reject(spec.name)  # зафиксировать отклонение задачи в аудите
            self._restore_state_with_optimizer(opt_snap)  # откатить состояние к снимку
            return self._hold(  # удержать задачу
                spec, "POISONED_LABEL_INJECTION", len(train_x), len(val_x)  # причина «внедрение отравленной метки», размеры train и val
            )  # закрытие вызова

        alpha_step = min(0.0005, self.ledger.remaining() * 0.5)  # шаг значимости: alpha_step = min(0.0005, 0.5·остаток бюджета значимости)
        ok_alpha, _radius = self.ledger.spend_anytime(spec.name, alpha_step, len(val_x))  # потратить alpha_step за |val| точек; ok_alpha — успех, _radius — радиус (не используется)
        if not ok_alpha:  # бюджет значимости исчерпан
            self._restore_state_with_optimizer(opt_snap)  # откатить состояние
            return self._hold(spec, "AUDIT_ALPHA_EXHAUSTED", len(train_x), len(val_x))  # удержать задачу: бюджет значимости исчерпан

        adapter: Optional[FactorizedAdapter] = None  # нейросетевой адаптер (по умолчанию нет)
        if train_neural and val_x and spec.family != "BoolVenn":  # нужно обучение нейросети, val непуст и задача не «BoolVenn»
            adapter = self.core.fit_residual(  # обучить адаптер остатка общего блока весов
                train_x, train_y, val_x, val_y, router=self.router  # на train и val с маршрутизатором
            )

        ex_idx = self._expert_index()  # индекс экспертов для стоимости в битах

        for nm, ex in sorted(self.experts.items()):  # по готовым экспертам в порядке имён
            if ex.program.arity != spec.arity:  # другая арность
                continue  # пропустить
            if not all(ex.supports(xs) for xs in domain):  # эксперт поддерживает не всю область задачи
                continue  # пропустить
            self.evals += len(train_x)  # число вычислений + |train|
            try:  # попытка проверить
                if any(  # если есть ошибка на train
                    ex.program.eval(xs, self.experts) != yy  # значение эксперта не равно значению учителя
                    for xs, yy in zip(train_x, train_y)  # по парам (вход, значение) train
                ):  # закрытие any
                    continue  # эксперт не подходит
                if any(  # если есть ошибка на val
                    ex.program.eval(xs, self.experts) != yy  # значение не равно yy
                    for xs, yy in zip(val_x, val_y)  # по парам val
                ):  # закрытие any
                    continue  # не подходит
                if any(ex.program.eval(xs, self.experts) != all_y[xs] for xs in domain):  # если есть ошибка хотя бы на одной точке всей области
                    continue  # не подходит
            except Exception:  # ошибка вычисления
                continue  # не подходит
            ref_bits = TOKEN_BITS + gamma_bits(ex_idx[nm] + 1)  # стоимость вызова: TOKEN_BITS + гамма-код номера (ex_idx[nm] + 1)
            ev = {  # словарь события
                "version": self.version,  # версия
                "task": spec.name,  # имя задачи
                "action": "REUSE",  # действие «переиспользование»
                "reason": f"verified-expert-coverage:{nm}",  # причина: покрытие проверенным экспертом nm
                "program": ex.program.pretty(),  # текст программы в читаемом виде
                "code_length": ref_bits,  # длина кода
                "train_examples": len(train_x),  # число точек train
                "validation_examples": len(val_x),  # число точек val
                "exact_accuracy": 1.0,  # точность 1.0
            }
            self.growth_log.append(ev)  # записать событие в журнал роста
            self.ledger.record_floor(spec.name, 1.0)  # зафиксировать нижнюю границу качества 1.0
            return ev  # вернуть событие

        def try_commit(  # Принять программу program как решение задачи, если она точна на train, val и всей области.
            program: Program, reason: str, parents: tuple[str, ...] = ()  # параметры
        ) -> Optional[dict[str, Any]]:  # возвращает событие или None
            try:  # попытка проверки
                self.evals += len(train_x)  # число вычислений + |train|
                if any(  # если есть ошибка на train
                    program.eval(xs, self.experts) != yy  # значение программы не равно yy
                    for xs, yy in zip(train_x, train_y)  # по парам train
                ):  # закрытие any
                    return None  # отклонить кандидата
                if val_x and any(  # val непуст и есть ошибка на val
                    program.eval(xs, self.experts) != yy for xs, yy in zip(val_x, val_y)  # значение не равно yy
                ):  # закрытие any
                    return None  # отклонить
                if any(program.eval(xs, self.experts) != all_y[xs] for xs in domain):  # есть ошибка хотя бы на одной точке всей области
                    return None  # отклонить
            except Exception:  # ошибка вычисления
                return None  # отклонить
            bits = program.bits(ex_idx)  # длина кода программы в битах (с учётом индекса экспертов)
            expert = Expert(  # новый эксперт
                spec.name,  # имя
                spec.family,  # семейство
                program,  # программа
                spec.domains,  # области аргументов
                spec.valid_rule,  # правило корректности
                bits,  # длина кода
                adapter,  # адаптер общих весов
                parents,  # родители
            )
            self.experts[spec.name] = expert  # записать эксперта в библиотеку
            self.version += 1  # версия + 1
            self.total_commits += 1  # принятых изменений + 1
            self._record_ram_fingerprint(spec.name, domain, all_y)  # записать отпечаток таблицы значений в память
            drop = 1.0  # снижение цели за принятого эксперта: drop = 1.0
            # точное уменьшение цели в сертификате: obj_after обязан равняться obj_before − drop
            # БЕЗ отсечки в нуле — нижняя граница 0.0 ломает это соотношение, как только число
            # принятых изменений превосходит начальный номинальный бюджет (obj0 = 100), и тогда
            # проверка сертификатов проваливалась бы навсегда. Цель может законно стать
            # отрицательной (задачи «переплачены»).
            obj_after = self.unsolved_objective - drop  # цель после шага: obj_after = unsolved_objective − drop
            b_spend = min(self.compute_budget, 0.5)  # расход бюджета вычислений: b_spend = min(остаток compute_budget, 0.5)
            r_spend = min(self.risk_rem, 0.01)  # расход риска: r_spend = min(остаток risk_rem, 0.01)
            self.compute_budget -= b_spend  # остаток бюджета вычислений − b_spend
            self.risk_rem -= r_spend  # остаток риска − r_spend
            floors_ok = self.ledger.record_floor(spec.name, 1.0)  # зафиксировать нижнюю границу качества 1.0; floors_ok — все границы соблюдены
            cert = CertStepGen(  # сертификат шага
                version=self.version,  # версия
                task=spec.name,  # имя задачи
                action="COMMIT",  # действие «принятие»
                obj_before=self.unsolved_objective,  # цель до
                obj_after=obj_after,  # цель после
                drop=drop,  # снижение
                budget_rem=self.compute_budget,  # остаток бюджета вычислений
                budget_spend=b_spend,  # расход бюджета вычислений
                risk_rem=self.risk_rem,  # остаток риска
                risk_spend=r_spend,  # расход риска
                delta_rem=self.ledger.remaining(),  # остаток значимости
                delta_spend=alpha_step,  # расход значимости
                floors_kept=floors_ok,  # нижние границы соблюдены
            )
            self.unsolved_objective = obj_after  # обновить цель
            self.cert_log.append(cert)  # записать сертификат в журнал
            ev = {  # словарь события
                "version": self.version,  # версия
                "task": spec.name,  # имя задачи
                "action": "COMMIT",  # действие «принятие»
                "reason": reason,  # причина
                "program": program.pretty(),  # текст программы
                "code_length": bits,  # длина кода
                "train_examples": len(train_x),  # число точек train
                "validation_examples": len(val_x),  # число точек val
                "exact_accuracy": 1.0,  # точность 1.0
            }
            self.growth_log.append(ev)  # записать событие в журнал роста
            return ev  # вернуть событие

        if spec.family == "BoolVenn":  # булевы задачи: перебор всех 16 булевых функций двух аргументов
            for m in range(16):  # по маске m = 0..15
                ev = try_commit(Program(2, venn_mask=m), "boolvenn-exact")  # программа с функцией Венна по маске m
                if ev is not None:  # принято
                    return ev  # вернуть событие

        if all(set(d) <= {0, 1} for d in spec.domains) and 1 <= spec.arity <= 6:  # все области из 0 и 1 и арность от 1 до 6
            if set(train_y) <= {0, 1}:  # значения train только 0 и 1
                full_inputs = list(itertools.product((0, 1), repeat=spec.arity))  # все 2^arity наборов из 0 и 1
                train_map = dict(zip(train_x, train_y))  # словарь вход → значение на train
                missing = [r for r in full_inputs if r not in train_map]  # наборы, отсутствующие в train
                if len(missing) <= 1:  # не хватает не более одного набора
                    anf_cands = []  # кандидаты (оценка, программа)
                    for fill in (0,) if not missing else (0, 1):  # заполнение недостающего набора: только 0, если ничего не хватает, иначе 0 и 1
                        tt = [train_map.get(r, fill) for r in full_inputs]  # таблица истинности: tt = train_map[r], для недостающего — fill
                        coeffs = anf_transform(tt)  # коэффициенты АНФ (полинома Жегалкина)
                        active_s = [s for s, c in enumerate(coeffs) if c]  # множества с ненулевым коэффициентом
                        max_deg = max((bin(s).count("1") for s in active_s), default=0)  # степень многочлена — максимум числа единиц в двоичной записи множества
                        cand_e = anf_to_expr(coeffs, spec.arity)  # выражение по коэффициентам АНФ
                        cand_p = Program(spec.arity, cand_e)  # программа-кандидат
                        score = (max_deg, len(active_s), cand_p.bits(ex_idx))  # оценка: (степень, число слагаемых, длина кода в битах) — меньше лучше
                        anf_cands.append((score, cand_p))  # добавить кандидата
                    anf_cands.sort(key=lambda it: it[0])  # отсортировать по оценке
                    for _, cand_p in anf_cands:  # по кандидатам в порядке оценки
                        ev = try_commit(cand_p, "anf-zhegalkin-butterfly+full-support")  # принять полином Жегалкина
                        if ev is not None:  # принято
                            return ev  # вернуть событие

        for p in candidate_programs(spec.arity):  # по программам-шаблонам для арности
            ev = try_commit(p, "direct-template+full-support")  # принять шаблон
            if ev is not None:  # принято
                return ev  # вернуть событие

        if 1 <= spec.arity <= 3 and not spec.family.startswith("Bool"):  # арность от 1 до 3 и небулево семейство
            if spec.arity == 1:  # один аргумент
                coeffs = _exact_poly_fit([r[0] for r in train_x], train_y, max_degree=5)  # точный рациональный многочлен степени не выше 5: y = Σ c_i·x^i
                if coeffs is not None:  # многочлен найден
                    ev = try_commit(  # принять
                        Program(1, Expr("poly", (Expr.var(0),), coeffs)),  # программа poly(x_0, коэффициенты)
                        "exact-rational-poly+full-support",  # причина
                    )  # закрытие вызова
                    if ev is not None:  # принято
                        return ev  # вернуть событие
            else:  # иначе — много аргументов
                terms = _exact_multivariate_poly_fit(  # точный многочлен от многих переменных степени не выше 4
                    train_x, train_y, spec.arity, max_degree=4  # арность и предел степени
                )  # закрытие вызова
                if terms is not None:  # найден
                    ev = try_commit(  # принять
                        Program(spec.arity, Expr("poly_nd", (), terms)),  # программа poly_nd с перечнем термов
                        "exact-multivar-poly+full-support",  # причина
                    )  # закрытие вызова
                    if ev is not None:  # принято
                        return ev  # вернуть событие

        pw = exact_piecewise_poly_fit(train_x, train_y)  # точная кусочно-полиномиальная подгонка
        if pw is not None:  # найдена
            ev = try_commit(  # принять
                Program(spec.arity, pw), "exact-piecewise-poly+full-support"  # программа с найденным выражением
            )  # закрытие вызова
            if ev is not None:  # принято
                return ev  # вернуть событие

        lib_list = [(ex_idx[nm], nm, self.experts[nm]) for nm in self.experts]  # библиотека для синтеза: (номер индекса, имя, эксперт)
        hit_expr, n_chk = synthesize_batched_mdl(  # перебор методом минимальной длины описания
            spec.arity,  # арность
            train_x,  # входы train
            train_y,  # значения train
            val_x,  # входы val
            val_y,  # значения val
            library_experts=lib_list,  # эксперты библиотеки
            max_evals=self.node_budget,  # предел проверенных выражений — бюджет узлов
            max_bits=160,  # бюджет бит на выражение 160
            deadline=deadline,  # предельное время
        )
        self.candidates_generated += n_chk  # счётчик сгенерированных кандидатов + число проверенных
        if hit_expr is not None:  # выражение найдено
            ev = try_commit(  # принять
                Program(spec.arity, hit_expr), "batched-mdl-mitm-dag+full-support"  # программа из найденного выражения
            )  # закрытие вызова
            if ev is not None:  # принято
                return ev  # вернуть событие

        self._restore_state_with_optimizer(opt_snap)  # ничего не принято: откатить состояние к снимку
        self.ledger.record_reject(spec.name)  # зафиксировать отклонение в аудите
        return self._hold(spec, "NO_VALIDATED_PROGRAM", len(train_x), len(val_x))  # удержать задачу: нет проверенной программы

    def _hold(  # Удержать задачу: записать сертификат и событие без изменения цели и бюджетов.
        self, spec: TaskSpec, reason: str, ntrain: int, nval: int  # параметры
    ) -> dict[str, Any]:  # возвращает словарь-событие
        self.total_holds += 1  # счётчик удержаний + 1
        cert = CertStepGen(  # сертификат шага
            version=self.version,  # версия
            task=spec.name,  # имя задачи
            action="HOLD",  # действие «удержание»
            obj_before=self.unsolved_objective,  # цель до
            obj_after=self.unsolved_objective,  # цель после — та же
            drop=0.0,  # снижение 0
            budget_rem=self.compute_budget,  # остаток бюджета вычислений
            budget_spend=0.0,  # расход 0
            risk_rem=self.risk_rem,  # остаток риска
            risk_spend=0.0,  # расход 0
            delta_rem=self.ledger.remaining(),  # остаток значимости
            delta_spend=0.0,  # расход 0
            floors_kept=True,  # нижние границы соблюдены
        )
        self.cert_log.append(cert)  # записать сертификат в журнал
        ev = {  # словарь события
            "version": self.version,  # версия
            "task": spec.name,  # имя задачи
            "action": "HOLD",  # действие «удержание»
            "reason": reason,  # причина
            "program": "",  # программы нет
            "code_length": 0,  # длина кода 0
            "train_examples": ntrain,  # число точек train
            "validation_examples": nval,  # число точек val
            "exact_accuracy": 0.0,  # точность 0.0
        }
        self.growth_log.append(ev)  # записать событие в журнал роста
        return ev  # вернуть событие

    def compress_library_mdl(self) -> dict[str, Any]:
        """Global MDL Call-DAG Refactoring with structural pattern unification & Zipf re-indexing."""
        ex_idx = self._expert_index()  # индекс имён экспертов для кодирования программ битами
        for ex in self.experts.values():  # по всем экспертам
            ex.code_length = ex.program.bits(ex_idx)  # длина кода эксперта = число бит его программы: bits = Σ длина(оператор)
        before_bits = sum(e.code_length for e in self.experts.values())  # суммарная длина до сжатия: Σ_e code_length(e)
        refactored = 0  # число переформированных экспертов

        for target_name, target_ex in list(self.experts.items()):  # по экспертам библиотеки (копия списка, чтобы менять словарь во время обхода)
            if target_ex.program.expr is None:  # у цели нет программы — нечего сжимать
                continue  # у эксперта нет программы
            cur_expr = target_ex.program.expr  # текущее выражение цели
            subs = [  # поддеревья-кандидаты на замену вызовом
                s  # элемент списка
                for s in cur_expr.subtrees()  # по всем поддеревьям cur_expr
                if s.op not in ("var", "const", "call") and s.cost >= 2  # не лист и не вызов, стоимость не меньше 2
            ]  # закрытие списка
            if not subs:  # заменять нечего
                continue  # следующий эксперт
            domain = [  # область задачи — все допустимые входы
                xs  # набор аргументов
                for xs in itertools.product(*target_ex.domains)  # декартово произведение областей эксперта
                if target_ex.supports(xs)  # только входы, которые эксперт поддерживает
            ]  # закрытие списка
            for sub in subs:  # по поддеревьям-кандидатам
                for lib_name, lib_ex in list(self.experts.items()):  # по экспертам библиотеки (копия списка)
                    if lib_name == target_name or lib_ex.program.expr is None:  # сам эксперт или эксперт без программы
                        continue  # пропустить
                    if _dag_reaches(self.experts, lib_name, target_name):  # замена создала бы цикл: lib_ex вызывает (прямо или косвенно) целевого эксперта
                        continue  # пропустить
                    bindings: dict[int, Expr] = {}  # связывания дыр образца: номер → поддерево
                    if not _match_pattern(lib_ex.program.expr, sub, bindings):  # программа библиотечного эксперта не сопоставляется с поддеревом
                        continue  # пропустить
                    if any(i not in bindings for i in range(lib_ex.program.arity)):  # какая-то дыра не связалась
                        continue  # пропустить
                    call_args = tuple(bindings[i] for i in range(lib_ex.program.arity))  # аргументы вызова — поддеревья, связанные с дырами 0..arity−1
                    call_e = Expr("call", call_args, lib_name)  # вызов lib_name с найденными аргументами
                    if call_e.bits(ex_idx) >= sub.bits(ex_idx):  # вызов не дешевле поддерева
                        continue  # пропустить
                    cand_expr = cur_expr.replace_subtree(sub, call_e)  # заменить поддерево sub вызовом call_e
                    cand_prog = Program(target_ex.program.arity, cand_expr)  # программа-кандидат той же арности
                    new_bits = cand_prog.bits(ex_idx)  # её длина в битах
                    if new_bits >= target_ex.code_length:  # не короче прежней
                        continue  # пропустить
                    try:  # попытка проверки
                        if all(  # если на всей области
                            cand_prog.eval(xs, self.experts)  # значение программы-кандидата
                            == target_ex.program.eval(xs, self.experts)  # совпадает со значением прежней программы
                            for xs in domain  # по всем входам области
                        ):  # закрытие all
                            target_ex.program = cand_prog  # принять новую программу
                            target_ex.code_length = new_bits  # запомнить новую длину кода
                            cur_expr = cand_expr  # текущее выражение — новое
                            refactored += 1  # счётчик преобразований + 1
                    except Exception:  # ошибка вычисления
                        pass  # оставить как было

        ex_idx = self._expert_index()  # пересчитать индекс после преобразований
        for ex in self.experts.values():  # по всем экспертам
            ex.code_length = ex.program.bits(ex_idx)  # пересчитать длину кода по новому индексу
        after_bits = sum(e.code_length for e in self.experts.values())  # суммарная длина после
        return {  # итог словарём
            "before_bits": before_bits,  # бит до
            "after_bits": after_bits,  # бит после
            "refactored": refactored,  # число преобразований
        }

    def predict(self, name: str, *xs: int, mode: str = "exact") -> dict[str, Any]:
        """Предсказать значение задачи name на входе xs.
        
                mode "exact" — точное значение программы эксперта; mode "neural" — приближение
                нейросетевым адаптером (или точной тернарной головой функции Венна, если адаптера
                нет). Если эксперта нет или вход вне его области — «ABSTAIN» (отказ). Возвращает
                словарь со статусом status, значением value, источником source и версией модели."""
        ex = self.experts.get(name)  # эксперт по имени
        if ex is None:  # эксперта нет
            return {"status": "ABSTAIN", "reason": "unknown operation"}  # отказ: неизвестная операция
        if not ex.supports(xs):  # вход вне области эксперта
            return {"status": "ABSTAIN", "reason": "outside certified finite domain"}  # отказ: вне проверенной конечной области
        if mode == "exact":  # точный режим
            return {  # словарь ответа
                "status": "OK",  # статус «ок»
                "value": ex.program.eval(xs, self.experts),  # значение программы
                "source": "verified-symbolic-dag",  # источник — проверенный символьный граф вызовов
                "model_version": self.version,  # версия модели
            }
        if mode == "neural":  # нейросетевой режим
            if ex.adapter is None:  # адаптера нет
                if ex.program.venn_mask is not None:  # но у программы есть маска функции Венна
                    return {  # словарь ответа
                        "status": "APPROXIMATE",  # статус «приближение»
                        "value": float(venn_head_eval(ex.program.venn_mask, xs)),  # точное значение тернарной головы функции Венна как число с плавающей точкой
                        "source": "venn-head-exact-ternary",  # источник — точная тернарная голова
                        "model_version": self.version,  # версия модели
                    }
                return {"status": "ABSTAIN", "reason": "neural adapter unavailable"}  # отказ: нейросетевой адаптер недоступен
            return {  # словарь ответа
                "status": "APPROXIMATE",  # статус «приближение»
                "value": ex.adapter.predict(self.core, xs),  # предсказание адаптера на общем блоке весов
                "source": "factorized-core-bitalloc-rvq",  # источник — общий блок с распределением бит
                "model_version": self.version,  # версия модели
            }
        raise ValueError("mode must be exact or neural")  # ошибка: режим обязан быть exact или neural

    def verify_all(self, specs: Sequence[TaskSpec]) -> dict[str, Any]:
        """Проверить все задачи из specs по их экспертам и вернуть итоги.
        
                Покрытой считается задача, для которой есть принятый эксперт или записанное
                в журнале роста переиспользование чужого проверенного эксперта. Для каждой
                покрытой задачи считается число точек, где программа совпадает с учителем;
                точность accuracy = correct / max(1, total). Непокрытые задачи перечислены
                в missing_tasks."""
        total = correct = 0  # всего проверено и совпадений
        # карта переиспользования: задача, решённая чужим проверенным экспертом, не имеет
        # собственной записи в self.experts; без этой карты число покрытых задач
        # занижалось бы, а вместе с ним и проверка нижних границ качества
        # Значения расходились на одну задачу.
        coverage: dict[str, str] = {}  # задача → имя эксперта, который её покрыл
        for ev in self.growth_log:  # по событиям журнала роста
            if ev.get("action") == "REUSE" and "verified-expert-coverage:" in ev.get("reason", ""):  # событие «переиспользование» с причиной «покрытие проверенным экспертом»
                coverage[ev["task"]] = ev["reason"].split(":", 1)[1]  # имя покрывшего эксперта — часть причины после первого двоеточия
        # непокрытая задача обязана снижать точность, а не пропускаться: иначе набор
        # задач {A, B} при экспертах {A, посторонний} давал бы точность 1.0.
        # Покрытая задача — принятый эксперт или проверенное переиспользование
        # Покрытая задача — принятый эксперт или проверенное переиспользование.
        missing: list[str] = []  # список непокрытых задач
        for sp in specs:  # по задачам
            ex = self.experts.get(sp.name) or self.experts.get(coverage.get(sp.name, ""))  # эксперт задачи или эксперт, покрывший её через переиспользование
            if ex is None:  # эксперта нет
                missing.append(sp.name)  # задача непокрыта
                continue  # следующая задача
            rows = sp.inputs()  # все входы задачи
            ok = sum(  # число совпадений с учителем
                int(ex.program.eval(xs, self.experts) == int(sp.teacher(*xs)))  # 1, если программа дала ответ учителя, иначе 0
                for xs in rows  # по входам
            )
            total += len(rows)  # всего + число входов
            correct += ok  # совпадений + ok
        return {  # итог словарём
            "checked": total,  # проверено точек
            "correct": correct,  # совпадений
            "accuracy": correct / max(1, total),  # точность correct / max(1, total)
            "committed": len(self.experts),  # число принятых экспертов
            "expected_tasks": len(specs),  # число задач в наборе
            "covered_tasks": len(specs) - len(missing),  # число покрытых задач
            "missing_tasks": missing,  # непокрытые задачи
            "suite_complete": not missing,  # набор покрыт целиком, если непокрытых нет
        }

    def verify_certificates(self) -> dict[str, Any]:
        """4328	Проверить сертификаты и журнал аудита: согласованность цели, неизменность состояния при удержаниях, корректность принятий, непересечение потоков аудита и полнота истории.
        
                Возвращает словарь признаков: telescope_ok — цель не превышает
                obj0 − Σ drop (сдвиг 1e−9); holds_ok — каждое удержание ничего не меняло;
                commits_ok — каждое принятие снизило цель и не ушло в минус по бюджетам;
                ledger_ok — потоки аудита непересекающиеся и чистые; state_complete — состояние
                управления загружено полностью; history_ok — число записей равно счётчикам;
                all_ok — все признаки вместе."""
        obj0 = self.cert_log[0].obj_before if self.cert_log else self.unsolved_objective  # начальная цель obj0 — цель до первого шага; если журнала нет, текущая цель
        total_drops = sum(c.drop for c in self.cert_log)  # сумма всех снижений Σ drop
        telescope_ok = self.unsolved_objective <= obj0 - total_drops + 1e-9  # цель не превосходит obj0 − Σ drop (допуск 1e−9)
        holds_ok = all(  # все удержания корректны
            c.drop == 0.0 and c.obj_after == c.obj_before and c.budget_spend == 0.0  # снижение 0, цель не изменилась и расход бюджета 0
            for c in self.cert_log  # по шагам журнала
            if c.action == "HOLD"  # только удержания
        )
        commits_ok = all(  # все принятия корректны
            c.drop > 0.0  # снижение больше 0
            and c.budget_rem >= -1e-12  # остаток бюджета вычислений не отрицателен (допуск 1e−12)
            and c.risk_rem >= -1e-12  # остаток риска не отрицателен
            and c.floors_kept  # нижние границы соблюдены
            for c in self.cert_log  # по шагам журнала
            if c.action == "COMMIT"  # только принятия
        )
        ledger_ok = self.ledger.verify_disjoint_and_clean()  # журнал аудита: потоки непересекающиеся и чистые
        # полнота истории: очищенный журнал сертификатов не должен читаться
        # как успешное доказательство
        state_complete = bool(getattr(self, "governance_state_complete", False))  # признак полноты состояния (по умолчанию False)
        logged_commits = sum(c.action == "COMMIT" for c in self.cert_log)  # число записей о принятиях
        logged_holds = sum(c.action == "HOLD" for c in self.cert_log)  # число записей об удержаниях
        history_ok = (  # история согласована
            logged_commits == self.total_commits  # принятий в журнале столько же, сколько счётчик
            and logged_holds == self.total_holds  # удержаний столько же, сколько счётчик
            and len(self.cert_log) == self.total_commits + self.total_holds  # и длина журнала равна сумме счётчиков
        )
        all_ok = (telescope_ok and holds_ok and commits_ok and ledger_ok  # все признаки вместе
                  and state_complete and history_ok)  # (продолжение условия)
        return {  # итог словарём
            "telescope_ok": telescope_ok,  # поле telescope_ok
            "holds_ok": holds_ok,  # поле holds_ok
            "commits_ok": commits_ok,  # поле commits_ok
            "ledger_ok": ledger_ok,  # поле ledger_ok
            "state_complete": state_complete,  # поле state_complete
            "history_ok": history_ok,  # поле history_ok
            "all_ok": all_ok,  # поле all_ok
        }

    def total_neural_bytes(self) -> int:
        """Суммарное число байт нейросетевой части: общий блок весов + голова функции Венна + адаптеры.
        
                Голова функции Венна занимает 13 байт, если есть хоть один эксперт семейства «BoolVenn».
                Возвращает целое число байт."""
        return _expert_total_neural_bytes(self)  # посчитать суммарный размер нейронной части в байтах

    def neural_quality_summary(self) -> dict[str, float]:
        """Сводка качества нейросетевых адаптеров: число адаптеров, средняя точность и средняя ошибка на val, суммарный размер.
        
                Средние считаются только по экспертам с адаптером; если адаптеров нет — 0.0.
                mean_val_mse — средний квадрат ошибки; total_neural_bytes — байты из total_neural_bytes()."""
        accs = [  # точности на val
            e.adapter.val_exact_acc_quant  # точность адаптера в сжатом (квантованном) виде
            for e in self.experts.values()  # по экспертам
            if e.adapter is not None  # с адаптером
        ]
        mses = [  # квадраты ошибки на val
            e.adapter.val_mse_quant  # средний квадрат ошибки адаптера в сжатом виде
            for e in self.experts.values()  # по экспертам
            if e.adapter is not None  # с адаптером
        ]
        return {  # итог словарём
            "num_adapters": len(accs),  # число адаптеров
            "mean_val_exact_acc": float(np.mean(accs)) if accs else 0.0,  # средняя точность: mean(accs)
            "mean_val_mse": float(np.mean(mses)) if mses else 0.0,  # средний квадрат ошибки: mean(mses)
            "total_neural_bytes": self.total_neural_bytes(),  # суммарное число байт нейросетевой части
        }

    def save(self, path: str | Path) -> None:
        """Сохранить модель в JSON-файл path: экспертов и состояние управления.
        
                Символических экспертов хватает для предсказания, но не для восстановления
                истории сертификатов и журнала аудита, поэтому format_version = 2 сохраняет
                и её: сертификаты, аудит, бюджеты, цель, веса общего блока, маршрутизатор
                и память. Файл создаётся с нужными каталогами; NaN запрещены."""
        """Сохранить модель в JSON-файл path: экспертов и состояние управления.
        
                Символических экспертов хватает для предсказания, но не для восстановления
                истории сертификатов и журнала аудита, поэтому format_version = 2 сохраняет
                и её: сертификаты, аудит, бюджеты, цель, веса общего блока, маршрутизатор
                и память. Файл создаётся с нужными каталогами; NaN запрещены."""
        gpm = self.core.gpm.snapshot_optimizer()  # снимок оптимизатора общего блока весов
        router_weights, router_loss = self.router.snapshot()  # веса маршрутизатора и его накопленные потери
        ledger = self.ledger  # журнал аудита
        governance_state = {  # состояние управления
            "schema": 1,  # версия схемы состояния
            "cert_log": [vars(c) for c in self.cert_log],  # сертификаты как словари полей
            "ledger": {  # журнал аудита
                "delta_total": ledger.delta_total,  # общий бюджет значимости
                "spent": dict(ledger.spent),  # расход по задачам
                "streams": {k: vars(v) for k, v in sorted(ledger.streams.items())},  # потоки как словари полей (по именам)
                "floors": dict(ledger.floors),  # нижние границы качества по задачам
                "rejected_tasks": sorted(ledger.rejected_tasks),  # список отклонённых задач
                "final_test_lo": ledger.final_test_lo,  # нижняя граница итоговой проверки
                "final_test_hi": ledger.final_test_hi,  # верхняя граница итоговой проверки
                "final_test_touched": ledger.final_test_touched,  # признак использования итоговой проверки
                "next_cursor_base": ledger.next_cursor_base,  # начало счётчика выборок
            },
            "compute_budget": self.compute_budget,  # бюджет вычислений
            "risk_rem": self.risk_rem,  # остаток риска
            "unsolved_objective": self.unsolved_objective,  # цель
            "evals": self.evals,  # число вычислений
            "candidates_generated": self.candidates_generated,  # число сгенерированных кандидатов
            "core_gpm": {  # общий блок весов
                "W": gpm["W"].tolist(),  # матрица W
                "U_basis": gpm["U_basis"].tolist(),  # базис U_basis
                "ef_v": gpm["ef_v"].tolist(),  # вектор ef_v
                "core_bytes": int(self.core.core_bytes),  # размер общего блока в байтах
            },
            "router": {  # маршрутизатор
                "K": int(self.router.K),  # число ветвей K
                "eta": float(self.router.eta),  # темп обучения eta
                "weights": router_weights.tolist(),  # веса ветвей
                "cum_loss": router_loss.tolist(),  # накопленные потери
            },
            "ram": self.ram.snapshot().tolist(),  # снимок памяти
        }
        data = {  # данные файла
            "format": MODEL_FILE_FORMAT,  # метка формата файла (строка-код: её читает load(); не переводится)
            "format_version": 2,  # версия формата 2
            "version": STATE_FORMAT,  # версия схемы
            "core_seed": self.core.seed,  # зерно общего блока
            "core_width": self.core.width,  # ширина общего блока
            "model_version": self.version,  # версия модели
            "node_budget": self.node_budget,  # бюджет узлов
            "total_commits": self.total_commits,  # принято изменений
            "total_holds": self.total_holds,  # удержаний
            "experts": {k: v.to_dict() for k, v in sorted(self.experts.items())},  # эксперты как словари (по именам)
            "theorems": [vars(t) for t in self.theorems],  # теоремы как словари полей
            "growth_log": self.growth_log,  # журнал роста
            "governance_state": governance_state,  # состояние управления
            "governance_state_complete": bool(self.governance_state_complete),  # признак полноты состояния управления
        }
        p = Path(path)  # путь к файлу
        p.parent.mkdir(parents=True, exist_ok=True)  # создать родительский каталог, если его нет
        p.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False),  # записать JSON с отступом 2, сортировкой ключей и без NaN
                     encoding="utf-8")  # кодировка UTF-8

    @classmethod  # метод класса — создаёт экземпляр
    def load(cls, path: str | Path) -> "GrowingHagiProcessor":
        """Загрузить модель из JSON-файла path, сохранённого save().n
                Метка формата в файле — «HAGI-MODEL» (строка-код); прежняя метка «HAGI-V9» тоже принимается. Восстанавливаются эксперты, теоремы, журнал роста
                и — только если состояние управления сохранено полностью и его схема та же —
                сертификаты, журнал аудита, маршрутизатор, память, веса общего блока, бюджеты
                и цель (иначе признак полноты сбрасывается в False). Формы и конечность
                чисел проверяются, при расхождении — ошибка ValueError. Возвращает процессор."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))  # прочитать и разобрать JSON
        if d.get("format") != MODEL_FILE_FORMAT and d.get("format") not in LEGACY_FILE_FORMATS[MODEL_FILE_FORMAT]:  # метка формата не опознана (ни текущая, ни прежняя)
            raise ValueError("not a HAGI model file")  # ошибка: это не файл модели HAGI
        m = cls(  # новый процессор
            seed=int(d.get("core_seed", 42)),  # зерно по умолчанию 42
            node_budget=int(d.get("node_budget", 8000)),  # бюджет узлов по умолчанию 8000
        )
        m.version = int(d.get("model_version", 0))  # версия модели
        m.total_commits = int(d.get("total_commits", 0))  # принято изменений
        m.total_holds = int(d.get("total_holds", 0))  # удержаний
        m.experts = {  # эксперты по именам
            k: Expert.from_dict(v, m.core) for k, v in d.get("experts", {}).items()  # каждый — из словаря, адаптеры привязываются к общему блоку m.core
        }  # закрытие словаря
        m.theorems = [Theorem(**t) for t in d.get("theorems", [])]  # теоремы из словарей полей
        m.growth_log = d.get("growth_log", [])  # журнал роста
        gs = d.get("governance_state")  # состояние управления (может отсутствовать)
        if (d.get("governance_state_complete") is not True  # если признак полноты не True
                or not isinstance(gs, dict) or gs.get("schema") != 1):  # или состояние не словарь, или версия схемы не 1
            # загрузка для предсказания совместима со старыми файлами, но отсутствие
            # истории аудита — не успешное пустое доказательство
            m.governance_state_complete = False  # сбросить признак полноты
            return m  # вернуть процессор без состояния управления
        ledger_data = gs.get("ledger")  # данные журнала аудита
        router_data = gs.get("router")  # данные маршрутизатора
        core_data = gs.get("core_gpm")  # данные общего блока
        ram_data = gs.get("ram")  # данные памяти
        if not all(isinstance(x, dict) for x in (ledger_data, router_data, core_data)):  # все три обязаны быть словарями
            raise ValueError("malformed governance state")  # ошибка: состояние управления повреждено
        m.cert_log = [CertStepGen(**row) for row in gs.get("cert_log", [])]  # сертификаты из словарей полей
        streams = {  # потоки аудита
            str(k): AuditStream(**v)  # по записям потоков
            for k, v in ledger_data.get("streams", {}).items()  # по ключам и значениям словаря потоков
        }  # закрытие словаря
        m.ledger = AuditLedger(  # журнал аудита
            delta_total=float(ledger_data["delta_total"]),  # общий бюджет значимости
            spent={str(k): float(v) for k, v in ledger_data.get("spent", {}).items()},  # расход по задачам
            streams=streams,  # потоки
            floors={str(k): float(v) for k, v in ledger_data.get("floors", {}).items()},  # нижние границы качества
            rejected_tasks=set(str(x) for x in ledger_data.get("rejected_tasks", [])),  # отклонённые задачи как множество строк
            final_test_lo=int(ledger_data.get("final_test_lo", 10_000_000)),  # нижняя граница итоговой проверки (по умолчанию 10^7)
            final_test_hi=int(ledger_data.get("final_test_hi", 20_000_000)),  # верхняя граница итоговой проверки (по умолчанию 2·10^7)
            final_test_touched=bool(ledger_data.get("final_test_touched", False)),  # признак использования итоговой проверки
            next_cursor_base=int(ledger_data.get("next_cursor_base", 0)),  # начало счётчика выборок
        )
        router_k = int(router_data["K"])  # число ветвей маршрутизатора
        if router_k != m.router.K:  # несовпадение с числом ветвей нового процессора
            raise ValueError("router expert count mismatch")  # ошибка: несовпадение числа экспертов маршрутизатора
        weights = np.asarray(router_data["weights"], dtype=np.float64)  # веса ветвей
        cum_loss = np.asarray(router_data["cum_loss"], dtype=np.float64)  # накопленные потери
        if weights.shape != m.router.weights.shape or cum_loss.shape != m.router.cum_loss.shape:  # несовпадение форм
            raise ValueError("router state shape mismatch")  # ошибка: несовпадение формы состояния маршрутизатора
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(cum_loss)):  # есть не конечные числа
            raise ValueError("non-finite router state")  # ошибка: не конечное состояние маршрутизатора
        m.router.eta = float(router_data["eta"])  # темп обучения
        m.router.restore((weights, cum_loss))  # восстановить (веса, потери)
        ram = np.asarray(ram_data, dtype=np.float64)  # матрица памяти
        if ram.shape != m.ram.mem.shape or not np.all(np.isfinite(ram)):  # форма не та или есть не конечные числа
            raise ValueError("RAM state shape/non-finite mismatch")  # ошибка: форма памяти не та или не конечные значения
        m.ram.restore(ram)  # восстановить память
        gpm_snapshot = {  # снимок оптимизатора общего блока
            "W": np.asarray(core_data["W"], dtype=np.float64),  # матрица W
            "U_basis": np.asarray(core_data["U_basis"], dtype=np.float64),  # базис U_basis
            "ef_v": np.asarray(core_data["ef_v"], dtype=np.float64),  # вектор ef_v
        }
        expected_gpm = m.core.gpm.snapshot_optimizer()  # ожидаемый снимок для сверки форм
        for key in ("W", "ef_v"):  # по W и ef_v
            if gpm_snapshot[key].shape != expected_gpm[key].shape:  # форма не та
                raise ValueError(f"core GPM state shape mismatch: {key}")  # ошибка: несовпадение формы состояния оптимизатора
            if not np.all(np.isfinite(gpm_snapshot[key])):  # есть не конечные числа
                raise ValueError(f"non-finite core GPM state: {key}")  # ошибка: не конечное состояние оптимизатора
        # ширина базиса U_basis переменная: закрепление добавляет направления
        u_basis = gpm_snapshot["U_basis"]  # базис
        if (u_basis.ndim != 2 or u_basis.shape[0] != expected_gpm["U_basis"].shape[0]  # если не матрица, или число строк не то
                or u_basis.shape[1] > u_basis.shape[0]):  # или число столбцов больше числа строк
            raise ValueError("core GPM U_basis shape mismatch")  # ошибка: форма базиса не та
        if not np.all(np.isfinite(u_basis)):  # есть не конечные числа
            raise ValueError("non-finite core GPM state: U_basis")  # ошибка: не конечный базис
        if u_basis.shape[1]:  # базис не пуст
            gram = u_basis.T @ u_basis  # матрица Грама U_basisᵀ·U_basis
            if not np.allclose(gram, np.eye(u_basis.shape[1]), atol=1e-7, rtol=0):  # отличается от единичной (допуск 1e−7)
                raise ValueError("core GPM U_basis is not orthonormal")  # ошибка: базис не ортонормирован
        m.core.gpm.restore_optimizer(gpm_snapshot)  # восстановить оптимизатор общего блока
        m.core.core_bytes = int(core_data.get("core_bytes", m.core.core_bytes))  # размер общего блока в байтах
        m.compute_budget = float(gs["compute_budget"])  # бюджет вычислений
        m.risk_rem = float(gs["risk_rem"])  # остаток риска
        m.unsolved_objective = float(gs["unsolved_objective"])  # цель
        m.evals = int(gs.get("evals", 0))  # число вычислений
        m.candidates_generated = int(gs.get("candidates_generated", 0))  # число сгенерированных кандидатов
        m.governance_state_complete = True  # состояние управления загружено полностью
        return m  # вернуть процессор

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'Expert', 'Theorem', 'TheoremMiner', 'GrowingHagiProcessor',  # имя интерфейса
]  # конец перечня
