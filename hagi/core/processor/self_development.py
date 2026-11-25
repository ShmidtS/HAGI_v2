"""Регулятор ресурсов и бесконечный цикл саморазвития символьного ядра.

Что делает файл:
  * ResourceGovernor — переводит свободную оперативную память и память
    видеокарты в ограниченные бюджеты поиска для одной задачи (число узлов
    перебора, размер популяции, размер порции проверки). Масштаб
    s = clamp((ram − floor) / (6 ГБ − floor), 0, 1); при ram < 1 ГБ
    s = min(s, max(0.05, (ram − floor) / (1 ГБ − floor))); если на
    видеокарте свободно меньше порога — s = 0. Бюджеты:
    node = max(1000, ⌊DEFAULT_NODE·s⌋), pop = max(2000, ⌊DEFAULT_POP·s⌋),
    chunk = max(8, ⌊CHUNK·min(1, 2s)⌋).
  * _propose_task_from_experts — строит новую задачу как композицию двух
    заученных программ outer(inner(...)) на конечном прямоугольном множестве
    входов, где обе родительские программы гарантированно определены.
  * verify_growth_lineage — исполняемая проверка: для каждой композиции на
    всём её объявленном множестве входов child(x) = outer(parent_input(x)).
  * grow_forever — цикл раундов: базовые задачи, самопредложенные композиции,
    добыча теорем, снимок состояния и журнал.

Ключевые имена: _ram_available_bytes, _vram_free_bytes, ResourceGovernor,
_propose_task_from_experts, verify_growth_lineage, grow_forever.
"""

from __future__ import annotations  # импорт стандартной библиотеки, нужный этой части
import itertools, json, os  # импорт стандартной библиотеки, нужный этой части
import sys, time  # импорт стандартной библиотеки, нужный этой части
from pathlib import Path  # импорт стандартной библиотеки, нужный этой части
from typing import TYPE_CHECKING, Any, Optional, Sequence  # импорт стандартной библиотеки, нужный этой части
import numpy as np  # импорт стандартной библиотеки, нужный этой части
from hagi.core.tasks.stored_tasks import (  # открыть явный импорт из предыдущей части: имена tasks
    TaskSpec  # TaskSpec — имя из предыдущей части
)  # конец импорта из предыдущей части
from typing import Any, Optional, Sequence  # имена из «stdlib:typing» (модуль, где они определены)
from hagi.core.synthesis.prefilter import DEFAULT_NODE_BUDGET, DEFAULT_POP_BUDGET, VALIDATION_CHUNK, _ensure_torch, _torch  # имена из «prefilter» (модуль, где они определены)
from hagi.core.tasks.stored_tasks import TaskSpec  # имена из «stored_tasks» (модуль, где они определены)
from hagi.core.synthesis.best_first import (  # открыть явный импорт из предыдущей части: имена best_first
    _composition_hint  # _composition_hint — имя из предыдущей части
)  # конец импорта из предыдущей части
from hagi.core.synthesis.induction import (  # открыть явный импорт из предыдущей части: имена induction
    TheoremMiner  # TheoremMiner — имя из предыдущей части
)  # конец импорта из предыдущей части
from hagi.core.processor.optimal_processor import OptimalMathProcessor  # имена из модуля «optimal_processor» (владелец определения)
if TYPE_CHECKING:  # только для проверки типов аннотаций (анализатор видит имя, импорта во время выполнения нет — нет цикла)
    from hagi.core.processor.benchmark import OptimalMathProcessor  # процессор из части «benchmark»: имя нужно лишь в строковых аннотациях

# ------------------------- регулятор ресурсов (неограниченный цикл) ----------


def _ram_available_bytes() -> int:
    """Свободная оперативная память в байтах (0, если определить нельзя).

        Способ 1: psutil.virtual_memory().available. Способ 2 (Windows без psutil):
        системный вызов GlobalMemoryStatusEx, поле ullAvailPhys. Иначе 0."""
    try:  # пробуем первый способ
        import psutil  # импортировать psutil (может быть не установлен)

        return int(psutil.virtual_memory().available)  # доступная память из psutil, в байтах
    except ImportError:  # psutil не установлен
        if sys.platform == "win32":  # на Windows есть системный вызов
            import ctypes  # доступ к функциям операционной системы

            class MEMORYSTATUSEX(ctypes.Structure):  # структура состояния памяти, которую заполняет система
                _fields_ = [  # поля структуры в порядке следования в памяти
                    ("dwLength", ctypes.c_ulong),  # размер структуры в байтах
                    ("dwMemoryLoad", ctypes.c_ulong),  # доля занятой физической памяти в процентах
                    ("ullTotalPhys", ctypes.c_ulonglong),  # вся физическая память
                    ("ullAvailPhys", ctypes.c_ulonglong),  # свободная физическая память
                    ("ullTotalPageFile", ctypes.c_ulonglong),  # предел файла подкачки
                    ("ullAvailPageFile", ctypes.c_ulonglong),  # свободно в файле подкачки
                    ("ullTotalVirtual", ctypes.c_ulonglong),  # предел виртуальной памяти процесса
                    ("ullAvailVirtual", ctypes.c_ulonglong),  # свободно виртуальной памяти процесса
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),  # свободная расширенная виртуальная память
                ]  # конец списка полей

            st = MEMORYSTATUSEX()  # выделить структуру (нули)
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)  # в dwLength записать размер структуры: его сверяет система
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # системный вызов заполняет st по указателю
            return int(st.ullAvailPhys)  # свободная физическая память в байтах
        return 0  # не Windows и нет psutil: размер неизвестен


def _vram_free_bytes(device: str) -> int:
    """Свободная память устройства device в байтах (0, если это не видеокарта).

        free = torch.cuda.mem_get_info(device)[0] — первый элемент пары (свободно, всего)."""
    if not (device.startswith("cuda") and _ensure_torch()):  # не cuda-устройство или torch недоступен
        return 0  # памяти видеокарты нет — вернуть 0
    try:  # запрос к драйверу может не удаться
        free, _total = _torch.cuda.mem_get_info(_torch.device(device))  # (свободно, всего) байт для device
        return int(free)  # свободные байты
    except Exception:  # ошибка драйвера
        return 0  # считать свободную память нулём


class ResourceGovernor:
    """Переводит свободную оперативную память и память видеокарты в бюджеты поиска одной задачи.

        Отдельный монотонный предел по времени и числу раундов ограничивает внешний цикл
        роста. Вместо того чтобы поиск занял всю память, бюджеты плавно уменьшаются.
        RAM_FLOOR_BYTES и VRAM_FLOOR_BYTES — пороги: ниже них задача пропускается."""

    RAM_FLOOR_BYTES = 300 * 1024 * 1024  # порог ОЗУ: 300·2^20 байт; ниже — пауза, а не падение по памяти
    VRAM_FLOOR_BYTES = 300 * 1024 * 1024  # порог памяти видеокарты: 300·2^20 байт

    def __init__(self, device: str = "cpu"):  # создать регулятор для устройства device (по умолчанию процессор)
        self.device = device  # запомнить устройство
        self.node_budget = DEFAULT_NODE_BUDGET  # бюджет узлов перебора — значение по умолчанию
        self.pop_budget = DEFAULT_POP_BUDGET  # бюджет размера популяции — значение по умолчанию
        self.chunk = VALIDATION_CHUNK  # размер порции проверки — значение по умолчанию
        self.total_skips = 0  # число пропущенных из-за нехватки ресурсов задач — пока 0

    def _scale(self) -> float:
        """Доля ресурсов s ∈ [0, 1] — во сколько раз уменьшать бюджеты.

            ram = 0 (неизвестно) → s = 1. Иначе
            s = min(1, max(0, (ram − floor) / (6 ГБ − floor)));
            при ram < 1 ГБ: s = min(s, max(0.05, (ram − floor) / (1 ГБ − floor)));
            на cuda при vram < VRAM_FLOOR_BYTES: s = 0."""
        ram = _ram_available_bytes()  # свободная оперативная память в байтах
        if ram <= 0:  # память определить не удалось
            return 1.0  # бюджеты не уменьшать
        # полная скорость выше 6 ГБ, линейное убывание до нуля на пороге floor
        hi, lo, floor = 6 * 2**30, 1 * 2**30, self.RAM_FLOOR_BYTES  # верх 6 ГБ, низ 1 ГБ, порог floor
        s = min(1.0, max(0.0, (ram - floor) / max(1, hi - floor)))  # по формуле s = clamp((ram − floor) / (hi − floor), 0, 1)
        if ram < lo:  # памяти меньше 1 ГБ — убывание сильнее
            s = min(s, max(0.05, (ram - floor) / max(1, lo - floor)))  # дополнительно s = min(s, max(0.05, (ram − floor) / (lo − floor)))
        if self.device.startswith("cuda") and _ensure_torch():  # cuda-устройство и torch доступен
            vram = _vram_free_bytes(self.device)  # свободная память видеокарты
            if vram < self.VRAM_FLOOR_BYTES:  # меньше порога
                s = 0.0  # масштаб ноль — задачу пропустить
        return s  # итоговый масштаб

    def before_task(self) -> bool:
        """Пересчитать бюджеты для следующей задачи; False — ресурсы исчерпаны, задачу пропустить.

            node = max(1000, ⌊DEFAULT_NODE_BUDGET·s⌋); pop = max(2000, ⌊DEFAULT_POP_BUDGET·s⌋);
            chunk = max(8, ⌊VALIDATION_CHUNK·min(1, 2s)⌋)."""
        s = self._scale()  # доля доступных ресурсов
        if s <= 0.0:  # ресурсов нет
            self.total_skips += 1  # пропусков: total_skips + 1
            return False  # сигнал пропустить задачу
        self.node_budget = max(1_000, int(DEFAULT_NODE_BUDGET * s))  # бюджет узлов = max(1000, ⌊DEFAULT_NODE_BUDGET·s⌋)
        self.pop_budget = max(2_000, int(DEFAULT_POP_BUDGET * s))  # бюджет популяции = max(2000, ⌊DEFAULT_POP_BUDGET·s⌋)
        self.chunk = max(8, int(VALIDATION_CHUNK * min(1.0, s * 2)))  # порция = max(8, ⌊VALIDATION_CHUNK·min(1, 2s)⌋)
        return True  # задачу можно запускать

    def recover(self, exc: BaseException) -> None:
        """После ошибки нехватки памяти: резко уменьшить бюджеты, сбросить кэш, работать дальше.

            exc не используется. node = min(node, 10000); pop = min(pop, 20000);
            chunk = max(4, ⌊chunk / 4⌋); на cuda — освободить кэш аллокатора."""
        DEFAULT_NODE_BUDGET_S = 10_000  # аварийный потолок бюджета узлов: 10000
        self.node_budget = min(self.node_budget, DEFAULT_NODE_BUDGET_S)  # бюджет узлов не выше 10000
        self.pop_budget = min(self.pop_budget, 20_000)  # бюджет популяции не выше 20000
        self.chunk = max(4, self.chunk // 4)  # порция = max(4, ⌊chunk / 4⌋)
        if self.device.startswith("cuda") and _ensure_torch():  # cuda-устройство и torch доступен
            try:  # очистка кэша может не удаться
                _torch.cuda.empty_cache()  # вернуть неиспользуемую память видеокарты драйверу
            except Exception:  # любая ошибка
                pass  # игнорировать и продолжать


# ---------------------- неограниченный цикл саморазвития ----------------------


def _propose_task_from_experts(
    experts: dict[str, Any],  # заученные эксперты по имени
    rng: np.random.Generator,  # генератор случайных чисел
    round_index: int,  # номер раунда
    depth: int = 1,  # глубина композиции (по умолчанию 1)
    domain_radius: int = 6,  # радиус области входов (по умолчанию 6)
) -> Optional[TaskSpec]:  # конец параметров; возврат — задача или None
    """Предложить композицию только на совместно подтверждённой области родителей.

        Ни одна родительская программа не вычисляется вне области, где эксперт
        считается корректным (Expert.supports). Задача наследует правило
        допустимости входов, и всё её конечное множество входов проверяется до
        возврата. Так растёт конечная программа композиций, а не независимая
        способность.
        experts — заученные эксперты; rng — генератор случайных чисел;
        round_index — номер раунда (входит в имя задачи); depth не используется;
        domain_radius — предел |x|, |y| (не более 12).
        Возвращает TaskSpec или None."""
    arity2 = [  # эксперты с двумя аргументами, кроме булевых
        e for e in experts.values() if e.program.arity == 2 and e.family != "BoolVenn"  # условие: арность 2 и семейство не BoolVenn
    ]  # конец списка
    arity1 = [  # эксперты с одним аргументом, кроме булевых
        e for e in experts.values() if e.program.arity == 1 and e.family != "BoolVenn"  # условие: арность 1 и семейство не BoolVenn
    ]  # конец списка
    bool2 = [  # булевы эксперты с двумя аргументами
        e for e in experts.values() if e.program.arity == 2 and e.family == "BoolVenn"  # условие: арность 2 и семейство BoolVenn
    ]  # конец списка

    # Булева композиция: каждый промежуточный и внешний вызов проверяется по области применимости.
    if bool2 and len(bool2) >= 2 and rng.random() < 0.4:  # есть ≥ 2 булевых экспертов и случайное число < 0.4
        for _ in range(8):  # до 8 попыток
            outer = bool2[int(rng.choice(len(bool2)))]  # внешний эксперт со случайным индексом из 0..len−1
            inner = bool2[int(rng.choice(len(bool2)))]  # внутренний эксперт со случайным индексом
            if inner is outer and len(bool2) > 1:  # выбран один и тот же эксперт, хотя есть другие
                continue  # повторить попытку
            rows = [  # пары (x, y) из {0,1}², допустимые для обеих программ
                (x, y)  # пара входов
                for x in (0, 1)  # x пробегает 0, 1
                for y in (0, 1)  # y пробегает 0, 1
                if inner.supports((x, y))  # внутренняя применима к (x, y)
                and outer.supports((int(inner.program.eval((x, y))), y))  # внешняя применима к (inner(x,y), y)
            ]  # конец списка
            if len(rows) != 4:  # нужны все 4 комбинации
                continue  # повторить попытку
            teacher = lambda x, y, o=outer, i=inner: int(  # учитель: o(i(x), y); o и i закреплены значениями по умолчанию
                o.program.eval((int(i.program.eval((int(x), int(y)))), int(y)))  # значение = outer(inner(int x, int y), int y)
            )  # конец лямбда-выражения
            return TaskSpec(  # новая задача
                f"AUTO_R{round_index}_{inner.name}_IN_{outer.name}",  # имя: номер раунда и имена родителей
                2,  # арность 2 (два аргумента)
                ((0, 1), (0, 1)),  # области аргументов: x ∈ {0,1}, y ∈ {0,1}
                teacher,  # эталонная функция
                family="auto-composed",  # семейство — автокомпозиция
            )  # конец TaskSpec

    if not arity2:  # нет двухаргументных экспертов — нет внешней программы
        return None  # предложить нечего
    inner_pool = arity2 + arity1  # кандидаты во внутренние эксперты: конкатенация списков

    # В предложении должно быть достаточно строк для разбиения на обучение и проверку.
    min_rows = 4  # минимум входных пар |X|·|Y| ≥ 4
    max_radius = max(1, min(int(domain_radius), 12))  # радиус домена = clamp(domain_radius, 1, 12)
    for _ in range(12):  # до 12 попыток
        outer = arity2[int(rng.choice(len(arity2)))]  # внешний эксперт со случайным индексом
        inner = inner_pool[int(rng.choice(len(inner_pool)))]  # внутренний эксперт со случайным индексом

        if inner.program.arity == 1:  # внутренняя программа с одним аргументом
            # outer(inner(x), y). Держим x внутри области унарного эксперта и
            # каждое inner(x) внутри области первого аргумента внешнего эксперта.
            xs0 = tuple(  # допустимые x
                x  # само значение
                for x in inner.domains[0]  # x из области определения внутреннего
                if abs(int(x)) <= max_radius  # |x| ≤ радиуса
                and inner.supports((int(x),))  # внутренний применим к x
                and int(inner.program.eval((int(x),))) in outer.domains[0]  # inner(x) входит в область 1-го аргумента внешнего
            )  # конец кортежа
            ys0 = tuple(  # допустимые y
                y  # само значение
                for y in outer.domains[1]  # y из области 2-го аргумента внешнего
                if abs(int(y)) <= max_radius  # |y| ≤ радиуса
                and (outer.valid_rule != "nonzero_y" or int(y) != 0)  # если правило «y ≠ 0», то y ≠ 0
            )  # конец кортежа
            if not xs0 or not ys0:  # хотя бы одно множество пусто
                continue  # повторить попытку
            for radius in range(max_radius, 0, -1):  # радиус от max_radius до 1 с шагом −1
                xs = tuple(x for x in xs0 if abs(int(x)) <= radius)  # оставшиеся xs = {x ∈ xs0 : |x| ≤ radius}
                ys = tuple(y for y in ys0 if abs(int(y)) <= radius)  # оставшиеся ys = {y ∈ ys0 : |y| ≤ radius}
                ys = tuple(  # оставляем только столбцы, безопасные для внешнего эксперта
                    y  # значение y
                    for y in ys  # по y из ys
                    if all(  # для всех x из xs
                        outer.supports((int(inner.program.eval((int(x),))), int(y)))  # внешняя применима к (inner(x), y)
                        for x in xs  # по x из xs
                    )  # конец all
                )  # конец кортежа
                if len(xs) * len(ys) < min_rows:  # входных пар |X|·|Y| меньше 4
                    continue  # уменьшить радиус
                if not all(inner.supports((int(x),)) for x in xs):  # внутренний применим не ко всем x
                    continue  # уменьшить радиус
                teacher = lambda x, y, o=outer, i=inner: int(  # учитель: o(i(x), y); o и i закреплены по умолчанию
                    o.program.eval((int(i.program.eval((int(x),))), int(y)))  # значение = outer(inner(int x), int y)
                )  # конец лямбда-выражения
                return TaskSpec(  # новая задача
                    f"AUTO_R{round_index}_{inner.name}_IN_{outer.name}",  # имя: номер раунда и имена родителей
                    2,  # арность 2 (два аргумента)
                    (xs, ys),  # области аргументов: x ∈ xs, y ∈ ys
                    teacher,  # эталонная функция
                    valid_rule=outer.valid_rule,  # правило допустимости — как у внешнего эксперта
                    family="auto-composed",  # семейство — автокомпозиция
                )  # конец TaskSpec

        elif inner.program.arity == 2:  # внутренняя программа с двумя аргументами
            # outer(x, inner(x,y)). Прямоугольник входов намеренно сужается,
            # пока каждый вызов обеих родительских программ не станет подтверждённым.
            xs0 = tuple(  # допустимые x
                x  # само значение
                for x in inner.domains[0]  # x из области 1-го аргумента внутреннего
                if x in outer.domains[0] and abs(int(x)) <= max_radius  # x входит в область 1-го аргумента внешнего и |x| ≤ радиуса
            )  # конец кортежа
            ys0 = tuple(y for y in inner.domains[1] if abs(int(y)) <= max_radius)  # ys0 = {y ∈ dom_2(inner) : |y| ≤ радиуса}
            if not xs0 or not ys0:  # хотя бы одно множество пусто
                continue  # повторить попытку
            for radius in range(max_radius, 0, -1):  # радиус от max_radius до 1 с шагом −1
                xs = tuple(x for x in xs0 if abs(int(x)) <= radius)  # оставшиеся xs = {x ∈ xs0 : |x| ≤ radius}
                ys = tuple(y for y in ys0 if abs(int(y)) <= radius)  # оставшиеся ys = {y ∈ ys0 : |y| ≤ radius}
                if not xs or not ys:  # хотя бы одно множество пусто
                    continue  # уменьшить радиус
                # Убираем столбцы y, если не для каждого выбранного x подтверждены
                # и внутреннее, и внешнее вычисление. Получается прямоугольное безопасное подмножество.
                safe_ys = []  # столбцы, безопасные для всех x из xs
                for y in ys:  # по каждому y
                    good = True  # столбец пока считается безопасным
                    for x in xs:  # по каждому x
                        if not inner.supports((int(x), int(y))):  # внутренняя не применима к (x, y)
                            good = False  # столбец небезопасен
                            break  # выйти из цикла по x
                        inner_value = int(inner.program.eval((int(x), int(y))))  # промежуточное inner_value = inner(int x, int y)
                        if not outer.supports((int(x), inner_value)):  # внешняя не применима к (x, inner_value)
                            good = False  # столбец небезопасен
                            break  # выйти из цикла по x
                    if good:  # все x прошли
                        safe_ys.append(y)  # добавить столбец y
                ys = tuple(safe_ys)  # ys = кортеж безопасных столбцов
                if len(xs) * len(ys) < min_rows:  # входных пар |X|·|Y| меньше 4
                    continue  # уменьшить радиус
                teacher = lambda x, y, o=outer, i=inner: int(  # учитель: o(x, i(x,y)); o и i закреплены по умолчанию
                    o.program.eval((int(x), int(i.program.eval((int(x), int(y))))))  # значение = outer(int x, inner(int x, int y))
                )  # конец лямбда-выражения
                return TaskSpec(  # новая задача
                    f"AUTO_R{round_index}_{inner.name}_IN_{outer.name}",  # имя: номер раунда и имена родителей
                    2,  # арность 2 (два аргумента)
                    (xs, ys),  # области аргументов: x ∈ xs, y ∈ ys
                    teacher,  # эталонная функция
                    valid_rule=inner.valid_rule,  # правило допустимости — как у внутреннего эксперта
                    family="auto-composed",  # семейство — автокомпозиция
                )  # конец TaskSpec

    return None  # все попытки исчерпаны — предложения нет


def verify_growth_lineage(model: "OptimalMathProcessor") -> dict[str, Any]:
    """Перепроверить каждую сохранённую композицию по паре её родителей.

        Это исполняемая проверка на всём объявленном конечном множестве входов
        потомка: child(x) = outer(parent_input(x)); вне этого множества она
        ничего не доказывает. Возвращает словарь с числом проверенных
        (lineage_checked), списком ошибок (lineage_errors) и флагом lineage_pass."""
    checked = 0  # число проверенных потомков — пока 0
    errors: list[dict[str, Any]] = []  # список ошибок — пока пуст
    for child in model.experts.values():  # по всем экспертам модели
        if not child.parents:  # у эксперта нет родителей — это не композиция
            continue  # пропустить
        if len(child.parents) != 2:  # родителей не двое
            errors.append({"expert": child.name, "reason": "invalid_parent_arity"})  # записать ошибку: неверное число родителей
            continue  # к следующему эксперту
        outer = model.experts.get(child.parents[0])  # внешний родитель — первый в списке (или None)
        inner = model.experts.get(child.parents[1])  # внутренний родитель — второй в списке (или None)
        if outer is None or inner is None:  # кого-то из родителей нет
            errors.append({"expert": child.name, "reason": "missing_parent"})  # записать ошибку: родитель отсутствует
            continue  # к следующему эксперту
        ok = True  # композиция пока верна
        checked += 1  # проверенных: checked + 1
        for xs in itertools.product(*child.domains):  # декартово произведение областей аргументов
            xs = tuple(int(x) for x in xs)  # каждый аргумент приведяется к целому
            if not child.supports(xs):  # потомок не подтверждён на этом входе
                continue  # пропустить вход
            try:  # вычисление родителей может вызвать арифметическую ошибку
                if (  # оба родителя — булевы программы
                    outer.program.venn_mask is not None  # у внешней есть маска булевой функции
                    and inner.program.venn_mask is not None  # и у внутренней есть маска
                ):  # конец условия
                    if not inner.supports(xs):  # внутренняя не подтверждена на xs
                        ok = False  # композиция неверна
                        break  # прервать перебор входов
                    middle = int(inner.program.eval(xs))  # промежуточное middle = inner(xs)
                    parent_input = (middle, xs[1])  # вход внешнего: (middle, y)
                elif inner.program.arity == 1:  # внутренняя программа с одним аргументом
                    if not inner.supports((xs[0],)):  # внутренняя не подтверждена на x
                        ok = False  # композиция неверна
                        break  # прервать перебор входов
                    middle = int(inner.program.eval((xs[0],)))  # промежуточное middle = inner(x)
                    parent_input = (middle, xs[1])  # вход внешнего: (inner(x), y)
                elif inner.program.arity == 2:  # внутренняя программа с двумя аргументами
                    if not inner.supports(xs):  # внутренняя не подтверждена на (x, y)
                        ok = False  # композиция неверна
                        break  # прервать перебор входов
                    middle = int(inner.program.eval(xs))  # промежуточное middle = inner(x, y)
                    parent_input = (xs[0], middle)  # вход внешнего: (x, inner(x,y))
                else:  # арность внутреннего родителя не 1 и не 2
                    ok = False  # композиция неверна
                    break  # прервать перебор входов
                if not outer.supports(parent_input):  # внешняя не подтверждена на своём входе
                    ok = False  # композиция неверна
                    break  # прервать перебор входов
                expected = int(outer.program.eval(parent_input))  # ожидаемое expected = outer(parent_input)
                if expected != int(child.program.eval(xs)):  # если ожидаемое expected ≠ child(xs)
                    ok = False  # композиция неверна
                    break  # прервать перебор входов
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # арифметическая ошибка родителей
                ok = False  # композиция неверна
                break  # прервать перебор входов
        if not ok:  # найдено расхождение с родителями
            errors.append(  # записать ошибку
                {"expert": child.name, "reason": "parent_composition_mismatch"}  # причина: композиция не совпадает с родителями
            )  # конец append
    return {  # итоговый словарь
        "lineage_checked": checked,  # число проверенных потомков
        "lineage_errors": errors,  # список ошибок
        "lineage_pass": not errors,  # проверка пройдена, если ошибок нет
    }  # конец словаря


def grow_forever(
    model: "OptimalMathProcessor",  # растущая модель
    suite: Sequence[TaskSpec],  # базовые задачи
    device: str = "cpu",  # устройство (по умолчанию процессор)
    max_rounds: Optional[int] = None,  # предел числа раундов (None — без предела)
    snapshot_every: int = 10,  # число фиксаций между снимками состояния
    out_dir: str = ".omc/hagi_grow",  # каталог вывода
    log_every: int = 1,  # печатать каждый раунд, если не ноль
    budget_cap: Optional[int] = None,  # потолок бюджета узлов (None — без потолка)
    deadline: Optional[float] = None,  # монотонный предел времени (None — без предела)
) -> dict[str, Any]:  # конец параметров; возврат — словарь отчёта
    """Ограниченный цикл композиционного роста с явными снимками состояния.

        Каждый раунд: (1) учит базовую программу задач; (2) предлагает конечные
        композиции только внутри областей родителей; (3) добывает теоремы;
        записывает снимок состояния. Вызывающий код может задать монотонный
        предел времени deadline и/или предел числа раундов max_rounds;
        регулятор ресурсов ограничивает каждый поиск. Это рекомбинация программы
        обучения, а не доказательство новой общей способности.
        model — растущая модель; suite — базовые задачи; device — устройство;
        max_rounds — предел раундов (None — без предела); snapshot_every —
        число фиксаций между снимками; out_dir — каталог вывода; log_every —
        печатать ли каждый раунд; budget_cap — потолок бюджета узлов;
        deadline — монотонный предел времени. Возвращает словарь отчёта."""
    out = Path(out_dir)  # каталог вывода
    out.mkdir(parents=True, exist_ok=True)  # создать его (вместе с родителями); существующий — не ошибка
    gov = ResourceGovernor(device)  # регулятор ресурсов для устройства
    log_path = out / "growth_log.jsonl"  # путь журнала раундов
    start = time.time()  # время начала (для общей длительности)
    rnd = np.random.default_rng(20260101)  # генератор случайных чисел с фиксированным зерном
    report: dict[str, Any] = {"device": device, "rounds": [], "stopped": None}  # отчёт: устройство, пустой список раундов, причина остановки не задана
    round_index = 0  # номер текущего раунда
    commits_since_snapshot = 0  # число фиксаций с последнего снимка
    idle_rounds = 0  # число раундов подряд без фиксаций
    depth = 1  # глубина композиции (передаётся в предложение задачи)
    domain_radius = 6  # радиус области входов для самопредложенных задач
    try:  # тело цикла прерывается по Ctrl+C
        while max_rounds is None or round_index < max_rounds:  # пока предел раундов не задан или не достигнут
            if deadline is not None and time.monotonic() >= deadline:  # монотонное время достигло предела
                report["stopped"] = "TIME_LIMIT"  # причина остановки — предел времени
                break  # выйти из цикла
            t0 = time.time()  # время начала раунда
            committed_this_round = 0  # фиксаций в этом раунде — пока 0
            # (1) базовая программа обучения (идемпотентно: уже зафиксированные задачи используются повторно)
            for spec in suite:  # по всем базовым задачам
                if deadline is not None and time.monotonic() >= deadline:  # монотонное время достигло предела
                    report["stopped"] = "TIME_LIMIT"  # причина остановки — предел времени
                    break  # прервать обход задач
                if not gov.before_task():  # ресурсы исчерпаны
                    break  # прервать обход задач
                model.node_budget = (  # бюджет узлов модели
                    min(gov.node_budget, budget_cap) if budget_cap else gov.node_budget  # min(node, cap), если cap задан, иначе node
                )  # конец присваивания
                model.pop_budget = (  # бюджет популяции модели
                    min(gov.pop_budget, budget_cap * 2) if budget_cap else gov.pop_budget  # min(pop, 2·cap), если cap задан, иначе pop
                )  # конец присваивания
                try:  # обучение может вызвать ошибку памяти или значений
                    ev = model.learn(  # обучиться на задаче
                        spec,  # задача
                        seed=10_000 + round_index * 7919 + len(spec.name),  # зерно = 10000 + раунд·7919 + длина имени
                        deadline=deadline,  # предел времени
                        train_neural=os.environ.get("HAGI_NO_NEURAL") != "1",  # обучать нейросеть, если HAGI_NO_NEURAL ≠ "1"
                    )  # конец learn
                    if ev.action == "COMMIT":  # результат — фиксация нового эксперта
                        committed_this_round += 1  # фиксаций в раунде + 1
                        commits_since_snapshot += 1  # фиксаций с последнего снимка + 1
                except (MemoryError, RuntimeError, ValueError) as exc:  # ошибка памяти или значений
                    gov.recover(exc)  # уменьшить бюджеты и продолжать
            # (2) самопредложенные задачи: композиции зафиксированных экспертов
            proposed = 0  # число запущенных на обучение предложений
            attempts = min(12, max(2, len(model.experts) // 3))  # число попыток = min(12, max(2, ⌊число экспертов / 3⌋))
            for k in range(attempts):  # по номерам попыток
                if deadline is not None and time.monotonic() >= deadline:  # монотонное время достигло предела
                    report["stopped"] = "TIME_LIMIT"  # причина остановки — предел времени
                    break  # прервать попытки
                if not gov.before_task():  # ресурсы исчерпаны
                    break  # прервать попытки
                spec = _propose_task_from_experts(  # предложить композицию
                    model.experts,  # текущие эксперты
                    rnd,  # генератор случайных чисел
                    round_index,  # номер раунда
                    depth=depth,  # глубина композиции
                    domain_radius=domain_radius,  # радиус области входов
                )  # конец вызова
                if spec is None or spec.name in model.experts:  # предложения нет или такой эксперт уже есть
                    continue  # следующая попытка
                model.node_budget = (  # бюджет узлов модели
                    min(gov.node_budget, budget_cap) if budget_cap else gov.node_budget  # min(node, cap), если cap задан, иначе node
                )  # конец присваивания
                model.pop_budget = (  # бюджет популяции модели
                    min(gov.pop_budget, budget_cap * 2) if budget_cap else gov.pop_budget  # min(pop, 2·cap), если cap задан, иначе pop
                )  # конец присваивания
                try:  # обучение может вызвать ошибку памяти или значений
                    hint = (  # подсказка формы решения
                        _composition_hint(spec)  # подсказка по композиции
                        if spec.family == "auto-composed"  # только для автокомпозиций
                        else None  # иначе без подсказки
                    )  # конец условного выражения
                    if spec.family == "auto-composed" and hint is None:  # автокомпозиция, а подсказки нет
                        # Предложение невыразимо в текущей безопасной грамматике
                        # подсказок. Слепой синтез вместо подсказки не используется.
                        continue  # пропустить предложение
                    ev = model.learn(  # обучиться на предложенной задаче
                        spec,  # задача
                        seed=500_000 + round_index * 104_729 + k,  # зерно = 500000 + раунд·104729 + номер попытки
                        deadline=deadline,  # предел времени
                        candidate_hint=hint,  # подсказка формы кандидата
                        train_neural=os.environ.get("HAGI_NO_NEURAL") != "1",  # обучать нейросеть, если HAGI_NO_NEURAL ≠ "1"
                    )  # конец learn
                    proposed += 1  # запущенных предложений + 1
                    if ev.action == "COMMIT":  # результат — фиксация нового эксперта
                        committed_this_round += 1  # фиксаций в раунде + 1
                        commits_since_snapshot += 1  # фиксаций с последнего снимка + 1
                except (  # ошибка памяти, времени выполнения или арифметики
                    MemoryError,  # нехватка памяти
                    RuntimeError,  # ошибка времени выполнения
                    ValueError,  # недопустимое значение
                    ZeroDivisionError,  # деление на ноль
                    OverflowError,  # арифметическое переполнение
                ) as exc:  # исключение — в переменной exc
                    gov.recover(exc)  # уменьшить бюджеты и продолжать
            # (3) автономная добыча теорем по подтверждённой базе экспертов
            try:  # добыча может вызвать ошибку памяти или значений
                new_theorems = TheoremMiner().mine(model, max_experts=16)  # новые теоремы по не более чем 16 экспертам
            except (  # ошибка памяти, времени выполнения или арифметики
                MemoryError,  # нехватка памяти
                RuntimeError,  # ошибка времени выполнения
                ValueError,  # недопустимое значение
                ZeroDivisionError,  # деление на ноль
                OverflowError,  # арифметическое переполнение
            ):  # конец списка исключений
                new_theorems = []  # считать, что новых теорем нет
            lineage = verify_growth_lineage(model)  # проверка композиций по родителям
            round_rec = {  # запись о раунде
                "round": round_index,  # номер раунда
                **lineage,  # поля lineage_checked, lineage_errors, lineage_pass
                "experts": len(model.experts),  # число экспертов
                "unique_programs": len({e.program for e in model.experts.values()}),  # число различных программ
                "unique_certified_scopes": len(  # число различных подтверждённых областей
                    {  # множество троек
                        (e.program, e.domains, e.valid_rule)  # (программа, области, правило допустимости)
                        for e in model.experts.values()  # по всем экспертам
                    }  # конец множества
                ),  # конец len
                "committed_this_round": committed_this_round,  # фиксаций в раунде
                "proposed_tasks": proposed,  # запущенных предложений
                "theorems_total": len(model.theorems),  # всего теорем
                "theorems_new": len(new_theorems),  # новых теорем
                "total_code_length": sum(e.code_length for e in model.experts.values()),  # суммарная длина кода экспертов
                "node_budget": gov.node_budget,  # бюджет узлов
                "pop_budget": gov.pop_budget,  # бюджет популяции
                "ram_avail_gb": round(_ram_available_bytes() / 2**30, 2),  # свободное ОЗУ, ГБ, с точностью 0.01
                "vram_free_gb": round(_vram_free_bytes(device) / 2**30, 2),  # свободная память видеокарты, ГБ, с точностью 0.01
                "elapsed": round(time.time() - t0, 2),  # длительность раунда, с, с точностью 0.01
            }  # конец записи
            report["rounds"].append(round_rec)  # добавить запись в отчёт
            with open(log_path, "a", encoding="utf-8") as f:  # открыть журнал на дозапись
                f.write(json.dumps(round_rec) + "\n")  # записать запись одной строкой JSON с переводом строки
            if log_every or committed_this_round:  # печать каждого раунда включена или есть фиксации
                print(  # напечатать строку состояния
                    f"[grow r{round_index}] experts={round_rec['experts']} "  # раунд и число экспертов
                    f"commits={committed_this_round} code={round_rec['total_code_length']}bits "  # фиксации и суммарная длина кода в битах
                    f"ram={round_rec['ram_avail_gb']}GB "  # свободное ОЗУ
                    f"vram={round_rec['vram_free_gb']}GB",  # свободная память видеокарты
                    flush=True,  # сразу сбросить буфер вывода
                )  # конец print
            if commits_since_snapshot >= snapshot_every:  # накопилось снимок фиксаций для снимка
                model.save(out / "hagi_grow_model.json")  # сохранить модель
                commits_since_snapshot = 0  # счётчик фиксаций обнулен
            if committed_this_round > 0:  # в раунде были фиксации
                idle_rounds = 0  # счётчик простоя обнулен
            else:  # фиксаций не было
                idle_rounds += 1  # раундов простоя + 1
                if idle_rounds >= 3:  # три раунда подряд без фиксаций — насыщение
                    # эскалация при насыщении: углубить алгебру композиций и
                    # расширить подтверждённые области — неограниченный рост программы обучения
                    depth += 1  # глубина + 1
                    domain_radius = min(12, domain_radius + 2)  # радиус = min(12, радиус + 2)
                    idle_rounds = 0  # счётчик простоя обнулен
            round_index += 1  # номер раунда + 1
            if deadline is not None and time.monotonic() >= deadline:  # монотонное время достигло предела
                report["stopped"] = "TIME_LIMIT"  # причина остановки — предел времени
                break  # выйти из цикла
            if (  # свободное ОЗУ известно и ниже порога
                _ram_available_bytes()  # свободное ОЗУ (0 — неизвестно)
                and _ram_available_bytes() < ResourceGovernor.RAM_FLOOR_BYTES  # и оно меньше порога
            ):  # конец условия
                report["stopped"] = "RAM_FLOOR"  # причина остановки — нижний порог ОЗУ
                break  # выйти из цикла
    except KeyboardInterrupt:  # цикл прерван пользователем
        report["stopped"] = "INTERRUPTED"  # причина остановки — прерывание
    finally:  # выполняется всегда
        model.save(out / "hagi_grow_model.json")  # сохранить модель
        (out / "theorems.json").write_text(  # записать теоремы в файл
            json.dumps(  # сериализовать в JSON
                {  # словарь
                    "count": len(model.theorems),  # число теорем
                    "theorems": [vars(t) for t in model.theorems],  # список словарей атрибутов теорем
                },  # конец словаря
                indent=2,  # с отступом 2 пробела
            ),  # конец json.dumps
            encoding="utf-8",  # кодировка UTF-8
        )  # конец write_text
        report["total_experts"] = len(model.experts)  # общее число экспертов
        report.update(verify_growth_lineage(model))  # добавить итоговую проверку композиций
        report["total_elapsed"] = round(time.time() - start, 1)  # общая длительность, с, с точностью 0.1
        (out / "growth_report.json").write_text(  # записать отчёт в файл
            json.dumps(report, indent=2), encoding="utf-8"  # JSON с отступом 2 пробела, кодировка UTF-8
        )  # конец write_text
    return report  # вернуть отчёт

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'ResourceGovernor', 'verify_growth_lineage', 'grow_forever',  # имя интерфейса
]  # конец перечня
