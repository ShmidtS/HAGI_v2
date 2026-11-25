"""Управление временем выполнения: сертифицированный шаг, журнал аудита, маршрутизатор — модуль символьного ядра HAGI.

Каждая строка снабжена русским комментарием: что делает и по какой формуле.
"""

import math  # импорт стандартной библиотеки, нужный этой части
import sys  # импорт стандартной библиотеки, нужный этой части
from dataclasses import dataclass, field  # импорт стандартной библиотеки, нужный этой части
from typing import Optional, Sequence  # импорт стандартной библиотеки, нужный этой части
import numpy as np  # импорт стандартной библиотеки, нужный этой части

# ==============================================================================
# 8. Формальное управление временем выполнения: CertStepGen, AuditLedger, HedgeRouter
# ==============================================================================


@dataclass  # поля класса автоматически становятся аргументами конструктора
class CertStepGen:  # канонический сертификат перехода
    """Канонический сертификат одного шага: что сделано, как изменилась цель и что истрачено.
    
        drop = obj_before − obj_after ≥ 0; budget_* и risk_* — остаток и расход бюджета и риска;
        delta_* — остаток и расход статистической значимости; floors_kept — все нижние границы соблюдены."""

    version: int  # версия схемы
    task: str  # имя задачи
    action: str  # действие
    obj_before: float  # цель до
    obj_after: float  # цель после
    drop: float  # снижение цели
    budget_rem: float  # остаток бюджета
    budget_spend: float  # расход бюджета
    risk_rem: float  # остаток риска
    risk_spend: float  # расход риска
    delta_rem: float  # остаток значимости
    delta_spend: float  # расход значимости
    floors_kept: bool  # границы соблюдены


class HedgeRouter:  # маршрутизатор Хеджа с ограниченными потерями
    """Маршрутизатор Хеджа (Hedge): эксперты с весами, убывающими по потерям.
    
        Начальные веса 1/K. После потерь l ∈ [0, 1]: w_i *= exp(−eta·l_i), затем нормировка.
        Нижняя граница против «запирания»: w_i ≥ (1/K)·exp(−eta·cum_loss_i) — проверяется.
        K — число экспертов, eta — скорость обучения, cum_loss — накопленные потери."""

    def __init__(self, n_experts: int, eta: float = 0.5):  # создать маршрутизатор: n_experts экспертов, скорость eta
        self.K = n_experts  # число экспертов
        self.eta = eta  # скорость обучения
        self.weights = np.ones(n_experts, dtype=np.float64) / max(1, n_experts)  # веса: w_i = 1 / max(1, K)
        self.cum_loss = np.zeros(n_experts, dtype=np.float64)  # накопленные потери — нули

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:  # снимок состояния
        return self.weights.copy(), self.cum_loss.copy()  # копии весов и потерь

    def restore(self, snap: tuple[np.ndarray, np.ndarray]) -> None:  # восстановить из снимка
        self.weights = snap[0].copy()  # веса из снимка
        self.cum_loss = snap[1].copy()  # потери из снимка

    def update(self, losses_01: np.ndarray) -> np.ndarray:  # обновить веса по потерям
        losses = np.clip(np.asarray(losses_01, dtype=np.float64), 0.0, 1.0)  # потери, обрезанные в [0, 1]
        self.cum_loss += losses  # накопить потери
        self.weights *= np.exp(-self.eta * losses)  # веса: w_i *= exp(−eta·l_i)
        self.weights /= np.sum(self.weights)  # нормировка: w_i = w_i / Σ w
        floor = (1.0 / self.K) * np.exp(-self.eta * self.cum_loss)  # нижняя граница: floor_i = (1/K)·exp(−eta·cum_loss_i)
        assert np.all(self.weights + 1e-12 >= floor)  # проверка: w_i + 1e-12 ≥ floor_i
        return self.weights.copy()  # вернуть копию весов

    def ordered(self, names: Optional[Sequence[str]] = None) -> list:  # порядок экспертов по убыванию веса
        """Порядок экспертов по убыванию веса (априорный порядок Хеджа).
        
                names: если задан — возвращаются индексы из names (отсортированные по весу),
                иначе — argsort весов. """
        order = np.argsort(-self.weights)  # порядок: argsort(−weights)
        if names is None:  # имена не заданы
            return order.tolist()  # список индексов
        return [i for i in order if i < len(names)]  # индексы, не выходящие за длину names


@dataclass  # поля класса автоматически становятся аргументами конструктора
class AuditStream:  # диапазон потока аудита: [lo, hi) и курсор
    lo: int  # начало диапазона
    hi: int  # конец диапазона (не включительно)
    cursor: int  # текущая позиция


@dataclass  # поля класса автоматически становятся аргументами конструктора
class AuditLedger:  # журнал аудита: непересекающиеся потоки, расход значимости, нижние границы
    """Журнал аудита статистических проверок.
    
        delta_total — общий бюджет значимости (по умолчанию 0.05); spent — истрачено по задачам;
        streams — потоки выборок [lo, hi) с курсором; floors — нижние границы качества по задачам;
        rejected_tasks — отклонённые задачи; [final_test_lo, final_test_hi) — зарезервированный
        диапазон финального теста, который нельзя трогать. Остаток значимости:
        remaining = max(0, delta_total − Σ spent)."""

    delta_total: float = 0.05  # общий бюджет значимости
    spent: dict[str, float] = field(default_factory=dict)  # истрачено по задачам
    streams: dict[str, AuditStream] = field(default_factory=dict)  # потоки выборок
    floors: dict[str, float] = field(default_factory=dict)  # нижние границы качества
    rejected_tasks: set[str] = field(default_factory=set)  # отклонённые задачи
    final_test_lo: int = 10_000_000  # начало финального теста
    final_test_hi: int = 20_000_000  # конец финального теста
    final_test_touched: bool = False  # финальный тест не тронут
    next_cursor_base: int = 0  # начало следующего потока

    def remaining(self) -> float:  # остаток бюджета значимости
        return max(0.0, self.delta_total - sum(self.spent.values()))  # остаток: remaining = max(0, delta_total − Σ spent)

    def allocate_stream(self, task: str, n_samples: int) -> AuditStream:  # выделить поток выборок задаче
        if task in self.streams:  # поток уже есть
            return self.streams[task]  # вернуть существующий
        lo = self.next_cursor_base  # начало: следующий свободный индекс
        hi = lo + max(1, int(n_samples))  # конец: lo + max(1, n_samples)
        assert (  # проверка
            hi <= self.final_test_lo  # поток не заходит в диапазон финального теста
        ), "audit stream overlapped held-out final test range"  # сообщение ошибки утверждения
        st = AuditStream(lo=lo, hi=hi, cursor=lo)  # создать поток с курсором в начале
        self.streams[task] = st  # запомнить поток
        self.next_cursor_base = hi  # следующий поток начнётся с hi
        return st  # вернуть поток

    def advance_stream(self, task: str, count: int) -> None:  # продвинуть курсор потока
        st = self.streams[task]  # поток задачи
        st.cursor = min(st.hi, st.cursor + max(0, int(count)))  # курсор: cursor = min(hi, cursor + max(0, count))

    def draw_disjoint_splits(  # разбить область на непересекающиеся train / cegis / val
        self,  # экземпляр
        task: str,  # имя задачи
        domain: list,  # список элементов
        seed: int = 0,  # зерно (смещается суммой кодов символов задачи)
        train_frac: float = 0.35,  # доля train
        cegis_frac: float = 0.43,  # доля cegis
    ) -> tuple[list, list, list]:  # возвращает (train, cegis, val)
        """Разбить domain на непересекающиеся train / cegis / val из ОДНОГО потока аудита.
        
                Перестановка определяется зерном seed + (Σ ord(символы task) mod 10000). Для малых
                областей доли деградируют честно: n ≤ 8 — train n − 1, cegis 0; n ≤ 16 — train n − 3,
                cegis 1; иначе train = max(6, round(n·train_frac)), cegis = max(4, round(n·cegis_frac)),
                а если сумма ≥ n — cegis = max(1, (n − n_tr) // 2). Курсор проходит весь поток."""
        n = len(domain)  # размер области
        st = self.allocate_stream(task, n)  # выделить поток на n элементов
        rng = np.random.default_rng(seed + (sum(map(ord, task)) % 10000))  # генератор с зерном seed + Σ ord(task) mod 10000
        perm = rng.permutation(n)  # случайная перестановка индексов
        if n <= 8:  # очень малая область
            n_tr, n_ceg = max(1, n - 1), 0  # обучение: train = n − 1, cegis = 0
        elif n <= 16:  # малая область
            n_tr, n_ceg = max(2, n - 3), 1  # обучение: train = n − 3, cegis = 1
        else:  # обычная область
            n_tr = max(6, int(round(n * train_frac)))  # обучение: train = max(6, round(n·train_frac))
            n_ceg = max(4, int(round(n * cegis_frac)))  # доводка: cegis = max(4, round(n·cegis_frac))
            if n_tr + n_ceg >= n:  # сумма не оставляет val — уменьшить cegis
                n_ceg = max(1, (n - n_tr) // 2)  # доводка: cegis = max(1, (n − n_tr) // 2)
        tr = [domain[int(i)] for i in perm[:n_tr]]  # train — первые n_tr перестановки
        ceg = [domain[int(i)] for i in perm[n_tr:n_tr + n_ceg]]  # cegis — следующие n_ceg
        val = [domain[int(i)] for i in perm[n_tr + n_ceg:]]  # val — остаток
        st.cursor = min(st.hi, st.lo + len(tr) + len(ceg) + len(val))  # курсор — за всеми использованными
        return tr, ceg, val  # вернуть три части

    @staticmethod  # статический метод
    def hoeffding_radius(alpha: float, n: int) -> float:  # радиус доверительного интервала Хёфдинга
        if n <= 0 or alpha <= 0.0:  # нет выборки или нет значимости
            return 1.0  # максимальный радиус
        return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))  # радиус: r = √(ln(2/alpha) / (2n))

    def spend_anytime(  # потратить значимость и вернуть радиус
        self, task: str, alpha: float, n_samples: int  # задача, значимость, число выборок
    ) -> tuple[bool, float]:  # возвращает (успех, радиус)
        if alpha < 0 or alpha > self.remaining() + 1e-12:  # некорректная или превышающая остаток значимость
            return False, 1.0  # отказ: радиус 1
        self.spent[task] = self.spent.get(task, 0.0) + alpha  # учесть расход: spent[task] += alpha
        self.advance_stream(task, n_samples)  # продвинуть курсор
        return True, self.hoeffding_radius(alpha, n_samples)  # успех и радиус Хёфдинга

    def record_floor(self, task: str, quality: float) -> bool:  # зафиксировать нижнюю границу качества
        prev = self.floors.get(task, -1e18)  # прежняя граница (если нет — −∞)
        if quality + 1e-12 < prev:  # качество ниже границы
            return False  # граница нарушена
        self.floors[task] = max(prev, quality)  # поднять границу: max(prev, quality)
        return True  # граница соблюдена

    def record_reject(self, task: str) -> None:  # зафиксировать отклонение задачи
        self.rejected_tasks.add(task)  # добавить в отклонённые
        if task in self.streams:  # если поток был
            st = self.streams[task]  # поток задачи
            st.cursor = st.hi  # исчерпать поток

    def verify_disjoint_and_clean(self) -> bool:  # проверить непересечение потоков и чистоту финального теста
        if self.final_test_touched:  # финальный тест тронут
            return False  # нарушение
        intervals = sorted((st.lo, st.hi) for st in self.streams.values())  # интервалы потоков по возрастанию
        for i in range(len(intervals) - 1):  # по соседним парам
            if intervals[i][1] > intervals[i + 1][0]:  # конец текущего дальше начала следующего — пересечение
                return False  # нарушение
        for lo, hi in intervals:  # по интервалам
            if not (hi <= self.final_test_lo or lo >= self.final_test_hi):  # интервал пересекает финальный тест
                return False  # нарушение
        return sum(self.spent.values()) <= self.delta_total + 1e-12  # итог: расход не превышает бюджет


def ram_available_bytes() -> int:  # доступная оперативная память в байтах
    try:  # попытка 1: /proc/meminfo
        with open("/proc/meminfo", "r", encoding="utf-8") as f:  # открыть файл сведений о памяти
            for line in f:  # по строкам
                if line.startswith("MemAvailable:"):  # строка доступной памяти
                    return int(line.split()[1]) * 1024  # значение в кБ → байты: × 1024
    except OSError:  # файл недоступен
        pass  # следующий способ
    try:  # попытка 2: psutil
        import psutil  # импорт библиотеки psutil

        return int(psutil.virtual_memory().available)  # доступная память
    except ImportError:  # библиотеки нет
        pass  # следующий способ
    if sys.platform == "win32":  # ветка для: Windows
        try:  # попытка 3: WinAPI
            import ctypes  # импорт ctypes

            class MEMORYSTATUSEX(ctypes.Structure):  # структура состояния памяти Windows
                _fields_ = [  # поля структуры:
                    ("dwLength", ctypes.c_ulong),  # размер структуры
                    ("dwMemoryLoad", ctypes.c_ulong),  # загрузка памяти, %
                    ("ullTotalPhys", ctypes.c_ulonglong),  # вся физическая память
                    ("ullAvailPhys", ctypes.c_ulonglong),  # доступная физическая память
                    ("ullTotalPageFile", ctypes.c_ulonglong),  # весь файл подкачки
                    ("ullAvailPageFile", ctypes.c_ulonglong),  # доступный файл подкачки
                    ("ullTotalVirtual", ctypes.c_ulonglong),  # всё виртуальное пространство
                    ("ullAvailVirtual", ctypes.c_ulonglong),  # доступное виртуальное пространство
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),  # доступное расширенное виртуальное пространство
                ]

            st = MEMORYSTATUSEX()  # создать структуру
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)  # заполнить размер
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # запросить состояние памяти
            return int(st.ullAvailPhys)  # доступная физическая память в байтах
        except Exception:  # ошибка WinAPI
            pass  # далее — вернуть 0
    return 0  # не удалось определить — 0

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'CertStepGen', 'HedgeRouter', 'AuditStream', 'AuditLedger', 'ram_available_bytes',  # имя интерфейса
]  # конец перечня
