"""Журнал учёта статистических проверок и определение доступной памяти.

AuditLedger ведёт счёт «альфы» — допустимого суммарного уровня ошибок первого
рода для внешних статистических тестов. У каждой задачи есть имя; потраченная
альфа суммируется: spent[task] += alpha. Остаток равен
    remaining = max(0, delta_total − Σ spent).
Трата разрешена, только если alpha ≤ remaining + 1e-12, alpha конечна и
положительна, имя непустое и итоговый тест ещё не вскрыт. После вызова
mark_final_test_touched дальнейшие траты запрещены. Полный перебор конечной
области детерминирован и альфу не расходует; сам журнал не является процедурой
построения доверительных границ.

ram_available_bytes возвращает объём доступной оперативной памяти в байтах:
из строки MemAvailable файла /proc/meminfo (значение в кБ · 1024), а при
недоступности файла — через атрибут пакета hagi.core.
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов
import math  # импорт стандартной библиотеки для разбора выражений, данных, графов и случайности
from dataclasses import dataclass, field  # декоратор класса-записи и фабрика значений по умолчанию
from hagi.core.processor.self_development import _ram_available_bytes  # имена из модуля «self_development» (прямой импорт вместо обращения через псевдоним пакета)

# ==============================================================================  # разделитель
# 7. Журнал аудита и регулятор RAM  # заголовок раздела
# ==============================================================================  # разделитель


@dataclass  # класс-запись
class AuditLedger:
    """Явный учёт альфы для внешне заданных статистических тестов.

        Точный перебор заявленной конечной области детерминирован и альфу не
        расходует. Сам журнал не является процедурой доверительных границ."""

    delta_total: float = 0.05  # общий бюджет альфы, по умолчанию 0.05
    spent: dict[str, float] = field(default_factory=dict)  # потрачено по задачам; по умолчанию пусто
    rejected_tasks: set[str] = field(default_factory=set)  # имена отвергнутых задач; по умолчанию пусто
    final_test_touched: bool = False  # итоговый тест вскрыт; по умолчанию нет

    def __post_init__(self) -> None:  # проверка после создания записи
        """Проверить бюджет альфы и записи о тратах; при нарушении — ValueError."""
        self.delta_total = float(self.delta_total)  # бюджет как число с плавающей точкой
        if not math.isfinite(self.delta_total) or not (0.0 < self.delta_total < 1.0):  # бюджет не конечен или вне (0, 1)
            raise ValueError("delta_total must be finite and in (0, 1)")  # ошибка: бюджет вне допустимого диапазона
        self.spent = {str(k): float(v) for k, v in self.spent.items()}  # ключи — строки, значения — числа
        if any(not k or not math.isfinite(v) or v < 0.0 for k, v in self.spent.items()):  # пустое имя, не конечное или отрицательное значение
            raise ValueError(  # ошибка
                "spent alpha entries must be named, finite, and nonnegative"  # текст ошибки
            )  # конец raise
        if sum(self.spent.values()) > self.delta_total + 1e-12:  # сумма трат больше бюджета (с допуском 1e-12)
            raise ValueError("serialized alpha spend exceeds delta_total")  # ошибка: траты превышают бюджет

    def remaining(self) -> float:  # остаток альфы
        """Остаток бюджета: max(0, delta_total − Σ spent)."""
        return max(0.0, self.delta_total - sum(self.spent.values()))  # max(0, budget − Σ потраченного)

    def spend_anytime(self, task: str, alpha: float) -> bool:  # потратить альфу на задачу
        """Потратить alpha на задачу task; True, если трата прошла.

            Отказ (False): итоговый тест вскрыт, имя пустое, alpha не конечна
            или ≤ 0, либо alpha > remaining + 1e-12."""
        alpha = float(alpha)  # alpha как число с плавающей точкой
        if (  # любая из причин отказа
            self.final_test_touched  # итоговый тест вскрыт
            or not task  # имя задачи пустое
            or not math.isfinite(alpha)  # alpha не конечна
            or alpha <= 0.0  # неположительная трата: alpha ≤ 0
        ):  # конец условия
            return False  # трата не прошла
        if alpha > self.remaining() + 1e-12:  # alpha больше остатка (с допуском)
            return False  # трата не прошла
        self.spent[task] = self.spent.get(task, 0.0) + alpha  # накопление: spent[task] := spent[task] + alpha
        return True  # трата прошла

    def record_reject(self, task: str) -> None:  # записать отвергнутую задачу
        """Добавить имя задачи task в множество отвергнутых."""
        if not task:  # имя пустое
            raise ValueError("task name must be nonempty")  # ошибка: имя должно быть непустым
        self.rejected_tasks.add(task)  # добавить в множество

    def mark_final_test_touched(self) -> None:  # отметить вскрытие итогового теста
        """Запретить дальнейшие траты альфы после вскрытия запечатанного итогового теста."""
        self.final_test_touched = True  # флаг: итоговый тест вскрыт


def ram_available_bytes() -> int:  # доступная память в байтах
    """Доступная оперативная память в байтах (из /proc/meminfo или через атрибут пакета hagi.core)."""
    try:  # файл может отсутствовать
        with open("/proc/meminfo", "r", encoding="utf-8") as f:  # открыть файл сведений о памяти
            for line in f:  # по строкам файла
                if line.startswith("MemAvailable:"):  # строка доступной памяти
                    return int(line.split()[1]) * 1024  # значение в кБ · 1024 = байты
    except OSError:  # файл недоступен
        pass  # перейти к запасному способу
    return _ram_available_bytes()  # запасной способ: атрибут пакета hagi.core

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'AuditLedger', 'ram_available_bytes',  # имя интерфейса
]  # конец перечня
