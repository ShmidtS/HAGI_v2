"""Единое место чтения настроек проекта из переменных окружения.

Правило: любой модуль, которому нужен_DEVICE, архитектура, каталог
контрольной точки или флаги режимов, берёт их здесь, а не читает os.environ
сам. Тогда список всех ручек проекта виден в одном файле, а значения
одинаковы во всех модулях.

Ручки (имя переменной → значение по умолчанию → смысл):
- HAGI_DEVICE        → "auto"      : устройство обучения и инференса: auto / cpu / cuda;
- HAGI_NEURAL_DEVICE → "auto"      : устройство нейронной части (та же шкала, нижний регистр);
- HAGI_THREADS       → не задано   : число потоков torch на процессоре (не задано = 1)
- HAGI_THREADS       → не задано   : число потоков torch на процессоре (не задано = 1)
- HAGI_ARCH          → "optimal"   : архитектура тела модели: optimal (причинная, по умолчанию) / hnet / rope;
- HAGI_CKPT_DIR      → <корень>/checkpoints : каталог весов, книги доменов и журналов;
- HAGI_SYMBOLIC      → "0"         : "1" — включить символьную фазу конвейера;
- HAGI_NO_NEURAL     → не задано   : "1" — не обучать нейронную часть;
- HAGI_DIM, HAGI_HIDDEN → 64, 128  : ширина и скрытый слой модели, если аргументы не заданы.

Все функции читают переменную В МОМЕНТ вызова (а не при импорте): тест
может подменить окружение, и модуль это увидит.
"""
from __future__ import annotations  # аннотации типов не вычисляются при импорте

import os  # чтение переменных окружения
from pathlib import Path  # пути к файлам

ROOT = Path(__file__).resolve().parent.parent  # корень проекта: каталог, в котором лежит пакет hagi

TOKEN_BITS = 6  # число битов на один токен выражения в префиксном коде (общий формат кодера и сертификатов)

# Обучение тела (train_guarded): размер пакета и шаг Adam. Выбран по ЗАБЫВАНИЮ, а не по
# скорости: приём одного документа (25 КБ, 2 эпохи), CE старых доменов до -> после:
#   batch 256 / lr 8e-3: новый 0.35, старые 2.45 -> 5.03 (дрейф +2.58, худший +6.24)
#   batch 128 / lr 2e-3: новый 0.37, старые 2.45 -> 3.75 (дрейф +1.29)
#   batch  64 / lr 1e-3: новый 0.42, старые 2.45 -> 3.45 (дрейф +1.00, худший +3.16)
# Крупный пакет с пропорциональным шагом быстрее, но кратно усиливает забывание
# старых доменов; ремонт забывания не покрывает такой разрыв.
# Переопределяются переменными окружения HAGI_TRAIN_BATCH и HAGI_TRAIN_LR.
TRAIN_BATCH = int(os.environ.get("HAGI_TRAIN_BATCH", "64"))  # размер пакета обучения тела
TRAIN_LR = float(os.environ.get("HAGI_TRAIN_LR", "1e-3"))  # шаг Adam обучения тела

# Единая метка формата сохраняемого состояния символьного ядра. Номеров версий нет:
# раньше у каждого слоя была своя строка ("5.0.0", "7.4.0-research", "8.4.0", "9.0.0"),
# теперь все файлы состояния несут одну метку, а загрузка принимает и прежние строки.
# Переопределяется переменной окружения HAGI_STATE_FORMAT.
STATE_FORMAT = os.environ.get("HAGI_STATE_FORMAT", "hagi-state")  # метка формата состояния
LEGACY_STATE_LABELS = frozenset({"5.0.0", "7.4.0-research", "8.4.0", "9.0.0"})  # прежние метки, которые загрузка ещё принимает
MODEL_FILE_FORMAT = "HAGI-MODEL"  # метка формата файла модели GrowingHagiProcessor (без номера версии)
STATE_FILE_FORMAT = "HAGI-STATE"  # метка формата файла состояния GrowthProcessorBase (без номера версии)
LEGACY_FILE_FORMATS = {"HAGI-MODEL": frozenset({"HAGI-V9"}), "HAGI-STATE": frozenset({"HAGI-V8-STATE"})}  # прежние метки файлов, которые загрузка ещё принимает


def device() -> str:
    """Устройство обучения/инференса: значение HAGI_DEVICE, иначе "auto"."""
    return os.environ.get("HAGI_DEVICE", "auto")  # читать в момент вызова


def cpu_threads() -> int | None:
    """Число потоков torch на процессоре: HAGI_THREADS, иначе None (тогда модель ставит 1)."""
    v = os.environ.get("HAGI_THREADS")  # прочитать в момент вызова
    return int(v) if v and v.strip() else None  # целое число или None


def cpu_threads() -> int | None:
    """Число потоков torch на процессоре: HAGI_THREADS, иначе None (тогда модель ставит 1)."""
    v = os.environ.get("HAGI_THREADS")  # прочитать в момент вызова
    return int(v) if v and v.strip() else None  # целое число или None


def neural_device_mode() -> str:
    """Режим выбора устройства нейронной части: HAGI_NEURAL_DEVICE в нижнем регистре, иначе "auto"."""
    return os.environ.get("HAGI_NEURAL_DEVICE", "auto").lower()  # читать и привести к нижнему регистру


def arch() -> str:
    """Архитектура тела модели: HAGI_ARCH, иначе "optimal" (причинная модель; hnet/rope — прежние)."""
    return os.environ.get("HAGI_ARCH", "optimal")  # читать в момент вызова


def symbolic_enabled() -> bool:
    """Включена ли символьная фаза: HAGI_SYMBOLIC == "1"."""
    return os.environ.get("HAGI_SYMBOLIC", "0") == "1"  # сравнение с "1"


def train_neural_enabled() -> bool:
    """Обучать ли нейронную часть: выключено только при HAGI_NO_NEURAL == "1"."""
    return os.environ.get("HAGI_NO_NEURAL") != "1"  # всё, кроме "1", значит «включено»


def model_dims(dim: int | None, hidden: int | None) -> tuple[int, int]:
    """Ширина и скрытый слой модели по аргументам и переменным HAGI_DIM / HAGI_HIDDEN.

    Правило приоритета: явный аргумент > переменная окружения > умолчание.
    Формула: d = dim, если dim задано; иначе int(HAGI_DIM), если задано; иначе 64.
    Аналогично для h с умолчанием 128."""
    d = int(os.environ.get("HAGI_DIM", dim or 64) if os.environ.get("HAGI_DIM") or dim is None else dim)  # ширина по правилу приоритета
    h = int(os.environ.get("HAGI_HIDDEN", hidden or 128) if os.environ.get("HAGI_HIDDEN") or hidden is None else hidden)  # скрытый слой по правилу приоритета
    return d, h  # вернуть пару (ширина, скрытый слой)
