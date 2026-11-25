"""Единый интерфейс командной строки — модуль символьного ядра HAGI.

Каждая строка снабжена русским комментарием: что делает и по какой формуле.
"""

import argparse  # импорт стандартной библиотеки, нужный этой части
import json  # импорт стандартной библиотеки, нужный этой части
import time  # импорт стандартной библиотеки, нужный этой части
from pathlib import Path  # импорт стандартной библиотеки, нужный этой части
import numpy as np  # импорт стандартной библиотеки, нужный этой части
from hagi.core.processor.unified_processor import GrowthProcessorBase  # имена из «unified_processor» (прежний общий импорт из пакета наследия)
from hagi.core.processor.stored_growth import grow_from_suite, self_test_unified  # имена из «stored_growth» (прежний общий импорт из пакета наследия)
from hagi.core.memory.selfcheck import self_test  # имена из «selfcheck» (прежний общий импорт из пакета наследия)
from hagi.core.processor.benchmark import self_test_optimal  # имена из «benchmark» (прежний общий импорт из пакета наследия)
from hagi.core.tasks.tasks import make_task_suite  # имена из модуля ядра «tasks» (как до слияния)
from hagi.core.processor.processor import (  # открыть явный импорт из предыдущей части: имена processor
    GrowingHagiProcessor  # GrowingHagiProcessor — имя из предыдущей части
)  # конец импорта из предыдущей части
from hagi.core.processor.growth import (  # открыть явный импорт из предыдущей части: имена growth
    grow_deep  # grow_deep — имя из предыдущей части
)  # конец импорта из предыдущей части
from hagi.core.memory.selftest import (  # открыть явный импорт из предыдущей части: имена selftest
    self_test_core  # self_test_core — имя из предыдущей части
)  # конец импорта из предыдущей части
from hagi.base.config import STATE_FORMAT  # единая метка формата состояния и перечень прежних меток

# ==============================================================================
# Единый интерфейс командной строки
# ==============================================================================


def _benchmark() -> None:
    """Замер ядра: выучить набор задач, выполнить рост, измерить время, точность и занимаемую память.
    
            Считает время обучения набора и двух раундов роста, точность на наборе, число
            экспертов и теорем после роста, метрики адаптеров, число байт тензоров в памяти
            (общий блок и вся модель), размер этого файла и корректность сертификатов.
            Печатает итог как JSON с отступом 2."""
    """Замер ядра: выучить набор задач, выполнить рост, измерить время, точность и занимаемую память.
    
            Считает время обучения набора и двух раундов роста, точность на наборе, число
            экспертов и теорем после роста, метрики адаптеров, число байт тензоров в памяти
            (общий блок и вся модель), размер этого файла и корректность сертификатов.
            Печатает итог как JSON с отступом 2."""

    suite = make_task_suite()  # набор задач
    model = GrowingHagiProcessor(node_budget=8000)  # процессор с бюджетом 8000 узлов
    t0 = time.time()  # начало отсчёта
    for i, sp in enumerate(suite):  # по задачам набора
        model.learn(sp, seed=10_000 + i, train_neural=True)  # выучить задачу с адаптером
    learn_elapsed = time.time() - t0  # время обучения набора
    check = model.verify_all(suite)  # проверка всего набора
    t0 = time.time()  # начало отсчёта
    rep = grow_deep(model, suite, max_rounds=2, out_dir=".omc/bench")  # два раунда роста
    grow_elapsed = time.time() - t0  # время роста
    nq = model.neural_quality_summary()  # сводка нейросетевого качества
    resident = sum(  # байты массивов NumPy среди атрибутов модели
        int(v.nbytes) for v in model.__dict__.values() if isinstance(v, np.ndarray)  # размер массива в байтах
    )
    core = model.core  # общий блок весов
    resident_core = sum(  # байты массивов среди атрибутов общего блока
        int(a.nbytes) for a in vars(core).values() if isinstance(a, np.ndarray)  # размер массива в байтах
    )
    resident_core += sum(  # прибавить массивы оптимизатора общего блока
        int(a.nbytes) for a in vars(core.gpm).values() if isinstance(a, np.ndarray)  # размер массива в байтах
    )
    result = {  # итог словарём
        "version": STATE_FORMAT,  # версия схемы
        "exact_committed": check["committed"],  # число принятых экспертов
        "exact_accuracy": check["accuracy"],  # точность на наборе
        "experts_after_growth": rep["experts"],  # число экспертов после роста
        "theorems": rep["theorems"],  # число теорем
        "neural_val_acc": round(nq["mean_val_exact_acc"], 4),  # средняя точность адаптеров на val, округлённая до 4 знаков
        "neural_val_mse": round(nq["mean_val_mse"], 4),  # средний квадрат ошибки на val, округлённый до 4 знаков
        "neural_payload_bytes": nq["total_neural_bytes"],  # байты упакованной нейросетевой части
        "resident_tensor_bytes_core": resident_core,  # байты тензоров общего блока
        "resident_tensor_bytes_model": resident + resident_core,  # байты тензоров всей модели
        "file_bytes": Path(__file__).stat().st_size,  # размер этого файла в байтах
        "learn_suite_seconds": round(learn_elapsed, 2),  # время обучения набора в секундах
        "grow_2rounds_seconds": round(grow_elapsed, 2),  # время двух раундов роста в секундах
        "certs_ok": model.verify_certificates()["all_ok"],  # сертификаты корректны
    }
    print(json.dumps(result, indent=2))  # напечатать итог как JSON с отступом 2


def _cli() -> None:
    """Разобрать аргументы командной строки и запустить выбранное действие.
    
            Флаги --self-test, --self-test-core, --self-test-optimal, --self-test-legacy — самопроверки
            разных версий ядра; --grow, --grow-deep — циклы роста; --benchmark-core — замер;
            --max-rounds — число раундов (по умолчанию 4); --out-dir — каталог вывода.
            Без флага печатается справка."""
    _p = argparse.ArgumentParser(  # парсер аргументов
        description="HAGI: символьное ядро — самопроверки, циклы роста, замеры"  # описание программы
    )
    _p.add_argument("--self-test", action="store_true", help="самопроверка: цикл роста единым процессором")  # флаг самопроверки версии 8.4
    _p.add_argument("--self-test-core", action="store_true", help="самопроверка канонического ядра")  # флаг самопроверки канонического ядра
    _p.add_argument("--self-test-optimal", action="store_true", help="самопроверка оптимального процессора")  # флаг самопроверки версии 7.4
    _p.add_argument("--self-test-legacy", action="store_true", help="самопроверка наследия")  # флаг самопроверки версии 5
    _p.add_argument("--grow", action="store_true", help="цикл роста единым процессором")  # флаг цикла роста версии 8.4
    _p.add_argument("--grow-deep", action="store_true", help="канонический цикл роста")  # флаг канонического цикла роста
    _p.add_argument("--benchmark-core", action="store_true", help="замер канонического ядра")  # флаг замера канонического ядра
    _p.add_argument("--max-rounds", type=int, default=4)  # максимальное число раундов (по умолчанию 4)
    _p.add_argument("--out-dir", default="grow_out")  # каталог вывода (по умолчанию «grow_out»)
    _a = _p.parse_args()  # разобранные аргументы
    if _a.self_test:  # если указана самопроверка единым процессором
        self_test_unified()  # запустить её
    elif _a.self_test_core:  # иначе если указана самопроверка ядра
        self_test_core()  # запустить её
    elif _a.self_test_optimal:  # иначе самопроверка оптимального процессора
        self_test_optimal()  # запустить её
    elif _a.self_test_legacy:  # иначе самопроверка наследия
        self_test()  # запустить её
    elif _a.grow:  # иначе цикл роста единым процессором
        _suite = make_task_suite()  # набор задач
        _m = GrowthProcessorBase()  # базовый растущий процессор
        print(  # напечатать итог роста
            json.dumps(  # превращение словаря в текст JSON
                grow_from_suite(_m, _suite, max_rounds=_a.max_rounds, out_dir=_a.out_dir),  # итог grow_from_suite с числом раундов и каталогом
                indent=2,  # отступ 2
            )
        )  # закрытие json.dumps
    elif _a.grow_deep:  # иначе канонический цикл роста
        _suite = make_task_suite()  # набор задач
        _m = GrowingHagiProcessor()  # канонический процессор
        print(  # напечатать итог роста
            json.dumps(  # превращение словаря в текст JSON
                grow_deep(_m, _suite, max_rounds=_a.max_rounds, out_dir=_a.out_dir),  # итог grow_deep с числом раундов и каталогом
                indent=2,  # отступ 2
            )
        )
    elif _a.benchmark_core:  # иначе замер ядра
        _benchmark()  # запустить замер
    else:  # иначе флагов нет
        _p.print_help()  # напечатать справку

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
]  # конец перечня
