"""Полная самопроверка — модуль символьного ядра HAGI.

Каждая строка снабжена русским комментарием: что делает и по какой формуле.
"""

import json  # импорт стандартной библиотеки, нужный этой части
import time  # импорт стандартной библиотеки, нужный этой части
from pathlib import Path  # импорт стандартной библиотеки, нужный этой части
import numpy as np  # импорт стандартной библиотеки, нужный этой части
from hagi.core.tasks.tasks import make_task_suite  # имена из модуля ядра «tasks» (как до слияния)
from hagi.core.synthesis.venv_anf import anf_compose3, anf_eval, anf_to_expr, pack_vector_dyadic5, unpack_vector_dyadic5  # имена из «venv_anf» (модуль, где они определены)
from hagi.core.synthesis.venv_anf import anf_transform, pack_venn_16_flat, unpack_venn_16_flat, venn_head_eval  # имена из модуля ядра «venv_anf» (как до слияния)
from hagi.core.neural.neural import CRAMEngine  # имена из модуля ядра «neural» (как до слияния)
from hagi.core.neural.neural import ContinualGPMMemory  # имена из модуля ядра «neural» (как до слияния)
from hagi.core.neural.segmentation import GDNRAM  # имена из модуля ядра «segmentation» (как до слияния)
from hagi.core.neural.segmentation import hnet_opt_ratio  # имена из «segmentation» (модуль, где они определены)
from hagi.core.processor.processor import (  # открыть явный импорт из предыдущей части: имена processor
    GrowingHagiProcessor  # GrowingHagiProcessor — имя из предыдущей части
)  # конец импорта из предыдущей части
from hagi.core.processor.growth import (  # открыть явный импорт из предыдущей части: имена growth
    grow_deep  # grow_deep — имя из предыдущей части
)  # конец импорта из предыдущей части

# ==============================================================================
# 12. Полная самопроверка
# ==============================================================================


def self_test_core() -> None:
    """Полная самопроверка ядра: выучить набор задач, проверить предсказания и сохранение, отказ по отравленной метке, рост, упаковку, теоремы полиномов Жегалкина, движок CRAM, память и ассоциативную память.
            При любом расхождении возбуждает AssertionError; в конце печатает строку с итогами и числом секунд."""
    t0 = time.time()  # начало отсчёта времени
    suite = make_task_suite()  # набор задач
    model = GrowingHagiProcessor(node_budget=8000)  # процессор с бюджетом 8000 узлов
    for i, sp in enumerate(suite):  # по задачам набора
        ev = model.learn(sp, seed=10_000 + i, train_neural=True)  # выучить задачу с адаптером
        assert ev["action"] in ("COMMIT", "REUSE"), (sp.name, ev)  # каждая задача либо принята, либо переиспользована
    check = model.verify_all(suite)  # проверка всего набора
    # Задач 62, экспертов 61: одна задача закрывается повторным использованием готового эксперта (REUSE),
    # поэтому верное условие — «покрыты все задачи с точностью 1.0», а не «число экспертов = числу задач»
    assert check["accuracy"] == 1.0 and check["covered_tasks"] == check["expected_tasks"] and not check["missing_tasks"], check  # точность 1.0, покрыты все задачи набора, пропущенных нет
    assert model.predict("ADD", 100, 3)["status"] == "ABSTAIN"  # ADD(100, 3) вне области — отказ

    pm = GrowingHagiProcessor()  # отдельный процессор для проверки отката
    snap_before = pm._snapshot_state_with_optimizer()  # снимок состояния до обучения
    pev = pm.learn(  # выучить ADD с «отравленной» меткой
        next(s for s in suite if s.name == "ADD"), seed=77, poison_one_label=True  # задача ADD и зерно 77
    )
    assert pev["action"] == "HOLD"  # задача удержана
    snap_after = pm._snapshot_state_with_optimizer()  # снимок состояния после удержания
    assert np.array_equal(  # веса общего блока не изменились
        snap_before["core"]["gpm"]["W"], snap_after["core"]["gpm"]["W"]  # сравнение матриц W до и после
    )
    assert np.array_equal(snap_before["router"][0], snap_after["router"][0])  # веса маршрутизатора не изменились

    rep = grow_deep(model, suite, max_rounds=2, out_dir="/tmp/grow_test")  # два раунда роста
    assert rep["experts"] > len(suite), rep  # экспертов стало больше, чем задач в наборе
    assert model.verify_certificates()["all_ok"], model.verify_certificates()  # сертификаты корректны

    vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5, -0.75, 0.125, 0.0, -3.2, 0.05])  # пробный вектор из 11 компонент
    raw, dec = pack_vector_dyadic5(vec, stages=2)  # упаковка в 2 ступени: сырые байты и декодированный вектор
    assert np.allclose(dec, unpack_vector_dyadic5(raw))  # декодирование сырых байтов возвращает декодированный вектор

    vb = pack_venn_16_flat()  # упакованная таблица 16 функций Венна
    assert len(vb) == 13 and unpack_venn_16_flat(vb) == list(range(16))  # её размер 13 байт и она содержит числа 0..15
    for m in range(16):  # по маскам m = 0..15
        for x in (0, 1):  # по x ∈ {0, 1}
            for y in (0, 1):  # по y ∈ {0, 1}
                assert venn_head_eval(m, (x, y)) == ((m >> ((x << 1) | y)) & 1)  # голова функции Венна даёт бит m с номером 2·x + y

    tt = anf_transform([0, 1, 1, 0])  # коэффициенты АНФ таблицы [0, 1, 1, 0] (исключающее ИЛИ)
    assert anf_to_expr(tt, 2).eval((1, 0)) == 1  # выражение даёт 1 на (1, 0)
    c3 = anf_compose3(lambda a, b: a | b, [0, 0, 0, 1], [0, 1, 1, 0])  # композиция: (a | b) с таблицами [0, 0, 0, 1] и [0, 1, 1, 0]
    for x in (0, 1):  # по x
        for y in (0, 1):  # по y
            assert anf_eval(c3, 2, (x, y)) == ((x & y) | (x ^ y))  # результат равен (x & y) | (x ^ y)

    col = CRAMEngine.solve_coloring(  # раскраска цикла из 4 вершин в 2 цвета с заданным цветом вершины 0
        [(0, 1), (1, 2), (2, 3), (3, 0)], 4, 2, givens={0: 1}  # рёбра цикла 0–1–2–3–0, 4 вершины, 2 цвета, цвет вершины 0 задан = 1
    )  # закрытие вызова
    assert col["validity_certificate"] and col["givens_respected"]  # раскраска корректна и учитывает заданные цвета
    adj_m = [[1], [0, 2], [1, 3], [2]]  # список смежности цепочки из 4 вершин
    lz = CRAMEngine.solve_latent_maze(4, adj_m, goal=0)  # решение скрытого лабиринта с целью в вершине 0
    assert abs(lz["z"][0] - 21.0 / 55.0) < 1e-4  # первая компонента решения равна 21/55 с точностью 1e−4
    mf = CRAMEngine.mean_field_kl_step(  # шаг среднего поля
        np.array([[1.0, 0.0], [0.0, 1.0]]),  # единичные свидетельства 2×2
        np.zeros((2, 2, 2, 2)),  # нулевые парные поля
        np.full((2, 2), 0.5),  # средние значения всех 0.5
    )
    assert mf["monotonic"]  # шаг монотонный
    alt = CRAMEngine.alternating_latent_visible_solve(4, adj_m, goal=0, num_colors=2)  # чередующееся решение скрытого лабиринта в 2 цвета
    assert alt["monotone_drop"]  # энергия монотонно убывает
    pl = CRAMEngine.train_cram_pseudolikelihood(  # обучение по псевдо-правдоподобию на трёх примерах в 2 цвета
        np.array([[0, 1], [0, 1], [1, 0]]), num_colors=2  # три примера-строки в 2 цвета
    )  # закрытие вызова
    assert pl["mean_pl_loss"] < 0.7  # средняя потеря меньше 0.7

    gpm = ContinualGPMMemory(6, 2, seed=1)  # память на 6 входов и 2 выхода
    X1 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])  # входы первой задачи: два орта
    Y1 = np.array([[1.0, -1.0], [0.5, 0.5]])  # цели первой задачи
    gpm.train_task_ef21_muon(X1, Y1, steps=20)  # обучить первую задачу за 20 шагов
    pred1_before = X1 @ gpm.W  # предсказание на первой задаче до закрепления: X1·W
    gpm.consolidate_subspace(X1)  # закрепить подпространство первой задачи
    X2 = np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])  # вход второй задачи
    Y2 = np.array([[2.0, -2.0]])  # цель второй задачи
    gpm.train_task_ef21_muon(X2, Y2, steps=20)  # обучить вторую задачу за 20 шагов
    pred1_after = X1 @ gpm.W  # предсказание на первой задаче после обучения второй
    assert float(np.max(np.abs(pred1_before - pred1_after))) < 1e-10  # предсказание не изменилось (допуск 1e−10) — забывания нет

    assert abs(hnet_opt_ratio(1.0, 4.0, 8.0) - 4.0) < 1e-9  # оптимальная доля: (2·4·8/1)^(1/3) = 4 с точностью 1e−9
    ram = GDNRAM(8, 4)  # память на 8 слотов размерности 4
    ram.store(2, np.array([1.0, 2.0, 3.0, 4.0]))  # записать в слот 2
    ram.store(5, np.array([9.0, 8.0, 7.0, 6.0]))  # записать в слот 5
    assert np.allclose(ram.load(2), [1.0, 2.0, 3.0, 4.0])  # чтение слота 2 возвращает записанное
    assert np.allclose(ram.load_stutter_head([2, 2, 5, 5]), [1.0, 2.0, 3.0, 4.0])  # «запинающаяся» голова [2, 2, 5, 5] читает первый адрес после схлопывания повторов — слот 2

    save_path = Path("v9_selftest_model.json")  # путь к файлу модели
    model.save(save_path)  # сохранить модель
    reloaded = GrowingHagiProcessor.load(save_path)  # загрузить модель
    assert reloaded.verify_all(suite)["accuracy"] == 1.0  # точность загруженной модели 1.0
    assert reloaded.predict("ADD", 7, -3)["value"] == 4  # проверка: ADD(7, −3) = 4
    assert reloaded.predict("ADD", 99, 1)["status"] == "ABSTAIN"  # ADD(99, 1) вне области — отказ
    for nm, ex in model.experts.items():  # по экспертам
        if ex.adapter is not None:  # у которых есть адаптер
            xs0 = tuple([ex.domains[0][0]] + [d[0] for d in ex.domains[1:]])  # первый допустимый вход: первые значения каждой области
            if ex.supports(xs0):  # эксперт поддерживает этот вход
                p1 = model.predict(nm, *xs0, mode="neural")["value"]  # нейросетевое предсказание до сохранения
                p2 = reloaded.predict(nm, *xs0, mode="neural")["value"]  # нейросетевое предсказание после загрузки
                assert abs(p1 - p2) < 1e-9, (nm, p1, p2)  # предсказания совпадают (допуск 1e−9)
    save_path.unlink(missing_ok=True)  # удалить файл, если он существует

    nq = model.neural_quality_summary()  # сводка нейросетевого качества
    dt = time.time() - t0  # затраченное время
    print(  # печать итога
        f"SELF_TEST PASS ({dt:.2f}s): {check['committed']} base experts acc=1.0 -> "  # первая часть строки: время, число экспертов, точность
        f"{rep['experts']} total experts, {len(model.theorems)} theorems, "  # всего экспертов и число теорем
        f"neural_acc={nq['mean_val_exact_acc']:.4f}, neural_mse={nq['mean_val_mse']:.4f}, "  # средняя точность и средний квадрат ошибки адаптеров
        f"neural_bytes={nq['total_neural_bytes']}, certs_ok=True"  # байты нейросетевой части и признак корректности сертификатов
    )  # закрытие print

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'self_test_core',  # имя интерфейса
]  # конец перечня
