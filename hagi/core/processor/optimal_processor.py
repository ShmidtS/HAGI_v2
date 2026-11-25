"""Оптимальный процессор и сертифицированный оптимизатор: обучает символьных экспертов на конечных областях.
Обучение: область делится на обучающую и отложенную части, кандидаты отсеваются по малой пробе, затем
проверяются точно на всех точках; применяются шаблоны, индукция многочленов и поиск по грамматике. Отдельно — подбор весов необязательного tanh-слоя.
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов
import json, os  # модули стандартной библиотеки: аргументы, дерево, копия, куча, произведения, JSON, математика, ОС, случайность
import time  # статистика, системные функции, время
from pathlib import Path  # пути к файлам
from typing import Optional, Sequence  # типы аннотаций
import numpy as np  # численные массивы numpy
from hagi.core.language.expr_dsl import (  # импорт из модуля expr_dsl
    Program  # Program — импортируемое имя
)  # конец импорта
from hagi.core.tasks.stored_tasks import (  # импорт из модуля tasks
    TaskSpec  # TaskSpec — импортируемое имя
)  # конец импорта
from hagi.core.tasks.candidates import (  # импорт из модуля candidates
    candidate_programs  # candidate_programs — импортируемое имя
)  # конец импорта
from hagi.core.neural.fiber import (  # импорт из модуля fiber
    NeuralFiber  # NeuralFiber — импортируемое имя
)  # конец импорта
from hagi.core.processor.experts import (  # импорт из модуля experts
    Expert,  # Expert — импортируемое имя
    GrowthEvent  # GrowthEvent — импортируемое имя
)  # конец импорта
from hagi.core.synthesis.prefilter import DEFAULT_NODE_BUDGET, DEFAULT_POP_BUDGET, SIEVE_SIZE, LegacyProcessor, VALIDATION_CHUNK  # имена из «prefilter» (модуль, где они определены)
from hagi.core.processor.experts import Expert, GrowthEvent  # имена из «experts» (модуль, где они определены)
from hagi.core.neural.fiber import NeuralFiber  # имена из «fiber» (модуль, где они определены)
from typing import Optional, Sequence  # имена из «stdlib:typing» (модуль, где они определены)
from hagi.core.language.expr_dsl import Program  # имена из «expr_dsl» (модуль, где они определены)
from hagi.core.tasks.stored_tasks import TaskSpec  # имена из «stored_tasks» (модуль, где они определены)
from hagi.core.tasks.candidates import candidate_programs  # имена из «candidates» (модуль, где они определены)
from hagi.core.language.code_length import (  # импорт из модуля code_length
    _TOKEN_BITS,  # _TOKEN_BITS — импортируемое имя
    _gamma_bits,  # _gamma_bits — импортируемое имя
    _program_description_bits,  # _program_description_bits — импортируемое имя
    batch_fits_safe  # batch_fits_safe — импортируемое имя
)  # конец импорта
from hagi.core.synthesis.best_first import (  # импорт из модуля best_first
    SynthesisStats,  # SynthesisStats — импортируемое имя
    iter_candidates_mdl  # iter_candidates_mdl — импортируемое имя
)  # конец импорта
from hagi.core.processor.experts import GrowingMathProcessor  # имена из модуля «experts» (владелец определения)
from hagi.core.synthesis.induction import Theorem  # имя Theorem определено в части «induction» (явный импорт вместо инъекции)
from hagi.core.synthesis.induction import _eval_poly  # имя _eval_poly определено в части «induction» (явный импорт вместо инъекции)
from hagi.core.synthesis.induction import _exact_multivariate_poly_fit  # имя _exact_multivariate_poly_fit определено в части «induction» (явный импорт вместо инъекции)
from hagi.core.synthesis.induction import _exact_poly_fit  # имя _exact_poly_fit определено в части «induction» (явный импорт вместо инъекции)
from hagi.core.synthesis.induction import _multivariate_poly_expr  # имя _multivariate_poly_expr определено в части «induction» (явный импорт вместо инъекции)
from hagi.core.synthesis.induction import _poly_expr  # имя _poly_expr определено в части «induction» (явный импорт вместо инъекции)
from hagi.base.config import LEGACY_STATE_LABELS, STATE_FORMAT  # единая метка формата состояния и перечень прежних меток

# ------------------------------- процессор оптимальный ----------------------------------


class OptimalMathProcessor(LegacyProcessor):  # обучатель символьных программ на конечной области с защитами ускорения
    """Обучатель на конечной области: для задачи ищется точная программа и записывается как эксперт. Ускоряющие проверки (проба, векторный фильтр) только отвергают кандидатов; принять кандидата можно лишь после точной скалярной проверки на всей области."""

    def __init__(  # создать процессор
        self,  # self — новый объект
        lookup_limit: int = 16,  # lookup_limit — предел размера таблицы поиска
        mdl_lambda: float = 0.001,  # mdl_lambda — вес MDL-штрафа при равных точных решениях
        node_budget: int = 20_000,  # node_budget — бюджет узлов поиска
        pop_budget: int = 50_000,  # pop_budget — бюджет популяции поиска
        device: str = "cpu",  # device — устройство для векторной проверки
    ):  # конец сигнатуры
        if float(mdl_lambda) <= 0:  # вес MDL-штрафа должен быть положительным
            raise ValueError(  # ошибка: вес должен быть положительным
                "mdl_lambda must be positive for the exact-fit MDL tie-break"  # ), конец вызова
            )  # инициализировать базовый процессор: lookup_limit, mdl_lambda
        super().__init__(lookup_limit, mdl_lambda)  # бюджет узлов поиска
        self.node_budget = int(node_budget)  # бюджет популяции поиска
        self.pop_budget = int(pop_budget)  # устройство векторной проверки
        self.device = device  # статистика синтеза
        self.stats = SynthesisStats()  # список теорем пока пуст
        self.theorems: list[Theorem] = []  # список теорем пока пуст

    def learn(  # def learn(, выучить задачу: найти точную программу и записать эксперта
        self,  # self — процессор
        spec: TaskSpec,  # spec — описание задачи
        seed: int = 0,  # seed — зерно случайности
        train_fraction: float = 0.70,  # train_fraction — доля обучающих точек
        poison_one_label: bool = False,  # poison_one_label — испортить одну метку (проверка защиты)
        train_neural: bool = False,  # train_neural — подбирать ли нейронный слой
        deadline: Optional[float] = None,  # deadline — предельное значение time.monotonic() или None
        candidate_hint: Optional[Program] = None,  # candidate_hint — программа-подсказка или None
    ) -> GrowthEvent:  # ) -> GrowthEvent:, возвращает событие роста
        if spec.name in self.experts:  # задача уже выучена
            return GrowthEvent(self.version, spec.name, "REUSE", "already learned")  # событие REUSE: программа уже есть
        if deadline is not None and time.monotonic() >= deadline:  # время истекло: deadline ≤ now
            return self._hold(spec, "TIME_BUDGET_EXHAUSTED", 0, 0)  # отказ: бюджет времени исчерпан
        domain = spec.inputs()  # domain — все допустимые входы задачи
        if not domain:  # область пуста
            return self._hold(spec, "EMPTY_VALID_DOMAIN", 0, 0)  # отказ: пустая область
        if len(domain) > 1:  # для области из более чем одной точки доля ограничивается
            train_fraction = min(max(float(train_fraction), 0.05), 0.95)  # доля обучающих точек: train_fraction = min(max(f, 0.05), 0.95)
        rng = np.random.default_rng(seed)  # генератор случайных чисел с зерном seed
        perm = rng.permutation(len(domain))  # случайная перестановка номеров 0..len(domain)−1
        if spec.family == "BoolVenn":  # класс задач BoolVenn
            n_train = max(1, len(domain) - 1)  # на обучение все точки, кроме одной: n_train = len − 1
        else:  # иначе
            n_train = max(  # число обучающих точек
                1, min(len(domain) - 1, int(round(len(domain) * train_fraction)))  # число обучающих точек: n_train = max(1, min(len − 1, round(len·train_fraction)))
            )  # ), конец max
        train_x = [domain[int(i)] for i in perm[:n_train]]  # обучающие входы — первые n_train элементов перестановки
        val_x = [domain[int(i)] for i in perm[n_train:]]  # отложенные входы — остальные элементы
        all_y = {xs: int(spec.teacher(*xs)) for xs in domain}  # метки учителя: all_y[x] = teacher(x) для всех входов
        train_y = [all_y[xs] for xs in train_x]  # метки обучающих входов
        val_y = [all_y[xs] for xs in val_x]  # метки отложенных входов
        if poison_one_label and train_y:  # режим порчи метки: обучающие метки не пусты
            train_y[0] = int(train_y[0]) + 1  # первая метка увеличена на 1
            # намеренная порча метки: по заведомо испорченным меткам программа не ищется
            # вместо поиска — отказ с кодом POISONED_LABEL_INJECTION
            return self._hold(  # отказ: в метки внедрена ошибка
                spec, "POISONED_LABEL_INJECTION", len(train_x), len(val_x)  # задача, код отказа, размеры обучающей и отложенной выборок
            )  # ), конец вызова

        neural: Optional[NeuralFiber] = None  # нейронный слой ещё не подобран

        def maybe_fit_neural() -> Optional[NeuralFiber]:  # подобрать нейронный слой (tanh-сеть), если задано
            if not train_neural or not val_x:  # подгонка не задана или нет отложенных точек
                return None  # слой не подбирается
            try:  # первая попытка — подгонка с защитой от роста ошибки
                return fit_fiber_guarded(  # вызвать защищённую подгонку
                    train_x, train_y, val_x, val_y, seed=seed + 991, epochs=120  # обучающие и отложенные данные, зерно seed + 991, 120 эпох
                )  # ), конец вызова
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):  # численная ошибка
                try:  # вторая попытка — обычная подгонка
                    return NeuralFiber.fit(  # вызвать подгонку слоя
                        train_x, train_y, val_x, val_y, seed=seed + 991, epochs=60  # те же данные, зерно seed + 991, 60 эпох
                    )  # ), конец вызова
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):  # снова численная ошибка
                    return None  # слоя нет

        def commit(  # принять программу и записать эксперта
            program: Program, reason: str, code_length: Optional[int] = None  # program — программа, reason — причина, code_length — длина в битах или None
        ) -> GrowthEvent:  # ) -> GrowthEvent:, возвращает событие COMMIT
            nonlocal neural  # нейронный слой берётся из внешней области видимости
            if neural is None and train_neural:  # слой ещё не подобран, а подгонка задана
                neural = maybe_fit_neural()  # подобрать слой
            bits = int(  # длина описания программы в битах
                _program_description_bits(program)  # bits = длина описания программы,
                if code_length is None  # если code_length не задан,
                else code_length  # иначе bits = code_length
            )  # ), конец int
            parent_names: tuple[str, ...] = ()  # имена родительских экспертов
            parent_defaults = getattr(spec.teacher, "__defaults__", None)  # значения по умолчанию учителя — родители композиции
            if (  # если
                candidate_hint is not None  # подсказка задана
                and parent_defaults  # и есть значения по умолчанию
                and len(parent_defaults) >= 2  # их не меньше двух
            ):  # ):, конец условия
                outer_parent, inner_parent = parent_defaults[0], parent_defaults[1]  # внешний и внутренний родитель
                if isinstance(outer_parent, Expert) and isinstance(  # внешний родитель — эксперт и
                    inner_parent, Expert  # внутренний родитель — эксперт
                ):  # ):, конец условия
                    parent_names = (outer_parent.name, inner_parent.name)  # имена внешнего и внутреннего родителя
            expert = Expert(  # новый эксперт
                spec.name,  # имя
                spec.family,  # семейство задач
                program,  # программа
                spec.domains,  # области аргументов
                spec.valid_rule,  # правило допустимости
                bits,  # длина кода в битах
                neural,  # нейронный слой или None
                len(domain),  # размер области
                0,  # число ошибок 0
                parent_names,  # имена родителей
            )  # ), конец Expert
            self.experts[spec.name] = expert  # записать эксперта в библиотеку
            self.version += 1  # версия модели + 1
            self.total_commits += 1  # число принятых + 1
            ev = GrowthEvent(  # событие COMMIT
                self.version,  # версия
                spec.name,  # имя задачи
                "COMMIT",  # тип события COMMIT
                reason,  # причина
                program.pretty(),  # текст программы
                bits,  # длина кода в битах
                len(train_x),  # размер обучающей выборки
                len(val_x),  # размер отложенной выборки
                1.0,  # точность 1.0
            )  # ), конец GrowthEvent
            self.growth_log.append(ev)  # записать событие в журнал роста
            return ev  # вернуть событие

        def validate_candidate(program: Program) -> Optional[GrowthEvent]:  # проверить кандидата и при успехе принять его
            if deadline is not None and time.monotonic() >= deadline:  # время истекло
                return None  # кандидат не рассматривается
            try:  # ошибка вычисления — кандидат отвергается
                if not all(  # сито: программа должна пройти все точки пробы
                    self._same(program.eval(xs), yy) for xs, yy in zip(probe_x, probe_y)  # program(x) == y на парах пробы
                ):  # ):, конец all
                    self.stats.evals += len(probe_x)  # учесть вычисления пробы
                    return None  # не прошёл сито
                self.stats.sieve_survivors += 1  # прошёл сито: число выживших + 1
                if not self._fits(program, train_x, train_y):  # не совпадает на обучающей выборке
                    return None  # отвергнуть
                if val_x and not self._fits(program, val_x, val_y):  # есть отложенные точки и на них не совпадает
                    return None  # отвергнуть
                # окончательная проверка на всей области целыми числами Python
                if any(not self._same(program.eval(xs), all_y[xs]) for xs in domain):  # есть точка области, где program(x) ≠ y
                    return None  # отвергнуть
                if program.cost > 32:  # стоимость программы выше предела 32
                    return None  # отвергнуть
                return commit(program, "exact-validation+full-finite-support", None)  # принять: точная проверка + вся конечная область
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                return None  # кандидат отвергнут

        # переиспользование: платится за ссылку на библиотеку, а не за произвольную скидку
        known_items = list(sorted(self.experts.items()))  # известные эксперты, отсортированные по имени
        for index, (name, ex) in enumerate(known_items):  # по экспертам с номером index
            if ex.program.arity != spec.arity:  # другая арность программы
                continue  # следующий эксперт
            # переиспользование без синонимов допустимо, только если сертификат эксперта
            # уже покрывает всю новую область; иначе та же формула может
            # получить более широкий сертифицированный диапазон
            if not all(ex.supports(xs) for xs in domain):  # эксперт поддерживает не все входы области
                continue  # следующий эксперт
            if not self._fits(ex.program, train_x, train_y):  # не совпадает на обучающей выборке
                continue  # следующий эксперт
            if not self._fits(ex.program, val_x, val_y):  # не совпадает на отложенной выборке
                continue  # следующий эксперт
            if any(not self._same(ex.program.eval(xs), all_y[xs]) for xs in domain):  # не совпадает на всей области
                continue  # следующий эксперт
            ref_bits = _TOKEN_BITS + _gamma_bits(index + 1)  # стоимость ссылки: биты токена + γ-код номера index + 1
            ev = GrowthEvent(  # событие REUSE
                self.version,  # версия
                spec.name,  # имя задачи
                "REUSE",  # тип события REUSE
                f"verified-expert-coverage:{name}",  # причина: покрытие проверенным экспертом name
                ex.program.pretty(),  # текст программы эксперта
                ref_bits,  # стоимость ссылки в битах
                len(train_x),  # размер обучающей выборки
                len(val_x),  # размер отложенной выборки
                1.0,  # точность 1.0
            )  # ), конец GrowthEvent
            self.growth_log.append(ev)  # записать событие в журнал
            return ev  # вернуть событие

        # малая проба — только сито отвержения, а не ключ семантической эквивалентности
        probe_n = min(SIEVE_SIZE, len(train_x))  # размер пробы: min(SIEVE_SIZE, число обучающих точек)
        probe_idx = rng.choice(len(train_x), size=probe_n, replace=False)  # номера точек пробы без повторов
        probe_x = [train_x[int(i)] for i in probe_idx]  # входы пробы
        probe_y = [train_y[int(i)] for i in probe_idx]  # метки пробы

        def make_train_arrays():  # обучающие данные как массивы int64 либо (None, None)
            lo64, hi64 = -(1 << 63), (1 << 63) - 1  # границы int64: lo64 = −2^63, hi64 = 2^63 − 1
            flat = [int(v) for row in train_x for v in row] + [int(v) for v in train_y]  # все значения входов и меток одним списком
            if any(v < lo64 or v > hi64 for v in flat):  # есть значение вне int64
                return None, None  # массивы не строятся
            return (  # return (, вернуть пару массивов
                np.asarray(train_x, dtype=np.int64),  # матрица входов int64
                np.asarray(train_y, dtype=np.int64),  # вектор меток int64
            )  # ), конец пары

        train_X, train_Yv = make_train_arrays()  # массивы для векторной проверки

        # сначала быстрые точные шаблоны: базовое семейство кандидатов покрывает частые
        # арифметические и булевы примитивы почти без затрат на поиск
        if spec.family == "BoolVenn":  # класс BoolVenn
            direct_candidates: Sequence[Program] = [  # кандидаты — программы от 2 переменных
                Program(2, venn_mask=m) for m in range(16)  # программа с маской Венна m, m = 0..15
            ]  # ], конец списка
        else:  # иначе
            direct_candidates = sorted(  # кандидаты базового семейства, отсортированные
                candidate_programs(spec.arity),  # по арности задачи
                key=lambda p: (_program_description_bits(p), p.pretty()),  # ключ сортировки: (длина описания в битах, текст программы)
            )  # ), конец sorted
        # подсказка композиции родителей — предложение, а не истина: она принимается
        # только после проверки на обучении, отложенных точках и всей области.
        # если заданная подсказка не прошла — отказ, а не неограниченный поиск другой функции
  # launching an unbounded blind search for a different function., подсказка задана
        if candidate_hint is not None:  # проверить подсказку
            hinted = validate_candidate(candidate_hint)  # подсказка принята
            if hinted is not None:  # причина события — проверенная подсказка композиции
                hinted.reason = "verified-composition-hint+full-finite-support"  # журнал роста не пуст
                if self.growth_log:  # поправить причину последней записи журнала
                    self.growth_log[-1].reason = hinted.reason  # вернуть событие
                return hinted  # отказ: подсказка не сертифицирована
            return self._hold(  # задача, код отказа, размеры выборок
                spec, "COMPOSITION_HINT_NOT_CERTIFIED", len(train_x), len(val_x)  # ), конец вызова
            )
  # по кандидатам-шаблонам
        for program in direct_candidates:  # проверить кандидата
            ev = validate_candidate(program)  # кандидат принят
            if ev is not None:  # вернуть событие
                return ev  # время истекло
            if deadline is not None and time.monotonic() >= deadline:  # отказ: бюджет времени исчерпан
                return self._hold(  # задача, код отказа, размеры выборок
                    spec, "TIME_BUDGET_EXHAUSTED", len(train_x), len(val_x)  # ), конец вызова
                )

        # индукция многочлена в замкнутой форме идёт до общего поиска по грамматике;
        # унарная интерполяция сохраняет компактную запись по Горнеру
        if spec.arity == 1 and not spec.family.startswith("Bool"):  # один аргумент и не булев класс задач
            coeffs = _exact_poly_fit([xs[0] for xs in train_x], train_y, max_degree=5)  # точная интерполяция многочлена степени ≤ 5 по обучающим точкам
            if coeffs is not None and val_x:  # коэффициенты найдены и есть отложенные точки
                if all(  # если на отложенных точках
                    _eval_poly(coeffs, xs[0]) == yy for xs, yy in zip(val_x, val_y)  # p(x) == y на отложенных точках
                ) and all(_eval_poly(coeffs, xs[0]) == all_y[xs] for xs in domain):  # и на всей области
                    program = Program(1, _poly_expr(coeffs))  # программа-многочлен от одного аргумента
                    if program.cost <= 32:  # стоимость в пределах 32
                        return commit(  # принять программу
                            program, "exact-rational-polynomial+full-finite-support"  # причина: точный рациональный многочлен + вся конечная область
                        )  # ), конец вызова
        # точная многомерная интерполяция распространяет индукцию на 2–3 переменные:
        # точная многомерная интерполяция распространяет индукцию на 2–3 переменные:
        # решается переопределённая рациональная система; кандидат годится только при
        # единственном целом векторе коэффициентов и прохождении проверок на отложенных
        # точках и на всей конечной области
        elif 2 <= spec.arity <= 3 and not spec.family.startswith("Bool"):  # от 2 до 3 аргументов и не булев класс задач
            terms = _exact_multivariate_poly_fit(  # точная многомерная интерполяция
                train_x, train_y, arity=spec.arity, max_degree=4  # обучающие данные, метки, арность, степень ≤ 4
            )  # ), конец вызова
            if terms is not None and val_x:  # одночлены найдены и есть отложенные точки
                program = Program(spec.arity, _multivariate_poly_expr(terms))  # программа-многочлен от arity переменных
                if (  # если
                    all(program.eval(xs) == yy for xs, yy in zip(val_x, val_y))  # совпадает на отложенных точках
                    and all(program.eval(xs) == all_y[xs] for xs in domain)  # и на всей области
                    and program.cost <= 32  # и стоимость в пределах 32
                ):  # ):, конец условия
                    return commit(  # принять программу
                        program,  # программа
                        "exact-multivariate-rational-polynomial+full-finite-support",  # причина: точный многомерный рациональный многочлен + вся конечная область
                    )  # ), конец вызова

        # ограниченный поиск по грамматике: векторный фильтр отвержения применяется
        # только к доказанно безопасным случаям int64; прочие идут через скалярную проверку
        if deadline is not None and time.monotonic() >= deadline:  # время истекло
            return self._hold(spec, "TIME_BUDGET_EXHAUSTED", len(train_x), len(val_x))  # отказ: бюджет времени исчерпан
        reuse_binops = [  # операции библиотеки той же арности для переиспользования
            (nm, ex.program)  # пара (имя эксперта, его программа)
            for nm, ex in self.experts.items()  # по экспертам
            if ex.program.arity == spec.arity  # арность программы совпадает с арностью задачи
        ]  # ], конец списка
        candidates = (  # генератор кандидатов
            Program(spec.arity, e)  # программа из выражения e
            for e in iter_candidates_mdl(  # перебор выражений, упорядоченный по длине описания
                spec.arity,  # арность
                self.node_budget,  # бюджет узлов
                probe_x,  # входы пробы
                probe_y,  # метки пробы
                reuse_binops,  # операции библиотеки
                self.pop_budget,  # бюджет популяции
                deadline,  # предельное время
            )  # ), конец вызова
        )  # ), конец генератора
        chunk: list[Program] = []  # буфер кандидатов для пакетной проверки

        def flush(buf: list[Program]) -> Optional[GrowthEvent]:  # проверить пакет кандидатов; при успехе вернуть событие
            if not buf:  # пакет пуст
                return None  # нечего проверять
            verdicts = batch_fits_safe(buf, train_X, train_Yv, device=self.device)  # вердикты True/False/None по пакету
            for program, ok in zip(buf, verdicts):  # по парам (кандидат, вердикт)
                self.stats.candidates_generated += 1  # сгенерировано кандидатов + 1
                # пропускать скалярную проверку можно только по False из защищённого
                # векторного подмножества; None всегда идёт на скалярную проверку
                if ok is False:  # кандидат векторно отвергнут
                    self.stats.evals += len(train_x)  # учесть вычисления обучающей выборки
                    continue  # следующий кандидат
                ev = validate_candidate(program)  # скалярная проверка кандидата
                if ev is not None:  # кандидат принят
                    return ev  # вернуть событие
            return None  # в пакете нет принятых

        for program in candidates:  # по кандидатам
            chunk.append(program)  # добавить кандидата в буфер
            if len(chunk) >= VALIDATION_CHUNK:  # буфер достиг размера пакета
                ev = flush(chunk)  # проверить пакет
                if ev is not None:  # кандидат принят
                    return ev  # вернуть событие
                chunk = []  # очистить буфер
            if deadline is not None and time.monotonic() >= deadline:  # время истекло
                return self._hold(  # отказ: бюджет времени исчерпан
                    spec, "TIME_BUDGET_EXHAUSTED", len(train_x), len(val_x)  # задача, код отказа, размеры выборок
                )  # ), конец вызова
        ev = flush(chunk)  # проверить остаток буфера
        if ev is not None:  # кандидат принят
            return ev  # вернуть событие
        return self._hold(spec, "NO_VALIDATED_PROGRAM", len(train_x), len(val_x))  # отказ: проверенной программы нет

    def save(self, path: str | Path) -> None:  # сохранить состояние процессора в JSON-файл
        data = {  # словарь состояния
            "format": "HAGI-GrowingMathProcessor",  # формат файла
            "version": STATE_FORMAT,  # версия формата
            "model_version": self.version,  # версия модели
            "lookup_limit": self.lookup_limit,  # предел таблицы поиска
            "mdl_lambda": self.mdl_lambda,  # вес MDL-штрафа
            "node_budget": self.node_budget,  # бюджет узлов
            "pop_budget": self.pop_budget,  # бюджет популяции
            "device": self.device,  # устройство
            "total_commits": self.total_commits,  # число принятых
            "total_holds": self.total_holds,  # число отказов
            "experts": {k: v.to_dict() for k, v in sorted(self.experts.items())},  # эксперты по имени, отсортированные
            "growth_log": [vars(x) for x in self.growth_log],  # журнал роста как словари атрибутов
            "theorems": [vars(x) for x in self.theorems],  # теоремы как словари атрибутов
            "synthesis_stats": {  # статистика синтеза
                "candidates_generated": self.stats.candidates_generated,  # сгенерировано кандидатов
                "evals": self.stats.evals,  # число вычислений
                "sieve_survivors": self.stats.sieve_survivors,  # прошло сито
            },  # },, конец вложенного словаря
        }  # }, конец словаря состояния
        target = Path(path)  # целевой путь
        target.parent.mkdir(parents=True, exist_ok=True)  # создать родительский каталог, если его нет
        tmp = target.with_name(target.name + ".tmp")  # временный файл рядом с целевым
        with tmp.open("w", encoding="utf-8") as handle:  # открыть временный файл на запись в UTF-8
            json.dump(data, handle, indent=2, sort_keys=True)  # записать JSON с отступом 2 и сортировкой ключей
            handle.flush()  # сбросить буфер записи
            os.fsync(handle.fileno())  # синхронизировать файл с диском
        tmp.replace(target)  # атомарно заменить целевой файл временным

    @classmethod  # @classmethod, метод класса
    def load(cls, path: str | Path) -> "OptimalMathProcessor":  # загрузить состояние процессора из JSON-файла
        data = json.loads(Path(path).read_text(encoding="utf-8"))  # прочитать и разобрать JSON
        if data.get("format") != "HAGI-GrowingMathProcessor":  # формат не совпадает
            raise ValueError("not a HAGI Growing Math Processor state file")  # raise ValueError(, ошибка: это не файл состояния процессора
        model = cls(  # создать процессор с параметрами из файла
            int(data.get("lookup_limit", 16)),  # lookup_limit из файла или 16
            float(data.get("mdl_lambda", 0.001)),  # mdl_lambda из файла или 0.001
            int(data.get("node_budget", DEFAULT_NODE_BUDGET)),  # node_budget из файла или значение по умолчанию
            int(data.get("pop_budget", DEFAULT_POP_BUDGET)),  # pop_budget из файла или значение по умолчанию
            str(data.get("device", "cpu")),  # device из файла или "cpu"
        )  # ), конец вызова
        model.version = int(data.get("model_version", 0))  # версия модели из файла или 0
        model.total_commits = int(data.get("total_commits", 0))  # число принятых из файла или 0
        model.total_holds = int(data.get("total_holds", 0))  # число отказов из файла или 0
        model.experts = {  # эксперты из файла
            k: Expert.from_dict(v) for k, v in data.get("experts", {}).items()  # по имени — эксперт, восстановленный из словаря
        }  # }, конец словаря экспертов
        if data.get("version") != STATE_FORMAT and data.get("version") not in LEGACY_STATE_LABELS:  # версия формата отличается от текущей
            for expert in model.experts.values():  # по всем экспертам
                expert.code_length = _program_description_bits(expert.program)  # пересчитать длину кода программы
        model.growth_log = [GrowthEvent(**x) for x in data.get("growth_log", [])]  # журнал роста из файла
        model.theorems = [Theorem(**x) for x in data.get("theorems", [])]  # теоремы из файла
        stats = data.get("synthesis_stats", {})  # статистика синтеза из файла или пустой словарь
        model.stats.candidates_generated = int(stats.get("candidates_generated", 0))  # сгенерировано кандидатов из файла или 0
        model.stats.evals = int(stats.get("evals", 0))  # число вычислений из файла или 0
        model.stats.sieve_survivors = int(stats.get("sieve_survivors", 0))  # прошло сито из файла или 0
        return model  # вернуть модель

    def _fits(  # проверка: program(x) == y на всех парах
        self, program: Program, xs_list: Sequence[Sequence[int]], ys: Sequence[int]  # program — программа, xs_list — входы, ys — метки
    ) -> bool:  # ) -> bool:, возвращает True, если совпадает на всех парах
        self.stats.evals += len(xs_list)  # учесть вычисления
        try:  # try:, ошибка вычисления — программа не подходит
            return all(self._same(program.eval(xs), yy) for xs, yy in zip(xs_list, ys))  # True, если program(x) == y на всех парах
        except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
            return False  # не подходит


# ------------------- сертифицированный оптимизатор с защитой -------------------


def _fiber_forward(params, x, hidden_scale=1.0):  # прямой ход tanh-слоя: y = tanh(x·W1 + b1)·W2 + b2
    w1, b1, w2, b2 = params  # разбор параметров: W1, b1, W2, b2
    return np.tanh(x @ w1 + b1) @ w2 + b2  # выход слоя: y = tanh(x·W1 + b1)·W2 + b2


def fit_fiber_guarded(  # подобрать веса tanh-слоя полным пакетом с защитой от роста ошибки
    inputs, targets, val_inputs, val_targets, seed, hidden: int = 24, epochs: int = 300  # inputs, targets — обучающие данные, val_inputs, val_targets — отложенные, seed — зерно, hidden — ширина слоя, epochs — число эпох
) -> "NeuralFiber":  # ) -> "NeuralFiber":, возвращает подобранный слой
    """Подбор весов необязательного tanh-слоя: полный пакет, средний квадрат ошибки (MSE).

    Шаг принимается, только если измеренная ошибка на обучении строго убывает. Это защита
    от регрессии на уровне реализации, а не доказательство шага обучения, не гарантия
    на отложенных данных и не теорема сходимости; лучшие веса выбираются по ошибке
    на отложенных данных.
    """
    x = np.asarray(inputs, dtype=np.float64).reshape(-1, len(inputs[0]))  # матрица входов (N, d) в float64
    y = np.asarray(targets, dtype=np.float64).reshape(-1, 1)  # вектор меток (N, 1)
    xv = np.asarray(val_inputs, dtype=np.float64).reshape(-1, x.shape[1])  # матрица отложенных входов
    yv = np.asarray(val_targets, dtype=np.float64).reshape(-1, 1)  # вектор отложенных меток
    sx = max(1.0, float(np.max(np.abs(x))))  # масштаб входов sx = max(1, max|x|)
    sy = max(1.0, float(np.max(np.abs(y))))  # масштаб меток sy = max(1, max|y|)
    x, y, xv, yv = x / sx, y / sy, xv / sx, yv / sy  # нормировка: x/sx, y/sy, xv/sx, yv/sy
    rng = np.random.default_rng(seed)  # генератор случайных чисел с зерном seed
    params = [  # начальные параметры
        rng.normal(0.0, 0.32, (x.shape[1], hidden)),  # W1 ~ N(0, 0.32) формы (d, hidden)
        np.zeros((1, hidden)),  # b1 = 0 формы (1, hidden)
        rng.normal(0.0, 0.12, (hidden, 1)),  # W2 ~ N(0, 0.12) формы (hidden, 1)
        np.zeros((1, 1)),  # b2 = 0 формы (1, 1)
    ]  # ], конец списка параметров

    def loss(p):  # ошибка MSE на обучении для параметров p
        err = _fiber_forward(p, x) - y  # невязка err = f(x) − y
        return float(np.mean(err * err))  # L = среднее err²

    cur = loss(params)  # текущая ошибка
    best = [q.copy() for q in params]  # копии лучших параметров
    best_val = float("inf")  # лучшая отложенная ошибка — бесконечность
    total_drop = 0.0  # суммарное убывание ошибки
    accepted = 0  # число принятых шагов
    held = 0  # число удержанных шагов
    for _ in range(epochs):  # по эпохам
        pred = _fiber_forward(params, x)  # предсказание на обучении
        err = pred - y  # невязка
        # направление шага — антиградиент средних потерь
        h_act = np.tanh(x @ params[0] + params[1])  # скрытая активация h = tanh(x·W1 + b1)
        dh = (2.0 / len(x) * err) @ params[2].T  # dh = (2/N)·err @ W2^T — градиент по скрытой активации
        dh = dh * (1.0 - h_act**2)  # dh = dh·(1 − h²) — производная tanh
        dp = 2.0 / len(x) * err  # dp = (2/N)·err — градиент по выходу
        grads = [  # градиенты параметров
            x.T @ dh,  # градиент по W1: dL/dW1 = x^T @ dh
            np.sum(dh, axis=0, keepdims=True),  # градиент по b1: dL/db1 = Σ dh
            h_act.T @ dp,  # градиент по W2: dL/dW2 = h^T @ dp
            np.sum(dp, axis=0, keepdims=True),  # градиент по b2: dL/db2 = Σ dp
        ]  # ], конец списка градиентов
        gnorm2 = sum(float(np.sum(g * g)) for g in grads)  # квадрат нормы градиента: Σ‖g‖²
        if gnorm2 <= 0:  # нулевой градиент
            held += 1  # шаг удержан
            continue  # следующая эпоха
        # кандидаты шага: сетка с обратным отслеживанием; принимается первый,
        # строго уменьшающий измеренную ошибку
        improved = False  # улучшения нет
        for step in (1.0, 0.5, 0.25, 0.1, 0.03):  # по размерам шага
            cand = [p - step * g for p, g in zip(params, grads)]  # кандидат параметров p − step·g
            cand_loss = loss(cand)  # ошибка кандидата
            if cand_loss < cur:  # строго меньше текущей
                drop = cur - cand_loss  # убывание drop = cur − cand_loss
                params = cand  # принять параметры
                cur = cand_loss  # текущая ошибка обновлена
                total_drop += drop  # суммарное убывание += drop
                accepted += 1  # принятых шагов + 1
                improved = True  # улучшение есть
                break  # выйти из перебора шагов
        if not improved:  # ни один шаг не принят
            held += 1  # удержанных шагов + 1
            # шаг удержан: параметры не меняются, убывание равно нулю
        hv = _fiber_forward(params, xv)  # предсказание на отложенных данных
        vl = float(np.mean((hv - yv) ** 2))  # отложенная ошибка vl = среднее (hv − yv)²
        if np.isfinite(vl) and vl < best_val:  # ошибка конечна и лучше лучшей
            best_val = vl  # лучшая отложенная ошибка
            best = [q.copy() for q in params]  # запомнить лучшие параметры
    w1, b1, w2, b2 = best  # лучшие параметры
    h = np.tanh(x @ w1 + b1)  # скрытая активация на обучении
    tr_loss = float(np.mean((h @ w2 + b2 - y) ** 2)) * sy * sy  # ошибка обучения в исходном масштабе: MSE·sy²
    hv = np.tanh(xv @ w1 + b1)  # скрытая активация на отложенных
    predv = (hv @ w2 + b2).reshape(-1) * sy  # предсказание в исходном масштабе: (h·W2 + b2)·sy
    val_loss = float(np.mean((predv - np.asarray(val_targets, dtype=float)) ** 2))  # отложенная ошибка в исходном масштабе
    exact = float(  # доля точных попаданий
        np.mean(  # среднее по отложенным точкам
            np.rint(predv).astype(np.int64) == np.asarray(val_targets, dtype=np.int64)  # округлённое предсказание равно метке
        )  # ), конец mean
    )  # ), конец float
    fiber = NeuralFiber(  # создать слой
        len(inputs[0]), sx, sy, w1, b1, w2, b2, tr_loss, val_loss, exact  # число входов, масштабы, параметры, ошибки, точность
    )  # ), конец NeuralFiber
    fiber.guard_certificate = {  # сертификат защиты
        "accepted_steps": accepted,  # число принятых шагов
        "hold_steps": held,  # число удержанных шагов
        "total_certified_drop": total_drop,  # суммарное убывание
    }  # }, конец сертификата
    return fiber  # вернуть слой


# --------------- сертифицированная тернарная компрессия ------------------------

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'OptimalMathProcessor', 'fit_fiber_guarded',  # имя интерфейса
]  # конец перечня
