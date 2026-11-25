"""HAGI — публичный интерфейс пакета: точка входа `import hagi`.

Назначение файла: собрать в пространстве имён `hagi` ЯВНЫЙ перечень
того, что считается внешним интерфейсом символьного ядра. Сам файл ничего
не вычисляет. Перечень задан списком `__all__` ниже; всё, чего в нём нет,
нужно импортировать из конкретного модуля.

HAGI — самообучающаяся модель, работающая с данными как с потоком байтов.
Ядро — символьный процессор: программа представляется выражением (дерево
операций над целыми числами), обучение — поиском программы, точно
воспроизводящей пары «вход → выход». Нейронная часть — языковая модель по
байтам p(b | контекст) — живёт в `hagi.models.lm` / `hagi.models.hnet_lm`, единая модель —
в `hagi.selfdev.unified`; сюда они не входят и импортируются явно.

Слои пакета (направление зависимостей — только вниз):
  интерфейс     hagi (этот файл), hagi.cli (импортируется явно), hagi.base.config
  мета-цикл     loop, cycle, genesis, selfmodel, fabric, law_critique, pipeline
  память/среды  memory_store/*, env_lut, env_temporal
  модели        lm, hnet_lm, unified, trunk_ops, quantized, embeddings, archiver
  символьное    core (+ legacy), representations, invention*, diagnosis,
                skill_engine, certificates, rewrite, school
  инфраструктура util, io, gpu

Группы интерфейса (что ниже в `__all__`):
  выражения и задачи   Expr, Program, CallDagProgram, TaskSpec, StoredExpert, VAR_NAMES
  таблицы операций     OPS, UNARY_OPS, BINARY_OPS, COMMUTATIVE_OPS, CONSTS,
                       DYADIC_LEVELS, REIFIED_OPS, TOKEN_BITS, gamma_bits, zigzag
  процессоры           GrowingMathProcessor(оптимальный), GrowingHagiProcessor(единый), базовый
  нейро-адаптеры ядра  FactorizedAdapter, FactorizedSharedCore, NeuralFiber,
                       HedgeRouter, newton_schulz_orthogonalize, batch_logit_safe_step_ce
  рост и проверка      grow_deep, grow_from_suite, self_test*, run_experiment, main
  синтез и аппроксимация synthesize_batched_mdl, exact_piecewise_poly_fit,
                       make_task_suite, propose_noncircular_task, solve_coloring
  служебные измерения  stutter_norm, boundary_chunks, boundary_heads,
                       ram_available_bytes, venn_*, anf_*
"""
from hagi.core import DYADIC_LEVELS, OPS, UNARY_OPS, BINARY_OPS, COMMUTATIVE_OPS, CONSTS, REIFIED_OPS, TOKEN_BITS, VAR_NAMES, Expr, Program, CallDagProgram, TaskSpec, StoredExpert, GrowingMathProcessor, OptimalMathProcessor, GrowingHagiProcessor, GrowthProcessorBase, FactorizedAdapter, FactorizedSharedCore, NeuralFiber, HedgeRouter, newton_schulz_orthogonalize, batch_logit_safe_step_ce, gamma_bits, zigzag, grow_deep, grow_from_suite, self_test, self_test_optimal, self_test_unified, self_test_core, run_experiment, main, synthesize_batched_mdl, exact_piecewise_poly_fit, make_task_suite, propose_noncircular_task, solve_coloring, stutter_norm, boundary_chunks, boundary_heads, ram_available_bytes, venn_atoms, venn_head_eval, pack_venn_16_flat, unpack_venn_16_flat, anf_transform, anf_support_vars  # явные имена ядра — только то, что входит в интерфейс
from hagi.data import genesis  # noqa: F401  — генезис доступен как атрибут: hagi.genesis
from hagi import core  # noqa: F401  — ядро доступно как атрибут: hagi.core.Expr
from hagi.base import config, util  # noqa: F401  — фундамент доступен как атрибуты: hagi.config.device, hagi.util.<имя>

__all__ = [  # явный интерфейс пакета: только эти имена выносит `from hagi import *`
    "Expr", "Program", "CallDagProgram", "TaskSpec", "StoredExpert", "VAR_NAMES",  # выражения и задачи
    "OPS", "UNARY_OPS", "BINARY_OPS", "COMMUTATIVE_OPS", "CONSTS", "DYADIC_LEVELS",  # таблицы операций
    "REIFIED_OPS", "TOKEN_BITS", "gamma_bits", "zigzag",  # реестр операций и кодирование длин
    "GrowingMathProcessor", "OptimalMathProcessor", "GrowingHagiProcessor", "GrowthProcessorBase",  # процессоры
    "FactorizedAdapter", "FactorizedSharedCore", "NeuralFiber", "HedgeRouter",  # нейро-адаптеры ядра
    "newton_schulz_orthogonalize", "batch_logit_safe_step_ce",  # численные процедуры
    "grow_deep", "grow_from_suite", "self_test", "self_test_optimal", "self_test_unified", "self_test_core", "run_experiment", "main",  # рост и проверка
    "synthesize_batched_mdl", "exact_piecewise_poly_fit", "make_task_suite", "propose_noncircular_task", "solve_coloring",  # синтез и аппроксимация
    "stutter_norm", "boundary_chunks", "boundary_heads", "ram_available_bytes",  # измерения
    "venn_atoms", "venn_head_eval", "pack_venn_16_flat", "unpack_venn_16_flat",  # диаграммы Венна
    "anf_transform", "anf_support_vars",  # алгебраическая нормальная форма
        "core", "config", "util", "genesis",  # подпакеты (cli импортируется явно: python -m hagi.cli)
]
