"""Публичный интерфейс символьного ядра HAGI: точка входа `import hagi.core`.

Ядро собрано из групп по смыслу (каталоги внутри hagi/core):
  language   язык выражений и код: expressions, expr_dsl, call_dag, code_length
  tasks      задачи и кандидаты: tasks, stored_tasks, candidates
  synthesis  поиск программ: synthesis, venv_anf, venn, prefilter, best_first, fast_mdl, induction
  neural     нейросетевая часть: neural, segmentation, fiber, certified_ce, energy, chunking
  processor  эксперты и процессоры: processor, growth, stored_growth, experts, optimal_processor,
             unified_processor, benchmark, self_development
  memory     журнал, аудит, алиасы, самопроверка, командная строка: audit, ledger, aliases, selfcheck,
             experiment, selftest, cli

Порядок звёздочных импортов задаёт, чьё имя действует при совпадении: импортированные
позже модули перекрывают прежние. `__all__` перечисляет то, что выносит
`from hagi.core import *`. Номеров версий в интерфейсе нет.
"""
from hagi.base.gpu import GpuGpmMixin  # noqa: F401  # примесь диспетчеризации на графический ускоритель: база ContinualGPMMemory
from hagi.core.language.expr_dsl import *  # noqa: F401,F403  — имена модуля expr_dsl: Expr, OPS, Program
from hagi.core.tasks.stored_tasks import *  # noqa: F401,F403  — имена модуля tasks: BOOL_NAMES, TaskSpec, make_task_suite
from hagi.core.tasks.candidates import *  # noqa: F401,F403  — имена модуля candidates: candidate_programs
from hagi.core.neural.fiber import *  # noqa: F401,F403  — имена модуля fiber: NeuralFiber
from hagi.core.processor.experts import *  # noqa: F401,F403  — имена модуля experts: Expert, GrowingMathProcessor, GrowthEvent, SafeExpression
from hagi.core.memory.experiment import *  # noqa: F401,F403  — имена модуля experiment: run_experiment
from hagi.core.memory.selfcheck import *  # noqa: F401,F403  — имена модуля selfcheck: main, query_model, self_test
from hagi.core.synthesis.prefilter import *  # noqa: F401,F403  — имена модуля prefilter: DEFAULT_NODE_BUDGET, DEFAULT_POP_BUDGET, MAX_REUSE_EXPANSIONS, ProcessPoolExecutor, SIEVE_SIZE, VALIDATION_CHUNK
from hagi.core.language.code_length import *  # noqa: F401,F403  — имена модуля code_length: batch_fits_safe
from hagi.core.synthesis.best_first import *  # noqa: F401,F403  — имена модуля best_first: BINARY_OPS, CONSTS, SynthesisStats, TERNARY_PRIMS, UNARY_OPS, iter_candidates_mdl
from hagi.core.processor.optimal_processor import *  # noqa: F401,F403  — имена модуля optimal_processor: OptimalMathProcessor, LegacyProcessor, fit_fiber_guarded
from hagi.core.synthesis.induction import *  # noqa: F401,F403  — индукция: solve_coloring, Theorem, TheoremMiner и точная подгонка многочленов
from hagi.core.processor.self_development import *  # noqa: F401,F403  — имена модуля self_development: ResourceGovernor, grow_forever, verify_growth_lineage
from hagi.core.processor.benchmark import *  # noqa: F401,F403  — имена модуля benchmark: run_optimal_experiment, self_test_optimal, OPTIMAL_LAYER_VERSION
from hagi.core.language.call_dag import *  # noqa: F401,F403  — имена модуля call_dag: DYADIC_LEVELS, CallDagExpr, MAX_RANGE_EVAL, CallDagProgram, TOKEN_BITS, VAR_NAMES, gamma_bits, zigzag
from hagi.core.synthesis.venn import *  # noqa: F401,F403  — имена модуля venn: anf_to_call_dag, anf_transform, pack_venn_16_flat, unpack_venn_16_flat (venn_atoms, venn_head_eval — в venv_anf)
from hagi.core.neural.certified_ce import *  # noqa: F401,F403  — имена модуля certified_ce: FactorizedAdapter, FactorizedSharedCore, batch_logit_safe_step_ce, newton_schulz_orthogonalize, quantize_5block_single, quantize_bitalloc_ef21
from hagi.core.neural.energy import *  # noqa: F401,F403  — имена модуля energy: CRAMEngine
from hagi.core.neural.chunking import *  # noqa: F401,F403  — имена модуля chunking: GDNRAM, HedgeRouter, boundary_chunks, boundary_heads, stutter_norm
from hagi.core.memory.ledger import *  # noqa: F401,F403  — имена модуля ledger: AuditLedger, ram_available_bytes
from hagi.core.synthesis.fast_mdl import *  # noqa: F401,F403  — имена модуля fast_mdl: REIFIED_OPERATIONS, bounded_fold_hypotheses, exact_piecewise_poly_fit, synthesize_batched_mdl_stored
from hagi.core.processor.unified_processor import *  # noqa: F401,F403  — имена модуля unified_processor: StoredExpert, GrowthProcessorBase, StoredTheoremMiner, verify_call_dag
from hagi.core.processor.stored_growth import *  # noqa: F401,F403  — имена модуля growth: CORE_LAYER_VERSION, grow_from_suite, propose_noncircular_task, self_test_unified
from hagi.core.memory.aliases import *  # noqa: F401,F403  — имена модуля aliases: COMMUTATIVE_OPS
from hagi.core.tasks.tasks import *  # noqa: F401,F403  — часть «tasks»: её публичные имена становятся именами подпакета
from hagi.core.language.expressions import *  # noqa: F401,F403  — часть «expressions»: её публичные имена становятся именами подпакета
from hagi.core.synthesis.venv_anf import *  # noqa: F401,F403  — часть «venv_anf»: её публичные имена становятся именами подпакета
from hagi.core.neural.neural import *  # noqa: F401,F403  — часть «neural»: её публичные имена становятся именами подпакета
from hagi.core.neural.segmentation import *  # noqa: F401,F403  — часть «segmentation»: её публичные имена становятся именами подпакета
from hagi.core.memory.audit import *  # noqa: F401,F403  — часть «audit»: её публичные имена становятся именами подпакета
from hagi.core.synthesis.synthesis import *  # noqa: F401,F403  — часть «synthesis»: её публичные имена становятся именами подпакета
from hagi.core.processor.processor import *  # noqa: F401,F403  — часть «processor»: её публичные имена становятся именами подпакета
from hagi.core.processor.growth import *  # noqa: F401,F403  — часть «growth»: её публичные имена становятся именами подпакета
from hagi.core.memory.selftest import *  # noqa: F401,F403  — часть «selftest»: её публичные имена становятся именами подпакета
from hagi.core.memory.cli import *  # noqa: F401,F403  — часть «cli»: её публичные имена становятся именами подпакета

# публичные имена, не попадающие через звёздочный импорт (определены в модулях, идущих раньше перекрывающих)
from hagi.core.memory.cli import _cli  # noqa: F401  # точка входа командной строки ядра
# приватные имена, через которые части ядра обращаются друг к другу как к атрибутам пакета
from hagi.core.memory.aliases import _PRIME_MOD  # noqa: F401  # имя доступно как атрибут hagi.core
from hagi.core.synthesis.prefilter import _bit_nand8  # noqa: F401  # имя доступно как атрибут hagi.core
from hagi.core.synthesis.prefilter import _bit_nor8  # noqa: F401  # имя доступно как атрибут hagi.core
from hagi.core.synthesis.prefilter import _bit_not8  # noqa: F401  # имя доступно как атрибут hagi.core
from hagi.core.synthesis.prefilter import _bit_xnor8  # noqa: F401  # имя доступно как атрибут hagi.core
from hagi.core.synthesis.induction import _exact_multivariate_poly_fit  # noqa: F401  # имя доступно как атрибут hagi.core
from hagi.core.synthesis.induction import _exact_poly_fit  # noqa: F401  # имя доступно как атрибут hagi.core
from hagi.core.synthesis.fast_mdl import _fit_poly_or_const  # noqa: F401  # имя доступно как атрибут hagi.core
from hagi.core.synthesis.induction import _monomial_exponents  # noqa: F401  # имя доступно как атрибут hagi.core
from hagi.core.synthesis.prefilter import _pow_int  # noqa: F401  # имя доступно как атрибут hagi.core
from hagi.core.synthesis.prefilter import _sign  # noqa: F401  # имя доступно как атрибут hagi.core
from hagi.core.processor.self_development import _ram_available_bytes  # noqa: F401  # имя доступно как атрибут hagi.core

__all__ = [  # явный интерфейс: только эти имена выносит `from hagi.core import *`
    'BOOL_NAMES', 'TaskSpec', 'make_task_suite', 'Expr', 'Program', 'REIFIED_OPS',  # имя интерфейса
    'candidate_programs', 'eval_expr_vec', 'anf_compose3', 'anf_eval', 'anf_support_vars', 'anf_to_expr',  # имя интерфейса
    'anf_transform', 'block_hadamard_rotate', 'block_hadamard_unrotate', 'pack_bitalloc_records', 'pack_vector_dyadic5', 'pack_venn_16_flat',  # имя интерфейса
    'quantize_5block_exact', 'sylvester_hadamard', 'unpack_bitalloc_records', 'unpack_vector_dyadic5', 'unpack_venn_16_flat', 'venn_atoms',  # имя интерфейса
    'venn_head_eval', 'CRAMEngine', 'ContinualGPMMemory', 'FactorizedAdapter', 'FactorizedSharedCore', 'batch_logit_safe_step_ce',  # имя интерфейса
    'bf16_nearest_float', 'newton_schulz_orthogonalize', 'quantize_bitalloc_exact', 'GDNRAM', 'boundary_chunks', 'boundary_heads',  # имя интерфейса
    'hnet_opt_ratio', 'stutter_norm', 'AuditLedger', 'AuditStream', 'CertStepGen', 'HedgeRouter',  # имя интерфейса
    'ram_available_bytes', 'exact_piecewise_poly_fit', 'synthesize_batched_mdl', 'Expert', 'GrowingHagiProcessor', 'Theorem',  # имя интерфейса
    'TheoremMiner', 'grow_deep', 'propose_noncircular_task', 'self_test_core', 'DYADIC_LEVELS', 'OPS',  # имя интерфейса
    'UNARY_OPS', 'BINARY_OPS', 'COMMUTATIVE_OPS', 'CONSTS', 'TOKEN_BITS', 'VAR_NAMES',  # имя интерфейса
    'CallDagProgram', 'StoredExpert', 'GrowingMathProcessor', 'OptimalMathProcessor', 'GrowthProcessorBase', 'NeuralFiber',  # имя интерфейса
    'gamma_bits', 'zigzag', 'grow_from_suite', 'self_test', 'self_test_optimal', 'self_test_unified',  # имя интерфейса
    'run_experiment', 'main', 'solve_coloring',  # имя интерфейса
]  # конец перечня интерфейса
