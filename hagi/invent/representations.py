#!/usr/bin/env python3
"""Реестр представлений: слой подключения встроенных и изобретённых конструкций.

Что даёт файл:
- Протокол Representation: единый интерфейс для встроенных и изобретённых решателей
  (supports_task / probe_potential / synthesize).
- Протокол Artifact: интерфейс результата синтеза (eval / supports / bits /
  verify_contract / to_dict / emit_python) — то, что потребляют Expert, rewrite.py,
  mdl.py и execute.py.
- REPR_REGISTRY: идентификатор -> Representation. Выражение Expr получает
  передающую операцию "construct" через construct_eval: если операция не входит в
  лестницу if-elif в core.py, вычисление уходит сюда, в реестр.
- Проверка контракта: проход по всей области + запись и чтение через JSON +
  обязательный отказ (ABSTAIN) вне сертифицированной области.

Честное ограничение: артефакт конструкции точен только на своей сертифицированной
области; вне её вызов бросает ошибку и никогда не угадывает.

Вторая часть — переификация: превращение изобретённой операции в именованный
оператор реестра REIFIED_OPS (hagi.core) с проверкой области в момент вызова,
а также сохранение и восстановление этих операторов с контрольной меткой.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов

from dataclasses import dataclass  # декоратор класса данных
from typing import Any, Callable, Protocol, Sequence, runtime_checkable  # типы для аннотаций и проверяемые протоколы

# --------------------------------------------------------------------------
# Протоколы
# --------------------------------------------------------------------------

@runtime_checkable  # протокол можно проверять выражением isinstance
class Artifact(Protocol):
    """Результат синтеза представления: вычислимая и сериализуемая конструкция."""

    def eval(self, xs: Sequence[int],  # вычислить значение на входе xs
             env: dict | None = None) -> int: ...  # env — среда имён; возвращает целое значение
    def supports(self, xs: Sequence[int]) -> bool: ...  # True, если вход xs лежит в сертифицированной области
    def bits(self, expert_index: dict | None = None) -> int: ...  # длина описания в битах (expert_index — индекс экспертов для ссылок)
    def verify_contract(self, oracle_or_domain: Any) -> Any: ...  # проверка контракта по учителю или области
    def to_dict(self) -> dict[str, Any]: ...  # словарь без объектов, пригодный для JSON
    def emit_python(self, fn_name: str, env: dict | None = None) -> str: ...  # текст функции на Python с именем fn_name


@runtime_checkable  # протокол можно проверять выражением isinstance
class Representation(Protocol):
    """Представление: решатель, подключаемый через реестр."""

    id: str  # уникальный идентификатор
    origin: str            # происхождение: "builtin" (встроенное), "invented" (изобретённое) или "ood_recovered"
    meta_bits: int         # честная длина описания самой конструкции в битах

    def supports_task(self, profile: dict) -> bool: ...  # годится ли представление для задачи с профилем profile
    def probe_potential(self, obs: dict) -> float: ...  # оценка потенциальной пользы по наблюдениям obs
    def synthesize(self, obs: dict, budget: float,  # синтезировать артефакт по наблюдениям obs
                   ledger: Any) -> Artifact | None: ...  # в рамках бюджета budget и журнала ledger; None, если не вышло


# --------------------------------------------------------------------------
# Реестр
# --------------------------------------------------------------------------

REPR_REGISTRY: dict[str, Representation] = {}  # реестр представлений: идентификатор -> объект Representation

CONSTRUCT_OP = "construct"  # имя операции узла Expr, ссылающейся на конструкцию реестра


def register(repr_obj: Representation) -> None:
    """Добавить представление repr_obj в реестр; повтор идентификатора — ошибка."""
    if repr_obj.id in REPR_REGISTRY:  # идентификатор уже занят
        raise ValueError(f"duplicate representation id: {repr_obj.id}")  # ошибка: представление с таким идентификатором уже есть
    REPR_REGISTRY[repr_obj.id] = repr_obj  # записать представление по своему идентификатору


def get(repr_id: str) -> Representation:
    """Вернуть представление реестра по идентификатору repr_id (нет такого — KeyError)."""
    return REPR_REGISTRY[repr_id]  # выборка из словаря реестра


# --------------------------------------------------------------------------
# Передача операции construct выражения Expr (вызывается из core через construct_eval)
# --------------------------------------------------------------------------

def construct_eval(value: Any, xs: Sequence[int],  # значение узла, вход
                   env: dict | None = None) -> int:  # среда имён
    """Вычислить узел construct: value = {"repr_id": ..., "artifact": ...}.

    Вне сертифицированной области артефакта бросает ValueError."""
    if not isinstance(value, dict) or "artifact" not in value:  # value не словарь или в нём нет ключа "artifact"
        raise ValueError("construct node missing artifact payload")  # ошибка: у узла нет полезной нагрузки-артефакта
    art = value["artifact"]  # артефакт конструкции
    if not art.supports(xs):  # вход xs вне сертифицированной области артефакта
        raise ValueError(f"construct {value.get('repr_id')} outside certified support")  # ошибка: конструкция вне сертифицированной области
    return int(art.eval(xs, env))  # значение артефакта на xs в среде env, приведённое к целому


def construct_bits(value: Any,  # значение узла construct
                   expert_index: dict | None = None) -> int:  # индекс экспертов для ссылок
    """Число бит, которое узел construct добавляет к длине описания: биты его артефакта."""
    art = value.get("artifact")  # артефакт конструкции (None, если ключа нет)
    if art is None:  # артефакта нет
        raise ValueError("construct node missing artifact payload")  # ошибка: у узла нет полезной нагрузки-артефакта
    return int(art.bits(expert_index))  # длина описания артефакта в битах


def construct_pretty(value: Any) -> str:
    """Текстовое имя узла construct: construct(<идентификатор представления>)."""
    return f"construct({value.get('repr_id')})"  # строка «construct(id)»


def construct_to_dict(value: Any) -> Any:
    """Словарь узла construct для сериализации: идентификатор представления и словарь артефакта."""
    return {"repr_id": value.get("repr_id"),  # идентификатор представления (None, если ключа нет)
            "artifact": value["artifact"].to_dict()}  # артефакт в виде словаря


# --------------------------------------------------------------------------
# Проверка контракта (вся область + запись-чтение + отказ вне области)
# --------------------------------------------------------------------------

@dataclass  # класс данных: методы сравнения и инициализации генерируются автоматически
class ContractResult:
    """Результат проверки контракта артефакта."""

    ok: bool  # контракт выполнен
    reason: str = ""  # причина отказа (пустая строка, если всё в порядке)
    checked_pairs: int = 0  # число пар (вход, ответ), проверенных до остановки
    counterexample: tuple | None = None  # вход, на котором найдено нарушение (None, если нет)


def verify_artifact_contract(artifact: Artifact, teacher: Callable[..., int],
                             domain: Sequence[Sequence[int]]) -> ContractResult:  # область точек; возвращает результат проверки
    """Проверка артефакта по всей области: 100% совпадение с учителем и отказ вне области.

    Для каждой точки xs области: артефакт обязан её поддерживать, иначе это отказ
    (ABSTAIN) внутри сертифицированной области; значение артефакта должно равняться
    ответу учителя. Затем словарь артефакта записывается в JSON и читается обратно —
    результат обязан совпасть. Поиск движется по контрпримеру: первое же нарушение
    возвращается, а не проглатывается."""
    checked = 0  # число проверенных точек
    for xs in domain:  # по точкам области
        xs_t = tuple(xs)  # точка как кортеж xs_t
        want = int(teacher(*xs_t))  # ожидаемый ответ want = int(teacher(xs_t))
        try:  # вычисление артефакта может бросить ошибку
            if not artifact.supports(xs_t):  # артефакт не поддерживает точку области
                return ContractResult(False, "ABSTAIN inside certified domain",  # отказ: отказ внутри сертифицированной области,
                                      checked, xs_t)  # с числом проверенных и контрпримером xs_t
            got = int(artifact.eval(xs_t))  # ответ артефакта got = int(артефакт(xs_t))
        except ValueError as e:  # артефакт бросил ValueError
            return ContractResult(False, f"raised: {e}", checked, xs_t)  # отказ: артефакт бросил ошибку e; контрпример xs_t
        if got != want:  # ответ артефакта не совпал с ответом учителя
            return ContractResult(False, "mismatch", checked, xs_t)  # отказ: несовпадение; контрпример xs_t
        checked += 1  # проверено точек: checked + 1
    # Запись в JSON и чтение обратно
    try:  # сериализация может бросить любую ошибку
        d = artifact.to_dict()  # словарь артефакта
        import json  # работа с JSON (локальный импорт)
        rt = json.loads(json.dumps(d))  # rt = json.loads(json.dumps(d)) — прочитанная обратно копия
        if rt != d:  # копия отличается от исходного словаря
            return ContractResult(False, "json round-trip mismatch", checked)  # отказ: несовпадение после записи-чтения
    except Exception as e:  # to_dict или JSON бросили ошибку
        return ContractResult(False, f"to_dict failed: {e}", checked)  # отказ: ошибка e при получении словаря
    return ContractResult(True, "verified", checked)  # контракт выполнен: проверено checked точек


# --------------------------------------------------------------------------
# Встроенные представления (обёртки над существующей лестницей синтеза)
# --------------------------------------------------------------------------

@dataclass  # класс данных: методы сравнения и инициализации генерируются автоматически
class BuiltinRepr:
    """Встроенное представление: предикат семейства задач и функция синтеза.

    id — идентификатор; family_pred — проверка профиля задачи; synthesize_fn —
    синтез по словарю наблюдений; origin — по умолчанию "builtin"."""

    id: str  # идентификатор представления
    family_pred: Callable[[dict], bool]  # предикат профиля задачи
    synthesize_fn: Callable[..., Any]  # функция синтеза
    origin: str = "builtin"  # происхождение — встроенное

    @property  # доступ как к атрибуту
    def meta_bits(self) -> int:  # длина описания конструкции в битах
        """Длина описания встроенного представления: один токен, 8 бит."""
        return 8  # один токен на встроенное представление

    def supports_task(self, profile: dict) -> bool:
        """Годится ли представление для задачи: предикат семейства; при его ошибке — False."""
        try:  # предикат может бросить ошибку на неполном профиле
            return bool(self.family_pred(profile))  # результат предиката как булево значение
        except Exception:  # предикат бросил любую ошибку
            return False  # считать, что задача не подходит

    def probe_potential(self, obs: dict) -> float:
        """Оценка потенциальной пользы: всегда 0.0.

        Встроенные представления упорядочивает существующая лестница синтеза,
        поэтому оценка потенциала не используется."""
        # встроенные представления упорядочивает лестница синтеза; потенциал не нужен
        return 0.0  # вернуть 0.0

    def synthesize(self, obs: dict, budget: float, ledger: Any) -> Any:
        """Синтез по наблюдениям obs через synthesize_fn(obs); при любой ошибке — None."""
        try:  # синтез может бросить ошибку
            return self.synthesize_fn(obs)  # результат функции синтеза (budget и ledger не передаются)
        except Exception:  # функция синтеза бросила любую ошибку
            return None  # синтез не удался


def register_builtins() -> None:
    """Обернуть три ветви лестницы обучения во встроенные представления и внести их в реестр.

    Функции синтеза всех трёх возвращают None: сами ветви (булевы диаграммы Венна,
    АНФ, шаблоны) обрабатывает прежняя лестница обучения, так что поведение
    обучения не меняется."""
    register(BuiltinRepr(  # зарегистрировать представление
        "boolvenn",  # идентификатор — булевы диаграммы Венна
        lambda prof: prof.get("family") == "BoolVenn",  # подходит задаче, у которой семейство — BoolVenn
        lambda obs: None,  # синтез не делает ничего: булевы задачи обрабатывает сама лестница
    ))  # конец регистрации
    register(BuiltinRepr(  # зарегистрировать представление
        "anf",  # идентификатор — алгебраическая нормальная форма
        lambda prof: prof.get("bool_domain", False) and prof.get("arity", 0) <= 6,  # подходит булевой задаче (по умолчанию False) с арностью не выше 6 (по умолчанию 0)
        lambda obs: None,  # синтез возвращает None
    ))  # конец регистрации
    register(BuiltinRepr(  # зарегистрировать представление
        "templates",  # идентификатор — прямые шаблоны
        lambda prof: prof.get("family") in ("Arith", "Poly", "Mixed"),  # подходит задаче арифметики, многочленов или смешанной
        lambda obs: None,  # синтез возвращает None
    ))  # конец регистрации


# --------------------------------------------------------------------------
# Сериализация артефактов: восстановление через реестр фабрик
# --------------------------------------------------------------------------

ARTIFACT_FACTORIES: dict[str, Callable[[dict], Any]] = {}  # фабрики артефактов: идентификатор представления -> функция словарь -> артефакт


def register_artifact_factory(repr_id: str,  # идентификатор представления
                              factory: Callable[[dict], Any]) -> None:  # фабрика артефакта
    """Связать идентификатор представления repr_id с фабрикой factory, восстанавливающей артефакт из словаря."""
    ARTIFACT_FACTORIES[repr_id] = factory  # записать фабрику по идентификатору (повторная запись заменяет прежнюю)


def artifact_to_payload(value: dict) -> dict:
    """Значение узла construct -> словарь для JSON: идентификатор представления и словарь артефакта."""
    art = value["artifact"]  # артефакт конструкции
    payload = art.to_dict()  # артефакт в виде словаря
    return {"repr_id": value.get("repr_id"), "artifact": payload}  # полезная нагрузка: идентификатор и словарь артефакта


def artifact_from_payload(payload: dict) -> dict:
    """Словарь из JSON -> значение узла construct: артефакт восстанавливается фабрикой его представления.

    Фабрики для такого идентификатора нет — ValueError."""
    rid = payload["repr_id"]  # идентификатор представления rid
    factory = ARTIFACT_FACTORIES.get(rid)  # его фабрика (None, если не зарегистрирована)
    if factory is None:  # фабрика не найдена
        raise ValueError(f"no artifact factory for repr {rid}")  # ошибка: нет фабрики артефактов для представления rid
    return {"repr_id": rid, "artifact": factory(payload["artifact"])}  # значение узла: артефакт = фабрика(словарь артефакта)


# Вспомогательные функции узлов construct для Expr (вызываются из core в to_dict/from_dict)

def construct_to_dict_safe(value: dict) -> dict:
    """Сериализовать значение узла construct: то же, что artifact_to_payload."""
    return artifact_to_payload(value)  # полезная нагрузка для JSON


def construct_from_dict_safe(payload: dict) -> dict:
    """Восстановить значение узла construct: то же, что artifact_from_payload."""
    return artifact_from_payload(payload)  # значение узла с восстановленным артефактом


# ============================================================
# ПЕРЕИФИКАЦИЯ: регистрация изобретённых операций
# ============================================================

from typing import Any, Callable  # повторный импорт типов (имена уже импортированы выше)
from hagi.core import REIFIED_OPS  # реестр макрооператоров выражений Expr
from hagi.invention import ConstructBlueprint  # чертёж изобретённой конструкции
REIFIED_OP_COST_BITS = 12  # длина описания одного переифицированного оператора в битах
def _normalize_domain(domain: Any, arity: int):
    """Сертифицированная область как неизменяемое множество кортежей с проверкой арности.

    Каждая точка domain превращается в кортеж целых (кортеж или список — по элементам,
    одиночное значение — кортеж из одного). Число элементов обязано равняться arity.
    Возвращает frozenset; при domain = None — None."""
    if domain is None:  # область не задана
        return None  # вернуть None — проверки области не будет
    support = set()  # пустое множество точек
    for point in domain:  # по точкам области
        key = tuple((int(v) for v in point)) if isinstance(point, (tuple, list)) else (int(point),)  # ключ-кортеж: (int(v) по элементам) для кортежа или списка, иначе (int(point),)
        if len(key) != arity:  # длина ключа не равна арности
            raise ValueError(f'domain point {point!r} has arity {len(key)}, expected {arity}')  # ошибка: у точки неверная арность
        support.add(key)  # добавить ключ в множество
    return frozenset(support)  # вернуть неизменяемое множество точек
def reify_construct(bp: ConstructBlueprint, name: str, fn: Callable[..., int], arity: int=1, domain: Any=None) -> str:
    """Зарегистрировать функцию fn как оператор 'reified:<name>' и вернуть имя оператора.

    bp — чертёж конструкции; name — имя; fn(*vals) -> int — чистая сертифицированная
    функция (её проверяет гейт переноса до этого вызова); arity — число аргументов;
    domain — сертифицированная область (множество точек). Область проверяется при
    каждом вызове: аргументы вне области бросают ValueError, а не молча дают число,
    на которое сертификат не распространяется."""
    op_name = f'reified:{name}'  # имя оператора: 'reified:' + name
    if op_name in REIFIED_OPS:  # оператор с таким именем уже зарегистрирован
        raise ValueError(f'reified op already exists: {op_name}')  # ошибка: повторная регистрация
    support = _normalize_domain(domain, arity)  # сертифицированная область как frozenset (или None)

    def _wrapped(*vals: int) -> int:  # обёртка: проверяет арность и область, затем вызывает fn
        if len(vals) != arity:  # число аргументов не равно арности
            raise ValueError(f'{op_name} expects {arity} args, got {len(vals)}')  # ошибка: неверное число аргументов
        key = tuple((int(v) for v in vals))  # ключ вызова: кортеж аргументов, приведённых к целым
        if support is not None and key not in support:  # область задана и ключ вне её
            raise ValueError(f'{op_name} outside certified support: {key}')  # ошибка: вызов вне сертифицированной области
        return int(fn(*key))  # результат int(fn(ключ)) — приведение к целому
    _wrapped.__name__ = op_name  # имя обёртки — имя оператора
    import inspect  # доступ к исходному тексту функции (локальный импорт)
    try:  # исходник получить не всегда возможно
        _candidate = inspect.getsource(fn).strip()  # исходный текст fn без крайних пробелов
        _source = _candidate if _candidate.startswith('def ') else None  # взять текст, только если он начинается с 'def ' (функция, а не лямбда), иначе None
    except (OSError, TypeError):  # исходник недоступен
        _source = None  # исходного текста нет
    _wrapped.reified_meta = {'op': op_name, 'arity': arity, 'meta_bits': REIFIED_OP_COST_BITS, 'origin': bp.q_name if hasattr(bp, 'q_name') else 'invented', 'domain': sorted(domain) if domain is not None else None, 'source': _source}  # метаданные: имя, арность, 12 бит, происхождение (имя фактора q_name у чертежа, иначе 'invented'), отсортированная область (или None), исходный текст (или None)
    REIFIED_OPS[op_name] = _wrapped  # внести обёртку в реестр макрооператоров
    return op_name  # вернуть имя оператора
def snapshot_reified() -> dict:
    """Сериализуемый снимок реестра макрооператоров: имя -> копия его метаданных (для сохранения в rewrite.py)."""
    out = {}  # пустой снимок
    for op, fn in REIFIED_OPS.items():  # по парам (имя оператора, функция)
        meta = getattr(fn, 'reified_meta', {})  # метаданные функции (пустой словарь, если атрибута нет)
        out[op] = dict(meta)  # копия метаданных в снимок
    return out  # вернуть снимок
def clear_reified() -> None:
    """Очистить реестр макрооператоров."""
    REIFIED_OPS.clear()  # удалить все операторы
def _state_sig_path(state_dir):
    """Путь к файлу контрольной метки сохранённых операторов: invented_reprs.sig внутри state_dir."""
    from pathlib import Path  # работа с путями (локальный импорт)
    return Path(state_dir) / 'invented_reprs.sig'  # путь к файлу подписи: state_dir/invented_reprs.sig
def _state_hmac(payload_bytes: bytes, state_dir) -> str:
    """Контрольная метка целостности, привязанная к машине: hex(SHA-256(ключ + байты)).

    Ключ — 32 случайных байта из файла .reified_key в state_dir (создаётся при первом
    вызове). Метка выявляет порчу или тихую подмену invented_reprs.json между
    сохранением и возобновлением; цифровые подписи личности хранятся отдельно,
    в certificates.py."""
    import hashlib  # хеширование (локальный импорт)
    import os  # генератор случайных байтов (локальный импорт)
    from pathlib import Path  # работа с путями (локальный импорт)
    key_file = Path(state_dir) / '.reified_key'  # путь к файлу ключа
    if key_file.exists():  # файл ключа уже есть
        key = key_file.read_bytes()  # прочитать ключ
    else:  # ключа ещё нет
        key = os.urandom(32)  # новый ключ — 32 случайных байта
        key_file.write_bytes(key)  # записать ключ в файл
    return hashlib.sha256(key + payload_bytes).hexdigest()  # hex-строка SHA-256(ключ + payload_bytes)
def save_reified_ops_state(state_dir) -> int:
    """Сохранить макрооператоры в state_dir в виде исходного текста и метаданных.

    Файл invented_reprs.json содержит по оператору с исходником пары код/мета; рядом
    пишется контрольная метка invented_reprs.sig. Возвращает число сохранённых.
    Операторы хранятся как исходный текст, поэтому переживают холодный перезапуск."""
    import json  # запись JSON (локальный импорт)
    from pathlib import Path  # работа с путями (локальный импорт)
    d = Path(state_dir)  # каталог состояния
    d.mkdir(parents=True, exist_ok=True)  # создать его вместе с родительскими (существует — не ошибка)
    out = {}  # что сохранять: имя оператора -> {код, метаданные}
    for op, fn in REIFIED_OPS.items():  # по операторам реестра
        meta = getattr(fn, 'reified_meta', {})  # метаданные оператора (пустой словарь, если нет)
        code = meta.get('source')  # исходный текст (None, если его нет)
        if code:  # исходник есть
            out[op] = {'code': code, 'meta': {k: v for k, v in meta.items() if k != 'source'}}  # сохранить код и метаданные без ключа source
    payload = json.dumps(out, ensure_ascii=False, indent=1).encode('utf-8')  # текст JSON с отступом 1 (кириллица как есть) в байтах
    (d / 'invented_reprs.json').write_bytes(payload)  # записать его в файл invented_reprs.json
    _state_sig_path(d).write_text(_state_hmac(payload, d))  # записать контрольную метку байтов payload
    return len(out)  # число сохранённых операторов
def restore_reified_ops_state(state_dir, require_tag: bool=True) -> int:
    """Восстановить макрооператоры из файла invented_reprs.json в state_dir.

    Прежде чем исполнять какой-либо сохранённый исходный текст, сверяется
    контрольная метка. Отсутствующая или несовпавшая метка при require_tag=True
    даёт ошибку (отказ по умолчанию). Возвращает число восстановленных операторов;
    уже зарегистрированные пропускаются."""
    import json  # чтение JSON (локальный импорт)
    from pathlib import Path  # работа с путями (локальный импорт)
    f = Path(state_dir) / 'invented_reprs.json'  # путь к сохранённым операторам
    if not f.exists():  # файла нет — сохранять было нечего
        return 0  # восстановлено 0 операторов
    payload = f.read_bytes()  # байты файла
    tag = _state_sig_path(state_dir)  # путь к файлу контрольной метки
    if tag.exists():  # метка есть
        if tag.read_text().strip() != _state_hmac(payload, state_dir):  # записанная метка (без крайних пробелов) не равна вычисленной
            raise ValueError('reified ops integrity check FAILED — invented_reprs.json modified after save; refusing to exec')  # ошибка: файл изменён после сохранения — исполнять отказ
    elif require_tag:  # метки нет и она обязательна
        raise ValueError('reified ops store has no integrity tag; refusing to exec unsigned state')  # ошибка: состояние без метки — исполнять отказ
    data = json.loads(payload.decode('utf-8'))  # разобрать JSON: имя оператора -> {код, метаданные}
    n = 0  # число восстановленных операторов
    import textwrap  # снятие общих отступов (локальный импорт)
    for op, rec in data.items():  # по парам (имя оператора, запись)
        if op in REIFIED_OPS:  # оператор уже зарегистрирован
            continue  # пропустить
        code = textwrap.dedent(rec['code'])  # исходный текст без общих отступов
        ns: dict = {'__name__': 'reified_restore'}  # пространство имён для исполнения: имя модуля — reified_restore
        exec(compile(code, f'<reified:{op}>', 'exec'), ns)  # исполнить текст как модуль; имя файла в ошибках — <reified:op>
        fns = [v for v in ns.values() if callable(v) and getattr(v, '__module__', None) == 'reified_restore']  # объекты, определяемые в пространстве: вызываемые и объявленные в нём
        if not fns:  # функции не найдены
            continue  # пропустить запись
        pure = fns[0]  # первая определённая функция — восстанавливаемая чистая функция
        meta = rec.get('meta', {})  # её метаданные (пустой словарь, если нет)
        arity = int(meta.get('arity', 1))  # арность (по умолчанию 1)
        # Восстановление теряло защиту сертифицированной области: до сохранения
        # вызов вне области бросал ValueError, после восстановления возвращал число.
        # Замыкание области заново строится по meta['domain'].
        domain = meta.get('domain')  # сертифицированная область из метаданных (None, если нет)
        if domain is not None:  # область задана
            support = frozenset(  # множество точек-кортежей целых
                (tuple(int(v) for v in pt)  # для кортежа или списка: (int(v) по элементам)
                 if isinstance(pt, (tuple, list)) else (int(pt),))  # для одиночного значения: (int(pt),)
                for pt in domain)  # по точкам области
        else:  # область не задана
            support = None  # проверки области не будет

        def _wrapped(*vals: int, _p=pure, _a=arity, _op=op,  # обёртка с аргументами по умолчанию: функция, арность, имя
                     _s=support):  # и область
            if len(vals) != _a:  # число аргументов не равно арности
                raise ValueError(f'{_op} expects {_a} args, got {len(vals)}')  # ошибка: неверное число аргументов
            key = tuple(int(v) for v in vals)  # ключ вызова: кортеж аргументов-целых
            if _s is not None and key not in _s:  # область задана и ключ вне её
                raise ValueError(  # ошибка:
                    f'{_op} outside certified support: {key}')  # вызов вне сертифицированной области
            return int(_p(*key))  # результат int(функция(ключ))
        _wrapped.__name__ = op  # имя обёртки — имя оператора
        _wrapped.reified_meta = dict(meta)  # копия метаданных в атрибуте обёртки
        REIFIED_OPS[op] = _wrapped  # внести обёртку в реестр
        n += 1  # восстановлено: n + 1
    return n  # вернуть число восстановленных операторов