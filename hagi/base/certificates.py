"""Переносимые сертификаты конечной области — проверка только стандартной библиотекой.

Сертификат — JSON-артефакт со списком задач. Для каждой задачи объявлена конечная
область входов (декартово произведение списков значений по измерениям, отфильтрованная
правилом допустимости), таблица ожидаемых ответов и программа — целочисленное
выражение (AST). Проверяющий перечисляет все строки области, вычисляет программу и
сравнивает с ответами: нет ни одного расхождения -> задача доказана на этой области.

Главные формулы и механизмы:
  - строки области: rows = {(x_1, ..., x_n) : x_i входит в domains[i]} и valid_rule(xs);
  - контроль целостности: artifact_sha256 = SHA-256(канонический JSON без этого поля),
    канонизация — ключи по алфавиту, разделители без пробелов;
  - подпись (RSA PKCS#1 v1.5, SHA-256): s^e mod n должно равняться
    00 01 FF..FF 00 || DigestInfo || SHA-256(сообщение), где сообщение =
    "HAGI-PORTABLE-CERT-SIGNATURE-V1" + нулевой байт + байты хеша артефакта;
  - длина кода узла в битах: 6 битов на токен + гамма-коды величин,
    гамма-код(n) = 2*floor(log2 n) + 1; зигзаг(n) = 2n при n >= 0 и -2n - 1 при n < 0.

Валидный артефакт доказывает только совпадение с вложенной конечной таблицей
ответов. Он НЕ устанавливает, кто поставил ответы, не доказывает утверждение на
бесконечной области и не гарантирует правильность вне записанной области.
Функции: _canonical_bytes, _sha256, _eval_program, _eval_expr, _expr_bits,
_validate_library, verify_portable_certificate, verify_portable_signature,
upgrade_portable_certificate_v1_to_v2, main."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import argparse  # разбор аргументов командной строки
import base64  # декодирование подписи из base64
import hashlib  # хеш-функция SHA-256
import hmac  # сравнение за постоянное время
import itertools  # декартово произведение областей определения
import json  # чтение и запись JSON
import math  # gcd и lcm для целых чисел
from pathlib import Path  # путь к файлу
from typing import Any, Mapping, Sequence  # аннотации типов

FORMAT_V1 = "HAGI-PORTABLE-CERT-V1"  # имя формата сертификата версии 1
FORMAT = "HAGI-PORTABLE-CERT-V2"  # имя формата сертификата версии 2
FORMAT_VERSION = 2  # номер текущей версии формата
AST_SCHEMA = "hagi.expr.v1"  # имя схемы выражений
VERIFIER_PROFILE = "hagi.portable-verifier.v2"  # имя профиля проверяющего
from hagi.base.config import TOKEN_BITS  # число битов на токен выражения — общий формат кодера и сертификатов
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024  # максимальный размер артефакта: 64·2^20 байтов
MAX_SIGNATURE_BYTES = 128 * 1024  # максимальный размер файла подписи или ключа: 128·2^10 байтов
MAX_DOMAIN_ROWS = 250_000  # максимальное число строк области одной задачи
MAX_TOTAL_DOMAIN_ROWS = 250_000  # максимальное суммарное число строк области по всем задачам
MAX_TASKS = 10_000  # максимальное число задач
MAX_EXPERTS = 10_000  # максимальное число экспертов в библиотеке
MAX_AST_NODES = 20_000  # максимальное число узлов выражения одного эксперта
MAX_TOTAL_AST_NODES = 250_000  # максимальное суммарное число узлов выражений всех экспертов
MAX_AST_DEPTH = 128  # максимальная глубина вложенности выражения
MAX_AST_PAYLOAD_ITEMS = 100_000  # максимальное число элементов полезной нагрузки узла
MAX_EVAL_WORK_PER_TASK = 20_000_000  # максимальная работа вычисления на одну задачу
# старое имя того же предела, оставленное для совместимости с локальными скриптами
MAX_EVAL_NODES_PER_TASK = MAX_EVAL_WORK_PER_TASK  # старое имя того же предела, оставленное для совместимости с локальными скриптами
MAX_INTEGER_BITS = 4096  # максимальная разрядность целого числа в битах

_UNARY_OPS = {"neg", "abs", "sign", "bit_not8", "square", "cube", "clamp3"}  # унарные операции: neg, abs, sign, bit_not8, square, cube, clamp3
_BINARY_OPS = {  # начало множества бинарных операций
    "add", "sub", "mul", "floordiv", "mod", "gcd", "lcm", "min", "max",  # арифметика: сложение, вычитание, умножение, деление нацело, остаток, НОД, НОК, минимум, максимум
    "eq", "ne", "lt", "le", "gt", "ge", "bit_and", "bit_or", "bit_xor",  # сравнения: равно, не равно, меньше, не больше, больше, не меньше
    "bit_nand8", "bit_nor8", "bit_xnor8", "pow_int", "absdiff",  # побитовые: не-и, не-или, не-искл. или (по 8 бит), степень, модуль разности
}
_TERNARY_OPS = {"fma", "clamp", "median3", "mul3"}  # тернарные операции: fma, clamp, median3, mul3
_AST_OP_ARITY = {  # таблица арностей операций
    "var": 0, "const": 0, "if_cond": 3, "call": None,  # var, const имеют 0 аргументов, if_cond — 3, арность call зависит от вызываемого
    "in_set": 1, "in_ranges": 1, "anf": 0, "poly": 1, "poly_nd": 0,  # in_set, in_ranges, poly — 1 аргумент; anf, poly_nd — 0
    **{op: 1 for op in _UNARY_OPS},  # каждая унарная операция имеет арность 1
    **{op: 2 for op in _BINARY_OPS},  # каждая бинарная операция имеет арность 2
    **{op: 3 for op in _TERNARY_OPS},  # каждая тернарная операция имеет арность 3
}


def _canonical_bytes(value: Any) -> bytes:
    """Канонические байты значения: JSON с ключами по алфавиту, без пробелов, UTF-8."""
    return json.dumps(  # сериализация в JSON: ключи по алфавиту, разделители без пробелов, юникод без экранирования
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,  # сортировка ключей, разделители ("," , ":"), без экранирования не-ASCII
        allow_nan=False,  # NaN и Infinity запрещены
    ).encode("utf-8")  # кодирование текста в байты UTF-8


def _sha256(value: Any) -> str:
    """SHA-256 канонических байтов значения; возвращает 64 шестнадцатеричных символа."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()  # хеш SHA-256 канонических байтов значения в виде шестнадцатеричной строки


_AST_SCHEMA_DESCRIPTOR = {  # описание схемы выражений (словарь, хеш которого вшит в сертификат)
    "schema": AST_SCHEMA,  # имя схемы
    "integer_model": "signed integer; every evaluated node result is limited to 4096 bits",  # модель целых: знаковые целые, результат каждого узла не шире 4096 битов
    "operators": {  # словарь операций
        "var": {"arity": 0, "semantics": "read zero-based input variable"},  # var: 0 аргументов, чтение входной переменной с нумерацией с нуля
        "const": {"arity": 0, "semantics": "integer literal"},  # const: 0 аргументов, целочисленная константа
        "if_cond": {"arity": 3, "semantics": "lazy; evaluate condition then selected branch"},  # if_cond: 3 аргумента, ленивое вычисление условия и выбранной ветви
        "call": {"arity": "callee-arity", "semantics": "acyclic call to expert; inputs must lie in callee support"},  # call: арность вызываемого, вызов эксперта без циклов, аргументы в его области действия
        "in_set": {"arity": 1, "semantics": "integer membership in strictly increasing finite set"},  # in_set: 1 аргумент, принадлежность строго возрастающему конечному множеству
        "in_ranges": {"arity": 1, "semantics": "inclusive membership in sorted disjoint non-adjacent intervals"},  # in_ranges: 1 аргумент, принадлежность несмежным отрезкам
        "anf": {"arity": 0, "semantics": "xor coefficients whose subset mask is contained in input bitmask"},  # anf: 0 аргументов, искл. или коэффициентов, чьи маски подмножеств входят в маску входа
        "poly": {"arity": 1, "semantics": "integer polynomial evaluated by Horner's method; coefficients low-to-high"},  # poly: 1 аргумент, многочлен по схеме Горнера, коэффициенты от младших к старшим
        "poly_nd": {"arity": 0, "semantics": "sum of integer coefficient times each input raised to its nonnegative power"},  # poly_nd: 0 аргументов, сумма коэффициентов, умноженных на входы в неотрицательных степенях
        "neg": {"arity": 1, "semantics": "-x"},  # отрицание: neg: −x
        "abs": {"arity": 1, "semantics": "absolute value"},  # abs: абсолютное значение
        "sign": {"arity": 1, "semantics": "-1, 0, or 1"},  # sign: −1, 0 или 1
        "bit_not8": {"arity": 1, "semantics": "(~x) & 255"},  # побитовое отрицание по модулю 256: bit_not8: (~x) & 255
        "square": {"arity": 1, "semantics": "x*x"},  # квадрат: square: x·x
        "cube": {"arity": 1, "semantics": "x*x*x"},  # куб: cube: x·x·x
        "clamp3": {"arity": 1, "semantics": "min(max(x,-3),3)"},  # усечение в отрезок [−3, 3]: clamp3: min(max(x, −3), 3)
        "add": {"arity": 2, "semantics": "a+b"},  # сложение: add: a + b
        "sub": {"arity": 2, "semantics": "a-b"},  # вычитание: sub: a − b
        "mul": {"arity": 2, "semantics": "a*b"},  # умножение: mul: a·b
        "floordiv": {"arity": 2, "semantics": "Python integer floor division"},  # floordiv: деление нацело по правилам Python
        "mod": {"arity": 2, "semantics": "Python integer modulo"},  # mod: остаток по правилам Python
        "gcd": {"arity": 2, "semantics": "nonnegative greatest common divisor"},  # gcd: неотрицательный наибольший общий делитель
        "lcm": {"arity": 2, "semantics": "nonnegative least common multiple"},  # lcm: неотрицательное наименьшее общее кратное
        "min": {"arity": 2, "semantics": "minimum"},  # min: минимум
        "max": {"arity": 2, "semantics": "maximum"},  # max: максимум
        "eq": {"arity": 2, "semantics": "1 iff a equals b, else 0"},  # eq: 1, если a = b, иначе 0
        "ne": {"arity": 2, "semantics": "1 iff a differs from b, else 0"},  # ne: 1, если a ≠ b, иначе 0
        "lt": {"arity": 2, "semantics": "1 iff a is less than b, else 0"},  # lt: 1, если a < b, иначе 0
        "le": {"arity": 2, "semantics": "1 iff a is at most b, else 0"},  # le: 1, если a ≤ b, иначе 0
        "gt": {"arity": 2, "semantics": "1 iff a is greater than b, else 0"},  # gt: 1, если a > b, иначе 0
        "ge": {"arity": 2, "semantics": "1 iff a is at least b, else 0"},  # ge: 1, если a ≥ b, иначе 0
        "bit_and": {"arity": 2, "semantics": "integer bitwise and"},  # bit_and: побитовое и
        "bit_or": {"arity": 2, "semantics": "integer bitwise or"},  # bit_or: побитовое или
        "bit_xor": {"arity": 2, "semantics": "integer bitwise xor"},  # bit_xor: побитовое исключающее или
        "bit_nand8": {"arity": 2, "semantics": "(~(a&b)) & 255"},  # побитовое «и-не»: bit_nand8: (~(a & b)) & 255
        "bit_nor8": {"arity": 2, "semantics": "(~(a|b)) & 255"},  # побитовое «или-не»: bit_nor8: (~(a | b)) & 255
        "bit_xnor8": {"arity": 2, "semantics": "(~(a^b)) & 255"},  # побитовое «исключающее или-не»: bit_xnor8: (~(a ^ b)) & 255
        "pow_int": {"arity": 2, "semantics": "a**b; exponent must be in [0,8]"},  # pow_int: a в степени b, показатель из отрезка [0, 8]
        "absdiff": {"arity": 2, "semantics": "abs(a-b)"},  # модуль разности: absdiff: |a − b|
        "fma": {"arity": 3, "semantics": "a*b+c"},  # умножение со сложением: fma: a·b + c
        "clamp": {"arity": 3, "semantics": "min(max(a,b),c)"},  # усечение снизу и сверху: clamp: min(max(a, b), c)
        "median3": {"arity": 3, "semantics": "median of three integers"},  # median3: медиана трёх целых
        "mul3": {"arity": 3, "semantics": "a*b*c"},  # тройное произведение: mul3: a·b·c
    },
}
AST_SCHEMA_SHA256 = _sha256(_AST_SCHEMA_DESCRIPTOR)  # хеш SHA-256 описания схемы, вшитый в проверяющего


def _int(value: Any, where: str) -> int:
    """Проверка целого: не bool, тип int, модуль не шире 4096 битов; where — имя места для ошибки."""
    if isinstance(value, bool) or not isinstance(value, int):  # булево значение не считается целым: значение обязано быть типа int
        raise ValueError(f"{where} must be an integer")  # ошибка: значение не целое
    if abs(value).bit_length() > MAX_INTEGER_BITS:  # модуль значения не шире 4096 битов
        raise ValueError(f"{where} exceeds integer bit limit")  # ошибка: превышен предел разрядности
    return value  # вернуть проверенное целое


def _gamma_bits(n: int) -> int:
    """Длина гамма-кода натурального n в битах: 2*(bit_length - 1) + 1 = 2*floor(log2 n) + 1."""
    if n < 1:  # гамма-код определён только для натурального n
        raise ValueError("gamma code requires positive integer")  # ошибка: n < 1
    return 2 * (n.bit_length() - 1) + 1  # длина гамма-кода: 2·⌊log2 n⌋ + 1, где bit_length − 1 = ⌊log2 n⌋


def _zigzag(n: int) -> int:
    """Зигзаг-код: n >= 0 -> 2n; n < 0 -> -2n - 1 (взаимно однозначное отображение в натуральные)."""
    return 2 * n if n >= 0 else -2 * n - 1  # зигзаг-код: n ≥ 0 -> 2n; n < 0 -> −2n − 1


def _valid_input(xs: Sequence[int], valid_rule: str) -> bool:
    """Допустим ли вход xs по правилу: all — всегда; nonzero_y — xs[1] != 0; ordered_yz — xs[1] <= xs[2]."""
    if valid_rule == "all":  # правило "all": любой вход допустим
        return True  # вернуть истину
    if valid_rule == "nonzero_y":  # правило "nonzero_y": y = xs[1] не ноль
        return len(xs) >= 2 and xs[1] != 0  # вернуть len(xs) ≥ 2 и xs[1] ≠ 0
    if valid_rule == "ordered_yz":  # правило "ordered_yz": xs[1] ≤ xs[2]
        return len(xs) >= 3 and xs[1] <= xs[2]  # вернуть len(xs) ≥ 3 и xs[1] ≤ xs[2]
    raise ValueError(f"unsupported valid_rule: {valid_rule}")  # ошибка: правило не поддерживается


def _validated_domain_dimensions(
    domains: Any, valid_rule: str, *, row_limit: int = MAX_DOMAIN_ROWS,  # аргументы: области, правило допустимости, именованный предел строк
) -> list[tuple[int, ...]]:  # возвращает список кортежей допустимых значений по измерениям
    """Проверить объявленную область и вернуть список кортежей значений по измерениям.
    
        Требования: список из 1..8 непустых измерений без повторов значений; произведение
        длин измерений не больше row_limit (иначе область слишком велика для перебора)."""
    if not isinstance(domains, list) or not domains or len(domains) > 8:  # список непустой и не длиннее 8 измерений
        raise ValueError("domains must be a non-empty list of at most 8 dimensions")  # ошибка: список пуст или длиннее 8
    if valid_rule not in {"all", "nonzero_y", "ordered_yz"}:  # правило допустимости должно быть одним из трёх
        raise ValueError(f"unsupported valid_rule: {valid_rule}")  # ошибка: правило не поддерживается
    parsed: list[tuple[int, ...]] = []  # разобранные измерения
    size = 1  # размер декартова произведения, начально 1
    for i, dim in enumerate(domains):  # по измерениям с номером i
        if not isinstance(dim, list) or not dim:  # измерение — непустой список
            raise ValueError(f"domain dimension {i} must be a non-empty list")  # ошибка: измерение не список или пусто
        vals = tuple(_int(v, f"domains[{i}][]") for v in dim)  # значения измерения как кортеж целых
        if len(set(vals)) != len(vals):  # значения в измерении различны
            raise ValueError(f"domain dimension {i} contains duplicate values")  # ошибка: есть повторяющиеся значения
        parsed.append(vals)  # добавить измерение
        size *= len(vals)  # размер: size = size · len(vals)
        if size > row_limit:  # произведение не превышает предел строк
            raise ValueError("declared Cartesian domain exceeds verifier row limit")  # ошибка: декартова область превышает предел строк
    return parsed  # вернуть разобранные измерения


def _domain_rows(domains: Any, valid_rule: str) -> list[tuple[int, ...]]:
    """Все строки области: декартово произведение измерений, оставлены только допустимые по valid_rule."""
    parsed = _validated_domain_dimensions(domains, valid_rule)  # разобрать измерения
    return [tuple(int(v) for v in row) for row in itertools.product(*parsed)  # строки декартова произведения (кортежи int), где вход допустим по правилу
            if _valid_input(row, valid_rule)]  # условие отбора строк


def _program_from_library(library: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Запись эксперта name из библиотеки; ошибка, если её нет или она не словарь."""
    item = library.get(name)  # запись библиотеки по имени
    if not isinstance(item, dict):  # запись обязана быть словарём
        raise ValueError(f"missing program library entry: {name}")  # ошибка: в библиотеке нет такой записи
    return item  # вернуть запись


def _supports(item: Mapping[str, Any], xs: Sequence[int]) -> bool:
    """Лежит ли вход xs в области действия эксперта: каждый аргумент в своём измерении и valid_rule истинно."""
    domains = item.get("domains")  # области определения эксперта
    if not isinstance(domains, list) or len(domains) != len(xs):  # число измерений должно равняться числу аргументов
        return False  # ложь: арность не совпадает
    if any(int(xs[i]) not in set(domains[i]) for i in range(len(xs))):  # каждый аргумент int(xs[i]) обязан входить в своё измерение
        return False  # ложь: аргумент вне своей области
    return _valid_input(xs, str(item.get("valid_rule", "all")))  # истина, если вход допустим по правилу valid_rule (по умолчанию "all")


def _eval_program(
    program: Mapping[str, Any], xs: Sequence[int], library: Mapping[str, Any],  # аргументы: программа, вход, библиотека, стек вызовов, счётчик работы
    stack: tuple[str, ...] = (), counter: list[int] | None = None,  # стек — кортеж имён, счётчик — список из одного элемента или None
) -> int:  # возвращает целое значение программы
    """Вычислить программу на входе xs (кортеж целых); возвращает целое.
    
        Либо маска таблицы истинности: результат = (mask >> (2*x0 + x1)) & 1 при бинарных
        x0, x1, либо выражение expr. stack — имена экспертов в стеке вызовов,
        counter — список из одного элемента со счётчиком работы (или None)."""
    arity = _int(program.get("arity"), "program.arity")  # арность программы — целое
    if len(xs) != arity:  # число аргументов обязано равняться арности
        raise ValueError("program arity mismatch")  # ошибка: арность не совпадает
    mask = program.get("venn_mask")  # маска таблицы истинности (BoolVenn) или None
    if mask is not None:  # маска задана
        mask = _int(mask, "program.venn_mask")  # маска — целое
        if arity != 2 or not 0 <= mask <= 15 or program.get("expr") is not None:  # для маски нужно: арность 2, 0 ≤ mask ≤ 15, выражение отсутствует
            raise ValueError("invalid BoolVenn program")  # ошибка: неверная программа BoolVenn
        if int(xs[0]) not in (0, 1) or int(xs[1]) not in (0, 1):  # оба входа обязаны быть 0 или 1
            raise ValueError("BoolVenn input is not binary")  # ошибка: вход BoolVenn не бинарный
        return (mask >> ((int(xs[0]) << 1) | int(xs[1]))) & 1  # бит маски с номером 2·x0 + x1: результат = (mask >> (2·x0 + x1)) & 1
    expr = program.get("expr")  # выражение программы
    if not isinstance(expr, dict):  # без маски и без выражения — ошибка
        raise ValueError("program has neither BoolVenn mask nor expression")  # ошибка: нет ни маски, ни выражения
    return _eval_expr(expr, xs, library, stack, counter, 0)  # вычислить выражение с глубиной 0


def _eval_expr_inner(
    expr: Mapping[str, Any], xs: Sequence[int], library: Mapping[str, Any],  # аргументы: узел, вход, библиотека, стек, счётчик, глубина
    stack: tuple[str, ...], counter: list[int] | None, depth: int,  # стек вызовов, счётчик работы и глубина узла
) -> int:  # возвращает целое значение узла
    """Вычислить узел выражения вместе с поддеревом; возвращает целое.
    
        Счётчик counter[0] растёт на 1 за узел и на размер полезной нагрузки
        (число точек, отрезков или коэффициентов; для poly_nd — термов*(n + 1));
        предел MAX_EVAL_WORK_PER_TASK. depth — текущая глубина вложенности."""
    if depth > MAX_AST_DEPTH:  # глубина вложенности не больше 128
        raise ValueError("AST nesting limit exceeded")  # ошибка: превышена глубина вложенности
    if counter is not None:  # счётчик работы задан
        counter[0] += 1  # счётчик: counter[0] + 1
        if counter[0] > MAX_EVAL_WORK_PER_TASK:  # предел работы на задачу исчерпан
            raise ValueError("evaluation work budget exceeded")  # ошибка: превышен бюджет работы
    op = expr.get("op")  # имя операции
    if not isinstance(op, str):  # имя обязано быть строкой
        raise ValueError("AST op must be a string")  # ошибка: op не строка
    args = expr.get("args", [])  # аргументы-выражения, по умолчанию пусто
    if not isinstance(args, list):  # аргументы обязаны быть списком
        raise ValueError("AST args must be a list")  # ошибка: args не список
    value = expr.get("value")  # полезная нагрузка узла
    if counter is not None:  # учёт стоимости нагрузки
        payload_cost = 0  # стоимость нагрузки, начально 0
        if op == "in_set" and isinstance(value, list):  # для in_set нагрузка — список точек
            payload_cost = len(value)  # стоимость = число точек
        elif op == "in_ranges" and isinstance(value, list):  # для in_ranges нагрузка — список отрезков
            payload_cost = len(value)  # стоимость = число отрезков
        elif op == "anf" and isinstance(value, list) and len(value) == 2 and isinstance(value[1], list):  # для anf нагрузка — пара [n_vars, коэффициенты]
            payload_cost = len(value[1])  # стоимость = число коэффициентов
        elif op == "poly" and isinstance(value, list):  # для poly нагрузка — список коэффициентов
            payload_cost = len(value)  # стоимость = число коэффициентов
        elif op == "poly_nd" and isinstance(value, list):  # для poly_nd нагрузка — список термов
            payload_cost = len(value) * (len(xs) + 1)  # стоимость = число термов · (число входов + 1)
        counter[0] += payload_cost  # добавить стоимость нагрузки к счётчику
        if counter[0] > MAX_EVAL_WORK_PER_TASK:  # предел работы с учётом нагрузки исчерпан
            raise ValueError("evaluation payload work budget exceeded")  # ошибка: превышен бюджет работы с учётом нагрузки
    if op == "var":  # операция чтения переменной
        idx = _int(value, "var.value")  # индекс переменной — целое
        if args or idx < 0 or idx >= len(xs):  # аргументов быть не должно и 0 ≤ idx < len(xs)
            raise ValueError("invalid variable reference")  # ошибка: неверная ссылка на переменную
        return int(xs[idx])  # вернуть xs[idx]
    if op == "const":  # операция константы
        if args:  # у константы не должно быть аргументов
            raise ValueError("constant cannot have arguments")  # ошибка: у константы есть аргументы
        return _int(value, "const.value")  # вернуть проверенную константу
    if op == "if_cond":  # условное выражение
        if len(args) != 3:  # ровно три аргумента
            raise ValueError("if_cond expects three arguments")  # ошибка: нужно три аргумента
# ветвь вычисляется лениво: это обязательно для сохранения смысла выражения
        cond = _eval_expr(args[0], xs, library, stack, counter, depth + 1)  # условие cond = значение args[0]
        branch = args[1] if cond else args[2]  # выбрать ветвь args[1], если cond ≠ 0, иначе args[2]
        return _eval_expr(branch, xs, library, stack, counter, depth + 1)  # вернуть значение выбранной ветви
    vals = [_eval_expr(a, xs, library, stack, counter, depth + 1)  # значения всех аргументов, глубина + 1
            for a in args]  # вторая физическая строка того же списка
    unary = _UNARY_OPS  # множество унарных операций
    binary = _BINARY_OPS  # множество бинарных операций
    if op in unary and len(vals) != 1:  # унарной операции нужен один аргумент
        raise ValueError(f"{op} expects one argument")  # ошибка: нужен один аргумент
    if op in binary and len(vals) != 2:  # бинарной операции нужно два аргумента
        raise ValueError(f"{op} expects two arguments")  # ошибка: нужно два аргумента
    if op == "call":  # вызов эксперта
        if not isinstance(value, str) or not value:  # имя вызываемого эксперта — непустая строка
            raise ValueError("call requires a named expert")  # ошибка: нужен именованный эксперт
        if value in stack:  # имя уже в стеке — рекурсия
            raise ValueError("recursive call cycle in certificate")  # ошибка: цикл рекурсивных вызовов
        callee = _program_from_library(library, value)  # запись вызываемого эксперта
        call_xs = tuple(vals)  # аргументы вызова = значения аргументов
        if not _supports(callee, call_xs):  # вход обязан лежать в области действия вызываемого
            raise ValueError(f"call outside support of {value}")  # ошибка: вызов вне области действия
        return _eval_program(callee["program"], call_xs, library,  # вычислить программу вызываемого эксперта с теми же входами
                             stack + (value,), counter)  # стек вызовов расширяется именем вызываемого
    if op == "in_set":  # принадлежность множеству
        if len(vals) != 1 or not isinstance(value, list):  # один аргумент и список точек
            raise ValueError("invalid in_set expression")  # ошибка: неверное выражение in_set
        points = tuple(_int(v, "in_set.value[]") for v in value)  # точки как кортеж целых
        if not points or tuple(sorted(set(points))) != points:  # точки непусты и строго возрастают
            raise ValueError("in_set values must be strictly increasing")  # ошибка: точки обязаны строго возрастать
        return int(vals[0] in points)  # результат 1, если vals[0] входит в points, иначе 0
    if op == "in_ranges":  # принадлежность отрезкам
        if len(vals) != 1 or not isinstance(value, list) or not value:  # один аргумент и непустой список
            raise ValueError("invalid in_ranges expression")  # ошибка: неверное выражение in_ranges
        ranges = [tuple(_int(z, "in_ranges bound") for z in pair)  # пары (lo, hi) целых
                  for pair in value]  # вторая физическая строка того же списка
        if any(len(pair) != 2 or pair[0] > pair[1] for pair in ranges):  # в каждой паре два элемента и lo ≤ hi
            raise ValueError("invalid in_ranges bounds")  # ошибка: неверные границы
        if any(a[1] + 1 >= b[0] for a, b in zip(ranges, ranges[1:])):  # для соседних пар hi_a + 1 < lo_b: без пересечений и с зазором
            raise ValueError("in_ranges must be sorted, disjoint and non-adjacent")  # ошибка: отрезки обязаны быть без пересечений и несмежными
        return int(any(lo <= vals[0] <= hi for lo, hi in ranges))  # результат 1, если lo ≤ vals[0] ≤ hi хотя бы для одного отрезка
    if op == "anf":  # алгебраическая нормальная форма
        if args or not isinstance(value, list) or len(value) != 2:  # без аргументов, нагрузка — пара [n_vars, коэффициенты]
            raise ValueError("invalid anf payload")  # ошибка: неверная нагрузка anf
        n_vars = _int(value[0], "anf.n_vars")  # число переменных
        coeffs = value[1]  # список коэффициентов
        if n_vars != len(xs) or not isinstance(coeffs, list):  # n_vars = числу входов, коэффициенты — список
            raise ValueError("anf arity/payload mismatch")  # ошибка: арность и нагрузка не согласованы
        mask_x = 0  # битовая маска входа, начально 0
        for i, x in enumerate(xs):  # по входам с номером i
            if x not in (0, 1):  # входы бинарные
                raise ValueError("anf expects binary inputs")  # ошибка: вход anf не бинарный
            if x:  # вход x_i = 1 задаёт бит n_vars − 1 − i
                mask_x |= 1 << (n_vars - 1 - i)  # маска входа: mask_x = mask_x | 1 << (n_vars − 1 − i)
        acc = 0  # аккумулятор, начально 0
        for pair in coeffs:  # по парам [подмножество, коэффициент]
            if not isinstance(pair, list) or len(pair) != 2:  # пара из двух элементов
                raise ValueError("invalid anf coefficient")  # ошибка: неверный коэффициент anf
            subset = _int(pair[0], "anf subset")  # номер подмножества
            coeff = _int(pair[1], "anf coefficient")  # коэффициент
            if subset < 0 or subset >= (1 << n_vars):  # номер подмножества: 0 ≤ subset < 2^n_vars
                raise ValueError("anf subset out of range")  # ошибка: подмножество вне диапазона
            if (subset & mask_x) == subset:  # подмножество входит в маску входа: subset & mask_x = subset
                acc ^= coeff  # аккумулятор: acc = acc ⊕ coeff
        return acc  # вернуть сумму по модулю 2 включённых коэффициентов
    if op == "poly":  # многочлен одной переменной
        if len(args) != 1 or not isinstance(value, list) or not value:  # один аргумент, непустой список коэффициентов
            raise ValueError("invalid poly expression")  # ошибка: неверное выражение poly
        acc = 0  # аккумулятор, начально 0
        for coef in reversed(value):  # по коэффициентам от старшего к младшему
            acc = acc * vals[0] + _int(coef, "poly coefficient")  # схема Горнера: acc = acc·x + c, где x = vals[0]
        return acc  # вернуть значение многочлена
    if op == "poly_nd":  # многочлен нескольких переменных
        if args or not isinstance(value, list):  # без аргументов, нагрузка — список термов
            raise ValueError("invalid poly_nd expression")  # ошибка: неверное выражение poly_nd
        total = 0  # сумма, начально 0
        for term in value:  # по термам [коэффициент, показатели]
            if not isinstance(term, list) or len(term) != 2:  # терм из двух элементов
                raise ValueError("invalid poly_nd term")  # ошибка: неверный терм poly_nd
            coef = _int(term[0], "poly_nd coefficient")  # коэффициент терма
            powers = term[1]  # показатели степеней
            if not isinstance(powers, list) or len(powers) != len(xs):  # показателей ровно len(xs)
                raise ValueError("poly_nd power arity mismatch")  # ошибка: число показателей не равно числу входов
            product = coef  # произведение, начально равно коэффициенту
            for x, power in zip(xs, powers):  # по парам (вход, показатель)
                p = _int(power, "poly_nd power")  # показатель — целое
                if p < 0 or p > 100:  # показатель из отрезка [0, 100]
                    raise ValueError("poly_nd power outside verifier limit")  # ошибка: показатель вне пределов
                product *= int(x) ** p  # произведение: product = product · x^p
            total += product  # сумма: total = total + product
        return total  # вернуть сумму термов
    if op in unary or op in binary:  # унарная или бинарная операция
# вычислитель явный: без динамической диспетчеризации и без eval
        if op == "neg": return -vals[0]  # отрицание: neg: −a
        if op == "abs": return abs(vals[0])  # модуль: abs: |a|
        if op == "sign": return (vals[0] > 0) - (vals[0] < 0)  # знак числа: sign: (a > 0) − (a < 0)
        if op == "bit_not8": return (~vals[0]) & 0xFF  # побитовое отрицание по модулю 256: bit_not8: (~a) & 255
        if op == "square": return vals[0] * vals[0]  # квадрат: square: a²
        if op == "cube": return vals[0] * vals[0] * vals[0]  # куб: cube: a³
        if op == "clamp3": return min(max(vals[0], -3), 3)  # усечение в отрезок [−3, 3]: clamp3: min(max(a, −3), 3)
        a, b = vals  # распаковать два аргумента a, b
        if op == "add": return a + b  # сложение: add: a + b
        if op == "sub": return a - b  # вычитание: sub: a − b
        if op == "mul": return a * b  # умножение: mul: a·b
        if op == "floordiv": return a // b  # деление нацело: floordiv: a // b
        if op == "mod": return a % b  # остаток от деления: mod: a mod b
        if op == "gcd": return math.gcd(a, b)  # gcd: НОД(a, b)
        if op == "lcm": return math.lcm(a, b)  # lcm: НОК(a, b)
        if op == "min": return min(a, b)  # минимум: min: min(a, b)
        if op == "max": return max(a, b)  # максимум: max: max(a, b)
        if op == "eq": return int(a == b)  # eq: 1, если a = b, иначе 0
        if op == "ne": return int(a != b)  # ne: 1, если a ≠ b, иначе 0
        if op == "lt": return int(a < b)  # lt: 1, если a < b, иначе 0
        if op == "le": return int(a <= b)  # le: 1, если a ≤ b, иначе 0
        if op == "gt": return int(a > b)  # gt: 1, если a > b, иначе 0
        if op == "ge": return int(a >= b)  # ge: 1, если a ≥ b, иначе 0
        if op == "bit_and": return a & b  # побитовое «и»: bit_and: a & b
        if op == "bit_or": return a | b  # побитовое «или»: bit_or: a | b
        if op == "bit_xor": return a ^ b  # побитовое «исключающее или»: bit_xor: a ^ b
        if op == "bit_nand8": return (~(a & b)) & 0xFF  # побитовое «и-не»: bit_nand8: (~(a & b)) & 255
        if op == "bit_nor8": return (~(a | b)) & 0xFF  # побитовое «или-не»: bit_nor8: (~(a | b)) & 255
        if op == "bit_xnor8": return (~(a ^ b)) & 0xFF  # побитовое «исключающее или-не»: bit_xnor8: (~(a ^ b)) & 255
        if op == "pow_int":  # pow_int: возведение в степень
            if b < 0 or b > 8:  # показатель из отрезка [0, 8]
                raise ValueError("pow_int exponent outside [0,8]")  # ошибка: показатель вне [0, 8]
            return a ** b  # вернуть a^b
        if op == "absdiff": return abs(a - b)  # модуль разности: absdiff: |a − b|
    if op in {"fma", "clamp", "median3", "mul3"}:  # тернарные операции
        if len(vals) != 3:  # ровно три аргумента
            raise ValueError(f"{op} expects three arguments")  # ошибка: нужно три аргумента
        a, b, c = vals  # распаковать a, b, c
        if op == "fma": return a * b + c  # умножение со сложением: fma: a·b + c
        if op == "clamp": return min(max(a, b), c)  # усечение снизу и сверху: clamp: min(max(a, b), c)
        if op == "median3": return sorted(vals)[1]  # медиана трёх: средний элемент отсортированного списка
        if op == "mul3": return a * b * c  # тройное произведение: mul3: a·b·c
    raise ValueError(f"unsupported or malformed AST op: {op}")  # ошибка: неизвестная или неверная операция узла


def _eval_expr(
    expr: Mapping[str, Any], xs: Sequence[int], library: Mapping[str, Any],  # аргументы: узел, вход, библиотека, стек, счётчик, глубина
    stack: tuple[str, ...], counter: list[int] | None, depth: int,  # стек вызовов, счётчик работы и глубина узла
) -> int:  # возвращает целое значение узла
    """Вычислить узел через _eval_expr_inner и проверить: модуль результата не шире 4096 битов."""
    value = _eval_expr_inner(expr, xs, library, stack, counter, depth)  # значение узла
    if abs(value).bit_length() > MAX_INTEGER_BITS:  # результат не шире 4096 битов
        raise ValueError("expression result exceeds integer bit limit")  # ошибка: результат выражения шире предела
    return int(value)  # вернуть int(value)


def _expr_bits(expr: Mapping[str, Any], expert_indices: Mapping[str, int]) -> int:
    """Длина записи узла в битах: 6 битов токена + гамма-коды нагрузки + сумма длин аргументов.
    
        var: g(idx + 1); const: g(зигзаг(c) + 1); call: g(индекс эксперта + 1);
        anf: g(n_vars + 1) + g(число коэффициентов + 1) + сумма g(подмножество + 1);
        poly: g(число коэффициентов + 1) + сумма g(зигзаг(c) + 1); poly_nd: по термам;
        in_set: g(число точек + 1) + g(зигзаг(p0) + 1) + сумма g(разностей соседних точек);
        in_ranges: g(число отрезков + 1) + g(зигзаг(lo0) + 1) + g(hi0 - lo0 + 1)
        + по соседним парам g(lo_j - hi_(j-1)) + g(hi_j - lo_j + 1); прочие: только аргументы.
        Здесь g — гамма-код."""
    op = expr["op"]  # имя операции
    value = expr.get("value")  # полезная нагрузка
    args = expr.get("args", [])  # аргументы, по умолчанию пусто
    if op == "var": return TOKEN_BITS + _gamma_bits(_int(value, "var") + 1)  # var: 6 битов токена + гамма-код(индекс + 1)
    if op == "const": return TOKEN_BITS + _gamma_bits(_zigzag(_int(value, "const")) + 1)  # const: 6 битов + гамма-код(зигзаг(константа) + 1)
    if op == "call":  # вызов эксперта
        if value not in expert_indices: raise ValueError(f"unknown call index: {value}")  # ошибка: неизвестный индекс вызываемого
        return TOKEN_BITS + _gamma_bits(int(expert_indices[value]) + 1) + sum(_expr_bits(a, expert_indices) for a in args)  # 6 битов + гамма-код(индекс эксперта + 1) + Σ битов аргументов
    if op == "anf":  # алгебраическая нормальная форма
        n_vars, coeffs = value  # число переменных и коэффициенты
        payload = _gamma_bits(int(n_vars) + 1) + _gamma_bits(len(coeffs) + 1)  # нагрузка = гамма-код(n_vars + 1) + гамма-код(число коэффициентов + 1)
        payload += sum(_gamma_bits(int(pair[0]) + 1) for pair in coeffs)  # нагрузка += Σ гамма-код(номер подмножества + 1)
        return TOKEN_BITS + payload  # anf: итог 6 битов + нагрузка
    if op == "poly":  # многочлен одной переменной
        payload = _gamma_bits(len(value) + 1) + sum(_gamma_bits(_zigzag(int(v)) + 1) for v in value)  # нагрузка = гамма-код(числа коэффициентов + 1) + Σ гамма-код(зигзаг(c) + 1)
        return TOKEN_BITS + payload + sum(_expr_bits(a, expert_indices) for a in args)  # 6 битов + нагрузка + Σ битов аргументов
    if op == "poly_nd":  # многочлен нескольких переменных
        payload = _gamma_bits(len(value) + 1)  # нагрузка = гамма-код(числа термов + 1)
        for coef, powers in value:  # по термам (коэффициент, показатели)
            payload += _gamma_bits(_zigzag(int(coef)) + 1) + _gamma_bits(len(powers) + 1)  # нагрузка += гамма-код(зигзаг(коэффициента) + 1) + гамма-код(числа показателей + 1)
            payload += sum(_gamma_bits(int(p) + 1) for p in powers)  # нагрузка += Σ гамма-код(показатель + 1)
        return TOKEN_BITS + payload  # 6 битов + нагрузка
    if op == "in_set":  # принадлежность множеству
        pts = [int(v) for v in value]  # точки как целые
        payload = _gamma_bits(len(pts) + 1) + _gamma_bits(_zigzag(pts[0]) + 1)  # нагрузка = гамма-код(числа точек + 1) + гамма-код(зигзаг(p0) + 1)
        payload += sum(_gamma_bits(b - a) for a, b in zip(pts, pts[1:]))  # нагрузка += Σ гамма-код(p_{i+1} − p_i) по соседним точкам
        return TOKEN_BITS + payload + sum(_expr_bits(a, expert_indices) for a in args)  # 6 битов + нагрузка + Σ битов аргументов
    if op == "in_ranges":  # принадлежность отрезкам
        ranges = value  # список отрезков
        payload = _gamma_bits(len(ranges) + 1) + _gamma_bits(_zigzag(int(ranges[0][0])) + 1)  # нагрузка = гамма-код(числа отрезков + 1) + гамма-код(зигзаг(lo0) + 1)
        payload += _gamma_bits(int(ranges[0][1]) - int(ranges[0][0]) + 1)  # нагрузка += гамма-код(hi0 − lo0 + 1)
        for prev, curr in zip(ranges, ranges[1:]):  # по соседним парам (prev, curr)
            payload += _gamma_bits(int(curr[0]) - int(prev[1]))  # нагрузка += гамма-код(lo_j − hi_(j−1))
            payload += _gamma_bits(int(curr[1]) - int(curr[0]) + 1)  # нагрузка += гамма-код(hi_j − lo_j + 1)
        return TOKEN_BITS + payload + sum(_expr_bits(a, expert_indices) for a in args)  # 6 битов + нагрузка + Σ битов аргументов
    return TOKEN_BITS + sum(_expr_bits(a, expert_indices) for a in args)  # прочие узлы: 6 битов + Σ битов аргументов


def _program_bits(program: Mapping[str, Any], expert_indices: Mapping[str, int]) -> int:
    """Длина программы в битах: 6 + 4 для маски таблицы истинности, иначе длина выражения."""
    if program.get("venn_mask") is not None:  # программа задана маской таблицы истинности
        return TOKEN_BITS + 4  # 6 битов токена + 4 бита маски
    expr = program.get("expr")  # выражение программы
    if not isinstance(expr, dict): raise ValueError("missing program expression")  # выражение отсутствует — ошибка
    return _expr_bits(expr, expert_indices)  # длина выражения в битах


def _require_keys(
    obj: Mapping[str, Any], required: set[str], optional: set[str], where: str,  # объект, обязательные и необязательные ключи, имя места
) -> None:  # ничего не возвращает; при ошибке бросает исключение
    """Проверить ключи объекта: все обязательные присутствуют, никаких других, кроме optional.
    
        obj — проверяемый словарь, required — обязательные ключи, optional — допустимые
        дополнительные, where — имя места в сообщении об ошибке. Ничего не возвращает."""
    if not isinstance(obj, dict):  # объект обязан быть словарём
        raise ValueError(f"{where} must be an object")  # ошибка: не объект
    keys = set(obj)  # множество ключей объекта
    missing = required - keys  # отсутствующие ключи: required − keys
    extra = keys - required - optional  # лишние ключи: keys − required − optional
    if missing:  # есть отсутствующие
        raise ValueError(f"{where} missing keys: {sorted(missing)}")  # ошибка: нет обязательных ключей
    if extra:  # есть лишние
        raise ValueError(f"{where} has unknown keys: {sorted(extra)}")  # ошибка: есть неизвестные ключи


def _validate_expr_shape(
    expr: Any, *, arity: int, library: Mapping[str, Any], where: str,  # выражение, арность программы, библиотека, имя места
    depth: int = 0, node_counter: list[int] | None = None,  # глубина и счётчики узлов (одного эксперта и суммарный)
    aggregate_counter: list[int] | None = None,  # второй счётчик
) -> set[str]:  # возвращает множество имён экспертов, которых вызывает выражение
    """Проверить форму выражения; возвращает множество имён экспертов, которых оно вызывает.
    
        expr — узел; arity — число входов программы; library — библиотека экспертов;
        where — имя места; depth — глубина; node_counter — счётчик узлов одного эксперта,
        aggregate_counter — суммарный счётчик узлов всех экспертов (создаётся при первом вызове).
        Проверяются: глубина, ключи op/args/value, арности, диапазоны индексов и нагрузок."""
    if depth > MAX_AST_DEPTH:  # глубина не больше 128
        raise ValueError(f"{where}: AST nesting limit exceeded")  # ошибка: превышена глубина вложенности
    if not isinstance(expr, dict):  # выражение обязано быть словарём
        raise ValueError(f"{where}: expression must be an object")  # ошибка: выражение не объект
    _require_keys(expr, {"op", "args", "value"}, set(), where)  # ровно три ключа: op, args, value
    if node_counter is None:  # счётчик узлов не задан
        node_counter = [0]  # локальный счётчик узлов, начально 0
    node_counter[0] += 1  # счётчик узлов + 1
    if node_counter[0] > MAX_AST_NODES:  # предел узлов одного эксперта
        raise ValueError(f"{where}: AST node limit exceeded")  # ошибка: превышен предел узлов
    if aggregate_counter is not None:  # суммарный счётчик задан
        aggregate_counter[0] += 1  # суммарный счётчик + 1
        if aggregate_counter[0] > MAX_TOTAL_AST_NODES:  # превышен суммарный предел
            raise ValueError("total expert AST node limit exceeded")  # ошибка: превышен суммарный предел узлов
    op = expr["op"]  # имя операции
    args = expr["args"]  # аргументы
    value = expr["value"]  # полезная нагрузка
    if not isinstance(op, str) or op not in _AST_OP_ARITY:  # операция обязана быть строкой из таблицы арностей
        raise ValueError(f"{where}: unsupported AST operator {op!r}")  # ошибка: операция не поддерживается
    if not isinstance(args, list):  # аргументы обязаны быть списком
        raise ValueError(f"{where}: args must be an array")  # ошибка: args не массив
    required_arity = _AST_OP_ARITY[op]  # требуемая арность операции
    if required_arity is not None and len(args) != required_arity:  # число аргументов обязано совпадывать с арностью
        raise ValueError(f"{where}: {op} expects {required_arity} args")  # ошибка: неверное число аргументов
    if op == "var":  # переменная
        idx = _int(value, f"{where}.value")  # индекс переменной — целое
        if idx < 0 or idx >= arity:  # границы индекса: 0 ≤ idx < arity
            raise ValueError(f"{where}: variable index outside program arity")  # ошибка: индекс переменной вне арности
    elif op == "const":  # константа
        _int(value, f"{where}.value")  # константа — целое
    elif op == "call":  # вызов эксперта
        if not isinstance(value, str) or not value:  # имя эксперта — непустая строка
            raise ValueError(f"{where}: call requires expert name")  # ошибка: нужно имя эксперта
        callee = _program_from_library(library, value)  # запись вызываемого эксперта
        callee_arity = _int(callee["program"]["arity"], f"{where}.callee_arity")  # арность вызываемого
        if len(args) != callee_arity:  # число аргументов равно арности вызываемого
            raise ValueError(f"{where}: call arity does not match {value}")  # ошибка: арность вызова не совпадает
    elif op == "in_set":  # множество точек
        if not isinstance(value, list) or not value or len(value) > MAX_AST_PAYLOAD_ITEMS:  # непустой список не длиннее предела нагрузки
            raise ValueError(f"{where}: invalid/oversized in_set payload")  # ошибка: неверная или слишком большая нагрузка in_set
        points = [_int(v, f"{where}.value[]") for v in value]  # точки — целые
        if points != sorted(set(points)):  # точки строго возрастают: points = sorted(set(points))
            raise ValueError(f"{where}: in_set must be strictly increasing")  # ошибка: точки обязаны строго возрастать
    elif op == "in_ranges":  # отрезки
        if not isinstance(value, list) or not value or len(value) > MAX_AST_PAYLOAD_ITEMS:  # непустой список не длиннее предела нагрузки
            raise ValueError(f"{where}: invalid/oversized in_ranges payload")  # ошибка: неверная или слишком большая нагрузка in_ranges
        ranges: list[tuple[int, int]] = []  # список отрезков (lo, hi)
        for pair in value:  # по парам
            if not isinstance(pair, list) or len(pair) != 2:  # пара — список из двух элементов
                raise ValueError(f"{where}: malformed in_ranges pair")  # ошибка: неверная пара in_ranges
            lo = _int(pair[0], f"{where}.range.lo")  # нижняя граница
            hi = _int(pair[1], f"{where}.range.hi")  # верхняя граница
            if lo > hi:  # проверка порядка: lo ≤ hi
                raise ValueError(f"{where}: reversed in_ranges interval")  # ошибка: перевёрнутый отрезок
            ranges.append((lo, hi))  # добавить отрезок
        if any(a[1] + 1 >= b[0] for a, b in zip(ranges, ranges[1:])):  # для соседних hi_a + 1 < lo_b
            raise ValueError(f"{where}: in_ranges must be sorted, disjoint, non-adjacent")  # ошибка: отрезки не отсортированы, пересекаются или соприкасаются
    elif op == "anf":  # алгебраическая нормальная форма
        if not isinstance(value, list) or len(value) != 2:  # нагрузка — список из двух элементов
            raise ValueError(f"{where}: malformed anf payload")  # ошибка: неверная нагрузка anf
        n_vars = _int(value[0], f"{where}.anf.n_vars")  # число переменных
        coeffs = value[1]  # коэффициенты
        if n_vars != arity or not isinstance(coeffs, list) or len(coeffs) > MAX_AST_PAYLOAD_ITEMS:  # n_vars = arity, коэффициенты — список в пределах нагрузки
            raise ValueError(f"{where}: invalid anf dimensions")  # ошибка: неверные размеры anf
        for pair in coeffs:  # по парам коэффициентов
            if not isinstance(pair, list) or len(pair) != 2:  # пара — список из двух элементов
                raise ValueError(f"{where}: malformed anf coefficient")  # ошибка: неверный коэффициент anf
            subset = _int(pair[0], f"{where}.anf.subset")  # номер подмножества
            _int(pair[1], f"{where}.anf.coefficient")  # коэффициент — целое
            if subset < 0 or subset >= (1 << n_vars):  # диапазон подмножества: 0 ≤ subset < 2^n_vars
                raise ValueError(f"{where}: anf subset outside arity")  # ошибка: подмножество вне арности
    elif op == "poly":  # многочлен одной переменной
        if not isinstance(value, list) or not value or len(value) > MAX_AST_PAYLOAD_ITEMS:  # непустой список коэффициентов в пределах нагрузки
            raise ValueError(f"{where}: invalid/oversized poly payload")  # ошибка: неверная или слишком большая нагрузка poly
        for coefficient in value:  # по коэффициентам
            _int(coefficient, f"{where}.poly.coefficient")  # коэффициент — целое
    elif op == "poly_nd":  # многочлен нескольких переменных
        if not isinstance(value, list) or len(value) > MAX_AST_PAYLOAD_ITEMS:  # список термов в пределах нагрузки
            raise ValueError(f"{where}: invalid/oversized poly_nd payload")  # ошибка: неверная или слишком большая нагрузка poly_nd
        for term in value:  # по термам
            if not isinstance(term, list) or len(term) != 2:  # терм — список из двух элементов
                raise ValueError(f"{where}: malformed poly_nd term")  # ошибка: неверный терм poly_nd
            _int(term[0], f"{where}.poly_nd.coefficient")  # коэффициент — целое
            powers = term[1]  # показатели
            if not isinstance(powers, list) or len(powers) != arity:  # число показателей равно арности
                raise ValueError(f"{where}: poly_nd power arity mismatch")  # ошибка: число показателей не равно арности
            for power in powers:  # по показателям
                p = _int(power, f"{where}.poly_nd.power")  # показатель — целое
                if p < 0 or p > 100:  # диапазон показателя: 0 ≤ p ≤ 100
                    raise ValueError(f"{where}: poly_nd power outside [0,100]")  # ошибка: показатель вне [0, 100]
    else:  # прочие операции
        if value is not None:  # нагрузка должна отсутствовать
            raise ValueError(f"{where}: operator {op} must not have a payload")  # ошибка: у операции не должно быть нагрузки
    calls: set[str] = {value} if op == "call" else set()  # имена вызываемых: {value} для call, иначе пусто
    for index, child in enumerate(args):  # по аргументам с номером index
        calls.update(_validate_expr_shape(  # добавить имена, вызываемые дочерним узлом
            child, arity=arity, library=library,  # дочерний узел с той же арностью и библиотекой
            where=f"{where}.args[{index}]", depth=depth + 1,  # имя места и глубина + 1
            node_counter=node_counter, aggregate_counter=aggregate_counter,  # те же счётчики узлов
        ))
    return calls  # вернуть множество имён вызываемых экспертов


def _validate_library(library: Any) -> tuple[dict[str, int], dict[str, set[str]]]:
    """Проверить библиотеку экспертов; возвращает (имя -> индекс, имя -> множество вызываемых).
    
        Для каждого эксперта: шесть обязательных ключей, непустое семейство, корректная
        область и valid_rule, программа с ключами arity/expr/venn_mask, форма выражения,
        неотрицательные expert_index и code_length. Затем: индексы уникальны и образуют
        0..n-1, все вызовы ведут в библиотеку, code_length = длина AST в битах, граф вызовов ацикличен."""
    if not isinstance(library, dict) or len(library) > MAX_EXPERTS:  # библиотека — словарь не длиннее предела экспертов
        raise ValueError("expert_library must be an object within expert limit")  # ошибка: библиотека не объект или слишком велика
    indices: dict[str, int] = {}  # индексы экспертов по именам
    graph: dict[str, set[str]] = {}  # граф вызовов: имя -> множество вызываемых
    aggregate_ast_nodes = [0]  # суммарный счётчик узлов, начально 0
    for name, item in library.items():  # по экспертам
        if not isinstance(name, str) or not name:  # имя — непустая строка
            raise ValueError("expert names must be non-empty strings")  # ошибка: имя эксперта пустое или не строка
        _require_keys(  # ровно шесть ключей записи эксперта
            item,  # проверяемая запись
            {"family", "program", "domains", "valid_rule", "code_length", "expert_index"},  # обязательные ключи
            set(), f"expert_library[{name}]",  # без необязательных ключей
        )
        if not isinstance(item["family"], str) or not item["family"]:  # семейство — непустая строка
            raise ValueError(f"{name}: family must be a non-empty string")  # ошибка: семейство не непустая строка
        valid_rule = item["valid_rule"]  # правило допустимости
        if not isinstance(valid_rule, str):  # правило — строка
            raise ValueError(f"{name}: valid_rule must be a string")  # ошибка: valid_rule не строка
# проверить метаданные области, не строя декартово произведение целиком
        dimensions = _validated_domain_dimensions(item["domains"], valid_rule)  # проверить метаданные области, не строя декартово произведение
        if not isinstance(item["program"], dict):  # программа — словарь
            raise ValueError(f"{name}: program must be an object")  # ошибка: программа не объект
        program = item["program"]  # программа эксперта
        _require_keys(program, {"arity", "expr", "venn_mask"}, set(), f"{name}.program")  # ровно три ключа: arity, expr, venn_mask
        arity = _int(program["arity"], f"{name}.program.arity")  # арность программы — целое
        if arity != len(dimensions):  # арность равна числу измерений
            raise ValueError(f"{name}: program arity/domain mismatch")  # ошибка: арность не совпадает с областью
        if program["venn_mask"] is not None:  # программа задана маской
            mask = _int(program["venn_mask"], f"{name}.venn_mask")  # маска — целое
            if arity != 2 or not 0 <= mask <= 15 or program["expr"] is not None:  # для маски: арность 2, 0 ≤ mask ≤ 15, выражение отсутствует
                raise ValueError(f"{name}: invalid BoolVenn program")  # ошибка: неверная программа BoolVenn
            dependencies: set[str] = set()  # у маски нет зависимостей
        else:  # иначе — программа с выражением
            if program["expr"] is None:  # выражение отсутствует
                raise ValueError(f"{name}: missing expression")  # ошибка: нет выражения
            dependencies = _validate_expr_shape(  # зависимости — имена экспертов, вызываемых выражением
                program["expr"], arity=arity, library=library,  # проверяемое выражение и арность
                where=f"{name}.expr", aggregate_counter=aggregate_ast_nodes,  # имя места и суммарный счётчик
            )
        idx = _int(item["expert_index"], f"{name}.expert_index")  # индекс эксперта — целое
        if idx < 0:  # индекс неотрицателен
            raise ValueError(f"{name}: expert_index must be non-negative")  # ошибка: expert_index отрицателен
        indices[name] = idx  # запомнить индекс эксперта
        graph[name] = dependencies  # зависимости эксперта в графе
        code_length = _int(item["code_length"], f"{name}.code_length")  # длина кода — целое
        if code_length < 0:  # длина кода неотрицательна
            raise ValueError(f"{name}: code_length must be non-negative")  # ошибка: code_length отрицательна
    if len(set(indices.values())) != len(indices):  # индексы различны
        raise ValueError("expert_index values must be unique")  # ошибка: индексы экспертов повторяются
    if set(indices.values()) != set(range(len(indices))):  # индексы образуют 0..n−1
        raise ValueError("expert_index values must be contiguous from zero")  # ошибка: индексы не идут подряд с нуля
    for name, item in library.items():  # по экспертам (второй проход)
        for dep in graph[name]:  # по вызываемым экспертам
            if dep not in library:  # вызываемый есть в библиотеке
                raise ValueError(f"{name}: call references unknown expert {dep}")  # ошибка: вызов неизвестного эксперта
        expected_bits = _program_bits(item["program"], indices)  # ожидаемая длина кода по AST
        if expected_bits != _int(item["code_length"], f"{name}.code_length"):  # совпадает с заявленной
            raise ValueError(f"{name}: code_length does not match AST")  # ошибка: code_length не совпадает с AST
# вызовы экспертов обязаны быть ациклическими: иначе результат конечного проверяющего
# зависел бы от глубины обхода, а не от замкнутой программы
    visiting: set[str] = set()  # эксперты в текущем обходе
    visited: set[str] = set()  # уже проверенные эксперты
    def visit(name: str) -> None:
        """Обход в глубину с пометками: эксперт в текущем обходе означает цикл вызовов."""
        if name in visiting:  # эксперт уже в текущем обходе — цикл
            raise ValueError(f"recursive expert call cycle includes {name}")  # ошибка: цикл рекурсивных вызовов через name
        if name in visited:  # уже проверен
            return  # выход без действий
        visiting.add(name)  # пометить как текущий
        for dep in graph[name]:  # по его зависимостям
            visit(dep)  # проверить зависимость
        visiting.remove(name)  # снять пометку текущего
        visited.add(name)  # пометить как проверенный
    for name in graph:  # по всем экспертам графа
        visit(name)  # проверить эксперта на циклы
    return indices, graph  # вернуть индексы экспертов и граф вызовов


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Хук разбора JSON: собрать словарь из пар ключ-значение, повтор ключа — ошибка."""
    result: dict[str, Any] = {}  # словарь результата
    for key, value in pairs:  # по парам ключ-значение
        if key in result:  # ключ уже встречался
            raise ValueError(f"duplicate JSON object key: {key}")  # ошибка: повторяющийся ключ JSON-объекта
        result[key] = value  # записать пару
    return result  # вернуть словарь


def _parse_artifact_source(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Прочитать артефакт из файла (не больше MAX_ARTIFACT_BYTES) или словаря; вернуть JSON-объект."""
    if isinstance(source, (str, Path)):  # источник — путь (строка или Path)
        p = Path(source)  # путь к файлу
        if p.stat().st_size > MAX_ARTIFACT_BYTES:  # размер файла больше предела
            raise ValueError("artifact exceeds size limit")  # ошибка: артефакт превышает предельный размер
        artifact = json.loads(  # разбор JSON файла
            p.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_object_keys,  # чтение UTF-8 с запретом повторяющихся ключей
        )
    else:  # иначе — словарь
        artifact = json.loads(  # разбор JSON
            _canonical_bytes(dict(source)).decode("utf-8"),  # канонические байты словаря как текст UTF-8
            object_pairs_hook=_reject_duplicate_object_keys,  # запрет повторяющихся ключей
        )
    if not isinstance(artifact, dict):  # корень не словарь
        raise ValueError("portable certificate root must be an object")  # ошибка: корень сертификата не объект
    return artifact  # вернуть артефакт


def _verify_artifact_hash(artifact: Mapping[str, Any]) -> str:
    """Сверить artifact_sha256 с SHA-256(канонический JSON артефакта без этого поля); вернуть хеш."""
    claimed_digest = artifact.get("artifact_sha256")  # заявленный хеш артефакта
    if not isinstance(claimed_digest, str) or len(claimed_digest) != 64:  # строка ровно из 64 символов
        raise ValueError("artifact_sha256 must be a 64-character hex digest")  # ошибка: artifact_sha256 — не 64-символьная шестнадцатеричная строка
    try:  # проверка шестнадцатеричности
        int(claimed_digest, 16)  # перевод hex-строки в целое
    except ValueError as exc:  # значение не шестнадцатеричное
        raise ValueError("artifact_sha256 is not hexadecimal") from exc  # ошибка: artifact_sha256 не шестнадцатеричный
    unsigned = dict(artifact)  # копия артефакта
    unsigned.pop("artifact_sha256", None)  # удалить поле с хешем
    actual = _sha256(unsigned)  # фактический хеш: SHA-256(артефакт без artifact_sha256)
    if claimed_digest != actual:  # хеши не совпадают
        raise ValueError("artifact SHA-256 mismatch")  # ошибка: хеш артефакта не совпадает
    return claimed_digest  # вернуть заявленный хеш


def _load_json_object(source: str | Path | Mapping[str, Any], *, limit: int, where: str) -> dict[str, Any]:
    """Прочитать JSON-объект из файла (не больше limit байтов) или словаря; where — имя места для ошибок."""
    if isinstance(source, (str, Path)):  # источник — путь
        path = Path(source)  # путь
        if path.stat().st_size > limit:  # размер больше limit
            raise ValueError(f"{where} exceeds size limit")  # ошибка: превышен предельный размер
        value = json.loads(  # разбор JSON файла
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_object_keys,  # чтение UTF-8 с запретом повторяющихся ключей
        )
    else:  # иначе — словарь
        value = json.loads(  # разбор JSON
            _canonical_bytes(dict(source)).decode("utf-8"),  # канонические байты словаря как текст UTF-8
            object_pairs_hook=_reject_duplicate_object_keys,  # запрет повторяющихся ключей
        )
    if not isinstance(value, dict):  # значение не словарь
        raise ValueError(f"{where} must be a JSON object")  # ошибка: не JSON-объект
    return value  # вернуть объект


def _rsa_key_id(n_hex: str, e: int) -> str:
    """Идентификатор ключа: SHA-256 от канонического JSON {algorithm, e, n_hex малыми буквами}."""
    return _sha256({  # идентификатор ключа = SHA-256(канонический JSON {algorithm, e, n_hex})
        "algorithm": "RSA_PKCS1_V1_5_SHA256",  # алгоритм подписи
        "e": e,  # открытая экспонента
        "n_hex": n_hex.lower(),  # модуль малыми буквами
    })


def _signature_message(artifact_digest: str) -> bytes:
    """Сообщение для подписи: служебная строка с нулевым байтом + 32 байта хеша артефакта."""
    return b"HAGI-PORTABLE-CERT-SIGNATURE-V1\0" + bytes.fromhex(artifact_digest)  # сообщение для подписи: служебная строка с нулевым байтом + байты хеша артефакта


def verify_portable_signature(
    artifact_source: str | Path | Mapping[str, Any],  # артефакт: путь или словарь
    signature_source: str | Path | Mapping[str, Any],  # подпись: путь или словарь
    trusted_public_key: str | Path | Mapping[str, Any],  # доверенный открытый ключ: путь или словарь
) -> dict[str, Any]:  # возвращает словарь с признаком valid и итогом проверки
    """Проверка отделённой подписи RSA/SHA-256 только стандартной библиотекой Python.
    
    artifact_source — артефакт (путь или словарь), signature_source — подпись,
    trusted_public_key — доверенный открытый ключ. Возвращает словарь: при успехе
    valid = True, algorithm, key_id, artifact_sha256, trust_basis; при любой ошибке
    valid = False и error."""
    try:  # любая ошибка -> valid = False
        artifact = _parse_artifact_source(artifact_source)  # разобрать артефакт
        artifact_result = verify_portable_certificate(artifact)  # проверить сертификат
        if not artifact_result.get("valid"):  # сертификат недействителен
            raise ValueError(f"certificate invalid: {artifact_result.get('error')}")  # ошибка: сертификат недействителен
        artifact_digest = _verify_artifact_hash(artifact)  # хеш артефакта

        signature = _load_json_object(  # прочитать файл подписи (не больше предела)
            signature_source, limit=MAX_SIGNATURE_BYTES, where="signature file",  # источник подписи, предел и имя места
        )
        _require_keys(  # ровно пять ключей подписи
            signature,  # проверяемый объект
            {"format", "algorithm", "key_id", "artifact_sha256", "signature_b64"},  # обязательные ключи
            set(), "signature",  # имя места
        )
        if signature["format"] != "HAGI-PORTABLE-SIG-V1":  # формат подписи
            raise ValueError("unsupported detached signature format")  # ошибка: формат отделённой подписи не поддерживается
        if signature["algorithm"] != "RSA_PKCS1_V1_5_SHA256":  # алгоритм подписи
            raise ValueError("unsupported signature algorithm")  # ошибка: алгоритм подписи не поддерживается
        if signature["artifact_sha256"] != artifact_digest:  # хеш в подписи совпадает с хешем артефакта
            raise ValueError("signature is for a different artifact digest")  # ошибка: подпись для другого хеша артефакта
        if not isinstance(signature["signature_b64"], str):  # подпись — строка
            raise ValueError("signature_b64 must be a string")  # ошибка: signature_b64 не строка
        try:  # блок с перехватом ошибки декодирования
            sig_bytes = base64.b64decode(signature["signature_b64"], validate=True)  # декодирование base64 со строгой проверкой
        except Exception as exc:  # ошибка декодирования
            raise ValueError("signature_b64 is malformed") from exc  # ошибка: signature_b64 повреждена

        public_key = _load_json_object(  # прочитать открытый ключ (не больше предела)
            trusted_public_key, limit=MAX_SIGNATURE_BYTES, where="trusted public key",  # источник ключа, предел и имя места
        )
        _require_keys(  # ровно пять ключей открытого ключа
            public_key,  # ключ
            {"format", "algorithm", "key_id", "n_hex", "e"},  # обязательные ключи
            set(), "public key",  # имя места
        )
        if public_key["format"] != "HAGI-PORTABLE-PUBKEY-V1":  # формат ключа
            raise ValueError("unsupported public key format")  # ошибка: формат открытого ключа не поддерживается
        if public_key["algorithm"] != "RSA_PKCS1_V1_5_SHA256":  # алгоритм ключа
            raise ValueError("unsupported public key algorithm")  # ошибка: алгоритм открытого ключа не поддерживается
        n_hex = public_key["n_hex"]  # модуль как hex-строка
        if not isinstance(n_hex, str) or not n_hex or len(n_hex) > 2048:  # непустая строка не длиннее 2048 символов
            raise ValueError("public modulus must be a bounded hexadecimal string")  # ошибка: модуль не ограниченная hex-строка
        try:  # блок с перехватом ошибки разбора модуля
            n = int(n_hex, 16)  # перевод hex-строки в целое
        except ValueError as exc:  # не шестнадцатеричное
            raise ValueError("public modulus is not hexadecimal") from exc  # ошибка: модуль не шестнадцатеричный
        e = _int(public_key["e"], "public exponent")  # экспонента — целое
        if n.bit_length() < 2048 or n.bit_length() > 8192:  # проверка модуля: 2048 ≤ bit_length(n) ≤ 8192
            raise ValueError("RSA modulus must be between 2048 and 8192 bits")  # ошибка: модуль вне 2048..8192 битов
        if e < 3 or e >= n or e % 2 == 0:  # e нечётно, e ≥ 3, e < n
            raise ValueError("invalid RSA public exponent")  # ошибка: неверная открытая экспонента
        key_id = _rsa_key_id(n_hex, e)  # идентификатор ключа
        if public_key["key_id"] != key_id or signature["key_id"] != key_id:  # совпадает с key_id в ключе и в подписи
            raise ValueError("signature/public-key key_id mismatch")  # ошибка: key_id ключа и подписи не совпадают

        k = (n.bit_length() + 7) // 8  # длина модуля в байтах: k = ceil(bit_length(n) / 8)
        if len(sig_bytes) != k:  # длина подписи равна k
            raise ValueError("RSA signature length does not match modulus")  # ошибка: длина подписи не равна длине модуля
        sig_int = int.from_bytes(sig_bytes, "big")  # подпись как целое: sig_int = int(байты, big)
        if sig_int >= n:  # проверка диапазона: sig_int < n
            raise ValueError("RSA signature representative out of range")  # ошибка: подпись вне диапазона
        encoded = pow(sig_int, e, n).to_bytes(k, "big")  # расшифровка: encoded = sig_int^e mod n в k байтах
        digest_info_prefix = bytes.fromhex("3031300d060960864801650304020105000420")  # префикс DigestInfo для SHA-256
        digest_info = digest_info_prefix + hashlib.sha256(_signature_message(artifact_digest)).digest()  # DigestInfo = префикс + SHA-256(сообщение)
        if len(encoded) < len(digest_info) + 11:  # длина не меньше len(DigestInfo) + 11
            raise ValueError("RSA encoded message too short")  # ошибка: расшифрованное сообщение слишком короткое
        padding_len = k - len(digest_info) - 3  # длина заполнения: k − len(DigestInfo) − 3
        expected = b"\x00\x01" + (b"\xff" * padding_len) + b"\x00" + digest_info  # эталон: 00 01 || FF·padding_len || 00 || DigestInfo
        if padding_len < 8 or not hmac.compare_digest(encoded, expected):  # padding_len ≥ 8 и побайтное равенство за постоянное время
            raise ValueError("detached RSA signature verification failed")  # ошибка: проверка отделённой подписи RSA не прошла

        return {  # вернуть словарь итога проверки
            "valid": True,  # признак успешной проверки
            "algorithm": "RSA_PKCS1_V1_5_SHA256",  # алгоритм
            "key_id": key_id,  # идентификатор ключа
            "artifact_sha256": artifact_digest,  # хеш артефакта
            "trust_basis": "caller-supplied public key; publisher identity requires out-of-band trust",  # основание доверия: ключ дан вызывающим
        }
    except Exception as exc:  # любое исключение
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}  # вернуть valid = False с текстом ошибки


def verify_portable_certificate(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Проверка сертификата V1 (старого) или V2 только стандартной библиотекой.
    
    source — путь или словарь. Возвращает словарь: valid, format, format_version,
    ast_schema, verifier_profile, artifact_sha256, tasks, proven_exact_finite, reused,
    held, domain_rows_checked, results (итог по каждой задаче), scope; при ошибке —
    valid = False и error. V2 обязан нести явный хеш схемы AST и профиль
    проверяющего; V1 читается по фиксированной схеме hagi.expr.v1 ради миграции."""
    try:  # начало блока с перехватом всех ошибок
        artifact = _parse_artifact_source(source)  # разобрать артефакт
        artifact_format = artifact.get("format")  # формат артефакта
        if artifact_format == FORMAT_V1:  # формат версии 1
            expected_keys = {  # обязательные ключи V1
                "format", "format_version", "hagi_version", "scope_notice",  # имена обязательных ключей: format, format_version, hagi_version, scope_notice
                "model_version", "total_commits", "task_count", "expert_library",  # имена обязательных ключей: model_version, total_commits, task_count, expert_library
                "tasks", "artifact_sha256",  # имена обязательных ключей: tasks, artifact_sha256
            }
            _require_keys(artifact, expected_keys, set(), "V1 artifact")  # ровно эти ключи
            version = _int(artifact["format_version"], "format_version")  # номер версии — целое
            if version != 1:  # номер версии равен 1
                raise ValueError("unsupported V1 format version")  # ошибка: неподдерживаемая версия формата V1
            ast_schema = AST_SCHEMA  # схема выражений V1 — фиксированная
            verifier_profile = "hagi.portable-verifier.v1-legacy"  # профиль проверяющего V1 — устаревший, задан здесь
        elif artifact_format == FORMAT:  # формат версии 2
            expected_keys = {  # обязательные ключи V2
                "format", "format_version", "ast_schema", "ast_schema_sha256",  # имена обязательных ключей: format, format_version, ast_schema, ast_schema_sha256
                "verifier_profile", "hagi_version", "scope_notice", "model_version",  # имена обязательных ключей: verifier_profile, hagi_version, scope_notice, model_version
                "total_commits", "task_count", "expert_library", "tasks",  # имена обязательных ключей: total_commits, task_count, expert_library, tasks
                "artifact_sha256",  # имена обязательных ключей: artifact_sha256
            }
            _require_keys(artifact, expected_keys, set(), "V2 artifact")  # ровно эти ключи
            version = _int(artifact["format_version"], "format_version")  # номер версии — целое
            if version != FORMAT_VERSION:  # номер версии равен FORMAT_VERSION = 2
                raise ValueError("unsupported V2 format version")  # ошибка: неподдерживаемая версия формата V2
            ast_schema = artifact["ast_schema"]  # схема выражений из артефакта
            if ast_schema != AST_SCHEMA:  # схема совпадает с вшитой
                raise ValueError(f"unsupported AST schema: {ast_schema!r}")  # ошибка: схема AST не поддерживается
            if artifact["ast_schema_sha256"] != AST_SCHEMA_SHA256:  # хеш схемы совпадает с вшитым
                raise ValueError("AST schema digest mismatch")  # ошибка: хеш схемы AST не совпадает
            verifier_profile = artifact["verifier_profile"]  # профиль проверяющего из артефакта
            if verifier_profile != VERIFIER_PROFILE:  # профиль совпадает с вшитым
                raise ValueError(f"unsupported verifier profile: {verifier_profile!r}")  # ошибка: профиль проверяющего не поддерживается
        else:  # иначе формат неизвестен
            raise ValueError("wrong portable certificate format")  # ошибка: неверный формат сертификата

        claimed_digest = _verify_artifact_hash(artifact)  # сверить хеш артефакта
        if not isinstance(artifact["hagi_version"], str) or not artifact["hagi_version"]:  # hagi_version — непустая строка
            raise ValueError("hagi_version must be a non-empty string")  # ошибка: hagi_version не непустая строка
        if not isinstance(artifact["scope_notice"], str) or not artifact["scope_notice"]:  # scope_notice — непустая строка
            raise ValueError("scope_notice must be a non-empty string")  # ошибка: scope_notice не непустая строка
        model_version = _int(artifact["model_version"], "model_version")  # версия модели — целое
        total_commits = _int(artifact["total_commits"], "total_commits")  # число коммитов — целое
        task_count = _int(artifact["task_count"], "task_count")  # число задач — целое
        if min(model_version, total_commits, task_count) < 0:  # все три неотрицательны: min(...) ≥ 0
            raise ValueError("model_version, total_commits and task_count must be non-negative")  # ошибка: model_version, total_commits, task_count обязаны быть неотрицательными
        if task_count > MAX_TASKS:  # число задач не больше предела
            raise ValueError("task_count exceeds verifier task limit")  # ошибка: task_count превышает предел

        library = artifact["expert_library"]  # библиотека экспертов
        tasks = artifact["tasks"]  # список задач
        if not isinstance(tasks, list) or len(tasks) != task_count:  # tasks — список длины task_count
            raise ValueError("task_count mismatch")  # ошибка: task_count не совпадает с длиной tasks
        expert_indices, _call_graph = _validate_library(library)  # проверить библиотеку; индексы экспертов и граф вызовов (граф далее не нужен)

        seen_names: set[str] = set()  # имена уже разобранных задач
        proof_rows = 0  # число проверенных строк области, начально 0
        total_domain_rows = 0  # суммарное число строк области, начально 0
        proved = 0  # число доказанных задач, начально 0
        holds = 0  # число задач в статусе HOLD, начально 0
        reused = 0  # число задач REUSE, начально 0
        per_task = []  # итог по каждой задаче
        for task in tasks:  # по задачам
            if not isinstance(task, dict):  # задача — словарь
                raise ValueError("task entry must be an object")  # ошибка: задача не объект
            _require_keys(  # проверить ключи задачи
                task,  # проверяемая задача
                {  # ровно тринадцать обязательных ключей
                    "name", "family", "arity", "domains", "valid_rule",  # имена обязательных ключей задачи: name, family, arity, domains, valid_rule
                    "domain_size", "expected_outputs", "labels_sha256", "action",  # имена обязательных ключей задачи: domain_size, expected_outputs, labels_sha256, action
                    "program_source", "program_bits", "reported_code_length", "reason",  # имена обязательных ключей задачи: program_source, program_bits, reported_code_length, reason
                },
                set(), "task",  # имя места
            )
            name = task["name"]  # имя задачи
            if not isinstance(name, str) or not name or name in seen_names:  # непустая строка, ещё не встречалась
                raise ValueError("empty or duplicate task name")  # ошибка: имя пустое или повторяется
            seen_names.add(name)  # запомнить имя
            if not isinstance(task["family"], str) or not task["family"]:  # семейство — непустая строка
                raise ValueError(f"{name}: family must be a non-empty string")  # ошибка: семейство не непустая строка
            arity = _int(task["arity"], f"{name}.arity")  # арность — целое
            domains = task["domains"]  # области по измерениям
            if not isinstance(domains, list) or len(domains) != arity:  # число измерений равно арности
                raise ValueError(f"{name}: domain arity mismatch")  # ошибка: арность не совпадает с областью
            valid_rule = task["valid_rule"]  # правило допустимости
            if not isinstance(valid_rule, str):  # правило — строка
                raise ValueError(f"{name}: valid_rule must be a string")  # ошибка: valid_rule не строка
            rows = _domain_rows(domains, valid_rule)  # строки области (декартово произведение, отфильтрованное правилом)
            domain_size = _int(task["domain_size"], f"{name}.domain_size")  # заявленный размер области — целое
            if len(rows) != domain_size:  # сравнение: len(rows) = domain_size
                raise ValueError(f"{name}: domain_size mismatch")  # ошибка: domain_size не совпадает
            total_domain_rows += len(rows)  # суммарно: total_domain_rows += len(rows)
            if total_domain_rows > MAX_TOTAL_DOMAIN_ROWS:  # не больше суммарного предела
                raise ValueError("total declared domain rows exceed verifier limit")  # ошибка: суммарное число строк области превышает предел

            outputs = task["expected_outputs"]  # ожидаемые ответы
            if not isinstance(outputs, list) or len(outputs) != len(rows):  # число ответов равно числу строк
                raise ValueError(f"{name}: incomplete expected_outputs")  # ошибка: expected_outputs неполны
            outputs = [_int(y, f"{name}.expected_outputs[]") for y in outputs]  # ответы — целые
            label_digest = _sha256({"inputs": [list(r) for r in rows], "outputs": outputs})  # хеш меток: SHA-256({inputs: rows, outputs})
            if label_digest != task["labels_sha256"]:  # совпадает с labels_sha256
                raise ValueError(f"{name}: labels SHA-256 mismatch")  # ошибка: хеш меток не совпадает

            action = task["action"]  # действие по задаче
            if not isinstance(action, str):  # действие — строка
                raise ValueError(f"{name}: action must be a string")  # ошибка: action не строка
            source_name = task["program_source"]  # имя эксперта — источника программы
            program_bits_value = task["program_bits"]  # заявленная длина программы в битах
            reported_bits = task["reported_code_length"]  # заявленная вызывающим длина кода
            if program_bits_value is not None:  # длина программы в битах задана
                _int(program_bits_value, f"{name}.program_bits")  # если задана — целое
            if reported_bits is not None:  # заявленная длина кода задана
                _int(reported_bits, f"{name}.reported_code_length")  # если задана — целое
            if task["reason"] is not None and not isinstance(task["reason"], str):  # reason — строка или None
                raise ValueError(f"{name}: reason must be a string or null")  # ошибка: reason не строка и не null

            if action in {"COMMIT", "REUSE"}:  # задача с программой (своя или переиспользованная)
                if not isinstance(source_name, str) or not source_name:  # имя источника — непустая строка
                    raise ValueError(f"{name}: committed/reused task lacks program source")  # ошибка: нет источника программы
                if action == "COMMIT" and source_name != name:  # COMMIT ссылается только на своего эксперта
                    raise ValueError(f"{name}: COMMIT must point to its own expert")  # ошибка: COMMIT обязан указывать на своего эксперта
                if source_name not in library:  # источник есть в библиотеке
                    raise ValueError(f"{name}: source expert {source_name} is missing")  # ошибка: эксперт-источник отсутствует
                source = _program_from_library(library, source_name)  # запись эксперта-источника
                for xs in rows:  # по всем строкам области
                    if not _supports(source, xs):  # источник поддерживает вход
                        raise ValueError(f"{name}: source expert {source_name} does not support {xs}")  # ошибка: эксперт-источник не поддерживает вход
                program = source["program"]  # программа источника
                total_eval_work = 0  # суммарная работа вычисления, начально 0
                for i, (xs, expected) in enumerate(zip(rows, outputs)):  # по парам (вход, ожидаемый ответ) с номером i
                    row_work = [0]  # работа одной строки, счётчик в списке
                    # имя эксперта-источника кладётся в стек вызовов, чтобы прямой
                    # вызов самого себя отвергался так же, как и косвенные циклы
                    got = _eval_program(program, xs, library, (source_name,), row_work)  # вычислить программу; стек вызовов начинается с имени источника
                    total_eval_work += row_work[0]  # итоговая работа: total_eval_work += row_work[0]
                    if total_eval_work > MAX_EVAL_WORK_PER_TASK:  # не больше предела на задачу
                        raise ValueError(f"{name}: evaluation work limit exceeded")  # ошибка: превышен предел работы вычисления
                    if got != expected:  # ответ совпал с ожидаемым
                        raise ValueError(  # ошибка: контрпример — строка i, вход, ожидаемый и полученный ответ
                            f"{name}: counterexample at row {i}, input={xs}, "  # текст ошибки: контрпример в строке
                            f"expected={expected}, got={got}"  # текст ошибки: ожидаемый и полученный ответы
                        )
                recomputed_bits = _program_bits(program, expert_indices)  # пересчитанная длина программы в битах
                if recomputed_bits != _int(program_bits_value, f"{name}.program_bits"):  # совпадает с заявленной
                    raise ValueError(f"{name}: program bit-length mismatch")  # ошибка: длина программы в битах не совпадает
                proof_rows += len(rows)  # счётчик проверенных строк: proof_rows += len(rows)
                proved += 1  # счётчик доказанных задач: proved += 1
                if action == "REUSE":  # задача переиспользована
                    reused += 1  # счётчик переиспользованных: reused += 1
                per_task.append({  # добавить итог задачи
                    "task": name, "status": "PROVEN_EXACT_FINITE",  # имя и статус «доказано точно на конечной области»
                    "checked": len(rows), "program_source": source_name,  # число проверенных строк и источник программы
                    "program_bits": recomputed_bits,  # пересчитанная длина программы
                })
            elif action == "HOLD":  # задача без программы (HOLD)
                if source_name is not None or program_bits_value is not None:  # у HOLD не должно быть источника и длины
                    raise ValueError(f"{name}: HOLD must not contain a program source or bit length")  # ошибка: HOLD не содержит источник программы или длину
                holds += 1  # счётчик задач в HOLD: holds += 1
                per_task.append({"task": name, "status": "HOLD_NO_PROGRAM", "checked": 0})  # итог: HOLD без программы, проверено 0 строк
            else:  # иначе действие неизвестно
                raise ValueError(f"{name}: unsupported action {action}")  # ошибка: действие не поддерживается

        return {  # вернуть итог проверки
            "valid": True,  # признак успешной проверки
            "format": artifact_format,  # формат артефакта
            "format_version": version,  # номер версии формата
            "ast_schema": ast_schema,  # схема выражений
            "verifier_profile": verifier_profile,  # профиль проверяющего
            "artifact_sha256": claimed_digest,  # хеш артефакта
            "tasks": len(tasks),  # число задач
            "proven_exact_finite": proved,  # число доказанных задач
            "reused": reused,  # число переиспользованных
            "held": holds,  # число задач в HOLD
            "domain_rows_checked": proof_rows,  # число проверенных строк области
            "results": per_task,  # итоги по задачам
            "scope": "finite-domain agreement with embedded labels; provenance not authenticated",  # область действия: совпадение с вложенными метками, происхождение не проверено
        }
    except Exception as exc:  # любое исключение
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}  # вернуть valid = False с текстом ошибки


def upgrade_portable_certificate_v1_to_v2(
    source: str | Path | Mapping[str, Any], destination: str | Path | None = None,  # аргументы: источник и путь назначения (None — не записывать)
) -> dict[str, Any]:  # возвращает итог проверки с полем artifact — новый артефакт
    """Миграция артефакта V1 в V2 без изменения меток и программ.
    
    source — путь или словарь, destination — путь записи (None — не записывать).
    Добавляются format, format_version, ast_schema, ast_schema_sha256,
    verifier_profile и новый artifact_sha256. Возвращает {artifact, ...итог проверки},
    при записи — с path и bytes. Происхождение меток не проверяется: хеш V2 —
    просто контрольная сумма целостности без ключа."""
    artifact = _parse_artifact_source(source)  # разобрать артефакт
    checked = verify_portable_certificate(artifact)  # проверить исходный сертификат
    if not checked.get("valid"):  # исходный недействителен
        raise ValueError(f"cannot migrate invalid portable certificate: {checked.get('error')}")  # ошибка: нельзя мигрировать недействительный сертификат
    if artifact.get("format") == FORMAT:  # уже V2 — берётся как есть
        upgraded = artifact  # артефакт уже в формате V2 — берётся как есть
    elif artifact.get("format") == FORMAT_V1:  # исходный V1
        upgraded = dict(artifact)  # копия артефакта
        upgraded.pop("artifact_sha256", None)  # удалить старый хеш
        upgraded["format"] = FORMAT  # записать формат: format = FORMAT
        upgraded["format_version"] = FORMAT_VERSION  # записать номер версии формата: format_version = FORMAT_VERSION
        upgraded["ast_schema"] = AST_SCHEMA  # явная схема AST
        upgraded["ast_schema_sha256"] = AST_SCHEMA_SHA256  # хеш схемы AST
        upgraded["verifier_profile"] = VERIFIER_PROFILE  # явный профиль проверяющего
        upgraded["artifact_sha256"] = _sha256(upgraded)  # новый хеш: artifact_sha256 = SHA-256(артефакт без хеша)
    else:  # иначе версия не поддерживается
        raise ValueError("unsupported source certificate version")  # ошибка: версия исходного сертификата не поддерживается
    result = verify_portable_certificate(upgraded)  # проверить новый артефакт
    if not result.get("valid"):  # новый недействителен
        raise ValueError(f"upgraded portable certificate failed verification: {result.get('error')}")  # ошибка: мигрированный сертификат не прошёл проверку
    if destination is not None:  # путь назначения задан
        p = Path(destination)  # путь к файлу
        p.parent.mkdir(parents=True, exist_ok=True)  # создать родительский каталог
        p.write_text(json.dumps(  # записать JSON
            upgraded, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,  # с отступом 2, ключи по алфавиту, без NaN
        ), encoding="utf-8")  # кодирование UTF-8
        result["path"] = str(p)  # итог: путь к файлу
        result["bytes"] = p.stat().st_size  # итог: размер файла в байтах
    return {"artifact": upgraded, **result}  # вернуть артефакт вместе с итогом

def main(argv: list[str] | None = None) -> int:
    """Точка входа командной строки: проверка сертификата, миграция V1 -> V2 или проверка подписи.
    
        argv — аргументы (None — из sys.argv). Возвращает 0, если итог valid, иначе 2.
        Итог печатается в stdout как JSON."""
    parser = argparse.ArgumentParser(description="Verify a HAGI portable finite-domain certificate")  # создать разборщик аргументов
    parser.add_argument("certificate", help="path to portable certificate JSON")  # позиционный аргумент — путь к сертификату JSON
    parser.add_argument("--upgrade-to-v2", metavar="OUTPUT", help="migrate a valid V1 artifact to V2")  # именованный аргумент — путь результата миграции V1 -> V2
    parser.add_argument("--signature", help="detached RSA/SHA-256 signature JSON")  # именованный аргумент — файл отделённой подписи
    parser.add_argument("--trusted-public-key", help="trusted public key JSON for signature verification")  # именованный аргумент — файл доверенного открытого ключа
    args = parser.parse_args(argv)  # разобрать аргументы
    if args.upgrade_to_v2:  # задана миграция
        try:  # блок с перехватом ошибки
            migrated = upgrade_portable_certificate_v1_to_v2(args.certificate, args.upgrade_to_v2)  # мигрировать сертификат и записать по указанному пути
            result = {k: v for k, v in migrated.items() if k != "artifact"}  # итог без самого артефакта
        except Exception as exc:  # любая ошибка
            print(json.dumps({"valid": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))  # напечатать итог с ошибкой
            return 2  # код выхода 2 — ошибка
    else:  # иначе — простая проверка
        result = verify_portable_certificate(args.certificate)  # проверить сертификат
    if bool(args.signature) != bool(args.trusted_public_key):  # подпись и ключ заданы не оба сразу
        result = {"valid": False, "error": "--signature and --trusted-public-key must be supplied together"}  # ошибка: --signature и --trusted-public-key нужно задавать вместе
    elif args.signature and args.trusted_public_key:  # подпись и ключ заданы
        signature_result = verify_portable_signature(  # проверить подпись
            args.certificate, args.signature, args.trusted_public_key,  # артефакт, подпись и ключ
        )
        result = {**result, "signature_verification": signature_result}  # добавить итог проверки подписи
        result["valid"] = bool(result.get("valid")) and bool(signature_result.get("valid"))  # valid = проверка сертификата и проверка подписи
    print(json.dumps(result, indent=2, ensure_ascii=False))  # напечатать итог JSON
    return 0 if result.get("valid") else 2  # код выхода 0, если valid, иначе 2


if __name__ == "__main__":  # pragma: no cover  # точка входа при запуске скрипта
    raise SystemExit(main())  # завершить процесс с кодом main()
