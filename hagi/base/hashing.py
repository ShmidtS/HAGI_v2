"""Общие хеш-функции и нормализация ключей (единое место для всех модулей).

Все функции детерминированы и не зависят от платформы: blake2b с фиксированной
длиной, младший байт первым (little-endian), без знака.
"""
from __future__ import annotations  # аннотации типов не вычисляются при импорте

import hashlib  # хеш-функция blake2b


def hash_token_u64(token: str) -> int:
    """Целое 64 бита из хеша слова: blake2b(UTF-8(token), 8 байт), little-endian, без знака.

    Формула: v = Σ_{i=0..7} b_i · 256^i, где b — байты хеша."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little", signed=False)  # хеш -> целое


def hash_token_sign(token: str) -> tuple[int, float]:
    """(целое 64 бита, знак +1/−1): знак = −1, если старший бит v равен 1, иначе +1.

    Формула: s = −1, если (v >> 63) == 1; иначе +1."""
    value = hash_token_u64(token)  # целое из хеша
    return value, (-1.0 if (value >> 63) else 1.0)  # пара (v, s): знак −1 при старшем бите 1


def hash_feature_bin(token: str, bins: int) -> int:
    """Номер ячейки признака: 0..bins−1. Формула: hash_token_u64(token) mod bins."""
    return hash_token_u64(token) % bins  # остаток от деления на число ячеек


def normalize_key(key: str) -> str:
    """Нормализованный ключ: нижний регистр (casefold), пробелы сжаты в один, крайние убраны.

    Формула: " ".join(str(key).casefold().split()); пустой результат — ValueError."""
    result = " ".join(str(key).casefold().split())  # str -> нижний регистр -> разбиение по пробелам -> соединение одним пробелом
    if not result:  # ключ пуст
        raise ValueError("fact key must be non-empty")  # ошибка (текст сообщения сохранён прежним — на него может опираться вызывающий код)
    return result  # нормализованный ключ
