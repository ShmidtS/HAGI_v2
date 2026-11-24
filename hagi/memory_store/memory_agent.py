"""Цикл «поиск в памяти — ответ — проверенная обратная связь».

Ключевые идеи:
  - Поиск гибридный: точный идентификатор из запроса (фиксирует сущность даже
    при перефразировании), затем семантический индекс, затем полнотекстовый
    поиск по всему запросу. Повторы удаляются по (page_id, char_start, char_end).
  - Компактная троичная байтовая модель переупорядочивает доказательства: для
    пары (запрос, доказательство) строится строка фиксированной длины context
      "|IDM=|ST=|R=" + "Q=<слова запроса>|E=<слова доказательства>",
    модель даёт logit следующего бита, оценка = logits['1'] − logits['0'].
    Модель не порождает ответ: ответ извлекается из выбранной записи.
  - Итоговый балл: combined = neural + utility_weight·utility + cache_bonus
    (если страница в кэше). Маржа = балл первого − балл второго.
  - Полезность страницы меняется только по внешней проверке: reward = +1 или −1.

Экспортирует: RankedEvidence, MemoryAnswer, pair_prompt,
pair_training_record, TernaryEvidenceReranker, MemoryAugmentedAgent.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов

from dataclasses import dataclass  # классы-записи
import re  # регулярные выражения
from typing import Iterable  # тип «итерируемый»

import torch  # тензоры

from .streaming import DiskBackedContext, MemoryHit  # дисковый архив и попадание поиска

_IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z_-]{1,}[0-9][A-Z0-9_-]*\b")  # идентификатор: заглавная буква, затем ≥ 1 символ из [A-Z_-], цифра, хвост из [A-Z0-9_-]
_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)  # слово: непрерывная цепочка букв/цифр (без подчёркивания)
_VALUE_RE = re.compile(r"\bvalue\s*[=:]\s*([A-Za-z0-9._/-]+)", flags=re.IGNORECASE)  # «value = X» или «value: X»; группа 1 — значение X
_FINAL_RE = re.compile(  # слова, помечающие запись как окончательную
    r"\b(final\s+decision|approved|adopted|selected|confirmed|current|authoritative)\b",  # без учёта регистра
    flags=re.IGNORECASE,  # флаг: регистр не различается
)
_OLD_RE = re.compile(  # слова, помечающие запись как устаревшую
    r"\b(earlier|preliminary|rejected|superseded|obsolete|draft|withdrawn|trial)\b",  # без учёта регистра
    flags=re.IGNORECASE,  # флаг: регистр не различается
)


@dataclass(frozen=True)  # неизменяемая класс-запись
class RankedEvidence:  # доказательство с баллами
    """Доказательство поиска с баллами: само попадание, балл нейросети, балл
    полезности, признак страницы в кэше и итоговый балл."""
    hit: MemoryHit  # попадание поиска (страница, границы, текст)
    neural_score: float  # балл нейросети: logits['1'] − logits['0']
    utility_score: float  # балл полезности от обучаемой оценки страницы
    cached: bool  # находится ли страница в кэше рабочего набора
    combined_score: float  # итог: neural + utility_weight·utility + cache_bonus·cached
@dataclass(frozen=True)  # неизменяемая класс-запись
class MemoryAnswer:  # ответ агента
    """Ответ на запрос: status — 'unknown' (ничего не найдено),
    'no_extractable_value' (в лучшей записи нет «value = X»), 'uncertain'
    (маржа меньше min_score_margin) или 'answered'; answer — извлечённое
    значение; evidence — выбранная запись; score и margin — её балл и маржа;
    candidates — все доказательства-кандидаты."""
    query: str  # текст запроса
    status: str  # статус ответа
    answer: str | None  # извлечённое значение или None
    evidence: MemoryHit | None  # выбранная запись-доказательство
    score: float | None  # итоговый балл выбранной записи
    margin: float | None  # маржа над вторым кандидатом
    candidates: tuple[RankedEvidence, ...]  # кортеж кандидатов после упорядочивания


def _first_identifier(text: str) -> str:  # первый идентификатор в тексте
    match = _IDENTIFIER_RE.search(str(text))  # найти первый идентификатор
    return match.group(0).casefold() if match else "none"  # идентификатор в нижнем регистре; если нет — "none"


def _status_tag(text: str) -> str:  # грубый статус записи
    final = bool(_FINAL_RE.search(text))  # есть ли слово окончательности
    old = bool(_OLD_RE.search(text))  # есть ли слово устарелости
    if old:  # слова устарелости важнее
        return "old"  # устаревшая
    if final:  # иначе есть слово окончательности
        return "final"  # окончательная
    return "neutral"  # иначе нейтральная


def _compact_words(text: str, max_chars: int = 45) -> str:  # сжатый текст
    words = [w.casefold() for w in _TOKEN_RE.findall(str(text))]  # все слова текста в нижнем регистре
    return " ".join(words)[:max_chars]  # слова через пробел, первые max_chars символов


def pair_prompt(query: str, evidence: str, context: int = 160) -> bytes:  # запрос, доказательство, длина
    """Строка фиксированной длины context байт — пара «запрос — доказательство»
    для байтовой модели.

        Явные признаки: IDM = 1, если первые идентификаторы запроса и
        доказательства совпадают (и не равны "none"), иначе 0; ST — статус
        доказательства. Хвост строки, на который модель предсказывает следующий
        байт: "1" (релевантно) или "0" (нерелевантно).
        Схема: fixed = "|IDM=<IDM>|ST=<ST>|R="; available = max(0, context −
        |fixed| − |"Q=|E="|) в байтах; half = available // 2; слова запроса
        обрезаются до half символов, слова доказательства — до available −
        длина(слова запроса). Итог: "Q=<q>|E=<e>" + fixed; если длиннее context —
        оставляется последние context байт; слева дополняется пробелами."""
    if context < 32:  # контекст слишком мал для признаков и текста
        raise ValueError("context must be >= 32")  # ошибка значения
    query_id = _first_identifier(query)  # идентификатор запроса
    evidence_id = _first_identifier(evidence)  # идентификатор доказательства
    id_match = int(query_id != "none" and query_id == evidence_id)  # IDM = 1, если идентификаторы совпадают и не "none"
    status = _status_tag(evidence)  # статус доказательства ST
    fixed = f"|IDM={id_match}|ST={status}|R="  # неизменяемая часть строки с признаками
    available = max(0, context - len(fixed.encode("utf-8")) - len("Q=|E="))  # место для слов: available = max(0, context − |fixed| − 6)
    half = available // 2  # половина места
    q_words = _compact_words(query, max_chars=half)  # слова запроса, до half символов
    e_words = _compact_words(evidence, max_chars=available - len(q_words))  # слова доказательства, до available − длина(запроса)
    body = f"Q={q_words}|E={e_words}{fixed}".encode("utf-8")  # тело: "Q=<q>|E=<e>" + fixed в байтах
    if len(body) > context:  # тело длиннее контекста
        body = body[-context:]  # оставить последние context байт
    return b" " * (context - len(body)) + body  # дополнить пробелами слева до context байт


def pair_training_record(query: str, evidence: str, relevant: bool,  # запрос, доказательство, релевантность
                         context: int = 160) -> bytes:  # длина контекста
    """Один обучающий пример предсказания следующего байта: «1» — релевантно,
    «0» — нерелевантно."""
    return pair_prompt(query, evidence, context=context) + (b"1" if relevant else b"0")  # строка пары + байт метки


class TernaryEvidenceReranker:  # переупорядочиватель доказательств
    """Оценивает найденные доказательства развернутой троичной байтовой моделью."""

    def __init__(self, model, context: int | None = None):  # модель и длина контекста
        self.model = model  # байтовая модель
        self.context = int(context if context is not None else getattr(model, "context", 160))  # длина контекста: данная, иначе context модели, иначе 160
        if int(getattr(model, "context", self.context)) != self.context:  # контекст модели задан и отличается
            raise ValueError("reranker context does not match model architecture")  # ошибка: контекст не совпадает с устройством модели
        self.model.eval()  # перевести модель в режим оценки

    @torch.no_grad()  # без вычисления градиентов
    def score(self, query: str, evidence: str) -> float:  # оценка пары
        """Вернуть разность логитов logits['1'] − logits['0'] для следующего
        байта строки пары."""
        encoded = pair_prompt(query, evidence, context=self.context)  # строка пары длиной context
        x = torch.tensor([list(encoded)], dtype=torch.long)  # тензор (1, context) кодов байтов
        logits = self.model(x)[0]  # логиты следующего байта: убрана ось пакета
        # Для упорядочивания разность логитов устойчивее, чем насыщающаяся вероятность.
        return float((logits[ord("1")] - logits[ord("0")]).item())  # score = logits['1'] − logits['0']


class MemoryAugmentedAgent:  # агент с памятью
    """Гибридный поиск (полнотекстовый + семантический), троичное
    переупорядочивание и проверенная обратная связь.

        Долговременный архив никогда не сокращается. Обратная связь меняет только
        предпочтение внутри ограниченного рабочего набора; архив остаётся
        источником, к которому всегда можно обратиться."""

    # конструктор агента: архив, индексы, переупорядочиватель и параметры балла
    def __init__(
        self,  # создаваемый агент
        memory: DiskBackedContext,  # дисковый архив с полнотекстовым поиском
        semantic_index,  # семантический индекс с методом search(query, limit) или None
        reranker: TernaryEvidenceReranker,  # переупорядочиватель доказательств
        utility_index=None,  # индекс полезности страниц или None
        utility_weight: float = 0.10,  # вес полезности в итоговом балле
        cache_bonus: float = 0.0,  # прибавка за страницу в кэше
        max_candidates: int = 24,  # максимум возвращаемых кандидатов
        min_score_margin: float = 0.0,  # минимальная маржа для ответа
    ):
        """Создать агент; вернуть кандидатов не более max_candidates."""
        if max_candidates < 1 or utility_weight < 0 or cache_bonus < 0 or min_score_margin < 0:  # неверные параметры: кандидатов < 1 или отрицательный вес/прибавка/маржа
            raise ValueError("invalid agent configuration")  # ошибка значения
        self.memory = memory  # дисковый архив
        self.semantic_index = semantic_index  # семантический индекс
        self.reranker = reranker  # переупорядочиватель
        self.utility_index = utility_index  # индекс полезности
        self.utility_weight = float(utility_weight)  # вес полезности
        self.cache_bonus = float(cache_bonus)  # прибавка за кэш
        self.max_candidates = int(max_candidates)  # максимум кандидатов
        self.min_score_margin = float(min_score_margin)  # минимальная маржа

    def retrieve(self, query: str) -> tuple[RankedEvidence, ...]:  # найти и упорядочить доказательства
        """Вернуть до max_candidates доказательств, упорядоченных по
        combined = neural + utility_weight·utility + cache_bonus·cached
        по убыванию; при равенстве — страница с меньшим page_id."""
        hits: list[MemoryHit] = []  # накопленные попадания без повторов
        seen: set[tuple[int, int, int]] = set()  # уже добавленные (page_id, char_start, char_end)

        def add(found: Iterable[MemoryHit]) -> None:  # добавить находки без повторов
            for hit in found:  # по каждой находке
                key = (hit.page_id, hit.char_start, hit.char_end)  # ключ: (страница, начало, конец)
                if key not in seen:  # такой записи ещё не было
                    seen.add(key)  # запомнить ключ
                    hits.append(hit)  # добавить находку

        # Точный поиск идентификатора фиксирует сущность, даже если пользователь
        # выразился иначе; семантический и полнотекстовый поиск добавляют связный контекст.
        identifier = _first_identifier(query)  # идентификатор запроса или "none"
        if identifier != "none":  # в запросе есть идентификатор
            add(self.memory.search(identifier, limit=self.max_candidates))  # добавить страницы с этим идентификатором
        if self.semantic_index is not None:  # семантический индекс подключён
            add(self.semantic_index.search(query, limit=self.max_candidates))  # добавить его находки по запросу
        add(self.memory.search(query, limit=self.max_candidates))  # добавить полнотекстовые находки по запросу

        if not hits:  # ничего не найдено
            return ()  # пустой кортеж

        cached_ids: set[int] = set()  # номера страниц в кэше
        learner = None  # оценка полезности страницы
        if self.utility_index is not None:  # индекс полезности подключён
            cached_ids = self.utility_index.working_set.cached_page_ids  # страницы рабочего набора в кэше
            learner = self.utility_index.working_set.learner  # обучаемая оценка текста страницы

        ranked: list[RankedEvidence] = []  # доказательства с баллами
        for hit in hits[: self.max_candidates * 3]:  # оцениваются не более 3·max_candidates находок
            neural = self.reranker.score(query, hit.text)  # балл нейросети: logits['1'] − logits['0']
            utility = learner.score(hit.text) if learner is not None else 0.0  # полезность текста страницы или 0
            cached = hit.page_id in cached_ids  # страница в кэше рабочего набора
            combined = neural + self.utility_weight * utility + (self.cache_bonus if cached else 0.0)  # итог combined = neural + utility_weight·utility + cache_bonus·cached
            ranked.append(RankedEvidence(hit, neural, utility, cached, combined))  # сохранить доказательство с баллами
        ranked.sort(key=lambda item: (item.combined_score, -item.hit.page_id), reverse=True)  # по combined по убыванию; при равенстве — по −page_id по убыванию (меньший page_id первым)
        return tuple(ranked[: self.max_candidates])  # первые max_candidates

    def answer(self, query: str) -> MemoryAnswer:  # ответ на запрос
        """Вернуть ответ: значение X из «value = X» лучшего кандидата, если он
        есть и маржа не меньше min_score_margin; иначе статус объясняет отказ."""
        candidates = self.retrieve(query)  # упорядоченные кандидаты
        if not candidates:  # кандидатов нет
            return MemoryAnswer(query, "unknown", None, None, None, None, ())  # статус "unknown", без ответа
        top = candidates[0]  # лучший кандидат
        second_score = candidates[1].combined_score if len(candidates) > 1 else None  # балл второго кандидата или None
        margin = None if second_score is None else top.combined_score - second_score  # margin = балл первого − балл второго
        value_match = _VALUE_RE.search(top.hit.text)  # найти «value = X» в тексте лучшего
        if value_match is None:  # извлекаемого значения нет
            return MemoryAnswer(query, "no_extractable_value", None, top.hit, top.combined_score,  # статус: нет извлекаемого значения
                                margin, candidates)  # с доказательством, баллом, маржой и кандидатами
        if margin is not None and margin < self.min_score_margin:  # есть второй кандидат и маржа меньше порога
            return MemoryAnswer(query, "uncertain", None, top.hit, top.combined_score,  # статус: неуверенно
                                margin, candidates)  # ответ не выдаётся
        return MemoryAnswer(query, "answered", value_match.group(1), top.hit,  # статус: отвечено, answer = X
                            top.combined_score, margin, candidates)  # с баллом, маржой и кандидатами

    def record_verified_feedback(self, result: MemoryAnswer, verified_correct: bool,  # ответ, проверенная правильность
                                 verifier: str, refresh: bool = False):  # имя проверяющего, пересчитать ли рабочее множество
        """Обновить полезность страницы только после внешней проверки.
        Возврат — результат записи обратной связи.

            Специального пути «самонаграды» или обновления по одной уверенности
            нет. reward = +1, если verified_correct, иначе −1."""
        if not isinstance(verified_correct, bool):  # правильность должна быть явным True/False
            raise TypeError("verified_correct must be an explicit bool")  # ошибка типа
        source = str(verifier).strip()  # имя проверяющего без крайних пробелов
        if not source:  # имя пусто
            raise ValueError("a verifier/source label is required")  # ошибка значения
        if result.evidence is None:  # доказательство не выбрано
            raise ValueError("cannot provide page feedback without selected evidence")  # ошибка: не к какой странице относить оценку
        if self.utility_index is None:  # индекс полезности не подключён
            raise RuntimeError("no utility index is attached to this agent")  # ошибка времени выполнения
        reward = 1.0 if verified_correct else -1.0  # reward = +1 при правильном ответе, иначе −1
        feedback = self.utility_index.record_page_feedback(result.evidence.page_id, reward=reward)  # записать оценку +reward странице доказательства
        if refresh:  # требуется пересборка
            self.utility_index.refresh()  # пересчитать рабочее множество
        return feedback  # результат записи
