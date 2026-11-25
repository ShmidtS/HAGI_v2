"""Структурированные факты с учётом конфликтов для постоянной памяти HAGI.

Слой не даёт непроверенному противоречащему наблюдению молча заменить принятое
значение. Каждое предложение сохраняется в журнале; обновление действующего факта
требует либо достаточного перевеса свидетельств, либо явной правки.

Экспортирует FactSnapshot (действующая версия), FactAnswer (ответ на запрос),
FactProposal (итог предложения) и ConflictAwareFactStore (журнал только с дополнениями
и защищённым обновлением текущего значения поверх DiskBackedContext)."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

from dataclasses import dataclass  # декоратор класса данных
import time  # время в наносекундах

from hagi.base.hashing import normalize_key  # нормализация ключа факта (общая реализация)
from .streaming import DiskBackedContext  # дисковое хранилище с базой SQLite


@dataclass(frozen=True)  # неизменяемый класс данных
class FactSnapshot:
    """Действующая версия факта со свидетельствами."""

    key: str  # нормализованный ключ факта
    revision: int  # номер версии
    value: str  # действующее значение
    source: str  # источник значения
    confidence: float  # оценка качества этого наблюдения (0..1)
    authority: float  # оценка положения источника (0..1)
    history_id: int  # номер записи в журнале, задавшей версию
    updated_ns: int  # время обновления в наносекундах


@dataclass(frozen=True)  # неизменяемый класс данных
class FactAnswer:
    """Ответ на запрос факта: статус, значение, действующая версия и альтернативы."""

    key: str  # нормализованный ключ факта
    status: str  # статус: unknown, conflict или found
    value: str | None  # значение, либо None (факта нет или конфликт не решён)
    current: FactSnapshot | None  # действующая версия (как происхождение), либо None
    alternatives: tuple[str, ...] = ()  # значения ожидающих конфликтных предложений


@dataclass(frozen=True)  # неизменяемый класс данных
class FactProposal:
    """Итог одного предложения значения."""

    key: str  # нормализованный ключ факта
    proposed_value: str  # предложенное значение
    status: str  # статус: accepted, reaffirmed или pending_conflict
    accepted: bool  # True для accepted и reaffirmed
    current: FactSnapshot  # действующая версия после предложения
    confidence: float  # оценка качества наблюдения
    authority: float  # оценка положения источника
    history_id: int  # номер записи предложения в журнале


class ConflictAwareFactStore:
    """Журнал свидетельств только с дополнениями и защищённым обновлением текущего значения.

    ``confidence`` оценивает качество наблюдения, ``authority`` — положение источника.
    Ни та, ни другая не является вероятностью истинности, которую система выдаёт сама:
    их задают или выводят вызывающие стороны."""

    def __init__(self, memory: DiskBackedContext, min_confidence_gain: float = 0.15,  # хранилище и минимальный перевес уверенности
                 authority_override: float = 0.20, authority_override_min_confidence: float = 0.60):  # перевес авторитета и нужная при нём уверенность
        """Создать журнал фактов поверх базы хранилища memory.

        min_confidence_gain — на сколько уверенность нового наблюдения должна превысить
        уверенность текущего; authority_override — на сколько его авторитет должен
        превысить авторитет текущего; authority_override_min_confidence — минимальная
        уверенность для замены по авторитету. Все параметры — числа из [0, 1]."""
        if not 0.0 <= min_confidence_gain <= 1.0:  # параметр вне отрезка [0, 1]
            raise ValueError("min_confidence_gain must be in [0,1]")  # ошибка: значение вне [0,1]
        if not 0.0 <= authority_override <= 1.0:  # параметр вне отрезка [0, 1]
            raise ValueError("authority_override must be in [0,1]")  # ошибка: значение вне [0,1]
        if not 0.0 <= authority_override_min_confidence <= 1.0:  # параметр вне отрезка [0, 1]
            raise ValueError("authority_override_min_confidence must be in [0,1]")  # ошибка: значение вне [0,1]
        self.memory = memory  # дисковое хранилище
        self.db = memory.db  # то же соединение с базой SQLite
        self.min_confidence_gain = float(min_confidence_gain)  # минимальный перевес уверенности
        self.authority_override = float(authority_override)  # минимальный перевес авторитета
        self.authority_override_min_confidence = float(authority_override_min_confidence)  # минимальная уверенность для замены по авторитету
        self._create_schema()  # создать таблицы журнала, если их нет

    def _create_schema(self) -> None:
        """Создать таблицу журнала guarded_fact_history с индексом и таблицу текущих фактов guarded_fact_current."""
        # журнал: номер записи — ключ; ключ факта, версия, значение, источник, уверенность,
        # авторитет, статус, признак явной правки (по умолчанию 0), время создания
        self.db.execute("""\
CREATE TABLE IF NOT EXISTS guarded_fact_history (
            history_id INTEGER PRIMARY KEY,
            fact_key TEXT NOT NULL,
            revision INTEGER NOT NULL,
            value TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence REAL NOT NULL,
            authority REAL NOT NULL,
            status TEXT NOT NULL,
            explicit_correction INTEGER NOT NULL DEFAULT 0,
            created_ns INTEGER NOT NULL)"""  # created_ns — время создания; конец текста создания guarded_fact_history
            )
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_guarded_history_key ON guarded_fact_history(fact_key, history_id)")  # индекс по (ключ факта, номер записи) для выборок истории одного ключа
        # текущие факты: по строке на ключ — версия, значение, источник, уверенность,
        # авторитет, номер записи журнала и время обновления
        self.db.execute("""\
CREATE TABLE IF NOT EXISTS guarded_fact_current (
            fact_key TEXT PRIMARY KEY,
            revision INTEGER NOT NULL,
            value TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence REAL NOT NULL,
            authority REAL NOT NULL,
            history_id INTEGER NOT NULL,
            updated_ns INTEGER NOT NULL)"""  # updated_ns — время обновления; конец текста создания guarded_fact_current
            )
        self.db.commit()  # зафиксировать создание таблиц

    @staticmethod  # статический метод: не получает self
    def normalize_key(key: str) -> str:  # ключ факта; возвращает нормализованный ключ
        """Нормализовать ключ: нижний регистр, пробелы сжаты в один, крайние убраны; пустой ключ — ValueError."""
        return normalize_key(key)  # общая реализация: " ".join(str(key).casefold().split()); пусто — ValueError

    @staticmethod  # статический метод: не получает self
    def normalize_value(value: str) -> str:  # значение; возвращает нормализованное значение
        """Нормализовать значение так же, как ключ: нижний регистр, пробелы сжаты в один, крайние убраны."""
        return " ".join(str(value).casefold().split())  # str(value) -> нижний регистр -> разбиение по пробелам -> соединение одним пробелом

    @staticmethod  # статический метод: не получает self
    def _validate_score(name: str, value: float) -> float:  # имя параметра и значение; возвращает число из [0, 1]
        """Проверить, что оценка value лежит в [0, 1]; иначе ValueError с именем name."""
        score = float(value)  # оценка как вещественное число
        if not 0.0 <= score <= 1.0:  # оценка вне отрезка [0, 1]
            raise ValueError(f"{name} must be in [0,1]")  # ошибка: значение вне [0,1]
        return score  # проверенная оценка

    @staticmethod  # статический метод: не получает self
    def _snapshot(row) -> FactSnapshot:  # строка базы; возвращает снимок факта
        """Превратить строку таблицы текущих фактов (8 столбцов) в FactSnapshot."""
        return FactSnapshot(str(row[0]), int(row[1]), str(row[2]), str(row[3]),  # ключ, версия, значение, источник
                            float(row[4]), float(row[5]), int(row[6]), int(row[7]))  # уверенность, авторитет, номер записи, время

    def get(self, key: str) -> FactSnapshot | None:
        """Вернуть действующую версию факта с ключом key либо None, если факта нет."""
        fact_key = self.normalize_key(key)  # нормализованный ключ факта
        row = self.db.execute(  # строка текущего факта (восемь столбцов снимка) по ключу
            """\
SELECT fact_key,revision,value,source,confidence,authority,history_id,updated_ns
               FROM guarded_fact_current WHERE fact_key=?""", (fact_key,)  # из таблицы текущих фактов по ключу
        ).fetchone()  # одна строка или None
        return None if row is None else self._snapshot(row)  # None, если строки нет, иначе снимок

    def answer(self, key: str) -> FactAnswer:
        """Дать ответ, только если нет нерешённых противоречащих предложений.

        При конфликте действующая версия выдаётся как происхождение, а ``value``
        намеренно равно None, чтобы вызывающая сторона могла воздержаться."""
        fact_key = self.normalize_key(key)  # нормализованный ключ факта
        current = self.get(fact_key)  # действующая версия (или None)
        if current is None:  # факта нет
            return FactAnswer(fact_key, "unknown", None, None, ())  # ответ: статус unknown, без значения и версии
        conflicts = self.pending_conflicts(fact_key)  # ожидающие конфликтные предложения по ключу
        if conflicts:  # есть нерешённый конфликт
            alternatives = tuple(dict.fromkeys(str(row["value"]) for row in conflicts))  # значения конфликтных предложений без повторов, в порядке появления
            return FactAnswer(fact_key, "conflict", None, current, alternatives)  # ответ: конфликт, значение скрыто
        return FactAnswer(fact_key, "found", current.value, current, ())  # ответ: найдено, со значением действующей версии

    def propose(self, key: str, value: str, source: str = "", confidence: float = 0.5,  # ключ, значение, источник, уверенность
                authority: float = 0.5, explicit_correction: bool = False) -> FactProposal:  # авторитет и признак явной правки
        """Записать предложение значения и, если правилами разрешено, обновить действующий факт.

        Новое значение принимается, если выполняется хотя бы одно: явная правка;
        уверенность ≥ уверенности текущего + min_confidence_gain при авторитете ≥ авторитета
        текущего; авторитет ≥ авторитета текущего + authority_override при уверенности
        ≥ authority_override_min_confidence. Иначе предложение получает статус
        pending_conflict (ожидает решения). Совпадающее с текущим значение даёт статус
        reaffirmed, и в текущей версии сохраняется более сильное из свидетельств.
        Возвращает итог предложения."""
        fact_key = self.normalize_key(key)  # нормализованный ключ факта
        proposed_value = str(value).strip()  # предложенное значение без крайних пробелов
        if not proposed_value:  # значение пусто
            raise ValueError("fact value must be non-empty")  # ошибка: значение факта пусто
        conf = self._validate_score("confidence", confidence)  # проверенная уверенность conf
        auth = self._validate_score("authority", authority)  # проверенный авторитет auth
        src = str(source)  # источник как строка
        now = time.time_ns()  # текущее время в наносекундах
        current = self.get(fact_key)  # действующая версия (или None)
        if current is None:  # факта ещё нет
            revision = 1  # первая версия
            status = "accepted"  # первое значение принимается
        elif self.normalize_value(proposed_value) == self.normalize_value(current.value):  # значение совпадает с текущим (нормализованным)
            revision = current.revision  # версия не меняется
            status = "reaffirmed"  # подтверждение
        else:  # значение противоречит текущему
            confidence_override = (  # замена по уверенности:
                auth >= current.authority  # авторитет не ниже текущего
                and conf >= current.confidence + self.min_confidence_gain  # и уверенность ≥ текущая + min_confidence_gain
            )  # конец условия
            authority_override = (  # замена по авторитету:
                auth >= current.authority + self.authority_override  # авторитет ≥ текущий + authority_override
                and conf >= self.authority_override_min_confidence  # и уверенность ≥ authority_override_min_confidence
            )  # конец условия
            accepted = bool(explicit_correction or confidence_override or authority_override)  # принять, если есть явная правка или один из перевесов
            revision = current.revision + 1  # номер новой версии = текущий + 1
            status = "accepted" if accepted else "pending_conflict"  # статус: принято либо конфликт
        cur = self.db.execute(  # дописать предложение в журнал (в любом случае)
            """\
INSERT INTO guarded_fact_history
               (fact_key,revision,value,source,confidence,authority,status,explicit_correction,created_ns)
               VALUES (?,?,?,?,?,?,?,?,?)""",  # девять параметров
            (fact_key, revision, proposed_value, src, conf, auth, status,  # ключ, версия, значение, источник, уверенность, авторитет, статус
             int(bool(explicit_correction)), now),  # признак правки как 0/1 и время
        )
        history_id = int(cur.lastrowid)  # номер записи = последний вставленный rowid
        if current is None:  # факта не было — вставить текущую версию
            self.db.execute(  # вставить текущую версию нового факта
                """\
INSERT INTO guarded_fact_current
                   (fact_key,revision,value,source,confidence,authority,history_id,updated_ns)
                   VALUES (?,?,?,?,?,?,?,?)""",  # восемь параметров
                (fact_key, 1, proposed_value, src, conf, auth, history_id, now),  # ключ, версия 1, значение, источник, уверенность, авторитет, номер записи, время
            )
        elif status == "reaffirmed":  # значение подтверждено
            # Сохранить более сильное свидетельство, не меняя действующее значение.
            new_conf = max(current.confidence, conf)  # уверенность = max(текущая, новая)
            new_auth = max(current.authority, auth)  # авторитет = max(текущий, новый)
            chosen_source = src if conf >= current.confidence else current.source  # источник нового, если его уверенность не ниже, иначе прежний
            self.db.execute(  # обновить свидетельства при подтверждении
                """\
UPDATE guarded_fact_current SET confidence=?,authority=?,source=?,updated_ns=?
                   WHERE fact_key=?""",  # обновить уверенность, авторитет, источник и время по ключу
                (new_conf, new_auth, chosen_source, now, fact_key),  # новые значения
            )
        elif status == "accepted":  # новая версия принята
            self.db.execute(  # пометить прежнюю запись журнала заменённой
                "UPDATE guarded_fact_history SET status='superseded' WHERE history_id=?",  # прежняя запись журнала получает статус «заменена»
                (current.history_id,),  # по её номеру
            )
            self.db.execute(  # заменить текущую версию принятой
                """\
UPDATE guarded_fact_current SET revision=?,value=?,source=?,confidence=?,authority=?,
                   history_id=?,updated_ns=? WHERE fact_key=?""",  # заменить версию, значение, источник, уверенность, авторитет, номер записи и время
                (revision, proposed_value, src, conf, auth, history_id, now, fact_key),  # новыми значениями по ключу
            )
        self.db.commit()  # зафиксировать изменения
        new_current = self.get(fact_key)  # действующая версия после предложения
        assert new_current is not None  # после вставки или обновления версия обязана существовать
        return FactProposal(fact_key, proposed_value, status,  # итог предложения: ключ, значение, статус
                            status in ("accepted", "reaffirmed"), new_current,  # принято для accepted и reaffirmed; действующая версия
                            conf, auth, history_id)  # уверенность, авторитет, номер записи

    def resolve(self, key: str, chosen_value: str, reviewer: str = "manual-review") -> FactProposal:
        """Явно решить ожидающие конфликты по ключу: выбор проверяющего имеет силу правки.

        Выбранное значение вносится как явная правка с уверенностью и авторитетом 1.0;
        предложения с тем же значением получают статус reaffirmed, остальные ожидающие — rejected."""
        result = self.propose(key, chosen_value, source=reviewer, confidence=1.0,  # внести выбор как правку с максимальными оценками
                              authority=1.0, explicit_correction=True)  # (явная правка)
        fact_key = self.normalize_key(key)  # нормализованный ключ факта
        # После решения проверяющим конкурирующие ожидающие предложения закрываются.
        self.db.execute(  # подтвердить совпадающие ожидающие предложения
            """\
UPDATE guarded_fact_history SET status='reaffirmed'
               WHERE fact_key=? AND status='pending_conflict' AND lower(trim(value))=lower(trim(?))""",  # совпадающие по значению (без регистра и крайних пробелов) ожидающие — подтверждены
            (fact_key, str(chosen_value).strip()),  # ключ и выбранное значение без крайних пробелов
        )
        self.db.execute(  # отклонить остальные ожидающие предложения
            """\
UPDATE guarded_fact_history SET status='rejected'
               WHERE fact_key=? AND status='pending_conflict' AND lower(trim(value))<>lower(trim(?))""",  # не совпадающие ожидающие — отклонены
            (fact_key, str(chosen_value).strip()),  # ключ и выбранное значение
        )
        self.db.commit()  # зафиксировать изменения
        return result  # итог предложения выбранного значения

    def history(self, key: str, include_pending: bool = True) -> list[dict]:
        """Вернуть журнал предложений по ключу key в порядке номеров записей — словари с девятью полями.

        include_pending=False исключает записи со статусом pending_conflict."""
        fact_key = self.normalize_key(key)  # нормализованный ключ факта
        # запрос истории: номер записи, версия, значение, источник, уверенность, авторитет, статус, признак правки, время
        sql = """\
SELECT history_id,revision,value,source,confidence,authority,status,
                        explicit_correction,created_ns
                 FROM guarded_fact_history WHERE fact_key=?"""  # по ключу факта
        params: list = [fact_key]  # параметры запроса
        if not include_pending:  # ожидающие исключаются
            sql += " AND status<>'pending_conflict'"  # добавить условие «статус не pending_conflict»
        sql += " ORDER BY history_id"  # упорядочить по номеру записи
        rows = self.db.execute(sql, params).fetchall()  # все строки истории
        keys = ("history_id", "revision", "value", "source", "confidence", "authority",  # имена полей результата: номер записи, версия, значение, источник, уверенность, авторитет,
                "status", "explicit_correction", "created_ns")  # статус, признак правки, время создания
        return [dict(zip(keys, row)) for row in rows]  # каждая строка -> словарь «имя поля -> значение»

    def pending_conflicts(self, key: str | None = None) -> list[dict]:
        """Вернуть ожидающие конфликтные предложения (по ключу key или все) в порядке номеров записей — словари с восемью полями."""
        # запрос: номер записи, ключ, версия, значение, источник, уверенность, авторитет, время создания
        sql = """\
SELECT history_id,fact_key,revision,value,source,confidence,authority,created_ns
                 FROM guarded_fact_history WHERE status='pending_conflict'"""  # только ожидающие конфликты
        params: tuple = ()  # параметров пока нет
        if key is not None:  # задан конкретный ключ
            sql += " AND fact_key=?"  # добавить условие по ключу
            params = (self.normalize_key(key),)  # параметр — нормализованный ключ
        sql += " ORDER BY history_id"  # упорядочить по номеру записи
        rows = self.db.execute(sql, params).fetchall()  # все строки
        keys = ("history_id", "fact_key", "revision", "value", "source", "confidence",  # имена полей: номер записи, ключ, версия, значение, источник, уверенность,
                "authority", "created_ns")  # авторитет, время создания
        return [dict(zip(keys, row)) for row in rows]  # каждая строка -> словарь «имя поля -> значение»
