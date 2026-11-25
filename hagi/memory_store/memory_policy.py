"""Локально обучаемый контроллер политики записи в долговременную память.

    Контроллер относит каждую запись к одному из трёх классов: transient (мимолётное),
    fact (факт), decision (решение). Он намеренно мал и работает на процессоре. Это фильтр
    записи, а не самостоятельный определитель истинности: чему он научится, ограничено
    метками и распределением обучающих данных.

    Признаки текста (вектор длины feature_bins + 13):
      * feature_bins ячеек хэширования слов и соседних пар слов со знаком:
        x[h mod feature_bins] += sign, затем x[:feature_bins] /= sqrt(число признаков);
      * 6 индикаторов групп слов-подсказок (решение, факт, временное, неуверенность,
        «не сохранять», обязательность), 3 индикатора явного отрицания или усиления,
        доля длины (min(число слов, 100)/100), наличие цифры, наличие «?»,
        доля «!» и «?» (min(число, 5)/5).

    Классификатор _PolicyNet: Linear(d, hidden) -> Tanh -> Linear(hidden, hidden/2) -> ReLU -> Linear(hidden/2, 3).

    Обучение fit: перекрёстная энтропия, AdamW (weight_decay 1e-4), пачки по 128, по классам
    выделяется доля validation_fraction на проверку; хранятся веса лучшей эпохи по потерям
    на проверке; останов после patience эпох без улучшения больше 1e-5.

    Экспортирует: MemoryPolicyController, PolicyMemoryStore, PolicyMetrics, PolicyFitReport."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

from dataclasses import dataclass  # классы-записи
import hashlib  # хэширование признаков и содержимого записей
from hagi.base.hashing import hash_token_sign  # (целое, знак) из хеша слова
import re  # регулярные выражения
import time  # замер времени
from pathlib import Path  # работа с путями файлов
from typing import Iterable  # аннотации типов

import numpy as np  # численные массивы
import torch  # тензоры и автоградиент
from torch import nn  # слои нейросети
from torch.nn import functional as F  # функции потерь

from .streaming import DiskBackedContext, MemoryHit  # дисковый архив памяти и запись результата поиска

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)  # слово: цепочка букв или цифр без подчёркивания
_CLASSES = ("transient", "fact", "decision")  # имена классов по номерам 0, 1, 2


def _stable_hash(token: str) -> tuple[int, float]:  # хэш слова: целое 64 бита и знак +1 или −1 (общая реализация — hagi.base.hashing)
    return hash_token_sign(token)  # (v, s): v = blake2b-целое, s = −1, если старший бит v равен 1


@dataclass(frozen=True)  # неизменяемая запись
class PolicyMetrics:
    """Качество классификатора на наборе текстов."""
    accuracy: float  # доля верных ответов
    macro_f1: float  # среднее F1 по трём классам
    keep_precision: float  # точность «сохранить»: tp / (tp + fp)
    keep_recall: float  # полнота «сохранить»: tp / (tp + fn)
    keep_f1: float  # F1 класса «сохранить»
    decision_recall: float  # полнота класса «решение»
    retention_rate: float  # доля записей, которые решено сохранить
    n: int  # число текстов


@dataclass(frozen=True)  # неизменяемая запись
class PolicyFitReport:
    """Итоги обучения контроллера."""
    epochs: int  # эпоха с наименьшими потерями на проверке
    best_validation_loss: float  # эти потери
    train_seconds: float  # время обучения в секундах
    train_metrics: PolicyMetrics  # метрики на обучающих текстах
    validation_metrics: PolicyMetrics  # метрики на проверочных текстах


class _PolicyNet(nn.Module):
    """Малая нейросеть-классификатор: Linear(d, hidden) -> Tanh -> Linear(hidden, hidden/2) -> ReLU -> Linear(hidden/2, 3)."""
    def __init__(self, input_dim: int, hidden: int = 48):  # создать сеть; input_dim — длина входа, hidden — ширина слоёв
        super().__init__()  # инициализация родительского класса nn.Module
        self.net = nn.Sequential(  # последовательность слоёв
            nn.Linear(input_dim, hidden), nn.Tanh(),  # линейный слой d -> hidden, затем tanh
            nn.Linear(hidden, hidden // 2), nn.ReLU(),  # линейный слой hidden -> hidden/2, затем relu
            nn.Linear(hidden // 2, 3),  # линейный слой hidden/2 -> 3 логита
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # прямой проход
        return self.net(x)  # логиты формы (N, 3)


class MemoryPolicyController:
    """Малый обучаемый классификатор: решает, что попадает в долговременную память.

        Хэширование признаков делает длину входа постоянной при любом размере архива.
        Предобученная модель и доступ к сети не используются."""

    def __init__(self, feature_bins: int = 256, hidden: int = 48, seed: int = 31):  # создать контроллер
        if feature_bins < 64 or hidden < 8:  # параметры слишком малы
            raise ValueError("feature_bins must be >=64 and hidden >=8")  # ошибка
        self.feature_bins = int(feature_bins)  # число ячеек хэширования
        self.hidden = int(hidden)  # ширина слоёв
        self.seed = int(seed)  # зерно случайности
        self.input_dim = self.feature_bins + 13  # длина входа: ячейки плюс 13 дополнительных признаков
        torch.manual_seed(self.seed)  # зерно генератора случайных чисел torch
        self.net = _PolicyNet(self.input_dim, self.hidden)  # сеть-классификатор
        self.trained = False  # обучена ли сеть
        self.fit_report: PolicyFitReport | None = None  # итоги последнего обучения

    def featurize_one(self, text: str) -> np.ndarray:  # вектор признаков одного текста
        """Вектор признаков текста длины feature_bins + 13 : ячейки хэширования слов и пар, 9 индикаторов групп слов и отрицаний, 4 признака длины и знаков."""
        x = np.zeros(self.input_dim, dtype=np.float32)  # нулевой вектор признаков
        tokens = [t.casefold() for t in _TOKEN_RE.findall(str(text))]  # слова текста в нижнем регистре
        # хэширование со знаком отдельных слов и соседних пар слов
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]  # признаки: слова и пары «a_b» соседних слов
        for token in features:  # по признакам
            hv, sign = _stable_hash(token)  # хэш и знак признака
            x[hv % self.feature_bins] += sign  # прибавить знак в ячейку: x[h mod feature_bins] += sign
        if features:  # признаки есть
            x[:self.feature_bins] /= np.sqrt(float(len(features)))  # нормирование ячеек: x[:feature_bins] /= sqrt(|признаков|)
        low = str(text).casefold()  # текст в нижнем регистре
        tail = x[self.feature_bins:]  # срез 13 дополнительных признаков (общая память с x)
        cue_groups = (  # шесть групп слов-подсказок
            ("decision", "decided", "agreed", "approved", "selected", "final choice", "settled on", "commitment", "binding choice"),  # группа 0: решение
            ("remember", "fact", "reference", "documented", "verified", "record indicates", "for future lookup", "reference detail"),  # группа 1: факт
            ("temporary", "today", "briefly", "scratchpad", "passing thought", "just for now", "speculative option"),  # группа 2: временное
            ("maybe", "might", "perhaps", "could", "unconfirmed", "not decided", "not final", "possible idea"),  # группа 3: неуверенность
            ("ignore", "discard", "not retain", "do not keep", "do not save", "not to retain"),  # группа 4: не сохранять
            ("must", "required", "constraint", "always", "never", "binding"),  # группа 5: обязательность
        )
        for i, group in enumerate(cue_groups):  # по группам
            tail[i] = float(any(cue in low for cue in group))  # tail[i] = 1, если в тексте есть хоть одна подсказка группы i
        # явное отрицание или мимолётность должно перевешивать «положительные» существительные
        # вроде «fact» во фразе «not a retained fact»
        tail[6] = float(any(cue in low for cue in (  # tail[6] = 1, если есть фраза явного отрицания факта
            "not a retained fact", "not a verified fact", "not something to keep",  # фразы отрицания
            "do not save", "not to retain", "no lasting information", "ignore this",  # фразы отрицания
        )))  # конец any(...)
        tail[7] = float(any(cue in low for cue in (  # tail[7] = 1, если есть фраза отрицания решения
            "not a decision", "not decided", "not final", "not approved", "not selected",  # фразы отрицания решения
        )))  # конец any(...)
        tail[8] = float(any(cue in low for cue in (  # tail[8] = 1, если есть фраза усиленного решения
            "final choice", "binding decision", "we agreed", "settled on", "commitment:",  # фразы усиленного решения
        )))  # конец any(...)
        tail[9] = min(len(tokens), 100) / 100.0  # tail[9] = min(число слов, 100) / 100
        tail[10] = float(any(ch.isdigit() for ch in low))  # tail[10] = 1, если в тексте есть цифра
        tail[11] = float("?" in low)  # tail[11] = 1, если есть знак вопроса
        tail[12] = min(sum(ch in "!?" for ch in low), 5) / 5.0  # tail[12] = min(число «!» и «?», 5) / 5
        return x  # вектор признаков

    def featurize(self, texts: Iterable[str]) -> np.ndarray:  # матрица признаков списка текстов
        return np.stack([self.featurize_one(t) for t in texts]).astype(np.float32)  # строки — векторы отдельных текстов, тип float32

    @staticmethod  # статический метод: не получает self
    def _metrics(y_true: np.ndarray, probs: np.ndarray) -> PolicyMetrics:  # метрики по истинным меткам и вероятностям
        pred = probs.argmax(axis=1)  # предсказание — класс с наибольшей вероятностью
        labels = (0, 1, 2)  # номера классов
        f1s = []  # F1 по классам
        for label in labels:  # по классам
            tp = int(np.sum((pred == label) & (y_true == label)))  # верно предсказанные этого класса
            fp = int(np.sum((pred == label) & (y_true != label)))  # предсказан этот класс, а истинный другой
            fn = int(np.sum((pred != label) & (y_true == label)))  # истинный этот класс, а предсказан другой
            precision = tp / max(tp + fp, 1)  # точность = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)  # полнота = tp / max(tp + fn, 1)
            f1s.append(2 * precision * recall / max(precision + recall, 1e-12))  # гармоническое среднее: F1 = 2·P·R / max(P + R, 1e-12)
        keep_pred = pred != 0  # решено сохранить: класс не «мимолётное»
        keep_true = y_true != 0  # истинно нужно сохранить
        tp = int(np.sum(keep_pred & keep_true))  # верно сохранённые
        fp = int(np.sum(keep_pred & ~keep_true))  # сохранённые зря
        fn = int(np.sum(~keep_pred & keep_true))  # пропущенные
        kp = tp / max(tp + fp, 1)  # точность «сохранить»
        kr = tp / max(tp + fn, 1)  # полнота «сохранить»
        kf = 2 * kp * kr / max(kp + kr, 1e-12)  # F1 «сохранить»
        decision_total = int(np.sum(y_true == 2))  # число истинных решений
        decision_recall = float(np.sum((pred == 2) & (y_true == 2)) / max(decision_total, 1))  # полнота решений = верные решения / max(число решений, 1)
        return PolicyMetrics(  # собрать метрики
            accuracy=float(np.mean(pred == y_true)), macro_f1=float(np.mean(f1s)),  # доля верных и среднее F1
            keep_precision=kp, keep_recall=kr, keep_f1=kf,  # метрики «сохранить»
            decision_recall=decision_recall, retention_rate=float(np.mean(keep_pred)),  # полнота решений и доля сохранённых
            n=int(len(y_true)),  # число текстов
        )

    def fit(self, texts: list[str], labels: list[int] | np.ndarray,  # обучить классификатор; texts — тексты, labels — метки 0/1/2
            validation_fraction: float = 0.2, epochs: int = 120,  # доля проверки и предел эпох
            lr: float = 0.01, patience: int = 15) -> PolicyFitReport:  # скорость обучения и терпение без улучшения
        """Обучить классификатор на текстах с метками 0, 1, 2; вернуть итоги обучения.

            Проверочная выборка выделяется по классам: из каждого класса случайно берётся доля
            validation_fraction. Остановка — после patience эпох без улучшения потерь на
            проверке больше 1e-5; восстанавливаются веса лучшей эпохи."""
        if len(texts) != len(labels) or len(texts) < 30:  # длины не совпадают или текстов меньше 30
            raise ValueError("texts and labels must have equal length >=30")  # ошибка
        if not 0.05 <= validation_fraction <= 0.4 or epochs < 1 or lr <= 0 or patience < 1:  # неверная конфигурация
            raise ValueError("invalid training configuration")  # ошибка
        y = np.asarray(labels, dtype=np.int64)  # метки как целочисленный массив
        if set(np.unique(y).tolist()) != {0, 1, 2}:  # должны присутствовать все три класса
            raise ValueError("labels must contain all classes 0, 1, 2")  # ошибка
        x = self.featurize(texts)  # матрица признаков
        rng = np.random.default_rng(self.seed)  # генератор случайных чисел с зерном seed
        train_idx: list[int] = []  # номера обучающих текстов
        valid_idx: list[int] = []  # номера проверочных текстов
        for cls in (0, 1, 2):  # по классам
            idx = np.flatnonzero(y == cls)  # номера текстов этого класса
            rng.shuffle(idx)  # перемешать их
            n_valid = max(1, int(round(len(idx) * validation_fraction)))  # число проверочных: max(1, round(доля·число текстов класса))
            valid_idx.extend(idx[:n_valid].tolist())  # первые n_valid — в проверку
            train_idx.extend(idx[n_valid:].tolist())  # остальные — в обучение
        rng.shuffle(train_idx)  # перемешать обучающие номера
        rng.shuffle(valid_idx)  # перемешать проверочные номера
        xt = torch.from_numpy(x[train_idx])  # обучающие признаки
        yt = torch.from_numpy(y[train_idx])  # обучающие метки
        xv = torch.from_numpy(x[valid_idx])  # проверочные признаки
        yv = torch.from_numpy(y[valid_idx])  # проверочные метки
        torch.manual_seed(self.seed)  # зерно для одинаковой начальной сети
        self.net = _PolicyNet(self.input_dim, self.hidden)  # заново создать сеть
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)  # оптимизатор AdamW со затуханием весов 1e-4
        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}  # копия весов лучшей сети (пока начальная)
        best_loss = float("inf")  # наименьшие потери на проверке
        bad_epochs = 0  # эпох подряд без улучшения
        start = time.perf_counter()  # начало замера времени
        best_epoch = 0  # эпоха лучших потерь
        for ep in range(1, epochs + 1):  # по эпохам 1..epochs
            self.net.train()  # режим обучения
            order = torch.randperm(len(xt))  # случайный порядок обучающих текстов
            for batch in order.split(128):  # пачками по 128
                logits = self.net(xt[batch])  # логиты пачки
                loss = F.cross_entropy(logits, yt[batch])  # перекрёстная энтропия: CE = среднее(−log softmax(logit)[метка])
                optimizer.zero_grad(set_to_none=True)  # обнулить градиенты
                loss.backward()  # вычислить градиенты
                optimizer.step()  # шаг оптимизатора
            self.net.eval()  # режим оценки
            with torch.no_grad():  # без градиентов
                val_loss = float(F.cross_entropy(self.net(xv), yv))  # потери на проверке
            if val_loss < best_loss - 1e-5:  # улучшение больше 1e-5
                best_loss = val_loss  # новые лучшие потери
                best_epoch = ep  # эпоха улучшения
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}  # копия весов
                bad_epochs = 0  # счётчик неудач обнулён
            else:  # улучшения нет
                bad_epochs += 1  # неудачных эпох: +1
                if bad_epochs >= patience:  # терпение исчерпано
                    break  # остановка обучения
        self.net.load_state_dict(best_state)  # восстановить лучшие веса
        self.net.eval()  # режим оценки
        elapsed = time.perf_counter() - start  # время обучения
        with torch.no_grad():  # без градиентов
            train_probs = torch.softmax(self.net(xt), dim=1).numpy()  # вероятности на обучении: softmax(logit)_j = e^{l_j} / Σ_k e^{l_k}
            valid_probs = torch.softmax(self.net(xv), dim=1).numpy()  # вероятности на проверке
        self.trained = True  # сеть обучена
        train_report = self._metrics(y[train_idx], train_probs)  # метрики на обучении
        valid_report = self._metrics(y[valid_idx], valid_probs)  # метрики на проверке
        self.fit_report = PolicyFitReport(best_epoch, best_loss, elapsed, train_report, valid_report)  # итоги обучения
        return self.fit_report  # вернуть итоги

    def predict_proba(self, texts: list[str]) -> np.ndarray:  # вероятности классов: матрица (N, 3)
        if not self.trained:  # сеть не обучена
            raise RuntimeError("fit or load the memory policy before prediction")  # ошибка
        if not texts:  # текстов нет
            return np.zeros((0, 3), dtype=np.float32)  # пустая матрица 0×3
        self.net.eval()  # режим оценки
        with torch.no_grad():  # без градиентов
            return torch.softmax(self.net(torch.from_numpy(self.featurize(texts))), dim=1).numpy()  # вероятности классов: p_j = e^{l_j} / Σ_k e^{l_k}

    def predict(self, text: str) -> tuple[str, float, np.ndarray]:  # класс текста, его вероятность и все вероятности
        probs = self.predict_proba([text])[0]  # вероятности классов текста
        cls = int(np.argmax(probs))  # номер класса с наибольшей вероятностью
        return _CLASSES[cls], float(probs[cls]), probs  # (имя класса, его вероятность, вектор вероятностей)

    def save(self, path: str | Path) -> None:  # сохранить обученный контроллер
        if not self.trained:  # сеть не обучена
            raise RuntimeError("cannot save an untrained policy")  # ошибка
        target = Path(path)  # путь как объект Path
        target.parent.mkdir(parents=True, exist_ok=True)  # создать каталог файла, если нужно
        torch.save({"feature_bins": self.feature_bins, "hidden": self.hidden,  # сохранить параметры
                    "seed": self.seed, "state_dict": self.net.state_dict(),  # зерно и веса сети
                    "classes": list(_CLASSES), "version": 1}, target)  # имена классов и версия 1

    @classmethod  # метод класса: создаёт объект, не имея экземпляра
    def load(cls, path: str | Path) -> "MemoryPolicyController":  # загрузить контроллер из файла
        """Загрузить контроллер из файла, созданного save; версия и классы обязаны совпадать."""
        payload = torch.load(path, map_location="cpu", weights_only=True)  # прочитать словарь параметров (только веса, на процессор)
        if payload.get("version") != 1 or payload.get("classes") != list(_CLASSES):  # версия или классы другие
            raise ValueError("unsupported memory-policy checkpoint")  # ошибка
        obj = cls(payload["feature_bins"], payload["hidden"], payload["seed"])  # создать контроллер с сохранёнными параметрами
        obj.net.load_state_dict(payload["state_dict"])  # загрузить веса сети
        obj.net.eval()  # режим оценки
        obj.trained = True  # сеть считается обученной
        return obj  # готовый контроллер

    def attach(self, memory: DiskBackedContext) -> "PolicyMemoryStore":  # присоединить к архиву памяти
        if not self.trained:  # сеть не обучена
            raise RuntimeError("fit or load the memory policy before attaching it")  # ошибка
        return PolicyMemoryStore(memory, self)  # хранилище с проверкой политики записи


class PolicyMemoryStore:
    """Таблица долговременной памяти с пропускным фильтром поверх исходного дискового архива."""

    def __init__(self, memory: DiskBackedContext, policy: MemoryPolicyController):  # создать хранилище по архиву и политике
        self.memory = memory  # исходный архив
        self.policy = policy  # контроллер политики
        db = memory.db  # соединение SQLite
        # таблица отобранных записей: номер, текст, класс, уверенность (вероятность сохранить),
        # источник, границы текста в архиве (начало и конец в символах), хэш текста (уникальный)
        db.execute("""\
CREATE TABLE IF NOT EXISTS policy_memory (
            memory_id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            content_hash TEXT NOT NULL UNIQUE)""")  # создать таблицу, если её нет
        # полнотекстовый индекс по тексту отобранных записей (внешняя таблица policy_memory, ключ memory_id)
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS policy_memory_fts USING fts5(content, content='policy_memory', content_rowid='memory_id')")  # создать индекс fts5
        # триггер после вставки: добавить текст новой записи в полнотекстовый индекс
        db.execute("""\
CREATE TRIGGER IF NOT EXISTS policy_memory_ai AFTER INSERT ON policy_memory BEGIN
            INSERT INTO policy_memory_fts(rowid,content) VALUES (new.memory_id,new.content);
        END""")  # создать триггер
        # триггер после удаления: убрать текст удалённой записи из полнотекстового индекса
        db.execute("""\
CREATE TRIGGER IF NOT EXISTS policy_memory_ad AFTER DELETE ON policy_memory BEGIN
            INSERT INTO policy_memory_fts(policy_memory_fts,rowid,content) VALUES('delete',old.memory_id,old.content);
        END""")  # создать триггер
        db.commit()  # зафиксировать создание объектов

    def ingest(self, records: Iterable[str], source: str = "", keep_threshold: float = 0.5,  # принять записи; source — источник; порог сохранения
               keep_decisions: bool = True) -> dict:  # keep_decisions — сохранять ли класс «решение»; возвращает счётчики
        """Пропустить записи через фильтр: записать все в архив, в отобранную таблицу — только полезные.

            Запись сохраняется, если p_fact + p_decision ≥ keep_threshold и (keep_decisions
            или класс не «decision»). Возвращает счётчики seen, kept, duplicates, total_retained
            и долю retention_rate = kept / max(seen, 1)."""
        if not 0.0 <= keep_threshold <= 1.0:  # порог вне [0, 1]
            raise ValueError("keep_threshold must be in [0,1]")  # ошибка
        seen = kept = duplicates = 0  # счётчики: просмотрено, сохранено, дубликаты
        for record in records:  # по записям
            text = str(record).strip()  # текст без крайних пробелов
            if not text:  # пустой текст
                continue  # пропустить
            seen += 1  # просмотрено: +1
            start = self.memory._indexed_chars + len(self.memory._pending)  # начало записи в архиве: проиндексировано символов + длина буфера
            category, confidence, probs = self.policy.predict(text)  # класс, его вероятность и все вероятности
            retain_prob = float(probs[1] + probs[2])  # вероятность сохранить: p_fact + p_decision
            decision_prob = float(probs[2])  # вероятность класса «решение» (в условии ниже не используется)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()  # хэш текста sha256 — ключ уникальности
            self.memory.append(text + "\n")  # записать текст в исходный архив (с переводом строки)
            if retain_prob >= keep_threshold and (keep_decisions or category != "decision"):  # вероятность сохранить не ниже порога и решение разрешено либо класс не решение
                cur = self.memory.db.execute(  # вставить отобранную запись (дубликат по хэшу игнорируется)
                    """\
INSERT OR IGNORE INTO policy_memory
                    (content,category,confidence,source,char_start,char_end,content_hash)
                    VALUES (?,?,?,?,?,?,?)""",  # вставка семи полей; конфликт по уникальному хэшу пропускается
                    (text, category, retain_prob, str(source), start, start + len(text), digest),  # текст, класс, вероятность сохранить, источник, границы, хэш
                )
                if cur.rowcount:  # строка вставлена
                    kept += 1  # сохранено: +1
                else:  # дубликат
                    duplicates += 1  # дубликатов: +1
        self.memory.flush()  # дописать буфер архива на диск
        self.memory.db.commit()  # зафиксировать вставки
        total = int(self.memory.db.execute("SELECT COUNT(*) FROM policy_memory").fetchone()[0])  # всего записей в отобранной таблице
        return {"seen": seen, "kept": kept, "duplicates": duplicates,  # итог: счётчики
                "total_retained": total, "retention_rate": kept / max(seen, 1)}  # всего сохранено и доля retention_rate = kept / max(seen, 1)

    def search(self, query: str, limit: int = 5) -> list[MemoryHit]:  # полнотекстовый поиск по отобранным записям
        """Полнотекстовый поиск по отобранным записям; результаты по возрастанию ранга bm25 (лучшие первыми)."""
        if limit < 1:  # лимит меньше 1
            raise ValueError("limit must be >=1")  # ошибка
        fts = self.memory._fts_query(query)  # текст запроса, подготовленный для fts5
        if not fts:  # запрос пуст
            return []  # ничего не найдено
        # поиск: для строк индекса, подходящих под запрос, берутся номер записи, границы в архиве,
        # ранг bm25 (меньше — лучше), текст и фрагмент с выделением совпадений в [match]...[/match],
        # соединение с policy_memory по номеру; сортировка по рангу, не более limit строк
        rows = self.memory.db.execute(  # выполнить поиск
            """\
SELECT p.memory_id,p.char_start,p.char_end,bm25(policy_memory_fts),p.content,
                      snippet(policy_memory_fts,0,'[match]','[/match]',' … ',32)
               FROM policy_memory_fts f JOIN policy_memory p ON p.memory_id=f.rowid
               WHERE policy_memory_fts MATCH ? ORDER BY bm25(policy_memory_fts) LIMIT ?""",  # текст запроса; фрагмент до 32 символов с « … » между кусками
            (fts, int(limit)),  # параметры: запрос и лимит
        ).fetchall()  # все строки результата
        return [MemoryHit(int(r[0]), int(r[1]), int(r[2]), float(r[3]), str(r[4]), str(r[5]))  # каждая строка -> MemoryHit(номер, начало, конец, ранг, текст, фрагмент)
                for r in rows]  # по строкам результата

    def stats(self) -> dict:  # сводка хранилища
        total_raw = int(self.memory.db.execute("SELECT COUNT(*) FROM page_meta").fetchone()[0])  # число исходных страниц архива
        retained = int(self.memory.db.execute("SELECT COUNT(*) FROM policy_memory").fetchone()[0])  # число отобранных записей
        return {"raw_pages": total_raw, "retained_records": retained}  # словарь с двумя числами
