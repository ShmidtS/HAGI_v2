"""Семантические векторы текста: обёртка над моделью embeddinggemma-2.

    Назначение: дать символьно-соседскому слою векторы смысла текстов.
    Класс GemmaEmbedder создаётся один раз на процесс, загружает модель из
    локального каталога (без выхода в сеть) и считает вектор текста так:
      1) токенизация текстов с дополнением до одной длины и обрезкой;
      2) h̃[n, j] = Σ_t h[n, t, j]·mask[n, t] / Σ_t mask[n, t] — среднее по
         настоящим токенам скрытых состояний h формы (N, T, 768);
         mask = 1 у токена, 0 у дополнения;
      3) v[n, j] = h̃[n, j] / sqrt(Σ_k h̃[n, k]²) — нормирование на единичную
         длину (L2).
    Сходство двух текстов — скалярное произведение их векторов.

    Экспортирует: класс GemmaEmbedder (методы get, embed, similarity).
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов

import os  # доступ к переменным окружения
from pathlib import Path  # работа с путями файлов

# каталог модели по умолчанию в локальном кэше huggingface (три части пути склеены в одну строку)
_DEFAULT_SNAP = (  # путь к снимку модели embeddinggemma-2
    r"C:\Users\shmid\.cache\huggingface\hub"  # корень локального кэша моделей
    r"\models--google--embeddinggemma-2\snapshots"  # подкаталог модели и её снимков
    r"\f9972ac22e696a105c482202646979e33e4b5a1e"  # каталог конкретного снимка
)


class GemmaEmbedder:
    """Ленивый одиночный экземпляр над embeddinggemma-2 (768 измерений,
    среднее по токенам + нормирование L2). Работает без сети из локального кэша."""

    _instance = None  # единственный созданный экземпляр (пока не создан — None)

    def __init__(self, snapshot: str | None = None):  # создать обёртку; snapshot — каталог модели или None
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # если HF_HUB_OFFLINE не задан, поставить "1" (не ходить в сеть)
        from transformers import AutoModel, AutoTokenizer  # загрузка модели и токенизатора библиотеки transformers

        self.snapshot = snapshot or os.environ.get("HAGI_EMBED_SNAPSHOT", _DEFAULT_SNAP)  # каталог = snapshot, иначе переменная HAGI_EMBED_SNAPSHOT, иначе путь по умолчанию
        if not Path(self.snapshot).exists():  # каталога модели нет на диске
            raise FileNotFoundError(  # прервать работу с ошибкой
                f"embeddinggemma-2 snapshot not found: {self.snapshot}"  # текст ошибки с путём
            )
        self.tok = AutoTokenizer.from_pretrained(self.snapshot)  # токенизатор из каталога модели
        self.model = AutoModel.from_pretrained(self.snapshot)  # сама модель из каталога модели
        # учитель должен лежать на том же графическом процессоре, что и ученик
        try:  # перенос на GPU необязателен: при любой ошибке остаёмся на процессоре
            import torch as _t  # библиотека работы с тензорами
            if _t.cuda.is_available():  # графический процессор доступен
                self.model = self.model.cuda()  # перенести модель на GPU
        except Exception:  # любая ошибка (нет torch, нет CUDA)
            pass  # остаёмся на процессоре (векторы те же, только медленнее)
        self.model.eval()  # режим оценки: без обучения и без случайного шума

    @classmethod  # метод класса: получает сам класс, а не экземпляр
    def get(cls) -> "GemmaEmbedder":  # вернуть единственный экземпляр класса
        if cls._instance is None:  # экземпляр ещё не создавался
            cls._instance = cls()  # создать и запомнить его
        return cls._instance  # вернуть сохранённый экземпляр

    def embed(self, texts: list[str]):
        """Векторы текстов: среднее по токенам, нормированное на единичную длину; тензор [N, 768].

            texts — список из N текстов. Возвращается тензор формы (N, 768):
            v[n, j] = h̃[n, j] / sqrt(Σ_k h̃[n, k]²),
            h̃[n, j] = Σ_t h[n, t, j]·mask[n, t] / Σ_t mask[n, t]."""
        import torch  # библиотека работы с тензорами

        b = self.tok(texts, return_tensors="pt", padding=True, truncation=True)  # словарь тензоров input_ids и attention_mask; короткие тексты дополнены, длинные обрезаны
        device = next(self.model.parameters()).device  # устройство (процессор или GPU), где лежит первый параметр модели
        b = {k: v.to(device) for k, v in b.items()}  # каждый тензор входа перенести на устройство модели
        with torch.no_grad():  # без вычисления градиентов (только прямой проход)
            out = self.model(**b)  # прогнать модель; out.last_hidden_state — скрытые состояния h формы (N, T, 768)
        mask = b["attention_mask"].unsqueeze(-1).float()  # маска формы (N, T, 1): 1 — настоящий токен, 0 — дополнение
        v = (out.last_hidden_state * mask).sum(1) / mask.sum(1)  # среднее по токенам: h̃[n, j] = Σ_t h[n, t, j]·mask[n, t] / Σ_t mask[n, t]
        return torch.nn.functional.normalize(v, dim=-1)  # нормируем: v[n, j] ← v[n, j] / sqrt(Σ_k v[n, k]²)

    def similarity(self, a: str, b: str) -> float:  # сходство текстов a и b: число от −1 до 1
        v = self.embed([a, b])  # векторы обоих текстов, форма (2, 768)
        return float(v[0] @ v[1])  # скалярное произведение Σ_j v[0, j]·v[1, j]
