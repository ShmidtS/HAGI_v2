#!/usr/bin/env python3  # интерпретатор Python 3
"""Фиксированный вневыборочный тест (OOS): честная кросс-энтропия модели на текстах, которые никогда не поступали в обучение.

Тексты удержаны вне доменов (их байты не подаются в ingest), поэтому набор одинаков при каждом запуске. Для каждого текста: X — окна из k байтов с шагом 5, y — следующий за окном байт; берутся первые 2000 пар; печатается CE = −ln P(истинный байт) по модели u.model.
"""  # документация модуля
import sys  # доступ к sys.path
from pathlib import Path  # вычисление корня проекта
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # добавить корень проекта в путь импорта
import numpy as np  # численные массивы
import torch as t  # тензорная библиотека
from hagi.selfdev import unified  # объединённая модель Unified

OOS_TEXTS = {  # словарь: имя текста -> его байты
    "english_tech": (  # технический текст на английском
        "The compiler applies constant folding before generating "  # строка-код: текст теста
        "assembly. Register allocation uses graph coloring. The "  # строка-код: текст теста
        "scheduler reorders instructions to hide memory latency. "  # строка-код: текст теста
        "Cache lines align to 64 bytes. The branch predictor is a "  # строка-код: текст теста
        "two-bit saturating counter. " * 10).encode(),  # повторить текст 10 раз и закодировать в байты UTF-8
    "python_code": (  # фрагмент кода на Python
        "def train_model(loader, epochs, lr):\n"  # строка-код: текст теста
        "    model = Net()\n"  # строка-код: текст теста
        "    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)\n"  # строка-код: текст теста
        "    for epoch in range(epochs):\n"  # строка-код: текст теста
        "        for batch in loader:\n"  # строка-код: текст теста
        "            loss = criterion(model(x), y)\n"  # строка-код: текст теста
        "            loss.backward()\n"  # строка-код: текст теста
        "            optimizer.step()\n" * 10).encode(),  # повторить текст 10 раз и закодировать в байты UTF-8
    "russian_text": (  # текст на русском
        "Компилятор применяет свёртку констант перед генерацией "  # строка-код: текст теста
        "ассемблера. Распределение регистров использует раскраску "  # строка-код: текст теста
        "графа. Планировщик переставляет инструкции. " * 10).encode("utf-8"),  # повторить текст 10 раз и закодировать в байты UTF-8
}


def main():
    """Измерить и напечатать CE удержанных текстов под загруженной моделью: OOS имя: значение."""
    u = unified.Unified.load()  # загрузить объединённую модель с диска
    for name, text in OOS_TEXTS.items():  # по всем удержанным текстам
        idx = np.frombuffer(text, dtype=np.uint8).astype(np.int64)  # idx_i = номер байта i текста (0..255) в int64
        k = u.k  # длина контекста модели
        X = np.stack([idx[i:i + k] for i in range(0, len(idx) - k, 5)])  # X_j = idx[5j .. 5j+k−1]: окна длины k с шагом 5
        y = idx[k::5][:len(X)]  # y_j = idx[5j + k]: байт, следующий за окном j (срез с k с шагом 5, обрезан по числу строк X)
        with t.no_grad():  # без вычисления градиентов
            ce = float(t.nn.functional.cross_entropy(  # ce = среднее по 2000 парам от −ln softmax(logits)[y]
                u.model.logits(t.as_tensor(X[:2000], dtype=t.long)),  # логиты модели на первых 2000 окнах
                t.as_tensor(y[:2000], dtype=t.long)))  # истинные следующие байты этих окон
        print(f"OOS {name}: {ce:.4f}")  # напечатать имя текста и CE с 4 знаками


if __name__ == "__main__":  # запуск только как сценарий
    main()  # измерить и напечатать CE
