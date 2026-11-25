"""Нейронное волокно — приближённая модель функции с малым числом аргументов.

Сеть: один скрытый слой из hidden нейронов и один линейный выход.
    h   = tanh(x·W1 + b1)          — скрытый слой, h имеет форму (N, hidden)
    ŷ  = h·W2 + b2                 — выход, форма (N, 1)
Входы и ответы заранее масштабируются: x/sx и y/sy, где sx = max(1, max|x|),
sy = max(1, max|y|), — так что все значения лежат в [−1, 1].

Обучение — полный батч, Adam: средний квадрат ошибки
    loss = (1/N)·Σ (ŷ_i − y_i)²
градиенты: dp = (2/N)·err; dW2 = hᵀ·dp; db2 = Σ dp; dh = (dp·W2ᵀ)·(1 − h²);
dW1 = xᵀ·dh; db1 = Σ dh. Шаг Adam: m = β1·m + (1−β1)·g; v = β2·v +
(1−β2)·g²; m̂ = m/(1−β1^t); v̂ = v/(1−β2^t); p −= lr·m̂/(√v̂ + ε).

После каждой эпохи считается ошибка на проверочной выборке; в модель
сохраняются веса эпохи с наименьшей проверочной ошибкой.
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов
from dataclasses import dataclass  # декоратор класса-записи и фабрика значений по умолчанию
from typing import Any, Sequence  # типы для аннотаций
import numpy as np  # численные массивы

# ------------------------------- Нейронное волокно ----------------------------------  # раздел нейронного волокна


@dataclass  # класс-запись: инициализатор и представление генерируются автоматически
class NeuralFiber:
    """Приближённая нейронная сеть «вход -> tanh-скрытый слой -> линейный выход».

        arity — число аргументов; input_scale sx и target_scale sy — делители
        входов и ответов; w1 (arity × hidden), b1 (1 × hidden), w2 (hidden × 1),
        b2 (1 × 1) — веса; train_loss и validation_loss — средний квадрат
        ошибки в исходном масштабе; validation_exact_accuracy — доля
        проверочных примеров, где округлённый прогноз совпал с ответом."""

    arity: int  # число аргументов функции
    input_scale: float  # делитель входов sx
    target_scale: float  # делитель ответов sy
    w1: np.ndarray  # веса вход -> скрытый слой
    b1: np.ndarray  # смещения скрытого слоя
    w2: np.ndarray  # веса скрытый слой -> выход
    b2: np.ndarray  # смещение выхода
    train_loss: float = float("nan")  # ошибка на обучении, по умолчанию NaN
    validation_loss: float = float("nan")  # ошибка на проверке, по умолчанию NaN
    validation_exact_accuracy: float = float("nan")  # точная точность на проверке, по умолчанию NaN

    @staticmethod  # метод без доступа к экземпляру
    def fit(  # обучить волокно
        inputs: Sequence[Sequence[int]],  # обучающие входы (N, arity)
        targets: Sequence[int],  # обучающие ответы (N,)
        val_inputs: Sequence[Sequence[int]],  # проверочные входы
        val_targets: Sequence[int],  # проверочные ответы
        seed: int,  # зерно генератора случайных чисел
        hidden: int = 24,  # число скрытых нейронов
        epochs: int = 140,  # число эпох
        lr: float = 0.025,  # скорость обучения
    ) -> "NeuralFiber":  # возвращает обученное волокно
        """Обучить волокно на примерах inputs -> targets методом Adam полным батчем.

            Возвращает волокно с весами эпохи, давшей наименьшую ошибку на
            проверочной выборке val_inputs -> val_targets."""
        x = np.asarray(inputs, dtype=np.float64).reshape(-1, len(inputs[0]))  # матрица входов x формы (N, arity)
        y = np.asarray(targets, dtype=np.float64).reshape(-1, 1)  # столбец ответов y формы (N, 1)
        xv = np.asarray(val_inputs, dtype=np.float64).reshape(-1, x.shape[1])  # проверочные входы xv
        yv = np.asarray(val_targets, dtype=np.float64).reshape(-1, 1)  # проверочные ответы yv
        sx = max(1.0, float(np.max(np.abs(x))))  # масштаб входа sx = max(1, max|x|)
        sy = max(1.0, float(np.max(np.abs(y))))  # масштаб выхода sy = max(1, max|y|)
        x, y, xv, yv = x / sx, y / sy, xv / sx, yv / sy  # нормировка: x/sx, y/sy, xv/sx, yv/sy
        rng = np.random.default_rng(seed)  # генератор случайных чисел с зерном seed
        w1 = rng.normal(0.0, 0.32, (x.shape[1], hidden))  # w1 — нормальные значения со средним 0 и отклонением 0.32, форма (arity, hidden)
        b1 = np.zeros((1, hidden), dtype=np.float64)  # b1 — нули формы (1, hidden)
        w2 = rng.normal(0.0, 0.12, (hidden, 1))  # w2 — нормальные значения со средним 0 и отклонением 0.12, форма (hidden, 1)
        b2 = np.zeros((1, 1), dtype=np.float64)  # b2 — нули формы (1, 1)
        params = [w1, b1, w2, b2]  # список обучаемых параметров
        m = [np.zeros_like(p) for p in params]  # первый момент Адама m — нули
        v = [np.zeros_like(p) for p in params]  # второй момент Адама v — нули
        beta1, beta2, eps = 0.9, 0.999, 1e-8  # коэффициенты Адама β1, β2 и малое ε
        best = [p.copy() for p in params]  # лучшие веса — копии начальных
        best_val = float("inf")  # наименьшая проверочная ошибка — пока +∞
        step = 0  # номер шага t
        for _ in range(epochs):  # по эпохам
            h = np.tanh(x @ w1 + b1)  # скрытый слой h = tanh(x·W1 + b1)
            pred = h @ w2 + b2  # прогноз ŷ = h·W2 + b2
            err = pred - y  # ошибка err = ŷ − y
            loss = float(np.mean(err * err))  # ошибка: loss = (1/N)·Σ err_i²
            dp = (2.0 / len(x)) * err  # градиент по выходу dp = (2/N)·err
            dw2 = h.T @ dp  # градиент: dW2 = hᵀ·dp
            db2 = np.sum(dp, axis=0, keepdims=True)  # градиент: db2 = Σ_i dp_i
            dh = (dp @ w2.T) * (1.0 - h * h)  # dh = (dp·W2ᵀ)·(1 − h²) — производная tanh
            dw1 = x.T @ dh  # градиент: dW1 = xᵀ·dh
            db1 = np.sum(dh, axis=0, keepdims=True)  # градиент: db1 = Σ_i dh_i
            grads = [dw1, db1, dw2, db2]  # градиенты в порядке параметров
            step += 1  # шаг t + 1
            for i, (p, g) in enumerate(zip(params, grads)):  # по парам (параметр, градиент)
                m[i] = beta1 * m[i] + (1 - beta1) * g  # первый момент: m = β1·m + (1−β1)·g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)  # второй момент: v = β2·v + (1−β2)·g²
                mh = m[i] / (1 - beta1**step)  # поправленный момент m̂ = m/(1 − β1^t)
                vh = v[i] / (1 - beta2**step)  # поправленный момент v̂ = v/(1 − β2^t)
                p -= lr * mh / (np.sqrt(vh) + eps)  # шаг p −= lr·m̂/(√v̂ + ε)
            hv = np.tanh(xv @ w1 + b1)  # скрытый слой проверочной выборки
            pv = hv @ w2 + b2  # прогноз на проверке
            vl = float(np.mean((pv - yv) ** 2))  # проверочная ошибка vl = mean((pv − yv)²)
            if np.isfinite(vl) and vl < best_val:  # ошибка конечна и меньше рекордной
                best_val = vl  # новый рекорд
                best = [p.copy() for p in params]  # сохранить копии весов
        w1, b1, w2, b2 = best  # взять лучшие веса
        h = np.tanh(x @ w1 + b1)  # скрытый слой на обучении
        tr_loss = float(np.mean((h @ w2 + b2 - y) ** 2)) * sy * sy  # ошибка обучения в исходном масштабе: mean((ŷ − y)²)·sy²
        hv = np.tanh(xv @ w1 + b1)  # скрытый слой на проверке
        predv = (hv @ w2 + b2).reshape(-1) * sy  # прогноз в исходном масштабе: ŷ·sy
        val_loss = float(np.mean((predv - np.asarray(val_targets, dtype=float)) ** 2))  # проверочная ошибка в исходном масштабе
        exact = float(  # точная точность
            np.mean(  # доля совпадений
                np.rint(predv).astype(np.int64)  # округлённый прогноз round(ŷ)
                == np.asarray(val_targets, dtype=np.int64)  # сравнение с целым ответом
            )  # конец mean
        )  # конец float
        return NeuralFiber(  # создать волокно
            len(inputs[0]), sx, sy, w1, b1, w2, b2, tr_loss, val_loss, exact  # arity, масштабы, веса, ошибки, точность
        )  # конец конструктора

    def predict(self, xs: Sequence[int]) -> float:  # прогноз для входа xs
        """Приближённое значение функции на входе xs: выход сети в исходном масштабе."""
        if len(xs) != self.arity:  # число аргументов не совпадает
            raise ValueError("wrong neural fiber arity")  # ошибка: неверная арность
        x = np.asarray(xs, dtype=np.float64).reshape(1, -1) / self.input_scale  # строка x/sx формы (1, arity)
        return float(  # вернуть число
            (np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2)[0, 0]  # выход сети: (tanh(x·W1 + b1)·W2 + b2)[0, 0]
            * self.target_scale  # умножить на sy — вернуться к исходному масштабу
        )  # конец float

    def to_dict(self) -> dict[str, Any]:  # словарь для JSON
        """Словарь полей волокна; массивы превращаются в списки."""
        return {  # вернуть словарь
            "arity": self.arity,  # число аргументов
            "input_scale": self.input_scale,  # масштаб входа
            "target_scale": self.target_scale,  # масштаб выхода
            "w1": self.w1.tolist(),  # веса w1 списком
            "b1": self.b1.tolist(),  # смещения b1 списком
            "w2": self.w2.tolist(),  # веса w2 списком
            "b2": self.b2.tolist(),  # смещение b2 списком
            "train_loss": self.train_loss,  # ошибка обучения
            "validation_loss": self.validation_loss,  # ошибка проверки
            "validation_exact_accuracy": self.validation_exact_accuracy,  # точная точность
        }  # конец словаря

    @staticmethod  # метод без доступа к экземпляру
    def from_dict(d: dict[str, Any]) -> "NeuralFiber":  # восстановить волокно
        """Восстановить волокно из словаря, созданного to_dict."""
        return NeuralFiber(  # собрать волокно
            int(d["arity"]),  # арность
            float(d["input_scale"]),  # масштаб входа
            float(d["target_scale"]),  # масштаб выхода
            np.asarray(d["w1"], dtype=np.float64),  # w1 из списка
            np.asarray(d["b1"], dtype=np.float64),  # b1 из списка
            np.asarray(d["w2"], dtype=np.float64),  # w2 из списка
            np.asarray(d["b2"], dtype=np.float64),  # b2 из списка
            float(d.get("train_loss", float("nan"))),  # ошибка обучения (NaN, если нет)
            float(d.get("validation_loss", float("nan"))),  # ошибка проверки (NaN, если нет)
            float(d.get("validation_exact_accuracy", float("nan"))),  # точная точность (NaN, если нет)
        )  # конец конструктора

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'NeuralFiber',  # имя интерфейса
]  # конец перечня
