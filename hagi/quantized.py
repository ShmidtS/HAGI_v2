"""Детерминированный упакованный троичный экспорт малых байтовых моделей HAGI.

    Формат хранит по пять троичных уровней (−1, 0, 1) в одном байте (основание 3:
    3^5 = 243 ≤ 256, упаковка и распаковка — в hagi.trunk_ops), по одному масштабу FP32 (или
    FP16) на строку матрицы и сдвиги в FP32/FP16. Квантование строки w:
      q_i = clip(round(w_i / s), −1, 1);  восстановленное значение: w̃_i = q_i·s.
    Масштабы заморожены при экспорте; загрузка никогда не пересчитывает их из уже
    квантованных весов.

    Способы выбора масштаба s на строку (scale_mode):
      mean_abs:    s = max(mean(|w_i|), 1e-12);
      mse_opt:     перебор кандидатов s_m = (сумма m наибольших |w_i|) / m по всем m, а также
                   mean|w| и sqrt(mean(w²)); берётся s с наименьшей Σ_i (w_i − q_i·s)²;
      fisher_opt:  кандидаты s_j = (Σ_{i≤j по убыванию |w|} v_i·|w_i|) / (Σ_{i≤j} v_i), v — веса
                   важности; берётся s с наименьшей взвешенной ошибкой Σ_i v_i·(w_i − q_i·s)²;
      explicit:    масштабы задаются извне.

    Контейнер: MAGIC (8 байт) | u32 длина заголовка | заголовок JSON | тело.
    Тело: для каждой матрицы упакованные троичные уровни, затем массив масштабов; затем сдвиги.
    Заголовок: формат, класс модели, context, dim, hidden, режимы, описания матриц и сдвигов.

    Для байтовой модели hagi.lm (атрибуты emb, W1, b1, W2, b2) есть адаптер: W1 и W2
    квантуются в троичные, а эмбеддинг дописывается в конец тела в FP16 (длина emb_fp16_len
    записана в заголовок), сдвиги хранятся как есть.

    Экспортирует: MAGIC, ScaleMode, export_ternary_payload, load_ternary_payload,
    instantiate_ternary_payload, export_hagi_lm_payload, instantiate_hagi_lm_payload."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import copy  # (импортирован, в коде ниже не используется)
import json  # чтение и запись заголовка
import struct  # упаковка целых чисел фиксированной длины
from typing import Literal  # тип-перечисление строк

import numpy as np  # численные массивы
import torch  # тензоры

from hagi.trunk_ops import pack_trits5, unpack_trits5  # упаковка и распаковка пяти троичных уровней в байт

MAGIC = b"HAGITRN1"  # сигнатура формата
ScaleMode = Literal["mean_abs", "mse_opt", "fisher_opt", "explicit"]  # имена способов выбора масштаба


def _levels(row: np.ndarray, scale: float) -> np.ndarray:
    """Троичные уровни строки: q_i = clip(round(w_i / s), −1, 1).

        row — строка весов w; scale — масштаб s. Если s ≤ 0 или не конечен, все уровни 0."""
    if scale <= 0.0 or not np.isfinite(scale):  # масштаб неположительный или не число
        return np.zeros(row.shape, dtype=np.int8)  # нулевые уровни
    return np.clip(np.rint(row / scale), -1, 1).astype(np.int8)  # q_i = clip(round(w_i / s), −1, 1), тип int8


def _best_mse_scale(row: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Найти наилучший масштаб строки среди кандидатов «среднее m наибольших модулей».

        row — строка весов w. Для фиксированного множества активных весов наименьшие квадраты
        даёт масштаб, равный среднему модулю активных весов. Перебираются все длины префикса m:
        s_m = (Σ_{i≤m} a_i) / m, где a — модули весов по убыванию; добавляются mean|w| и
        sqrt(mean(w²)). Для каждого кандидата s считается настоящая ошибка квантования
        E(s) = Σ_i (w_i − q_i·s)²; возвращается (s с наименьшей E, его уровни q, E)."""
    x = np.asarray(row, dtype=np.float32)  # строка в float32
    a = np.abs(x).astype(np.float64)  # модули a_i = |w_i|
    if not np.any(a):  # все веса нулевые
        return 0.0, np.zeros(x.shape, dtype=np.int8), 0.0  # масштаб 0, уровни 0, ошибка 0
    ordered = np.sort(a)[::-1]  # модули по убыванию
    candidates = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)  # s_m = (Σ_{i≤m} a_i) / m для m = 1..n
    # добавляются mean|w| и sqrt(mean(w²)) как запасные кандидаты для случаев,
    # когда округление на границе меняет множество активных весов
    candidates = np.unique(np.concatenate((candidates, [a.mean(), np.sqrt(np.mean(a*a))])))  # все кандидаты без повторов
    best_scale, best_levels, best_error = 0.0, np.zeros(x.shape, dtype=np.int8), float(np.dot(x, x))  # старт: ошибка Σ w_i² (как при s = 0)
    for candidate in candidates:  # по кандидатам
        s = float(np.float32(candidate))  # масштаб в точности float32
        if s <= 0.0:  # неположительный
            continue  # пропустить
        q = _levels(x, s)  # уровни при этом масштабе
        err = float(np.square(x.astype(np.float64) - q.astype(np.float64) * s).sum())  # ошибка строки: E = Σ_i (w_i − q_i·s)²
        if err < best_error:  # ошибка меньше
            best_scale, best_levels, best_error = s, q, err  # запомнить кандидата
    return best_scale, best_levels, best_error  # лучший масштаб, его уровни и ошибка


def _quantize_rows(matrix: np.ndarray, mode: ScaleMode, importance: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float, float]:  # квантовать матрицу построчно
    """Квантовать матрицу построчно; вернуть (уровни, масштабы строк, ошибка цели, обычная ошибка).

        matrix — матрица W (n строк); mode — способ масштаба; importance — веса важности v той же
        формы для fisher_opt. Возвращает уровни Q формы W, вектор масштабов s_i, сумму ошибок
        цели по строкам и сумму обычных ошибок Σ_i (w_i − q_i·s)² (при mean_abs и mse_opt они совпадают)."""
    if mode not in ("mean_abs", "mse_opt", "fisher_opt"):  # неизвестный способ
        raise ValueError(f"unsupported scale mode: {mode}")  # ошибка
    if mode == "fisher_opt":  # нужны веса важности
        if importance is None:  # их нет
            raise ValueError("fisher_opt requires a nonnegative importance matrix")  # ошибка
        importance = np.asarray(importance, dtype=np.float64)  # веса как float64
        if importance.shape != np.asarray(matrix).shape or not np.isfinite(importance).all() or np.any(importance < 0):  # форма другая, есть нечисла или отрицательные
            raise ValueError("importance must match matrix and be finite/nonnegative")  # ошибка
    w = np.asarray(matrix, dtype=np.float32)  # матрица в float32
    if w.ndim != 2:  # не двумерная
        raise ValueError("ternary matrices must be rank 2")  # ошибка
    levels = np.zeros(w.shape, dtype=np.int8)  # уровни Q
    scales = np.zeros(w.shape[0], dtype=np.float32)  # масштабы строк s_i
    objective_error = 0.0  # сумма ошибок цели
    plain_error = 0.0  # сумма обычных ошибок
    for i, row in enumerate(w):  # по строкам
        if mode == "mean_abs":  # средний модуль
            scale = float(np.mean(np.abs(row), dtype=np.float32))  # масштаб строки: s = mean|w_i|
            scale = max(scale, 1e-12)  # не меньше 1e-12: s = max(s, 1e-12)
            q = _levels(row, scale)  # уровни
            row_error = float(np.square(row.astype(np.float64) - q.astype(np.float64)*scale).sum())  # ошибка строки: E = Σ (w_i − q_i·s)²
            row_plain_error = row_error  # обычная ошибка та же
        elif mode == "mse_opt":  # перебор кандидатов
            scale, q, row_error = _best_mse_scale(row)  # лучший масштаб, уровни, ошибка
            row_plain_error = row_error  # обычная ошибка та же
        else:  # fisher_opt: взвешенный перебор
            a = np.abs(np.asarray(row, dtype=np.float64))  # модули a_i
            importance_row = np.asarray(importance[i], dtype=np.float64)  # веса важности строки v_i
            order = np.argsort(-a, kind="stable")  # порядок убывания модулей (устойчивый)
            ao, wo = a[order], importance_row[order]  # модули и веса в этом порядке
            mass = np.cumsum(wo)  # накопленная масса: mass_j = Σ_{i≤j} v_i
            weighted_sum = np.cumsum(wo * ao)  # накопленная взвешенная сумма: wsum_j = Σ_{i≤j} v_i·a_i
            candidates = [float(np.mean(a))]  # первый кандидат — mean|w|
            candidates.extend((weighted_sum[j] / mass[j]) for j in range(len(a)) if mass[j] > 0)  # s_j = wsum_j / mass_j, где mass_j > 0
            best_scale, best_q, best_error = max(float(np.max(a)), 1e-12), np.zeros(a.shape, dtype=np.int8), float(np.dot(importance_row, a*a))  # старт: s = max|w|, уровни 0, ошибка Σ v_i·w_i²
            for candidate in np.unique(np.asarray(candidates, dtype=np.float64)):  # по кандидатам без повторов
                candidate = float(np.float32(candidate))  # масштаб в точности float32
                if candidate <= 0:  # неположительный
                    continue  # пропустить
                q_candidate = _levels(np.asarray(row, dtype=np.float32), candidate)  # уровни кандидата
                e = np.asarray(row, dtype=np.float64) - q_candidate.astype(np.float64) * candidate  # ошибка e_i = w_i − q_i·s
                weighted_error = float(np.dot(importance_row, e*e))  # взвешенная ошибка: E_v = Σ v_i·e_i²
                if weighted_error < best_error:  # взвешенная ошибка меньше
                    best_scale, best_q, best_error = candidate, q_candidate, weighted_error  # запомнить
            scale, q, row_error = best_scale, best_q, best_error  # итог строки
            row_plain_error = float(np.square(np.asarray(row, dtype=np.float64) - q.astype(np.float64)*scale).sum())  # обычная ошибка E = Σ (w_i − q_i·s)²
        levels[i] = q  # записать уровни строки
        scales[i] = scale  # записать масштаб строки
        objective_error += row_error  # сумма ошибок цели: + ошибка строки
        plain_error += row_plain_error  # сумма обычных ошибок: + ошибка строки
    return levels, scales, objective_error, plain_error  # уровни, масштабы, две суммы ошибок


def export_ternary_payload(model, scale_mode: ScaleMode = "mean_abs", importance: dict[str, np.ndarray] | None = None, explicit_scales: dict[str, np.ndarray] | None = None, scale_dtype: str = "f32", bias_dtype: str = "f32") -> tuple[bytes, dict]:
    """Экспортировать ByteLM или TernaryAwareByteLM в переносимый упакованный контейнер.

        model — модель с emb, W1, W2 (каждый с .weight и .bias у W1, W2), context, dim, hidden.
        scale_mode — способ масштаба; importance — словарь весов важности по имени матрицы;
        explicit_scales — заданные масштабы по имени матрицы; scale_dtype и bias_dtype — "f32" или "f16".
        Возвращает (payload, отчёт): payload = MAGIC | u32 длины заголовка | заголовок JSON | тело."""
    if scale_dtype not in ("f32", "f16") or bias_dtype not in ("f32", "f16"):  # неподдерживаемый тип хранения
        raise ValueError("scale_dtype and bias_dtype must be f32 or f16")  # ошибка
    specs = [  # матрицы к экспорту: (имя, параметр, транспонирована ли)
        ("emb.weight", model.emb.weight, False),  # эмбеддинг как есть
        ("W1.weight", model.W1.weight, True),  # W1 транспонируется (строка = выходной нейрон)
        ("W2.weight", model.W2.weight, True),  # W2 транспонируется
    ]
    chunks: list[bytes] = []  # куски тела
    matrices = []  # описания матриц для заголовка
    offset = 0  # текущая позиция в теле
    total_sq_error = 0.0  # сумма обычных ошибок по матрицам
    total_objective_error = 0.0  # сумма ошибок цели
    n_weights = 0  # всего квантованных весов
    for name, param, transposed in specs:  # по матрицам
        src_t = param.detach().cpu().T if transposed else param.detach().cpu()  # исходник на процессоре, при необходимости транспонированный
        src = src_t.numpy().astype(np.float32, copy=True)  # его копия в float32
        if scale_mode == "mean_abs":  # средний модуль
            # среднее считается теми же средствами torch, что и при обучении с квантованием:
            # пересчёт средствами NumPy может отличаться на 1 ulp и разрушить точный
            # круговой поход «упаковка → распаковка → те же логиты»
            scale_t = src_t.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)  # s_j = max(mean_i |w_ji|, 1e-12) по строкам
            level_t = torch.round(src_t / scale_t).clamp(-1, 1).to(torch.int8)  # уровни: q_ji = clip(round(w_ji / s_j), −1, 1)
            scales = scale_t.squeeze(1).numpy().astype(np.float32, copy=True)  # вектор масштабов
            levels = level_t.numpy().astype(np.int8, copy=True)  # матрица уровней
            q = levels.astype(np.float32) * scales[:, None]  # восстановленные веса w̃_ji = q_ji·s_j
            err = float(np.square(src.astype(np.float64) - q.astype(np.float64)).sum())  # ошибка: E = Σ (w_ji − w̃_ji)²
            plain_err = err  # обычная ошибка та же
        elif scale_mode == "explicit":  # заданные масштабы
            if explicit_scales is None or name not in explicit_scales:  # масштабы не переданы
                raise ValueError(f"explicit scales are missing for {name}")  # ошибка
            scales = np.asarray(explicit_scales[name], dtype=np.float32).reshape(-1)  # вектор заданных масштабов
            if len(scales) != src.shape[0] or not np.isfinite(scales).all() or np.any(scales < 0):  # длина не равна числу строк, есть нечисла или отрицательные
                raise ValueError(f"invalid explicit scales for {name}")  # ошибка
            levels = np.stack([_levels(row, float(scale)) for row, scale in zip(src, scales)])  # уровни построчно с заданными масштабами
            q = levels.astype(np.float32) * scales[:, None]  # восстановленные веса w̃_ji = q_ji·s_j
            err = float(np.square(src.astype(np.float64) - q.astype(np.float64)).sum())  # ошибка: E = Σ (w_ji − w̃_ji)²
            plain_err = err  # обычная ошибка та же
        else:  # mse_opt или fisher_opt
            imp = None  # веса важности
            if importance is not None and name in importance:  # для этой матрицы переданы
                imp = np.asarray(importance[name], dtype=np.float64)  # как float64
                if transposed:  # матрица транспонирована
                    imp = imp.T  # веса важности транспонируются тоже
            levels, scales, err, plain_err = _quantize_rows(src, scale_mode, imp)  # построчное квантование
        packed, n = pack_trits5(levels.ravel().tolist())  # упаковать уровни (по строкам подряд) в байты; n — число уровней
        scale_numpy_dtype = "<f4" if scale_dtype == "f32" else "<f2"  # тип хранения масштабов: 4 или 2 байта, младший первым
        scale_blob = scales.astype(scale_numpy_dtype, copy=False).tobytes()  # масштабы в байты
        packed_offset = offset  # начало упакованных уровней
        chunks.append(packed)  # добавить уровни в тело
        offset += len(packed)  # сдвиг позиции
        scale_offset = offset  # начало массива масштабов
        chunks.append(scale_blob)  # добавить масштабы в тело
        offset += len(scale_blob)  # сдвиг позиции
        matrices.append({  # описание матрицы
            "name": name,  # имя
            "parameter_shape": list(param.shape),  # форма параметра
            "source_shape": list(src.shape),  # форма исходника (после транспонирования)
            "transposed": transposed,  # транспонирована ли
            "trit_count": n,  # число уровней
            "packed_offset": packed_offset,  # начало уровней
            "packed_length": len(packed),  # длина уровней в байтах
            "scale_offset": scale_offset,  # начало масштабов
            "scale_count": len(scales),  # число масштабов
            "scale_length": len(scale_blob),  # длина масштабов в байтах
            "scale_dtype": scale_dtype,  # тип хранения масштабов
            "sq_error": plain_err,  # обычная ошибка
            "scale_objective_error": err,  # ошибка цели
        })
        total_sq_error += plain_err  # сумма обычных ошибок: + ошибка матрицы
        total_objective_error += err  # сумма ошибок цели: + ошибка матрицы
        n_weights += n  # всего весов: + число уровней

    biases = []  # описания сдвигов
    bias_numpy_dtype = "<f4" if bias_dtype == "f32" else "<f2"  # тип хранения сдвигов
    for name, param in (("W1.bias", model.W1.bias), ("W2.bias", model.W2.bias)):  # по сдвигам
        arr = param.detach().cpu().numpy().astype(bias_numpy_dtype, copy=True)  # сдвиг в байты выбранного типа
        blob = arr.tobytes()  # байтовое представление
        biases.append({"name": name, "shape": list(param.shape),  # описание сдвига: имя и форма
                       "offset": offset, "length": len(blob), "dtype": bias_dtype})  # начало, длина, тип
        chunks.append(blob)  # добавить сдвиг в тело
        offset += len(blob)  # сдвиг позиции

    header = {  # заголовок JSON
        "format": "HAGI-TERNARY-1",  # имя формата
        "model": type(model).__name__,  # класс модели
        "context": int(model.context),  # контекст
        "dim": int(model.dim),  # размерность
        "hidden": int(model.hidden),  # скрытый слой
        "scale_mode": scale_mode,  # способ масштаба
        "scale_dtype": scale_dtype,  # тип хранения масштабов
        "bias_dtype": bias_dtype,  # тип хранения сдвигов
        "matrices": matrices,  # описания матриц
        "biases": biases,  # описания сдвигов
    }
    header_blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")  # заголовок компактным JSON с сортированными ключами
    body = b"".join(chunks)  # тело — все куски подряд
    payload = MAGIC + struct.pack("<I", len(header_blob)) + header_blob + body  # контейнер: сигнатура, длина заголовка, заголовок, тело
    fp32_size = sum(p.numel() * 4 for p in model.parameters())  # размер модели в FP32: Σ (число элементов·4)
    report = {  # отчёт об экспорте
        "payload_bytes": len(payload),  # размер контейнера
        "header_bytes": len(MAGIC) + 4 + len(header_blob),  # размер заголовочной части
        "body_bytes": len(body),  # размер тела
        "fp32_parameter_bytes": fp32_size,  # размер FP32-модели
        "compression_ratio": fp32_size / len(payload),  # сжатие = FP32-размер / размер контейнера
        "weights": n_weights,  # число квантованных весов
        "sum_squared_weight_error": total_sq_error,  # сумма обычных ошибок
        "scale_objective_value": total_objective_error,  # сумма ошибок цели
        "scale_mode": scale_mode,  # способ масштаба
        "scale_dtype": scale_dtype,  # тип хранения масштабов
        "bias_dtype": bias_dtype,  # тип хранения сдвигов
    }
    return payload, report  # контейнер и отчёт


def load_ternary_payload(model, payload: bytes):
    """Загрузить контейнер в существующую совместимую модель и включить её квантованный путь.

        model — модель той же архитектуры (context, dim, hidden); payload — контейнер из
        export_ternary_payload. Веса переписываются восстановленными значениями w̃ = q·s,
        сдвиги — из тела; устанавливаются флаги _deployed_ternary и заголовок _ternary_header.
        Возвращает ту же model."""
    if not payload.startswith(MAGIC) or len(payload) < len(MAGIC) + 4:  # нет сигнатуры или контейнер короче заголовка длины
        raise ValueError("invalid HAGI ternary payload magic")  # ошибка
    header_len = struct.unpack("<I", payload[len(MAGIC):len(MAGIC)+4])[0]  # длина заголовка — u32 после сигнатуры
    header_start = len(MAGIC) + 4  # начало заголовка
    body_start = header_start + header_len  # начало тела
    if body_start > len(payload):  # тело за пределами контейнера
        raise ValueError("truncated HAGI ternary header")  # ошибка: обрезанный заголовок
    header = json.loads(payload[header_start:body_start].decode("utf-8"))  # разобрать заголовок
    if header.get("format") != "HAGI-TERNARY-1":  # другой формат
        raise ValueError("unsupported ternary format")  # ошибка
    if (header["context"], header["dim"], header["hidden"]) != (model.context, model.dim, model.hidden):  # архитектура не совпадает
        raise ValueError("payload architecture does not match model")  # ошибка
    body = memoryview(payload)[body_start:]  # тело без копирования
    params = dict(model.named_parameters())  # имя параметра -> тензор
    with torch.no_grad():  # без градиентов
        for item in header["matrices"]:  # по матрицам заголовка
            packed = bytes(body[item["packed_offset"]:item["packed_offset"]+item["packed_length"]])  # байты упакованных уровней
            scale_blob = bytes(body[item["scale_offset"]:item["scale_offset"]+item["scale_length"]])  # байты масштабов
            count = int(item["trit_count"])  # число уровней
            levels = unpack_trits5(packed, count).reshape(item["source_shape"])  # распаковать уровни в форму исходника
            scale_np_dtype = "<f4" if item.get("scale_dtype", "f32") == "f32" else "<f2"  # тип хранения масштабов
            scales = np.frombuffer(scale_blob, dtype=scale_np_dtype).astype(np.float32)  # масштабы строк
            if len(scales) != item["scale_count"] or len(scales) != levels.shape[0]:  # число масштабов не равно числу строк
                raise ValueError("scale count does not match matrix rows")  # ошибка
            if item.get("scale_dtype", "f32") not in ("f32", "f16"):  # неизвестный тип хранения
                raise ValueError("unsupported scale storage dtype")  # ошибка
            q_src = levels.astype(np.float32) * scales[:, None]  # восстановленные веса w̃_ji = q_ji·s_j
            q_param = q_src.T if item["transposed"] else q_src  # если матрица экспортировалась транспонированной — транспонировать обратно
            name = item["name"]  # имя параметра
            if name not in params or list(params[name].shape) != item["parameter_shape"]:  # параметра нет или форма другая
                raise ValueError(f"parameter shape mismatch: {name}")  # ошибка
            params[name].copy_(torch.from_numpy(np.ascontiguousarray(q_param)).to(params[name].device))  # записать веса в параметр (на его устройстве)
        for item in header["biases"]:  # по сдвигам заголовка
            blob = bytes(body[item["offset"]:item["offset"]+item["length"]])  # байты сдвига
            bias_storage_dtype = item.get("dtype", "f32")  # тип хранения (по умолчанию f32)
            if bias_storage_dtype not in ("f32", "f16"):  # неизвестный тип
                raise ValueError("unsupported bias storage dtype")  # ошибка
            bias_np_dtype = "<f4" if bias_storage_dtype == "f32" else "<f2"  # numpy-тип
            arr = np.frombuffer(blob, dtype=bias_np_dtype).astype(np.float32).reshape(item["shape"])  # сдвиг в нужной форме
            name = item["name"]  # имя параметра
            if name not in params or list(params[name].shape) != item["shape"]:  # параметра нет или форма другая
                raise ValueError(f"bias shape mismatch: {name}")  # ошибка
            params[name].copy_(torch.from_numpy(arr).to(params[name].device))  # записать сдвиг в параметр
    model._deployed_ternary = True  # модель работает по квантованному пути
    model._ternary_header = header  # сохранить заголовок
    return model  # та же модель




def instantiate_ternary_payload(payload: bytes):
    """Создать поддерживаемую малую модель по метаданным контейнера и загрузить в неё веса.

        payload — контейнер из export_ternary_payload. Класс берётся из заголовка (ByteLM или
        TernaryAwareByteLM), контекст, dim и hidden — оттуда, seed = 0. Возвращает модель."""
    if not payload.startswith(MAGIC) or len(payload) < len(MAGIC) + 4:  # нет сигнатуры или контейнер слишком короткий
        raise ValueError("invalid HAGI ternary payload magic")  # ошибка
    header_len = struct.unpack("<I", payload[len(MAGIC):len(MAGIC)+4])[0]  # длина заголовка
    start = len(MAGIC) + 4  # начало заголовка
    end = start + header_len  # конец заголовка
    if end > len(payload):  # заголовок выходит за пределы
        raise ValueError("truncated HAGI ternary header")  # ошибка
    header = json.loads(payload[start:end].decode("utf-8"))  # разобрать заголовок
    from hagi.lm import ByteLM, TernaryAwareByteLM  # поддерживаемые классы моделей
    classes = {"ByteLM": ByteLM, "TernaryAwareByteLM": TernaryAwareByteLM}  # имя класса -> класс
    cls = classes.get(header.get("model"))  # класс из заголовка или None
    if cls is None:  # класс не поддерживается
        raise ValueError("unsupported model class in ternary payload")  # ошибка
    model = cls(context=int(header["context"]), dim=int(header["dim"]),  # создать модель с параметрами из заголовка
                hidden=int(header["hidden"]), seed=0)  # скрытый слой и зерно 0
    return load_ternary_payload(model, payload)  # загрузить веса и вернуть модель


def export_hagi_lm_payload(model, scale_dtype: str = "f16"  # тип хранения масштабов и сдвигов
                           ) -> tuple[bytes, dict]:  # возвращает (контейнер, метаданные)
    """Адаптер: экспортировать байтовую модель hagi.lm (атрибуты emb, W1, b1, W2, b2 — простые
    параметры, без nn.Linear) в упакованный контейнер HAGITRN1.

        W1 и W2 квантуются в троичные; эмбеддинг хранится в FP16 и дописывается в конец тела
        (длина emb_fp16_len записывается в заголовок); b1 и b2 хранятся как сдвиги.
        Возвращает (payload, meta), где meta — отчёт экспорта с полем hagi_lm (k, dim, hidden, emb_fp16_len)."""
    import numpy as np  # (локальный импорт, имя уже есть в модуле)
    import torch  # (локальный импорт, имя уже есть в модуле)
    class _W:  # представление параметра с атрибутом .weight для существующего экспортёра
        """Обёртка тензора с атрибутом weight; при transpose=True хранится транспонированная копия."""
        def __init__(self, tensor, transpose):  # tensor — тензор; transpose — транспонировать ли
            self.weight = tensor if not transpose else tensor.T.contiguous()  # weight = tensor или tensorᵀ (непрерывная копия)
    # W1 и W2 — троичные; они экспортируются транспонированными
    specs_tr = [("W1.weight", model.W1, True),  # W1
                ("W2.weight", model.W2, True)]  # W2
    class _Model:  # объект-«утка» для export_ternary_payload
        """Объект с атрибутами, которых ждёт export_ternary_payload."""
        pass  # атрибуты задаются ниже
    mm = _Model()  # модель-«утка»
    # экспортёру нужен эмбеддинг — подставляется заглушка 1×1 из нулей (настоящий
    # эмбеддинг едёт хвостом FP16 ниже; модели, обученные с квантованием, экспортируются
    # с троично-согласованными масштабами W1/W2)
    class _Stub:  # заглушка модуля эмбеддинга
        """Заглушка: атрибут weight задаётся после создания."""
        class weight:  # класс-«утка» под запись ("emb.weight", model.emb.weight)
            """Пустой класс-заполнитель; ниже заменяется тензором."""
            pass  # тело пустое
    _stub = _Stub()  # заглушка
    import torch as _torch  # torch под другим именем
    _stub.weight = _torch.zeros(1, 1)  # вес заглушки — нулевой тензор 1×1
    mm.emb = _stub  # эмбеддинг модели-«утки»
    for name, attr, tr in specs_tr:  # по троичным матрицам
        setattr(mm, name.split('.')[0], _W(attr.detach(), tr))  # mm.W1 = _W(W1 без градиента, True); аналогично W2
    mm.context = model.k  # контекст модели-«утки» — это k исходной модели
    mm.dim = model.dim  # размерность
    mm.hidden = model.hidden  # скрытый слой
    _params = [model.emb, model.W1, model.b1, model.W2, model.b2]  # все параметры исходной модели
    mm.parameters = lambda: _params  # экспортёр перебирает параметры для подсчёта размера
    mm.b1 = model.b1.detach()  # b1 без градиента
    mm.b2 = model.b2.detach()  # b2 без градиента
    mm.W1.bias = model.b1.detach()  # сдвиг W1 — b1
    mm.W2.bias = model.b2.detach()  # сдвиг W2 — b2
    payload, meta = export_ternary_payload(mm, scale_mode="mean_abs",  # экспорт модели-«утки» со средним модулем
                                           scale_dtype=scale_dtype,  # тип хранения масштабов
                                           bias_dtype=scale_dtype)  # тип хранения сдвигов — тот же
    # в заголовок записывается настоящий класс модели, чтобы instantiate создал
    # совпадающий шаблон
    real_cls = type(model).__name__  # имя класса исходной модели
    # заголовок читается заново (длина u32 после сигнатуры)
    hl = struct.unpack("<I", payload[len(MAGIC):  # длина заголовка
                                     len(MAGIC) + 4])[0]  # срез из 4 байт
    hd = json.loads(payload[len(MAGIC) + 4:  # разбор заголовка
                            len(MAGIC) + 4 + hl])  # срез заголовка
    hd["hagi_model_class"] = real_cls  # настоящий класс модели
    hd["context"] = int(getattr(model, "k", 16))  # контекст = k модели или 16
    hb = json.dumps(hd, sort_keys=True,  # заголовок снова компактным JSON
                    separators=(",", ":")).encode("utf-8")  # в байты
    # контейнер собирается заново: сигнатура, новая длина, новый заголовок, прежнее тело
    payload = (MAGIC + struct.pack("<I", len(hb)) + hb  # сигнатура + u32 + заголовок
               + payload[len(MAGIC) + 4 + hl:])  # + прежнее тело
    emb_fp16 = model.emb.detach().to(torch.float16).numpy().tobytes()  # эмбеддинг в FP16-байты
    # длина хвоста emb_fp16 обязана лежать в заголовке контейнера, а не только во внешних
    # метаданных: потерянные метаданные молча оставили бы в шаблоне эмбеддинг зерна 0
    header_len = struct.unpack("<I", payload[len(MAGIC):  # длина заголовка
                                              len(MAGIC) + 4])[0]  # срез из 4 байт
    header = json.loads(payload[len(MAGIC) + 4:  # разбор заголовка
                                 len(MAGIC) + 4 + header_len])  # срез заголовка
    header["emb_fp16_len"] = len(emb_fp16)  # длина хвоста эмбеддинга в байтах
    hb = json.dumps(header, sort_keys=True,  # заголовок снова компактным JSON
                     separators=(",", ":")).encode("utf-8")  # в байты
    # контейнер: сигнатура, длина, заголовок, тело, хвост эмбеддинга
    payload = (MAGIC + struct.pack("<I", len(hb)) + hb  # сигнатура + u32 + заголовок
               + payload[len(MAGIC) + 4 + header_len:] + emb_fp16)  # + тело + хвост
    meta["hagi_lm"] = {"k": model.k, "dim": model.dim,  # сведения о модели в метаданных
                       "hidden": model.hidden,  # скрытый слой
                       "emb_fp16_len": len(emb_fp16)}  # длина хвоста
    return payload, meta  # контейнер и метаданные


def instantiate_hagi_lm_payload(payload: bytes, meta: dict | None = None):  # восстановить модель hagi.lm из контейнера
    """Восстановить байтовую модель hagi.lm из контейнера: заголовок управляет шаблоном,
    эмбеддинг читается из хвоста FP16.

        payload — контейнер из export_hagi_lm_payload; meta — метаданные экспорта (запасной
        источник emb_fp16_len, если в заголовке его нет). Класс шаблона берётся из
        hagi_model_class заголовка (по умолчанию CertifiedStepLM), k, dim, hidden — оттуда.
        Возвращает модель с восстановленными emb, W1, b1, W2, b2."""
    import json as _json  # (локальный импорт, не используется)
    import numpy as np  # (локальный импорт, имя уже есть в модуле)
    import struct as _struct  # (локальный импорт, не используется)
    import torch  # (локальный импорт, имя уже есть в модуле)
    if not payload.startswith(MAGIC):  # нет сигнатуры
        raise ValueError("invalid HAGI ternary payload magic")  # ошибка
    header_len = struct.unpack("<I", payload[len(MAGIC):len(MAGIC)+4])[0]  # длина заголовка
    start = len(MAGIC) + 4  # начало заголовка
    header = json.loads(payload[start:start + header_len].decode("utf-8"))  # разобрать заголовок
    meta = meta or {}  # метаданные (пустой словарь, если None)
    body = payload[start + header_len:]  # тело
    import hagi.lm as _lm  # модуль моделей
    cls_name = header.get("hagi_model_class", "CertifiedStepLM")  # имя класса из заголовка
    tpl_cls = getattr(_lm, cls_name, _lm.CertifiedStepLM)  # класс; если его нет — CertifiedStepLM
    if cls_name == "RopeAttentionByteLM":  # у этой модели нет аргумента k
        template = tpl_cls(dim=int(header["dim"]),  # создать шаблон по dim и hidden
                           hidden=int(header["hidden"]), seed=0)  # зерно 0
    else:  # обычная модель
        template = tpl_cls(k=int(header["context"]),  # шаблон по k, dim, hidden
                           dim=int(header["dim"]),  # размерность
                           hidden=int(header["hidden"]), seed=0)  # скрытый слой, зерно 0
    params = {"emb.weight": None, "W1.weight": template.W1,  # целевые параметры матриц; эмбеддинг — None (едёт хвостом)
              "W2.weight": template.W2}  # W2
    for m in header["matrices"]:  # по матрицам заголовка
        if m["name"] == "emb.weight":  # это заглушка
            continue  # настоящий эмбеддинг восстанавливается ниже
        tgt = params[m["name"]]  # целевой параметр
        src_shape = tuple(m["source_shape"])  # форма исходника
        rows, cols = src_shape  # строки и столбцы
        # масштабы читаются по смещению и длине из описания матрицы
        scales = np.frombuffer(body[m["scale_offset"]:m["scale_offset"]  # срез байтов масштабов
                                    + m["scale_length"]],  # до конца массива
                               dtype="<f4" if m["scale_dtype"] == "f32"  # тип: 4 байта
                               else "<f2").astype(np.float32)  # или 2 байта; результат в float32
        packed = body[m["packed_offset"]:m["packed_offset"]  # срез байтов уровней
                      + m["packed_length"]]  # до конца
        # уровни записаны подряд по всей матрице (levels.ravel()), а не по строкам
        lv = unpack_trits5(packed, rows * cols)  # распаковать rows·cols уровней
        mat = (lv.astype(np.float32).reshape(rows, cols) * scales[:, None])  # восстановленные веса: w̃_ji = q_ji·s_j
        t = torch.as_tensor(mat)  # тензор
        # адаптер выше хранит W1ᵀᵀ = исходный W1, то есть source_shape совпадает с формой
        # параметра; дополнительное транспонирование не нужно
        if tuple(t.shape) != tuple(tgt.shape) and m.get("transposed"):  # форма не совпала и матрица была транспонирована
            t = t.T.contiguous()  # настоящий случай nn.Linear — транспонировать
        if tuple(t.shape) != tuple(tgt.shape):  # форма всё ещё не совпадает
            raise ValueError(  # ошибка
                f"shape mismatch for {m['name']}: {tuple(t.shape)} vs {tuple(tgt.shape)}")  # текст с формами
        with torch.no_grad():  # без градиентов
            tgt.copy_(t.to(tgt.dtype))  # записать веса в параметр
    bd = header.get("bias_dtype", "f32")  # тип хранения сдвигов
    for bm in header["biases"]:  # по сдвигам заголовка
        name, tgt = (("W1.bias", template.b1) if bm["name"] == "W1.bias"  # сдвиг W1 -> b1
                     else ("W2.bias", template.b2))  # иначе сдвиг W2 -> b2
        arr = np.frombuffer(body[bm["offset"]:bm["offset"] + bm["length"]],  # байты сдвига
                            dtype="<f2" if bd == "f16" else "<f4")  # 2 или 4 байта на число
        with torch.no_grad():  # без градиентов
            tgt.copy_(torch.as_tensor(arr.astype(np.float32),  # записать сдвиг (в float32)
                                      dtype=tgt.dtype))  # в типе параметра
    # эмбеддинг в FP16 дописан после сдвигов; заголовок — единственный источник истины,
    # метаданные остаются запасным вариантом
    emb_len = int(header.get("emb_fp16_len",  # длина хвоста из заголовка
                             (meta or {}).get("hagi_lm", {})  # иначе из метаданных
                             .get("emb_fp16_len", 0)))  # иначе 0
    if emb_len:  # хвост есть
        tail = body[len(body) - emb_len:]  # последние emb_len байт тела
        emb = np.frombuffer(tail, dtype="<f2").astype(np.float32)  # эмбеддинг из FP16 в float32
        with torch.no_grad():  # без градиентов
            template.emb.copy_(torch.as_tensor(  # записать в эмбеддинг шаблона
                emb.reshape(template.emb.shape), dtype=template.emb.dtype))  # в его форме и типе
    return template  # восстановленная модель
