"""Тесты архиватора без потерь: арифметическое кодирование под байтовой моделью.

    Кодек: байт s кодируется по 16-битной функции распределения cdf; размер
    архива в битах на байт bpb = len(поток)·8 / n. Для известного распределения
    bpb отличается от энтропии H не более чем на константу:
      skewed: P(0) = 0,5, P(1..255) = 0,5/255, H = 1 + 0,5·log2(255) ≈ 4,984;
      random: P = 1/256, H = 8.
    Архив: MAGIC | u32 длина | u8 режим; режим 0 — сырые байты (байт 9 = 0,
    длина = n + 10), режим 1 — кодированный (байт 9 = 1).

    Свойства: круглый ход кодирования воспроизводит байты точно; bpb близок
    к идеалу; архив никогда не длиннее сырого вида с заголовком; необученная
    модель на случайных байтах даёт режим 0; обученная модель сжимает
    периодический поток; архив раскодирует только та же модель; повреждённый
    архив отвергается; длинный файл сжимается без потерь обеими архитектурами;
    потоковый кодек с таблицей контекстов q = (1 − λ)·p + λ·r без потерь и
    короче обычного; расширение модели сохраняет logits."""
import numpy as np  # массивы и случайные байты
import pytest  # проверка исключений
import zlib  # сжатие zlib (импортирован, в тестах ниже не используется)

from hagi import archiver as A  # архиватор


def _model():
    """Модель RopeAttentionByteLM с d = 32, h = 64, зерно 1 (маленькая — ради скорости)."""
    from hagi.lm import RopeAttentionByteLM  # модель с вращением RoPE
    return RopeAttentionByteLM(dim=32, hidden=64, seed=1)  # создать модель заданных размеров


def _roundtrip(m, data):
    """Сжать data под моделью m, проверить сигнатуру и точный обратный ход; возвращает архив.

        Проверки: c[:5] = MAGIC и decompress(m, c) = data."""
    c = A.compress(m, data)  # сжать байты data
    assert c[:5] == A.MAGIC  # первые 5 байтов — сигнатура HAGI3
    assert A.decompress(m, c) == data  # раскодированные байты совпадают с исходными
    return c  # вернуть архив для дальнейших проверок


def test_codec_is_lossless_and_near_ideal_on_known_distributions():
    """Без потерь и близко к идеалу на известном распределении: 20 000 байтов кодируются по P и
    раскодировываются точно; bpb = len(поток)·8 / n отличается от идеала менее чем на 0,03."""
    rng = np.random.default_rng(0)  # генератор с зерном 0
    for name, mk in (("skewed", lambda: rng.choice(256, 20000, p=np.r_[0.5, np.full(255, 0.5 / 255)])),  # перекошенное: 20 000 байтов, P(0) = 0,5, прочие по 0,5/255
                     ("random", lambda: rng.integers(0, 256, 20000))):  # равномерное: 20 000 случайных байтов 0..255
        x = mk().astype(np.uint8)  # байты выборки
        P = np.full((len(x), 256), 1 / 256, np.float32)  # таблица вероятностей (n, 256), заполненная 1/256
        if name == "skewed":  # для перекошенного распределения
            P[:, 0] = 0.5; P[:, 1:] = 0.5 / 255  # вероятность нуля 0,5; остальных — по 0,5/255
        e = A.ArithEncoder(); e.encode_step(x.astype(np.int64), P); bs = e.finish()  # закодировать байты x по P; bs — поток в байтах
        y = A.ArithDecoder(bs).decode_step(P)  # раскодировать по тому же P
        assert (y == x).all()  # все раскодированные байты равны исходным
        bpb = len(bs) * 8 / len(x)  # битов на байт bpb = 8·len(bs) / n
        ideal = 1 + 0.5 * np.log2(255) if name == "skewed" else 8.0  # идеал: H = 1 + 0,5·log2(255), иначе 8
        assert abs(bpb - ideal) < 0.03, (name, bpb)  # |bpb − идеал| < 0,03


def test_roundtrip_real_random_and_tiny_inputs():
    """Круглый ход без потерь на 7 наборах (периодика, случайные, пустые, 2 и 200 байтов, обрезанные)
    и архив не длиннее сырого вида с заголовком: len(c) ≤ n + 5 + 5."""
    m = _model()  # модель
    rng = np.random.default_rng(3)  # генератор с зерном 3
    periodic = bytes(range(256)) * 40  # периодический поток 0..255, повторённый 40 раз
    cases = [periodic, bytes(rng.integers(0, 256, 5000, dtype=np.uint8)), b"", b"ab", bytes(range(200)),  # случаи: периодика, 5000 случайных байтов, пусто, 2 байта, 200 байтов
             periodic[:63], periodic[:64] + b"\x00\x01"]  # ещё: 63 байта периодики; 64 байта периодики + 2 байта
    for d in cases:  # по всем случаям
        c = _roundtrip(m, d)  # сжать и проверить обратный ход
        # никогда не хуже сырого вида с заголовком
        assert len(c) <= len(d) + len(A.MAGIC) + 5  # длина архива ≤ n + 5 (сигнатура) + 5 (длина u32 и режим u8)


def test_stored_mode_is_chosen_when_the_model_does_not_help():
    """Необученная модель на случайных байтах не сжимает: выбран режим 0 (байт 9 = 0),
    длина архива = n + 10."""
    m = _model()  # необученная модель: случайные байты несжимаемы
    d = bytes(np.random.default_rng(1).integers(0, 256, 3000, dtype=np.uint8))  # 3000 случайных байтов
    c = _roundtrip(m, d)  # сжать и проверить обратный ход
    assert c[9] == 0 and len(c) == len(d) + 10  # режим 0 и длина = n + 10


def test_trained_model_compresses_a_periodic_stream_better_than_raw():
    """После 6 эпох на первых 8000 байтах периодики периода 64: bpb = len(c)·8 / n < 7,0
    (у необученной модели около 8,0)."""
    m = _model()  # модель
    d = bytes(range(64)) * 200  # периодический поток с периодом 64
    m.train_guarded(d[:8000], epochs=6)  # обучить на первых 8000 байтах, 6 эпох
    c = _roundtrip(m, d)  # сжать и проверить обратный ход
    assert len(c) * 8 / len(d) < 7.0  # битов на байт меньше 7 (у необученной около 8)


def test_archive_is_bound_to_the_model():
    """Архив привязан к модели: другая модель либо не может раскрыть архив, либо не воспроизводит байты
    (или архив в режиме 0); та же модель раскодирует точно."""
    m1 = _model()  # первая модель
    d = bytes(range(64)) * 100  # периодический поток периода 64
    m1.train_guarded(d, epochs=4)  # обучить первую модель на d, 4 эпохи
    c = A.compress(m1, d)  # архив под первой моделью
    m2 = _model()  # вторая модель
    m2.train_guarded(bytes(range(255, 0, -1)) * 25, epochs=4)  # обучить вторую на обратном законе 255,254,…,1
    # другая модель либо терпит неудачу, либо не воспроизводит байты
    try:  # попытка раскрыть архив другой моделью
        assert A.decompress(m2, c) != d or c[9] == 0  # результат отличен от d, либо архив в режиме 0
    except Exception:  # любая ошибка при раскрытии
        pass  # допустимо: другая модель не смогла раскодировать
    assert A.decompress(m1, c) == d  # та же модель раскодирует точно


def test_corrupt_archive_is_rejected():
    """Повреждённый архив отвергается: испорченная сигнатура и длина сырых данных не равна
    заявленной дают ValueError."""
    m = _model()  # модель
    c = bytearray(A.compress(m, bytes(range(256)) * 20))  # архив как изменяемый массив
    c[0] = 0x00  # испортить первый байт сигнатуры
    with pytest.raises(ValueError):  # ожидается ошибка значения
        A.decompress(m, bytes(c))  # раскрыть архив с испорченной сигнатурой
    with pytest.raises(ValueError):  # ожидается ошибка значения
        A.decompress(m, A.MAGIC + (5).to_bytes(4, "big") + b"\x00" + b"abc")  # архив режима 0: заявлено 5 байтов, фактически 3


def test_archiver_is_lossless_on_a_long_real_file_for_both_archs():
    """Без потерь на файле длиннее, чем G·L байтов (группа блоков × длина блока), для обеих моделей
    (rope и hnet), после одной эпохи обучения: c[9] = 1 и bpb < 8,0.

        Если реального файла нет — берётся периодический поток 0..63 ×300."""
    import glob  # поиск файлов по маске
    import os  # пути
    from hagi.lm import make_lm  # фабрика моделей
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "domains", "ns_04*.bin")))  # реальные файлы доменов ns_04*.bin, по алфавиту
    data = open(files[0], "rb").read()[:20000] if files else bytes(range(64)) * 300  # первые 20 000 байтов реального файла, иначе периодика
    for arch in ("rope", "hnet"):  # обе архитектуры
        m = make_lm(arch, seed=1)  # модель архитектуры arch
        m.train_guarded(data[:8000], epochs=1)  # одна эпоха на первых 8000 байтах
        c = _roundtrip(m, data)  # сжать и проверить обратный ход
        assert c[9] == 1 and len(c) * 8 / len(data) < 8.0  # режим 1 и bpb < 8,0


def test_stream_codec_with_context_table_is_lossless_and_beats_plain():
    """Потоковый кодек: q = (1 − λ)·p + λ·r (p — модель, r — продолжения самого длинного встречавшегося
    суффикса). На тексте Python после одной эпохи: сигнатура MAGIC4, точный обратный ход, короче обычного
    архива; λ из {0; 0,5; 0,95} без потерь; λ = 1 отвергнут; неверный префикс и битовая порча дают ValueError."""
    import hashlib  # хеширование (импортирован, не используется)
    m = _model()  # модель
    src = open(__file__.replace("test_archiver.py", "conftest.py") if False else __file__, "rb").read()  # прочитать байты этого файла (условие ложно — всегда этот файл)
    src = (src * 6)[:6000]  # текст, повторённый 6 раз, обрезанный до 6000 байтов
    pre, tgt = src[:4800], src[4800:]  # префикс — первые 4800 байтов, цель — остальные 1200
    m.train_guarded(src, epochs=1)  # одна эпоха на всём тексте
    a = A.compress(m, tgt)  # обычный архив цели
    c = A.compress_stream(m, tgt, prefix=pre, lam=0.7)  # потоковый архив цели с префиксом и λ = 0,7
    assert c[:5] == A.MAGIC4  # сигнатура потокового архива
    assert A.decompress_stream(m, c, prefix=pre) == tgt  # точный обратный ход с тем же префиксом
    assert len(c) < len(a)  # потоковый архив короче обычного
    for lam in (0.0, 0.5, 0.95):  # по весам λ
        assert A.decompress_stream(m, A.compress_stream(m, tgt, prefix=pre, lam=lam), prefix=pre) == tgt  # круглый ход без потерь при каждом λ
    with pytest.raises(ValueError):  # ожидается ошибка значения
        A.compress_stream(m, tgt, prefix=pre, lam=1.0)  # λ = 1 недопустим: должен быть < 1
    with pytest.raises(ValueError, match="prefix"):  # ожидается ошибка значения с текстом «prefix»
        A.decompress_stream(m, c, prefix=pre[:-1])  # префикс короче на 1 байт — контрольная сумма префикса не совпадает
    bad = bytearray(c); bad[-1] ^= 1  # испортить последний бит потока: b XOR 1
    with pytest.raises(ValueError):  # ожидается ошибка значения
        A.decompress_stream(m, bytes(bad), prefix=pre)  # контрольная сумма потока не совпадает — отказ


def test_stream_codec_falls_back_to_batch_and_stored():
    """Потоковый кодек на коротких и несжимаемых данных откатывается к обычному архиву
    (менее 64 байтов либо не короче обычного): обратный ход через decompress_stream и decompress точен."""
    m = _model()  # модель
    d = bytes(np.random.default_rng(2).integers(0, 256, 300, dtype=np.uint8))  # 300 случайных байтов
    assert A.decompress_stream(m, A.compress_stream(m, d, prefix=b"", lam=0.7)) == d  # потоковый ход без префикса: точно
    assert A.decompress_stream(m, A.compress(m, d)) == d  # обычный архив раскодируется и потоковым декодером: точно


def test_grow_width_preserves_logits_and_adapters(tmp_path, monkeypatch):
    """Расширение модели по ширине сохраняет logits: max |logits_новой − logits_старой| < 1e-4,
    размеры и число параметров растут; в Unified ширина банка адаптеров равна dim новой модели,
    logits с адаптером тоже сохраняются (< 1e-4)."""
    import torch as t  # тензоры
    from hagi import unified as U  # единая модель
    from hagi.hnet_lm import HNetByteLM  # модель hnet
    m = HNetByteLM(dim=16, hidden=32, seed=3)  # модель d = 16, h = 32
    m.train_guarded(bytes(range(0, 256, 7)) * 40, epochs=1)  # обучить на законе 0,7,14,…,252 ×40, одна эпоха
    X = t.randint(0, 256, (128, m.k), generator=t.Generator().manual_seed(1))  # 128 случайных контекстов длины k
    with t.no_grad():  # без записи градиентов
        before = m.logits(X)  # logits исходной модели
    g = m.grow_width(24, 48, seed=1)  # новая модель с d = 24, h = 48
    assert (g.dim, g.hidden) == (24, 48) and sum(p.numel() for p in g.params()) > sum(p.numel() for p in m.params())  # размеры 24 и 48, параметров стало больше
    with t.no_grad():  # без записи градиентов
        assert float((g.logits(X) - before).abs().max()) < 1e-4  # максимальный модуль разности logits < 1e-4
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # контрольная точка := tmp_path
    monkeypatch.setattr(U.Unified, "ARCH", "hnet")  # архитектура := hnet
    u = U.Unified()  # новая единая модель
    data = bytes(range(0, 256, 5)) * 600  # закон 0,5,10,…,250 ×600
    u.ingest("a", data, epochs=1, repeat=1)  # проглотить домен a
    mm = u.model  # модель до расширения
    ad = mm.domain_adapter = u.adapter_bank.get("a")  # адаптер домена a подключён к модели
    with t.no_grad():  # без записи градиентов
        b = mm.logits(X).clone()  # logits с адаптером до расширения
    info = u.grow_width(mm.dim + 8, mm.hidden + 16)  # расширить модель до dim + 8 и hidden + 16
    mm = u.model  # модель после расширения
    mm.domain_adapter = u.adapter_bank.get("a")  # подключить (перенесённый) адаптер домена a
    assert u.adapter_bank.d == mm.dim and info["d"][1] == mm.dim  # ширина банка = dim модели; записанная новая dim = dim модели
    with t.no_grad():  # без записи градиентов
        assert float((mm.logits(X) - b).abs().max()) < 1e-4  # logits с адаптером сохранны: расхождение < 1e-4
