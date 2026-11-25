"""HAGI как архиватор без потерь: байтовая языковая модель задаёт распределение
вероятностей для арифметического кодера. Размер архива отличается от
−log2 P(данные) лишь на константу, поэтому коэффициент сжатия равен
кросс-энтропии модели в битах на байт — внешняя мера того, чему модель научилась.

Кодек: 32-битный диапазонный кодер с 16-битной функцией распределения
(частота каждого байта не меньше 1, так что ни один байт не получает нулевую
вероятность). Кодирование пакетное по столбцам: один проход модели на столбец
для целой группы блоков; декодер подаёт модели только уже раскодированные
байты, поэтому симметричен кодеру по построению.

Контейнер:
  MAGIC | u32 total_len | u8 mode
  mode 0 (stored): сырые байты — модель не выиграла у сырого вида
  mode 1: u8 L | группы: u16 nb | u32 bit_len | битовый поток ... | хвост

Архив раскодирует только ТА ЖЕ модель: содержимое привязано к весам,
k, dim, hidden и прочим параметрам модели и разбиению на блоки.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import struct  # упаковка и распаковка двоичных заголовков

import numpy as np  # численные массивы

MAGIC = b"HAGI3"  # сигнатура пакетного архива
ZERO = bytes([0])  # метка режима 0 (сырые данные)
TOP, HALF, Q1, Q3 = 0xFFFFFFFF, 0x80000000, 0x40000000, 0xC0000000  # константы кодера: 2^32−1, 2^31, 2^30, 3·2^30


def probs_to_cdf(P: np.ndarray) -> np.ndarray:
    """Строки вероятностей -> 16-битные функции распределения: cdf (N, 257) int32.

        P: (N, 256). cdf[n, 0] = 0, cdf[n, j] = Σ_{i<j} freq[i], cdf[n, 256] = 65536.
        freq[i] = max(1, round(65200·P[n, i])); остаток 65536 − Σ freq добавляется
        самому частому байту, чтобы сумма равнялась 65536."""
    N = len(P)  # N — число строк (контекстов)
    freq = np.maximum(1, np.round(P * 65200.0).astype(np.int32))  # частота freq_i = max(1, round(65200·P_i))
    freq[np.arange(N), np.argmax(freq, 1)] += 65536 - freq.sum(1)  # самому частому байту += 65536 − Σ_i freq_i
    cdf = np.zeros((N, 257), np.int32)  # нулевая матрица N×257 (cdf[0] = 0)
    cdf[:, 1:] = np.cumsum(freq, 1)  # накопленная сумма cdf[j] = Σ_{i<j} freq[i], j = 1..256
    return cdf  # функции распределения всех строк


class ArithEncoder:
    """Арифметический кодер: сжимает байты в битовый поток.

        Хранит текущий интервал [lo, hi] 32-битных целых. Байт s с функцией
        распределения cdf сужает интервал до долей cdf[s]/65536 и cdf[s+1]/65536;
        когда старший бит интервала определён, он выдаётся наружу."""
    def __init__(self):  # создать кодер
        self.lo, self.hi, self.uf, self.bits = 0, TOP, 0, []  # lo = 0, hi = 2^32−1, ожидающих битов 0, поток битов пуст

    def _put(self, b: int):
        """Выдать бит b, а за ним все ожидающие биты недолёта (каждый равен 1 − b):
            именно он достраивает любое число из [lo, hi] после отбрасывания бита."""
        self.bits.append(b)  # добавить бит b
        while self.uf:  # пока есть ожидающие биты недолёта
            self.bits.append(1 - b)  # выдать инвертированный бит 1 − b
            self.uf -= 1  # ожидающих: uf − 1

    def encode_step(self, syms: np.ndarray, P: np.ndarray):
        """Закодировать по одному байту на каждую строку P; строки идут в общий поток.

            syms: (N,) кодируемые байты, P: (N, 256) их вероятности. Для байта s:
            rl = hi − lo + 1; hi' = lo + floor(rl·cdf[s+1]/65536) − 1;
            lo' = lo + floor(rl·cdf[s]/65536)."""
        cdf = probs_to_cdf(P)  # таблица cdf: (N, 257)
        r = np.arange(len(syms))  # номера строк r = 0..N−1
        c_lo = cdf[r, syms].astype(np.int64).tolist()  # нижние границы cdf[n, s_n]
        c_hi = cdf[r, syms + 1].astype(np.int64).tolist()  # верхние границы cdf[n, s_n + 1]
        lo, hi = self.lo, self.hi  # текущий интервал [lo, hi]
        for lc, hc in zip(c_lo, c_hi):  # по парам (нижняя lc, верхняя hc) граница
            rl = hi - lo + 1  # длина интервала rl = hi − lo + 1
            hi, lo = lo + (rl * hc) // 65536 - 1, lo + (rl * lc) // 65536  # новые границы: hi = lo + ⌊rl·hc/65536⌋ − 1; lo = lo + ⌊rl·lc/65536⌋
            while True:  # нормализация: пока интервал в половине или средней четверти
                if hi < HALF:  # интервал в нижней половине [0, 2^31)
                    self._put(0); lo <<= 1; hi = (hi << 1) | 1  # выдать 0; lo = 2·lo; hi = 2·hi + 1
                elif lo >= HALF:  # интервал в верхней половине [2^31, 2^32)
                    self._put(1); lo = (lo - HALF) << 1; hi = ((hi - HALF) << 1) | 1  # выдать 1; lo = 2(lo − 2^31); hi = 2(hi − 2^31) + 1
                elif lo >= Q1 and hi < Q3:  # интервал в средней четверти [2^30, 3·2^30)
                    self.uf += 1; lo = (lo - Q1) << 1; hi = ((hi - Q1) << 1) | 1  # недолёт +1; lo = 2(lo − 2^30); hi = 2(hi − 2^30) + 1
                else:  # иначе: интервал перекрывает границу половины
                    break  # выход из цикла, когда интервал перекрывает границу
        self.lo, self.hi = lo, hi  # запомнить новый интервал

    def finish(self) -> bytes:
        """Закрыть поток и вернуть биты, упакованные в байты (старший бит вперёд).

            Добавляет uf + 1 завершающих бит: выбирается число Q1, если lo < Q1
            (тогда hi ≥ Q3 и Q1 лежит в [lo, hi]), иначе число HALF = 2^31. Хвост
            дополняется нулями до кратности 8; байт = Σ_{i=0..7} b_i·2^(7−i)."""
        self.uf += 1  # завершающий бит считается одним «недолётом»: uf + 1
        self._put(0 if self.lo < Q1 else 1)  # выдать 0 (число Q1), если lo < Q1, иначе 1 (число HALF)
        while len(self.bits) % 8:  # пока число битов не кратно 8
            self.bits.append(0)  # добавить нулевой бит
        b = np.asarray(self.bits, np.uint8).reshape(-1, 8)  # биты в матрицу (число байтов, 8)
        return np.sum(b << np.arange(7, -1, -1, dtype=np.uint8), 1).astype(np.uint8).tobytes()  # байт_j = Σ_{i=0..7} b_{j,i}·2^(7−i)


class ArithDecoder:
    """Арифметический декодер, обратный ArithEncoder.

        Читает биты потока по одному (за концом потока — нули), хранит 32-битное
        значение val и интервал [lo, hi]; повторяет шаги кодера, выбирая байт, чей
        диапазон cdf содержит val."""
    def __init__(self, payload: bytes):  # создать декодер по байтам потока
        self.bs = ((np.frombuffer(payload, np.uint8)[:, None]  # байты потока в столбец
                    >> np.arange(7, -1, -1, dtype=np.uint8)) & 1).ravel().tolist() if payload else []  # бит = (байт >> (7 − i)) & 1, старший первым; пустой поток — []
        self.bp = 0  # позиция следующего бита
        self.val = 0  # текущее 32-битное значение val
        for _ in range(32):  # прочитать первые 32 бита
            self.val = (self.val << 1) | self._nb()  # val = 2·val + следующий бит
        self.lo, self.hi = 0, TOP  # начальный интервал [0, 2^32 − 1]

    def _nb(self) -> int:
        """Следующий бит потока; за концом потока — 0."""
        if self.bp < len(self.bs):  # биты ещё есть
            r = self.bs[self.bp]; self.bp += 1; return r  # вернуть бит bp и сдвинуть позицию на 1
        return 0  # поток кончился — бит 0

    def decode_step(self, P: np.ndarray) -> np.ndarray:
        """Раскодировать по байту на каждую строку P (N, 256); возвращает (N,) int64.

            Для строки t с функцией распределения ct и rl = hi − lo + 1:
            x = floor(((val − lo + 1)·65536 − 1)/rl);
            s — наибольшее, для которого ct[s] ≤ x < ct[s + 1]. Затем интервал сужается и
            нормализуется как в кодере; при каждом удвоении lo в val вдвигается новый
            бит: val = (2·val mod 2^32) | бит."""
        cdf = probs_to_cdf(P)  # таблица cdf: (N, 257)
        lo, hi, val = self.lo, self.hi, self.val  # текущие lo, hi, val
        out = np.empty(len(P), np.int64)  # результат длины N
        for t in range(len(P)):  # по строкам t = 0..N−1
            ct = cdf[t]; rl = hi - lo + 1  # строка cdf и длина интервала rl = hi − lo + 1
            sym = int(np.searchsorted(ct, ((val - lo + 1) * 65536 - 1) // rl, side="right")) - 1  # x = ⌊((val−lo+1)·65536 − 1)/rl⌋; s = (индекс вставки x в ct справа) − 1, т.е. ct[s] ≤ x < ct[s+1]
            out[t] = sym  # записать раскодированный байт
            hi = lo + (rl * int(ct[sym + 1])) // 65536 - 1  # новая верхняя граница hi = lo + ⌊rl·ct[s+1]/65536⌋ − 1
            lo = lo + (rl * int(ct[sym])) // 65536  # новая нижняя граница lo = lo + ⌊rl·ct[s]/65536⌋
            while True:  # нормализация — как в кодере
                if hi < HALF:  # интервал в нижней половине
                    lo <<= 1; hi = (hi << 1) | 1; val = ((val << 1) & TOP) | self._nb()  # lo = 2lo; hi = 2hi + 1; val = (2·val mod 2^32) | бит
                elif lo >= HALF:  # интервал в верхней половине
                    lo = (lo - HALF) << 1; hi = ((hi - HALF) << 1) | 1  # сдвиг: lo = 2(lo − 2^31); hi = 2(hi − 2^31) + 1
                    val = (((val - HALF) << 1) & TOP) | self._nb()  # val = (2(val − 2^31) mod 2^32) | бит
                elif lo >= Q1 and hi < Q3:  # интервал в средней четверти
                    lo = (lo - Q1) << 1; hi = ((hi - Q1) << 1) | 1  # сдвиг: lo = 2(lo − 2^30); hi = 2(hi − 2^30) + 1
                    val = (((val - Q1) << 1) & TOP) | self._nb()  # val = (2(val − 2^30) mod 2^32) | бит
                else:  # иначе: интервал перекрывает границу половины
                    break  # выход из цикла нормализации
        self.lo, self.hi, self.val = lo, hi, val  # запомнить lo, hi, val
        return out  # раскодированные байты всех строк


def _probs(model, ctx_np: np.ndarray) -> np.ndarray:
    """P(следующий байт | контекст), float32 (B, 256).

    ctx_np: (B, n) байты контекста от СТАРОГО к новому — тот же порядок,
    что строит сборка контекстов при обучении (последний столбец n−1 — самый
    новый байт, он же запрос; веса давности −λ·(L−1−j) на нём согласованы).
    Считается softmax: p_i = exp(l_i − max l) / Σ_j exp(l_j − max l)."""
    t = model.t  # тензорная библиотека модели t
    ctx = t.as_tensor(np.ascontiguousarray(ctx_np), dtype=t.long)  # контекст (B, n) в целый тензор
    with t.no_grad():  # без вычисления градиентов
        lg = model.logits(ctx)  # логиты lg (B, 256)
        lg = lg - lg.amax(dim=-1, keepdim=True)  # lg_i − max_j lg_j (устойчивость экспоненты)
        e = t.exp(lg)  # экспоненты e_i = exp(l_i − max l)
        return (e / e.sum(dim=-1, keepdim=True)).float().cpu().numpy()  # вероятности p_i = e_i / Σ_j e_j -> float32 numpy


def _groups(n_blk: int, group: int):
    """Не используется (старый вариант по числу блоков). Выдаёт пары
        (старт, размер) кусков не длиннее `group`: (0, g), (g, g), ..., последний короче."""
    s = 0  # старт первого куска s = 0
    while s < n_blk:  # пока старт меньше числа блоков
        yield s, min(group, n_blk - s)  # пара (старт s, размер min(group, n_blk − s))
        s += group  # следующий старт s + group


def _groups(blocks: np.ndarray, group: int):
    """Разбить массив блоков на группы по `group` штук: blocks[0:g], blocks[g:2g], ..."""
    for s in range(0, len(blocks), group):  # старты s = 0, group, 2·group, ...
        yield blocks[s:s + group]  # срез blocks[s : s+group−1]


def compress(model, data: bytes, L: int | None = None, group: int = 256) -> bytes:
    """Архив без потерь для `data` под `model`; возвращает байты архива.

    Блоки по L байтов НЕЗАВИСИМЫ (контекст каждого начинается с нулевого
    дополнения на старте блока) — именно это делает пакетный по столбцам
    декодер причинным: столбец col каждого блока группы кодируется и
    декодируется одним проходом модели по одним и тем же уже известным байтам.
    Измерленная цена дополнения на старте блока против сплошного контекста
    (2 эпохи на реальном файле): +0.04 nat (Rope), +0.13 nat (HNet). Сплошной
    контекст длины k = L нельзя батчить по столбцам: блоку j нужен весь блок
    j−1, то есть блоки образуют цепочку; три опробованных фазовых расписания
    нарушили беспотерность на файлах длиннее G·L байтов.

    L по умолчанию = model.k; group — число блоков в группе. Если архив не короче
    сырого вида, возвращается вид mode 0."""
    head = MAGIC + struct.pack(">I", len(data))  # заголовок: сигнатура + длина данных (u32, big-endian)
    stored = head + ZERO + data  # режим 0: заголовок + метка 0 + сырые данные
    L = L or model.k  # L = L, если задано, иначе k модели
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)  # данные как массив целых байтов
    n_blk = len(arr) // L  # число полных блоков n_blk = ⌊n / L⌋
    if len(data) < 64 or n_blk == 0:  # данных меньше 64 байтов или нет ни одного блока
        return stored  # вернуть сырой вид
    k = model.k  # k — длина контекста модели
    blocks = arr[:n_blk * L].reshape(n_blk, L)  # первые n_blk·L байтов в матрицу (n_blk, L)
    out = bytearray(head) + struct.pack(">BB", 1, L)  # архив: заголовок + режим 1 (u8) + L (u8)
    for grp in _groups(blocks, group):  # по группам блоков
        nb = len(grp)  # nb — число блоков в группе
        pad = np.zeros((nb, k), dtype=np.int64)  # нулевое дополнение k байтов на каждый блок
        buf = np.concatenate([pad, grp], axis=1)  # буфер: дополнение + блоки, форма (nb, k+L)
        enc = ArithEncoder()  # свой кодер на группу
        for col in range(L):  # по столбцам col = 0..L−1
            enc.encode_step(grp[:, col], _probs(model, buf[:, col:col + k]))  # закодировать байты столбца col по контексту buf[:, col : col+k−1]
        bs = enc.finish()  # битовый поток группы в байтах
        out += struct.pack(">HI", nb, len(bs)) + bs  # записать nb (u16), длину потока bl (u32), поток
    out += arr[n_blk * L:].astype(np.uint8).tobytes()  # хвост: байты с n_blk·L до конца, некодированные
    return bytes(out) if len(out) < len(stored) else stored  # архив, если он короче сырого вида, иначе сырой вид


def decompress(model, payload: bytes) -> bytes:
    """Точный обратный к compress под ТОЙ ЖЕ моделью; возвращает исходные байты."""
    if payload[:len(MAGIC)] != MAGIC:  # сигнатура не совпадает
        raise ValueError("not a HAGI archive")  # ошибка: это не архив HAGI
    total = struct.unpack(">I", payload[5:9])[0]  # длина исходных данных total из байтов 5..8
    mode = payload[9]  # режим архива — байт 9
    if mode == 0:  # режим 0 — сырой вид
        raw = payload[10:]  # данные с байта 10 до конца
        if len(raw) != total:  # длина не совпадает с заявленной
            raise ValueError("corrupt archive")  # ошибка: архив повреждён
        return bytes(raw)  # вернуть сырые данные
    if mode != 1:  # режим не 0 и не 1
        raise ValueError("unsupported archive mode")  # ошибка: режим архива не поддерживается
    L = payload[10]  # L — длина блока, байт 10
    k = model.k  # k — длина контекста модели
    n_blk = total // L  # число блоков n_blk = ⌊total / L⌋
    tail_len = total - n_blk * L  # длина хвоста tail_len = total − n_blk·L
    pos, done = 11, 0  # позиция чтения 11 (после заголовка), раскодировано блоков 0
    parts = []  # раскодированные части
    while done < n_blk:  # пока раскодировано меньше n_blk блоков
        nb, bl = struct.unpack(">HI", payload[pos:pos + 6]); pos += 6  # nb (u16), длина потока bl (u32); pos += 6
        dec = ArithDecoder(payload[pos:pos + bl]); pos += bl  # декодер на потоке payload[pos : pos+bl−1]; pos += bl
        buf = np.zeros((nb, k + L), dtype=np.int64)  # буфер группы: k нулей дополнения, затем блоки (nb, k+L)
        blk = buf[:, k:]  # ВИД (не копия): запись в blk видна в контексте buf
        for col in range(L):  # по столбцам col = 0..L−1
            blk[:, col] = dec.decode_step(_probs(model, buf[:, col:col + k]))  # столбец col = раскодированные байты по контексту buf[:, col : col+k−1]
        parts.append(blk.astype(np.uint8).tobytes())  # добавить байты раскодированных блоков группы
        done += nb  # раскодировано блоков done + nb
    tail = payload[pos:pos + tail_len]  # хвост: payload[pos : pos+tail_len−1]
    if len(tail) != tail_len:  # хвост короче заявленного
        raise ValueError("corrupt archive")  # ошибка: архив повреждён
    return b"".join(parts) + tail  # все блоки + хвост


# ---- Единый фасад -------------------------------------------------------
def compress_bytes(u, data: bytes, domain: str | None = None, **kw) -> bytes:
    """Архив `data` под загруженной объединённой моделью u; возвращает байты архива.

    domain: необязательный адаптер домена, включается на время вызова так же,
    как при обычном выводе модели."""
    ad = u._live_adapter(domain) if domain else None  # адаптер домена, если domain задан, иначе None
    u.model.domain_adapter = ad  # подключить адаптер к модели
    try:  # кодирование — адаптер снимается в любом случае
        return compress(u.model, data, **kw)  # архив под моделью u.model с параметрами kw
    finally:  # всегда после кодирования
        u.model.domain_adapter = None  # отключить адаптер домена


def decompress_bytes(u, payload: bytes, domain: str | None = None, **kw) -> bytes:
    """Раскодировать `payload` под загруженной моделью u; возвращает исходные байты.

        domain: необязательный адаптер домена — тот же, что при сжатии."""
    ad = u._live_adapter(domain) if domain else None  # адаптер домена, если domain задан, иначе None
    u.model.domain_adapter = ad  # подключить адаптер к модели
    try:  # декодирование — адаптер снимается в любом случае
        return decompress(u.model, payload, **kw)  # раскодировать под моделью u.model с параметрами kw
    finally:  # всегда после декодирования
        u.model.domain_adapter = None  # отключить адаптер домена


def compress_file(u, path, out_path, domain: str | None = None) -> tuple[int, int]:
    """Без потерь: файл -> архив. Возвращает (размер исходника, размер архива), байты."""
    import pathlib  # работа с путями
    data = pathlib.Path(path).read_bytes()  # прочитать исходный файл целиком
    payload = compress_bytes(u, data, domain)  # архив байтов файла
    pathlib.Path(out_path).write_bytes(payload)  # записать архив на диск
    return len(data), len(payload)  # пара (число исходных байтов, число байтов архива)


def decompress_file(u, path, out_path, domain: str | None = None) -> bytes:
    """Без потерь: архив -> файл. Возвращает исходные байты (они же записаны в out_path)."""
    import pathlib  # работа с путями
    data = decompress_bytes(u, pathlib.Path(path).read_bytes(), domain)  # прочитать архив и раскодировать его
    pathlib.Path(out_path).write_bytes(data)  # записать раскодированные байты
    return data  # вернуть исходные байты


# ---- Нейросеть + таблица контекстов нескольких порядков (потоковый кодек) ----
# Для каждого байта полное распределение нейросети p смешивается с эмпирическим
# распределением продолжений r САМОГО ДЛИННОГО уже встречавшегося
# контекста-суффикса (16/12/8/4/2 байта):
#     q = (1 − λ)·p + λ·r,   λ < 1
# так что каждый байт сохраняет положительную вероятность, а неожиданности
# уходят в остаток. Таблица пополняется только байтами, которые декодер уже
# держит, — для этого нужна ПОСЛЕДОВАТЕЛЬНАЯ декодировка (один проход модели
# на байт), отсюда отдельный кодек, а не пакетный кодек выше.
CONTEXT_LENGTHS = (16, 12, 8, 4, 2)  # длины контекстов от длинного к короткому
MAGIC4 = b"HAGI4"  # сигнатура потокового архива


class ContextTable:
    """Причинная таблица суффиксных контекстов: порядки от длинного к короткому,
    контекст учитывается, только если в нём ≥ min_count продолжений."""

    def __init__(self, prefix: bytes = b"", lengths=CONTEXT_LENGTHS, min_count: int = 1):  # создать таблицу по префиксу
        self.lengths, self.min_count = tuple(lengths), int(min_count)  # сохранить длины порядков и минимальный счётчик
        self.t = {L: {} for L in self.lengths}  # для каждого порядка L пустой словарь: контекст -> {байт: счётчик}
        for L in self.lengths:  # по порядкам L
            for i in range(L - 1, len(prefix) - 1):  # по позициям i от L−1 до len(prefix)−2 (есть L-байтов контекст и следующий байт)
                d = self.t[L].setdefault(bytes(prefix[i - L + 1:i + 1]), {})  # словарь продолжений контекста prefix[i−L+1 : i]
                b = prefix[i + 1]  # следующий байт b = prefix[i + 1]
                d[b] = d.get(b, 0) + 1  # счётчик продолжения b +1

    def choose(self, hist):
        """Самый длинный порядок, чей контекст (последние L байтов hist) есть в таблице
            и имеет Σ счётчиков ≥ min_count; возвращает {байт: счётчик} или None."""
        for L in self.lengths:  # от длинного порядка к короткому
            if len(hist) < L:  # истории меньше L байтов
                continue  # перейти к более короткому порядку
            d = self.t[L].get(bytes(hist[-L:]))  # продолжения контекста hist[−L : −1] или None
            if d is not None and sum(d.values()) >= self.min_count:  # контекст найден и сумма счётчиков ≥ min_count
                return d  # распределение продолжений этого контекста
        return None  # подходящего контекста нет

    def update(self, hist, sym):
        """Записать в таблицу: после истории hist идёт байт sym (для всех порядков)."""
        for L in self.lengths:  # по порядкам L
            if len(hist) >= L:  # истории не меньше L байтов
                d = self.t[L].setdefault(bytes(hist[-L:]), {})  # словарь продолжений последних L байтов истории
                d[sym] = d.get(sym, 0) + 1  # счётчик sym +1


def mix_probs(p: np.ndarray, counts, lam: float) -> np.ndarray:
    """Смешать распределение модели p с эмпирическим counts при весе lam.

        counts: {байт: счётчик} или None. Если counts нет или lam ≤ 0 — p без изменений.
        Иначе q_s = (1 − λ)·p_s + λ·counts[s]/Σ_c counts[c] (по s из counts),
        затем q нормируется: q_i = q_i / Σ_j q_j."""
    if counts is None or lam <= 0.0:  # таблицы нет или вес нулевой
        return p  # вернуть p без изменений
    q = (1.0 - lam) * p  # основная часть q = (1 − λ)·p
    tot = sum(counts.values())  # сумма счётчиков tot = Σ_c counts[c]
    for s, c in counts.items():  # по парам (байт s, счётчик c)
        q[int(s)] += lam * c / tot  # добавка q_s += λ·c / tot
    return q / q.sum()  # нормировать: q_i = q_i / Σ_j q_j


def model_fingerprint(model) -> bytes:
    """Отпечаток модели: первые 16 байт sha256 от строки «ARCH|k|dim|hidden»
        и необработанных байтов всех параметров модели."""
    import hashlib  # хеширование
    h = hashlib.sha256()  # хеш sha256
    h.update(f"{getattr(model, 'ARCH', '?')}|{model.k}|{model.dim}|{model.hidden}".encode())  # добавить байты строки 'ARCH|k|dim|hidden'
    for p in model.params():  # по всем параметрам модели
        h.update(p.detach().cpu().numpy().tobytes())  # добавить необработанные байты параметра
    return h.digest()[:16]  # первые 16 байт хеша


def _pad_ctx(hist: bytes, k: int) -> np.ndarray:
    """Контекст длины k: последние k байтов hist, слева дополненные нулями; (1, k)."""
    tail = hist[-k:]  # последние k байтов истории (или все, если их меньше k)
    x = np.zeros(k, dtype=np.int64)  # массив из k нулей
    x[k - len(tail):] = np.frombuffer(tail, dtype=np.uint8).astype(np.int64)  # хвост в конец: x[k − len(tail) : k−1] = tail
    return x[None, :]  # добавить ось размера 1: форма (1, k)


def _pack_lam(lam: float) -> int:
    """Квантовать вес λ в целое: min(65534, round(65535·λ)); требует 0 ≤ λ < 1."""
    if not 0.0 <= lam < 1.0:  # λ вне отрезка [0, 1)
        raise ValueError("lambda must be in [0, 1): the neural prior keeps full support")  # ошибка: λ должен лежать в [0, 1) — нейросеть сохраняет полный носитель
    return min(65534, int(round(lam * 65535)))  # квантование: min(65534, round(65535·λ))


def compress_stream(model, data: bytes, prefix: bytes = b"", lam: float = 0.70,
                    min_count: int = 1) -> bytes:  # min_count — минимальное число продолжений, чтобы контекст учитывался
    """Последовательный кодек с таблицей контекстов (один проход модели на байт).

    `prefix` должен быть известен декодеру (например, начало того же файла).
    λ квантуется: λ_q = min(65534, round(65535·λ)), в смешивании используется
    λ_q/65535. Если результат не короче обычного архива — возвращается обычный."""
    import hashlib  # хеширование
    import zlib  # контрольная сумма crc32
    if len(data) < 64:  # данных меньше 64 байтов
        return compress(model, data)  # обычный пакетный архив
    lam_q = _pack_lam(lam)  # квантованный вес λ_q
    tab = ContextTable(prefix, min_count=min_count)  # таблица, наполненная префиксом
    hist = bytearray(prefix)  # история = префикс
    enc = ArithEncoder()  # арифметический кодер
    for s in data:  # по байтам s данных
        q = mix_probs(_probs(model, _pad_ctx(bytes(hist), model.k))[0].astype(np.float64), tab.choose(hist), lam_q / 65535.0)  # q = (1−λ)·p + λ·r: p от модели на контексте hist, r от таблицы, λ = λ_q/65535
        enc.encode_step(np.asarray([s], dtype=np.int64), q[None, :].astype(np.float32))  # закодировать байт s с распределением q (1, 256)
        tab.update(hist, s)  # внести в таблицу пару (история, байт s)
        hist.append(s)  # история += s
    payload = enc.finish()  # битовый поток в байтах
    head = (MAGIC4 + struct.pack(">IBBHI", len(data), model.k, min_count, lam_q, len(payload))  # n (u32), k (u8), min_count (u8), λ_q (u16), длина потока plen (u32)
            + model_fingerprint(model) + hashlib.sha256(prefix).digest()  # отпечаток модели (16 байт) + sha256 префикса (32 байта)
            + hashlib.sha256(data).digest()[:16] + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF))  # первые 16 байт sha256 данных + crc32 потока (u32)
    arc = head + payload  # архив целиком
    return arc if len(arc) < len(compress(model, data)) else compress(model, data)  # потоковый архив, если он короче обычного, иначе обычный


def decompress_stream(model, payload: bytes, prefix: bytes = b"") -> bytes:
    """Раскодировать потоковый архив (обратно к compress_stream); возвращает байты.

        Если сигнатура не HAGI4 — архив обрабатывается как обычный (decompress)."""
    import hashlib  # хеширование
    import zlib  # контрольная сумма crc32
    if payload[:len(MAGIC4)] != MAGIC4:  # это не потоковый архив
        return decompress(model, payload)  # раскодировать как обычный архив
    n, k, min_count, lam_q, plen = struct.unpack(">IBBHI", payload[5:17])  # n, k, min_count, λ_q, plen из байтов 5..16
    off = 17  # смещение после заголовка off = 17
    fp, ph, th = payload[off:off + 16], payload[off + 16:off + 48], payload[off + 48:off + 64]; off += 64  # отпечаток модели (16 б), sha256 префикса ph (32 б), sha256 данных th (16 б)
    (crc,) = struct.unpack(">I", payload[off:off + 4]); off += 4  # контрольная сумма потока из 4 байт; off += 4
    body = payload[off:]  # тело — весь оставшийся поток
    if k != model.k:  # k не совпадает с контекстом модели
        raise ValueError("model context length mismatch")  # ошибка: длина контекста модели не совпадает
    if fp != model_fingerprint(model):  # отпечаток не совпадает
        raise ValueError("model fingerprint mismatch")  # ошибка: отпечаток модели не совпадает
    if ph != hashlib.sha256(prefix).digest():  # sha256 префикса не совпадает
        raise ValueError("prefix checksum mismatch")  # ошибка: контрольная сумма префикса не совпадает
    if len(body) != plen or (zlib.crc32(body) & 0xFFFFFFFF) != crc:  # длина тела ≠ plen или crc32 не совпадает
        raise ValueError("payload corrupt")  # ошибка: поток повреждён
    tab = ContextTable(prefix, min_count=min_count)  # таблица, наполненная префиксом
    hist = bytearray(prefix)  # история = префикс
    dec = ArithDecoder(body)  # декодер на теле архива
    out = bytearray()  # выходные байты
    lam = lam_q / 65535.0  # восстановленный вес λ = λ_q / 65535
    for _ in range(n):  # n раз — по байтам результата
        q = mix_probs(_probs(model, _pad_ctx(bytes(hist), model.k))[0].astype(np.float64), tab.choose(hist), lam)  # q = (1−λ)·p + λ·r — так же, как при сжатии
        s = int(dec.decode_step(q[None, :].astype(np.float32))[0])  # s — раскодированный байт по распределению q
        out.append(s)  # добавить байт s
        tab.update(hist, s)  # внести в таблицу пару (история, s)
        hist.append(s)  # история += s
    if hashlib.sha256(bytes(out)).digest()[:16] != th:  # первые 16 байт sha256 результата ≠ th
        raise ValueError("decoded bytes do not match the archive checksum")  # ошибка: раскодированные байты не совпадают с контрольной суммой
    return bytes(out)  # вернуть исходные байты


# ---- Архиватор v2: потоковый предиктор (модель + PPM 0..4, Mirror-Hedge) + арифметический кодек ----
MAGIC6 = b"HAGI6"  # сигнатура потокового архива с хеджем и PPM 0..4


def compress_stream_v2(model, data: bytes, eta: float = 0.25) -> bytes:
    """Сжать data побайтно предиктором MirrorHedgeStreamPredictor (модель на окне + онлайн PPM 0..4).
        Без потерь: обновления предиктора зависят только от уже закодированных байтов."""
    import hashlib  # хеширование
    import struct  # упаковка заголовка
    import zlib  # контрольная сумма потока
    from hagi.models.optimal_lm import MirrorHedgeStreamPredictor  # предиктор
    eta_q = min(65534, int(round(eta * 65535)))  # η квантованный — кодер и декодер используют одно и то же значение
    pr = MirrorHedgeStreamPredictor(model, eta=eta_q / 65535.0)  # предиктор с квантованным η
    enc = ArithEncoder()  # арифметический кодер
    for s in data:  # по байтам
        q = pr.predict_probs().astype(np.float32)  # распределение следующего байта
        q = np.maximum(q, 1e-9); q /= q.sum()  # полный носитель
        enc.encode_step(np.asarray([s], dtype=np.int64), q[None, :])  # закодировать
        pr.update(s)  # обновить предиктор фактическим байтом
    payload = enc.finish()  # битовый поток
    head = (MAGIC6 + struct.pack(">IBH", len(data), model.k, eta_q)  # число байтов, контекст, шаг хеджа
            + model_fingerprint(model) + hashlib.sha256(data).digest()[:16]  # отпечаток модели, sha256 данных (16 байт)
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF))  # контрольная сумма потока
    return head + payload  # архив


def decompress_stream_v2(model, payload: bytes) -> bytes:
    """Раскодировать архив HAGI6 (обратно к compress_stream_v2); проверяет crc и sha256."""
    import hashlib  # хеширование
    import struct  # распаковка заголовка
    import zlib  # контрольная сумма потока
    from hagi.models.optimal_lm import MirrorHedgeStreamPredictor  # предиктор
    if payload[: len(MAGIC6)] != MAGIC6:  # не тот формат
        raise ValueError("not a HAGI6 archive")  # ошибка
    n, k, eta_q = struct.unpack(">IBH", payload[5:12])  # число байтов, контекст, шаг хеджа
    off = 12; fp, th = payload[off: off + 16], payload[off + 16: off + 32]; off += 32  # отпечаток и sha256
    (crc,) = struct.unpack(">I", payload[off: off + 4]); off += 4  # контрольная сумма потока
    body = payload[off:]  # тело
    if k != model.k:  # контекст другой
        raise ValueError("model context length mismatch")  # ошибка
    if fp != model_fingerprint(model):  # модель другая
        raise ValueError("model fingerprint mismatch")  # ошибка
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc:  # поток повреждён
        raise ValueError("payload corrupt")  # ошибка
    pr = MirrorHedgeStreamPredictor(model, eta=eta_q / 65535.0)  # предиктор с тем же η
    dec = ArithDecoder(body)  # декодер
    out = bytearray()  # результат
    for _ in range(n):  # n байтов
        q = pr.predict_probs().astype(np.float32)  # распределение
        q = np.maximum(q, 1e-9); q /= q.sum()  # полный носитель
        s = int(dec.decode_step(q[None, :])[0])  # раскодировать байт
        out.append(s); pr.update(s)  # добавить и обновить (та же история, что у кодера)
    if hashlib.sha256(bytes(out)).digest()[:16] != th:  # контрольная сумма не сошлась
        raise ValueError("decoded bytes do not match the archive checksum")  # ошибка
    return bytes(out)  # исходные байты
