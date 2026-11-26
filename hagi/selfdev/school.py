"""Школа: книга уроков, учёт вклада фаз учебного плана и интерфейс «ума».

Три части:
- Книга уроков — накопительный JSON-словарь «урок -> текст»; corpus склеивает все
  тексты в единый обучающий поток байтов.
- Учёт фаз: каждая фаза учебного плана даёт вклад delta/d_cert; сумма вкладов не
  зависит от порядка фаз и служит сертификатом учебного плана.
- Ум: гибридный интерфейс вопроса и ответа — сначала точная программа (если запрос
  подходит под маршрут), иначе вероятностная догадка байтовой языковой модели.
Экспортирует load_book, corpus, PhaseRecord, curriculum_total, certificate,
ask (ответ на вопрос), retrain (переобучение модели на собственных артефактах) и main."""
from __future__ import annotations  # отложенное вычисление аннотаций типов


# ============================================================
# КНИГА УРОКОВ (накопительная, полный повтор)
# ============================================================

import json  # чтение книги уроков
from pathlib import Path  # пути к файлам проекта
from hagi.base.util import CKPT_DIR as _CKPT  # каталог контрольных точек
BOOK = _CKPT / 'nadya_lessons.json'  # путь к книге уроков в каталоге контрольных точек
def load_book() -> dict[str, str]:
    """Прочитать книгу уроков: словарь «имя урока -> текст»; если файла нет — пустой словарь."""
    if BOOK.exists():  # файл книги существует
        return json.loads(BOOK.read_text(encoding='utf-8'))  # разбор JSON-текста файла в словарь
    return {}  # книги ещё нет — пустой словарь
def corpus() -> bytes:
    """Собрать обучающий поток: тексты всех уроков через пробел, в байтах UTF-8."""
    return ' '.join(load_book().values()).encode('utf-8')  # ' '.join(значения книги) -> байты UTF-8
SCHOOL_CKPT = _CKPT / 'nadya_school.pt'  # путь к контрольной точке «школы»

# ============================================================
# УЧЁТ ВКЛАДА ФАЗ УЧЕБНОГО ПЛАНА
# ============================================================

from dataclasses import dataclass, asdict  # класс данных и преобразование в словарь
from typing import Any  # тип для аннотаций
PHASES = ('GROW', 'DERIVE', 'OPTIMIZE', 'GATE')  # четыре фазы цикла: рост, вывод, оптимизация, гейт
@dataclass  # класс данных: методы сравнения и инициализации генерируются автоматически
class PhaseRecord:
    """Запись об одной фазе учебного плана: прирост и сертифицированная доля."""

    phase: str  # имя фазы
    delta: float  # прирост за фазу
    d_cert: float  # сертифицированный шаг
    steps: int = 1  # число шагов фазы (по умолчанию 1)

    @property  # доступ как к атрибуту
    def contribution(self) -> float:  # вклад фазы
        """Вклад фазы в сертификат: delta/d_cert, если d_cert > 0, иначе 0.0."""
        return self.delta / self.d_cert if self.d_cert > 0 else 0.0  # delta/d_cert при d_cert > 0, иначе 0.0
def curriculum_total(records: list[PhaseRecord]) -> float:
    """Сертификат, не зависящий от порядка: сумма вкладов фаз с 6 знаками.

    total = Σ_i delta_i/d_cert_i (для d_cert_i > 0); сумма коммутативна по построению."""
    return round(sum((r.contribution for r in records)), 6)  # Σ contribution по записям, округление до 6 знаков
def certificate(state: dict[str, Any]) -> dict[str, Any]:
    """Добавить или обновить сертификат учебного плана в словаре состояния state.

    Записи берутся из state['curriculum']; в state['curriculum_certificate']
    кладутся фазы (в виде словарей), сумма вкладов и признаки независимости от порядка."""
    recs = [PhaseRecord(**r) for r in state.get('curriculum', [])]  # записи фаз из state['curriculum'] (пустой список, если ключа нет)
    state['curriculum_certificate'] = {'phases': [asdict(r) for r in recs], 'total': curriculum_total(recs), 'order_free': True, 'theorem': 'curriculum_phase_accounting (CurriculumCert.lean)'}  # сертификат: фазы как словари, сумма вкладов, порядок не важен, имя утверждения
    return state  # вернуть тот же словарь state

# ============================================================
# УМ: гибридный интерфейс «точное / символьное»
# ============================================================

import argparse  # разбор аргументов командной строки
import re  # маршруты запросов по регулярным выражениям
from typing import Callable  # типы для аннотаций
from hagi.data.io import execute, execute_text  # вызов написанной моделью программы (числовой и текстовый)
from hagi.models import lm  # байтовая языковая модель
# Фактический состав модулей после слияний (io поглотил execute и ingest; generated_programs лежит в scripts/, если есть)
SELF_SOURCES = ['lm.py', 'io.py', 'loop.py', 'rewrite.py']  # собственные файлы, чей текст идёт в обучение модели
_ROUTES: dict[str, tuple[str, Callable[..., tuple]]] = {'^(?:чему равен )?gcd (\\d+) (\\d+)$': ('GCD', lambda m: (int(m[1]), int(m[2]))), '^(?:чему равен )?нок (\\d+) (\\d+)$': ('LCM', lambda m: (int(m[1]), int(m[2]))), '^abs (-?\\d+)$': ('ABS', lambda m: (int(m[1]),)), '^square (-?\\d+)$': ('SQUARE', lambda m: (int(m[1]),)), '^cube (-?\\d+)$': ('CUBE', lambda m: (int(m[1]),)), '^mul (-?\\d+) (-?\\d+)$': ('MUL', lambda m: (int(m[1]), int(m[2]))), '^add (-?\\d+) (-?\\d+)$': ('ADD', lambda m: (int(m[1]), int(m[2]))), '^xor (.+) (-?\\d+)$': ('LANG8_XOR_CIPHER', lambda m: (m[1], int(m[2]))), '^caesar (.+)$': ('LANG8_CAESAR1', lambda m: (m[1],))}  # маршруты: шаблон запроса -> (имя программы, разбор групп совпадения в аргументы)
def _exact_answer(query: str) -> tuple[str, str] | None:
    """Попытаться ответить точно: найти первый подходящий маршрут и выполнить его программу.

    Запрос приводится к нижнему регистру без крайних пробелов. Для совпавшего шаблона
    аргументы разбираются лямбдой маршрута, имя программы заменяется устойчивым хешем
    (из rewrite._fn_hash). Текстовые программы (шифры) выполняются через execute_text
    с ключом или без, остальные — через execute. Возвращает (имя программы, текст ответа)
    либо None, если ни один маршрут не подошёл."""
    q = query.strip().lower()  # нормализованный запрос: без крайних пробелов, в нижнем регистре
    from hagi.selfdev.rewrite import _fn_hash  # устойчивое хеш-имя программы (локальный импорт)
    for pat, (prog, parse) in _ROUTES.items():  # по маршрутам: шаблон, имя программы, разбор аргументов
        m = re.match(pat, q)  # совпадение запроса с шаблоном (с начала строки) или None
        if not m:  # шаблон не подошёл
            continue  # следующий маршрут
        args = parse(m)  # аргументы программы из групп совпадения
        prog_h = _fn_hash(prog)  # хеш-имя программы prog_h = _fn_hash(prog)
        if prog in ('LANG8_XOR_CIPHER', 'LANG8_CAESAR1', 'LANG_ATBASH', 'LANG8_SHIFT'):  # текстовая программа
            text, key = args if len(args) == 2 else (args[0], None)  # текст и ключ: два аргумента, если они есть, иначе текст и ключ None
            out = execute_text(prog_h, text, key=key) if key is not None else execute_text(prog_h, text)  # выполнить текстовую программу с ключом или без него
        else:  # числовая программа
            out = execute(prog_h, *args)  # выполнить программу с аргументами, результат — целое
        return (prog, str(out))  # вернуть (имя программы, текст ответа)
    return None  # точного маршрута нет
def ask(query: str) -> str:
    """Ответить на вопрос: точно (если есть маршрут) или вероятностной догадкой модели.

    Точный ответ имеет вид «[EXACT · программа] ответ». Иначе: без файла весов модели —
    сообщение с подсказкой переобучить; с моделью — 100 байт генерации с температурой 0.6
    с пометкой, что это не сертифицированный ответ."""
    exact = _exact_answer(query)  # точный ответ или None
    if exact:  # найден точный маршрут
        prog, out = exact  # имя программы и ответ
        return f'[EXACT · {prog}] {out}'  # точный ответ с именем программы
    ckpt = lm.CKPT  # путь к файлу весов модели
    if not ckpt.exists():  # модель не обучена
        return '[NEURAL · нет чекпойнта] сначала: python -m hagi.mind retrain'  # сообщение: сначала переобучить
    model = lm.load()  # загрузить байтовую модель
    return '[NEURAL · вероятностная догадка, не сертифицировано] ' + model.generate(query, n_bytes=100, temp=0.6)  # пометка + 100 байт генерации с температурой 0.6
def retrain() -> str:
    """Переобучить байтовую модель на тексте собственных файлов проекта; вернуть строку-итог.

    Материал: scripts/generated_programs.py (если есть), затем io.py, loop.py, rewrite.py
    и два корневых документа проекта — только существующие файлы, их байты склеиваются.
    Модель ByteLM(k=10, dim=96, hidden=192), 90 эпох, журнал каждые 30; веса сохраняются."""
    root = Path(__file__).resolve().parent.parent  # корень проекта (каталог над hagi)
    pkg = root / 'hagi'  # каталог пакета
    from hagi.base.util import CKPT_DIR  # каталог контрольных точек (локальный импорт)
    base = CKPT_DIR / 'self'  # путь к контрольной точке «самого себя» (переменная далее не используется)
    gen_prog = root / 'scripts' / 'generated_programs.py'  # файл сгенерированных программ
    files = ([gen_prog] if gen_prog.exists() else []) + [root / f for f in SELF_SOURCES[1:]] + [root / 'README.md', root / 'THEORY_ALIGNMENT.md']  # файлы обучения: сгенерированные программы (если есть) + io.py, loop.py, rewrite.py + два корневых документа
    files = [f for f in files if f.exists()]  # оставить только существующие
    data = b''.join((f.read_bytes() for f in files))  # обучающие байты: конкатенация содержимого файлов
    model = lm.ByteLM(k=10, dim=96, hidden=192)  # новая байтовая модель: контекст 10, размерность 96, скрытый слой 192
    rep = model.train_guarded(data, epochs=90, log_every=30)  # обучение с гейтом: 90 эпох, запись в журнал каждые 30
    lm.CKPT.parent.mkdir(parents=True, exist_ok=True)  # создать каталог для файла весов
    lm.save(model)  # сохранить веса модели
    return f"LM перетренирована на {len(data)} байтах собственных артефактов ({len(files)} файлов): holdout CE {rep['best_holdout_ce']}"  # итог: число байтов, число файлов, лучшая кросс-энтропия отложенной части
def main() -> None:
    """Командная строка: ask <вопрос> — ответить; retrain — переобучить модель и напечатать итог."""
    p = argparse.ArgumentParser(description='HAGI hybrid mind')  # разборщик аргументов
    sub = p.add_subparsers(dest='cmd', required=True)  # обязательная подкоманда
    a1 = sub.add_parser('ask')  # подкоманда ask
    a1.add_argument('query')  # её единственный позиционный аргумент — вопрос
    sub.add_parser('retrain')  # подкоманда retrain
    a = p.parse_args()  # разобрать аргументы
    print(retrain() if a.cmd == 'retrain' else ask(a.query))  # выполнить подкоманду и напечатать результат
if __name__ == '__main__':  # запуск файла как скрипта
    main()  # точка входа командной строки
