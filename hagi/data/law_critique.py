"""Автоматическая самокритика по законам, уже принятым в реестр законов модели.

Источник истины — реестр law_registry; вручную написанных законов нет.
Участвуют только записи со статусом проверки PROVEN_EXACT, INVENTED или
PARTIAL_LAW и с сохранёнными примерами байтов, согласованными с законом.
Статус доказательства закону не повышается: PARTIAL_LAW остаётся PARTIAL_LAW.

Метрика называется teacher_forced_replay_accuracy: каждое предсказание
модели получает истинный хронологический контекст. Это не свободная
авторегрессионная генерация и не утверждение об обобщении на новые данные.

Ход работы (run_law_self_critique):
  1) взять принятые законы, отсортировать по (домен, feature, отпечаток);
  2) закон оценивается, если generation − last_generation >= reevaluate_every,
     где generation — число доменов, не начинающихся с "chat_";
  3) отобрать примеры, где bp.apply(feature_context) == target;
  4) accuracy = 1 − |ошибки| / max(1, |примеры|);
  5) при ошибках и новом отпечатке error_signature — атомарно записать
     .chat-файл в каталог inbox; повтор того же набора ошибок не записывается.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import hashlib  # хеширование
import json  # чтение и запись JSON
import os  # работа с путями и переменными окружения
import re  # регулярные выражения
import time  # отметки времени
from pathlib import Path  # пути к файлам
from typing import Any, Callable, Iterable  # типы «любой», «вызываемый объект», «итерируемый»


_ACCEPTED_STATUSES = {"PROVEN_EXACT", "INVENTED", "PARTIAL_LAW"}  # статусы проверки, при которых закон допускается к критике: PROVEN_EXACT, INVENTED, PARTIAL_LAW
_DEFAULT_REEVALUATE_EVERY = 4  # переоценивать закон не чаще, чем раз в 4 поколения
_DEFAULT_MAX_LAWS_PER_RUN = 8  # не более 8 законов за один запуск
_DEFAULT_MAX_EVIDENCE = 256  # не более 256 примеров на закон
_DEFAULT_MAX_ERROR_EXAMPLES = 64  # не более 64 ошибок в тексте критики
_DEFAULT_MAX_GOOD_EXAMPLES = 16  # не более 16 верных пар в тексте критики


def _json_default(value: Any) -> Any:  # value — произвольное значение
    if isinstance(value, (set, frozenset, tuple)):  # множество, замороженное множество или кортеж
        return list(value)  # превратить в список
    if hasattr(value, "tolist"):  # у массива numpy или тензора есть tolist
        return value.tolist()  # взять список значений
    return repr(value)  # иначе — текстовое представление


def _blueprint_payload(entry: dict[str, Any]) -> Any:  # entry — запись реестра
    bp = entry.get("bp")  # bp — чертёж закона (объект с apply) или None
    if bp is None:  # чертежа нет
        return entry.get("blueprint", {})  # взять словарь blueprint из записи или пустой словарь
    if hasattr(bp, "to_dict"):  # у объекта есть to_dict
        try:  # вызов может не удаться
            return bp.to_dict()  # сериализовать чертёж словарём
        except Exception:  # любая ошибка сериализации
            pass  # пропустить и собрать словарь ниже
    return {  # словарь из атрибутов чертежа
        "q_name": getattr(bp, "q_name", None),  # имя характеристики или None
        "q_param": getattr(bp, "q_param", None),  # параметр характеристики или None
        "basis": getattr(bp, "basis", None),  # имя базиса или None
        "payload": getattr(bp, "payload", None),  # нагрузка базиса или None
    }  # закрытие словаря


def law_fingerprint(domain: str, entry: dict[str, Any]) -> str:  # domain — домен, entry — запись реестра
    """Стабильный идентификатор одного принятого закона."""
    explicit = entry.get("commit_id") or entry.get("law_id")  # явный идентификатор: commit_id или law_id
    if explicit:  # он задан
        return str(explicit)  # вернуть его как строку
    material = {  # материал для отпечатка
        "domain": domain,  # домен
        "feature": entry.get("feature"),  # характеристика
        "verify": entry.get("verify"),  # статус проверки
        "support": entry.get("support"),  # опора
        "blueprint": _blueprint_payload(entry),  # нагрузка чертежа
    }  # закрытие словаря
    raw = json.dumps(material, sort_keys=True, ensure_ascii=False,  # raw = JSON материала с сортировкой ключей, без лишних пробелов
                     default=_json_default, separators=(",", ":"))  # продолжение json.dumps: default и разделители
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]  # отпечаток: первые 20 шестнадцатеричных цифр sha256(raw в utf-8)


def _accepted(entry: dict[str, Any]) -> bool:  # entry — запись реестра
    """Запись принята в реестр; проверка остаётся отдельным вопросом."""
    if entry.get("accepted_to_registry", True) is not True:  # флаг accepted_to_registry не равен True
        return False  # не принята
    return str(entry.get("verify", "")).upper() in _ACCEPTED_STATUSES  # принята, если статус проверки (в верхнем регистре) входит в набор допустимых


def iter_accepted_laws(  # реестр: домен -> список записей
    law_registry: dict[str, list[dict[str, Any]]],  # продолжение типа первого аргумента
) -> Iterable[tuple[str, dict[str, Any]]]:  # возвращает пары (домен, запись)
    for domain in sorted(law_registry):  # по доменам в алфавитном порядке
        for entry in law_registry.get(domain, []) or []:  # по записям домена (пустой список, если ключа нет)
            if isinstance(entry, dict) and _accepted(entry):  # запись — словарь и принята
                yield domain, entry  # выдать пару (домен, запись)


def _normalize_evidence(  # начало определения: запись реестра
    entry: dict[str, Any],  # entry — запись реестра
    model_context_length: int | None,  # model_context_length — контекст модели или None
    max_examples: int = _DEFAULT_MAX_EVIDENCE,  # max_examples — предел примеров
) -> list[dict[str, Any]]:  # возврат — список примеров
    """Вернуть только примеры, согласованные с принятым символьным законом.

        context — полный хронологический контекст модели; feature_context —
        короткий кортеж, который использует символьный чертёж (обычно последние
        четыре байта)."""
    bp = entry.get("bp")  # bp — чертёж закона или None
    raw = entry.get("evidence") or []  # raw — исходные примеры (или пустой список)
    rows: list[dict[str, Any]] = []  # накопитель строк
    for item in raw:  # по примерам
        if not isinstance(item, dict):  # пример не словарь
            continue  # пропустить
        try:  # разбор полей может не удаться
            context = [int(v) for v in item["context"]]  # context — список целых
            feature_context = [int(v) for v in item["feature_context"]]  # feature_context — список целых
            target = int(item["target"])  # target — целое
        except (KeyError, TypeError, ValueError):  # нет поля, неверный тип или значение
            continue  # пропустить пример
        if not context or not feature_context or not 0 <= target <= 255:  # пустой контекст, пустой feature_context или target вне 0..255
            continue  # пропустить пример
        # # модель может использовать более короткий контекст домена (Unified.k),
        # # чем максимальное окно внимания (model.k); оба допустимы.
        if model_context_length and len(context) > model_context_length:  # контекст модели задан и пример длиннее его
            continue  # пропустить пример
        if any(v < 0 or v > 255 for v in context + feature_context):  # есть байт вне 0..255 в context или feature_context
            continue  # пропустить пример
        # # перед использованием примера для самообучения закон проверяется ещё раз.
        if bp is not None:  # чертёж задан
            try:  # применение может вызвать ошибку
                if int(bp.apply(tuple(feature_context))) != target:  # проверка закона: bp.apply(feature_context) != target
                    continue  # пример противоречит закону — пропустить
            except Exception:  # любая ошибка применения
                continue  # пропустить пример
        rows.append({  # добавить строку
            "context": context,  # полный контекст
            "feature_context": feature_context,  # контекст характеристики
            "target": target,  # целевой байт
            "source_index": int(item.get("source_index", len(rows))),  # исходный индекс или номер строки
        })  # закрытие словаря (строка-примера)
    if len(rows) <= max_examples:  # примеров не больше предела
        return rows  # вернуть все
    # # детерминированная равномерная выборка вместо смещённого начала списка
    last = len(rows) - 1  # индекс последнего элемента: last = len(rows) − 1
    indices = [round(i * last / (max_examples - 1))  # индексы: indices_i = round(i·last/(max_examples − 1)), i = 0..max_examples−1
               for i in range(max_examples)] if max_examples > 1 else [0]  # если max_examples <= 1 — только индекс 0
    return [rows[i] for i in indices]  # вернуть строки с этими индексами


def _model_predictions(model: Any, contexts: list[list[int]],  # model — модель, contexts — списки контекстов, batch_size — размер пакета
                       batch_size: int = 64) -> list[int]:  # продолжение сигнатуры
    """Жадные предсказания модели; хронологический порядок контекста сохранён."""
    import torch  # библиотека тензоров

    if not contexts:  # контекстов нет
        return []  # пустой список
    device = getattr(model, "device", "cpu")  # устройство модели (по умолчанию cpu)
    predictions: list[int] = []  # накопитель предсказаний
    with torch.no_grad():  # без градиентов
        for start in range(0, len(contexts), batch_size):  # по пакетам контекстов
            batch = contexts[start:start + batch_size]  # текущий пакет
            x = torch.as_tensor(batch, dtype=torch.long, device=device)  # тензор пакета целых на устройстве модели
            logits = model.logits(x)  # логиты model.logits(x)
            predictions.extend(int(v) for v in logits.argmax(dim=-1).detach().cpu().tolist())  # дополнить предсказаниями argmax по последней оси, в int, на cpu
    return predictions  # список предсказаний


def _safe_name(value: str) -> str:  # value — исходное имя
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")  # заменить всё вне латиницы, цифр, «_», «.», «−» на «_», срезать крайние точки и подчёркивания
    return cleaned[:48] or "law"  # первые 48 символов или "law"


def _hex(values: list[int]) -> str:  # values — список байтов
    return bytes(values).hex()  # шестнадцатеричный текст байтов


def _render_critique(domain: str, entry: dict[str, Any], law_id: str,  # domain, entry, law_id, rows, predictions, errors, generation — данные критики
                     rows: list[dict[str, Any]], predictions: list[int],  # rows — примеры, predictions — предсказания,
                     errors: list[int], generation: int) -> str:  # errors — номера ошибок, generation — поколение
    status = str(entry.get("verify", "UNKNOWN")).upper()  # статус проверки в верхнем регистре (или UNKNOWN)
    support = entry.get("support")  # опора закона или None
    support_text = "не указан" if support is None else f"{float(support):.3f}"  # текст опоры: «не указан» или число с тремя знаками
    feature = str(entry.get("feature", "unnamed"))  # имя характеристики (или unnamed)
    lines = [  # начало списка строк текста критики
        "АВТОСАМОКРИТИКА HAGI: обучающий материал построен по закону,",  # строка заголовка 1
        "уже принятому в law_registry. Это не новый закон и не новое доказательство.",  # строка заголовка 2
        f"Закон: {feature}",  # строка: закон
        f"Домен: {domain}",  # строка: домен
        f"Статус проверки закона: {status}",  # строка: статус проверки
        f"Поддержка закона: {support_text}",  # строка: опора
        f"Идентификатор коммита закона: {law_id}",  # строка: идентификатор
        f"Поколение обучения: {generation}",  # строка: поколение
        "Пары ниже повторно проверены самой символьной программой закона.",  # строка описания
        "Контекст записан в хронологическом порядке; после стрелки — верный следующий байт.",  # строка описания
        "Исправляй нейросетевое предсказание по этим подтверждённым парам.",  # строка описания
        "",  # пустая строка
        "ОШИБКИ МОДЕЛИ (context_hex -> target_byte_hex ; predicted_byte_hex):",  # заголовок списка ошибок
    ]  # конец списка
    for i in errors[:_DEFAULT_MAX_ERROR_EXAMPLES]:  # по первым _DEFAULT_MAX_ERROR_EXAMPLES ошибкам
        row = rows[i]  # строка примера
        lines.append(  # добавить строку
            f"{_hex(row['context'])} -> {row['target']:02x} ; "  # строка «контекст → цель»: context_hex -> target_hex ;
            f"predicted={predictions[i]:02x}"  # predicted=предсказание в hex
        )  # закрытие вызова
    error_set = set(errors)  # множество номеров ошибок
    good = [i for i in range(len(rows)) if i not in error_set]  # номера примеров без ошибки
    if good:  # они есть
        lines.extend(["", "КОНТРОЛЬНЫЕ ВЕРНЫЕ ПАРЫ:"])  # добавить пустую строку и заголовок верных пар
        for i in good[:_DEFAULT_MAX_GOOD_EXAMPLES]:  # по первым _DEFAULT_MAX_GOOD_EXAMPLES верным
            row = rows[i]  # строка примера
            lines.append(f"{_hex(row['context'])} -> {row['target']:02x}")  # строка «контекст → цель»: context_hex -> target_hex
    lines.extend([  # добавить заключительные строки
        "",  # пустая строка
        "Метрика выше — teacher-forced replay accuracy: каждый контекст подан",  # строка пояснения метрики 1
        "в исходном порядке с правильной историей. Это не свободная генерация",  # строка пояснения метрики 2
        "и не доказательство обобщения за пределы поддержки закона.",  # строка пояснения метрики 3
    ])  # закрытие списка
    return "\n".join(lines) + "\n"  # соединить строки через перевод строки и добавить перевод строки в конце


def _atomic_json(path: Path, value: Any) -> None:  # path — путь, value — значение
    path.parent.mkdir(parents=True, exist_ok=True)  # создать родительский каталог при необходимости
    tmp = path.with_suffix(path.suffix + ".tmp")  # временный файл: путь с добавленным суффиксом .tmp
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,  # записать JSON (без экранирования не-ASCII, отступ 2,
                              sort_keys=True), encoding="utf-8")  # с сортировкой ключей) во временный файл
    os.replace(tmp, path)  # атомарно заменить целевой файл временным


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:  # path — путь, value — словарь
    path.parent.mkdir(parents=True, exist_ok=True)  # создать родительский каталог при необходимости
    with path.open("a", encoding="utf-8") as fh:  # открыть файл на дозапись
        fh.write(json.dumps(value, ensure_ascii=False, sort_keys=True,  # дописать JSON одной строкой (сортировка ключей,
                            default=_json_default) + "\n")  # default для нестандартных типов) и перевод строки


def _default_paths() -> tuple[Path, Path, Path]:  # возврат — пути inbox, файла состояния и журнала
    try:  # попытка импорта
        from hagi.base.util import CKPT_DIR  # каталог сохранённых состояний
        root = Path(CKPT_DIR)  # корень — этот каталог
    except Exception:  # импорт не удался
        root = Path(os.environ.get("HAGI_CKPT_DIR", "checkpoints"))  # корень — из переменной окружения HAGI_CKPT_DIR или "checkpoints"
    return (root / "inbox", root / "law_critique_state.json",  # возврат: путь inbox, продолжение на следующих строках
            root / "law_critique_ledger.jsonl")  # путь файла состояния и путь журнала


def run_law_self_critique(  # самокритика законов; параметры — только именованные
    u: Any,  # u — единая модель
    *,  # только именованные аргументы
    inbox_dir: str | Path | None = None,  # каталог inbox или None (по умолчанию)
    state_path: str | Path | None = None,  # путь файла состояния или None
    ledger_path: str | Path | None = None,  # путь журнала или None
    reevaluate_every: int = _DEFAULT_REEVALUATE_EVERY,  # переоценка не чаще, чем раз в reevaluate_every поколений
    max_laws_per_run: int = _DEFAULT_MAX_LAWS_PER_RUN,  # не более max_laws_per_run законов за запуск
    max_evidence: int = _DEFAULT_MAX_EVIDENCE,  # не более max_evidence примеров на закон
    predict_fn: Callable[[Any, list[list[int]]], list[int]] | None = None,  # функция предсказания или None (тогда сама модель)
) -> dict[str, Any]:  # ) -> словарь результата
    """Оценить только что принятые законы и периодически возвращаться к ним.

        При ошибках модели атомарно поставить в очередь .chat-файл в каталог
        входящих сообщений живого обучения; живой цикл читает его обычным
        путём. Повтор того же набора ошибок в очередь не ставится."""
    default_inbox, default_state, default_ledger = _default_paths()  # пути по умолчанию
    inbox = Path(inbox_dir) if inbox_dir is not None else default_inbox  # каталог inbox: данный или по умолчанию
    state_file = Path(state_path) if state_path is not None else default_state  # файл состояния: данный или по умолчанию
    ledger_file = Path(ledger_path) if ledger_path is not None else default_ledger  # файл журнала: данный или по умолчанию
    inbox.mkdir(parents=True, exist_ok=True)  # создать inbox при необходимости

    registry = getattr(u, "law_registry", {}) or {}  # реестр законов (или пустой словарь)
    model = getattr(u, "model", None)  # модель (или None)
    if model is None:  # модели нет
        return {"status": "SKIP", "reason": "no_model", "evaluated": 0,  # пропуск: нет модели,
                "queued": 0}  # оценено 0, в очереди 0
    model_k = getattr(model, "k", None)  # контекст модели model.k (или None)
    # # домены chat_, созданные критикой, НЕ должны считаться «новыми обучающими
    # # доменами»: иначе каждое исправляющее сообщение сдвигает часы переоценки,
    # # и цикл самоускоряется — критика почти на каждом куске в ущерб реальным
    # # кускам с диска (за одну сессию поколения 79 -> 91).
    # nearly every chunk, starving real disk chunks).  # закрытие комментария-блока (пустая строка-комментарий не требуется)
    _doms = getattr(u, "domains", {}) or {}  # домены модели (или пустой словарь)
    generation = max(0, sum(1 for d in _doms  # generation = max(0, число доменов d,
                            if not str(d).startswith("chat_")))  # имя которых не начинается с "chat_")

    try:  # чтение состояния
        state = json.loads(state_file.read_text(encoding="utf-8"))  # state = JSON файла состояния
        if not isinstance(state, dict):  # state не словарь
            state = {}  # пустой словарь
    except (OSError, ValueError, TypeError):  # файл не читается или JSON повреждён
        state = {}  # пустой словарь
    if not isinstance(state.get("laws"), dict):  # state["laws"] не словарь
        state["laws"] = {}  # пустой словарь законов

    evaluated = 0  # счётчик оценённых законов
    queued = 0  # счётчик поставленных в очередь
    skipped = 0  # счётчик пропущенных
    reports: list[dict[str, Any]] = []  # список отчётов
    laws = list(iter_accepted_laws(registry))  # список принятых законов (домен, запись)
    laws.sort(key=lambda pair: (pair[0], str(pair[1].get("feature", "")),  # сортировка по (домен, feature,
                                law_fingerprint(pair[0], pair[1])))  # отпечаток закона)

    for domain, entry in laws:  # по законам
        if evaluated >= max_laws_per_run:  # достигнут предел числа законов за запуск
            break  # прервать цикл
        law_id = law_fingerprint(domain, entry)  # отпечаток закона
        previous = state["laws"].get(law_id, {})  # прежнее состояние закона (или пустой словарь)
        if not isinstance(previous, dict):  # не словарь
            previous = {}  # пустой словарь
        last_generation = int(previous.get("last_generation", -10**9))  # поколение прошлой оценки или −10^9 (никогда не оценивался)
        if generation - last_generation < max(1, reevaluate_every):  # прошло меньше max(1, reevaluate_every) поколений
            skipped += 1  # пропущено: +1
            continue  # следующий закон

        evidence = _normalize_evidence(entry, model_k, max_evidence)  # примеры, согласованные с законом
        if not evidence:  # примеров нет
            skipped += 1  # пропущено: +1
            state["laws"][law_id] = {  # записать состояние закона:
                **previous,  # прежние поля,
                "domain": domain,  # домен,
                "feature": entry.get("feature"),  # характеристика,
                "verify": entry.get("verify"),  # статус проверки,
                "reason": "no_law_consistent_full_context_evidence",  # причина: нет согласованных примеров с полным контекстом,
                "last_generation": generation,  # поколение оценки,
            }  # закрытие словаря
            continue  # следующий закон

        contexts = [row["context"] for row in evidence]  # список полных контекстов примеров
        try:  # предсказание может вызвать ошибку
            if predict_fn is None:  # своя функция предсказания не задана
                predictions = _model_predictions(model, contexts)  # предсказания модели
            else:  # иначе
                predictions = [int(v) for v in predict_fn(model, contexts)]  # предсказания predict_fn как целые
            if len(predictions) != len(evidence):  # число предсказаний не равно числу примеров
                raise ValueError("predict_fn returned a different number of predictions")  # ошибка значения
        except Exception as exc:  # любая ошибка
            skipped += 1  # пропущено: +1
            state["laws"][law_id] = {  # записать состояние закона:
                **previous,  # прежние поля,
                "domain": domain,  # домен,
                "feature": entry.get("feature"),  # характеристика,
                "verify": entry.get("verify"),  # статус проверки,
                "last_generation": generation,  # поколение оценки,
                "error": repr(exc),  # текст ошибки,
            }  # закрытие словаря
            reports.append({"law_id": law_id, "domain": domain,  # добавить отчёт: law_id, домен,
                            "feature": entry.get("feature"),  # характеристика,
                            "status": "EVAL_ERROR", "error": repr(exc)})  # статус EVAL_ERROR и текст ошибки
            continue  # следующий закон

        targets = [int(row["target"]) for row in evidence]  # целевые байты примеров
        errors = [i for i, (pred, target) in enumerate(zip(predictions, targets))  # номера ошибок: i, где pred_i != target_i
                  if pred != target]  # продолжение условия
        accuracy = 1.0 - len(errors) / max(1, len(evidence))  # точность: accuracy = 1 − |errors| / max(1, |evidence|)
        signature_material = [(i, predictions[i], targets[i]) for i in errors]  # материал отпечатка ошибок: (i, pred_i, target_i) по ошибкам
        signature = hashlib.sha256(json.dumps(  # signature = первые 16 цифр sha256(JSON материала
            signature_material, separators=(",", ":")).encode("ascii")  # с разделителями без пробелов, в ascii)
        ).hexdigest()[:16]  # продолжение хеширования
        report = {  # отчёт:
            "law_id": law_id,  # идентификатор закона
            "domain": domain,  # домен
            "feature": entry.get("feature"),  # характеристика
            "verification_status": str(entry.get("verify", "UNKNOWN")),  # статус проверки
            "accepted_to_registry": True,  # принят в реестр: True
            "support": entry.get("support"),  # опора
            "generation": generation,  # поколение
            "metric": "teacher_forced_replay_accuracy",  # имя метрики
            "n": len(evidence),  # число примеров
            "correct": len(evidence) - len(errors),  # число верных
            "errors": len(errors),  # число ошибок
            "accuracy": round(accuracy, 6),  # точность с 6 знаками
            "error_signature": signature,  # отпечаток ошибок
        }  # закрытие словаря
        reports.append(report)  # добавить отчёт
        _append_jsonl(ledger_file, report)  # дописать отчёт в журнал

        old_signature = previous.get("error_signature")  # прежний отпечаток ошибок
        queued_this_law = False  # для этого закона в очередь ничего не поставлено
        if errors and signature != old_signature:  # есть ошибки и отпечаток новый
            message = _render_critique(domain, entry, law_id, evidence,  # текст критики
                                       predictions, errors, generation)  # продолжение вызова
            name = _safe_name(domain)  # безопасное имя домена для файла
            chat_path = inbox / f"critique_law_{name}_{law_id}_g{generation}.chat"  # путь .chat-файла: critique_law_<домен>_<law_id>_g<generation>.chat
            if not chat_path.exists():  # файла ещё нет
                tmp = chat_path.with_suffix(chat_path.suffix + ".tmp")  # временный файл .tmp
                tmp.write_text(message, encoding="utf-8")  # записать текст критики
                os.replace(tmp, chat_path)  # атомарно переименовать в целевой файл
                queued += 1  # в очереди: +1
                queued_this_law = True  # для этого закона поставлено

        state["laws"][law_id] = {  # записать состояние закона:
            "domain": domain,  # домен
            "feature": entry.get("feature"),  # характеристика
            "verify": entry.get("verify"),  # статус проверки
            "support": entry.get("support"),  # опора
            "last_generation": generation,  # поколение оценки
            "error_signature": signature,  # отпечаток ошибок
            "accuracy": round(accuracy, 6),  # точность с 6 знаками
            "n": len(evidence),  # число примеров
            "queued": queued_this_law,  # поставлен ли файл в очередь
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),  # время обновления ГОД-МЕСЯЦ-ДЕНЬ ЧЧ:ММ:СС
        }  # закрытие словаря
        evaluated += 1  # оценено: +1

    _atomic_json(state_file, state)  # атомарно сохранить состояние
    return {  # возврат:
        "status": "OK" if evaluated or queued else "NO_PENDING_LAWS",  # статус OK, если что-то оценено или поставлено в очередь, иначе NO_PENDING_LAWS
        "registry_laws": len(laws),  # число законов в реестре
        "evaluated": evaluated,  # число оценённых
        "queued": queued,  # число поставленных в очередь
        "skipped": skipped,  # число пропущенных
        "reports": reports,  # отчёты
        "state_path": str(state_file),  # путь файла состояния
        "ledger_path": str(ledger_file),  # путь журнала
    }  # закрытие словаря
