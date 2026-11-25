"""Тесты самокритики законов run_law_self_critique: проверка нейросетевой модели по принятым в реестр законам и постановка исправляющего задания в очередь.

    Схема. Для каждого принятого закона (accepted_to_registry = True и статус
    PROVEN_EXACT / INVENTED / PARTIAL_LAW) берутся примеры-свидетельства,
    которые подтверждает сама символьная программа закона bp:
    bp.apply(feature_context) = target. Контексты подаются модели в
    хронологическом порядке, предсказание жадное: ŷ = argmax(logits).
    Метрика — точность повтора с верной историей:
      accuracy = 1 − (число ошибок) / n.
    Подпись ошибок = первые 16 знаков sha256 от списка (i, ŷ_i, target_i) по
    ошибочным примерам; .chat-файл ставится в inbox, только если ошибки есть
    и подпись изменилась. Поколение = число доменов без «chat_»; закон
    проверяется не чаще, чем раз в reevaluate_every поколений.

    Свойства: критика строится по закону и не повышает статус PARTIAL_LAW;
    контекст идёт в хронологическом порядке; непринятые и неизвестные законы
    игнорируются; одинаковые ошибки не ставятся в очередь повторно;
    периодическая проверка переоценивает закон только в срок; свидетельства
    сверяются с законом; метрика — повтор с верной историей; короткий контекст
    допустим; верный закон пишет отчёт, но ничего не ставит в очередь."""
import json  # чтение state.json и ledger.jsonl
from pathlib import Path  # типы путей
from types import SimpleNamespace  # объект-заглушка вместо Unified

import pytest  # сравнение с допуском (approx)
import torch  # тензоры логитов

from hagi.data.law_critique import run_law_self_critique  # самокритика законов


class IdentityLaw:
    """Закон-тождество: apply(ctx) = ctx[-1]; to_dict даёт описание закона для его отпечатка."""

    def apply(self, ctx):  # применить закон к кортежу байтов
        return int(ctx[-1])  # предсказание = последний байт контекста

    def to_dict(self):  # описание закона
        return {"basis": "test-identity", "payload": {"kind": "identity"}}  # базис и вид закона


class LastByteModel:
    """Модель-заглушка: наибольший логит всегда у байта always, поэтому argmax = always.

        k: длина контекста; device: устройство; always: байт-победитель;
        last_inputs: последний поданный батч контекстов (вложенные списки)."""

    def __init__(self, k=4, always=0):  # создать модель: длина контекста k, байт-победитель always
        self.k = k  # длина контекста
        self.device = "cpu"  # устройство — процессор
        self.always = always  # байт, который модель всегда называет
        self.last_inputs = None  # батч ещё не подавался

    def logits(self, x):  # логиты для батча x формы (n, k)
        self.last_inputs = x.detach().cpu().tolist()  # запомнить поданные контексты
        out = torch.zeros((x.shape[0], 256), dtype=torch.float32)  # нулевые логиты формы (n, 256)
        out[:, self.always] = 1.0  # логит байта always := 1 (остальные 0)
        return out  # вернуть логиты


def _u(status="PARTIAL_LAW", accepted=True, domains=1):
    """Заглушка Unified: реестр с одним законом IdentityLaw в домене «domain-A».

        status: статус проверки; accepted: принят ли закон в реестр;
        domains: число доменов d0..d(domains−1), оно же поколение."""
    evidence = [  # свидетели: контекст, короткий контекст признака, цель, номер источника
        {"context": [1, 2, 3, 7], "feature_context": [1, 2, 3, 7],  # пример 1
         "target": 7, "source_index": 10},  # цель 7, источник 10
        {"context": [4, 5, 6, 8], "feature_context": [4, 5, 6, 8],  # пример 2
         "target": 8, "source_index": 11},  # цель 8, источник 11
        {"context": [9, 10, 11, 9], "feature_context": [9, 10, 11, 9],  # пример 3
         "target": 9, "source_index": 12},  # цель 9, источник 12
    ]
    entry = {  # запись реестра: имя, статус, поддержка, принятие, программа, свидетели
        "feature": "feature_from_data",  # имя закона
        "verify": status,  # статус проверки
        "support": 0.62,  # поддержка 0,62
        "accepted_to_registry": accepted,  # принят в реестр
        "bp": IdentityLaw(),  # символьная программа закона
        "evidence": evidence,  # свидетели
    }
    return SimpleNamespace(  # объект-заглушка
        model=LastByteModel(k=4, always=0),  # модель всегда предсказывает байт 0
        law_registry={"domain-A": [entry]},  # реестр: один домен с одной записью
        domains={f"d{i}": {} for i in range(domains)},  # домены d0..d(domains−1)
    )


def test_generates_critique_from_registered_law_and_keeps_partial_status(tmp_path):
    """По принятому закону строится критика: оценён 1 закон, в очередь поставлен 1 файл .chat;
    в тексте есть статус PARTIAL_LAW и название метрики, нет hex-нулей, есть имя закона;
    статус PARTIAL_LAW не повышается."""
    u = _u()  # заглушка с законом в статусе PARTIAL_LAW
    inbox = tmp_path / "inbox"  # каталог очереди
    result = run_law_self_critique(  # прогнать самокритику
        u, inbox_dir=inbox, state_path=tmp_path / "state.json",  # очередь и файл состояния
        ledger_path=tmp_path / "ledger.jsonl",  # журнал отчётов
    )
    files = list(inbox.glob("*.chat"))  # файлы очереди
    assert result["evaluated"] == 1  # оценён 1 закон
    assert result["queued"] == 1  # поставлен в очередь 1 файл
    assert len(files) == 1  # в очереди ровно один файл
    text = files[0].read_text(encoding="utf-8")  # текст критики
    assert "Статус проверки закона: PARTIAL_LAW" in text  # в тексте статус PARTIAL_LAW
    assert "teacher-forced replay accuracy" in text  # названа метрика — точность повтора с верной историей
    assert "00000000" not in text  # в тексте нет hex-нулей
    assert "feature_from_data" in text  # в тексте есть имя закона
    # критика никогда не повышает статус неполного закона
    assert u.law_registry["domain-A"][0]["verify"] == "PARTIAL_LAW"  # статус проверки остался PARTIAL_LAW


def test_model_context_is_passed_in_chronological_order(tmp_path):
    """Контекст подаётся модели в хронологическом порядке: первый [1,2,3,7], второй [4,5,6,8]."""
    u = _u()  # заглушка
    model = u.model  # модель-заглушка
    run_law_self_critique(  # прогнать самокритику
        u, inbox_dir=tmp_path / "inbox", state_path=tmp_path / "state.json",  # очередь и файл состояния
        ledger_path=tmp_path / "ledger.jsonl",  # журнал отчётов
    )
    assert model.last_inputs[0] == [1, 2, 3, 7]  # первый контекст — как в свидетеле, без переворота
    assert model.last_inputs[1] == [4, 5, 6, 8]  # второй контекст — как в свидетеле


def test_unaccepted_or_unknown_laws_are_ignored(tmp_path):
    """Непринятый в реестр закон и закон со статусом NO_LAW не рассматриваются: registry_laws = 0,
    evaluated = 0, файлов .chat нет."""
    u = _u(accepted=False)  # закон не принят в реестр
    result = run_law_self_critique(  # прогнать самокритику
        u, inbox_dir=tmp_path / "inbox", state_path=tmp_path / "state.json",  # очередь и файл состояния
        ledger_path=tmp_path / "ledger.jsonl",  # журнал отчётов
    )
    assert result["registry_laws"] == 0  # в реестре нет принимаемых законов
    assert result["evaluated"] == 0  # ничего не оценено
    assert list((tmp_path / "inbox").glob("*.chat")) == []  # очередь пуста

    u = _u(status="NO_LAW")  # закон с неизвестным статусом
    result = run_law_self_critique(  # прогнать самокритику
        u, inbox_dir=tmp_path / "inbox2", state_path=tmp_path / "state2.json",  # другие очередь и состояние
        ledger_path=tmp_path / "ledger2.jsonl",  # другой журнал
    )
    assert result["registry_laws"] == 0  # неизвестный статус тоже не принимается


def test_repeated_identical_errors_are_not_queued_again(tmp_path):
    """Второй запуск с теми же ошибками не ставит задание в очередь: оценено 0, поставлено 0,
    в очереди по-прежнему один файл."""
    u = _u()  # заглушка
    args = dict(inbox_dir=tmp_path / "inbox",  # аргументы запуска: очередь,
                state_path=tmp_path / "state.json",  # файл состояния,
                ledger_path=tmp_path / "ledger.jsonl")  # журнал отчётов
    first = run_law_self_critique(u, **args)  # первый запуск
    second = run_law_self_critique(u, **args)  # второй запуск — то же состояние
    assert first["queued"] == 1  # первый запуск поставил 1 задание
    assert second["evaluated"] == 0  # второй запуск ничего не оценивал (поколение не дошло)
    assert second["queued"] == 0  # второй запуск ничего не поставил
    assert len(list((tmp_path / "inbox").glob("*.chat"))) == 1  # файл в очереди один


def test_periodic_reevaluation_only_requeues_changed_error_signature(tmp_path):
    """Периодическая проверка (раз в 2 поколения): при добавлении одного домена закон ещё не проверяется
    (2 − 1 < 2); после второго добавления и смены предсказаний модели (байт 7 вместо 0) закон проверяется
    и ставится в очередь; точность = 1 − 2/3 = 1/3 (верен только пример с целью 7)."""
    u = _u(domains=1)  # один домен d0 — поколение 1
    args = dict(inbox_dir=tmp_path / "inbox",  # аргументы запуска: очередь,
                state_path=tmp_path / "state.json",  # файл состояния,
                ledger_path=tmp_path / "ledger.jsonl",  # журнал отчётов,
                reevaluate_every=2)  # переоценивать раз в 2 поколения
    first = run_law_self_critique(u, **args)  # первый запуск: поколение 1
    assert first["queued"] == 1  # поставлен 1 файл
    u.domains["d1"] = {}  # добавлен домен d1 — поколение 2
    skipped = run_law_self_critique(u, **args)  # 2 − 1 = 1 < 2 — рано
    assert skipped["evaluated"] == 0  # закон не переоценён
    u.domains["d2"] = {}  # добавлен домен d2 — поколение 3
    u.model.always = 7  # модель теперь всегда предсказывает байт 7
    changed = run_law_self_critique(u, **args)  # 3 − 1 = 2 — пора, и подпись ошибок изменилась
    assert changed["evaluated"] == 1  # закон переоценён
    assert changed["queued"] == 1  # изменившаяся подпись — поставлен новый файл
    state = json.loads((tmp_path / "state.json").read_text())  # прочитать файл состояния
    assert list(state["laws"].values())[0]["accuracy"] == pytest.approx(1 / 3)  # точность = 1 − 2/3 = 1/3


def test_evidence_must_match_the_symbolic_law(tmp_path):
    """Свидетельство, противоречащее закону, отбрасывается: при испорченной цели 99 (закон даёт 7)
    оценены 2 примера из 3."""
    u = _u()  # заглушка
    # портим одну цель: принятая программа закона больше не подтверждает эту строку
    u.law_registry["domain-A"][0]["evidence"][0]["target"] = 99  # цель первого примера := 99, а закон даёт 7
    result = run_law_self_critique(  # прогнать самокритику
        u, inbox_dir=tmp_path / "inbox", state_path=tmp_path / "state.json",  # очередь и файл состояния
        ledger_path=tmp_path / "ledger.jsonl",  # журнал отчётов
    )
    assert result["evaluated"] == 1  # закон оценён
    report = result["reports"][0]  # отчёт по закону
    assert report["n"] == 2  # учтены 2 примера: противоречащий закону отброшен


def test_accuracy_report_is_teacher_forced_not_rollout(tmp_path):
    """Отчёт в журнале называет метрику teacher_forced_replay_accuracy и содержит n = 3.

        Повтор с верной историей: каждый контекст подан с правильными прежними байтами,
        а не получен свободной генерацией."""
    u = _u()  # заглушка
    result = run_law_self_critique(  # прогнать самокритику
        u, inbox_dir=tmp_path / "inbox", state_path=tmp_path / "state.json",  # очередь и файл состояния
        ledger_path=tmp_path / "ledger.jsonl",  # журнал отчётов
    )
    line = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]  # первая строка журнала
    report = json.loads(line)  # отчёт как словарь
    assert report["metric"] == "teacher_forced_replay_accuracy"  # метрика — точность повтора с верной историей
    assert report["n"] == 3  # оценены 3 примера


def test_shorter_unified_context_is_valid_below_attention_window(tmp_path):
    """Контекст короче окна внимания модели допустим: при k = 96 окна контекста 4 оценены все 3 примера."""
    u = _u()  # заглушка
    u.model.k = 96  # окно внимания модели 96; контексты свидетелей остаются длиной 4
    result = run_law_self_critique(  # прогнать самокритику
        u, inbox_dir=tmp_path / "inbox", state_path=tmp_path / "state.json",  # очередь и файл состояния
        ledger_path=tmp_path / "ledger.jsonl",  # журнал отчётов
    )
    assert result["evaluated"] == 1  # закон оценён
    assert result["reports"][0]["n"] == 3  # все 3 примера допустимы: 4 ≤ 96


def test_successful_law_still_writes_audit_but_does_not_queue_training(tmp_path):
    """Закон без ошибок пишет отчёт (accuracy = 1, errors = 0), но ничего не ставит в очередь.

        predict_fn подменяет предсказания модели верными целями [7, 8, 9]."""
    u = _u()  # заглушка
    result = run_law_self_critique(  # прогнать самокритику
        u,  # заглушка
        inbox_dir=tmp_path / "inbox",  # очередь
        state_path=tmp_path / "state.json",  # файл состояния
        ledger_path=tmp_path / "ledger.jsonl",  # журнал отчётов
        predict_fn=lambda _model, _contexts: [7, 8, 9],  # предсказания совпадают с целями
    )
    assert result["evaluated"] == 1  # закон оценён
    assert result["queued"] == 0  # в очередь ничего не поставлено
    assert not list((tmp_path / "inbox").glob("*.chat"))  # очередь пуста
    report = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[0])  # первая запись журнала
    assert report["accuracy"] == 1.0  # точность 1
    assert report["errors"] == 0  # ошибок 0
