"""Тест охвата старых доменов гейтом и ремонтом.

Правило: гейт забывания и ремонт измеряют CE ВСЕХ старых доменов книги
(настройка OLD_GATE_DOMAINS, 0 — без лимита), а не случайную выборку из 8.
Бюджет окон делится между доменами, поэтому цена полного охвата близка к
цене прежних 8 доменов.

Сторож: если охват снова ограничить 8 доменами, тест падает — домен,
разрушенный за пределами выборки, становится невидимым для гейта
(замер на живой книге: 55 из 130 доменов проседали больше 0.25 ната,
а гейт на 8 доменах сообщал «принят»)."""

import numpy as np  # массивы
import torch as t  # тензоры

from hagi.selfdev.unified import Unified  # цикл


class _Tiny(t.nn.Module):  # подставная модель: logits зависят от одного веса
    def __init__(self) -> None:  # создать
        super().__init__(); self.w = t.nn.Parameter(t.zeros(1)); self.k = 4; self.device = "cpu"; self.t = t  # вес, контекст
        self.bypass_t5: dict = {}; self.bypass_t3: dict = {}  # пустые словари обхода

    def logits(self, x: t.Tensor) -> t.Tensor:  # логиты следующего байта (смещение первого класса)
        return (t.zeros((x.shape[0], 256)) + self.w).requires_grad_(True)  # логиты последней позиции: форма (B, 256)

    def params(self):  # параметры
        return [self.w]  # список

    def param_names(self):  # имена
        return ["w"]  # список


class _Fake(Unified):  # заглушка цикла: только книга доменов и подставная модель
    def __init__(self, n_domains: int) -> None:  # создать с n доменами
        self.model = _Tiny(); self.domains = {f"d{i:03d}": {"ce": 1.0} for i in range(n_domains)}  # книга
        self._files = {f"d{i:03d}": bytes((i * 7 + j) % 256 for j in range(2000)) for i in range(n_domains)}  # байты доменов

    def _domain_bytes(self, name: str, entry: dict) -> bytes:  # байты домена из памяти
        return self._files[name]  # вернуть


def test_gate_covers_all_old_domains():  # гейт мерит все старые домены
    u = _Fake(n_domains=30)  # 30 доменов
    names, vec = u._old_domains_ce_vec(exclude="d029")  # замер вектора CE
    assert names is not None and vec is not None, "нет замера"  # замер есть
    assert len(names) == 29, f"гейт видит {len(names)} из 29 старых доменов — остальные невидимы"  # полный охват
    assert "d029" not in names, "новый домен попал в замер старых"  # новый исключён
    assert len(vec) <= u.OLD_GATE_WINDOWS + 29 * 16, "бюджет окон превышен"  # бюджет соблюдён


def test_gate_limit_is_configurable():  # лимит можно вернуть настройкой
    u = _Fake(n_domains=30)  # 30 доменов
    names, _ = u._old_domains_ce_vec(exclude="d029", max_domains=8)  # явный лимит 8
    assert len(names) == 8, "явный лимит не действует"  # ровно 8
    assert len(set(names)) == 8, "домены повторяются"  # без повторов
