#!/usr/bin/env python3
"""RDL (Representation Discovery Loop) tests: invent-a-construction cycle."""

from hagi import rdl


class ModWorld:
    clock_action = 0

    def __init__(self, n):
        self._n = n
        self._s = 0

    def reset(self):
        self._s = 0
        return 0

    def step(self, a):
        if a == 0:
            self._s = (self._s + 1) % 16
        elif a == 1:
            self._s = rdl.mk_cnt_mod(self._n)(self._s)
        return self._s

    def true_step(self, s, a):
        if a == 0:
            return (s + 1) % 16
        if a == 1:
            return rdl.mk_cnt_mod(self._n)(s)
        return s


class LfsrWorld:
    clock_action = 0

    def __init__(self, p):
        self._p = p
        self._s = 1  # LFSR stalls at 0 — start at 1

    def reset(self):
        self._s = 1
        return 1

    def step(self, a):
        if a == 0:
            self._s = (self._s + 1) % 16
        elif a == 1:
            self._s = rdl.mk_lfsr(self._p)(self._s)
        return self._s

    def true_step(self, s, a):
        if a == 0:
            return (s + 1) % 16
        if a == 1:
            return rdl.mk_lfsr(self._p)(s)
        return s


def test_rdl_discovers_counter_mod5():
    r = rdl.run_rdl(ModWorld(5))
    assert r["proven"], r
    assert r["invented_class"] == "cnt_mod"
    assert r["param"] == 5
    assert r["savings_pct"] > 0


def test_rdl_discovers_lfsr():
    r = rdl.run_rdl(LfsrWorld(9))
    assert r["proven"], r
    assert r["invented_class"] == "lfsr"
    assert r["param"] == 9


def test_rdl_holds_on_unknown():
    # мир с динамикой вне всех семей: xor с константой 7
    class XorWorld:
        clock_action = 0

        def reset(self):
            return 0

        def step(self, a):
            return 0

        def true_step(self, s, a):
            return (s ^ 7) if a == 1 else s

    r = rdl.run_rdl(XorWorld())
    assert not r["proven"]
    assert r["invented_class"] is None


def test_families_space():
    fams = rdl.generate_families()
    assert len(fams) >= 30
    names = {n for n, _, _ in fams}
    assert {"lfsr", "cnt_mod", "sat", "gray", "shift2"} <= names
