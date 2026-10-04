"""Tests for the universality metric (F2).

"GENERAL" used to be a self-assessment: the benchmark printed a spread
and a human decided whether it counted. These tests pin the number to
properties that make it impossible to pass by accident -- in particular
that it actually MOVES when a model gets worse on one domain, which the
spread alone would not guarantee.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "growth_benchmark", ROOT / "scripts" / "growth_benchmark.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gb = _load()


# --- the anchors ---------------------------------------------------------


def test_the_floor_is_the_measured_unigram_entropy_not_ln_v():
    """ln V was tried first and made the metric collapse.

    A floor ABOVE every model's score leaves no range to measure, so the
    floor has to be the corpus's own unigram entropy (8.06), not the
    vocabulary entropy (10.40) that the code also knows about.
    """
    assert gb.UNIGRAM_CE == pytest.approx(8.06)
    assert gb.UNIGRAM_CE < 10.40  # strictly below ln V


def test_the_real_gen3_numbers_score_above_zero():
    """The metric must discriminate on the models that exist.

    Both generations score well clear of zero, and gen-3 -- the one that
    is better on all four domains -- scores higher.
    """
    gen3 = gb.universality({"CODE": 1.7223, "EN": 3.9425, "MATH": 3.7323,
                            "RU": 3.8959})
    gen2 = gb.universality({"CODE": 1.8051, "EN": 4.0358, "MATH": 3.8105,
                            "RU": 4.0863})
    assert gen3["universality"] > 0.5
    assert gen2["universality"] > 0.5
    assert gen3["universality"] > gen2["universality"]


# --- the bounds ---------------------------------------------------------


def test_identical_domains_score_one():
    assert gb.universality({"A": 3.0, "B": 3.0})["universality"] == \
        pytest.approx(1.0)


def test_a_worst_domain_at_the_floor_scores_zero():
    """No better than unigram frequencies on its worst domain."""
    assert gb.universality({"A": 2.0, "B": gb.UNIGRAM_CE})["universality"] == \
        pytest.approx(0.0)


def test_uniform_noise_scores_zero():
    assert gb.universality({"A": 10.4, "B": 10.4})["universality"] == 0.0


def test_the_score_never_leaves_the_unit_interval():
    """Even for absurd inputs, including a model worse than the floor."""
    for d in (
        {"A": 1.0, "B": 2.0},
        {"A": 20.0, "B": 30.0},          # everything worse than the floor
        {"A": 0.0, "B": 0.0},
        {"A": gb.UNIGRAM_CE - 1, "B": 0.5},
    ):
        u = gb.universality(d)["universality"]
        assert 0.0 <= u <= 1.0


def test_a_model_worse_than_the_floor_everywhere_is_zero_not_negative():
    d = {"A": 12.0, "B": 14.0}
    assert gb.universality(d)["universality"] == 0.0


# --- it must move -------------------------------------------------------


def test_degrading_one_domain_lowers_the_score():
    """The property spread does NOT guarantee, which is why this metric exists.

    Doubling the worst domain's CE changes the spread only slightly, but
    it should visibly move a metric that is anchored to a real floor.
    """
    good = {"CODE": 2.0, "EN": 3.0, "MATH": 3.0, "RU": 3.0}
    worse = dict(good, RU=4.5)
    u_good = gb.universality(good)["universality"]
    u_worse = gb.universality(worse)["universality"]
    assert u_worse < u_good
    # ... by an amount comparable to the change, not a rounding error.
    assert u_good - u_worse > 0.2


def test_uniform_mediocrity_is_not_generality():
    """The failure mode this metric exists to catch."""
    mediocre = {"A": 6.0, "B": 6.0}          # small spread, poor everywhere
    sharp = {"A": 2.0, "B": 7.0}             # large spread, one good domain
    assert mediocre["A"] == mediocre["B"]
    assert gb.universality(mediocre)["universality"] > \
        gb.universality(sharp)["universality"]


def test_improving_the_best_domain_alone_lowers_the_score():
    """A better best domain widens the gap the worst domain must cover."""
    fixed_worst = {"A": 2.0, "B": 4.0}
    better_best = {"A": 1.0, "B": 4.0}
    assert gb.universality(better_best)["universality"] < \
        gb.universality(fixed_worst)["universality"]


def test_improving_the_worst_domain_raises_the_score():
    d = {"A": 2.0, "B": 6.0}
    better = {"A": 2.0, "B": 4.0}
    assert gb.universality(better)["universality"] > \
        gb.universality(d)["universality"]


def test_only_the_two_extreme_domains_move_the_score():
    """The metric is anchored to the best and worst domains, by design.

    Worsening a MIDDLE domain changes neither extreme, so the score does
    not move. That is a property, not a defect: "is this model general"
    is a question about the extremes, and a domain sitting between them is
    evidence about neither.
    """
    base = {"A": 2.0, "B": 3.0, "C": 4.0}
    u_base = gb.universality(base)["universality"]
    # B is the middle: 3.0 -> 3.9 stays under C, so nothing changes.
    assert gb.universality(dict(base, B=3.9))["universality"] == \
        pytest.approx(u_base)
    # Push B past C and it becomes the new worst: the score falls.
    assert gb.universality(dict(base, B=4.5))["universality"] < u_base


def test_worsening_the_best_domain_can_RAISE_the_score_and_that_is_documented():
    """The metric is a RATIO, so the counter-intuitive case is real.

    (F - worst) / (F - best): if the best domain degrades, the denominator
    shrinks faster than the numerator and the score goes UP. So the
    number is "how close is the worst domain to the best one, in
    unigram-floor units" -- it is NOT "how good is the model". A model
    that got uniformly worse can look more general here.

    This is why the metric is read alongside ``avg`` and never alone, and
    it is pinned as a test so the behaviour cannot drift silently.
    """
    d = {"A": 2.0, "B": 3.0, "C": 4.0}
    worse_best = dict(d, A=2.9)
    assert gb.universality(worse_best)["universality"] > \
        gb.universality(d)["universality"]
    # ... and the model really did get worse on average.
    assert sum(worse_best.values()) / 3 > sum(d.values()) / 3


# --- the shape of the result --------------------------------------------


def test_avg_is_not_treated_as_a_domain():
    """``AVG`` is a summary; counting it would move the worst domain."""
    without = gb.universality({"A": 2.0, "B": 4.0})
    with_avg = gb.universality({"A": 2.0, "B": 4.0, "AVG": 3.0})
    assert with_avg["n_domains"] == without["n_domains"] == 2
    assert with_avg["universality"] == pytest.approx(without["universality"])


def test_the_reported_anchors_match_the_input():
    d = {"CODE": 1.7223, "EN": 3.9425, "MATH": 3.7323, "RU": 3.8959}
    r = gb.universality(d)
    assert r["best_ce"] == pytest.approx(1.7223)
    assert r["worst_ce"] == pytest.approx(3.9425)
    assert r["n_domains"] == 4
    assert r["floor_ce"] == pytest.approx(gb.UNIGRAM_CE)


def test_the_floor_is_overridable_for_a_different_corpus():
    """A corpus with a different entropy must not borrow this one's floor."""
    r = gb.universality({"A": 5.0, "B": 6.0}, floor_ce=7.0)
    assert r["floor_ce"] == pytest.approx(7.0)


def test_an_empty_domain_set_is_refused_not_silently_scored():
    with pytest.raises(ValueError):
        gb.universality({})


# --- integration with report() -------------------------------------------


def test_report_carries_the_number():
    r = gb.report("gen3", {"CODE": 1.7223, "EN": 3.9425, "MATH": 3.7323,
                           "RU": 3.8959, "AVG": 3.3233})
    assert r["universality"] > 0.5
    assert r["n_domains"] == 4
    assert r["avg"] == pytest.approx(3.3233)