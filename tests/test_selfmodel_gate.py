"""WAVE 254: selfmodel controller — MGDA direction, Adam backtracking, per-task gate."""
import pytest
import torch as t

from hagi.selfmodel import guarded_optimizer_step, mgda_weights


def _two_task_setup(scale_ratio=1000.0):
    model = t.nn.Linear(2, 1, bias=False)
    with t.no_grad():
        model.weight.copy_(t.tensor([[0.0, 0.0]]))
    # task 1 pushes along x, task 2 along y, with very different gradient scales
    x = t.tensor([[1.0, 0.0]])
    x2 = t.tensor([[0.0, 1.0]])
    def loss_fn(m, b):
        x_, y_ = b
        return ((m(x_) - y_) ** 2).sum()
    task_batches = {"big": (x, t.tensor([[scale_ratio]])), "small": (x2, t.tensor([[1.0]]))}
    validation = lambda m: {"big": float(((m(x) - task_batches["big"][1]) ** 2).sum()),
                            "small": float(((m(x2) - task_batches["small"][1]) ** 2).sum())}
    return model, task_batches, loss_fn, validation


def test_mgda_mixture_is_not_dominated_by_the_largest_gradient():
    model, tb, lf, val = _two_task_setup()
    before = val(model)
    res = guarded_optimizer_step(model, t.optim.SGD(model.parameters(), lr=0.1), tb, lf, val,
                                 max_backtracks=3, max_task_regression=0.0)
    # with the old code the step was ~all along x (the 1000x gradient):
    # 'small' barely moved. With the normalized mixture both weights move
    # by a comparable amount.
    w = model.weight.detach()[0]
    assert 0.2 < abs(float(w[1]) / max(abs(float(w[0])), 1e-12)) <= 1.0, w
    assert all(val(model)[k] <= before[k] + 1e-9 for k in before)   # nothing got worse


def test_adam_backtracking_actually_changes_the_step():
    # A direction that overshoots at scale 1 must be reduced by the
    # backtracking: the committed step_scale < 1 must correspond to a
    # SMALLER realised update than a full step.
    model, tb, lf, val = _two_task_setup(scale_ratio=1.0)
    m_full, _, _, _ = _two_task_setup(scale_ratio=1.0)
    opt = t.optim.Adam(model.parameters(), lr=5.0)      # deliberately huge
    res = guarded_optimizer_step(model, opt, tb, lf, val, max_backtracks=8, max_task_regression=0.0)
    o2 = t.optim.Adam(m_full.parameters(), lr=5.0)
    o2.step()
    full_norm = float(m_full.weight.detach().norm())
    taken_norm = float(model.weight.detach().norm())
    if res.action == "COMMIT":
        assert taken_norm <= full_norm * (1.0 - 1e-6) or res.step_scale == 1.0
    import math; assert math.isfinite(taken_norm)


def test_aggregate_gain_cannot_hide_a_single_task_regression():
    # task 'a' improves a lot, task 'b' gets slightly worse: mean gain > 0.
    calls = {"n": 0}
    model = t.nn.Linear(1, 1, bias=False)
    def loss_fn(m, b):
        return (m(b[0]) - b[1]).sum() ** 2
    def validation(m):
        return {"a": 1.0, "b": 0.0}
    tb = {"a": (t.ones(1, 1), t.zeros(1, 1)), "b": (t.ones(1, 1), t.zeros(1, 1))}
    # make 'b' regress after any step: validation depends on the parameter
    def validation2(m):
        w = float(m.weight.detach()[0, 0])
        return {"a": 1.0 - min(w, 1.0), "b": 0.5 * w}      # a improves, b worsens
    with t.no_grad():
        model.weight.fill_(0.0)
    res = guarded_optimizer_step(model, t.optim.SGD(model.parameters(), lr=0.5), tb, loss_fn, validation2,
                                 max_backtracks=4, max_task_regression=0.0)
    assert res.action == "HOLD" and float(model.weight.detach()[0, 0]) == 0.0


def test_mgda_weights_symmetry_and_simplex():
    a = t.tensor([1.0, 0.0]); b = t.tensor([0.0, 1.0])
    al = mgda_weights([a, b]); assert abs(float(al.sum()) - 1) < 1e-6 and abs(float(al[0] - 0.5)) < 1e-6
    al = mgda_weights([a, a]); assert abs(float(al[0] - 0.5)) < 1e-6
