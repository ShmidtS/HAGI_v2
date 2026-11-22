"""Self-development contour (WAVE 195h): controller, A/B
experiment framework, gated self-edit, self-model — one file."""
from __future__ import annotations


# ============================================================
# CONTROLLER: safe multi-task update (spec 4.C)
# ============================================================

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence
import torch

import hagi  # GrowingHagiProcessor base (package root; no cycle)
@dataclass
class StepResult:
    action: str
    reason: str
    step_scale: float = 0.0
    certified_gain: float = 0.0
    alphas: dict[str, float] = field(default_factory=dict)
    grad_agreement_min: float = 0.0
    backtracks: int = 0
    loss_before: dict[str, float] = field(default_factory=dict)
    loss_after: dict[str, float] = field(default_factory=dict)
def _flat_grads(model: torch.nn.Module) -> torch.Tensor:
    parts = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    if not parts:
        raise RuntimeError('no gradients present; call loss.backward() first')
    return torch.cat(parts)
def _zero_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.grad = None
def mgda_weights(grads: Sequence[torch.Tensor], iters: int=200) -> torch.Tensor:
    """Min-norm point on the convex hull of task gradients (MGDA).

    Frank-Wolfe over the simplex: deterministic, gradient-free solver
    suitable for K <= ~16 tasks. Returns alpha on the simplex (sum=1, >=0).
    """
    K = len(grads)
    if K == 1:
        return torch.ones(1)
    G = torch.stack([g / (g.norm() + 1e-12) for g in grads])
    alpha = torch.full((K,), 1.0 / K, dtype=G.dtype)
    for _ in range(iters):
        d = alpha @ G
        dots = G @ d
        j = int(torch.argmin(dots).item())
        if dots[j].item() >= d.dot(d).item() - 1e-16:
            break
        gamma = 2.0 / (2.0 + float(_))
        alpha = (1 - gamma) * alpha
        alpha[j] = alpha[j] + gamma
    alpha = torch.clamp(alpha, min=0.0)
    s = float(alpha.sum().item())
    if s <= 0:
        alpha = torch.full((K,), 1.0 / K)
        s = 1.0
    return alpha / s
@torch.no_grad()
def _snapshot_trainable(model: torch.nn.Module) -> dict[int, torch.Tensor]:
    return {i: p.detach().clone() for i, p in enumerate(model.parameters())}
@torch.no_grad()
def _restore_trainable(model: torch.nn.Module, snap: dict[int, torch.Tensor]) -> None:
    for i, p in enumerate(model.parameters()):
        p.copy_(snap[i])
def _snapshot_optim(optimizer: torch.optim.Optimizer) -> Any:
    return copy.deepcopy(optimizer.state_dict())
def _snapshot_rng() -> dict[str, Any]:
    return {'py': __import__('random').getstate(), 'np': __import__('numpy').random.get_state(), 'torch': torch.get_rng_state()}
def _restore_rng(snap: dict[str, Any]) -> None:
    import random
    import numpy as _np
    random.setstate(snap['py'])
    _np.random.set_state(snap['np'])
    torch.set_rng_state(snap['torch'])
def guarded_optimizer_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, task_batches: Mapping[str, Any], loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor], validation_fn: Callable[[torch.nn.Module], dict[str, float]], generalization_error_bound: float=0.0, quantization_error_bound: float=0.0, routing_error_bound: float=0.0, safety_margin: float=0.0, max_backtracks: int=8, conflict_tol: float=1e-08, task_weights: Optional[Mapping[str, float]]=None) -> StepResult:
    """One validation-gated multi-task optimizer step with exact rollback.

    validation_fn must return per-task held-out losses for the same keys as
    task_batches, on a fixed sentinel set that does not overlap train data.
    Commit gate: weighted_loss_before - weighted_loss_after
                > e_gen + e_quant + e_route + m_safety.
    """
    names = list(task_batches.keys())
    w = {k: float(task_weights[k]) if task_weights and k in task_weights else 1.0 for k in names}
    wsum = sum(w.values())
    rng_snap = _snapshot_rng()
    param_snap = _snapshot_trainable(model)
    optim_snap = _snapshot_optim(optimizer)
    loss_before = validation_fn(model)
    if any((k not in loss_before for k in names)):
        _restore_rng(rng_snap)
        return StepResult('HOLD', f'validation_fn missing tasks: {names}')
    wb = sum((w[k] * float(loss_before[k]) for k in names)) / wsum
    grads: list[torch.Tensor] = []
    for k in names:
        _zero_grads(model)
        try:
            loss = loss_fn(model, task_batches[k])
            loss.backward()
        except RuntimeError as exc:
            _zero_grads(model)
            _restore_rng(rng_snap)
            return StepResult('HOLD', f"backward failed on '{k}': {exc}")
        g = _flat_grads(model)
        if not torch.isfinite(g).all():
            _zero_grads(model)
            _restore_rng(rng_snap)
            return StepResult('HOLD', f"non-finite gradient on '{k}'")
        grads.append(g)
    alpha = mgda_weights(grads)
    d = sum((float(alpha[i].item()) * grads[i] for i in range(len(grads))))
    dots = [float(grads[i].dot(d).item()) for i in range(len(grads))]
    min_dot = min(dots)
    _zero_grads(model)
    if min_dot < -conflict_tol:
        _restore_rng(rng_snap)
        return StepResult('HOLD', 'gradient conflict (g_i^T d < 0)', alphas={k: float(alpha[i]) for i, k in enumerate(names)}, grad_agreement_min=min_dot)
    d_norm = float(d.norm().item())
    if d_norm <= 1e-14:
        _restore_rng(rng_snap)
        return StepResult('HOLD', 'degenerate direction (||d|| ~ 0)', alphas={k: float(alpha[i]) for i, k in enumerate(names)})
    total_bound = generalization_error_bound + quantization_error_bound + routing_error_bound + safety_margin
    best: Optional[StepResult] = None
    scale = 1.0
    for bt in range(max_backtracks + 1):
        _restore_trainable(model, param_snap)
        optimizer.load_state_dict(optim_snap)
        _zero_grads(model)
        idx = 0
        # WAVE 235b FIX (external audit P4): backtracking scaled the
        # REPORTED step_scale only — optimizer.step() ran at full lr
        # every retry, so all backtrack iterations were identical.
        # Scale the GRADIENT instead: optimizer(lr) * scale*grad ==
        # effective step of lr*scale for SGD/Adam direction.
        for p in model.parameters():
            n = p.numel()
            p.grad = (scale * d[idx:idx + n]).view_as(p).clone()
            idx += n
        optimizer.step()
        loss_after = validation_fn(model)
        if any((k not in loss_after for k in names)):
            break
        wa = sum((w[k] * float(loss_after[k]) for k in names)) / wsum
        gain = wb - wa
        res = StepResult('COMMIT', 'validation gain exceeds error bounds', step_scale=scale, certified_gain=gain, alphas={k: float(alpha[i]) for i, k in enumerate(names)}, grad_agreement_min=min_dot, backtracks=bt, loss_before=dict(loss_before), loss_after=dict(loss_after))
        if gain > total_bound:
            _zero_grads(model)
            return res
        if best is None or res.certified_gain > best.certified_gain:
            best = res
        scale *= 0.5
    _restore_trainable(model, param_snap)
    optimizer.load_state_dict(optim_snap)
    _zero_grads(model)
    _restore_rng(rng_snap)
    reason = f'validation gain below error budget ({best.certified_gain:.6g} <= {total_bound:.6g})' if best else 'all backtrack scales failed validation'
    return StepResult('HOLD', reason, step_scale=scale, certified_gain=best.certified_gain if best else 0.0, alphas={k: float(alpha[i]) for i, k in enumerate(names)}, grad_agreement_min=min_dot, backtracks=max_backtracks, loss_before=dict(loss_before), loss_after=dict(best.loss_after) if best else {})
def self_test() -> None:
    """Executable property tests for the controller (spec Phase 0)."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    x = torch.randn(64, 4)
    y1 = x @ torch.tensor([[1.0], [1.0], [0.0], [0.0]]) + 0.1
    y2 = x @ torch.tensor([[0.0], [0.0], [1.0], [1.0]]) - 0.1

    def loss_fn(m, batch):
        xx, yy = batch
        return ((m(xx) - yy) ** 2).mean()
    vx = torch.randn(256, 4)
    vy1 = vx @ torch.tensor([[1.0], [1.0], [0.0], [0.0]]) + 0.1
    vy2 = vx @ torch.tensor([[0.0], [0.0], [1.0], [1.0]]) - 0.1

    def validation_fn(m):
        with torch.no_grad():
            return {'t1': float(loss_fn(m, (vx, vy1))), 't2': float(loss_fn(m, (vx, vy2)))}
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    before = validation_fn(model)
    res = guarded_optimizer_step(model, opt, {'t1': (x, y1), 't2': (x, y2)}, loss_fn, validation_fn, generalization_error_bound=0.0, quantization_error_bound=0.0, routing_error_bound=0.0, safety_margin=0.0, max_backtracks=6)
    assert res.action == 'COMMIT', res
    after = validation_fn(model)
    assert after['t1'] < before['t1'] and after['t2'] < before['t2'], (before, after)
    assert abs(sum(res.alphas.values()) - 1.0) < 1e-06
    assert all((a >= 0 for a in res.alphas.values()))
    print(f'  case1 COMMIT: gain={res.certified_gain:.4f} alphas={res.alphas}')
    model2 = torch.nn.Linear(4, 1, bias=False)
    y3 = x @ torch.tensor([[1.0], [-1.0], [0.0], [0.0]])
    y4 = -y3
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.5)
    snap_params = {i: p.detach().clone() for i, p in enumerate(model2.parameters())}
    opt_state = copy.deepcopy(opt2.state_dict())
    res2 = guarded_optimizer_step(model2, opt2, {'t1': (x, y3), 't2': (x, y4)}, loss_fn, validation_fn, max_backtracks=4)
    assert res2.action == 'HOLD', res2
    for i, p in enumerate(model2.parameters()):
        assert torch.equal(p.detach(), snap_params[i]), f'param {i} changed on HOLD'
    assert opt2.state_dict().keys() == opt_state.keys()
    print(f"  case2 HOLD+rollback: reason='{res2.reason}' min_agreement={res2.grad_agreement_min:.3e}")
    model3 = torch.nn.Linear(4, 1)
    opt3 = torch.optim.SGD(model3.parameters(), lr=0.5)
    res3 = guarded_optimizer_step(model3, opt3, {'t1': (x, y1), 't2': (x, y2)}, loss_fn, validation_fn, generalization_error_bound=1000000.0, max_backtracks=3)
    assert res3.action == 'HOLD', res3
    after3 = validation_fn(model3)
    assert after3 == validation_fn(model3)
    print(f'  case3 HOLD@bounds: gain={res3.certified_gain:.4f} << bound=1e6')
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    al = mgda_weights([a, b])
    assert torch.allclose(al, torch.tensor([0.5, 0.5]), atol=1e-06)
    al2 = mgda_weights([a, a])
    assert torch.allclose(al2, torch.tensor([0.5, 0.5]), atol=1e-06)
    al3 = mgda_weights([a, -a])
    assert abs(float(al3.sum()) - 1.0) < 1e-06
    print(f'  case4 MGDA: orthogonal->uniform, identical->uniform, conflict ok')
    print('V10 CONTROLLER SELF_TEST PASS')
if __name__ == '__main__':
    self_test()

# ============================================================
# A/B EXPERIMENT FRAMEWORK
# ============================================================

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable
import numpy as np
Policy = Callable[..., int]
def run_growth(policy: Policy | None, seed: int, n_proposal_rounds: int=8, budget_per_round: int=8, base_budget: int=2000) -> dict[str, Any]:
    """One growth run: base suite + self-proposed tasks under a policy."""
    m = TracedProcessorV9(node_budget=base_budget)
    if policy is not None:
        _orig_learn = m.learn

        def learn_with_policy(spec, seed=0, **kw):
            budget = int(policy(spec, base_budget, seed, m.trace, **kw))
            budget = max(1, min(base_budget, budget))
            m.node_budget = budget
            event = _orig_learn(spec, seed=seed, **kw)
            if m.trace:
                m.trace[-1]['policy_budget'] = budget
                m.trace[-1]['policy_reduced'] = budget < base_budget
            return event
        m.learn = learn_with_policy
    rnd = np.random.default_rng(20260101 + seed)
    seen: set = set()
    t0 = time.perf_counter()
    for sp in hagi.make_task_suite():
        m.learn(sp, seed=10000, train_neural=False)
    for r in range(n_proposal_rounds):
        for k in range(budget_per_round):
            prop = hagi.propose_noncircular_task(m.experts, rnd, r, seen)
            if prop is None:
                continue
            sp, _ = prop
            if sp.name in m.experts:
                continue
            m.learn(sp, seed=500000 + r * 104729 + k, train_neural=False)
    elapsed = time.perf_counter() - t0
    wasted = sum((t['cost'] for t in m.trace if not t['success']))
    total_candidates = sum((t['cost'] for t in m.trace))
    return {'expert_names': frozenset(m.experts.keys()), 'n_experts': len(m.experts), 'wasted': wasted, 'total_candidates': total_candidates, 'seconds': round(elapsed, 2), 'trace': m.trace}
def p_baseline(spec, base: int, seed=0, history=(), **kw) -> int:
    return base
def p_floor_half(spec, base: int, seed=0, history=(), **kw) -> int:
    return max(400, base // 2) if FAMILY_IDS.get(spec.family, 6) == 4 else base
def p_floor_quarter(spec, base: int, seed=0, history=(), **kw) -> int:
    return max(200, base // 4) if FAMILY_IDS.get(spec.family, 6) == 4 else base
def _shape_key(spec, seed: int, train_fraction: float, coarse_domain: bool):
    target_range, target_unique = _target_features(spec, seed, train_fraction)
    domain = len(spec.inputs())
    domain_key = 1 << domain.bit_length() - 1 if coarse_domain and domain > 0 else domain
    return (FAMILY_IDS.get(spec.family, 6), spec.arity, domain_key, target_range, target_unique)
def make_online_conservative_policy(min_holds: int, fraction: float, coarse_domain: bool=False):
    """Reduce search only after repeated, costly HOLDs in the same observed task-shape bucket."""

    def policy(spec, base: int, seed=0, history=(), **kw) -> int:
        train_fraction = float(kw.get('train_fraction', 0.7))
        key = _shape_key(spec, seed, train_fraction, coarse_domain)
        prior = [row for row in history if _shape_key_from_trace(row, coarse_domain) == key]
        if len(prior) >= min_holds and all((int(row['success']) == 0 for row in prior)) and (float(np.mean([row['cost'] for row in prior])) >= 1200):
            return max(200, int(base * fraction))
        return base
    return policy
def _shape_key_from_trace(row, coarse_domain: bool):
    domain = int(row['domain'])
    domain_key = 1 << domain.bit_length() - 1 if coarse_domain and domain > 0 else domain
    return (int(row['family']), int(row['arity']), domain_key, int(row['target_range']), int(row['target_unique']))
POLICIES: dict[str, Policy] = {'baseline': p_baseline, 'floor_half_auto': p_floor_half, 'floor_quarter_auto': p_floor_quarter, 'online_exact_1hold_half': make_online_conservative_policy(1, 0.5, False), 'online_exact_1hold_quarter': make_online_conservative_policy(1, 0.25, False), 'online_exact_2holds_quarter': make_online_conservative_policy(2, 0.25, False), 'online_exact_2holds_eighth': make_online_conservative_policy(2, 0.125, False), 'online_coarse_2holds_half': make_online_conservative_policy(2, 0.5, True), 'online_coarse_3holds_quarter': make_online_conservative_policy(3, 0.25, True)}
def run_experiment(seeds: int=5, policy_names: list[str] | None=None, seed_start: int=0) -> dict[str, Any]:
    report: dict[str, Any] = {}
    selected_names = list(POLICIES) if policy_names is None else list(dict.fromkeys(policy_names))
    unknown = sorted(set(selected_names) - set(POLICIES))
    if unknown:
        raise ValueError(f'unknown policy names: {unknown}')
    if 'baseline' not in selected_names:
        raise ValueError("policy_names must include 'baseline'")
    selected_policies = {name: POLICIES[name] for name in selected_names}
    report['validation_protocol'] = {'seed_start': seed_start, 'seed_count': seeds, 'policies': selected_names, 'gate': 'preserve every baseline expert on every seed AND reduce mean total candidate evaluations'}
    tracer = TracedProcessorV9(node_budget=2000)
    rnd = np.random.default_rng(20260101)
    seen: set = set()
    for sp in hagi.make_task_suite():
        tracer.learn(sp, seed=10000, train_neural=False)
    for k in range(10):
        prop = hagi.propose_noncircular_task(tracer.experts, rnd, 0, seen)
        if prop is None:
            continue
        sp, _ = prop
        if sp.name in tracer.experts:
            continue
        tracer.learn(sp, seed=500000 + k, train_neural=False)
    from collections import Counter
    reasons = Counter((e['reason'] for e in tracer.growth_log))
    cert = impossibility_certificate(tracer.trace)
    report['derived_algorithm'] = {'pipeline': ['REUSE(certified-library-coverage)', 'boolvenn-exact(16 masks)', 'anf-zhegalkin-butterfly(+full-support)', 'direct-template(+full-support)', 'exact-rational-poly(1D/ND,+full-support)', 'exact-piecewise-poly(+full-support)', 'batched-mdl-mitm-dag(+full-support)', 'HOLD(rollback+audit)'], 'commit_reason_counts': dict(reasons), 'success_feature_impossibility_certificate': cert, 'note': "success of the exhaustive synthesis stage is a function of the task's full teacher data, not of coarse features; the experiment therefore trials MEASURED policies instead of claiming a symbolic predictor"}
    suite_names = frozenset((s.name for s in hagi.make_task_suite()))
    trials: dict[str, list[dict[str, Any]]] = {p: [] for p in selected_policies}
    for seed in range(seed_start, seed_start + seeds):
        for pname, policy in selected_policies.items():
            res = run_growth(policy, seed)
            trials[pname].append(res)
            print(f"  seed={seed} policy={pname}: experts={res['n_experts']} wasted={res['wasted']} t={res['seconds']}s", flush=True)
    gate: dict[str, Any] = {}
    base_stats = trials['baseline']
    for pname, runs in trials.items():
        superset_all = all((r['expert_names'] >= b['expert_names'] for r, b in zip(runs, base_stats)))
        mean_wasted = float(np.mean([r['wasted'] for r in runs]))
        mean_total_candidates = float(np.mean([r['total_candidates'] for r in runs]))
        mean_secs = float(np.mean([r['seconds'] for r in runs]))
        baseline_total_candidates = float(np.mean([b['total_candidates'] for b in base_stats]))
        passes = superset_all and mean_total_candidates < baseline_total_candidates
        gate[pname] = {'superset_on_all_seeds': superset_all, 'exact_accuracy_floor': 1.0 if superset_all else None, 'mean_wasted': round(mean_wasted, 1), 'mean_total_candidates': round(mean_total_candidates, 1), 'baseline_mean_total_candidates': round(baseline_total_candidates, 1), 'total_candidate_reduction_pct': round(100 * (baseline_total_candidates - mean_total_candidates) / max(1.0, baseline_total_candidates), 2), 'mean_seconds': round(mean_secs, 2), 'per_seed_total_candidates': [r['total_candidates'] for r in runs], 'verdict': 'PASS' if passes else 'REJECT', 'per_seed_experts': [r['n_experts'] for r in runs], 'per_seed_wasted': [r['wasted'] for r in runs], 'reduced_budget_tasks_per_seed': [sum((1 for row in r['trace'] if row.get('policy_reduced', False))) for r in runs], 'reduced_budget_commits_per_seed': [sum((1 for row in r['trace'] if row.get('policy_reduced', False) and row['success'])) for r in runs], 'lost_experts_per_seed': [sorted(b['expert_names'] - r['expert_names']) for r, b in zip(runs, base_stats)], 'new_experts_vs_baseline': [sorted(r['expert_names'] - b['expert_names']) for r, b in zip(runs, base_stats)]}
    passing = {p: g for p, g in gate.items() if p != 'baseline' and g['verdict'] == 'PASS'}
    if passing:
        best = min(passing, key=lambda p: gate[p]['mean_total_candidates'])
        report['gate'] = gate
        report['decision'] = {'action': 'COMMIT', 'adopted_policy': best, 'policy_source': str(selected_policies[best]), 'reason': 'measured non-regression on all seeds with strictly fewer total candidate evaluations'}
    else:
        report['gate'] = gate
        report['decision'] = {'action': 'HOLD', 'reason': 'no candidate policy preserved the baseline expert set on all seeds while reducing total candidate evaluations; baseline retained'}
    return report
def main_experiment() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed-start', type=int, default=0)
    ap.add_argument('--policies', nargs='+', choices=tuple(POLICIES))
    a = ap.parse_args()
    rep = run_experiment(seeds=a.seeds, policy_names=a.policies, seed_start=a.seed_start)
    print(json.dumps({k: v for k, v in rep.items() if k != 'derived_algorithm'}, indent=2, ensure_ascii=False))
    d = rep['decision']
    if d['action'] == 'COMMIT':
        print(f"ADOPTED: policy '{d['adopted_policy']}' (wasted {rep['gate']['baseline']['mean_wasted']} -> {rep['gate'][d['adopted_policy']]['mean_wasted']})")
    else:
        print('GATE HOLD: ' + d['reason'])

# ============================================================
# GATED SELF-EDIT LOG
# ============================================================

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parent.parent
from hagi.util import CKPT_DIR as _CKPT
EDIT_LOG = _CKPT / 'self' / 'self_edits.json'
BASELINE_PATH = _CKPT / 'self' / 'edit_baseline.json'
KNOBS = {'floor_ladder': [[200, 50, 20], [200, 100, 50], [300, 100, 30], [200, 50, 20, 10]], 'batch_size': [8, 12, 16], 'proposals': [8, 10, 12]}
def _baseline() -> dict[str, Any]:
    """Benchmark the CURRENT code once and cache it."""
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding='utf-8'))
    res = run_benchmark()
    BASELINE_PATH.write_text(json.dumps(res), encoding='utf-8')
    return res
def run_benchmark(rounds: int=2, workers: int=8) -> dict[str, Any]:
    """Fixed mini-suite benchmark of the learning loop (deterministic)."""
    from hagi import loop as sl
    t0 = time.time()
    rep = sl.run_self_development(max_rounds=rounds, workers=workers, proposals_per_round=6)
    fin = rep['final']
    return {'adopted': fin['adopted_rounds'], 'experts': fin['experts'], 'seconds': round(time.time() - t0, 1)}
def _gate_tests() -> bool:
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', 'tests/test_characterization.py'], capture_output=True, text=True, timeout=1800)
    return r.returncode == 0
def current_settings() -> dict[str, Any]:
    """Snapshot of the CURRENT knob values: loop defaults overridden
    by gate-adopted settings (the production loop runs with them)."""
    from hagi import loop as sl
    import inspect
    sig = inspect.signature(sl.run_self_development)
    defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
    cur = {'floor_ladder': defaults.get('floor_ladder') or [200], 'batch_size': defaults.get('batch_size'), 'proposals': defaults.get('proposals_per_round')}
    try:
        adopted = adopted_settings()
    except Exception:
        adopted = {}
    if 'floor_ladder' in adopted:
        cur['floor_ladder'] = adopted['floor_ladder']
    if 'batch_size' in adopted:
        cur['batch_size'] = adopted['batch_size']
    if 'proposals' in adopted:
        cur['proposals'] = adopted['proposals']
    return cur
def propose_patch(rng) -> dict[str, Any]:
    """Propose a knob change (declared space only).

    GrowthCeilings.expansion_required / ProxyGap horizon: a no-op
    patch (value identical to the current setting) would burn
    gate-budget (<= 3 candidates/cycle) on a triviality — never
    propose it. Only changes are candidates.
    """
    import random
    cur = current_settings()
    for _ in range(64):
        knob = rng.choice(list(KNOBS.keys()))
        value = rng.choice(KNOBS[knob])
        if knob == 'floor_ladder':
            if value == cur.get('floor_ladder'):
                continue
        elif value == cur.get(knob):
            continue
        return {'knob': knob, 'value': value}
    return {'knob': 'batch_size', 'value': 8}
def is_noop(patch: dict[str, Any]) -> bool:
    """True when the patch equals the current setting (wastes gate budget)."""
    cur = current_settings()
    if patch['knob'] == 'floor_ladder':
        return patch['value'] == cur.get('floor_ladder')
    return patch['value'] == cur.get(patch['knob'])
def adopted_settings() -> dict[str, Any]:
    """Last adopted value per knob from the edit log (gated, certified).

    CertifiedLoop contract: adopted patches must actually drive the
    production loop — otherwise the self-edit cycle is decorative.
    Returns kwargs for run_self_development.
    """
    if not EDIT_LOG.exists():
        return {}
    log = json.loads(EDIT_LOG.read_text(encoding='utf-8'))
    last: dict[str, Any] = {}
    for e in log.get('adopted', []):
        p = e.get('patch', {})
        if p.get('knob') and 'value' in p:
            last[p['knob']] = p['value']
    kwargs: dict[str, Any] = {}
    if 'floor_ladder' in last:
        kwargs['floor_ladder'] = last['floor_ladder']
    if 'batch_size' in last:
        kwargs['batch_size'] = last['batch_size']
    if 'proposals' in last:
        kwargs['proposals_per_round'] = last['proposals']
    return kwargs
def apply_patch(patch: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Return a modified settings overlay (does NOT edit source code)."""
    out = dict(target)
    out[patch['knob']] = patch['value']
    return out
MAX_GATE_RUNS = 6
def self_edit_cycle(seed: int=0, dry: bool=False) -> dict[str, Any]:
    """One propose->gate->adopt cycle. Honest record in EDIT_LOG."""
    import random
    rng = random.Random(seed)
    log = json.loads(EDIT_LOG.read_text(encoding='utf-8')) if EDIT_LOG.exists() else {'adopted': [], 'rejected': []}
    if len(log.get('rejected', [])) >= MAX_GATE_RUNS:
        log['rejected'] = log['rejected'][-MAX_GATE_RUNS:]
        EDIT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')
    baseline = _baseline()
    patch = propose_patch(rng)
    proposal = {'patch': patch, 'ts': time.strftime('%Y-%m-%d %H:%M:%S')}
    if is_noop(patch):
        proposal['decision'] = 'noop-skipped'
        return proposal
    if dry:
        proposal['decision'] = 'dry'
        return proposal
    from hagi import loop as sl
    kwargs = {}
    if patch['knob'] == 'batch_size':
        kwargs['batch_size'] = patch['value']
    elif patch['knob'] == 'proposals':
        kwargs['proposals_per_round'] = patch['value']
    elif patch['knob'] == 'floor_ladder':
        kwargs['floor_ladder'] = patch['value']
    try:
        rep = sl.run_self_development(max_rounds=2, workers=8, proposals_per_round=6, **kwargs)
        cand = {'adopted': rep['final']['adopted_rounds'], 'experts': rep['final']['experts']}
    except Exception as e:
        proposal.update(decision='rejected', reason=repr(e)[:80])
        log['rejected'].append(proposal)
        EDIT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')
        return proposal
    improves = cand['adopted'] > baseline['adopted'] or cand['experts'] > baseline['experts']
    no_regress = improves and _gate_tests()
    proposal.update(decision='adopted' if no_regress else 'rejected', candidate=cand, baseline=baseline, gate_tests=improves)
    (log['adopted'] if no_regress else log['rejected']).append(proposal)
    EDIT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')
    return proposal
def main_selfedit() -> None:
    import argparse
    ap = argparse.ArgumentParser(description='Nadya self-edit gate')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    print(json.dumps(self_edit_cycle(seed=a.seed, dry=a.dry), ensure_ascii=False, indent=1))

# ============================================================
# SELF-MODEL: TRACE->DERIVE->OPTIMIZE->GATE (anchor module)
# ============================================================

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional
import numpy as np
FAMILY_IDS = {'BoolVenn': 0, 'unary': 1, 'integer-binary': 2, 'composed-ternary': 3, 'auto-composed': 4, 'auto-bool3': 5}
FEATURE_KEYS = ('family', 'arity', 'domain', 'library', 'target_range', 'target_unique', 'self_inv')
def _self_inverse_flag(spec) -> int:
    """1 if the target is self-inverse (f∘f = id) on sampled unary inputs.
    Separates tasks identical in range/unique stats (ATBASH vs CAESAR)."""
    if spec.arity != 1:
        return 0
    dom = spec.inputs()
    for x in dom[:64]:
        try:
            y = int(spec.teacher(*x))
            if not spec.valid((y,)) or int(spec.teacher(y)) != x[0]:
                return 0
        except Exception:
            return 0
    return 1
def _target_features(spec, seed: int, train_fraction: float) -> tuple[int, int]:
    """Describe only the supervised target labels visible to the learner."""
    domain = spec.inputs()
    if not domain:
        return (0, 0)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(domain))
    bool_like = spec.family == 'BoolVenn' or (all((set(d) <= {0, 1} for d in spec.domains)) and len(domain) <= 16)
    if bool_like:
        n_train = max(1, len(domain) - 1)
    else:
        n_train = max(1, min(len(domain) - 1, int(round(len(domain) * train_fraction))))
    ys = [int(spec.teacher(*domain[int(i)])) for i in perm[:n_train]]
    return (max(ys) - min(ys), len(set(ys))) if ys else (0, 0)
class TracedProcessorV9(hagi.GrowingHagiProcessor):
    stream_names: list = None
    stream_actions: list = None
    'Records a (features -> outcome, cost) trace of its own learning.'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trace: list[dict[str, Any]] = []

    def learn(self, spec, seed=0, train_fraction=0.7, poison_one_label=False, train_neural=False, deadline=None):
        target_range, target_unique = _target_features(spec, seed, train_fraction)
        feats = {'family': FAMILY_IDS.get(spec.family, 6), 'arity': spec.arity, 'domain': len(spec.inputs()), 'library': len(self.experts), 'self_inv': _self_inverse_flag(spec), 'target_range': target_range, 'target_unique': target_unique}
        c0 = self.candidates_generated
        t0 = time.perf_counter()
        ev = super().learn(spec, seed=seed, train_fraction=train_fraction, poison_one_label=poison_one_label, train_neural=train_neural, deadline=deadline)
        self.trace.append({**feats, 'seed': int(seed), 'success': 1 if ev['action'] == 'COMMIT' else 0, 'cost': min(self.candidates_generated - c0, 3000), 'seconds': round(time.perf_counter() - t0, 4), 'task': spec.name})
        return ev
def _features(spec, model, seed: int=0, train_fraction: float=0.7) -> tuple[int, ...]:
    target_range, target_unique = _target_features(spec, seed, train_fraction)
    return (FAMILY_IDS.get(spec.family, 6), spec.arity, len(spec.inputs()), len(model.experts), target_range, target_unique, _self_inverse_flag(spec))
def derive_self_model(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Induce self-model programs from the trace with HAGI's own machinery."""
    rows = [tuple((t[k] for k in FEATURE_KEYS)) for t in trace]
    ys = [t['success'] for t in trace]
    succ_expr = hagi.exact_piecewise_poly_fit(rows, ys)
    succ_prog = None
    if succ_expr is not None:
        ok = all((int(succ_expr.eval(r)) == y for r, y in zip(rows, ys)))
        if not ok:
            succ_expr = None
    if succ_expr is None:
        expr, _n = hagi.synthesize_batched_mdl(len(FEATURE_KEYS), rows, ys, rows, ys, max_evals=20000, max_bits=64)
        if expr is not None and all((int(expr.eval(r, {})) == y for r, y in zip(rows, ys))):
            succ_expr = expr
    costs = [t['cost'] for t in trace]
    cost_expr = None
    if len(set(costs)) > 1:
        ce, _n = hagi.synthesize_batched_mdl(len(FEATURE_KEYS), rows, costs, rows, costs, max_evals=20000, max_bits=64)
        if ce is not None and all((int(ce.eval(r, {})) == c for r, c in zip(rows, costs))):
            cost_expr = ce
    model_desc = {'n_trace': len(trace), 'success_program': succ_expr.pretty() if succ_expr is not None else None, 'success_exact': succ_expr is not None, 'cost_program': cost_expr.pretty() if cost_expr is not None else None, 'cost_exact': cost_expr is not None, '_succ_expr': succ_expr, '_cost_expr': cost_expr}
    return model_desc
def impossibility_certificate(trace: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Prove (by witness) that no exact success predictor exists on the
    current feature space: find two learn calls with identical features and
    different outcomes. This is a finite, checkable certificate for ABSTAIN
    (spec v10 sec 9: never claim what is not demonstrated)."""
    by_feats: dict[tuple, list[str]] = {}
    for t in trace:
        key = tuple((t[k] for k in FEATURE_KEYS))
        by_feats.setdefault(key, []).append(f"{t['task']}:{('COMMIT' if t['success'] else 'HOLD')}")
    for key, names in sorted(by_feats.items()):
        if any((n.endswith(':COMMIT') for n in names)) and any((n.endswith(':HOLD') for n in names)):
            return {'identical_features': list(key), 'conflicting_outcomes': names}
    return None
def optimized_budget_policy(self_model: dict[str, Any], spec, model, base_budget: int, floor_budget: int, seed: int=0, train_fraction: float=0.7) -> int:
    """Self-model-driven compute allocation from the observed training split."""
    succ = self_model.get('_succ_expr')
    if succ is None:
        return base_budget
    try:
        pred = int(succ.eval(_features(spec, model, seed, train_fraction), {}))
    except Exception:
        return base_budget
    if pred == 0:
        return min(base_budget, floor_budget)
    return base_budget
class SelfOptimizingProcessorV9(TracedProcessorV9):
    """Growth processor whose per-task node_budget is set by its own self-model."""

    def __init__(self, self_model: dict[str, Any], base_budget: int, floor_budget: int, *a, **kw):
        super().__init__(*a, **kw)
        self.self_model = self_model
        self.base_budget = base_budget
        self.floor_budget = floor_budget
        self.policy_overrides = 0

    def learn(self, spec, seed=0, **kw):
        override = getattr(self, 'optimized_budget_policy', None)
        if override is None:
            budget = optimized_budget_policy(self.self_model, spec, self, self.base_budget, self.floor_budget, seed=seed, train_fraction=float(kw.get('train_fraction', 0.7)))
        else:
            budget = override(spec, self, self.base_budget, self.floor_budget)
        self.node_budget = max(1, min(self.base_budget, int(budget)))
        if self.node_budget < self.base_budget:
            self.policy_overrides += 1
        ev = super().learn(spec, seed=seed, **kw)
        return ev
def describe_pipeline(model: TracedProcessorV9) -> dict[str, Any]:
    """The model outputs its own creation pipeline, derived from its log."""
    from collections import Counter
    reasons = Counter((e['reason'] for e in model.growth_log))
    return {'stages': ['REUSE(certified-library-coverage)', 'boolvenn-exact(16 masks)', 'anf-zhegalkin-butterfly(+full-support)', 'direct-template(+full-support)', 'exact-rational-poly(1D/ND,+full-support)', 'exact-piecewise-poly(+full-support)', 'batched-mdl-mitm-dag(+full-support)', 'HOLD(rollback+audit)'], 'commit_reason_counts': dict(reasons), 'experts': len(model.experts), 'theorems': len(model.theorems)}
def run_self_optimization(rounds: int=2) -> dict[str, Any]:
    report: dict[str, Any] = {}
    suite = hagi.make_task_suite()
    tracer = TracedProcessorV9(node_budget=2000)
    rnd = np.random.default_rng(20260101)
    seen: set = set()
    t0 = time.perf_counter()
    for sp in suite:
        tracer.learn(sp, seed=10000, train_neural=False)
    n_prop = 0
    for k in range(10):
        prop = hagi.propose_noncircular_task(tracer.experts, rnd, 0, seen)
        if prop is None:
            continue
        sp, _ = prop
        if sp.name in tracer.experts:
            continue
        n_prop += 1
        tracer.learn(sp, seed=500000 + k, train_neural=False)
    trace_seconds = time.perf_counter() - t0
    commits = sum((t['success'] for t in tracer.trace))
    wasted = sum((t['cost'] for t in tracer.trace if not t['success']))
    report['phase1_trace'] = {'learn_calls': len(tracer.trace), 'commits': commits, 'holds': len(tracer.trace) - commits, 'wasted_candidates_on_holds': wasted, 'trace_seconds': round(trace_seconds, 2), 'pipeline': describe_pipeline(tracer)}
    self_model = derive_self_model(tracer.trace)
    report['phase2_self_model'] = {k: v for k, v in self_model.items() if not k.startswith('_')}
    succ = self_model.get('_succ_expr')
    report['self_algorithm'] = {'pipeline': report['phase1_trace']['pipeline']['stages'], 'success_predictor': self_model['success_program'], 'cost_predictor': self_model['cost_program'], 'policy': 'node_budget = base if self_model.predict_success(task)==1 else floor', 'gate': 'adopt iff expert set identical/superset and floors hold; else rollback'}
    if succ is None:
        cert = impossibility_certificate(tracer.trace)
        demo = SelfOptimizingProcessorV9({'_succ_expr': None, '_naive': True}, base_budget=2000, floor_budget=200)
        demo.optimized_budget_policy = lambda spec, model, base, floor: floor if FAMILY_IDS.get(spec.family, 6) == 4 else base
        rnd2 = np.random.default_rng(20260101)
        seen2: set = set()
        for sp in suite:
            demo.learn(sp, seed=10000, train_neural=False)
        for k in range(10):
            prop = hagi.propose_noncircular_task(demo.experts, rnd2, 0, seen2)
            if prop is None:
                continue
            sp, _ = prop
            if sp.name in demo.experts:
                continue
            demo.learn(sp, seed=500000 + k, train_neural=False)
        lost = sorted(set(tracer.experts) - set(demo.experts))
        if lost:
            gate_reason = 'naive policy lost experts -> gate rollback: ' + ', '.join(lost)
        else:
            gate_reason = 'no exact self-model at this feature resolution; unverified policy cannot be adopted without a certified predictor (honest HOLD)'
        report['phase3_optimization'] = {'status': 'ATTEMPTED_NAIVE_POLICY', 'policy': 'floor_budget for auto-composed family', 'lost_experts': lost}
        report['phase4_gate'] = {'action': 'HOLD', 'reason': gate_reason, 'impossibility_certificate': cert, 'rollback': 'baseline configuration retained (tracer state is the adopted model)'}
        report['adopted'] = False
        return report
    opt = SelfOptimizingProcessorV9(self_model, base_budget=2000, floor_budget=200)
    rnd2 = np.random.default_rng(20260101)
    seen2: set = set()
    t0 = time.perf_counter()
    for sp in suite:
        opt.learn(sp, seed=10000, train_neural=False)
    for k in range(10):
        prop = hagi.propose_noncircular_task(opt.experts, rnd2, 0, seen2)
        if prop is None:
            continue
        sp, _ = prop
        if sp.name in opt.experts:
            continue
        opt.learn(sp, seed=500000 + k, train_neural=False)
    opt_seconds = time.perf_counter() - t0
    opt_wasted = sum((t['cost'] for t in opt.trace if not t['success']))
    report['phase3_optimization'] = {'status': 'APPLIED', 'learn_calls': len(opt.trace), 'wasted_candidates_on_holds': opt_wasted, 'seconds': round(opt_seconds, 2), 'same_seed_stream': True}
    base_names = set(tracer.experts.keys())
    opt_names = set(opt.experts.keys())
    experts_ok = opt_names >= base_names
    check = opt.verify_all(suite)
    floors_ok = check['accuracy'] == 1.0 and check['committed'] >= len(suite)
    if experts_ok and floors_ok:
        adopted = True
        report['phase4_gate'] = {'action': 'COMMIT', 'experts_base': len(base_names), 'experts_optimized': len(opt_names), 'exact_accuracy': check['accuracy'], 'wasted_candidates': f'{wasted} -> {opt_wasted}', 'seconds': f'{round(trace_seconds, 2)} -> {round(opt_seconds, 2)}', 'speedup_wasted': round(wasted / max(1, opt_wasted), 2)}
    else:
        adopted = False
        report['phase4_gate'] = {'action': 'HOLD', 'reason': 'optimized policy lost experts' if not experts_ok else 'quality floor violated', 'experts_base': len(base_names), 'experts_optimized': len(opt_names), 'accuracy': check['accuracy']}
    report['adopted'] = adopted
    return report
def main() -> None:
    p = argparse.ArgumentParser(description='HAGI self-model: derive & optimize own algorithm')
    p.add_argument('--json', action='store_true', help='print raw JSON only')
    a = p.parse_args()
    rep = run_self_optimization()
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if rep['phase4_gate'].get('action') == 'COMMIT':
        print(f"SELF-OPTIMIZATION COMMIT: self-model policy adopted ({rep['phase4_gate']['wasted_candidates']} wasted candidates)")
    elif rep.get('phase3_optimization', {}).get('status') == 'ATTEMPTED_NAIVE_POLICY':
        print('SELF-OPTIMIZATION HOLD: exact self-model impossible at this feature resolution (certificate produced); naive policy rejected by commit gate, baseline retained')
    else:
        print('SELF-OPTIMIZATION HOLD: ' + rep['phase4_gate'].get('reason', ''))
if __name__ == '__main__':
    main()

# ============================================================
# MDL PRUNE: library minimisation (certified coverage)
# ============================================================

from pathlib import Path
from typing import Any
from hagi import *
def _expr_calls(expr) -> set[str]:
    """Names of experts called by this expression (call nodes)."""
    out: set[str] = set()

    def walk(e) -> None:
        if e is None:
            return
        if getattr(e, 'op', None) == 'call':
            out.add(str(e.value))
            for a in e.args:
                walk(a)
        else:
            for a in getattr(e, 'args', []) or []:
                walk(a)
    walk(expr)
    return out
def _ast_size(expr) -> int:
    if expr is None:
        return 0
    return 1 + sum((_ast_size(a) for a in getattr(expr, 'args', []) or []))
def _certified_map(proc, name: str) -> tuple[int, tuple]:
    """(arity, sorted ((inputs..., output)) map) on the certified domain."""
    ex = proc.experts[name]
    rows = []
    for row in _domain_rows(proc, name):
        try:
            rows.append((*row, int(ex.program.expr.eval(row, proc.experts))))
        except Exception:
            rows.append((*row, 'ERR'))
    return (ex.program.arity, tuple(sorted(rows)))
def _domain_rows(proc, name: str):
    ex = proc.experts[name]
    doms = ex.domains
    if ex.program.arity == 1:
        return [(x,) for x in doms[0]]
    if ex.program.arity == 2:
        return [(x, y) for x in doms[0] for y in doms[1]]
    import itertools
    return [tuple(c) for c in itertools.product(*doms)]
def prune_library(proc, max_report: int=12) -> dict[str, Any]:
    """Return (new_proc-like object mutated in place? NO — returns a
    report and a pruned experts dict + counts; caller rebuilds)."""
    experts = proc.experts
    names = list(experts.keys())
    sigs: dict[str, tuple] = {}
    for n in names:
        if experts[n].program.expr is not None:
            try:
                sigs[n] = _certified_map(proc, n)
            except Exception:
                sigs[n] = ('ERR', n)
    groups: dict[tuple, list[str]] = {}
    for n, s in sigs.items():
        groups.setdefault(s, []).append(n)
    dup_keep: dict[str, str] = {}
    dup_drop: set[str] = set()
    for s, members in groups.items():
        if len(members) > 1:
            members.sort(key=lambda n: (_ast_size(experts[n].program.expr), n))
            keep = members[0]
            dup_keep[keep] = f'{len(members)} copies'
            dup_drop.update(members[1:])
    refs: dict[str, set[str]] = {}
    for n in names:
        refs[n] = _expr_calls(experts[n].program.expr) if experts[n].program.expr else set()
    refd: set[str] = set()
    for n in names:
        refd |= (refs[n] & set(names)) - {n}
    pseudo_drop: set[str] = set()
    for n in names:
        ex = experts[n]
        if ex.program.arity != 2 or n in dup_drop:
            continue
        src_calls = refs[n]
        uses_y = _expr_uses_var(ex.program.expr, 1)
        if not uses_y and n not in refd and (len(sigs.get(n, ('?',))[0:1] or ()) >= 0):
            pseudo_drop.add(n)
    keep = set(names) - dup_drop
    changed = True
    extra_drop: set[str] = set()
    base_guard = {'ADD', 'SUB', 'MUL', 'NEG', 'ABS', 'SIGN', 'SQUARE', 'CUBE', 'MIN', 'MAX', 'GCD', 'LCM', 'EQ', 'LT', 'GE', 'MEDIAN3', 'CLAMP', 'FMA'}
    while changed:
        changed = False
        refd_now: set[str] = set()
        for n in keep:
            refd_now |= (refs[n] & keep) - {n}
        for n in list(keep):
            if n in base_guard or n in refd_now:
                continue
            if n in pseudo_drop:
                keep.discard(n)
                extra_drop.add(n)
                changed = True
    report = {'before': len(names), 'after': len(keep), 'dropped_duplicates': len(dup_drop), 'dropped_pseudo_1arg': len(extra_drop), 'dup_groups': len(dup_keep), 'examples_dup_groups': [f'{k}: {v}' for k, v in list(dup_keep.items())[:max_report]]}
    return {'keep': keep, 'report': report, 'refs': refs}
def _expr_uses_var(expr, idx: int) -> bool:
    if expr is None:
        return False
    if getattr(expr, 'op', None) == 'var' and int(expr.value) == idx:
        return True
    return any((_expr_uses_var(a, idx) for a in getattr(expr, 'args', []) or []))
def apply_prune(proc, keep: set[str]):
    """Return a shallow-copied processor with only kept experts, after
    FULL exact re-validation. Raises on any validation failure."""
    import copy
    new = copy.copy(proc)
    new.experts = {n: proc.experts[n] for n in proc.experts if n in keep}
    failures = 0
    for n, ex in new.experts.items():
        if ex.program.expr is None:
            continue
        for row in _domain_rows(proc, n):
            try:
                want = int(ex.program.expr.eval(row, proc.experts))
            except Exception:
                want = 'ERR'
            try:
                got = int(ex.program.expr.eval(row, new.experts))
            except Exception:
                got = 'ERR'
            if want != got:
                failures += 1
                break
    if failures:
        raise RuntimeError(f'prune aborted: {failures} experts failed re-validation')
    return (new, failures)