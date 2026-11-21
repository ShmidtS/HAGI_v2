"""Integration wires: disjoint splits, WhileLoopIR, neural prior."""
import numpy as np
from hagi.core import AuditLedger
from hagi.while_loop import (discover_while_loop_candidates,
                             synthesize_while_loop)
from hagi.prior import NeuralProposalPrior, profile_features, FAMILIES
from hagi.diagnosis import ResidualProfile


def test_draw_disjoint_splits_no_overlap():
    led = AuditLedger()
    tr, ceg, val = led.draw_disjoint_splits("t", list(range(64)), seed=1)
    allpts = set(tr) | set(ceg) | set(val)
    assert len(allpts) == len(tr) + len(ceg) + len(val)


def test_draw_disjoint_splits_deterministic():
    led = AuditLedger()
    a = led.draw_disjoint_splits("t2", list(range(64)), seed=3)
    b = led.draw_disjoint_splits("t2", list(range(64)), seed=3)
    assert a == b  # cached stream -> same permutation


def test_while_loop_discovers_halve_steps():
    pairs = [(x, x.bit_length()) for x in range(1, 64)]
    cands = discover_while_loop_candidates([p[0] for p in pairs],
                                           [p[1] for p in pairs])
    bp = next((c[1] for c in cands if c[1].step_name == "halve"), None)
    assert bp is not None
    art = synthesize_while_loop(bp, pairs, range(128))
    assert art.eval(100) == 100 .bit_length()


def test_while_loop_noise_yields_nothing():
    rng = np.random.default_rng(5)
    xs = list(range(1, 40))
    ys = [int(v) for v in rng.integers(0, 9, 39)]
    cands = discover_while_loop_candidates(xs, ys)
    # noise may pass the relation test only by accident; require that
    # synthesis on noise never certifies on held-out half
    for _, bp in cands:
        art = synthesize_while_loop(bp, list(zip(xs[:20], ys[:20])),
                                    range(64))
        if art is not None:
            ok = all(art.eval(x) == y for x, y in zip(xs[20:], ys[20:]))
            assert not ok, "noise must not certify"


def test_prior_learns_family_preference():
    pr = NeuralProposalPrior(lr=1.5)
    # periodic profile -> win "mod" 20 times
    prof = ResidualProfile(n=64, k_density=1.0, uniq_outputs=5,
                           sparsity_gaps_entropy=0.5, n_runs=3,
                           autocorr_shift=0.9, autocorr_reflect=0.1,
                           autocorr_xor=0.1, residual_frac_builtin=0.5)
    feat = profile_features(prof)
    for _ in range(20):
        pr.update(feat, "mod")
    h = pr.hints_from_profile(prof)
    assert h["mod"] is True
    assert sum(h.values()) == 1   # exactly one hint (top-1)


def test_reified_ops_as_batched_leaves():
    """Reification wire: invented op composes INSIDE batched synthesis
    (2*popcount(x) solved via reified leaf, fast)."""
    from hagi.legacy_kernel import synthesize_batched_mdl_v8
    pop = lambda x: bin(int(x)).count('1')
    tx = [(x,) for x in range(16)]
    ty = [2 * pop(x) for x in range(16)]
    vx = [(x,) for x in range(16, 24)]
    vy = [2 * pop(x) for x in range(16, 24)]
    expr, n = synthesize_batched_mdl_v8(1, tx, ty, vx, vy,
                                        reified_ops={'popcount': (pop, 12)})
    assert expr is not None
    assert all(expr.eval((x,)) == 2 * pop(x) for x in range(64))
