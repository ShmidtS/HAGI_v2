"""V10-3: two-part MDL capability gate."""
from hagi.measurement import (CapabilitySnapshot, capability_delta,
                             total_mdl, heldout_suite, noise_panel)


def snap(mdl, rate, samples, noise=0):
    return CapabilitySnapshot(1, library_meta_bits=29, experts_bits=mdl,
                              unsolved_penalty_bits=0,
                              total_mdl_bits=mdl,
                              heldout_solve_rate=rate,
                              mean_samples_to_solve=samples,
                              noise_false_accepts=noise)


def test_real_gain_requires_mdl_and_transfer():
    before = snap(1000, 0.2, 30.0)
    good = snap(800, 0.5, 20.0)   # MDL down, transfer up, efficient
    assert capability_delta(before, good)["REAL_GAIN"] is True
    md_only = snap(800, 0.2, 30.0)   # compression without transfer
    assert capability_delta(before, md_only)["REAL_GAIN"] is False
    tr_only = snap(1200, 0.5, 20.0)  # transfer at any cost
    assert capability_delta(before, tr_only)["REAL_GAIN"] is False


def test_noise_acceptance_breaks_gain():
    before = snap(1000, 0.2, 30.0)
    noisy = snap(800, 0.5, 20.0, noise=1)
    d = capability_delta(before, noisy)
    assert d["REAL_GAIN"] is False


def test_heldout_panel_counts_and_efficiency():
    tasks = [("a", lambda x: x % 5, range(32), 0),
             ("b", lambda x: (x * 2) % 7, range(32), 0),
             ("c", None, range(32), 0)]  # unsolvable -> HOLD
    def solver(name, dom, target):
        if name == "c":
            return None
        return (42, 8)  # bits, samples
    solved, rate, mean_s = heldout_suite(tasks, solver)
    assert solved == 2 and abs(rate - 2 / 3) < 1e-9 and mean_s == 8.0


def test_noise_panel_zero_for_honest_inventor():
    rng_tables = [{x: (x * 7919) % 2 for x in range(64)}]
    honest = lambda tbl: None      # rejects noise
    sneaky = lambda tbl: object()  # accepts everything
    assert noise_panel(rng_tables, honest) == 0
    assert noise_panel(rng_tables, sneaky) == 1
