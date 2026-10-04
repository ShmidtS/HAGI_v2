"""Round-34 growth verdict table: the (G_F, R_repr) decision grid.

Verifies the Python port against the Lean theorem hypotheses
(GrowthGate.lean verdict_ttt / verdict_grow / verdict_exhausted):

- TTT/LORA: G_F >= eps_g AND R <= eps_r      (non-strict on R)
- GROW:     G_F >= eps_g AND eps_r < R       (strict on R)
- STOP:     G_F < eps_g AND R < eps_r
- TEACHER_CHECK: G_F < eps_g AND R >= eps_r

The R-boundary is the one that matters operationally: at R == eps_r
the verdict must be TTT/LORA (the cheaper mechanism wins the tie),
per verdict_ttt's hypothesis ``R ≤ eps``.
"""
from hagi.model.formal import growth_verdict

EPS = 0.01


def test_ttt_cell():
    assert growth_verdict(0.5, 0.001, EPS, EPS) == "TTT/LORA"


def test_grow_cell():
    assert growth_verdict(0.5, 0.5, EPS, EPS) == "GROW"


def test_stop_cell():
    assert growth_verdict(0.001, 0.001, EPS, EPS) == "STOP"


def test_teacher_check_cell():
    assert growth_verdict(0.001, 0.5, EPS, EPS) == "TEACHER_CHECK"


def test_boundary_r_equals_eps_is_ttt():
    # Lean verdict_ttt: R <= eps (non-strict); verdict_grow: eps < R
    # (strict). At the tie the cheaper mechanism (LoRA) wins.
    assert growth_verdict(0.5, EPS, EPS, EPS) == "TTT/LORA"


def test_boundary_g_equals_eps_is_active():
    # Lean verdict_ttt/grow both take eps <= GF (non-strict).
    assert growth_verdict(EPS, 0.001, EPS, EPS) == "TTT/LORA"
    assert growth_verdict(EPS, EPS + 1e-9, EPS, EPS) == "GROW"
