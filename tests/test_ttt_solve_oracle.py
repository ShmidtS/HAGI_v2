"""Tests for the closed-form-solve oracle (``scripts/ttt_solve_oracle.py``).

Two jobs. First, the committed fixture must be *self-consistent*: the arrays satisfy the normal
equations the manifest documents, the guard verdicts recompute, and ``A`` is the orthonormal
draw it claims to be — otherwise the C++ solver would be proven against a fiction. Second, the
manifest's constants must be the *shipped* ones, read off ``TttRls.__init__``'s signature
rather than retyped, so a change to ``src/hagi/train/ttt.py`` breaks here instead of leaving a
stale oracle behind.

The regeneration test compares with ``allclose``, not bitwise: ``torch.linalg.qr``/``solve`` run
through LAPACK/BLAS, whose last bits differ across builds. The verdicts and scalars are
compared exactly, because those are what the C++ side must reproduce.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ttt_solve_oracle as O  # noqa: E402

from hagi.train.ttt import TttRls  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures"
MANIFEST = FIXTURES / "ttt_solve_oracle.json"
PAYLOAD = FIXTURES / "ttt_solve_oracle.npz"


@pytest.fixture(scope="session")
def manifest() -> dict:
    assert MANIFEST.exists(), f"missing committed fixture: {MANIFEST} (run scripts/ttt_solve_oracle.py)"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def arrays() -> dict[str, np.ndarray]:
    with np.load(PAYLOAD) as z:
        return {k: np.array(z[k]) for k in z.files}


def test_payload_matches_manifest(manifest, arrays):
    assert set(arrays) == set(manifest["arrays"]), "payload and manifest disagree on array names"
    for name, spec in manifest["arrays"].items():
        a = arrays[name]
        assert list(a.shape) == spec["shape"], f"{name} shape {a.shape} != {spec['shape']}"
        assert a.dtype == np.dtype(spec["dtype"]) == np.float32, f"{name} dtype {a.dtype}"
        digest = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        assert digest == spec["sha256"], f"{name} bytes differ from the manifest (stale payload?)"


def test_shapes_are_the_documented_geometry(manifest, arrays):
    c = manifest["constants"]
    assert arrays["A"].shape == (c["hidden"], c["rank"]) == (5120, 8)
    assert arrays["X"].shape == (c["t_rows"], c["hidden"]) == (64, 5120)
    for name in ("Y_default", "Y_cap_bound"):
        assert arrays[name].shape == (64, 5120)
    for case in ("default", "cap_bound"):
        assert arrays[f"B_{case}"].shape == (5120, 8)
        assert arrays[f"G_{case}"].shape == (8, 8)
        assert arrays[f"C_{case}"].shape == (8, 5120)


def test_A_is_the_orthonormal_draw(manifest, arrays):
    """``A`` is a frozen orthonormal buffer (adapters.py:212-218), so ``A.T @ A == I``."""
    A = arrays["A"]
    gram = A.T @ A
    assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-5), gram
    assert np.allclose(np.linalg.norm(A, axis=0), 1.0, atol=1e-5)
    # Regenerating the draw from the documented seed reproduces the fixture's A.
    fresh = O._lora(manifest["seeds"]["A_init_seed"]).lora_A.detach().numpy()
    assert np.allclose(fresh, A, rtol=1e-5, atol=1e-7)


def test_phi_is_X_projected_on_A(manifest, arrays):
    """``phi = feat @ lora_A`` (ttt.py:345) — the C++ side's first step."""
    assert np.allclose(arrays["phi"], arrays["X"] @ arrays["A"], rtol=1e-4, atol=1e-6)


def test_constants_are_the_shipped_defaults(manifest):
    """Read off the real signature, so a change in ttt.py cannot leave a stale oracle behind."""
    params = inspect.signature(TttRls.__init__).parameters
    for key in ("stream_frac", "reg", "lam", "refit_rows", "prior", "max_delta_rms_frac", "rows_max"):
        assert manifest["constants"][key] == params[key].default, (
            f"manifest {key}={manifest['constants'][key]!r} != shipped default {params[key].default!r}"
        )
    assert manifest["constants"]["alpha"] == pytest.approx(1.0)
    assert manifest["constants"]["scaling"] == pytest.approx(1.0 / manifest["constants"]["rank"])


def test_holdout_case_is_rejected_by_the_refit_gate(manifest, arrays):
    """64 rows with holdout=True leaves 52 training rows < refit_rows=64: nothing is solved."""
    case = manifest["cases"]["holdout_refit_gate"]
    assert case["updated"] is False and case["delta_rms_frac"] == 0.0
    assert case["n_holdout"] == 64 // 5 == 12 and case["n_train"] == 52
    assert case["n_train"] < manifest["constants"]["refit_rows"]
    assert case["reject_reason"] == "refit_rows gate: n_train < refit_rows"
    assert case["b_is_zero"] is True
    assert "B_holdout_refit_gate" not in arrays


def test_default_case_satisfies_the_normal_equations(manifest, arrays):
    """``B`` is the ridge solution of ``(G + ridge*I) B^T = C`` (ttt.py:437-438)."""
    case = manifest["cases"]["default"]
    G, C, B = arrays["G_default"], arrays["C_default"], arrays["B_default"]
    assert case["updated"] is True and case["cap_bound"] is False
    assert not np.all(B == 0)

    ridge = case["ridge"]
    assert ridge == pytest.approx(float(np.mean(np.diag(G))) * manifest["constants"]["reg"], rel=1e-6)
    resid = (G + ridge * np.eye(G.shape[0])) @ B.T - C
    assert np.max(np.abs(resid)) < 1e-3 * max(1.0, float(np.max(np.abs(C)))), resid.max()

    # The accumulators themselves: G = prior*lam^nt*I + phi^T phi, C = phi^T y (ttt.py:422-424).
    phi, y = arrays["phi"], arrays["Y_default"]
    nt = case["n_train"]
    expected_G = manifest["constants"]["prior"] * case["decay"] * np.eye(8) + phi.T @ phi
    expected_C = phi.T @ y
    assert np.allclose(G, expected_G, rtol=1e-5, atol=1e-5)
    assert np.allclose(C, expected_C, rtol=1e-5, atol=1e-5)
    assert case["decay"] == pytest.approx(manifest["constants"]["lam"] ** nt, rel=1e-9)


def test_delta_rms_frac_recomputes(manifest, arrays):
    """``frac = rms(scaling * phi @ B.T) / rms(phi)`` (ttt.py:452-454) — the guard's own number."""
    phi = arrays["phi"]
    stream = np.sqrt((phi**2).mean())
    for case_name, y_name in (("default", "Y_default"), ("cap_bound", "Y_cap_bound")):
        case = manifest["cases"][case_name]
        B = arrays[f"B_{case_name}"]
        # y is the *unscaled* target, so rms(y) == stream_frac * rms(X) / scaling (ttt.py:342-346).
        y = arrays[y_name]
        assert float(np.sqrt((y**2).mean())) == pytest.approx(
            case["stream_frac"] * float(np.sqrt((arrays["X"] ** 2).mean())) / case["scaling"], rel=1e-3
        ), case_name
        frac = float(np.sqrt(((case["scaling"] * (phi @ B.T)) ** 2).mean()) / stream)
        assert frac == pytest.approx(case["delta_rms_frac"], rel=1e-4), case_name


def test_cap_bound_case_rescales_exactly(manifest, arrays):
    """The rescale branch (ttt.py:455-459): written ``B`` is the candidate times ``cap/frac_pre``."""
    case = manifest["cases"]["cap_bound"]
    cap = manifest["constants"]["max_delta_rms_frac"]
    assert case["updated"] is True and case["cap_bound"] is True
    assert case["delta_rms_frac"] == pytest.approx(cap, rel=0, abs=0)  # clamped to the cap exactly
    assert case["delta_rms_frac_pre_cap"] > cap
    ratio = cap / case["delta_rms_frac_pre_cap"]
    assert np.allclose(arrays["B_cap_bound"], arrays["B_cand_cap_bound"] * ratio, rtol=1e-4, atol=1e-8)
    # The gate still ran on the unscaled candidate, and it passed (candidate < current == 1.0).
    assert case["resid_gate_current"] == pytest.approx(1.0, rel=1e-6)
    assert case["resid_gate_candidate"] < case["resid_gate_current"]
    assert case["resid_final"] > case["resid_gate_candidate"]  # shrinking B worsens the fit


def test_regenerating_the_fixture_reproduces_it(manifest, arrays, tmp_path: Path):
    """Reproducibility: the committed output is a function of this script alone."""
    fresh_manifest, fresh_arrays = O.generate()
    assert fresh_manifest["constants"] == manifest["constants"]
    assert fresh_manifest["seeds"] == manifest["seeds"]
    assert fresh_manifest["cases"].keys() == manifest["cases"].keys()
    for name, case in fresh_manifest["cases"].items():
        for key, value in case.items():
            if isinstance(value, float):
                assert value == pytest.approx(manifest["cases"][name][key], rel=1e-9), (name, key)
            else:
                assert value == manifest["cases"][name][key], (name, key)
    assert set(fresh_arrays) == set(arrays)
    for name in arrays:
        assert np.allclose(fresh_arrays[name], arrays[name], rtol=1e-5, atol=1e-8), name


def test_main_writes_into_another_directory(tmp_path: Path):
    """The generator is not hard-wired to ``tests/fixtures`` — a caller can regenerate to temp."""
    rc = O.main(["--out-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / O.MANIFEST_NAME).exists() and (tmp_path / O.PAYLOAD_NAME).exists()
    with np.load(tmp_path / O.PAYLOAD_NAME) as z:
        assert np.allclose(np.array(z["B_default"]), np.load(PAYLOAD)["B_default"], rtol=1e-5, atol=1e-8)
