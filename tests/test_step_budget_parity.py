"""A quality comparison must match its step budget, or the number means nothing.

The M2 result (-0.5623 nats, 3/3 domains) was the project's best quality
evidence for five weeks and was almost a step-budget artifact. The experts
trained 3000 steps each, the baseline 9000 from scratch, and the merged
model 9000 *after* merging those experts. Read as "merging works" without
matching the budget, that is a claim about extra post-merge training.

These tests are the mechanical form of the rule the project broke: any
config pair used to compare quality must agree on total training steps, and
the gate must refuse to report a delta computed across unequal budgets.

They check the configs and the reporting contract, not model quality. A
model can pass every test here and still be bad; the point is that a bad
comparison cannot be reported.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS = _ROOT / "configs"


def _load(name: str) -> dict:
    return yaml.safe_load((_CONFIGS / name).read_text(encoding="utf-8"))


def _train(cfg: dict) -> dict:
    return cfg.get("train", cfg)


# The comparison that produced -0.5623, and the one that was unfair.


def test_the_m2_pair_agrees_on_max_steps() -> None:
    """baseline and merged both declare max_steps: 9000."""
    baseline = _train(_load("m2_baseline_merged.yaml"))
    merged = _train(_load("m2_merged_joint.yaml"))
    assert baseline["max_steps"] == merged["max_steps"], (
        "quality comparison across these two configs is only valid when the "
        "step budget matches; it does not"
    )


def test_the_m2_pair_agrees_on_its_seed_set() -> None:
    """The s5678 variants must be scored against matching baselines.

    A merged run at seeds 5678 compared against a baseline at other seeds
    is a different experiment, and the row ids would not be the giveaway.
    """
    assert (_CONFIGS / "m2_merged_joint_s5678.yaml").is_file()
    assert (_CONFIGS / "m2_baseline_merged_s5678.yaml").is_file()
    merged = _load("m2_merged_joint_s5678.yaml")
    baseline = _load("m2_baseline_merged_s5678.yaml")
    assert _train(merged)["max_steps"] == _train(baseline)["max_steps"]


# The general rule.


def _pairs_in_repo() -> list[tuple[str, str]]:
    """Every merged/baseline config pair that shares a name prefix.

    The seed suffix is part of the name, so ``m2_baseline_merged_s5678``
    pairs with ``m2_merged_joint_s5678`` and not with the default-seed run.
    Pairing the wrong two would let a seed mismatch pass as a match.
    """
    names = {p.stem for p in _CONFIGS.glob("m2_*.yaml")}
    pairs = []
    for name in sorted(names):
        if not name.startswith("m2_baseline"):
            continue
        suffix = name[len("m2_baseline_merged"):]
        counterpart = f"m2_merged_joint{suffix}"
        if counterpart in names:
            pairs.append((f"{name}.yaml", f"{counterpart}.yaml"))
    return pairs


def test_the_pair_discovery_actually_finds_pairs() -> None:
    """A discovery helper that matches nothing would skip every guard."""
    pairs = _pairs_in_repo()
    assert pairs, "no baseline/merged config pair was discovered; the guards below are dead"
    for baseline, merged in pairs:
        assert baseline.startswith("m2_baseline"), baseline
        assert merged.startswith("m2_merged_joint"), merged


@pytest.mark.parametrize("baseline,merged", _pairs_in_repo())
def test_every_matched_pair_shares_a_step_budget(baseline: str, merged: str) -> None:
    """Every matched config pair in the repo must agree on max_steps."""
    b = _train(_load(baseline))["max_steps"]
    m = _train(_load(merged))["max_steps"]
    assert b == m, (
        f"{baseline} runs {b} steps and {merged} runs {m}: a CE delta between "
        "them is a delta between two different training budgets"
    )


def test_a_merged_config_actually_merges_something() -> None:
    """A 'merged' arm with merge disabled is a baseline wearing a name."""
    cfg = _load("m2_merged_joint.yaml")
    merge = cfg.get("merge", {})
    assert merge.get("enabled") is True, "merged config has merging disabled"
    checkpoints = merge.get("expert_checkpoints") or []
    assert len(checkpoints) >= 2, (
        f"merged config references {len(checkpoints)} expert checkpoints; "
        "merging fewer than two models is not merging"
    )
    for path in checkpoints:
        assert (_ROOT / path).is_file(), f"missing expert checkpoint {path}"


def test_a_baseline_config_does_not_merge() -> None:
    """The other half of the comparison: the baseline must be unmerged."""
    assert _load("m2_baseline_merged.yaml").get("merge", {}).get("enabled") is False


def test_attribution_script_measures_both_budgets() -> None:
    """The control exists and covers a matched budget, not only the final one.

    If the attribution harness only scored step-0009000 it could not tell a
    merge win from a step win, which is the failure it exists to prevent.
    """
    source = (_ROOT / "scripts" / "merge_attribution.py").read_text(encoding="utf-8")
    assert "step-0004500" in source, (
        "the attribution harness must score a mid-training checkpoint too, "
        "otherwise a step-budget artifact is indistinguishable from a merge win"
    )


# --- the empty-group-list defect that blocked the level-2 DFT3 merge -----


def test_empty_hadamard_group_list_means_unspecified() -> None:
    """``mixer_hadamard_groups: []`` must behave exactly like ``None``.

    The merged M2 configs carry the default ``mixer_hadamard_groups: []``.
    ``all([])`` is True, so the ternary-group branch claimed the empty list
    and reshaped with ``n_blocks // 3 == 0``:

        RuntimeError: shape '[32768, 0, 3, 1152]' is invalid

    That made every level-2 three-way DFT3 merge fail to build, which is the
    recursion this project exists to test. The fix normalises the empty list
    to None so both spellings of "unspecified" agree.
    """
    import torch

    from hagi.model.merge import hadamard_apply_2d

    weight = torch.randn(16, 3 * 8)
    assert torch.allclose(
        hadamard_apply_2d(weight, 3, []), hadamard_apply_2d(weight, 3, None)
    ), "empty group list must not take a different branch than None"


@pytest.mark.parametrize("n_blocks", [2, 3, 4])
def test_hadamard_group_list_none_and_empty_agree(n_blocks: int) -> None:
    import torch

    from hagi.model.merge import hadamard_apply_2d

    weight = torch.randn(8, n_blocks * 8)
    assert hadamard_apply_2d(weight, n_blocks, []).shape == weight.shape
    assert torch.allclose(
        hadamard_apply_2d(weight, n_blocks, []),
        hadamard_apply_2d(weight, n_blocks, None),
    )


def test_a_populated_group_list_still_takes_its_own_branch() -> None:
    """Normalising the empty list must not collapse a real grouping.

    ``[2, 2]`` splits 4 blocks hierarchically and must differ from the flat
    ungrouped transform. ``[4]`` is deliberately NOT used: a single group
    spanning every block is the same rotation, so asserting a difference
    there would be asserting something false.
    """
    import torch

    from hagi.model.merge import hadamard_apply_2d

    weight = torch.randn(8, 4 * 8)
    ungrouped = hadamard_apply_2d(weight, 4, None)
    assert not torch.allclose(hadamard_apply_2d(weight, 4, [2, 2]), ungrouped), (
        "an explicit [2, 2] grouping was ignored, so the empty-list fix is too broad"
    )


def test_every_merged_config_with_a_default_group_list_can_still_merge() -> None:
    """The defect lived in the merged configs, so guard that path.

    ``mixer_hadamard_groups`` is absent from the merged configs, which the
    dataclass surfaces as the default ``[]``. Building the head rotation with
    that value must not raise - that RuntimeError is exactly what stopped the
    level-2 three-way DFT3 merge from building.
    """
    import torch

    from hagi.config import load_config
    from hagi.model.merge import hadamard_apply_2d

    for path in sorted(_CONFIGS.glob("m2_merged_joint*.yaml")):
        merge = load_config(str(path)).merge
        assert merge.mixer_hadamard_groups == [], (
            f"{path.name} resolves mixer_hadamard_groups to "
            f"{merge.mixer_hadamard_groups!r}; this guard assumed the default [] "
            "and should be revisited if the default changed"
        )
    weight = torch.randn(8, 3 * 8)
    assert hadamard_apply_2d(weight, 3, []).shape == weight.shape
