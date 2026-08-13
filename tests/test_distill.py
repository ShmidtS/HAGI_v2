"""Tests for the distillation module (orthogonal-expert merge glue)."""

import pytest
import torch

from hagi.distill import FeatureAdapter, distill_loss, feature_mse, logit_kl


def test_logit_kl_vanishes_when_student_matches_teacher():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    assert logit_kl(logits, logits) < 1e-5


def test_logit_kl_rewards_matching_teacher():
    teacher = torch.tensor([[5.0, 1.0, 0.0]])
    good = torch.tensor([[4.5, 1.0, 0.2]])  # close to teacher
    bad = torch.tensor([[0.0, 0.0, 5.0]])   # wrong mode
    assert logit_kl(good, teacher) < logit_kl(bad, teacher)


def test_logit_kl_requires_positive_temperature():
    with pytest.raises(ValueError):
        logit_kl(torch.zeros(1, 2), torch.zeros(1, 2), temperature=0.0)


def test_feature_mse_shape_mismatch_raises():
    with pytest.raises(ValueError):
        feature_mse(torch.zeros(2, 4), torch.zeros(2, 8))


def test_feature_adapter_projects_teacher_dim():
    adapter = FeatureAdapter(teacher_dim=8, student_dim=4, norm=False)
    out = adapter(torch.zeros(3, 8))
    assert tuple(out.shape) == (3, 4)


def test_distill_loss_dispatch():
    logits = torch.tensor([[1.0, 0.0]])
    hidden = torch.zeros(1, 4)
    # logit mode only needs logits.
    assert distill_loss(logits, logits, mode="logit") < 1e-5
    # feature mode requires hidden states.
    with pytest.raises(ValueError):
        distill_loss(logits, logits, mode="feature")
    assert distill_loss(logits, logits, mode="feature",
                        student_hidden=hidden, teacher_hidden=hidden) < 1e-5
    with pytest.raises(ValueError):
        distill_loss(logits, logits, mode="nope")
