"""Receiver: chunked cross-entropy exactness, source prior, receiver gain.

The chunked cross-entropy is a hand-written ``autograd.Function`` with a
recompute in backward. That is the one place in the model where a wrong gradient
would not raise, not produce NaN, and not fail a shape check — it would simply
train to a worse optimum. So it is checked against the dense reference in fp64,
value and gradient, including the gradient with respect to the receiver gain.
"""

from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pytest
import torch

from hagi.config import HeadConfig
from hagi.model.head import LMHead, load_unigram_logprior
from tests.conftest import assert_finite


def dense_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(ce, z)`` computed the obvious way, materializing all logits."""
    logits = (hidden * scale) @ weight.t()
    if bias is not None:
        logits = logits + bias
    lse = torch.logsumexp(logits, dim=-1)
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (lse - picked).mean(), lse.pow(2).mean()


def make_head(vocab=64, hidden=16, chunk=7, z=1e-3, prior_path=None) -> LMHead:
    cfg = HeadConfig(
        unigram_prior=prior_path is not None,
        unigram_path=str(prior_path or ""),
        z_loss_weight=z,
        ce_chunk_rows=chunk,
    )
    return LMHead(hidden, vocab, cfg)


class TestChunkedCrossEntropy:
    @pytest.mark.parametrize("chunk", [1, 7, 23, 1000], ids=["one", "uneven", "exact", "oversized"])
    def test_value_matches_dense(self, chunk):
        head = make_head(chunk=chunk).double()
        h = torch.randn(23, 16, dtype=torch.float64)
        t = torch.randint(0, 64, (23,))
        ce, z = head.loss(h, t)
        ce_ref, z_ref = dense_reference(h, head.weight.detach(), t, head.logit_scale.detach(), None)
        assert abs(float((ce - ce_ref).detach())) < 1e-12
        assert abs(float((z - z_ref).detach())) < 1e-12

    def test_gradients_match_dense(self):
        head = make_head(chunk=7).double()
        h = torch.randn(23, 16, dtype=torch.float64, requires_grad=True)
        t = torch.randint(0, 64, (23,))
        ce, z = head.loss(h, t)
        (ce + 1e-3 * z).backward()

        h2 = h.detach().clone().requires_grad_(True)
        w2 = head.weight.detach().clone().requires_grad_(True)
        s2 = head.logit_scale.detach().clone().requires_grad_(True)
        ce2, z2 = dense_reference(h2, w2, t, s2, None)
        (ce2 + 1e-3 * z2).backward()

        assert float((h.grad - h2.grad).abs().max()) < 1e-12
        assert float((head.projection.weight.grad - w2.grad).abs().max()) < 1e-12
        assert float((head.logit_scale.grad - s2.grad).abs()) < 1e-12

    def test_gradient_reaches_the_gain(self):
        head = make_head()
        h = torch.randn(8, 16, requires_grad=True)
        ce, z = head.loss(h, torch.randint(0, 64, (8,)))
        (ce + z).backward()
        assert head.logit_scale.grad is not None
        assert float(head.logit_scale.grad.abs()) > 0

    def test_z_loss_zero_is_not_computed(self):
        head = make_head(z=0.0).double()
        h = torch.randn(9, 16, dtype=torch.float64)
        _, z = head.loss(h, torch.randint(0, 64, (9,)))
        assert float(z.detach()) == 0.0

    def test_empty_input(self):
        head = make_head()
        ce, z = head.loss(torch.zeros(0, 16), torch.zeros(0, dtype=torch.long))
        assert float(ce) == 0.0 and float(z) == 0.0

    def test_shape_mismatch_raises(self):
        head = make_head()
        with pytest.raises(ValueError):
            head.loss(torch.randn(4, 16), torch.zeros(5, dtype=torch.long))
        with pytest.raises(ValueError):
            head.loss(torch.randn(2, 4, 16), torch.zeros(8, dtype=torch.long))

    def test_logits_path_agrees_with_loss(self):
        head = make_head().double()
        h = torch.randn(5, 16, dtype=torch.float64)
        t = torch.randint(0, 64, (5,))
        logits = head.logits(h)
        manual = torch.nn.functional.cross_entropy(logits, t)
        ce, _ = head.loss(h, t)
        assert abs(float((ce - manual).detach())) < 1e-12


class TestReceiverGain:
    def test_default_is_inverse_sqrt_hidden(self):
        head = make_head(hidden=64)
        assert float(head.logit_scale.detach()) == pytest.approx(64**-0.5)

    def test_explicit_value_respected(self):
        cfg = HeadConfig(ce_chunk_rows=8, logit_scale_init=0.5)
        assert float(LMHead(16, 32, cfg).logit_scale.detach()) == pytest.approx(0.5)

    def test_tied_head_starts_at_the_prior(self, tmp_path):
        """The point of the gain: a tied codebook must not overwrite the prior.

        Without it the input token's own code word scores ``H * init_std`` against
        itself and the starting cross-entropy is several times the prior's.
        """
        vocab, hidden = 256, 512
        counts = np.random.default_rng(0).integers(1, 10_000, size=vocab)
        path = tmp_path / "unigram.npy"
        np.save(path, counts)

        prior = load_unigram_logprior(str(path), vocab)
        probs = prior.exp()
        entropy = float(-(probs * prior).sum())

        codebook = torch.randn(vocab, hidden) * 0.02
        head = LMHead(
            hidden,
            vocab,
            HeadConfig(unigram_prior=True, unigram_path=str(path), ce_chunk_rows=64),
            tied_weight=codebook,
        )
        # The residual stream at init is dominated by the input token's code word,
        # rescaled to unit RMS by out_norm.
        ids = torch.randint(0, vocab, (128,))
        h = codebook[ids]
        h = h / h.pow(2).mean(-1, keepdim=True).sqrt()
        targets = torch.multinomial(probs, 128, replacement=True)

        ce, _ = head.loss(h, targets)
        value = float(ce.detach())
        assert value < entropy * 1.10, f"ce {value:.3f} vs unigram entropy {entropy:.3f}"


class TestUnigramPrior:
    def test_normalized(self, tmp_path):
        counts = np.array([5, 1, 4, 0, 10], dtype=np.int64)
        path = tmp_path / "u.npy"
        np.save(path, counts)
        prior = load_unigram_logprior(str(path), 5, smoothing=1.0)
        assert abs(float(prior.exp().sum()) - 1.0) < 1e-6
        assert_finite(prior, "prior")

    def test_smoothing_keeps_zeros_finite(self, tmp_path):
        path = tmp_path / "u.npy"
        np.save(path, np.array([0, 0, 100], dtype=np.int64))
        prior = load_unigram_logprior(str(path), 3, smoothing=1.0)
        assert math.isfinite(float(prior[0]))

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_unigram_logprior(str(tmp_path / "nope.npy"), 5)

    def test_wrong_length(self, tmp_path):
        path = tmp_path / "u.npy"
        np.save(path, np.array([1, 2, 3], dtype=np.int64))
        with pytest.raises(ValueError):
            load_unigram_logprior(str(path), 5)

    def test_prior_lowers_ce_on_prior_samples(self, tmp_path):
        """A prior-matched target distribution must cost less than ``ln V``."""
        vocab = 128
        counts = np.random.default_rng(1).integers(1, 5000, size=vocab)
        path = tmp_path / "u.npy"
        np.save(path, counts)
        head = make_head(vocab=vocab, hidden=32, chunk=32, prior_path=path)
        with torch.no_grad():
            head.projection.weight.zero_()  # learned part contributes nothing
        probs = head.log_prior.exp()
        targets = torch.multinomial(probs, 4096, replacement=True)
        ce, _ = head.loss(torch.randn(4096, 32), targets)
        entropy = float(-(probs * head.log_prior).sum())
        value = float(ce.detach())
        assert abs(value - entropy) < 0.2
        assert value < math.log(vocab)


class TestTying:
    def test_tied_weight_is_shared(self):
        codebook = torch.nn.Parameter(torch.randn(32, 16))
        head = LMHead(16, 32, HeadConfig(ce_chunk_rows=8), tied_weight=codebook)
        assert head.weight is codebook
        assert head.projection is None
        assert head.is_tied

    def test_tied_weight_is_not_stored_twice(self):
        """``state_dict`` does not deduplicate; ``named_parameters`` does.

        A registered tied codebook therefore looks correct in every parameter
        count and still doubles the checkpoint — 537M extra entries at the 1B
        configuration.
        """
        codebook = torch.nn.Parameter(torch.randn(32, 16))
        head = LMHead(16, 32, HeadConfig(ce_chunk_rows=8), tied_weight=codebook)
        assert not any(v.numel() == codebook.numel() for v in head.state_dict().values())

    def test_untied_head_reports_untied(self):
        head = LMHead(16, 32, HeadConfig(ce_chunk_rows=8))
        assert not head.is_tied
        assert head.projection is not None

    def test_wrong_tied_shape_raises(self):
        with pytest.raises(ValueError):
            LMHead(16, 32, HeadConfig(), tied_weight=torch.randn(32, 8))


class TestSampledSoftmax:
    """Local partition CE: train-only surrogate of full CE (Jean et al.)."""

    def test_finite_and_backprops(self):
        cfg = HeadConfig(ce_chunk_rows=8, sampled_softmax_k=8, z_loss_weight=1e-3)
        head = LMHead(16, 64, cfg)
        h = torch.randn(12, 16, requires_grad=True)
        t = torch.randint(0, 64, (12,))
        ce, z = head.loss(h, t)
        assert math.isfinite(float(ce.detach())) and math.isfinite(float(z.detach()))
        ce.backward()
        assert h.grad is not None and torch.isfinite(h.grad).all()

    def test_k_zero_is_full_ce(self):
        cfg = HeadConfig(ce_chunk_rows=8, sampled_softmax_k=0, z_loss_weight=0.0)
        head = LMHead(16, 64, cfg).double()
        h = torch.randn(9, 16, dtype=torch.float64)
        t = torch.randint(0, 64, (9,))
        ce, _ = head.loss(h, t)
        ref, _ = dense_reference(h, head.weight.detach(), t, head.logit_scale.detach(), None)
        assert float(ce.detach()) == pytest.approx(float(ref.detach()), rel=1e-6, abs=1e-8)

    def test_empty_rows(self):
        cfg = HeadConfig(sampled_softmax_k=4)
        head = LMHead(16, 32, cfg)
        ce, z = head.loss(torch.zeros(0, 16), torch.zeros(0, dtype=torch.long))
        assert float(ce) == 0.0 and float(z) == 0.0

    def test_target_always_wins_index_zero(self):
        """Candidate set is [target, neg...]; sharp target peak → low local CE."""
        torch.manual_seed(0)
        cfg = HeadConfig(sampled_softmax_k=16, z_loss_weight=0.0, logit_scale_init=1.0)
        head = LMHead(8, 128, cfg)
        with torch.no_grad():
            head.weight.zero_()
            head.logit_scale.fill_(1.0)
            # Only the true class rows are nonzero; every negative correlates to 0.
            for tid in (3, 7, 11, 15):
                head.weight.data[tid] = torch.ones(8)
        h = torch.ones(4, 8) * 5.0
        t = torch.tensor([3, 7, 11, 15])
        ce, _ = head.loss(h, t)
        # Random among K+1 would be log(17)≈2.83; peaked target must beat that hard.
        assert float(ce.detach()) < 0.5

    def test_prior_proposal_is_finite_and_backprops(self, tmp_path):
        """Matched IS (prior proposal) must be finite and differentiable."""
        counts = np.array([50, 30, 10, 5, 3, 2, 1, 1, 1, 1], dtype=np.int64)
        path = tmp_path / "u.npy"
        np.save(path, counts)
        cfg = HeadConfig(
            unigram_prior=True,
            unigram_path=str(path),
            sampled_softmax_k=8,
            sampled_proposal="prior",
            z_loss_weight=1e-3,
        )
        head = LMHead(16, 10, cfg)
        h = torch.randn(12, 16, requires_grad=True)
        t = torch.randint(0, 10, (12,))
        ce, z = head.loss(h, t)
        assert math.isfinite(float(ce.detach())) and math.isfinite(float(z.detach()))
        ce.backward()
        assert h.grad is not None and torch.isfinite(h.grad).all()

    def test_prior_proposal_competes_against_frequent_token(self, tmp_path):
        """A source-matched bank must contain the dominant interferer."""
        counts = np.array([900, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int64)
        path = tmp_path / "u.npy"
        np.save(path, counts)
        # Target is a RARE token (id 1); the frequent token (id 0) is the
        # dominant interferer the proposal must surface.
        cfg = HeadConfig(
            unigram_prior=True,
            unigram_path=str(path),
            sampled_softmax_k=64,
            sampled_proposal="prior",
            z_loss_weight=0.0,
        )
        head = LMHead(16, 10, cfg)
        h = torch.randn(2000, 16)
        t = torch.ones(2000, dtype=torch.long)
        ce, _ = head.loss(h, t)
        # A random head cannot separate a rare target from the frequent token
        # when the frequent token is present as a negative. The CE must be
        # substantial (near log of the effective candidate mass), not ~0.
        assert float(ce.detach()) > 0.5

    def test_prior_proposal_does_not_double_count_source_bias(self, tmp_path):
        """Conditional NCE uses q=prior, so log_prior must not enter twice."""
        counts = np.array([100, 10, 1, 1], dtype=np.int64)
        path = tmp_path / "u.npy"
        np.save(path, counts)
        cfg = HeadConfig(
            unigram_prior=True,
            unigram_path=str(path),
            sampled_softmax_k=2,
            sampled_proposal="prior",
            z_loss_weight=0.0,
            logit_scale_init=1.0,
        )
        head = LMHead(4, 4, cfg)
        with torch.no_grad():
            head.weight.zero_()
        h = torch.zeros(3, 4)
        t = torch.tensor([0, 1, 1])
        negatives = torch.tensor([2, 3])
        with mock.patch("torch.multinomial", return_value=negatives):
            ce, _ = head.loss(h, t)
        assert float(ce.detach()) == pytest.approx(math.log(3), rel=1e-6)

    def test_shared_bank_matches_explicit_candidates(self):
        """A fixed shared bank must equal the explicit local partition."""
        cfg = HeadConfig(sampled_softmax_k=3, z_loss_weight=0.0, logit_scale_init=1.0)
        head = LMHead(4, 8, cfg)
        h = torch.randn(5, 4)
        t = torch.tensor([0, 1, 2, 3, 4])
        negatives = torch.tensor([5, 6, 7])

        with mock.patch("torch.randint", return_value=negatives):
            ce, _ = head.loss(h, t)

        scaled = h * head.logit_scale
        target_logits = (scaled * head.weight[t]).sum(dim=-1, keepdim=True)
        negative_logits = scaled @ head.weight[negatives].t()
        reference = -torch.log_softmax(torch.cat([target_logits, negative_logits], dim=1), dim=-1)[:, 0].mean()
        assert torch.allclose(ce, reference.float(), atol=1e-6, rtol=1e-6)

    def test_in_batch_interleaving_covers_deterministic_targets(self):
        cfg = HeadConfig(sampled_softmax_k=8, sampled_in_batch_fraction=0.5)
        head = LMHead(4, 32, cfg)
        targets = torch.arange(16)
        with mock.patch("torch.randint", return_value=torch.tensor([20, 21, 22, 23])):
            negatives = head._sample_negatives(targets, 8)
        assert torch.equal(negatives[:4], torch.tensor([0, 4, 8, 12]))
        assert torch.equal(negatives[4:], torch.tensor([20, 21, 22, 23]))

    def test_target_collision_is_masked(self):
        """A target drawn into the shared bank must not compete with itself."""
        cfg = HeadConfig(sampled_softmax_k=2, z_loss_weight=0.0, logit_scale_init=1.0)
        head = LMHead(4, 8, cfg)
        h = torch.randn(3, 4)
        t = torch.tensor([1, 2, 3])
        negatives = torch.tensor([1, 7])

        with mock.patch("torch.randint", return_value=negatives):
            ce, _ = head.loss(h, t)

        scaled = h * head.logit_scale
        target_logits = (scaled * head.weight[t]).sum(dim=-1, keepdim=True)
        negative_logits = scaled @ head.weight[negatives].t()
        negative_logits[0, 0] = float("-inf")
        reference = -torch.log_softmax(torch.cat([target_logits, negative_logits], dim=1), dim=-1)[:, 0].mean()
        assert torch.allclose(ce, reference.float(), atol=1e-6, rtol=1e-6)

    def test_logit_scale_cap(self):
        """logit_scale_max must clamp the receiver gain after a step."""
        from hagi.model.model import HAGI
        from tests.conftest import tiny_config

        cfg = tiny_config(**{"model.head.logit_scale_max": 1.0})
        model = HAGI(cfg)
        with torch.no_grad():
            model.head.logit_scale.fill_(5.0)
        model.commit_controller_updates()
        assert float(model.head.logit_scale.detach()) == pytest.approx(1.0)


class TestReceiver:
    """Receiver behavior with and without a source prior."""

    def _make_prior_path(self, tmp_path) -> str:
        counts = np.array([20, 15, 10, 5, 3, 2, 1, 1, 1, 1], dtype=np.int64)
        path = tmp_path / "u.npy"
        np.save(path, counts)
        return str(path)

    def _make_head(self, tmp_path) -> LMHead:
        path = self._make_prior_path(tmp_path)
        head_cfg = HeadConfig(unigram_prior=True, unigram_path=path, ce_chunk_rows=8)
        return LMHead(16, 10, head_cfg)

    def test_receiver_uses_exact_path_with_prior(self, tmp_path):
        head = self._make_head(tmp_path)
        h = torch.randn(64, 16, requires_grad=True)
        t = torch.randint(0, 10, (64,))
        ce, z = head.loss(h, t)
        (ce + 1e-3 * z).backward()
        assert head.logit_scale.grad is not None
        assert torch.isfinite(h.grad).all()

    def test_receiver_loss_is_finite_with_sampled_partition(self):
        head = LMHead(16, 32, HeadConfig(sampled_softmax_k=4, ce_chunk_rows=8))
        h = torch.randn(8, 16, requires_grad=True)
        t = torch.randint(0, 32, (8,))
        ce, z = head.loss(h, t)
        (ce + z).backward()
        assert torch.isfinite(ce) and torch.isfinite(z)

    def test_receiver_without_prior_is_ordinary_ce(self):
        head = LMHead(16, 32, HeadConfig(ce_chunk_rows=8))
        h = torch.randn(8, 16, requires_grad=True)
        t = torch.randint(0, 32, (8,))
        ce, _ = head.loss(h, t)
        ce.backward()
        assert torch.isfinite(ce)

        ce, z = head.loss(h, t)
        (ce + z).backward()
        assert head.logit_scale.grad is not None
