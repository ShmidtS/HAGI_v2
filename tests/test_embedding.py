"""Source coder: codebook, causal filter, and decode-state exactness.

The causal-leak assertion is the important one. A symmetric convolution puts
token ``t+1`` inside position ``t``'s representation, which makes training loss
drop beautifully and generation produce garbage — the exact failure this project
already shipped once.
"""

from __future__ import annotations

import pytest
import torch

from hagi.model.embedding import SourceEncoder
from tests.conftest import assert_finite


class TestShapes:
    def test_output_shape(self):
        enc = SourceEncoder(64, 32, conv_kernel=4)
        out = enc(torch.randint(0, 64, (2, 7)))
        assert out.shape == (2, 7, 32)
        assert_finite(out, "encoder output")

    def test_conv_kernel_one_disables_the_filter(self):
        enc = SourceEncoder(64, 32, conv_kernel=1)
        assert enc.conv is None and enc.norm is None
        ids = torch.randint(0, 64, (2, 5))
        assert torch.equal(enc(ids), enc.embedding(ids))

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError):
            SourceEncoder(64, 32, conv_kernel=0)

    def test_weight_property_exposes_the_codebook(self):
        enc = SourceEncoder(64, 32)
        assert enc.weight is enc.embedding.weight
        assert enc.weight.shape == (64, 32)


class TestCausality:
    @pytest.mark.parametrize("kernel", [2, 3, 4, 8])
    def test_future_tokens_do_not_reach_the_past(self, kernel):
        enc = SourceEncoder(64, 32, conv_kernel=kernel).eval()
        ids = torch.randint(0, 64, (1, 12))
        with torch.no_grad():
            base = enc(ids)
            changed = ids.clone()
            changed[0, 6] = (int(ids[0, 6]) + 1) % 64
            perturbed = enc(changed)
        assert torch.allclose(base[:, :6], perturbed[:, :6], atol=1e-6), (
            f"kernel {kernel}: changing token 6 changed an earlier position"
        )
        assert not torch.allclose(base[:, 6], perturbed[:, 6]), "the change had no effect at all"

    def test_left_pad_width(self):
        enc = SourceEncoder(64, 32, conv_kernel=5)
        assert enc.left_pad == 4


class TestDecodeState:
    def test_incremental_matches_full(self):
        """Token-at-a-time encoding must be bit-comparable to one full pass."""
        enc = SourceEncoder(64, 32, conv_kernel=4).eval()
        ids = torch.randint(0, 64, (2, 9))
        with torch.no_grad():
            full = enc(ids)
            enc.reset_state()
            steps = [enc(ids[:, t : t + 1], use_state=True) for t in range(9)]
        incremental = torch.cat(steps, dim=1)
        assert float((full - incremental).abs().max()) < 1e-5

    def test_prefill_then_decode_matches_full(self):
        enc = SourceEncoder(64, 32, conv_kernel=4).eval()
        ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            full = enc(ids)
            enc.reset_state()
            prefill = enc(ids[:, :6], use_state=True)
            rest = [enc(ids[:, t : t + 1], use_state=True) for t in range(6, 10)]
        combined = torch.cat([prefill, *rest], dim=1)
        assert float((full - combined).abs().max()) < 1e-5

    def test_reset_clears_history(self):
        enc = SourceEncoder(64, 32, conv_kernel=4).eval()
        ids = torch.randint(0, 64, (1, 3))
        with torch.no_grad():
            enc(ids, use_state=True)
            assert enc._state is not None
            enc.reset_state()
            assert enc._state is None

    def test_state_is_not_in_the_graph(self):
        enc = SourceEncoder(64, 32, conv_kernel=4)
        ids = torch.randint(0, 64, (1, 2))
        enc(ids, use_state=True)
        assert not enc._state.requires_grad


class TestInit:
    def test_filter_starts_near_passthrough(self):
        """The last tap dominates, so the filter cannot corrupt the code at step 0."""
        enc = SourceEncoder(64, 32, conv_kernel=4)
        weight = enc.conv.weight.detach()
        assert float(weight[:, 0, -1].mean()) > 0.9
        assert float(weight[:, 0, :-1].abs().mean()) < 0.1

    def test_codebook_std(self):
        enc = SourceEncoder(4096, 64, init_std=0.02)
        assert float(enc.embedding.weight.detach().std()) == pytest.approx(0.02, rel=0.1)
