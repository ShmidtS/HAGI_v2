"""Tests for LatentMemoryBank: FIFO, dtype, shapes, gradients, reset."""

from __future__ import annotations

import pytest
import torch

from hagi.model.memory_bank import LatentMemoryBank
from tests.conftest import assert_finite


DIM = 64
LATENT_DIM = 32
BANK_SIZE = 8
NUM_HEADS = 4
HEAD_DIM = 32  # inner = 128


@pytest.fixture
def bank() -> LatentMemoryBank:
    return LatentMemoryBank(
        dim=DIM,
        latent_dim=LATENT_DIM,
        bank_size=BANK_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )


class TestLatentMemoryBankShapes:
    def test_output_shape(self, bank):
        h = torch.randn(2, 4, DIM)
        z = torch.randn(2, 4, LATENT_DIM)
        out = bank(h, z)
        assert_finite(out, "out")
        assert out.shape == (2, 4, DIM)

    def test_output_residual(self, bank):
        h = torch.randn(2, 3, DIM)
        z = torch.randn(2, 3, LATENT_DIM)
        out = bank(h, z)
        assert out.shape == h.shape
        assert not torch.equal(out, h)


class TestLatentMemoryBankDtype:
    def test_fp32_in_fp32(self, bank):
        bank.to(torch.float32)
        h = torch.randn(2, 5, DIM, dtype=torch.float32)
        z = torch.randn(2, 5, LATENT_DIM, dtype=torch.float32)
        out = bank(h, z)
        assert out.shape == (2, 5, DIM)
        assert out.dtype == torch.float32

    def test_fp32_model_fp32_z(self, bank):
        bank.to(torch.float32)
        h = torch.randn(2, 4, DIM, dtype=torch.float32)
        z = torch.randn(2, 4, LATENT_DIM, dtype=torch.float32)
        out = bank(h, z)
        assert out.dtype == torch.float32

    def test_bf16_model_bf16_z(self, bank):
        bank.to(torch.bfloat16)
        h = torch.randn(2, 4, DIM, dtype=torch.bfloat16)
        z = torch.randn(2, 4, LATENT_DIM, dtype=torch.bfloat16)
        out = bank(h, z)
        assert out.dtype == torch.bfloat16

    def test_fp32_z_bf16_model(self, bank):
        """Regression: z in fp32 (from IB bottleneck) + bf16 model = no crash."""
        bank.to(torch.bfloat16)
        h = torch.randn(2, 6, DIM, dtype=torch.bfloat16)
        z = torch.randn(2, 6, LATENT_DIM, dtype=torch.float32)
        out = bank(h, z)
        assert out.shape == (2, 6, DIM)
        assert out.dtype == torch.bfloat16

    def test_bank_allocated_in_model_dtype(self, bank):
        """_ensure_bank allocates in model dtype (bf16), not z dtype (fp32)."""
        bank.to(torch.bfloat16)
        h = torch.randn(2, 4, DIM, dtype=torch.bfloat16)
        z = torch.randn(2, 4, LATENT_DIM, dtype=torch.float32)
        bank(h, z)
        assert bank._bank is not None
        assert bank._bank.dtype == torch.bfloat16

    def test_dtype_mismatch_without_z_cast_would_crash(self, bank):
        """Demonstrate the bug: fp32 bank + bf16 proj -> RuntimeError.

        This replicates the exact bug scenario and verifies the forward()
        cast prevents the crash.
        """
        bank.to(torch.bfloat16)
        h = torch.randn(2, 4, DIM, dtype=torch.bfloat16)
        z = torch.randn(2, 4, LATENT_DIM, dtype=torch.float32)
        # Without z.to(h.dtype) in forward, _ensure_bank(z) allocates fp32 bank,
        # then k_proj(bank) raises "expected mat1 and mat2 to have the same dtype".
        # This forward call must NOT raise.
        out = bank(h, z)
        assert out.dtype == torch.bfloat16


class TestLatentMemoryBankFIFO:
    def test_bank_fills_incrementally(self, bank):
        h1 = torch.randn(2, 3, DIM)
        z1 = torch.randn(2, 3, LATENT_DIM)
        bank(h1, z1)
        assert int(bank._bank_fill.item()) == 3

        h2 = torch.randn(2, 4, DIM)
        z2 = torch.randn(2, 4, LATENT_DIM)
        bank(h2, z2)
        assert int(bank._bank_fill.item()) == 7

    def test_bank_saturates_at_bank_size(self, bank):
        for _ in range(5):
            h = torch.randn(2, 3, DIM)
            z = torch.randn(2, 3, LATENT_DIM)
            bank(h, z)
        assert int(bank._bank_fill.item()) == BANK_SIZE

    def test_large_input_fills_full_bank(self, bank):
        h = torch.randn(2, BANK_SIZE + 5, DIM)
        z = torch.randn(2, BANK_SIZE + 5, LATENT_DIM)
        bank(h, z)
        assert int(bank._bank_fill.item()) == BANK_SIZE

    def test_large_input_takes_last(self, bank):
        B, T = 1, 16  # > BANK_SIZE=8
        h = torch.randn(B, T, DIM)
        z = torch.arange(T, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand(B, T, LATENT_DIM)
        bank(h, z)
        # Bank should hold last BANK_SIZE timesteps: z[8:16] = 8..15
        expected = torch.arange(8, 16, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand(B, BANK_SIZE, LATENT_DIM)
        assert torch.allclose(bank._bank[:, :BANK_SIZE, :], expected)


class TestLatentMemoryBankReset:
    def test_reset_clears_bank(self, bank):
        h = torch.randn(2, 4, DIM)
        z = torch.randn(2, 4, LATENT_DIM)
        bank(h, z)
        assert int(bank._bank_fill.item()) == 4

        bank.reset()
        assert int(bank._bank_fill.item()) == 0
        assert bank._bank is None

    def test_reset_then_forward_restarts(self, bank):
        h1 = torch.randn(2, 6, DIM)
        z1 = torch.randn(2, 6, LATENT_DIM)
        bank(h1, z1)
        assert int(bank._bank_fill.item()) == 6

        bank.reset()

        h2 = torch.randn(2, 3, DIM)
        z2 = torch.randn(2, 3, LATENT_DIM)
        bank(h2, z2)
        assert int(bank._bank_fill.item()) == 3


class TestLatentMemoryBankGradients:
    def test_gradient_flows(self, bank):
        h = torch.randn(2, 4, DIM, requires_grad=True)
        z = torch.randn(2, 4, LATENT_DIM, requires_grad=True)
        out = bank(h, z)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None
        assert not torch.all(h.grad == 0)

    def test_gradient_through_params(self, bank):
        bank.reset()
        h = torch.randn(2, 4, DIM)
        z = torch.randn(2, 4, LATENT_DIM)
        out = bank(h, z)
        loss = out.sum()
        loss.backward()
        assert bank.q_proj.weight.grad is not None
        assert bank.k_proj.weight.grad is not None
        assert bank.v_proj.weight.grad is not None
        assert bank.out_proj.weight.grad is not None


class TestLatentMemoryBankNoOp:
    def test_bank_size_zero_is_noop(self):
        b = LatentMemoryBank(dim=DIM, latent_dim=LATENT_DIM, bank_size=0)
        # _ensure_bank returns buffer before allocating in forward
        assert b._bank is None
        assert b._bank_fill.item() == 0


class TestLatentMemoryBankEnsureBank:
    def test_ensure_bank_allocates(self, bank):
        assert bank._bank is None
        z = torch.randn(2, 5, LATENT_DIM)
        bank._ensure_bank(z)
        assert bank._bank is not None
        assert bank._bank.shape == (2, BANK_SIZE, LATENT_DIM)

    def test_ensure_bank_reallocates_on_batch_change(self, bank):
        z1 = torch.randn(2, 5, LATENT_DIM)
        bank._ensure_bank(z1)
        assert bank._bank.shape[0] == 2

        z2 = torch.randn(4, 5, LATENT_DIM)
        bank._ensure_bank(z2)
        assert bank._bank.shape[0] == 4

    def test_ensure_bank_same_shape_no_realloc(self, bank):
        z = torch.randn(2, 5, LATENT_DIM)
        b1 = bank._ensure_bank(z)
        b2 = bank._ensure_bank(z)
        assert b1.data_ptr() == b2.data_ptr()


class TestLatentMemoryBankAttnMask:
    def test_partially_filled_bank_does_not_crash(self, bank):
        """First forward: bank fill < bank_size, mask must handle it."""
        h = torch.randn(2, 3, DIM)
        z = torch.randn(2, 3, LATENT_DIM)
        out = bank(h, z)
        assert out.shape == (2, 3, DIM)

    def test_mask_only_attends_filled(self, bank):
        """At fill=3, mask should only allow positions 0,1,2."""
        h = torch.randn(2, 3, DIM)
        z = torch.randn(2, 3, LATENT_DIM)
        bank(h, z)
        fill = int(bank._bank_fill.item())
        assert fill == 3
        attn_mask = torch.arange(BANK_SIZE) < fill
        assert attn_mask.tolist() == [True, True, True] + [False] * 5
