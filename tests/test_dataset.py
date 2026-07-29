"""Tests for MemmapDataset and validate_terminal_eos."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from hagi.data.dataset import MemmapDataset, validate_terminal_eos, dataset_path


class TestValidateTerminalEOS:
    def test_valid(self):
        ids = torch.tensor([[1, 2, 3, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]])
        valid = validate_terminal_eos(ids, eos_token_id=0, pad_token_id=9)
        assert valid[0, :4].all() and not valid[0, 4:].any()

    def test_rejects_no_eos(self):
        with pytest.raises(ValueError, match="exactly one"):
            validate_terminal_eos(torch.tensor([[1, 2, 3, 4]]), eos_token_id=0, pad_token_id=9)

    def test_rejects_multiple_eos(self):
        with pytest.raises(ValueError, match="exactly one"):
            validate_terminal_eos(torch.tensor([[0, 1, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]),
                                  eos_token_id=0, pad_token_id=9)


@pytest.fixture
def bin_path():
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        np.array([1, 2, 3, 0, 9, 9], dtype=np.uint16).tofile(path)
        yield path
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


class TestMemmapDataset:
    def test_len_positive(self, bin_path):
        ds = MemmapDataset(bin_path, seq_len=4, vocab_size=1000, eos_token_id=0, pad_token_id=9)
        assert len(ds) > 0

    def test_item_shapes(self, bin_path):
        ds = MemmapDataset(bin_path, seq_len=8, vocab_size=1000, eos_token_id=0, pad_token_id=9)
        item = ds[0]
        assert item["input_ids"].shape == (8,) and item["input_ids"].dtype == torch.long
        assert "valid_target_mask" in item and item["valid_target_mask"].shape == (8,)

    def test_eos_structure(self, bin_path):
        ds = MemmapDataset(bin_path, seq_len=8, vocab_size=1000, eos_token_id=0, pad_token_id=9)
        ids = ds[0]["input_ids"]
        eos_pos = (ids == 0).nonzero(as_tuple=True)[0]
        assert len(eos_pos) == 1
        assert (ids[eos_pos[0] + 1:] == 9).all()

    def test_without_eos_pad(self, bin_path):
        ds = MemmapDataset(bin_path, seq_len=4, vocab_size=1000)
        assert "valid_target_mask" not in ds[0]

    def test_rejects_seq_len_1(self, bin_path):
        with pytest.raises(ValueError):
            MemmapDataset(bin_path, seq_len=1, vocab_size=1000)

    def test_rejects_eos_without_pad(self, bin_path):
        with pytest.raises(ValueError, match="together"):
            MemmapDataset(bin_path, seq_len=8, vocab_size=1000, eos_token_id=0)


class TestDatasetPath:
    def test_valid(self):
        assert dataset_path("data", "tinystories").name == "tinystories.bin"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            dataset_path("data", "")

    def test_rejects_traversal(self):
        with pytest.raises(ValueError):
            dataset_path("data", "..")
