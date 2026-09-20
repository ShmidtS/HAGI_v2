import tempfile
import unittest
from pathlib import Path

import torch

from hagi.inference.lora import HeadLoRA


class HeadLoRATests(unittest.TestCase):
    def test_zero_initialized_adapter_is_exact_noop(self):
        adapter = HeadLoRA(16, 32, rank=4)
        x = torch.randn(3, 16)
        self.assertTrue(torch.equal(adapter(x), torch.zeros(3, 32)))

    def test_parameter_count_and_roundtrip(self):
        adapter = HeadLoRA(8, 11, rank=3)
        self.assertEqual(adapter.trainable_params, 3 * 8 + 11 * 3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "adapter.pt"
            adapter.save(path)
            restored = HeadLoRA.load(path, 8, 11, torch.device("cpu"))
            self.assertTrue(torch.equal(adapter.A, restored.A))
            self.assertTrue(torch.equal(adapter.B, restored.B))


if __name__ == "__main__":
    unittest.main()
