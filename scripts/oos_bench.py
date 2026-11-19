#!/usr/bin/env python3
"""Fixed OOS benchmark: honest out-of-sample CE across waves.
Texts are held OUT of the domains (never ingested) — same every run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch as t
from hagi import unified

OOS_TEXTS = {
    "english_tech": (
        "The compiler applies constant folding before generating "
        "assembly. Register allocation uses graph coloring. The "
        "scheduler reorders instructions to hide memory latency. "
        "Cache lines align to 64 bytes. The branch predictor is a "
        "two-bit saturating counter. " * 10).encode(),
    "python_code": (
        "def train_model(loader, epochs, lr):\n"
        "    model = Net()\n"
        "    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)\n"
        "    for epoch in range(epochs):\n"
        "        for batch in loader:\n"
        "            loss = criterion(model(x), y)\n"
        "            loss.backward()\n"
        "            optimizer.step()\n" * 10).encode(),
    "russian_text": (
        "Компилятор применяет свёртку констант перед генерацией "
        "ассемблера. Распределение регистров использует раскраску "
        "графа. Планировщик переставляет инструкции. " * 10).encode("utf-8"),
}


def main():
    u = unified.Unified.load()
    for name, text in OOS_TEXTS.items():
        idx = np.frombuffer(text, dtype=np.uint8).astype(np.int64)
        k = u.k
        X = np.stack([idx[i:i + k] for i in range(0, len(idx) - k, 5)])
        y = idx[k::5][:len(X)]
        with t.no_grad():
            ce = float(t.nn.functional.cross_entropy(
                u.model.logits(t.as_tensor(X[:2000], dtype=t.long)),
                t.as_tensor(y[:2000], dtype=t.long)))
        print(f"OOS {name}: {ce:.4f}")


if __name__ == "__main__":
    main()
