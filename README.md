# HAGI

HAGI is a **ternary RD-channel causal language model**:
source coder → (optional multimodal bridge) → ternary transformer channel →
tied receiver.

Current release: **V41** (`hagi-channel-v41`) —
[docs/V41_ARCHITECTURE.md](docs/V41_ARCHITECTURE.md)
(interleaved in-batch/prior conditional receiver).

## Quick start

```bash
pip install -e .
python scripts/train.py --config configs/v41_1b.yaml --data-dir data
```
