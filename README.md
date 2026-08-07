# HAGI

HAGI is a **ternary RD-channel causal language model**:
source coder → (optional multimodal bridge) → ternary transformer channel →
tied receiver.

Current releases:
- **V42** (`hagi-channel-v42`) —
  [docs/V42_ARCHITECTURE.md](docs/V42_ARCHITECTURE.md)
  (full causal attention + T=512 + punctured receiver; 43% faster than V41).
- **V41** (`hagi-channel-v41`) —
  [docs/V41_ARCHITECTURE.md](docs/V41_ARCHITECTURE.md)
  (interleaved in-batch/prior conditional receiver).

## Quick start

```bash
pip install -e .
python scripts/train.py --config configs/v42_1b.yaml --data-dir data
```
