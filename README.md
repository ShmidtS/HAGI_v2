# HAGI

HAGI is a **ternary RD-channel causal language model** (BitNet b1.58):
source coder → (optional multimodal bridge) → ternary transformer channel →
tied receiver. Body-weight quantization is the only channel noise.

Current release: **V41** (`hagi-channel-v41`) —
[docs/V41_ARCHITECTURE.md](docs/V41_ARCHITECTURE.md)
(interleaved in-batch/prior conditional receiver). Prior:
[V40](docs/V40_ARCHITECTURE.md), [V39](docs/V39_ARCHITECTURE.md),
[V35](docs/V35_ARCHITECTURE.md).

## Quick start

```bash
pip install -e .
python scripts/train.py --config configs/v41_1b.yaml --dry-run
python scripts/train.py --config configs/v41_1b.yaml
```

## Layout

- `src/hagi/` — package
- `configs/v41_1b.yaml` — ship config
- `docs/` — architecture references
