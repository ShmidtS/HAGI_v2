# HAGI

HAGI is a **ternary RD-channel causal language model** (BitNet b1.58):
source coder → (optional multimodal bridge) → ternary transformer channel →
tied receiver. Body-weight quantization is the only channel noise.

Current release: **V39** (`hagi-channel-v39`) —
[docs/V39_ARCHITECTURE.md](docs/V39_ARCHITECTURE.md)
(L=4 + punctured CE + sampled softmax). Prior:
[V38](docs/V38_ARCHITECTURE.md), [V37](docs/V37_ARCHITECTURE.md),
[V36](docs/V36_ARCHITECTURE.md), [V35](docs/V35_ARCHITECTURE.md).

## Quick start

```bash
pip install -e .
python scripts/train.py --config configs/v39_1b.yaml --dry-run
python scripts/train.py --config configs/v39_1b.yaml
```

## Layout

- `src/hagi/` — package
- `configs/v39_1b.yaml` — ship config
- `docs/` — architecture references
