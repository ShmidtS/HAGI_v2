# HAGI

HAGI is a **ternary RD-channel causal language model**:
source coder → (optional multimodal bridge) → ternary transformer channel →
tied receiver.

The current architecture is **V42** (`hagi-channel-v42`) — full causal
attention + T=512 + punctured receiver. See
[docs/V42_ARCHITECTURE.md](docs/V42_ARCHITECTURE.md).

## Recursive growth pipeline

HAGI grows a large model from small domain experts instead of training from
scratch. The idea and results live in
[GROWING_HYPOTHESIS.md](GROWING_HYPOTHESIS.md).

The pipeline (level-1, current):

1. **Level-0**: train 18 per-corpus experts (H=128) to saturation, then
   block-diagonally merge them into one wide model (H=2304).
2. **Level-1**: copy the merged level-0 model as a shared prior, specialize 4
   copies on 4 domain groups (ru/en/math/instruct), then merge them into
   H=9216 with a **recursive/local Hadamard mixer** (`mixer_hadamard_groups:
   [2,2]`).
3. **Joint training**: short run teaching the blocks to interact.

### Scripts

- `scripts/train.py` — train / resume / init-from a model.
- `scripts/merge_level1.py` — merge the 4 level-1 experts into H=9216.
- `scripts/merge_experts.py` — low-level block-diagonal merge CLI.
- `scripts/run_level1_pipeline.sh` — end-to-end level-1 driver.
- `scripts/eval_domains.py` — per-domain exact CE / perplexity eval.

### Configs

- `configs/level1/expert_*.yaml` — the 4 level-1 expert configs.
- `configs/level1_merged.yaml` — the merged H=9216 model (Hadamard mixer).
- `configs/level0_merged.yaml` — the level-0 merged base.

## Quick start

```bash
pip install -e .
python scripts/train.py --config configs/level1_merged.yaml --resume checkpoints_l1_merged/step-0000000.pt
```
