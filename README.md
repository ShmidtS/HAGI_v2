# HAGI

HAGI is a **ternary RD-channel causal language model** (BitNet b1.58). The
transformer body is ternary (`{-1, 0, +1}` weights), and that quantization is
the *genuine* discrete communication channel — its noise is the only
impairment. A factorized source encoder (causal conv, no future leak) feeds
the ternary body; a variational information bottleneck acts as an auxiliary
KL-rate regularizer off the main LM path; optional predictive-decoder and
multimodal fusion are opt-in.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## Quick start

```bash
pip install -e .

# train (all hyperparameters come from the YAML config)
python scripts/train.py --config configs/smollm2.yaml --no-distill

# generate
python scripts/infer.py --checkpoint checkpoints/step-010000.pt --interactive
```

## Layout

- `src/hagi/` — the package (`config.py`, `data/`, `model/`, `train/`, `inference/`)
- `configs/` — canonical deployment configs (`smollm2.yaml`, `google.yaml`)
- `scripts/` — `train.py`, `infer.py`, `download_model.py`
- `docs/ARCHITECTURE.md` — architecture reference

Every tunable parameter lives in `src/hagi/config.py` (and the YAML configs);
the code contains no hardcoded constants.
