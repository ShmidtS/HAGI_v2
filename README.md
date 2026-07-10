# HAGI v4

**H**ierarchical **A**ttention with **G**eometric **I**ntelligence — a communication-theoretic language model.

HAGI reformulates a neural language model as a **digital communication channel** in the sense of Claude Shannon. Instead of the standard autoregressive transformer, it treats text generation as a *codec* problem: tokens are transmitted symbols, the attention mechanism is replaced by frequency-domain (OFDM-style) processing, and iterative refinement behaves like error-correcting (LDPC / turbo) decoding.

> *"AI is just a communication channel. Hallucinations are prediction errors; repetition is echo."* — project thesis

## Why this exists

The author's working hypothesis is that a language model is isomorphic to a communication system. If that is true, then the mature tooling of information theory — modulation, channel coding, equalization, capacity allocation — should be reusable to *design* the model. HAGI v4 is an experiment in building a language model bottom-up from those primitives, rather than stacking transformer blocks.

## Key ideas

| Model component | Communication-theory analogue |
| --- | --- |
| Token embedding | Transport block + CRC |
| Random masking | Binary erasure channel |
| 2D FFT (`FreqCoding2D`) | OFDM modulation / demodulation |
| FFT bottleneck (H → C) | Rate matching (puncturing) |
| `MultiScaleGP2D` | LDPC parity check |
| `KalmanFilter` | Optimal channel estimation |
| `MSA` memory | HARQ buffer + decision-feedback equalizer |
| LM head | Demodulation to bits |

Architectural choices worth highlighting:

- **Attention-free core (default).** The forward pass uses `FreqCoding2D`: a real FFT along the time and head dimensions, learnable frequency gating, phase modulation (PSK), a low-rank complex equalizer, then an inverse FFT. Cost is `O(T·log T)` instead of `O(T²)` for attention — no QKV, no softmax, no RoPE.
- **Clifford algebra `Cl(3,0,0)`.** The hidden state is structured into the 8 blades of a 3D geometric algebra (scalar, 3 vectors, 3 bivectors, trivector). The geometric product between adjacent positions acts as a parity channel for error correction.
- **Kalman-filter iterative refinement.** Each refinement iteration blends a prediction (frequency blocks) with a parity measurement (`GP2D`) via an optimal Bayesian update.
- **Plane prediction (non-autoregressive).** Random tokens are masked and all of them are predicted simultaneously, then refined over several iterations (BERT / LLaDA style) rather than one token at a time.
- **Knowledge distillation.** Training distills from `HuggingFaceTB/SmolLM2-360M` (and its embedding table from the 135M variant).
- **Muon + AdamW hybrid optimizer.** Newton–Schulz-orthogonalized Muon for 2-D weight matrices, AdamW for the rest.

## Requirements

- **Python ≥ 3.12**
- **PyTorch ≥ 2.0**
- A **CUDA GPU** is required for training; inference also expects GPU.
- Core dependencies: `numpy`, `torch`, `pyyaml`.

## Installation

```bash
git clone https://github.com/ShmidtS/HAGI_v2.git
cd HAGI_v2
pip install -e .
# optional developer tooling
pip install -e ".[dev]"   # pytest, ruff
```

Optional integrations (not required to run the core model):

```bash
pip install transformers   # for knowledge distillation
pip install triton        # for fused frequency kernels
```

## Configuration

All model dimensions are derived from a single `target_params` value through `auto_configure()` (see `src/hagi_v4/config.py`). Three presets ship with the repo:

| Config | Approx. params | VRAM target | Tokenizer (vocab) |
| --- | --- | --- | --- |
| `configs/8gb_canonical.yaml` | ~17.5M | 8 GB | SmolLM2 (49 154) |
| `configs/8gb_google.yaml` | ~75M | 8 GB | Gemma (262 146) |
| `configs/24gb_google.yaml` | ~75M | 24 GB | Gemma (262 146) |

## Training

```bash
python scripts/train_v4.py --config configs/8gb_canonical.yaml
```

Useful flags (see `scripts/train_v4.py`):

```bash
python scripts/train_v4.py --dry-run          # validate config + data without training
python scripts/train_v4.py --no-distill       # disable knowledge distillation
python scripts/train_v4.py --steps 50000      # override max_steps
python scripts/train_v4.py --resume           # resume from latest checkpoint
```

Training reads pre-tokenized `.bin` datasets from a `data/` directory (with a `mix.json` describing ratios) and follows a curriculum over several corpora. Checkpoints are written to `checkpoints/` (excluded from git).

## Inference

```bash
python scripts/infer_v4.py --config configs/8gb_canonical.yaml --checkpoint checkpoints/<step>.pt
```

Generation is block-parallel with iterative refinement and an optional speculative branch (see `src/hagi_v4/inference/`).

## Project layout

```
src/hagi_v4/
  config.py                 # dataclasses + auto_configure()
  algebra/clifford.py       # Cl(3,0,0) geometric product (Cayley table)
  model/
    hagi_v4.py              # HAGIv4 model + TurboLoop
    freq_layer.py           # FreqCoding2D (FFT-based attention replacement)
    gp2d.py                 # GeometricProduct2D / parity channel
    multiscale_gp2d.py      # multi-window LDPC-like parity
    kalman.py               # KalmanFilter refinement
    msa.py                  # Memory Sparse Attention (HARQ buffer)
    attention.py            # Grouped-query attention (fallback path)
    transformer_block.py    # standard block (fallback path)
    moe.py, hrm.py, gdr.py  # Mixture-of-Experts, Hierarchical Recurrent Memory, Grade-Decomposed Recurrence
    norms.py, cqi.py, masking.py, cast.py, variational_bottleneck.py,
    turbo_decoder.py, water_filling.py, wave_routing.py, wave_block.py,
    contrastive.py, cross_modal_*.py, multimodal_input.py, multimodal_masking.py,
    triton_kernels.py, outputs.py
  train/
    optim.py, losses.py, loop.py, distillation.py, checkpoint.py
  data/
    dataset.py, sequential.py
  research/
    dataset.py              # TinyStories loader
  inference/
    generate.py, speculative.py, kv_cache.py
scripts/
  train_v4.py, infer_v4.py, download_model.py,
  loss_bpt_search.py, loss_bpt_parallel.py, _train_utils.py
configs/
  8gb_canonical.yaml, 8gb_google.yaml, 24gb_google.yaml
docs/
  ARCHITECTURE_V5.md        # detailed architecture write-up (Russian)
```

## Status

This is an **active research experiment**, not a production system.

- The full architecture is implemented and wired end-to-end.
- Several subsystems are implemented but **disabled by default**: `multimodal`, `wave_routing`, and the Triton fused kernels (the core uses pure-PyTorch frequency layers).
- There is currently **no test suite** and **no published checkpoints**.
- The codebase is still evolving — V4 (HRM-era) and V7 (turbo-loop) components coexist and are selected via config flags.

Contributions, reproductions, and discussion are welcome.

## License

No license is specified yet. If you intend to use or redistribute this code, please coordinate with the author.
