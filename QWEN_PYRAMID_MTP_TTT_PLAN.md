# Qwen Pyramid MTP/TTT plan (frozen ternary base + separate online LoRA)

Status: research → design → implementation.

Baseline: `ProCreations/Ternary-Bonsai-2-27B-MTP`
(`Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf`, 7,657,489,728 bytes,
SHA256 `83a0aea0d7c3e7c8a9bd4e8d4ef85a3a14b33c92d1eebb161b40afb239c81bb3`).

Target: a text model with a pyramid-connected FFN path and a separate online
TTT-LoRA contour, running on AMD Radeon 8060S / ROCm 7.13. The GLM/STE track is
closed and must not be resumed.

## 1. Understand

Use the already-trained Bonsai MTP artifact instead of repeating STE. Keep the
ternary base frozen. Add an experimental pyramid path and a low-rank online
adaptation path as separate, checkpointed modules. Preserve an honest
validation metric and save only on measured improvement.

## 2. Verified architecture contract

- The language backbone is derived from `Qwen/Qwen3.8-27B` at pinned revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- The local runtime/config class is `Qwen3_5TextConfig` / Qwen3.5-compatible
  text architecture. The repository name and the runtime class are therefore
  not interchangeable labels.
- 64 text layers, hidden size 5120, intermediate size 17408, 24 attention heads,
  4 KV heads, head dimension 256, vocabulary 248320, MRoPE sections
  `[11, 11, 10]`, context 262144.
- `layer_types` alternates linear-attention and full-attention layers; this is
  not a MoE model.
- One MTP layer follows layer 64. Its 15 tensors are recorded in
  `mtp_config.json` and match `model_mtp.safetensors`.
- The MTP forward contract is `embedding[t]` plus the previous hidden state,
  through `mtp.fc`, one decoder layer, and `mtp.norm`.
- The published MTP objective is forward KL to frozen soft targets plus
  `0.1 * CE(argmax target)`; every fourth step adds a two-step
  self-conditioning loss with weight `0.25`.

## 3. Runtime and memory constraints

- `torch` is `2.10.0+rocm7.13.0a20260512`; the device is AMD Radeon 8060S with
  107.9 GiB VRAM.
- The Prism runtime is Linux/CUDA 13.3/SM120-only and cannot be used here.
- `peft`, `bitsandbytes`, `diffusers`, `torchao`, `trl`, and `optimum` are not
  installed. Use a thin manual LoRA/RLS implementation unless a verified
  dependency is added deliberately.
- PQ2_0 is a custom GGML type (`142`). The installed `gguf.GGUFReader` rejects
  it before tensor loading, so E0 must include a verified custom reader path;
  stock Transformers/llama.cpp cannot be assumed to work.
- PQ2_0 occupies 7.21 GB on disk. A full BF16 materialization is roughly
  55 GB before activations and KV cache. The 107.9 GiB device therefore has
  limited headroom for full-context prefill; use layer-wise/offloaded loading,
  batch size 1, and int8 KV cache. Do not claim that a full BF16 model plus
  cache fits without measuring it.

## 4. Local code map

- `scripts/dsv4_pyramid.py` at commit `f09584d`: DSv4 MoE pyramid prototype.
  It is a topology reference only and must not be imported into the dense Qwen
  path unchanged.
- `scripts/dsv4_generate_ttt.py`: anchored RLS, `Hva/Yva` holdout buffers,
  `TTT_MIN_GAIN`, and low-rank persistence. Reuse the invariants, not its
  DSv4-specific hooks.
- `llama-glm5/conversion/qwen.py`: Qwen MTP remapping (`mtp.*` to layer 64)
  and Qwen3.5 tensor handling.
- `llama-ds4/src/models/qwen35.cpp` and
  `llama-ds4/tests/snapshots/qwen3.5-27b.schema`: runtime tensor/layout reference.
- `models/ternary-bonsai-2-27b-mtp/training/head.py`: exact MTP module contract.
- `models/ternary-bonsai-2-27b-mtp/training/train.py` and
  `smoke_train.py`: objective and validation contract.
- `models/qwen3_8-27b-tokenizer/`: official Qwen3.8 tokenizer assets. Its BPE
  vocab is 248044; model vocab is 248320. The extra model IDs are primarily
  vision/control IDs. E3 must validate the tokenizer/model ID contract rather
  than silently clip arbitrary text IDs.

## 5. Atomic plan and gates

### E0 — custom frozen GGUF loader and tensor parity

Create `scripts/bonsai_gguf_load.py` with:

- `--gguf`, `--manifest`, `--sha256`, and `--smoke` arguments;
- SHA256 verification before parsing;
- a custom PQ2_0-aware GGUF reader path, with stock `gguf.GGUFReader` used only
  for supported metadata/tensors;
- extraction of at least one base tensor and all 15 MTP tensors;
- shape/dtype checks against `mtp_config.json`;
- round-trip parity against `model_mtp.safetensors` for the MTP tensors.

Gate: `py_compile` passes, checksum matches, all 15 MTP tensors are found, and
the parity smoke exits 0. No base tensor may be modified or written back.

### E1 — identity-preserving pyramid FFN adapter

Create a dense-Qwen adapter, not a copy of the DSv4 MoE prototype:

- select a small, explicit set of text-layer FFN outputs;
- run selected FFN branches in parallel on the same frozen-base activation;
- mix with a fixed residual + mean path first;
- keep attention, recurrent state, tokenizer, and base weights unchanged;
- expose a flat reference path and a pyramid path with identical inputs.

Gate: on a bounded activation slice, pyramid output equals the flat reference
within a predeclared relative-error tolerance, with no NaN/Inf and no change to
the frozen base. A learned pyramid objective is not allowed until this
identity gate passes.

### E2 — separate online TTT-LoRA contour

Create `scripts/qwen_ttt_lora.py` with manual low-rank `A/B` parameters:

- target only explicit MTP projections (initially MLP gate/up/down; attention
  projections are opt-in, not implicit);
- keep the ternary base and the frozen MTP anchor separate;
- maintain an anchored RLS state and a 1/5-row honest holdout (`Hva/Yva`);
- persist a separate checkpoint containing only LoRA state, optimizer state,
  step, validation metric, and RNG state;
- save atomically only when honest validation improves by at least the declared
  threshold;
- never use training rows for the save decision.

Gate: a bounded smoke shows finite loss, no base-parameter gradients, honest
validation rows excluded from updates, and at least one measured improvement or
an explicit no-save result. KL alone is not a sufficient quality proxy.

### E3 — ROCm smoke generation

After E0–E2 gates, run a batch-1 smoke with the official Qwen tokenizer and the
custom loader:

- short prompt, bounded context, 128-token maximum;
- verify finite logits, valid token IDs, coherent finish behavior, and no NaN;
- run a small objective-answer subset and record both strict and normalized
  results;
- report device memory, prompt throughput, decode throughput, and whether the
  pyramid/TTT paths were active.

Gate: the smoke exits 0 and produces a machine-readable result file. It is a
runtime smoke, not a claim of full benchmark parity.

## 6. Verification evidence

- E0: command output, checksum, tensor count/shapes, and parity values.
- E1: flat-vs-pyramid error and memory measurement on the exact smoke input.
- E2: train/validation split, update exclusion proof, checkpoint diff, and
  honest metric before/after.
- E3: generated-token count, finish reason, objective checks, and ROCm memory.

## 7. Fallbacks

- If custom PQ2_0 decoding is incomplete, stop E1–E3; do not fall back to a
  stock reader that silently misinterprets type 142.
- If VRAM is insufficient, use CPU-pinned tensor storage, layer-wise loading,
  and int8 KV cache before changing the model contract.
- If `peft` is unavailable, retain the manual LoRA implementation; do not add
  an unmeasured dependency.
- If the tokenizer/model ID contract is not validated, E3 is blocked rather
  than silently clipping IDs.

## 8. Risks and decision gates

- The largest unknown is the custom PQ2_0 decode/kernel contract on AMD ROCm.
- The DSv4 pyramid code is not portable to dense Qwen without a new adapter.
- A single MTP layer is a narrow adaptation surface; E2 must remain separate
  from the frozen ternary base.
- Full-generation quality and throughput remain unverified until E3.
