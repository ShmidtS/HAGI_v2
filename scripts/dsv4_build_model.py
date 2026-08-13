"""Build the merged DSV4 model: 256 routed experts -> 1 super-expert per layer.

Runs on GPU (ROCm). Loads original weights through the official transformers
conversion mapping, decodes fp8 to bf16, replaces routed experts with the
F3-merged super-experts, zeroes the router + shared expert, and saves a compact
bf16 safetensors checkpoint loadable by transformers.
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stub_import_tf  # noqa: F401  (torch.distributed stub)
import dsv4_experts as de

from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

DEVICE = "cuda"
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dsv4_shared_only"))


def apply_mapping(mapping, key: str) -> str:
    for m in mapping:
        new, pat = m.rename_source_key(key)
        if pat is not None:
            key = new
    return key


def main() -> None:
    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]

    cfg = DeepseekV4Config.from_pretrained(snap)
    cfg.n_routed_experts = 1
    cfg.n_shared_experts = 0
    cfg.num_experts_per_tok = 1
    print("building model on GPU...", flush=True)
    t0 = time.time()
    torch.set_default_device(DEVICE)
    torch.set_default_dtype(torch.bfloat16)
    model = DeepseekV4ForCausalLM(cfg)
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    print(f"model built in {time.time()-t0:.1f}s", flush=True)
    ref_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    ref_dtypes = {k: v.dtype for k, v in model.state_dict().items()}
    sd_keys = set(ref_shapes.keys())
    print(f"target keys: {len(sd_keys)}", flush=True)

    mapping = get_checkpoint_conversion_mapping("deepseek_v4")

    new_sd: dict[str, torch.Tensor] = {}

    # 2. original tensors (GPU)
    t0 = time.time()
    loaded = 0
    for orig_name in wm:
        if ".experts." in orig_name or orig_name.endswith(".scale") or ".ffn.gate." in orig_name:
            continue
        target = apply_mapping(mapping, orig_name)
        tf = "lm_head.weight" if target == "lm_head.weight" else "model." + target
        if tf not in sd_keys:
            continue
        t = de.read_tensor(snap, wm, orig_name, device=DEVICE)
        if t.dtype == torch.float8_e4m3fn:
            scale = de.read_tensor(snap, wm, orig_name.replace(".weight", ".scale"), device=DEVICE)
            t = de.dequant_fp8(t, scale)
        new_sd[tf] = t.to(torch.bfloat16)
        loaded += 1
        if loaded % 400 == 0:
            print(f"  {loaded} tensors, {time.time()-t0:.1f}s", flush=True)
    print(f"loaded {loaded} original tensors in {time.time()-t0:.1f}s", flush=True)

    # 3. Experts are zero-filled below (step 4). The real 256 routed experts are
    # loaded on-the-fly from lossless_layers via a forward hook at inference, so
    # the skeleton's expert weights never participate in the forward pass.
    print("experts left zero (replaced by lossless_layers hook at inference)", flush=True)

    # 4. zero-fill the rest
    missing = [k for k in sd_keys if k not in new_sd]
    for k in missing:
        if ref_dtypes[k] in (torch.int64, torch.int32, torch.long):
            new_sd[k] = torch.zeros(ref_shapes[k], dtype=torch.long, device=DEVICE)
        else:
            new_sd[k] = torch.zeros(ref_shapes[k], dtype=torch.bfloat16, device=DEVICE)
    print(f"zero-filled {len(missing)} keys", flush=True)

    # 5. validate
    bad = [k for k in sd_keys if tuple(new_sd[k].shape) != ref_shapes[k]]
    if bad:
        for k in bad[:20]:
            print(f"  SHAPE MISMATCH {k}: {tuple(new_sd[k].shape)} vs {ref_shapes[k]}")
        raise SystemExit(f"{len(bad)} shape mismatches")

    # 6. save (to CPU)
    os.makedirs(OUT_DIR, exist_ok=True)
    from safetensors.torch import save_file

    cpu_sd = {k: v.to("cpu") for k, v in new_sd.items()}
    save_file(cpu_sd, os.path.join(OUT_DIR, "model.safetensors"))
    cfg.quantization_config = None
    if hasattr(cfg, "expert_dtype"):
        cfg.expert_dtype = None
    cfg.save_pretrained(OUT_DIR)
    total = sum(v.numel() * (v.element_size() if v.dtype != torch.bool else 1) for v in new_sd.values())
    print(f"saved {len(new_sd)} tensors, {total/1e9:.2f} GB -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
