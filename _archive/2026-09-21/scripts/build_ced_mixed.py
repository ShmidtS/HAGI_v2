"""Build mixed-precision GGUF: TQ1 on most layers, fp16 on last N decoder layers.
Hypothesis: clean decoder layers at end fix cascade collapse from TQ1 encoder."""
import numpy as np, os, sys, io, json, time, gc, glob
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch

D, I, E = 4096, 2048, 288
HF = r'//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']

DECODER_LAYERS = [43, 44, 45]  # Last 3 layers = fp16 "decoder"
SRC = 'glm5_gguf/glm5-ternary-skel8.gguf'
DST = 'glm5_gguf/glm5-ternary-ced.gguf'

def load_expert_fp16(il, pname):
    """Load original fp8 → fp32 → fp16 for one expert layer"""
    from safetensors import safe_open
    kt = f'model.language_model.layers.{il}.mlp.experts.{{e}}.{pname}_proj.weight'
    
    # Load all 288 experts
    W = torch.empty((E, I, D), dtype=torch.float16, device='cpu')
    for e in range(E):
        key = kt.format(e=e)
        shard = _wm[key]
        with safe_open(f'{HF}/{shard}', 'pt', 'cpu') as sf:
            w = sf.get_tensor(key).to(torch.float32)
            s = sf.get_tensor(key.replace('.weight', '.weight_scale_inv'))
            W_e = (w.reshape(s.shape[0], 128, s.shape[1], 128)
                   * s[:, None, :, None].to(torch.float32))
            W[e] = W_e.reshape(I, D).to(torch.float16)
    return W.numpy()

print("Loading fp16 weights for decoder layers...", flush=True)
fp16_tensors = {}
for il in DECODER_LAYERS:
    for pname in ('gate', 'up'):
        key = f'blk.{il}.ffn_{pname}_exps.weight'
        t0 = time.time()
        data = load_expert_fp16(il, pname)
        fp16_tensors[key] = data
        dt = time.time() - t0
        print(f"  {key}: {data.shape} = {data.nbytes/1e9:.1f}GB in {dt:.0f}s", flush=True)
        del data; gc.collect()

print(f"\nBuilding GGUF...", flush=True)
from gguf import GGUFReader, GGUFWriter
import gc as gc_mod

r = GGUFReader(SRC)
tensors_src = [(t.name, np.array(t.data, copy=True), t.tensor_type) for t in r.tensors]
arch = r.fields['general.architecture'].contents()
kv = [(k, v.contents(), v.types[0]) for k, v in r.fields.items() if not k.startswith('GGUF.')]
del r; gc_mod.collect()

w = GGUFWriter(DST, arch, use_temp_file=True)
for k, val, typ in kv:
    try: w.add_key_value(k, val, typ)
    except: pass

for name, data, dtype in tensors_src:
    if name in fp16_tensors:
        # Replace TQ1 with fp16 — shape must be logical (4096, 2048, 288)
        fp16_flat = fp16_tensors[name]  # (E, I, D) = (288, 2048, 4096)
        # GGUF stores as (D, I, E) = (4096, 2048, 288) for tensor_type=1
        fp16_gguf = np.ascontiguousarray(fp16_flat.transpose(2, 1, 0))  # (4096, 2048, 288)
        w.add_tensor(name, fp16_gguf, raw_dtype=1, raw_shape=fp16_gguf.shape)
        print(f"  REPLACED {name}: TQ1 → fp16 ({fp16_gguf.shape})", flush=True)
    else:
        w.add_tensor(name, data, raw_dtype=dtype)
    gc_mod.collect()

w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"DONE -> {DST}", flush=True)

del fp16_tensors; gc_mod.collect()
print("GGUF built, testing generation...", flush=True)

import subprocess
result = subprocess.run(
    ['llama-glm5/build-vulkan/bin/llama.exe', 'cli',
     '-m', DST, '-ngl', '99', '-c', '4096',
     '--temp', '0.7', '-n', '60',
     '-p', 'Привет'],
    capture_output=True, text=True, timeout=300, cwd='C:/HAGI_v2'
)
out = (result.stdout or "")[-500:].strip()
err = (result.stderr or "")[-300:].strip()
print(f"\nGENERATION:", out, flush=True)
if err and ('error' in err.lower() or 'fail' in err.lower()):
    print(f"STDERR:", err, flush=True)
print("DONE", flush=True)