"""Streaming GGUF split: skel8 ternary + last K layers' experts fp16.
FIXED layout: do NOT transpose fp16 buffers. Store as (E, I, D) for gate/up,
(E, D, I) for down, raw_shape = that shape. llama.cpp reads as reversed.
Usage: python scripts/glm_split_stream.py <K_fp16_layers>
"""
import numpy as np, os, sys, io, json, time, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch
D, I, E = 4096, 2048, 288
K_FP16 = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SRC = 'glm5_gguf/glm5-ternary-skel8.gguf'
DST = f'glm5_gguf/glm5-ternary-split-K{K_FP16}.gguf'
HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']

all_blocks = list(range(3, 46))
fp16_blocks = set(all_blocks[-K_FP16:])
tern_blocks = set(all_blocks) - fp16_blocks
print(f"ternary: {sorted(tern_blocks)} | fp16: {sorted(fp16_blocks)}", flush=True)

def load_layer_fp16(il):
    """Return dict name->data for gate/up/down fp16.
    gate/up: shape (E, I, D) → raw_shape that → logical [D, I, E].
    down:    shape (E, D, I) → raw_shape that → logical [I, D, E].
    """
    from safetensors import safe_open
    g = np.empty((E, I, D), np.float16)
    u = np.empty((E, I, D), np.float16)
    d = np.empty((E, D, I), np.float16)
    for e in range(E):
        for pname, store, o, i in [('gate_proj', g, I, D), ('up_proj', u, I, D), ('down_proj', d, D, I)]:
            key = f'model.language_model.layers.{il}.mlp.experts.{e}.{pname}.weight'
            with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
                w = sf.get_tensor(key).to(torch.float32)
                s = sf.get_tensor(key.replace('.weight','.weight_scale_inv')).to(torch.float32)
                # fp16 teacher stores (o, i) in native format
                store[e] = (w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(o,i).numpy().astype(np.float16)
    return {
        f'blk.{il}.ffn_gate_exps.weight': g,  # (E, I, D) → logical [D, I, E]
        f'blk.{il}.ffn_up_exps.weight':   u,
        f'blk.{il}.ffn_down_exps.weight': d,  # (E, D, I) → logical [I, D, E]
    }

from gguf import GGUFReader, GGUFWriter
r = GGUFReader(SRC)
arch = r.fields['general.architecture'].contents()
kv = [(k, v.contents(), v.types[0]) for k, v in r.fields.items() if not k.startswith('GGUF.')]

w = GGUFWriter(DST, arch, use_temp_file=True)
for k, val, typ in kv:
    try: w.add_key_value(k, val, typ)
    except: pass

print("loading fp16 teacher layers...", flush=True)
for blk in sorted(fp16_blocks):
    t0 = time.time()
    bufs = load_layer_fp16(blk)
    for name, data in bufs.items():
        w.add_tensor(name, data, raw_dtype=1, raw_shape=data.shape)  # GGML_TYPE_F16=1
    print(f"  L{blk}: {time.time()-t0:.0f}s", flush=True)

# Write ternary & other tensors from memmap
n_tern = 0
for t in r.tensors:
    name = t.name
    parts = name.split('.')
    try: blk = int(parts[1])
    except: blk = -1
    is_expert = name.endswith('_exps.weight') and ('ffn' in name)
    if is_expert and blk in tern_blocks:
        w.add_tensor(name, t.data, raw_dtype=34, raw_shape=t.data.shape)
        n_tern += 1
    elif not is_expert:
        w.add_tensor(name, t.data, raw_dtype=t.tensor_type, raw_shape=t.data.shape)
    gc.collect()

w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"ternary experts written: {n_tern}, fp16 layers: {sorted(fp16_blocks)} -> {DST}", flush=True)
print("DONE", flush=True)