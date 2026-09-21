"""Split GGUF: layers 0..K_ternary ternary (skel8), rest fp16 from original safetensors.
Localizes how many ternary layers the cascade tolerates before collapse.
Usage: python scripts/glm_split_build.py [K_ternary]
K_ternary = number of layers (3..14, i.e. absolute block idx) to keep ternary.
"""
import numpy as np, os, sys, io, json, time, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch
D, I, E = 4096, 2048, 288

K_MAX_TEN = int(sys.argv[1]) if len(sys.argv) > 1 else 14  # blocks 3..K_MAX_TEN ternary
SRC = 'glm5_gguf/glm5-ternary-skel8.gguf'
DST = f'glm5_gguf/glm5-ternary-split-K{K_MAX_TEN}.gguf'

HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']

def load_layer_fp16(il):
    """Load all three expert matrices (gate,up,down) fp32 for layer il."""
    from safetensors import safe_open
    out = {}
    res_gate = torch.empty((E, I, D), dtype=torch.float16)
    res_up = torch.empty((E, I, D), dtype=torch.float16)
    res_down = torch.empty((E, D, I), dtype=torch.float16)
    for e in range(E):
        for pname, arr, o, i in [('gate_proj', res_gate, I, D), ('up_proj', res_up, I, D), ('down_proj', res_down, D, I)]:
            key = f'model.language_model.layers.{il}.mlp.experts.{e}.{pname}.weight'
            with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
                w = sf.get_tensor(key).to(torch.float32)
                s = sf.get_tensor(key.replace('.weight','.weight_scale_inv')).to(torch.float32)
                arr[e] = (w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(o,i).to(torch.float16)
    return res_gate.numpy(), res_up.numpy(), res_down.numpy()

from gguf import GGUFReader, GGUFWriter
r = GGUFReader(SRC)
by = {t.name: t for t in r.tensors}
# metadata
arch = r.fields['general.architecture'].contents()
kv = [(k, v.contents(), v.types[0]) for k, v in r.fields.items() if not k.startswith('GGUF.')]

# Determine which blocks are ternary (3..K_MAX_TEN) vs fp16 (K_MAX_TEN+1..45)
tern_blocks = set(range(3, 43))
print(f"ternary blocks: {sorted(tern_blocks)}", flush=True)

# We'll rewrite ALL tensors: copy from skel8 for ternary + non-expert, load fp16 for expert tensors of fp16 blocks
# Load fp16 for all non-ternary block expert tensors
print("loading fp16 teacher layers...", flush=True)
fp16_tensors = {}
for il in range(3, 46):
    if il in tern_blocks:
        continue
    if il == 45:  # block 45 has no cal/fp16? it exists in safetensors. load it.
        pass
    t0 = time.time()
    g, u, d = load_layer_fp16(il)
    # store in LOGICAL layout matching GGUF reader t.shape:
    # gate/up logical [D,I,E]  -> g is (E,I,D) => transpose(2,1,0)
    # down  logical [I,D,E]  -> d is (E,D,I) => transpose(2,1,0)
    fp16_tensors[f'blk.{il}.ffn_gate_exps.weight'] = np.ascontiguousarray(g.transpose(2,1,0))
    fp16_tensors[f'blk.{il}.ffn_up_exps.weight'] = np.ascontiguousarray(u.transpose(2,1,0))
    fp16_tensors[f'blk.{il}.ffn_down_exps.weight'] = np.ascontiguousarray(d.transpose(2,1,0))  # (I,D,E)
    print(f"  L{il}: {time.time()-t0:.0f}s", flush=True)
    del g,u,d; gc.collect()

print("writing GGUF...", flush=True)
w = GGUFWriter(DST, arch, use_temp_file=True)
for k, val, typ in kv:
    try: w.add_key_value(k, val, typ)
    except: pass

# Need full tensor list: iterate skel8 tensors, but for fp16 blocks override expert tensors
processed_expert = set()
for t in r.tensors:
    name = t.name
    # parse block
    parts = name.split('.')  # ['blk','3','ffn_gate_exps','weight']
    try:
        blk = int(parts[1])
    except:
        blk = -1
    is_expert = ('ffn_gate_exps' in name or 'ffn_up_exps' in name or 'ffn_down_exps' in name)
    
    if is_expert and blk in tern_blocks:
        # ternary: copy from skel8 raw packed
        data = np.array(t.data, copy=True)
        w.add_tensor(name, data, raw_dtype=34, raw_shape=data.shape)
        processed_expert.add(name)
    elif name in fp16_tensors and not is_expert:
        pass  # not expert, skip (handled below)
    elif is_expert and blk not in tern_blocks and name in fp16_tensors:
        # fp16 expert: logical shape differs from packed! must add with logical shape
        logical = list(t.shape)  # e.g. [4096,2048,288]
        data = fp16_tensors[name]
        # already in LOGICAL layout (D,I,E) or (I,D,E); pass with that shape
        w.add_tensor(name, data.astype(np.float16) if data.dtype != np.float16 else data,
                     raw_dtype=1, raw_shape=data.shape)
        processed_expert.add(name)
    elif not is_expert:
        # non-expert tensor: copy dtype/bytes as-is
        data = np.array(t.data, copy=True)
        w.add_tensor(name, data, raw_dtype=t.tensor_type, raw_shape=data.shape)
    else:
        print(f"  WARN unhandled {name}", flush=True)

# handle any expert not processed (shouldn't happen)
missing = [n for n in fp16_tensors if n not in processed_expert]
print(f"missing expert writes: {len(missing)}", flush=True)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"DONE -> {DST}", flush=True)