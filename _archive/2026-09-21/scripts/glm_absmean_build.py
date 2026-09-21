"""Build GGUF from per-channel absmean ternary (tq1abs_L*.npy) for gate/up/down.
Preserve down as its own tq1abs; test whether HAGI's canonical b1.58 scheme
beats skel8 (PPL 27.5) at same 1.58 bpw.
"""
import numpy as np, os, sys, io, glob, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
from gguf import GGUFReader, GGUFWriter

SRC = 'glm5_gguf/glm5-ternary-skel8.gguf'
DST = 'glm5_gguf/glm5-ternary-absmean.gguf'
TQ1_0 = 34

abs_w = {}
for f in glob.glob('glm5_pod/tq1abs_L*.npy'):
    parts = os.path.basename(f).replace('tq1abs_','').replace('.npy','').split('_')
    blk = parts[0][1:]
    pname = parts[1]
    abs_w[f'blk.{blk}.ffn_{pname}_exps.weight'] = np.load(f)
print(f'{len(abs_w)} absmean tensors', flush=True)

r = GGUFReader(SRC)
src = [(t.name, np.array(t.data, copy=True), t.tensor_type) for t in r.tensors]
arch = r.fields['general.architecture'].contents()
kv = [(k, v.contents(), v.types[0]) for k, v in r.fields.items() if not k.startswith('GGUF.')]
del r; gc.collect()

w = GGUFWriter(DST, arch, use_temp_file=True)
for k, val, typ in kv:
    try: w.add_key_value(k, val, typ)
    except: pass
for name, data, dtype in src:
    if name in abs_w:
        w.add_tensor(name, abs_w[name], raw_dtype=TQ1_0, raw_shape=abs_w[name].shape)
    else:
        w.add_tensor(name, data, raw_dtype=dtype)
    gc.collect()
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f'DONE -> {DST}', flush=True)