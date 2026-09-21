"""Build GGUF replacing gate/up with activation-aware ternary, keep skel8 down.
Uses tq1aw_L{layer}_{gate,up}.npy produced by glm_actaware_all.py.
"""
import numpy as np, os, sys, io, glob, time, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
from gguf import GGUFReader, GGUFWriter

SRC = 'glm5_gguf/glm5-ternary-skel8.gguf'
DST = 'glm5_gguf/glm5-ternary-actaware.gguf'
TQ1_0 = 34

aw = {}
for f in glob.glob('glm5_pod/tq1aw_L*.npy'):
    parts = os.path.basename(f).replace('tq1aw_','').replace('.npy','').split('_')
    blk = parts[0][1:]  # '3'
    pname = parts[1]    # gate/up
    aw[f'blk.{blk}.ffn_{pname}_exps.weight'] = np.load(f)
print(f'{len(aw)} activation-aware tensors', flush=True)

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
    if name in aw:
        w.add_tensor(name, aw[name], raw_dtype=TQ1_0, raw_shape=aw[name].shape)
    else:
        w.add_tensor(name, data, raw_dtype=dtype)
    gc.collect()
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f'DONE -> {DST}', flush=True)