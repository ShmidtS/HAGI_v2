"""Extract TQ1 tensors from skel8 GGUF → tq1raw_L*.npy"""
import numpy as np, os, gc
from gguf import GGUFReader

os.chdir(r'C:/HAGI_v2')
GGUF = 'glm5_gguf/glm5-ternary-skel8.gguf'
OUT = 'glm5_pod'
os.makedirs(OUT, exist_ok=True)

r = GGUFReader(GGUF)
count = 0
for t in r.tensors:
    name = t.name
    # ffn_gate_exps.weight → tq1raw_L{blk}_gate.npy
    # ffn_up_exps.weight → tq1raw_L{blk}_up.npy
    if 'ffn_gate_exps' in name or 'ffn_up_exps' in name:
        blk = name.split('.')[1]  # '3'
        pname = 'gate' if 'gate' in name else 'up'
        fname = f'tq1raw_L{blk}_{pname}.npy'
        data = np.array(t.data, copy=True)
        np.save(f'{OUT}/{fname}', data)
        print(f'{fname}: {data.shape}', flush=True)
        count += 1

print(f'DONE: {count} tensors extracted', flush=True)