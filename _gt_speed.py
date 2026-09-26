import io, json, os, time
import gigatoken
BS=chr(92); RAW=BS*2+"Home"+BS+"1"+BS+"raw"
def load(fn,n=1200):
    out=[];p=os.path.join(RAW,fn)
    with io.open(p,"r",encoding="utf-8",errors="replace") as h:
        for line in h:
            if len(out)>=n: break
            try: r=json.loads(line)
            except: continue
            t=r.get("text") or r.get("content") or r.get("prompt") or ""
            if isinstance(t,str) and len(t)>200: out.append(t[:4000])
    return out
docs=[]
for fn in ("wikipedia_ru.jsonl","slimpajama.jsonl","openwebmath.jsonl","python_instruct.jsonl"):
    docs += load(fn)
total=sum(len(d) for d in docs)
print(f"документов {len(docs)}, символов {total/1e6:.1f}M")
tk=gigatoken.Tokenizer("google/gemma-4-E2B-it")
f=tk.encode_batch_list
t0=time.time(); f(docs); dt=time.time()-t0
print(f"gigatoken (gemma-262k): {total/1e6/dt:.2f} Mсимв/с  ({dt:.1f} c на {total/1e6:.1f}M)")
from transformers import AutoTokenizer
h=AutoTokenizer.from_pretrained("google/gemma-4-E2B-it",local_files_only=True)
t0=time.time(); [h.encode(d,add_special_tokens=False) for d in docs]; dt2=time.time()-t0
print(f"HF tokenizers (gemma):   {total/1e6/dt2:.2f} Mсимв/с  ({dt2:.1f} c)")
print(f"ускорение gigatoken/HF:  {dt2/dt:.2f}x")
