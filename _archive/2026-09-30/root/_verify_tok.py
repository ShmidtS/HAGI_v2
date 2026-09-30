import io, json, os, time
from tokenizers import Tokenizer
import gigatoken
BS=chr(92); RAW=BS*2+"Home"+BS+"1"+BS+"raw"
def load(fn,n=1500):
    out=[];p=os.path.join(RAW,fn)
    with io.open(p,"r",encoding="utf-8",errors="replace") as h:
        for line in h:
            if len(out)>=n: break
            try: r=json.loads(line)
            except: continue
            t=r.get("text") or r.get("content") or r.get("prompt") or ""
            if isinstance(t,str) and len(t)>200: out.append(t[:4000])
    return "\n".join(out)
doms={"ru":"wikipedia_ru.jsonl","ru_web":"oscar_ru.jsonl","en":"slimpajama.jsonl",
      "math":"openwebmath.jsonl","code":"python_instruct.jsonl"}
blobs={d:load(f)[:1_500_000] for d,f in doms.items()}
mine=Tokenizer.from_file("data/hagi_32k.tokenizer.json")
gemma=gigatoken.Tokenizer("google/gemma-4-E2B-it")
def enc(tk,b):
    if isinstance(tk,Tokenizer): return len(tk.encode(b).ids)
    return len(tk.encode(b))
print("=== токенов на 1000 символов (реальный корпус) ===")
print("tokenizer".ljust(14)+"".join(d.ljust(10) for d in doms)+"vocab")
for lab,tk in (("HAGI-32k",mine),("gemma-262k",gemma)):
    row=lab.ljust(14)
    for d,b in blobs.items():
        row+=f"{enc(tk,b)/(len(b)/1000):.1f}".ljust(10)
    v=mine.get_vocab_size() if isinstance(mine,Tokenizer) else 262144
    print(row+str(v))
print("\n=== ru: токенов на слово ===")
ru=blobs["ru"]; w=len(ru.split())
for lab,tk in (("HAGI-32k",mine),("gemma-262k",gemma)):
    print(f"{lab:12s} {enc(tk,ru)/w:.2f} ток/слово")
