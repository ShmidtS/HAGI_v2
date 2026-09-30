import io, json, os, collections
from transformers import AutoTokenizer
BS=chr(92); RAW=BS*2+"Home"+BS+"1"+BS+"raw"
SAMPLES={"ru":"wikipedia_ru.jsonl","en":"slimpajama.jsonl","math":"openwebmath.jsonl","code":"python_instruct.jsonl","ru_web":"oscar_ru.jsonl"}
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
CANDS={"gemma-262k":"google/gemma-4-E2B-it","Qwen3.8-248k":"models/qwen3_8-27b-tokenizer"}
blobs={d:load(f)[:1_500_000] for d,f in SAMPLES.items()}
print("=== после огранения до 32768: токенов на 1000 символов ===")
print("tokenizer".ljust(16)+"".join(d.ljust(9) for d in blobs)+"живых_в_32k")
for lab,repo in CANDS.items():
    tk=AutoTokenizer.from_pretrained(repo,local_files_only=True)
    cnt=collections.Counter(); row=lab.ljust(16)
    for d,b in blobs.items():
        ids=tk.encode(b,add_special_tokens=False); cnt.update(ids)
        row+=f"{len(ids)/(len(b)/1000):.1f}".ljust(9)
    top=cnt.most_common(32768); mass=sum(c for _,c in top)/sum(cnt.values())
    print(row+f"{len(top)}  покрытие={mass*100:.2f}%")
