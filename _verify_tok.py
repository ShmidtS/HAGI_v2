import io, json, os, time
import gigatoken
BS=chr(92); RAW=BS*2+"Home"+BS+"1"+BS+"raw"
payload=open("data/hagi_32k.tiktoken.json",encoding="utf-8").read()
mine=gigatoken.Tokenizer.from_json(payload)
gemma=gigatoken.Tokenizer("google/gemma-4-E2B-it")
s='Съешь же ещё этих мягких французских булок, да выпей чаю. Кинза на укус.'
ids=mine.encode(s)
print("vocab", mine.vocab_size if hasattr(mine,'vocab_size') else len(mine.vocab))
print("ru: %d симв -> %d ток (%.2f ток/слово)"%(len(s),len(ids),len(ids)/len(s.split())))
print("roundtrip ok:", mine.decode(ids).strip()==s.strip())
def load(fn,n=1200):
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
print("\n=== токенов на 1000 символов ===")
print("tokenizer".ljust(14)+"".join(d.ljust(10) for d in doms))
for lab,tk in (("HAGI-32k",mine),("gemma-262k",gemma)):
    row=lab.ljust(14)
    for d,f in doms.items():
        b=load(f)[:1_500_000]; n=len(tk.encode(b))
        row+=f"{n/(len(b)/1000):.1f}".ljust(10)
    print(row)
print("\n=== скорость (gigatoken, одинаковый путь) ===")
docs=[]
for f in doms.values(): docs += load(f,400).split("\n")
tot=sum(len(d) for d in docs)
for lab,tk in (("HAGI-32k",mine),("gemma-262k",gemma)):
    t0=time.time(); tk.encode_batch_list(docs); dt=time.time()-t0
    print(f"{lab:12s} {tot/1e6/dt:6.2f} Mсимв/с")
