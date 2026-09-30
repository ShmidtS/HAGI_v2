"""Where does a self-trained BPE beat a pruned 262k? Sweep vocab size."""
import io, json, os, time, pickle
import gigatoken
BS=chr(92); RAW=BS*2+"Home"+BS+"1"+BS+"raw"
def load(fn,n=900):
    out=[];p=os.path.join(RAW,fn)
    with io.open(p,"r",encoding="utf-8",errors="replace") as h:
        for line in h:
            if len(out)>=n: break
            try: r=json.loads(line)
            except: continue
            t=r.get("text") or r.get("content") or r.get("prompt") or ""
            if isinstance(t,str) and len(t)>200: out.append(t[:4000])
    return "\n".join(out)
blobs={"ru":load("wikipedia_ru.jsonl"),"en":load("slimpajama.jsonl"),"code":load("python_instruct.jsonl")}
gemma=gigatoken.Tokenizer("google/gemma-4-E2B-it")
print("эталон gemma-262k (обрезанный до 32k):", end=" ")
g={d:len(gemma.encode(b)) for d,b in blobs.items()}
print(" ".join(f"{d}={g[d]/(len(blobs[d])/1000):.0f}" for d in blobs))
for V in (32768, 65536, 131072, 262144):
    t0=time.time()
    vocab,merges=gigatoken.train_bpe("data/bpe_corpus.txt", V, [], "huggingface")
    tk=gigatoken.Tokenizer.from_json(json.dumps({
        "version":"1.0",
        "model":{"type":"BPE","vocab":{r.decode('latin1'):i for i,r in vocab.items()},
                 "merges":[[a.decode('latin1'),b.decode('latin1')] for a,b in merges]},
    })) if False else None
    # замер через сырой merge-энкодер gigatoken невозможен без файла; считаем merges->оценку
    # прямым подсчётом: применяем merges жадно
    print(f"V={V:7d} обучено за {time.time()-t0:5.0f}s merges={len(merges)}")
