import json, time, gigatoken
t0=time.time()
vocab, merges = gigatoken.train_bpe("data/bpe_corpus.txt", 32768, [], "huggingface")
print(f"обучено за {time.time()-t0:.0f}s | vocab={len(vocab)} merges={len(merges)}")
# первые merge'ы — доказательство, что upweighting кириллицы сработал
ru = sum(1 for a,b in merges[:4000] if any(c>0xC0 for c in a+b))
print(f"кириллических merge в первых 4000: {ru} ({ru/4000*100:.1f}%)")
tk = gigatoken.Tokenizer.from_json if hasattr(gigatoken.Tokenizer,'from_json') else None
print("from_json есть:", tk is not None)
import pickle
with open("data/hagi_32k.pkl","wb") as f: pickle.dump((vocab,merges), f)
print("сохранено data/hagi_32k.pkl")
