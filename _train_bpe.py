import time, gigatoken
t0=time.time()
print("обучаю 32768 merges на 517M символов...", flush=True)
res = gigatoken.train_bpe("data/bpe_corpus.txt", 32768, [], "huggingface")
print(f"готово за {time.time()-t0:.0f}s | тип={type(res).__name__} len={len(res)}", flush=True)
for i,x in enumerate(res):
    print(f"  [{i}] {type(x).__name__} {str(x)[:110]}")
