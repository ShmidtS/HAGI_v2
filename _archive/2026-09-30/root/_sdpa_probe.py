import torch, time
torch.manual_seed(0)
dev="cuda"
B,H,T,D=8,18,1024,64
q=torch.randn(B,H,T,D,device=dev,dtype=torch.bfloat16)
k=torch.randn_like(q); v=torch.randn_like(q)
mask_bool=~torch.triu(torch.ones(T,T,device=dev,dtype=torch.bool),1)
bias=torch.triu(torch.full((T,T),float('-inf'),device=dev,dtype=torch.bfloat16),1)
F=torch.nn.functional
def bench(fn,n=20,warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
print("=== forward only ===")
with torch.no_grad():
    print("  mask bool :", round(bench(lambda: F.scaled_dot_product_attention(q,k,v,attn_mask=mask_bool)),2),"ms")
    print("  mask bf16 :", round(bench(lambda: F.scaled_dot_product_attention(q,k,v,attn_mask=bias)),2),"ms")
    print("  is_causal :", round(bench(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True)),2),"ms")
    print("  no mask   :", round(bench(lambda: F.scaled_dot_product_attention(q,k,v)),2),"ms")
print("=== forward+backward ===")
for lab,fn in (("mask bool",lambda qk: F.scaled_dot_product_attention(qk,k,v,attn_mask=mask_bool)),
               ("is_causal",lambda qk: F.scaled_dot_product_attention(qk,k,v,is_causal=True)),
               ("no mask",  lambda qk: F.scaled_dot_product_attention(qk,k,v))):
    q2=q.clone().requires_grad_(True)
    def fb(fn=fn,q2=q2):
        o=fn(q2); o.float().sum().backward()
    print(f"  {lab:10s}:", round(bench(fb),2),"ms")
