#!/usr/bin/env python3
"""Reusable proof-carrying ANF front-end for HAGI Fusion.

This module is an optional adapter layer, not a patch to upstream HAGI_v2.
It supports two honest modes:
  1) trusted degree contract -> exact unisolvent skeleton interpolation;
  2) no contract -> full-support Mobius transform only up to a configured cap,
     otherwise HOLD.

The compiler emits only constant/AND/XOR Venn-LUT gates. An independent
symbolic transfer checker verifies the generated network exactly in ANF.
"""
from __future__ import annotations
import hashlib,heapq,itertools,json
from dataclasses import dataclass
from typing import Callable,Iterable

@dataclass(frozen=True)
class DegreeContract:
    n_inputs:int
    max_anf_degree:int
    issuer:str
    version:int=1

    def validate(self, trusted_issuers:set[str])->tuple[bool,str]:
        if self.n_inputs<1:return False,"n_inputs must be positive"
        if not 0<=self.max_anf_degree<=self.n_inputs:return False,"degree bound outside [0,n]"
        if self.issuer not in trusted_issuers:return False,"untrusted degree-contract issuer"
        return True,"trusted contract"

@dataclass
class ANFCandidate:
    n_inputs:int
    outputs:tuple[frozenset[int],...]
    netlist:dict
    query_count:int
    mode:str
    certificate:dict
    metrics:dict

    def semantic_digest(self)->str:
        payload={"n":self.n_inputs,"outputs":[sorted(x) for x in self.outputs]}
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

class ProofCarryingANFLearner:
    def __init__(self,trusted_issuers:Iterable[str]=(),max_full_support_inputs:int=14):
        self.trusted_issuers=set(trusted_issuers)
        self.max_full_support_inputs=max_full_support_inputs

    @staticmethod
    def subsets_upto(n:int,d:int)->list[int]:
        return [sum(1<<i for i in c) for k in range(d+1) for c in itertools.combinations(range(n),k)]

    @staticmethod
    def _interpolate(n:int,d:int,oracle:Callable[[int],tuple[int,...]],n_outputs:int):
        terms=[set() for _ in range(n_outputs)];labels=[]
        for s in ProofCarryingANFLearner.subsets_upto(n,d):
            y=tuple(int(v)&1 for v in oracle(s));labels.append((s,*y))
            t=(s-1)&s if s else -1;proper=[]
            while t>=0:
                proper.append(t)
                if t==0:break
                t=(t-1)&s
            for o in range(n_outputs):
                v=y[o]
                for sub in proper:
                    if sub in terms[o]:v^=1
                if v:terms[o].add(s)
        return tuple(frozenset(t) for t in terms),labels

    @staticmethod
    def _full_mobius(n:int,oracle:Callable[[int],tuple[int,...]],n_outputs:int):
        N=1<<n
        observed=[tuple(int(v)&1 for v in oracle(x)) for x in range(N)]
        if any(len(row)!=n_outputs for row in observed):
            raise ValueError("oracle output arity mismatch")
        tables=[[observed[x][o] for x in range(N)] for o in range(n_outputs)]
        terms=[]
        for vals in tables:
            a=list(vals)
            for bit in range(n):
                step=1<<bit
                for x in range(N):
                    if x&step:a[x]^=a[x^step]
            terms.append(frozenset(i for i,v in enumerate(a) if v))
        return tuple(terms),N

    @staticmethod
    def compile(n:int,outputs:tuple[frozenset[int],...])->dict:
        gates=[];depth=[0]*n;cache={};const1=None;const0=None
        def add(a,b,mask):
            w=n+len(gates)
            if not(0<=a<w and 0<=b<w):raise ValueError("compiler produced forward edge")
            gates.append({"a":a,"b":b,"mask":mask});depth.append(1+max(depth[a],depth[b]));return w
        def monomial(s):
            nonlocal const1
            if s in cache:return cache[s]
            if s==0:
                if const1 is None:const1=add(0,0,15)
                cache[s]=const1;return const1
            if s&(s-1)==0:
                w=s.bit_length()-1;cache[s]=w;return w
            bits=[1<<i for i in range(n) if (s>>i)&1];mid=len(bits)//2
            a=sum(bits[:mid]);b=sum(bits[mid:])
            w=add(monomial(a),monomial(b),8);cache[s]=w;return w
        outs=[]
        for out_terms in outputs:
            leaves=[monomial(s) for s in sorted(out_terms)]
            if not leaves:
                if const0 is None:const0=add(0,0,0)
                out=const0
            elif len(leaves)==1:out=leaves[0]
            else:
                heap=[(depth[w],w) for w in leaves];heapq.heapify(heap)
                while len(heap)>1:
                    da,a=heapq.heappop(heap);db,b=heapq.heappop(heap)
                    w=add(a,b,6);heapq.heappush(heap,(1+max(da,db),w))
                out=heap[0][1]
            outs.append(out)
        nw=n+len(gates);fan=[0]*nw
        for g in gates:fan[g["a"]]+=1;fan[g["b"]]+=1
        for o in outs:fan[o]+=1
        wb=max(1,(nw-1).bit_length());ob=max(1,nw.bit_length())
        return {"n_inputs":n,"gates":gates,"outs":outs,
                "metrics":{"gate_count":len(gates),"depth":max((depth[o] for o in outs),default=0),
                           "max_fanout":max(fan,default=0),
                           "program_bits":len(gates)*(4+2*wb)+len(outs)*ob}}

    @staticmethod
    def symbolic_verify(netlist:dict,expected:tuple[frozenset[int],...])->bool:
        n=netlist["n_inputs"];wires=[frozenset({1<<i}) for i in range(n)]
        for i,g in enumerate(netlist["gates"]):
            lim=n+i
            if not(0<=g["a"]<lim and 0<=g["b"]<lim):return False
            a=wires[g["a"]];b=wires[g["b"]];m=g["mask"]
            if m==15:z=frozenset({0})
            elif m==0:z=frozenset()
            elif m==6:z=a.symmetric_difference(b)
            elif m==8:
                if len(a)!=1 or len(b)!=1:return False
                z=frozenset({next(iter(a))|next(iter(b))})
            else:return False
            wires.append(z)
        if any(o<0 or o>=len(wires) for o in netlist["outs"]):return False
        return tuple(wires[o] for o in netlist["outs"])==expected

    def fit(self,n_inputs:int,n_outputs:int,oracle:Callable[[int],tuple[int,...]],
            contract:DegreeContract|None=None,allow_full_support:bool=True,
            audit_full_support:bool=False)->tuple[ANFCandidate|None,dict]:
        if n_inputs<1 or n_outputs<1:raise ValueError("input/output arity must be positive")
        if contract is not None:
            valid,reason=contract.validate(self.trusted_issuers)
            if valid and contract.n_inputs==n_inputs:
                d=contract.max_anf_degree
                outputs,labels=self._interpolate(n_inputs,d,oracle,n_outputs)
                net=self.compile(n_inputs,outputs)
                basis=[x[0] for x in labels]
                label_bytes=bytes(v for row in labels for v in row[1:])
                basis_hash=hashlib.sha256(",".join(map(str,basis)).encode()).hexdigest()
                label_hash=hashlib.sha256(label_bytes).hexdigest()
                cert={"kind":"trusted-degree-unisolvent","issuer":contract.issuer,
                      "contract_version":contract.version,"max_anf_degree":d,
                      "basis_size":len(basis),"basis_points_sha256":basis_hash,
                      "labels_sha256":label_hash,"unit_triangular":True,
                      "conditional_claim":"unique function in the declared degree<=d ANF class"}
                if not self.symbolic_verify(net,outputs):return None,{"action":"HOLD","reason":"symbolic netlist verification failed"}
                audit_checks=0
                if audit_full_support and n_inputs<=self.max_full_support_inputs:
                    mismatch=False
                    for x in range(1<<n_inputs):
                        predicted=tuple(sum(1 for s in terms if (s&x)==s)&1 for terms in outputs)
                        if predicted!=tuple(int(v)&1 for v in oracle(x)):
                            mismatch=True;break
                        audit_checks+=1
                    if mismatch:
                        if not allow_full_support:
                            return None,{"action":"HOLD","reason":"degree contract failed full-support audit","audit_checks":audit_checks}
                        full_outputs,q=self._full_mobius(n_inputs,oracle,n_outputs)
                        full_net=self.compile(n_inputs,full_outputs)
                        if not self.symbolic_verify(full_net,full_outputs):
                            return None,{"action":"HOLD","reason":"fallback symbolic verification failed"}
                        full_cert={"kind":"full-support-mobius","full_support":True,"queries":q,
                                   "reason":"degree contract failed audit; escalated to full support"}
                        cand=ANFCandidate(n_inputs,full_outputs,full_net,q,"full-support-escalated",full_cert,full_net["metrics"])
                        return cand,{"action":"COMMIT_CANDIDATE","mode":"full-support-escalated","query_count":q,"support_size":q,"degree_contract_rejected":True}
                cert["audit_full_support"] = bool(audit_full_support and n_inputs<=self.max_full_support_inputs)
                cert["audit_rows_checked"] = audit_checks
                cand=ANFCandidate(n_inputs,outputs,net,len(labels),"degree-contract",cert,net["metrics"])
                return cand,{"action":"COMMIT_CANDIDATE","mode":"degree-contract","query_count":len(labels),"support_size":1<<n_inputs,"audit_rows_checked":audit_checks}
            reason=reason if not valid else "contract input arity mismatch"
        else:reason="no degree contract"
        if not allow_full_support or n_inputs>self.max_full_support_inputs:
            return None,{"action":"HOLD","reason":reason+"; exact full-support fallback unavailable"}
        outputs,q=self._full_mobius(n_inputs,oracle,n_outputs)
        net=self.compile(n_inputs,outputs)
        if not self.symbolic_verify(net,outputs):return None,{"action":"HOLD","reason":"full-support symbolic netlist verification failed"}
        cert={"kind":"full-support-mobius","full_support":True,"queries":q,
              "claim":"exact on the full Boolean cube because every truth-table row was consumed"}
        cand=ANFCandidate(n_inputs,outputs,net,q,"full-support",cert,net["metrics"])
        return cand,{"action":"COMMIT_CANDIDATE","mode":"full-support","query_count":q,"support_size":q}

class ExactMDLLibrary:
    """Exact semantic deduplication; keeps the cheapest circuit per behavior."""
    def __init__(self):self._items={}
    def commit(self,candidate:ANFCandidate)->dict:
        if not ProofCarryingANFLearner.symbolic_verify(candidate.netlist,candidate.outputs):
            return {"action":"HOLD","reason":"candidate proof invalid"}
        key=candidate.semantic_digest();old=self._items.get(key)
        if old is None:
            self._items[key]=candidate
            return {"action":"COMMIT","digest":key,"library_size":len(self._items)}
        old_score=(old.metrics["program_bits"],old.metrics["depth"],old.metrics["gate_count"])
        new_score=(candidate.metrics["program_bits"],candidate.metrics["depth"],candidate.metrics["gate_count"])
        if new_score<old_score:self._items[key]=candidate
        return {"action":"DEDUP","digest":key,"library_size":len(self._items),"kept":"new" if new_score<old_score else "old"}
    def __len__(self):return len(self._items)
