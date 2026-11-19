#!/usr/bin/env python3
from hagi.fusion_anf import DegreeContract,ProofCarryingANFLearner,ExactMDLLibrary
import random

def make_task(n,d,seed,density=.25):
    rng=random.Random(seed);pool=[s for k in range(d+1) for s in _comb_masks(n,k)];outs=[]
    for _ in range(2):
        t={s for s in pool if rng.random()<density}
        if d>0:t.add(rng.choice([s for s in pool if s.bit_count()==d]))
        outs.append(t)
    def oracle(x):return tuple(sum(1 for s in ts if (s&x)==s)&1 for ts in outs)
    return tuple(frozenset(x) for x in outs),oracle

def _comb_masks(n,k):
    import itertools
    return [sum(1<<i for i in c) for c in itertools.combinations(range(n),k)]

def test_degree_contract_n32():
    terms,oracle=make_task(32,2,1)
    learner=ProofCarryingANFLearner({"test-generator"},max_full_support_inputs=14)
    cand,rep=learner.fit(32,2,oracle,DegreeContract(32,2,"test-generator"))
    assert cand is not None and rep["mode"]=="degree-contract"
    assert rep["query_count"]==1+32+32*31//2==529
    assert learner.symbolic_verify(cand.netlist,cand.outputs)

def test_untrusted_contract_falls_back_full_support():
    terms,oracle=make_task(8,2,2)
    learner=ProofCarryingANFLearner({"trusted"},max_full_support_inputs=10)
    calls=[0]
    def counted(x):
        calls[0]+=1
        return oracle(x)
    cand,rep=learner.fit(8,2,counted,DegreeContract(8,2,"untrusted"))
    assert cand is not None and rep["mode"]=="full-support"
    assert rep["query_count"]==256 and calls[0]==256

def test_misspecified_contract_is_audited_and_escalated():
    terms,oracle=make_task(8,2,3)
    learner=ProofCarryingANFLearner({"test-generator"},max_full_support_inputs=10)
    cand,rep=learner.fit(8,2,oracle,DegreeContract(8,1,"test-generator"),audit_full_support=True)
    assert cand is not None and rep["mode"]=="full-support-escalated"
    assert rep["degree_contract_rejected"]
    assert cand.outputs==terms

def test_misspecified_contract_without_fallback_holds():
    terms,oracle=make_task(8,2,4)
    learner=ProofCarryingANFLearner({"test-generator"},max_full_support_inputs=10)
    cand,rep=learner.fit(8,2,oracle,DegreeContract(8,1,"test-generator"),allow_full_support=False,audit_full_support=True)
    assert cand is None and rep["action"]=="HOLD"

def test_mdl_deduplicates_exact_behavior():
    terms,oracle=make_task(6,2,5)
    learner=ProofCarryingANFLearner({"test-generator"},max_full_support_inputs=10)
    c1,_=learner.fit(6,2,oracle,DegreeContract(6,2,"test-generator"))
    c2,_=learner.fit(6,2,oracle,DegreeContract(6,2,"test-generator"))
    lib=ExactMDLLibrary()
    assert lib.commit(c1)["action"]=="COMMIT"
    assert lib.commit(c2)["action"]=="DEDUP"
    assert len(lib)==1

def main():
    tests=[test_degree_contract_n32,test_untrusted_contract_falls_back_full_support,
           test_misspecified_contract_is_audited_and_escalated,
           test_misspecified_contract_without_fallback_holds,
           test_mdl_deduplicates_exact_behavior]
    for t in tests:t()
    print(f"PASS {len(tests)}/{len(tests)} fusion core tests")
if __name__=="__main__":main()
