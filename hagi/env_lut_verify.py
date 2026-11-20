#!/usr/bin/env python3
"""Independent replay verifier for HAGI Environment Growth v4.6.
Does not import the learning engine. Reconstructs deterministic worlds and checks
all saved probes, exact transition tables, ANFs, and shortest plans.
"""
from __future__ import annotations
from hagi.util import sha
import argparse, json, random
from collections import deque, Counter
from pathlib import Path

N=16
A=4

def b(s,i): return (s>>i)&1

def ob(s): return b(s,0)|(b(s,1)<<1)|((b(s,2)^b(s,3))<<2)

def clock(s): return (s+1)&15

def rotate(s): return ((s<<1)&15)|(s>>3)

def lut2(lut,x,y): return (lut>>(x|(y<<1)))&1

def transition(s, role, f, g, f3=None, ood=False):
    x0,x1,x2,x3=(b(s,i) for i in range(4))
    if role=="clock": return clock(s)
    if role=="shift": return rotate(s)
    if role=="toggle": return s ^ (4 if (x0 and not x1) else 0)
    if role=="coupled" and not ood:
        nx0=x0 ^ lut2(f,x1,x2)
        nx3=x3 ^ lut2(g,nx0,x1)
        return nx0|(x1<<1)|(x2<<2)|(nx3<<3)
    if role=="coupled" and ood:
        idx=x1|(x2<<1)|(x3<<2)
        nx0=x0 ^ ((f3>>idx)&1)
        nx3=x3 ^ lut2(g,nx0,x1)
        return nx0|(x1<<1)|(x2<<2)|(nx3<<3)
    raise ValueError(role)

def reconstruct_world(seed, ood=False):
    rng=random.Random(seed ^ 0xC0FFEE)
    roles=["clock","shift","toggle","coupled"]
    rng.shuffle(roles)
    f=rng.randint(1,14); g=rng.randint(1,14)
    f3=128
    if ood:
        choices=[x for x in range(1,256) if (x.bit_count()&1) and x not in (0x81,0xA5)]
        f3=rng.choice(choices)
    goal=rng.randrange(1,16)
    table=[[transition(s,roles[a],f,g,f3,ood) for a in range(A)] for s in range(N)]
    return roles,f,g,f3,goal,table

def sig_for(table, state, action, suffix):
    s=table[state][action]
    out=[ob(s)]
    ca=next(i for i,r in enumerate(ROLES) if r=="clock")
    for _ in range(suffix):
        s=table[s][ca]; out.append(ob(s))
    return out

def bfs(table, goal):
    q=deque([0]); prev={0:None}; pa={}
    while q:
        s=q.popleft()
        if s==goal: break
        for a in range(A):
            t=table[s][a]
            if t not in prev: prev[t]=s; pa[t]=a; q.append(t)
    if goal not in prev: return None
    out=[]; s=goal
    while prev[s] is not None: out.append(pa[s]); s=prev[s]
    return list(reversed(out))

def verify_plan(table, goal, path, trace):
    s=0
    if path is None or len(path)!=len(trace): return False
    for idx,a in enumerate(path):
        s=table[s][a]
        row=trace[idx]
        if row.get("action")!=a or row.get("predicted_state")!=s or row.get("observed")!=ob(s): return False
    return s==goal and len(path)==len(bfs(table,goal) or [])

def verify_run(run, ood=False):
    seed=run["seed"]
    roles,f,g,f3,goal,table=reconstruct_world(seed,ood)
    clock_a=roles.index("clock")
    checks={"seed":seed,"pass":True,"probes":0,"plan":False,"table_digest":False}
    def fail(msg): checks["pass"]=False; checks.setdefault("errors",[]).append(msg)
    if run.get("clock_action")!=clock_a: fail("wrong clock action")
    if not ood:
        if run.get("status")!="PASS_SAMPLED": fail("wrong status label")
        if run.get("model_evidence")!="SAMPLED_ONLY" or run.get("validation_status")!="PASS_SAMPLED_ONLY": fail("model not marked provisional")
        if run.get("candidate_left")!=1: fail("candidate not unique")
        if sha(table)!=run.get("model_digest"): fail("candidate digest differs from true model")
        if run.get("f")!=f or run.get("g")!=g: fail("latent parameters mismatch")
        for item in run.get("evidence",[]):
            if sig_for(table,item["state"],item["action"],item["suffix"])!=item["obs"]: fail("training evidence mismatch")
            checks["probes"]+=1
        for item in run.get("validation",[]):
            if sig_for(table,item["state"],item["action"],4)!=item["observed"]: fail("holdout validation mismatch")
            if item["observed"]!=item["predicted"]: fail("holdout prediction mismatch")
            checks["probes"]+=1
        if len(run.get("validation",[]))!=6: fail("unexpected holdout count")
        expected_cost=64+sum(x["cost"] for x in run.get("evidence",[]))+sum(x["cost"] for x in run.get("validation",[]))
        if expected_cost!=run.get("learn_interactions"): fail("interaction accounting mismatch")
        if run.get("goal")!=goal: fail("goal mismatch")
        path=bfs(table,goal)
        checks["plan"]=verify_plan(table,goal,run.get("plan"),run.get("execution",[]))
        if not checks["plan"]: fail("plan is not shortest or trace mismatch")
        checks["table_digest"]=True
    else:
        if run.get("status")!="PASS": fail("OOD full-table task not pass")
        if run.get("true_f3")!=f3 or run.get("true_g")!=g: fail("OOD generator params mismatch")
        if run.get("novel_rule_exact") is not True: fail("novel rule not exactly recovered")
        inf=run.get("inferred_novel_rule") or {}
        if inf.get("f3")!=f3 or inf.get("g")!=g or roles[inf.get("action",-1)]!="coupled": fail("inferred mechanism mismatch")
        rows=run.get("rows",[])
        expected={(s,a) for s in range(N) for a in range(A) if a!=clock_a}
        got={(r.get("state"),r.get("action")) for r in rows}
        if got!=expected or len(rows)!=48: fail("incomplete or duplicate full transition table")
        for r in rows:
            if r.get("next_state")!=table[r["state"]][r["action"]]: fail("full transition row mismatch")
        if run.get("fallback_transition_pairs")!=48: fail("wrong transition pair count")
        expected_cost=sum(r["cost"] for r in rows)
        if expected_cost!=run.get("fallback_interactions"): fail("full fallback cost mismatch")
        if run.get("ood_total_interactions") != run.get("active_interactions_before_hold",0)+expected_cost: fail("OOD total interaction accounting mismatch")
        if run.get("grammar_status") not in ("GRAMMAR_MISMATCH","PROVISIONAL_REJECTED_BY_FULL_AUDIT"): fail("unexpected grammar status")
        audit=run.get("sampled_candidate_full_audit")
        if run.get("grammar_status")=="PROVISIONAL_REJECTED_BY_FULL_AUDIT":
            if not audit or audit.get("pass") or audit.get("pairs_checked")!=48: fail("failed to reject sampled candidate by full audit")
            if len(audit.get("mismatches",[]))==0: fail("full audit had no mismatch")
        anfs=run.get("anfs",[])
        if len(anfs)!=A or any(len(x)!=4 for x in anfs): fail("ANF arity")
        for s in range(N):
            for a in range(A):
                predicted=sum((1<<bit_i) for bit_i,terms in enumerate(anfs[a]) if sum(1 for t in terms if (s&int(t))==int(t))&1)
                if predicted!=table[s][a]: fail("ANF value mismatch")
                checks["probes"]+=1
        if run.get("goal")!=goal: fail("goal mismatch")
        checks["plan"]=verify_plan(table,goal,run.get("plan"),run.get("execution_trace",[]))
        if not checks["plan"]: fail("OOD plan mismatch")
        if sha(table)!=run.get("model_digest"): fail("OOD table digest mismatch")
        checks["table_digest"]=True
    return checks

# Used in sig_for; set on each call in verify_bundle.
ROLES=None

def verify_bundle(root: Path):
    global ROLES
    p=root/"results"/"panel20"
    active=json.loads((p/"active_runs.json").read_text())
    random_runs=json.loads((p/"random_runs.json").read_text())
    ood=json.loads((p/"ood_runs.json").read_text())
    active_audits=json.loads((p/"active_exact_audits.json").read_text())
    random_audits=json.loads((p/"random_exact_audits.json").read_text())
    out={"active":[],"random":[],"ood":[],"active_exact_audits":[],"random_exact_audits":[]}
    for target,items,is_ood in (("active",active,False),("random",random_runs,False),("ood",ood,True)):
        for run in items:
            roles,*_=reconstruct_world(run["seed"],is_ood)
            ROLES=roles
            out[target].append(verify_run(run,is_ood))
    for target, audits, runs in (("active_exact_audits",active_audits,active),("random_exact_audits",random_audits,random_runs)):
        for audit,run in zip(audits,runs):
            roles,f,g,f3,goal,table=reconstruct_world(run["seed"],False)
            ok=(audit.get("status")=="PROVEN_EXACT" and audit.get("pairs_checked")==48 and audit.get("total_pairs")==64
                and audit.get("interactions")==600 and audit.get("candidate_digest")==sha(table)
                and audit.get("seed")==run.get("seed"))
            out[target].append({"seed":run["seed"],"pass":ok,"pairs_checked":audit.get("pairs_checked"),
                                "status":audit.get("status"),"errors":[] if ok else ["exact audit record invalid"]})
    summary={"active_pass":sum(x["pass"] for x in out["active"]),"random_pass":sum(x["pass"] for x in out["random"]),
             "ood_pass":sum(x["pass"] for x in out["ood"]),
             "active_exact_audits_pass":sum(x["pass"] for x in out["active_exact_audits"]),
             "random_exact_audits_pass":sum(x["pass"] for x in out["random_exact_audits"]),
             "total":sum(len(v) for v in out.values()),
             "all_pass":all(x["pass"] for k in out for x in out[k]),
             "ood_rule_recoveries":sum(1 for x in ood if x.get("novel_rule_exact")),
             "panel_summary":json.loads((p/"panel_summary.json").read_text())}
    result={"summary":summary,"details":out}
    (root/"results"/"independent_verification.json").write_text(json.dumps(result,indent=2))
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    args=ap.parse_args(); r=verify_bundle(args.root); print(json.dumps(r["summary"],indent=2))
    raise SystemExit(0 if r["summary"]["all_pass"] else 1)
if __name__=="__main__": main()
