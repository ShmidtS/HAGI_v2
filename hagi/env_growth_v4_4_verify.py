#!/usr/bin/env python3
"""Independent verifier for HAGI Environment Growth v4.4.

Does not import the learner/agent module. Recomputes hidden transition tables from
serialized audit facts, checks all plans, query traces, model mappings, checkpoints,
and tamper cases.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import deque
from pathlib import Path

N=32; A=4
MODEL_BANK=[]

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def next_state(role, s, p):
    if role == "clock": return (s+1)%N
    if role == "xor": return s ^ int(p["xor_mask"])
    if role == "guard":
        return (s ^ int(p["effect_mask"])) if ((s & int(p["guard_mask"])) == int(p["guard_value"])) else s
    if role == "delay":
        return ((s & 15) ^ int(p["delay_mask"])) if ((s>>4)&1) else ((s&15)|16)
    raise ValueError(role)

def make_table(audit):
    if audit["out_of_family"]:
        return [[int(v) for v in row] for row in audit["table"]]
    roles=audit["roles"]; p=audit["params"]
    return [[next_state(roles[a],s,p) for a in range(A)] for s in range(N)]

def shortest_len(table,start,goal):
    q=deque([(start,0)]); seen={start}
    while q:
        s,d=q.popleft()
        if s==goal:return d
        for a in range(A):
            t=table[s][a]
            if t not in seen: seen.add(t); q.append((t,d+1))
    return None

def verify_record(rec, audit, label):
    table=make_table(audit)
    if rec.get("mode")=="HOLD_NONDETERMINISTIC" and rec.get("model_digest") is None:
        if not rec.get("nondeterministic",True): raise AssertionError(f"{label}: nondeterministic hold flag missing")
        return {"label":label,"verified_model_table":False,"plan_verified":False,"status":"HOLD"}
    if rec.get("model_digest") != digest(table):
        raise AssertionError(f"{label}: model digest mismatch")
    if rec.get("mode") in ("FULL_TABLE_UNTRUSTED", "FULL_TABLE_FALLBACK"):
        if int(rec.get("unique_observed_pairs",0)) != N*A:
            raise AssertionError(f"{label}: full-table fallback did not log all {N*A} pairs")
        clock_for_coverage=rec.get("clock_action")
        covered={(0,a) for a in range(A)}
        if clock_for_coverage is not None:
            covered.update((s,int(clock_for_coverage)) for s in range(N))
        covered.update((int(s),int(a)) for s,a,y in rec.get("query_order",[]))
        if len(covered) != N*A:
            raise AssertionError(f"{label}: full-table evidence coverage is {len(covered)}/{N*A}")
    if rec.get("clock_action") is not None:
        clock=rec["clock_action"]
        if audit["roles"][clock] != "clock":
            raise AssertionError(f"{label}: wrong clock action")
    mapping=rec.get("role_mapping")
    if mapping is not None:
        norm={int(k):v for k,v in mapping.items()}
        expected={a:r for a,r in enumerate(audit["roles"]) if r!="clock"}
        if norm != expected:
            raise AssertionError(f"{label}: role mapping mismatch")
        if rec.get("model_id") != audit.get("model_id"):
            raise AssertionError(f"{label}: model identity mismatch")
        if rec.get("mode")=="TRANSFER_CONFIRMED":
            checks={(int(s),int(a)) for s,a in rec.get("transfer_checks",[])}
            params=audit["params"]
            target=int(audit["model_id"])
            for alt in range(12):
                if alt==target: continue
                separated=False
                for s,a in checks:
                    role=norm[a]
                    if next_state(role,s,params) != next_state(role,s,MODEL_BANK[alt]):
                        separated=True; break
                if not separated:
                    raise AssertionError(f"{label}: transfer checks do not distinguish model {alt}")
    for s,a,y in rec.get("query_order",[]):
        if table[int(s)][int(a)] != int(y):
            raise AssertionError(f"{label}: query evidence mismatch at {(s,a)}")
    if rec.get("nondeterministic"):
        if rec.get("mode") != "HOLD_NONDETERMINISTIC" or rec.get("plan") is not None:
            raise AssertionError(f"{label}: nondeterministic oracle was not held")
        return {"label":label,"verified_model_table":True,"plan_verified":False,"status":"HOLD"}
    plan=rec.get("plan")
    if plan is None:
        if rec.get("success"):
            raise AssertionError(f"{label}: successful record has no plan")
        return {"label":label,"verified_model_table":True,"plan_verified":False,"status":rec.get("mode")}
    state=0
    for a in plan:
        state=table[state][int(a)]
    if state != int(audit["goal"]) or not rec.get("success"):
        raise AssertionError(f"{label}: plan does not reach goal")
    opt=shortest_len(table,0,int(audit["goal"]))
    if len(plan) != opt:
        raise AssertionError(f"{label}: plan not shortest ({len(plan)} vs {opt})")
    if rec.get("plan_length") != len(plan):
        raise AssertionError(f"{label}: plan length metadata mismatch")
    return {"label":label,"verified_model_table":True,"plan_verified":True,"shortest":True,"status":rec.get("mode")}

def verify_checkpoint(cp, label):
    body=dict(cp); got=body.pop("digest",None)
    if got != digest(body): raise AssertionError(f"{label}: checkpoint digest")
    if body.get("version") != "4.4": raise AssertionError(f"{label}: version")
    # Every committed context model must be consistent with its latest non-HOLD history entry.
    latest={}
    for rec in body.get("history",[]):
        if rec.get("context_id") and rec.get("model_id") is not None and not str(rec.get("mode","")).startswith("HOLD"):
            latest[rec["context_id"]]=rec
    for ctx,model in body.get("contexts",{}).items():
        if int(model["model_id"]) >= 12 or int(model["model_id"]) < 0:
            raise AssertionError(f"{label}: model id bounds")
        mapping={int(k):v for k,v in model["mapping"].items()}
        if len(mapping)!=3 or set(mapping.values())!={"xor","guard","delay"}:
            raise AssertionError(f"{label}: invalid role mapping")
        clock=next(a for a in range(4) if a not in mapping)
        if not MODEL_BANK or int(model["model_id"]) >= len(MODEL_BANK):
            raise AssertionError(f"{label}: model bank unavailable")
        p=MODEL_BANK[int(model["model_id"])]
        table=[[next_state("clock",s,p) if a==clock else next_state(mapping[a],s,p) for a in range(4)] for s in range(N)]
        if digest(table) != model.get("model_digest"):
            raise AssertionError(f"{label}: checkpoint model digest mismatch")
        if ctx not in latest:
            raise AssertionError(f"{label}: missing lineage record for {ctx}")
        rec=latest[ctx]
        if int(rec["model_id"]) != int(model["model_id"]):
            raise AssertionError(f"{label}: checkpoint/history model lineage mismatch for {ctx}")
        if rec.get("model_digest") != model.get("model_digest"):
            raise AssertionError(f"{label}: checkpoint/history digest lineage mismatch for {ctx}")
    return {"label":label,"digest":True,"contexts":len(body.get("contexts",{})),"history":len(body.get("history",[]))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--panel",type=Path,default=Path("results/panel20/panel_runs.json")); ap.add_argument("--summary",type=Path,default=Path("results/panel20/panel_summary.json")); ap.add_argument("--out",type=Path,default=Path("results/independent_verification.json")); ap.add_argument("--checkpoint-dir",type=Path,default=Path("results/panel20"))
    ap.add_argument("--bank",type=Path,default=Path("MODEL_BANK.json"))
    a=ap.parse_args(); runs=json.loads(a.panel.read_text()); summary=json.loads(a.summary.read_text())
    global MODEL_BANK
    MODEL_BANK=json.loads(a.bank.read_text())
    checked=[]
    for r in runs:
        seed=r["seed"]
        checked.append(verify_record(r["train"],r["train_audit"],f"{seed}/train"))
        checked.append(verify_record(r["train_random"],r["train_random_audit"],f"{seed}/train_random"))
        for i,x in enumerate(r["stable"]):
            checked.append(verify_record(x["memory"],x["audit"],f"{seed}/stable{i}/memory"))
            checked.append(verify_record(x["fresh_active"],x["audit"],f"{seed}/stable{i}/fresh"))
        checked.append(verify_record(r["drift"],r["drift_audit"],f"{seed}/drift"))
        checked.append(verify_record(r["drift_fresh"],r["drift_audit"],f"{seed}/drift_fresh"))
        checked.append(verify_record(r["new_context"],r["new_context_audit"],f"{seed}/new_context"))
        checked.append(verify_record(r["ood"],r["ood_audit"],f"{seed}/ood"))
        checked.append(verify_record(r["spoof"],r["spoof_audit"],f"{seed}/spoof"))
        checked.append(verify_record(r["noisy"],r["noisy_audit"],f"{seed}/noisy"))
        cpfile=a.checkpoint_dir/f"checkpoint_{seed}.json"
        cp=json.loads(cpfile.read_text())
        checked.append(verify_checkpoint(cp,f"{seed}/checkpoint"))
    # Tamper: change a context model id but recompute the outer hash. Cross-link to history must catch it.
    tamper_tests=[]
    if runs:
        cp=json.loads((a.checkpoint_dir/f"checkpoint_{runs[0]['seed']}.json").read_text())
        bad=json.loads(json.dumps(cp)); ctx=next(iter(bad["contexts"]))
        bad["contexts"][ctx]["model_id"]=(int(bad["contexts"][ctx]["model_id"])+1)%12
        body=dict(bad); body.pop("digest",None); bad["digest"]=digest(body)
        rejected=False
        try: verify_checkpoint(bad,"tamper/model-id")
        except AssertionError: rejected=True
        if not rejected: raise AssertionError("tampered checkpoint model-id not rejected")
        tamper_tests.append({"test":"checkpoint_model_id_with_rehashed_outer_digest","rejected":True})
    # Summary consistency checks.
    if summary["train_exact"] != len(runs) or summary["train_random_exact"] != len(runs): raise AssertionError("summary train count mismatch")
    if summary["stable_memory_success"] != 2*len(runs): raise AssertionError("summary stable successes mismatch")
    if summary["drift_relearned"] != len(runs): raise AssertionError("summary drift count mismatch")
    if summary["ood_full_table"] != len(runs): raise AssertionError("summary OOD fallback mismatch")
    if summary["spoof_full_table"] != len(runs): raise AssertionError("summary spoof fallback mismatch")
    if summary["noisy_holds"] != len(runs): raise AssertionError("summary noisy hold mismatch")
    result={"status":"PASS","runs":len(runs),"verified_records":len(checked),
            "verified_plans":sum(x.get("plan_verified",False) for x in checked),
            "verified_holds":sum(x.get("status")=="HOLD" for x in checked),
            "checkpoint_audits":sum("checkpoint" in x["label"] for x in checked),
            "tamper_tests":tamper_tests,"summary_checks":"PASS","records":checked}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))

if __name__=="__main__": main()
