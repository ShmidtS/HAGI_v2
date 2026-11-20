#!/usr/bin/env python3
"""HAGI Environment-Grounded Growth v4.5.

Bounded experiment: partially observable 4-bit environment, active model discovery
from a compositional Boolean-LUT grammar, exact OOD fallback, and model-based planning.

Standard library only. This is not a claim of general intelligence.
"""
from __future__ import annotations
from hagi.util import sha
import argparse, itertools, json, math, random, statistics, time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

N_BITS = 4
N_STATES = 1 << N_BITS
N_ACTIONS = 4
_MODEL_CACHE = {}
_PROBE_INDEX_CACHE = {}



def bit(s, i):
    return (s >> i) & 1

def lut2(lut, a, b):
    return (lut >> (a | (b << 1))) & 1

def obs(s):
    # 4 latent bits -> 3 observed bits; x2 and x3 are aliased through parity.
    return bit(s, 0) | (bit(s, 1) << 1) | ((bit(s, 2) ^ bit(s, 3)) << 2)

def clock(s):
    return (s + 1) & 15

def rotate4(s):
    return ((s << 1) & 15) | (s >> 3)

def toggle(s):
    # Fixed primitive: toggle x2 iff x0 and not x1.
    return s ^ (4 if (bit(s, 0) and not bit(s, 1)) else 0)

def coupled(s, f, g):
    # Same-tick depth-2 composition. First update x0 using LUT2(x1,x2),
    # then update x3 using LUT2(updated_x0,x1).
    x0, x1, x2, x3 = (bit(s, i) for i in range(4))
    nx0 = x0 ^ lut2(f, x1, x2)
    nx3 = x3 ^ lut2(g, nx0, x1)
    return nx0 | (x1 << 1) | (x2 << 2) | (nx3 << 3)

def deep_ood(s, f3, g):
    # Out-of-base-grammar action: 3-input LUT update of x0, followed by x3 update.
    x0, x1, x2, x3 = (bit(s, i) for i in range(4))
    idx = x1 | (x2 << 1) | (x3 << 2)
    nx0 = x0 ^ ((f3 >> idx) & 1)
    nx3 = x3 ^ lut2(g, nx0, x1)
    return nx0 | (x1 << 1) | (x2 << 2) | (nx3 << 3)

def apply_role(s, role, f=0, g=0, ood=None):
    if role == "clock": return clock(s)
    if role == "shift": return rotate4(s)
    if role == "toggle": return toggle(s)
    if role == "coupled": return coupled(s, f, g)
    if role == "deep_ood": return deep_ood(s, ood[0], ood[1])
    raise ValueError(role)

@dataclass
class World:
    seed: int
    roles: tuple[str, ...]  # opaque action id -> role
    f: int
    g: int
    goal: int
    ood: bool = False
    f3: int = 0
    state: int = 0
    steps: int = 0
    resets: int = 0
    noisy_repeat: bool = False
    seen: dict = None

    def __post_init__(self):
        self.rng = random.Random(self.seed ^ 0x45BEEF)
        self.seen = {} if self.seen is None else self.seen

    @property
    def clock_action(self): return self.roles.index("clock")

    def reset(self):
        self.state = 0; self.resets += 1
        return obs(self.state)

    def true_step(self, s, a):
        role = self.roles[a]
        if role == "coupled" and self.ood:
            return apply_role(s, "deep_ood", self.f, self.g, (self.f3, self.g))
        return apply_role(s, role, self.f, self.g)

    def step(self, a):
        old = self.state
        nxt = self.true_step(old, a)
        self.state = nxt; self.steps += 1
        key = (old, a); cnt = self.seen.get(key, 0); self.seen[key] = cnt + 1
        y = obs(nxt)
        if self.noisy_repeat and cnt >= 1:
            y ^= 4
        return y

    def goal_test(self):
        return self.state == self.goal

def transition_table(roles, f, g):
    return [[apply_role(s, roles[a], f, g) for a in range(N_ACTIONS)] for s in range(N_STATES)]

def model_digest(table): return sha(table)

def generate_models(clock_action):
    if clock_action in _MODEL_CACHE:
        return _MODEL_CACHE[clock_action]
    rest = [a for a in range(N_ACTIONS) if a != clock_action]
    models = []
    seen = set()
    for perm in itertools.permutations(("shift", "toggle", "coupled")):
        roles = [None] * N_ACTIONS; roles[clock_action] = "clock"
        for a, r in zip(rest, perm): roles[a] = r
        for f in range(16):
            for g in range(16):
                tab = transition_table(tuple(roles), f, g)
                key = model_digest(tab)
                if key in seen: continue
                seen.add(key)
                models.append({"roles": tuple(roles), "f": f, "g": g, "table": tab, "digest": key})
    _MODEL_CACHE[clock_action] = models
    return models

def observation_signature_from_state(s, suffix_len):
    out = [obs(s)]
    for _ in range(suffix_len):
        s = clock(s); out.append(obs(s))
    return tuple(out)

def identify_clock(world):
    expected = tuple(obs((i + 1) & 15) for i in range(16))
    matches = []
    cost = 0
    for a in range(N_ACTIONS):
        world.reset(); seq=[]
        for _ in range(16): seq.append(world.step(a)); cost += 1
        if tuple(seq) == expected: matches.append(a)
    return (matches[0] if len(matches) == 1 else None), cost, len(matches)

def probe_world(world, c_action, state, action, suffix_len):
    world.reset(); cost = 0
    for _ in range(state): world.step(c_action); cost += 1
    seq = [world.step(action)]; cost += 1
    for _ in range(suffix_len): seq.append(world.step(c_action)); cost += 1
    return tuple(seq), cost

def query_signature(model, state, action, suffix_len, c_action):
    s = model["table"][state][action]
    out = [obs(s)]
    for _ in range(suffix_len):
        s = model["table"][s][c_action]
        out.append(obs(s))
    return tuple(out)

def build_probe_index(c_action):
    """Precompute exact signature partitions for the finite hypothesis grammar.

    A candidate set is represented as a Python integer bitset. This avoids
    repeatedly simulating every hypothesis for every possible experiment.
    """
    if c_action in _PROBE_INDEX_CACHE:
        return _PROBE_INDEX_CACHE[c_action]
    models = generate_models(c_action)
    partitions = {}
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            if a == c_action:
                continue
            for k in range(5):
                groups = {}
                key = (s, a, k)
                for idx, model in enumerate(models):
                    sig = query_signature(model, s, a, k, c_action)
                    groups[sig] = groups.get(sig, 0) | (1 << idx)
                partitions[key] = groups
    out = (models, partitions, (1 << len(models)) - 1)
    _PROBE_INDEX_CACHE[c_action] = out
    return out

def choose_active_probe_index(c_action, candidate_mask, used):
    models, partitions, _ = build_probe_index(c_action)
    n = candidate_mask.bit_count()
    best = None
    for key, all_groups in partitions.items():
        if key in used:
            continue
        counts = [((candidate_mask & mask).bit_count()) for mask in all_groups.values()]
        counts = [v for v in counts if v]
        if len(counts) <= 1:
            continue
        h = -sum((v / n) * math.log2(v / n) for v in counts)
        s, a, k = key
        cost = s + 1 + k
        expected = sum(v * v for v in counts) / n
        score = (h - 0.02 * (expected / n)) / cost
        item = (score, -cost, -s, -a, -k, key)
        if best is None or item[:5] > best[:5]:
            best = item
    return None if best is None else best[5]

def entropy(groups, n):
    h=0.0
    for c in groups.values():
        p=c/n
        h -= p*math.log2(p)
    return h

def choose_active_probe(models, c_action, used):
    rest = [a for a in range(N_ACTIONS) if a != c_action]
    n=len(models); best=None
    for s in range(N_STATES):
        for a in rest:
            for k in range(5):
                key=(s,a,k)
                if key in used: continue
                groups={}
                for m in models:
                    sig=query_signature(m,s,a,k,c_action)
                    groups[sig]=groups.get(sig,0)+1
                if len(groups)<=1: continue
                h=entropy(groups,n)
                cost=s+1+k
                # Information per interaction, with a small preference for lower worst-case remainder.
                expected=sum(v*v for v in groups.values())/n
                score=(h - 0.02*(expected/n))/cost
                item=(score,-cost,-s,-a,-k,key,groups)
                if best is None or item[:5] > best[:5]: best=item
    return None if best is None else best[5]

def learn_active(world, initial_models=None, max_queries=80, strategy="active"):
    c_action, clock_cost, nclock = identify_clock(world)
    if c_action is None:
        return {"status":"HOLD_CLOCK_AMBIGUOUS","clock_cost":clock_cost,"queries":0,"interactions":clock_cost}

    indexed = initial_models is None
    if indexed:
        models, partitions, full_mask = build_probe_index(c_action)
        candidate_mask = full_mask
        start_n = len(models)
    else:
        models = list(initial_models)
        candidate_mask = None
        partitions = None
        start_n = len(models)

    used=set(); q=0; interactions=clock_cost; evidence=[]
    rng=random.Random(world.seed ^ 0xA51)
    while q < max_queries:
        current_n = candidate_mask.bit_count() if indexed else len(models)
        if current_n <= 1:
            break
        if strategy == "active":
            probe = choose_active_probe_index(c_action, candidate_mask, used) if indexed else choose_active_probe(models,c_action,used)
        else:
            options=[(s,a,k) for s in range(N_STATES) for a in range(N_ACTIONS) if a!=c_action for k in range(5) if (s,a,k) not in used]
            probe=rng.choice(options) if options else None
        if probe is None: break
        s,a,k=probe; used.add(probe)
        observed,cost=probe_world(world,c_action,s,a,k)
        interactions += cost; q += 1
        before=current_n
        if indexed:
            group_mask = partitions[probe].get(observed, 0)
            candidate_mask &= group_mask
            after = candidate_mask.bit_count()
        else:
            models=[m for m in models if query_signature(m,s,a,k,c_action)==observed]
            after=len(models)
        evidence.append({"state":s,"action":a,"suffix":k,"obs":list(observed),"before":before,"after":after,"cost":cost})
        if after == 0:
            return {"status":"GRAMMAR_MISMATCH","clock_action":c_action,
                    "queries":q,"interactions":interactions,"evidence":evidence,"candidate_start":start_n}

    if indexed:
        models = [m for i,m in enumerate(models) if (candidate_mask >> i) & 1]
    if len(models)==1 or len({m["digest"] for m in models})==1:
        # Six hold-out traces are evidence, not a proof of membership in the grammar.
        candidate=models[0]; nonclock=[a for a in range(N_ACTIONS) if a!=c_action]
        checks=[(14,a) for a in nonclock]
        vrng=random.Random(world.seed ^ 0xA0D17)
        checks += [(vrng.randrange(N_STATES), vrng.choice(nonclock)) for _ in range(3)]
        validation=[]; mismatch=[]
        for vs,va in checks:
            observed,cost=probe_world(world,c_action,vs,va,4); interactions+=cost
            predicted=query_signature(candidate,vs,va,4,c_action)
            item={"state":vs,"action":va,"observed":list(observed),"predicted":list(predicted),"cost":cost}
            validation.append(item)
            if observed!=predicted: mismatch.append(item)
        if mismatch:
            return {"status":"GRAMMAR_MISMATCH","clock_action":c_action,
                    "queries":q,"interactions":interactions,"evidence":evidence,"candidate_start":start_n,
                    "candidate_left":len(models),"validation":validation,"mismatch":mismatch,"provisional_model":candidate}
        return {"status":"PROVISIONAL_SAMPLED","clock_action":c_action,"model":candidate,"clock_cost":clock_cost,
                "queries":q,"interactions":interactions,"evidence":evidence,"candidate_start":start_n,
                "candidate_left":len(models),"validation":validation,"validation_status":"PASS_SAMPLED_ONLY"}
    return {"status":"HOLD_AMBIGUOUS","clock_action":c_action,"clock_cost":clock_cost,
            "queries":q,"interactions":interactions,"evidence":evidence,"candidate_start":start_n,"candidate_left":len(models)}

def decode_hidden_from_clock_signature(seq):
    # The 5-observation signature along clock uniquely identifies all 16 latent states.
    sigmap={observation_signature_from_state(s,4):s for s in range(16)}
    return sigmap.get(tuple(seq))

def full_table_fallback(world, c_action):
    table=[[None]*N_ACTIONS for _ in range(N_STATES)]
    for s in range(N_STATES): table[s][c_action]=clock(s)
    interactions=0; rows=[]
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            if a==c_action: continue
            seq,cost=probe_world(world,c_action,s,a,4)
            nxt=decode_hidden_from_clock_signature(seq)
            interactions += cost
            if nxt is None: raise RuntimeError("clock suffix failed to decode latent state")
            table[s][a]=nxt
            rows.append({"state":s,"action":a,"observations":list(seq),"next_state":nxt,"cost":cost})
    if any(v is None for row in table for v in row): raise AssertionError("incomplete table")
    return table,interactions,rows

def apply_mobius(values, n):
    a=[int(v)&1 for v in values]
    for bit_i in range(n):
        step=1<<bit_i
        for x in range(1<<n):
            if x & step: a[x] ^= a[x^step]
    return [i for i,v in enumerate(a) if v]

def table_to_anf(table):
    # Each action is a 4-input, 4-output deterministic state transition.
    actions=[]
    for a in range(N_ACTIONS):
        outs=[]
        for ob in range(N_BITS):
            vals=[bit(table[s][a],ob) for s in range(N_STATES)]
            outs.append(apply_mobius(vals,N_BITS))
        actions.append(outs)
    return actions

def eval_anf(terms,s):
    out=0
    for t in terms:
        if (s&t)==t: out ^= 1
    return out

def verify_table_anf(table, anfs):
    checks=0
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            got=sum(eval_anf(anfs[a][b],s)<<b for b in range(N_BITS))
            if got != table[s][a]: return False,checks,(s,a,got,table[s][a])
            checks += 1
    return True,checks,None

def bfs_plan(table,start,goal):
    q=deque([start]); prev={start:None}; prev_a={}
    while q:
        s=q.popleft()
        if s==goal: break
        for a in range(N_ACTIONS):
            t=table[s][a]
            if t not in prev:
                prev[t]=s; prev_a[t]=a; q.append(t)
    if goal not in prev: return None
    path=[]; cur=goal
    while prev[cur] is not None:
        path.append(prev_a[cur]); cur=prev[cur]
    return list(reversed(path))

def execute_plan(world, table, path, goal):
    world.reset(); s=0; trace=[]
    for a in path:
        pred=table[s][a]
        observed=world.step(a)
        if observed != obs(pred): return False,trace,"observation mismatch"
        s=pred; trace.append({"action":a,"predicted_state":s,"observed":observed})
    return world.goal_test() and s==goal,trace,"ok" if world.goal_test() and s==goal else "goal mismatch"

def make_world(seed, ood=False, same_dynamics=None, noisy=False, f3=None):
    rng=random.Random(seed ^ 0xC0FFEE)
    roles_list=list(("clock","shift","toggle","coupled")); rng.shuffle(roles_list)
    roles=tuple(roles_list)
    if same_dynamics is None:
        f=rng.randint(1,14); g=rng.randint(1,14)
    else: f,g=same_dynamics
    # For OOD tasks draw a genuinely degree-3 Boolean function. For a 3-input
    # truth-table bitmask this is equivalent to odd parity of its eight values.
    if f3 is None:
        f3=128
        if ood:
            choices=[x for x in range(1,256) if (x.bit_count() & 1) and x not in (0x81,0xA5)]
            f3=rng.choice(choices)
    goal=rng.randrange(1,16)
    return World(seed,roles,f,g,goal,ood,f3,0,0,0,noisy,{})

def audit_candidate_full(world, c_action, model):
    """Exhaustively check all 16 x 3 non-clock transitions via reset + clock addressing."""
    rows=[]; mismatches=[]; interactions=0
    for state in range(N_STATES):
        for action in range(N_ACTIONS):
            if action == c_action:
                if model["table"][state][action] != clock(state):
                    mismatches.append({"state":state,"action":action,"reason":"clock transition"})
                continue
            observed,cost=probe_world(world,c_action,state,action,4)
            interactions += cost
            got=decode_hidden_from_clock_signature(observed)
            want=model["table"][state][action]
            row={"state":state,"action":action,"observed_next":got,"predicted_next":want,"cost":cost}
            rows.append(row)
            if got != want: mismatches.append(row)
    return {"complete":len(rows)==N_STATES*(N_ACTIONS-1),"pairs_checked":len(rows),
            "total_pairs":N_STATES*N_ACTIONS,"interactions":interactions,
            "pass":len(rows)==N_STATES*(N_ACTIONS-1) and not mismatches,"mismatches":mismatches,"rows":rows}

def table_from_full_audit(audit, c_action):
    if not audit.get("complete") or len(audit.get("rows", [])) != N_STATES * (N_ACTIONS - 1):
        raise ValueError("cannot build table from incomplete audit")
    table = [[None] * N_ACTIONS for _ in range(N_STATES)]
    rows = []
    for state in range(N_STATES):
        table[state][c_action] = clock(state)
    for r in audit["rows"]:
        state, action, nxt = r["state"], r["action"], r["observed_next"]
        if table[state][action] is not None:
            raise ValueError("duplicate transition in audit")
        table[state][action] = nxt
        rows.append({"state": state, "action": action, "observations": [],
                     "next_state": nxt, "cost": r["cost"], "source": "full_candidate_audit"})
    if any(v is None for row in table for v in row):
        raise ValueError("full audit left missing transitions")
    return table, audit["interactions"], rows

def run_adversarial_unseen_transition(seed):
    """Construct an OOD environment differing at one state/action not sampled by learner."""
    baseline=make_world(seed)
    base_result=learn_active(baseline)
    if base_result.get("status")!="PROVISIONAL_SAMPLED":
        return {"status":"SETUP_FAILED","base_status":base_result.get("status")}
    c=base_result["clock_action"]
    visited={(e["state"],e["action"]) for e in base_result["evidence"]}
    visited.update((v["state"],v["action"]) for v in base_result.get("validation",[]))
    visited.update((s,c) for s in range(N_STATES))
    target=next(((s,a) for s in range(N_STATES) for a in range(N_ACTIONS) if a!=c and (s,a) not in visited),None)
    if target is None: return {"status":"SETUP_FAILED","reason":"no unseen pair"}
    world=make_world(seed)
    original=world.true_step
    ts,ta=target
    def altered_true_step(self,s,a):
        nxt=original(s,a)
        return nxt ^ 1 if (s,a)==(ts,ta) else nxt
    world.true_step=altered_true_step.__get__(world,World)
    learned=learn_active(world)
    if learned.get("status")!="PROVISIONAL_SAMPLED":
        return {"status":"UNEXPECTED_DETECTION_DURING_SAMPLE","target":list(target),"learn_status":learned.get("status")}
    candidate=learned["model"]
    full=audit_candidate_full(world,learned["clock_action"],candidate)
    return {"status":"PASS" if (not full["pass"] and target in [(m["state"],m["action"]) for m in full["mismatches"]]) else "FAIL",
            "target":list(target),"learner_status":learned["status"],"sampled_validation":learned.get("validation_status"),
            "full_audit_detected":not full["pass"],"pairs_checked":full["pairs_checked"],"mismatches":full["mismatches"]}

def run_seed(seed, strategy="active"):
    world=make_world(seed)
    learned=learn_active(world,strategy=strategy)
    if learned["status"] not in ("PROVISIONAL_SAMPLED","IDENTIFIED","IDENTIFIED_EQUIVALENCE_CLASS"):
        return {"seed":seed,"status":learned["status"],"learn":learned}
    model=learned["model"]; table=model["table"]
    path=bfs_plan(table,0,world.goal)
    ok,trace,reason=execute_plan(world,table,path,world.goal)
    return {"seed":seed,"status":"PASS_SAMPLED" if ok else "FAIL","model_evidence":"SAMPLED_ONLY",
            "clock_action":learned["clock_action"],
            "f":world.f,"g":world.g,"model_digest":model["digest"],"candidate_start":learned["candidate_start"],
            "candidate_left":learned["candidate_left"],"queries":learned["queries"],"learn_interactions":learned["interactions"],
            "evidence":learned["evidence"],"validation":learned.get("validation",[]),
            "validation_status":learned.get("validation_status","NONE"),
            "goal":world.goal,"plan":path,"plan_length":len(path),
            "execution":trace,"execution_ok":ok,"reason":reason}

def infer_deep_ood_rule(table, clock_action):
    """Recover a novel 3-input update followed by a 2-input update from the exact table."""
    for action in range(N_ACTIONS):
        if action == clock_action:
            continue
        fvals={}; gvals={}; ok=True
        for s in range(N_STATES):
            x0,x1,x2,x3=(bit(s,i) for i in range(4))
            nxt=table[s][action]
            nx0,nx1,nx2,nx3=(bit(nxt,i) for i in range(4))
            if nx1 != x1 or nx2 != x2:
                ok=False; break
            idx=x1 | (x2<<1) | (x3<<2)
            fv=x0 ^ nx0
            if idx in fvals and fvals[idx] != fv:
                ok=False; break
            fvals[idx]=fv
            idx2=nx0 | (x1<<1)
            gv=x3 ^ nx3
            if idx2 in gvals and gvals[idx2] != gv:
                ok=False; break
            gvals[idx2]=gv
        if not ok or len(fvals)!=8 or len(gvals)!=4:
            continue
        f3=sum(v<<i for i,v in fvals.items())
        g=sum(v<<i for i,v in gvals.items())
        if (f3.bit_count() & 1)==0:
            continue
        reconstructed=[None]*N_ACTIONS
        reconstructed[clock_action]=[clock(s) for s in range(N_STATES)]
        reconstructed[action]=[deep_ood(s,f3,g) for s in range(N_STATES)]
        for a in range(N_ACTIONS):
            if a in (clock_action,action):
                continue
            reconstructed[a]=[table[s][a] for s in range(N_STATES)]
        if any(reconstructed[action][s] != table[s][action] for s in range(N_STATES)):
            continue
        return {"action":action,"f3":f3,"g":g,"reconstructed_rows":N_STATES}
    return None

def run_ood(seed):
    world = make_world(seed, ood=True)
    learned = learn_active(world)
    c = learned.get("clock_action")
    sampled_audit = None
    if learned["status"] == "PROVISIONAL_SAMPLED":
        # A sampled pass is not proof of grammar membership: exhaustively audit.
        sampled_audit = audit_candidate_full(world, c, learned["model"])
        if sampled_audit["pass"]:
            return {"seed": seed, "status": "BASE_EQUIVALENT_EXACT", "clock_action": c,
                    "sampled_candidate_full_audit": sampled_audit, "true_f3": world.f3, "true_g": world.g}
        grammar_status = "PROVISIONAL_REJECTED_BY_FULL_AUDIT"
        table, cost, rows = table_from_full_audit(sampled_audit, c)
        full_table_source = "REUSED_EXHAUSTIVE_CANDIDATE_AUDIT"
    elif learned["status"] == "GRAMMAR_MISMATCH":
        grammar_status = "GRAMMAR_MISMATCH"
        table, cost, rows = full_table_fallback(world, c)
        full_table_source = "DIRECT_FULL_TABLE_FALLBACK"
    else:
        return {"seed": seed, "status": "LEARN_FAILURE", "learn": learned}

    anfs = table_to_anf(table)
    vok, checks, cex = verify_table_anf(table, anfs)
    path = bfs_plan(table, 0, world.goal)
    ok, trace, reason = execute_plan(world, table, path, world.goal)
    inferred = infer_deep_ood_rule(table, c)
    rule_exact = (inferred is not None and inferred["f3"] == world.f3 and inferred["g"] == world.g)
    total_interactions = learned.get("interactions", 0) + cost
    return {"seed": seed, "status": "PASS" if vok and ok and rule_exact else "FAIL", "clock_action": c,
            "grammar_status": grammar_status, "active_queries_before_hold": learned.get("queries", 0),
            "active_interactions_before_hold": learned.get("interactions", 0),
            "sampled_candidate_full_audit": sampled_audit,
            "fallback_transition_pairs": len(rows), "fallback_interactions": cost,
            "full_table_source": full_table_source, "ood_total_interactions": total_interactions,
            "anf_checks": checks, "anf_verified": vok, "counterexample": cex,
            "model_digest": model_digest(table), "goal": world.goal, "plan": path,
            "plan_length": len(path), "execution_ok": ok, "execution_trace": trace,
            "reason": reason, "rows": rows, "anfs": anfs,
            "true_f3": world.f3, "true_g": world.g,
            "inferred_novel_rule": inferred, "novel_rule_exact": rule_exact}

def self_tests():
    tests=[]
    # Partial observation really aliases pairs, while 5-step clock observations identify every hidden state.
    aliases={}
    for s in range(16): aliases.setdefault(obs(s),[]).append(s)
    assert len(aliases)==8 and all(len(v)==2 for v in aliases.values())
    sigs=[observation_signature_from_state(s,4) for s in range(16)]
    assert len(set(sigs))==16
    tests.append({"name":"partial_observation_and_clock_disambiguation","pass":True,"observation_classes":len(aliases),"unique_clock_signatures":len(set(sigs))})
    # All 16 two-input LUTs are representable, including XNOR and implication cases.
    lut_checks=0
    for lut in range(16):
        for a,b in itertools.product((0,1),repeat=2):
            assert lut2(lut,a,b)==((lut>>(a|(b<<1)))&1); lut_checks+=1
    tests.append({"name":"all_lut2_functions","pass":True,"checks":lut_checks})
    # Active discovery of a generated composite action.
    w=make_world(45001); r=learn_active(w)
    assert r["status"] == "PROVISIONAL_SAMPLED"
    assert r["model"]["digest"]==model_digest(transition_table(w.roles,w.f,w.g))
    tests.append({"name":"active_compositional_discovery","pass":True,"queries":r["queries"],"candidate_start":r["candidate_start"]})
    # Out-of-grammar three-input interaction must expand to exact full-table skill.
    o=run_ood(45002); assert o["status"]=="PASS" and o["fallback_transition_pairs"]==48 and o["anf_checks"]==64
    tests.append({"name":"ood_full_table_and_anf_expansion","pass":True,"pairs":48,"anf_checks":64})
    # Tamper check: alter one action transition and ensure digest mismatch.
    tab=transition_table(w.roles,w.f,w.g); original=model_digest(tab); tab[3][w.clock_action]=(tab[3][w.clock_action]^1)
    assert model_digest(tab)!=original
    tests.append({"name":"checkpoint_model_tamper_detection","pass":True})
    adv=run_adversarial_unseen_transition(45003)
    assert adv["status"]=="PASS"
    tests.append({"name":"unseen_transition_defeats_sampled_audit_but_full_audit_catches_it","pass":True,
                  "target":adv["target"],"full_pairs_checked":adv["pairs_checked"]})
    recovered=[]
    for j in range(8):
        rr=run_ood(45100+j)
        recovered.append(rr.get("novel_rule_exact",False))
    assert all(recovered)
    tests.append({"name":"recover_novel_three_input_lut_from_ood_full_table","pass":True,"trials":len(recovered)})
    return tests

def exact_audit_record(run):
    if run.get("status") not in ("PASS_SAMPLED", "PASS"):
        return {"seed":run.get("seed"),"status":"HOLD_NO_CANDIDATE"}
    world=make_world(run["seed"])
    c=run["clock_action"]
    models=generate_models(c)
    matches=[m for m in models if m["digest"]==run.get("model_digest")]
    if len(matches)!=1:
        return {"seed":run["seed"],"status":"HOLD_CANDIDATE_NOT_IN_GRAMMAR","match_count":len(matches)}
    audit=audit_candidate_full(world,c,matches[0])
    return {"seed":run["seed"],"status":"PROVEN_EXACT" if audit["pass"] else "HOLD_COUNTEREXAMPLE",
            "pairs_checked":audit["pairs_checked"],"total_pairs":audit["total_pairs"],
            "interactions":audit["interactions"],"mismatches":audit["mismatches"],
            "candidate_digest":run.get("model_digest")}

def run_panel(start_seed, n, outdir):
    outdir.mkdir(parents=True,exist_ok=True); active=[]; random_runs=[]; ood=[]
    for i in range(n):
        seed=start_seed+i
        active.append(run_seed(seed,"active"))
        random_runs.append(run_seed(seed,"random"))
        ood.append(run_ood(seed+100000))
        if (i+1)%5==0:
            (outdir/f"progress_{i+1}.json").write_text(json.dumps({"done":i+1},indent=2))
    active_exact=[exact_audit_record(x) for x in active]
    random_exact=[exact_audit_record(x) for x in random_runs]
    def avg(xs): return statistics.mean(xs) if xs else 0.0
    a_ok=sum(x.get("status") in ("PASS","PASS_SAMPLED") for x in active)
    r_ok=sum(x.get("status") in ("PASS","PASS_SAMPLED") for x in random_runs)
    o_ok=sum(x.get("status")=="PASS" for x in ood)
    good=lambda x: x.get("status") in ("PASS","PASS_SAMPLED")
    a_q=[x.get("queries",0) for x in active if good(x)]
    r_q=[x.get("queries",0) for x in random_runs if good(x)]
    a_cost=[x.get("learn_interactions",0) for x in active if good(x)]
    r_cost=[x.get("learn_interactions",0) for x in random_runs if good(x)]
    o_fallback=[x.get("fallback_interactions",0) for x in ood if x.get("status")=="PASS"]
    summary={"version":"4.6","seeds":n,"active_success":a_ok,"random_success":r_ok,"ood_success":o_ok,
             "active_mean_queries":avg(a_q),"random_mean_queries":avg(r_q),
             "active_mean_interactions":avg(a_cost),"random_mean_interactions":avg(r_cost),
             "active_interaction_reduction_pct":(1-avg(a_cost)/avg(r_cost))*100 if avg(r_cost) else 0,
             "ood_full_table_pairs_mean":avg([x.get("fallback_transition_pairs",0) for x in ood]),
             "ood_fallback_interactions_mean":avg(o_fallback),
             "ood_total_interactions_mean":avg([x.get("ood_total_interactions",0) for x in ood if x.get("status")=="PASS"]),
             "ood_inferred_novel_rules":sum(bool(x.get("novel_rule_exact")) for x in ood),
             "active_exact_commit_passes":sum(x.get("status")=="PROVEN_EXACT" for x in active_exact),
             "random_exact_commit_passes":sum(x.get("status")=="PROVEN_EXACT" for x in random_exact),
             "active_total_cost_mean_with_exact_audit":avg([a.get("learn_interactions",0)+e.get("interactions",0) for a,e in zip(active,active_exact)]),
             "random_total_cost_mean_with_exact_audit":avg([a.get("learn_interactions",0)+e.get("interactions",0) for a,e in zip(random_runs,random_exact)]),
             "exact_commit_policy":"PROVEN_EXACT_ONLY_AFTER_48_PAIR_AUDIT",
             "mean_plan_length":avg([x.get("plan_length",0) for x in active if good(x)]),
             "candidate_commit_status":"PROVISIONAL_SAMPLED_ONLY",
             "active_candidate_start":active[0].get("candidate_start"),
             "all_ood_anf_checks":all(x.get("anf_verified",False) for x in ood if x.get("status")=="PASS")}
    (outdir/"active_runs.json").write_text(json.dumps(active,indent=2))
    (outdir/"random_runs.json").write_text(json.dumps(random_runs,indent=2))
    (outdir/"ood_runs.json").write_text(json.dumps(ood,indent=2))
    (outdir/"active_exact_audits.json").write_text(json.dumps(active_exact,indent=2))
    (outdir/"random_exact_audits.json").write_text(json.dumps(random_exact,indent=2))
    (outdir/"panel_summary.json").write_text(json.dumps(summary,indent=2))
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--panel",type=int,default=20)
    ap.add_argument("--seed",type=int,default=20261120)
    ap.add_argument("--out",type=Path,default=Path("results/panel20"))
    args=ap.parse_args()
    if args.self_test:
        tests=self_tests(); print(json.dumps({"all_pass":all(t["pass"] for t in tests),"tests":tests},indent=2)); return
    t0=time.time(); summary=run_panel(args.seed,args.panel,args.out); summary["elapsed_seconds"]=round(time.time()-t0,3)
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
