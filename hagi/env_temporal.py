#!/usr/bin/env python3
"""HAGI Temporal Environment Growth.

A finite-state, action-driven environment benchmark with:
- hidden 5-bit temporal dynamics and four opaque action symbols;
- active identification over a declared finite family of 12 mechanisms;
- memory transfer with exact finite-family distinguishing sets;
- drift detection and active relearning;
- exact full transition-table fallback for untrusted/out-of-family worlds;
- model-based shortest-path planning;
- checkpointed memory and adversarial consistency tests.

This is a bounded synthetic system-identification experiment, not an AGI claim.
"""
from __future__ import annotations
from hagi.util import sha
import argparse, hashlib, itertools, json, math, random, statistics, time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


N_BITS = 5
N_STATES = 1 << N_BITS
N_ACTIONS = 4
TRUSTED_FAMILY = "TEMPORAL_AFFINE_DELAY_FAMILY_V1"
TRUSTED_ISSUER = "HAGI_TEMPORAL_SIM_V1"
ROLES = ("xor", "guard", "delay")




def clock_fn(s: int) -> int:
    return (s + 1) % N_STATES


def role_fn(role: str, s: int, p: dict) -> int:
    if role == "clock":
        return clock_fn(s)
    if role == "xor":
        return s ^ int(p["xor_mask"])
    if role == "guard":
        if (s & int(p["guard_mask"])) == int(p["guard_value"]):
            return s ^ int(p["effect_mask"])
        return s
    if role == "delay":
        pending = (s >> 4) & 1
        low = s & 15
        if pending == 0:
            return low | 16
        return (low ^ int(p["delay_mask"]))
    raise ValueError(role)


def generate_bank(n_models: int = 12) -> list[dict]:
    rng = random.Random(440044)
    bank = []
    seen = set()
    guard_masks = [m for m in range(1, 32) if m.bit_count() >= 2]
    while len(bank) < n_models:
        p = {
            "xor_mask": 6,  # deliberately shared: a single reset probe must not reveal model identity
            "guard_mask": rng.choice(guard_masks),
            "guard_value": 0,
            "effect_mask": rng.choice([1, 2, 4, 8, 16, 3, 5, 6, 9, 10, 12]),
            "delay_mask": rng.randint(1, 15),
        }
        sub = [x for x in range(1, 32) if (x & p["guard_mask"]) == x]
        p["guard_value"] = rng.choice(sub)
        sig = tuple(role_fn(r, s, p) for r in ("xor", "guard", "delay") for s in range(N_STATES))
        if sig in seen:
            continue
        # Avoid identical primitive roles within one model.
        role_sigs = [tuple(role_fn(r, s, p) for s in range(N_STATES)) for r in ("xor", "guard", "delay")]
        if len(set(role_sigs)) != 3:
            continue
        seen.add(sig)
        p["model_id"] = len(bank)
        bank.append(p)
    # Assert each model + action permutation is behaviorally distinct.
    signatures = set()
    for p in bank:
        for perm in itertools.permutations(ROLES):
            sig = tuple(role_fn("clock", s, p) for s in range(N_STATES))
            for role in perm:
                sig += tuple(role_fn(role, s, p) for s in range(N_STATES))
            if sig in signatures:
                raise AssertionError("duplicate mechanism template")
            signatures.add(sig)
    return bank


BANK = generate_bank()


@dataclass
class TemporalWorld:
    seed: int
    model_id: int | None
    action_roles: tuple[str, ...]
    params: dict | None = None
    context_id: str = "shared"
    family: str | None = TRUSTED_FAMILY
    issuer: str = TRUSTED_ISSUER
    goal: int = 0
    out_of_family: bool = False
    noisy_repeat: bool = False
    state: int = 0
    steps: int = 0
    resets: int = 0
    _pair_seen: dict[tuple[int, int], int] = field(default_factory=dict)
    _rng: random.Random = field(init=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed ^ 0x44A1)
        if self.params is None and self.model_id is not None:
            self.params = BANK[self.model_id]
        if len(self.action_roles) != N_ACTIONS or self.action_roles.count("clock") != 1:
            raise ValueError("action_roles must map exactly one clock and 3 nonclock actions")
        if self.out_of_family and self.params is None:
            self._ood_table = [[None for _ in range(N_ACTIONS)] for _ in range(N_STATES)]
            # Action role `clock` remains a known navigation primitive; all other roles are arbitrary.
            for s in range(N_STATES):
                for a, role in enumerate(self.action_roles):
                    if role == "clock":
                        self._ood_table[s][a] = clock_fn(s)
                    else:
                        self._ood_table[s][a] = self._rng.randrange(N_STATES)
        else:
            self._ood_table = None

    @property
    def clock_action(self) -> int:
        return self.action_roles.index("clock")

    def reset(self) -> int:
        self.state = 0
        self.resets += 1
        return self.state

    def true_transition(self, s: int, a: int) -> int:
        if self.out_of_family:
            return int(self._ood_table[s][a])
        return role_fn(self.action_roles[a], s, self.params)

    def step(self, action: int) -> int:
        if not 0 <= action < N_ACTIONS:
            raise ValueError("action out of range")
        old = self.state
        nxt = self.true_transition(old, action)
        self.state = nxt
        self.steps += 1
        observed = nxt
        key = (old, action)
        count = self._pair_seen.get(key, 0)
        self._pair_seen[key] = count + 1
        if self.noisy_repeat and count >= 1:
            observed ^= 1  # deterministic repeat inconsistency, internal state remains correct.
        return observed


@dataclass
class AgentMemory:
    version: str = "4.4"
    contexts: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    alerts: list = field(default_factory=list)

    def payload(self) -> dict:
        body = {"version": self.version, "contexts": self.contexts,
                "history": self.history, "alerts": self.alerts}
        return {**body, "digest": sha(body)}

    @classmethod
    def restore(cls, payload: dict) -> "AgentMemory":
        body = dict(payload); got = body.pop("digest", None)
        if got != sha(body) or body.get("version") != "4.4":
            raise ValueError("checkpoint integrity/version failure")
        return cls(version=body["version"], contexts=body["contexts"],
                   history=body["history"], alerts=body["alerts"])


def candidate_records(clock_action: int, model_ids: Iterable[int] | None = None) -> list[dict]:
    nonclock = [a for a in range(N_ACTIONS) if a != clock_action]
    mids = list(range(len(BANK))) if model_ids is None else list(model_ids)
    out = []
    for mid in mids:
        for perm in itertools.permutations(ROLES):
            mapping = {a: role for a, role in zip(nonclock, perm)}
            out.append({"model_id": mid, "mapping": mapping})
    return out


def predict_candidate(cand: dict, params_bank: list[dict], state: int, action: int,
                      clock_action: int) -> int:
    if action == clock_action:
        return clock_fn(state)
    return role_fn(cand["mapping"][action], state, params_bank[cand["model_id"]])


def identify_clock(world: TemporalWorld) -> tuple[int | None, dict, int]:
    """Find the unique +1 mod-32 action. If reset probes collide, verify full orbit."""
    observations = {}; cost = 0; found = []
    first_step = {}
    for a in range(N_ACTIONS):
        world.reset(); cost += 1
        nxt = world.step(a); cost += 1
        observations[(0, a)] = nxt
        first_step[a] = nxt
        if nxt == 1:
            found.append(a)
    if len(found) == 1:
        candidate=found[0]
        # Do not trust a single 0->1 coincidence: verify the entire navigation orbit.
        world.reset(); cost += 1
        state=0; valid=True
        for expected in range(1,N_STATES):
            observed=world.step(candidate); cost += 1
            prior_value=observations.get((state,candidate))
            if prior_value is not None and prior_value != observed:
                world.reset(); cost += 1
                return -2, observations, cost
            observations[(state,candidate)] = observed
            if observed != expected:
                valid=False; break
            state=observed
        if valid:
            observed=world.step(candidate); cost += 1
            prior_value=observations.get((state,candidate))
            if prior_value is not None and prior_value != observed:
                world.reset(); cost += 1
                return -2, observations, cost
            observations[(state,candidate)] = observed
            valid=(observed==0)
        world.reset(); cost += 1
        if valid:
            return candidate, observations, cost
    # Robust fallback: an actual clock must implement the complete 32-state +1 orbit.
    orbit_found = []
    for a in range(N_ACTIONS):
        world.reset(); cost += 1
        state = 0; ok = True
        for expected in range(1, N_STATES):
            observed = world.step(a); cost += 1
            prior_value=observations.get((state,a))
            if prior_value is not None and prior_value != observed:
                world.reset(); cost += 1
                return -2, observations, cost
            observations[(state, a)] = observed
            if observed != expected:
                ok = False
                break
            state = observed
        if ok:
            observed = world.step(a); cost += 1
            prior_value=observations.get((state,a))
            if prior_value is not None and prior_value != observed:
                world.reset(); cost += 1
                return -2, observations, cost
            observations[(state, a)] = observed
            if observed == 0:
                orbit_found.append(a)
    world.reset(); cost += 1
    if len(orbit_found) == 1:
        return orbit_found[0], observations, cost
    return None, observations, cost


def navigate_query(world: TemporalWorld, clock_action: int, current_state: int,
                   target_state: int, action: int) -> tuple[int, int, int]:
    nav = (target_state - current_state) % N_STATES
    cur = current_state
    for _ in range(nav):
        cur = world.step(clock_action)
    observed = world.step(action)
    return observed, nav, 1


def query_pair(world: TemporalWorld, clock_action: int, current_state: int,
               pair: tuple[int, int]) -> tuple[int, int, int]:
    s, a = pair
    observed, nav, probes = navigate_query(world, clock_action, current_state, s, a)
    return observed, nav, probes


def choose_pair(candidates: list[dict], untested: list[tuple[int, int]],
                params_bank: list[dict], clock_action: int, current_state: int,
                rng: random.Random, strategy: str) -> tuple[int, int] | None:
    if not untested:
        return None
    if strategy == "random":
        return rng.choice(untested)
    best_pair = None; best_score = None
    for s, a in untested:
        groups = {}
        for c in candidates:
            y = predict_candidate(c, params_bank, s, a, clock_action)
            groups[y] = groups.get(y, 0) + 1
        if len(groups) <= 1:
            continue
        total = len(candidates)
        entropy = -sum((v/total) * math.log2(v/total) for v in groups.values())
        worst = max(groups.values())
        nav = (s - current_state) % N_STATES
        score = (entropy, -worst, -nav, -s, -a)
        if best_score is None or score > best_score:
            best_score, best_pair = score, (s, a)
    return best_pair


def filter_candidates(candidates: list[dict], observations: dict[tuple[int, int], int],
                      params_bank: list[dict], clock_action: int) -> list[dict]:
    return [c for c in candidates if all(
        predict_candidate(c, params_bank, s, a, clock_action) == y
        for (s, a), y in observations.items())]


def active_identify(world: TemporalWorld, clock_action: int, observations: dict,
                    current_state: int, strategy: str = "active", rng=None,
                    initial_candidates: list[dict] | None = None) -> tuple[dict | None, dict]:
    rng = rng or random.Random(world.seed)
    candidates = filter_candidates(initial_candidates or candidate_records(clock_action), observations, BANK, clock_action)
    untested = [(s, a) for s in range(N_STATES) for a in range(N_ACTIONS)
                if a != clock_action and (s, a) not in observations]
    nav_total = probe_total = 0
    query_order = []
    while len(candidates) > 1:
        pair = choose_pair(candidates, untested, BANK, clock_action, current_state, rng, strategy)
        if pair is None:
            break
        y, nav, probes = query_pair(world, clock_action, current_state, pair)
        s, a = pair; current_state = y
        nav_total += nav; probe_total += probes; query_order.append([s, a, y])
        observations[pair] = y
        untested.remove(pair)
        candidates = filter_candidates(candidates, {pair: y}, BANK, clock_action)
        if not candidates:
            break
    chosen = candidates[0] if len(candidates) == 1 else None
    return chosen, {"current_state": current_state, "nav_steps": nav_total,
                    "probe_queries": probe_total, "query_order": query_order,
                    "remaining_candidates": len(candidates)}


def minimum_distinguishing_pairs(target_model: int, mapping: dict, params_bank: list[dict],
                                 clock_action: int, prior_observations: dict) -> list[tuple[int, int]]:
    """Exact minimum-cardinality set cover separating the stored model from other bank models."""
    alts = [mid for mid in range(len(params_bank)) if mid != target_model]
    if not alts:
        return []
    full = (1 << len(alts)) - 1
    covered = 0
    for (s, a), y in prior_observations.items():
        if a == clock_action:
            continue
        expected = role_fn(mapping[a], s, params_bank[target_model])
        if expected != y:
            return []
        for i, mid in enumerate(alts):
            if role_fn(mapping[a], s, params_bank[mid]) != expected:
                covered |= (1 << i)
    if covered == full:
        return []
    tests = []
    for s in range(N_STATES):
        for a, role in mapping.items():
            exp = role_fn(role, s, params_bank[target_model])
            mask = 0
            for i, mid in enumerate(alts):
                if role_fn(role, s, params_bank[mid]) != exp:
                    mask |= 1 << i
            if (s, a) not in prior_observations and mask:
                tests.append((mask, (s, a)))
    # BFS on the small 2^11 coverage lattice yields an exact minimum number of probes.
    queue = deque([covered]); parent = {covered: None}; used_test = {}
    target = None
    while queue:
        cur = queue.popleft()
        if cur == full:
            target = cur; break
        for mask, pair in tests:
            nxt = cur | mask
            if nxt not in parent:
                parent[nxt] = cur; used_test[nxt] = pair; queue.append(nxt)
    if target is None:
        raise RuntimeError("no distinguishing set for model bank")
    result = []
    while target != covered:
        result.append(used_test[target]); target = parent[target]
    result.reverse()
    return result


def build_candidate_table(cand: dict, clock_action: int, params_bank: list[dict]) -> list[list[int]]:
    table = [[0] * N_ACTIONS for _ in range(N_STATES)]
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            if a == clock_action: table[s][a] = clock_fn(s)
            else: table[s][a] = role_fn(cand["mapping"][a], s, params_bank[cand["model_id"]])
    return table


def full_table_learn(world: TemporalWorld, clock_action: int, observations: dict,
                     current_state: int) -> tuple[list[list[int]], dict]:
    nav_total = probe_total = 0; query_order = []
    table = [[clock_fn(s) if a == clock_action else None for a in range(N_ACTIONS)] for s in range(N_STATES)]
    for (s, a), y in observations.items():
        if a != clock_action:
            table[s][a] = y
    remaining = {(s, a) for s in range(N_STATES) for a in range(N_ACTIONS)
                 if a != clock_action and table[s][a] is None}
    # Route queries in nearest-clock-reachable order to reduce environment interaction cost.
    while remaining:
        pair = min(remaining, key=lambda p: ((p[0] - current_state) % N_STATES, p[1], p[0]))
        s, a = pair
        y, nav, probes = query_pair(world, clock_action, current_state, pair)
        current_state = y; nav_total += nav; probe_total += probes; query_order.append([s, a, y]); table[s][a] = y
        observations[(s,a)] = y
        remaining.remove(pair)
    return table, {"current_state": current_state, "nav_steps": nav_total,
                   "probe_queries": probe_total, "query_order": query_order}


def bfs_plan(table: list[list[int]], start: int, goal: int) -> list[int] | None:
    q = deque([start]); prev = {start: None}; prev_a = {}
    while q:
        s = q.popleft()
        if s == goal: break
        for a in range(N_ACTIONS):
            ns = table[s][a]
            if ns not in prev:
                prev[ns] = s; prev_a[ns] = a; q.append(ns)
    if goal not in prev: return None
    path=[]; s=goal
    while prev[s] is not None:
        path.append(prev_a[s]); s=prev[s]
    return list(reversed(path))


class TemporalGrowthAgent:
    def __init__(self, memory: AgentMemory | None = None, strategy: str = "active"):
        self.memory = memory or AgentMemory(); self.strategy = strategy

    def run_task(self, world: TemporalWorld, rng=None) -> dict:
        rng = rng or random.Random(world.seed ^ 0x771)
        start_steps, start_resets = world.steps, world.resets
        clock_action, observations, clock_cost = identify_clock(world)
        if clock_action == -2:
            self.memory.alerts.append({"context":world.context_id,"kind":"NONDETERMINISTIC_CLOCK_PROBE"})
            return {"mode":"HOLD_NONDETERMINISTIC","success":False,"exact_model":False,
                    "nondeterministic":True,"probe_queries":0,"navigation_steps":0,
                    "total_interactions":world.steps-start_steps+(world.resets-start_resets),"total_steps":world.steps-start_steps}
        if clock_action is None:
            return {"mode":"HOLD_CLOCK_UNIDENTIFIED","success":False,"exact_model":False,
                    "probe_queries":0,"navigation_steps":0,"total_steps":world.steps-start_steps}
        current_state = 0
        trusted = world.family == TRUSTED_FAMILY and world.issuer == TRUSTED_ISSUER
        prior = self.memory.contexts.get(world.context_id) if trusted else None
        mode="FRESH_ACTIVE"; selected=None; table=None
        nav_total=0; probe_total=0
        # Include non-clock transition samples collected while validating the clock orbit.
        query_order=[[s,a,y] for (s,a),y in observations.items() if a != clock_action and s != 0]
        transfer_checks=[]
        if trusted and prior is not None:
            mode="TRANSFER_VALIDATE"
            prior_map_cands = candidate_records(clock_action, [int(prior["model_id"])])
            mapping_cand, info = active_identify(world, clock_action, observations, current_state,
                                                 "active", rng, prior_map_cands)
            current_state = info["current_state"]; nav_total += info["nav_steps"]; probe_total += info["probe_queries"]; query_order.extend(info["query_order"])
            if mapping_cand is not None:
                mapping = mapping_cand["mapping"]
                try:
                    checks = minimum_distinguishing_pairs(int(prior["model_id"]), mapping, BANK, clock_action, observations)
                except RuntimeError:
                    checks=[]
                transfer_ok=True
                for pair in checks:
                    y, nav, probes = query_pair(world, clock_action, current_state, pair)
                    current_state=y; nav_total += nav; probe_total += probes; query_order.append([pair[0],pair[1],y]); observations[pair]=y
                    expected=role_fn(mapping[pair[1]],pair[0],BANK[int(prior["model_id"])])
                    if y != expected:
                        transfer_ok=False; break
                if transfer_ok:
                    selected=mapping_cand; table=build_candidate_table(selected,clock_action,BANK)
                    mode="TRANSFER_CONFIRMED"
                    transfer_checks=checks
                else:
                    mode="DRIFT_DETECTED"
            else:
                mode="DRIFT_DETECTED"
            if selected is None:
                full_cands=candidate_records(clock_action)
                selected, info2=active_identify(world,clock_action,observations,current_state,
                                               self.strategy,rng,full_cands)
                current_state=info2["current_state"]; nav_total += info2["nav_steps"]; probe_total += info2["probe_queries"]; query_order.extend(info2["query_order"])
                if selected is not None:
                    table=build_candidate_table(selected,clock_action,BANK); mode="DRIFT_RELEARNED"
        elif trusted:
            selected, info=active_identify(world,clock_action,observations,current_state,self.strategy,rng)
            current_state=info["current_state"]; nav_total += info["nav_steps"]; probe_total += info["probe_queries"]; query_order.extend(info["query_order"])
            if selected is not None:
                table=build_candidate_table(selected,clock_action,BANK); mode="LEARNED_ACTIVE"
        else:
            mode="FULL_TABLE_UNTRUSTED"
        if table is None:
            table, info=full_table_learn(world,clock_action,observations,current_state)
            current_state=info["current_state"]; nav_total += info["nav_steps"]; probe_total += info["probe_queries"]; query_order.extend(info["query_order"])
            mode="FULL_TABLE_FALLBACK" if trusted else "FULL_TABLE_UNTRUSTED"
        # Repeat one previously seen pair from a clean reset to detect inconsistent outputs.
        audit_pair = next(((s,a) for s,a in observations if a != clock_action), None)
        nondeterministic=False
        if audit_pair is not None:
            world.reset(); current_state=0
            audit_s,audit_a=audit_pair
            for _ in range(audit_s):
                current_state=world.step(clock_action)
            y2=world.step(audit_a)
            expected_seen=observations[audit_pair]
            if y2 != expected_seen:
                nondeterministic=True; mode="HOLD_NONDETERMINISTIC"
                self.memory.alerts.append({"context":world.context_id,"pair":[audit_s,audit_a],"expected":expected_seen,"observed":y2})
        exact_model = table is not None and not nondeterministic
        path=None; success=False; plan_len=None
        if exact_model:
            path=bfs_plan(table,0,world.goal)
            if path is not None:
                world.reset(); state=0
                for a in path:
                    state=world.step(a)
                success=(state==world.goal)
                plan_len=len(path)
        if trusted and selected is not None and not nondeterministic:
            self.memory.contexts[world.context_id]={"model_id":selected["model_id"],
                "mapping":{str(k):v for k,v in selected["mapping"].items()},
                "source_seed":world.seed,"version":int(prior.get("version",0)+1) if prior else 1,
                "family":TRUSTED_FAMILY,"issuer":TRUSTED_ISSUER,
                "model_digest":sha([[table[s][a] for a in range(N_ACTIONS)] for s in range(N_STATES)])}
        action_count=world.steps-start_steps
        rec={"context_id":world.context_id,"mode":mode,"model_id":selected["model_id"] if selected else None,
             "clock_action":clock_action,"goal":world.goal,"success":success,"exact_model":exact_model,
             "plan_length":plan_len,"probe_queries":sum(1 for (_,a) in observations if a != clock_action),"unique_observed_pairs":len(observations),
             "navigation_steps":nav_total,"clock_discovery_cost":clock_cost,
             "total_interactions":action_count+(world.resets-start_resets),"resets":world.resets-start_resets,
             "query_order":query_order,"transfer_checks":[list(x) for x in transfer_checks],
             "role_mapping":({str(k):v for k,v in selected["mapping"].items()} if selected else None),
             "plan":path,"nondeterministic":nondeterministic,"trusted":trusted,
             "model_digest":sha(table) if table is not None else None}
        self.memory.history.append(rec)
        return rec


def random_action_roles(seed: int) -> tuple[str,...]:
    p=list(("clock",)+ROLES); random.Random(seed).shuffle(p); return tuple(p)


def make_world(seed: int, model_id: int | None, context: str, goal: int,
               *, family=TRUSTED_FAMILY, issuer=TRUSTED_ISSUER,
               out_of_family=False, noisy_repeat=False, roles=None) -> TemporalWorld:
    roles=roles or random_action_roles(seed)
    params=BANK[model_id] if model_id is not None else None
    return TemporalWorld(seed,model_id,roles,params,context,family,issuer,goal,out_of_family,noisy_repeat)


def _audit_world(world: TemporalWorld) -> dict:
    if world.out_of_family:
        table = [[world.true_transition(s, a) for a in range(N_ACTIONS)] for s in range(N_STATES)]
        return {"seed":world.seed,"model_id":None,"roles":list(world.action_roles),"goal":world.goal,
                "family":world.family,"issuer":world.issuer,"out_of_family":True,"table":table}
    return {"seed":world.seed,"model_id":world.model_id,"roles":list(world.action_roles),"goal":world.goal,
            "family":world.family,"issuer":world.issuer,"out_of_family":False,
            "params":dict(world.params)}


def run_seed(seed: int) -> dict:
    rng=random.Random(seed)
    base_model=rng.randrange(len(BANK)); mem=AgentMemory(); agent=TemporalGrowthAgent(mem,"active")
    train_world=make_world(seed+100,base_model,"core",rng.randrange(1,N_STATES))
    train=agent.run_task(train_world,random.Random(seed+200)); train_audit=_audit_world(train_world)
    random_train_world=make_world(seed+100,base_model,"core",train_world.goal,roles=train_world.action_roles)
    train_random=TemporalGrowthAgent(AgentMemory(),"random").run_task(random_train_world,random.Random(seed+200))
    train_random_audit=_audit_world(random_train_world)
    stable=[]
    for j in range(2):
        w=make_world(seed+300+j,base_model,"core",rng.randrange(1,N_STATES))
        mem_agent=agent.run_task(w,random.Random(seed+400+j)); mem_audit=_audit_world(w)
        fw=make_world(seed+300+j,base_model,"core",w.goal,roles=w.action_roles)
        fresh=TemporalGrowthAgent(AgentMemory(),"active").run_task(fw,random.Random(seed+400+j))
        stable.append({"memory":mem_agent,"fresh_active":fresh,"audit":mem_audit})
    drift_model=(base_model+1+rng.randrange(len(BANK)-1))%len(BANK)
    drift_world=make_world(seed+500,drift_model,"core",rng.randrange(1,N_STATES))
    drift=agent.run_task(drift_world,random.Random(seed+501)); drift_audit=_audit_world(drift_world)
    dfw=make_world(seed+500,drift_model,"core",drift_world.goal,roles=drift_world.action_roles)
    drift_fresh=TemporalGrowthAgent(AgentMemory(),"active").run_task(dfw,random.Random(seed+501))
    new_world=make_world(seed+600,(base_model+4)%len(BANK),"new_context",rng.randrange(1,N_STATES))
    new_context=agent.run_task(new_world,random.Random(seed+601)); new_audit=_audit_world(new_world)
    ood_world=make_world(seed+700,None,"ood",rng.randrange(1,N_STATES),family=None,issuer="UNKNOWN",out_of_family=True)
    ood=agent.run_task(ood_world,random.Random(seed+701)); ood_audit=_audit_world(ood_world)
    spoof_world=make_world(seed+800,base_model,"spoof",rng.randrange(1,N_STATES),family=TRUSTED_FAMILY,issuer="ROGUE_ISSUER")
    spoof=agent.run_task(spoof_world,random.Random(seed+801)); spoof_audit=_audit_world(spoof_world)
    noisy_world=make_world(seed+900,base_model,"core",rng.randrange(1,N_STATES),noisy_repeat=True)
    noisy=agent.run_task(noisy_world,random.Random(seed+901)); noisy_audit=_audit_world(noisy_world)
    cp=mem.payload(); restored=AgentMemory.restore(cp)
    audits={"train":train_audit,"stable":[x["audit"] for x in stable],"drift":drift_audit,
            "new_context":new_audit,"ood":ood_audit,"spoof":spoof_audit,"noisy":noisy_audit}
    return {"seed":seed,"base_model":base_model,"train":train,"train_audit":train_audit,
            "train_random":train_random,"train_random_audit":train_random_audit,
            "stable":stable,"drift":drift,"drift_audit":drift_audit,"drift_fresh":drift_fresh,
            "new_context":new_context,"new_context_audit":new_audit,"ood":ood,"ood_audit":ood_audit,
            "spoof":spoof,"spoof_audit":spoof_audit,"noisy":noisy,"noisy_audit":noisy_audit,
            "audits":audits,"checkpoint":cp,"checkpoint_roundtrip":restored.payload()==cp}


def self_tests() -> list[dict]:
    out=[]
    # All generated models have unique complete transition tables and clock signature.
    assert len(BANK)==12
    out.append({"test":"bank_size_and_distinctness","pass":True,"models":len(BANK)})
    for i,p in enumerate(BANK):
        assert all(role_fn("clock",s,p)==(s+1)%N_STATES for s in range(N_STATES))
        assert role_fn("xor",0,p)!=1 and role_fn("guard",0,p)!=1 and role_fn("delay",0,p)!=1
    out.append({"test":"unique_clock_discovery_signature","pass":True})
    # Exact set cover distinguishes every bank template under fixed role mapping.
    mapping={1:"xor",2:"guard",3:"delay"}
    tests=minimum_distinguishing_pairs(0,mapping,BANK,0,{})
    for mid in range(1,len(BANK)):
        assert any(role_fn(mapping[a],s,BANK[mid])!=role_fn(mapping[a],s,BANK[0]) for s,a in tests)
    out.append({"test":"exact_minimum_distinguishing_set","pass":True,"queries":len(tests)})
    # Every template and all 6 non-clock action permutations can be identified and planned.
    cases=0
    for mi in range(len(BANK)):
        for perm in itertools.permutations(ROLES):
            roles_list=[None]*N_ACTIONS; clock_pos=(mi+cases)%N_ACTIONS; roles_list[clock_pos]="clock"
            other=[a for a in range(N_ACTIONS) if a!=clock_pos]
            for a,role in zip(other,perm): roles_list[a]=role
            w=make_world(2000+cases,mi,"self",(cases*7+3)%N_STATES,roles=tuple(roles_list))
            a=TemporalGrowthAgent(AgentMemory())
            r=a.run_task(w)
            assert r["exact_model"] and r["success"] and r["model_id"]==mi
            assert r["role_mapping"]=={str(a):role for a,role in enumerate(roles_list) if role!="clock"}
            cases+=1
    out.append({"test":"all_bank_models_all_action_permutations_identifiable_and_plannable","pass":True,"cases":cases})
    # Drift, OOD fallback, rogue issuer and repeat inconsistency are explicit safety paths.
    m=AgentMemory(); ag=TemporalGrowthAgent(m)
    first=ag.run_task(make_world(3101,0,"drift",17)); changed=ag.run_task(make_world(3102,3,"drift",21))
    assert first["exact_model"] and changed["mode"]=="DRIFT_RELEARNED" and changed["model_id"]==3
    out.append({"test":"contextual_drift_detected_and_relearned","pass":True})
    ood=TemporalGrowthAgent(AgentMemory()).run_task(make_world(3201,None,"ood",7,family=None,issuer="UNKNOWN",out_of_family=True))
    assert ood["mode"]=="FULL_TABLE_UNTRUSTED" and ood["success"] and ood["probe_queries"]>=90
    out.append({"test":"out_of_family_exact_full_table_fallback","pass":True,"probes":ood["probe_queries"]})
    spoof=TemporalGrowthAgent(AgentMemory()).run_task(make_world(3202,0,"spoof",9,issuer="ROGUE_ISSUER"))
    assert spoof["mode"]=="FULL_TABLE_UNTRUSTED" and spoof["success"]
    out.append({"test":"rogue_issuer_forces_full_table","pass":True})
    noisy=TemporalGrowthAgent(AgentMemory()).run_task(make_world(3203,0,"noise",9,noisy_repeat=True))
    assert noisy["mode"]=="HOLD_NONDETERMINISTIC" and not noisy["success"]
    out.append({"test":"repeated_observation_inconsistency_holds","pass":True})
    # Tampered checkpoint rejected.
    mm=AgentMemory(); mm.contexts["x"]={"model_id":0}; cp=mm.payload(); bad=json.loads(json.dumps(cp)); bad["contexts"]["x"]["model_id"]=1
    rejected=False
    try: AgentMemory.restore(bad)
    except ValueError: rejected=True
    assert rejected
    out.append({"test":"checkpoint_tamper_rejected","pass":True})
    return out


def run_panel(start_seed: int, n_seeds: int, outdir: Path) -> dict:
    outdir.mkdir(parents=True,exist_ok=True); runs=[]
    for s in range(start_seed,start_seed+n_seeds):
        r=run_seed(s)
        cpfile=outdir/f"checkpoint_{s}.json"
        cpfile.write_text(json.dumps(r["checkpoint"],indent=2,sort_keys=True))
        r.pop("checkpoint")
        runs.append(r)
    # Flatten and summarize.
    stable_mem=[x["memory"] for r in runs for x in r["stable"]]
    stable_fresh=[x["fresh_active"] for r in runs for x in r["stable"]]
    def descriptive(values):
        vals=list(values); mean=statistics.mean(vals) if vals else 0.0
        sd=statistics.stdev(vals) if len(vals)>1 else 0.0
        half=1.96*sd/math.sqrt(len(vals)) if vals else 0.0
        return {"mean_pct":round(mean,3),"sd_pct":round(sd,3),
                "ci95_low":round(mean-half,3),"ci95_high":round(mean+half,3),
                "min_pct":round(min(vals),3) if vals else 0.0,
                "max_pct":round(max(vals),3) if vals else 0.0}
    paired_train_probe=[100*(1-r["train"]["probe_queries"]/max(1,r["train_random"]["probe_queries"])) for r in runs]
    paired_train_inter=[100*(1-r["train"]["total_interactions"]/max(1,r["train_random"]["total_interactions"])) for r in runs]
    paired_stable_probe=[100*(1-sum(x["memory"]["probe_queries"] for x in r["stable"])/max(1,sum(x["fresh_active"]["probe_queries"] for x in r["stable"]))) for r in runs]
    paired_stable_inter=[100*(1-sum(x["memory"]["total_interactions"] for x in r["stable"])/max(1,sum(x["fresh_active"]["total_interactions"] for x in r["stable"]))) for r in runs]
    paired_drift_inter=[100*(1-r["drift"]["total_interactions"]/max(1,r["drift_fresh"]["total_interactions"])) for r in runs]
    summary={"version":"4.4","seeds":n_seeds,
      "paired_seed_descriptive_reductions":{"active_vs_random_probes":descriptive(paired_train_probe),
          "active_vs_random_total_interactions":descriptive(paired_train_inter),
          "memory_vs_fresh_stable_probes":descriptive(paired_stable_probe),
          "memory_vs_fresh_stable_total_interactions":descriptive(paired_stable_inter),
          "memory_vs_fresh_drift_total_interactions":descriptive(paired_drift_inter)},
      "train_exact":sum(bool(r["train"]["exact_model"]) for r in runs),
      "train_random_exact":sum(bool(r["train_random"]["exact_model"]) for r in runs),
      "train_active_probes":sum(r["train"]["probe_queries"] for r in runs),
      "train_random_probes":sum(r["train_random"]["probe_queries"] for r in runs),
      "train_active_vs_random_probe_reduction_pct":round((1-sum(r["train"]["probe_queries"] for r in runs)/max(1,sum(r["train_random"]["probe_queries"] for r in runs)))*100,3),
      "train_active_interactions":sum(r["train"]["total_interactions"] for r in runs),
      "train_random_interactions":sum(r["train_random"]["total_interactions"] for r in runs),
      "stable_memory_success":sum(bool(x["success"]) for x in stable_mem),
      "stable_fresh_success":sum(bool(x["success"]) for x in stable_fresh),
      "stable_memory_probes":sum(x["probe_queries"] for x in stable_mem),
      "stable_fresh_probes":sum(x["probe_queries"] for x in stable_fresh),
      "drift_relearned":sum(r["drift"]["mode"]=="DRIFT_RELEARNED" for r in runs),
      "drift_success":sum(bool(r["drift"]["success"]) for r in runs),
      "drift_fresh_success":sum(bool(r["drift_fresh"]["success"]) for r in runs),
      "new_context_success":sum(bool(r["new_context"]["success"]) for r in runs),
      "ood_full_table":sum(r["ood"]["mode"]=="FULL_TABLE_UNTRUSTED" for r in runs),
      "ood_success":sum(bool(r["ood"]["success"]) for r in runs),
      "spoof_full_table":sum(r["spoof"]["mode"]=="FULL_TABLE_UNTRUSTED" for r in runs),
      "spoof_success":sum(bool(r["spoof"]["success"]) for r in runs),
      "noisy_holds":sum(r["noisy"]["mode"]=="HOLD_NONDETERMINISTIC" for r in runs),
      "all_checkpoint_roundtrip":all(r["checkpoint_roundtrip"] for r in runs),
      "mean_train_probes":round(statistics.mean(r["train"]["probe_queries"] for r in runs),3),
      "mean_train_random_probes":round(statistics.mean(r["train_random"]["probe_queries"] for r in runs),3),
      "mean_transfer_probes":round(statistics.mean(x["memory"]["probe_queries"] for r in runs for x in r["stable"]),3),
      "mean_fresh_probes":round(statistics.mean(x["fresh_active"]["probe_queries"] for r in runs for x in r["stable"]),3),
      "memory_probe_reduction_pct":round((1-sum(x["probe_queries"] for x in stable_mem)/max(1,sum(x["probe_queries"] for x in stable_fresh)))*100,3),
      "mean_drift_probes":round(statistics.mean(r["drift"]["probe_queries"] for r in runs),3),
      "mean_ood_probes":round(statistics.mean(r["ood"]["probe_queries"] for r in runs),3),
      "mean_ood_interactions":round(statistics.mean(r["ood"]["total_interactions"] for r in runs),3),
      "stable_memory_interactions":sum(x["memory"]["total_interactions"] for r in runs for x in r["stable"]),
      "stable_fresh_interactions":sum(x["fresh_active"]["total_interactions"] for r in runs for x in r["stable"]),
      "stable_interaction_reduction_pct":round((1-sum(x["memory"]["total_interactions"] for r in runs for x in r["stable"])/max(1,sum(x["fresh_active"]["total_interactions"] for r in runs for x in r["stable"])))*100,3),
      "drift_memory_interactions":sum(r["drift"]["total_interactions"] for r in runs),
      "drift_fresh_interactions":sum(r["drift_fresh"]["total_interactions"] for r in runs),
      "mean_train_plan_length":round(statistics.mean(r["train"]["plan_length"] or 0 for r in runs),3)}
    (outdir/"panel_runs.json").write_text(json.dumps(runs,indent=2,sort_keys=True))
    (outdir/"panel_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--seed",type=int,default=20261119)
    ap.add_argument("--panel",type=int,default=20)
    ap.add_argument("--out",type=Path,default=Path("results/panel20"))
    args=ap.parse_args()
    if args.self_test:
        print(json.dumps(self_tests(),indent=2)); return
    print(json.dumps(run_panel(args.seed,args.panel,args.out),indent=2))

if __name__=="__main__": main()
