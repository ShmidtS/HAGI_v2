"""Data IO: ingest any file as a domain; execute written programs."""
from __future__ import annotations


# ============================================================
# INGEST: learn from arbitrary data
# ============================================================

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable
from hagi import *
import hagi
def _int_or_none(v: str) -> int | None:
    try:
        return int(v.strip())
    except (ValueError, AttributeError):
        return None
def _numeric_columns(path: Path) -> dict[str, list[int]]:
    delim = '\t' if path.suffix.lower() in ('.tsv', '.tab') else ','
    cols: dict[str, list[int]] = {}
    with path.open('r', encoding='utf-8', errors='replace', newline='') as fh:
        reader = csv.reader(fh, delimiter=delim)
        for row in reader:
            if not row:
                continue
            vals = [_int_or_none(c) for c in row]
            if all((v is None for v in vals)):
                names = [c.strip() or f'c{i}' for i, c in enumerate(row)]
                for n in names:
                    cols.setdefault(n, [])
                continue
            if not cols:
                cols = {f'c{i}': [] for i in range(len(row))}
            keys = list(cols.keys())
            for i, v in enumerate(vals):
                if v is not None and i < len(keys):
                    cols[keys[i]].append(v)
    return cols
def _json_numbers(obj: Any) -> list[int]:
    out: list[int] = []

    def walk(o: Any) -> None:
        if isinstance(o, bool):
            return
        if isinstance(o, int):
            out.append(o)
        elif isinstance(o, float) and o.is_integer():
            out.append(int(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)
    return out
def _pair_tasks(xs: list[int], ys: list[int], name: str) -> list[hagi.TaskSpec]:
    """Exact-function test: y must be unambiguous for each observed x."""
    mapping: dict[int, int] = {}
    for x, y in zip(xs, ys):
        if x in mapping and mapping[x] != y:
            return []
        mapping[x] = y
    if len(mapping) < 4:
        return []
    dom = tuple(sorted(mapping))

    def teacher(*args, _m=mapping) -> int:
        v = _m.get(int(args[0]))
        if v is None:
            raise ValueError('outside ingested domain')
        return v
    return [hagi.TaskSpec(name, 1, (dom,), teacher, family='ingest')]
def tasks_from_numeric_file(path: Path, max_tasks: int=16) -> list[hagi.TaskSpec]:
    tasks: list[hagi.TaskSpec] = []
    if path.suffix.lower() == '.json':
        nums = _json_numbers(json.loads(path.read_text(encoding='utf-8', errors='replace')))
        if len(nums) >= 8:
            tasks += _pair_tasks(nums[::2][:512], nums[1::2][:512], f'INGEST_{path.stem}_json_pairs')
    else:
        cols = _numeric_columns(path)
        names = [n for n in cols if len(cols[n]) >= 8]
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j or len(tasks) >= max_tasks:
                    continue
                tasks += _pair_tasks(cols[names[i]], cols[names[j]], f'INGEST_{path.stem}_{names[i]}_to_{names[j]}')
    return tasks
def tasks_from_text_file(path: Path) -> list[hagi.TaskSpec]:
    data = path.read_bytes()
    nxt: dict[int, int] = {}
    conflict: set[int] = set()
    for a, b in zip(data, data[1:]):
        if a in nxt and nxt[a] != b:
            conflict.add(a)
        else:
            nxt[a] = b
    for a in conflict:
        nxt.pop(a, None)
    if len(nxt) < 8:
        return []
    dom = tuple(sorted(nxt))

    def teacher(*args, _m=nxt) -> int:
        v = _m.get(int(args[0]))
        if v is None:
            raise ValueError('outside deterministic subset')
        return v
    return [hagi.TaskSpec(f'INGEST_{path.stem}_nextbyte', 1, (dom,), teacher, family='ingest')]
def ingest_file(path: Path) -> list[hagi.TaskSpec]:
    if path.suffix.lower() == '.json':
        return tasks_from_numeric_file(path)
    if path.suffix.lower() in ('.csv', '.tsv', '.tab'):
        return tasks_from_numeric_file(path)
    tasks = tasks_from_text_file(path)
    numeric = tasks_from_numeric_file(path)
    return tasks + numeric
def learn_from_data(paths: list[Path], budget: int=200000) -> dict[str, Any]:
    proc = hagi.GrowingHagiProcessor(node_budget=budget)
    report: dict[str, Any] = {'files': [], 'tasks': 0, 'committed': 0}
    for path in paths:
        try:
            tasks = ingest_file(path)
        except Exception as e:
            report['files'].append({'file': str(path), 'error': repr(e)[:120]})
            continue
        f_info: dict[str, Any] = {'file': str(path), 'tasks': len(tasks), 'results': []}
        for sp in tasks:
            ev = proc.learn(sp, seed=10000, train_neural=True)
            f_info['results'].append({'task': sp.name, 'action': ev['action'], 'reason': ev.get('reason', '')[:40], 'program': ev.get('program', '')[:80]})
            report['tasks'] += 1
            report['committed'] += int(ev['action'] == 'COMMIT')
        report['files'].append(f_info)
    return report
def ingest_main() -> None:
    p = argparse.ArgumentParser(description='Learn exact programs from any data')
    p.add_argument('inputs', nargs='+', type=Path)
    p.add_argument('--budget', type=int, default=200000)
    p.add_argument('--json', action='store_true')
    a = p.parse_args()
    paths: list[Path] = []
    for i in a.inputs:
        paths.extend(sorted(i.iterdir()) if i.is_dir() else [i])
    rep = learn_from_data(paths, budget=a.budget)
    if a.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        for f in rep['files']:
            print(f"{f['file']}: {f.get('tasks', '?')} tasks")
            for r in f.get('results', []):
                print(f"  {r['task']}: {r['action']} ({r['reason']})")
        print(f"total: {rep['committed']}/{rep['tasks']} committed")

# ============================================================
# EXECUTE: run written programs
# ============================================================

import argparse
import importlib.util
from pathlib import Path
from hagi.util import CKPT_DIR as _CKPT
DEFAULT_PROGRAMS = _CKPT / 'self' / 'generated_programs.py'
_mod = None
_load_path: Path | None = None
def _load(path: Path | None=None):
    global _mod, _load_path
    if _mod is not None and (path or DEFAULT_PROGRAMS) == _load_path:
        return _mod
    p = path or DEFAULT_PROGRAMS
    _load_path = p
    if not p.exists():
        raise FileNotFoundError(f'{p} not found — run hagi_selfrewrite first so the model writes its programs')
    spec = importlib.util.spec_from_file_location('hagi_generated', p)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {p}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _mod = mod
    return mod
def available(path: Path | None=None) -> list[str]:
    return [n for n in dir(_load(path)) if not n.startswith('_') and n != 'hagi_generated']
def execute(name: str, *args: int, programs: Path | None=None) -> int:
    fn = getattr(_load(programs), name, None)
    if fn is None:
        raise KeyError(f'no program named {name!r} (model has {len(available())})')
    return int(fn(*args))
def execute_text(name: str, text: str, key: int | None=None) -> str:
    """Stream UTF-8 text byte-by-byte through a byte-level program."""
    data = text.encode('utf-8')
    arity = 2 if key is not None else 1
    out = bytearray()
    for b in data:
        out.append(execute(name, b, int(key or 0)) & 255 if arity == 2 else execute(name, b) & 255)
    return out.decode('utf-8', errors='replace')
def execute_main() -> None:
    p = argparse.ArgumentParser(description='Execute programs the HAGI model wrote')
    p.add_argument('--programs', type=Path, default=DEFAULT_PROGRAMS)
    p.add_argument('--list', action='store_true')
    p.add_argument('--task')
    p.add_argument('--text')
    p.add_argument('--key', type=int, default=None)
    p.add_argument('values', nargs='*', type=int, help='direct integer arguments, e.g. --task GCD 12 18')
    a = p.parse_args()
    if a.list:
        names = available(a.programs)
        print(f'{len(names)} programs:')
        for n in sorted(names):
            print(' ', n)
        return
    if not a.task:
        p.error('--task or --list required')
    if a.text is not None:
        print(execute_text(a.task, a.text, a.key))
    else:
        print(execute(a.task, *a.values, programs=a.programs))
if __name__ == '__main__':
    execute_main()