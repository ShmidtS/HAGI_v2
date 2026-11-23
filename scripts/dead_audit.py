"""Top-level names that no entry point (other modules, scripts, tests, CLI
workflows) can reach. Read-only report. Unreachable != dead: check
primes source-anchors and docs before deleting."""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def read(p):
    return open(p, encoding="utf-8", errors="ignore").read()


def sources(exclude):
    for base in ("hagi", "scripts", "tests", "training", ".github"):
        for r, _, fs in os.walk(base):
            for f in fs:
                p = os.path.join(r, f)
                if (os.path.normpath(p) != os.path.normpath(exclude)
                        and f.endswith((".py", ".yml", ".yaml", ".sh", ".ps1"))):
                    yield p
    for f in os.listdir("."):
        if f.endswith(".py"):
            yield f


def markdown():
    for r, _, fs in os.walk("."):
        if "checkpoints" in r or ".git" in r:
            continue
        for f in fs:
            if f.endswith(".md"):
                yield os.path.join(r, f)


def nodes(tree, lines):
    """name -> source text. Nodes: every top-level def/class and every
    name BOUND at module level, including inside `if`/`try`/`for`
    blocks; a binding node's text is the whole top-level statement that
    binds it (so `QUOTIENTS.append(q_shift(d))` inside `if` counts).
    Non-binding top-level statements form the pseudo-node
    `<module code>`. Blind spots fixed along the way: entry nodes never
    marked reachable, the module leaking into its own corpus,
    defs-only traversal, pseudo-node spans swallowing unrelated defs."""
    out = {}
    MODULE = "<module code>"

    def text(stmt):
        return "\n".join(lines[stmt.lineno - 1:stmt.end_lineno])

    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out[n.name] = text(n)
            continue
        owner = text(n)
        out[MODULE] = out.get(MODULE, "") + "\n" + owner
        for st in ast.walk(n):
            if isinstance(st, ast.Assign):
                targets = st.targets
            elif isinstance(st, ast.AnnAssign):
                targets = [st.target]
            else:
                continue
            for t_ in targets:
                if isinstance(t_, ast.Name):
                    out.setdefault(t_.id, owner)
    return out


mod = sys.argv[1]
path = f"hagi/{mod}.py"
src = read(path)
tree = ast.parse(src)
lines = src.splitlines()
tops = nodes(tree, lines)
funcs = {n.name: (n.lineno, n.end_lineno) for n in tree.body
         if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
corpus = list(sources(path))
assert os.path.normpath(path) not in {os.path.normpath(p) for p in corpus}, \
    "module leaked into its own corpus"
others = "\n".join(read(p) for p in corpus)
docs = "\n".join(read(p) for p in markdown())
word = lambda n: r"\b%s\b" % re.escape(n)
deps = {n: {m for m in tops if m != n and re.search(word(m), b)}
        for n, b in tops.items()}
entry = {n for n in tops if re.search(word(n), others)} | {"main"}
entry.add("<module code>")          # module-level statements RUN at import
reach = {e for e in entry if e in tops}   # entries themselves are reachable
stack = list(reach)
while stack:
    n = stack.pop()
    for m in deps[n]:
        if m not in reach:
            reach.add(m)
            stack.append(m)
total = 0
for n in sorted(tops):
    if n in reach or n not in funcs:
        continue
    a, b = funcs[n]
    total += b - a + 1
    print(f"{path}:{a}-{b}  {n:34} {b - a + 1:4} lines  "
          f"code={'yes' if re.search(word(n), others) else 'no '} "
          f"docs={'yes' if re.search(word(n), docs) else 'no'}")
print(f"{path}: unreachable {total} lines")
