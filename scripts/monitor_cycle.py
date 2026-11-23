"""One-line cycle health: chunk count, median time, chat/critique/semantic."""
import os, re, time
LOG = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "checkpoints", "cycle.log")
raw = open(LOG, "rb").read()
txt = raw.decode("utf-16-le", errors="ignore")
ns = re.findall(r"\[(ns_\d+)\] neural: .*?CE (\d+\.\d+) \((\d+\.\d)s\)", txt)
secs = sorted(float(s) for _, _, s in ns)
sem = re.findall(r"semantic loss (\d+\.\d+)", txt)
fresh = int(time.time() - os.path.getmtime(LOG))
print(f"fresh {fresh}s | chunks {len(ns)} last {ns[-1] if ns else None} | "
      f"median {secs[len(secs) // 2] if secs else 0:.0f}s | "
      f"chat {txt.count('[chat]')} crit {txt.count('critique stream')} | "
      f"sem {sem[:1]}->{sem[-1:]} | err {txt.count('Traceback')}")
