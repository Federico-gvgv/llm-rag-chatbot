import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/results.csv")

def to_bool(x):
    if isinstance(x, bool): return x
    if isinstance(x, str): return x.strip().lower() == "true"
    return bool(x)

rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
# Basic per-strategy accuracy
totals = Counter()
corrects = Counter()
by_q = defaultdict(dict)

for r in rows:
    s = r["strategy"]
    ok = to_bool(r["correct"])
    qid = r["id"] or r.get("question")  # prefer id
    totals[s] += 1
    corrects[s] += ok
    by_q[qid][s] = ok

print(f"File: {csv_path}\n")
for s in totals:
    acc = corrects[s]/totals[s] if totals[s] else 0.0
    print(f"{s:7} {corrects[s]}/{totals[s]} = {acc:.1%}")

# Head-to-head: where RAG beats / loses vs baseline
wins = [qid for qid, d in by_q.items() if d.get("rag") and not d.get("no_ctx")]
losses = [qid for qid, d in by_q.items() if d.get("no_ctx") and not d.get("rag")]

print("\nHead-to-head:")
print(f"- RAG wins vs baseline on {len(wins)} question(s): {wins}")
print(f"- RAG loses to baseline on {len(losses)} question(s): {losses}")

# Optional: write a tiny report file
out = Path("eval/summary.txt")
with out.open("w", encoding="utf-8") as f:
    for s in totals:
        acc = corrects[s]/totals[s] if totals[s] else 0.0
        f.write(f"{s}: {corrects[s]}/{totals[s]} = {acc:.1%}\n")
    f.write(f"\nRAG wins: {wins}\nRAG losses: {losses}\n")
print(f"\nSaved {out}")