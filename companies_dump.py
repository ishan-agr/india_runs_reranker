"""Dump every distinct company with count + dominant industry + dominant size (for curation)."""
import json
from collections import Counter, defaultdict

CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
        r"\India_runs_data_and_ai_challenge\candidates.jsonl")
comp = Counter()
ind = defaultdict(Counter)
size = defaultdict(Counter)
with open(CAND, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        for j in c.get("career_history", []) or []:
            nm = (j.get("company") or "").strip()
            if not nm:
                continue
            comp[nm] += 1
            if j.get("industry"):
                ind[nm][j["industry"]] += 1
            if j.get("company_size"):
                size[nm][j["company_size"]] += 1
for nm, ct in comp.most_common():
    i = ind[nm].most_common(1)[0][0] if ind[nm] else "?"
    s = size[nm].most_common(1)[0][0] if size[nm] else "?"
    print(f"{ct:7d}  {nm:26s} ind={i:18s} size={s}")
print("TOTAL distinct:", len(comp))
