"""Inspect teacher labels: tier/archetype spread, honeypot handling, teacher-vs-cosine reorder."""
import os
import sys
import json
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import honeypot
import company_quality

ART = os.path.join(HERE, "artifacts")
CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
        r"\India_runs_data_and_ai_challenge\candidates.jsonl")

labels = [json.loads(l) for l in open(os.path.join(ART, "teacher_labels.jsonl"), encoding="utf-8") if l.strip()]
by_id = {x["candidate_id"]: x for x in labels}
need = set(by_id)
recs = {}
with open(CAND, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        if c["candidate_id"] in need:
            recs[c["candidate_id"]] = c
            if len(recs) == len(need):
                break

rows = []
for cid, lab in by_id.items():
    c = recs.get(cid, {})
    prof = c.get("profile", {}) or {}
    hp = bool(honeypot.flags(c))
    rows.append((lab.get("fit", 0), lab.get("tier"), lab.get("archetype"), cid,
                 prof.get("current_title"), prof.get("current_company"),
                 round(lab.get("retrieval_score", 0), 3), hp, lab.get("concerns", "")))
rows.sort(reverse=True)

print(f"{'fit':>3} {'T':>1} {'archetype':<13} {'id':<13} {'title':<24} {'company':<12} {'cos':>5} hp")
for fit, tier, arch, cid, title, comp, cos, hp, conc in rows:
    print(f"{fit:>3} {tier!s:>1} {str(arch):<13} {cid:<13} {str(title)[:24]:<24} {str(comp)[:12]:<12} {cos:>5.3f} {'HP' if hp else ''}")

print("\ntier histogram:", dict(sorted(Counter(l.get('tier') for l in labels).items())))
print("archetype histogram:", dict(Counter(l.get('archetype') for l in labels).most_common()))

# honeypot cross-check: did teacher tier-0 / flag the impossible ones?
print("\n-- honeypot cross-check --")
for cid in by_id:
    if honeypot.flags(recs.get(cid, {})):
        lab = by_id[cid]
        print(f"  {cid} deterministic=HONEYPOT -> teacher tier={lab.get('tier')} arch={lab.get('archetype')}")

# teacher vs cosine reorder
try:
    from scipy.stats import spearmanr
    cos = [l.get("retrieval_score", 0) for l in labels]
    fit = [l.get("fit", 0) for l in labels]
    rho, p = spearmanr(cos, fit)
    print(f"\nSpearman(retrieval_cosine, teacher_fit) = {rho:.3f}  (low => teacher reorders the compressed top)")
except Exception as e:
    print("spearman skipped:", e)
