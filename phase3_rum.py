"""
Phase 3 — RUM (Runner-Up hard-negative Mining) for the CPRD distillation set.

The teacher labeled the retrieval pool (all HIGH cosine). RUM's insight: the dangerous
negatives are the "runner-ups" — candidates the encoder ranks high (high cosine) that the
teacher judged LOW. Those are the confusable cases the distilled encoder must learn to push
DOWN. We assemble:
  positive  : teacher tier >= 4          (genuine fits — pull toward JD)
  mid       : teacher tier == 3          (relevant)
  hard_neg  : teacher tier <= 2          (retrieved but low-fit = runner-ups)
  honeypot  : gate-flagged or tier 0     (hardest negatives — push far)
  easy_neg  : random non-pool candidates (obviously irrelevant; low cosine by construction)

Output artifacts/rum_train.jsonl : {candidate_id, role, target(0-100), tier, cosine}
target = teacher fit for labeled; honeypot=0; easy_neg=2. Phase 4 looks up embeddings by id.
"""
import os
import sys
import json
import random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import honeypot

ART = os.path.join(HERE, "artifacts")
CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
        r"\India_runs_data_and_ai_challenge\candidates.jsonl")
N_EASY = 400
random.seed(42)

LBL = os.path.join(ART, "teacher_labels.jsonl")
if not os.path.exists(LBL):
    sys.exit(
        "No teacher_labels.jsonl found. The teacher step is OPTIONAL — create the labels with EITHER:\n"
        "  python phase2_teacher.py --limit 400 --model claude-opus-4-8   (LLM teacher; needs ANTHROPIC key)\n"
        "  python phase2_fallback.py                                       (deterministic heuristic; no key)"
    )
labels = [json.loads(l) for l in open(LBL, encoding="utf-8") if l.strip()]
by_id = {x["candidate_id"]: x for x in labels}
pool = json.load(open(os.path.join(ART, "retrieval_topk.json"), encoding="utf-8"))
pool_ids = {p["id"] for p in pool}
all_ids = json.load(open(os.path.join(ART, "cand_ids.json"), encoding="utf-8"))

# records for the labeled set (for honeypot flagging + display)
recs = {}
need = set(by_id)
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


def role_of(cid):
    lab = by_id[cid]
    tier = lab.get("tier")
    if honeypot.flags(recs.get(cid, {})) or tier == 0 or lab.get("archetype") == "honeypot":
        return "honeypot", 0.0
    if tier is not None and tier >= 4:
        return "positive", float(lab.get("fit", 80))
    if tier == 3:
        return "mid", float(lab.get("fit", 60))
    return "hard_neg", float(lab.get("fit", 30))


train = []
for cid in by_id:
    role, target = role_of(cid)
    train.append({"candidate_id": cid, "role": role, "target": target,
                  "tier": by_id[cid].get("tier"), "cosine": round(by_id[cid].get("retrieval_score", 0), 4)})

# easy negatives — random candidates outside the retrieval pool (low cosine by construction)
outside = [i for i in all_ids if i not in pool_ids and i not in by_id]
for cid in random.sample(outside, N_EASY):
    train.append({"candidate_id": cid, "role": "easy_neg", "target": 2.0, "tier": None, "cosine": None})

with open(os.path.join(ART, "rum_train.jsonl"), "w", encoding="utf-8") as f:
    for r in train:
        f.write(json.dumps(r) + "\n")

from collections import Counter
roles = Counter(r["role"] for r in train)
print("RUM training set:", dict(roles), "| total", len(train))

# show the most dangerous hard negatives / honeypots: highest cosine but low fit
conf = [r for r in train if r["role"] in ("hard_neg", "honeypot") and r["cosine"]]
conf.sort(key=lambda r: -r["cosine"])
print("\n-- top runner-up hard-negatives (high cosine, low teacher fit) --")
for r in conf[:12]:
    c = recs[r["candidate_id"]]
    p = c.get("profile", {}) or {}
    print(f"  cos={r['cosine']:.3f} tier={r['tier']} {r['role']:8s} {r['candidate_id']} "
          f"{str(p.get('current_title'))[:24]:24s} @ {str(p.get('current_company'))[:14]}")
print("\nsaved", os.path.join(ART, "rum_train.jsonl"))
