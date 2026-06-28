"""
FP-test candidate honeypot rules against the full 100K before adopting any.
Reports: fire-count on 100K (precision proxy — should stay near ~80, not thousands),
and recall on the 9 gate-missed teacher honeypots. Adopt only precise rules.
"""
import os
import sys
import json
from datetime import date
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import honeypot

CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
        r"\India_runs_data_and_ai_challenge\candidates.jsonl")
ART = os.path.join(HERE, "artifacts")
TODAY = date(2026, 6, 28)


def parse(s):
    try:
        y, m, d = str(s).split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def months(a, b):
    return (b.year - a.year) * 12 + (b.month - a.month)


def feats(c):
    p = c.get("profile", {}) or {}
    yoe = p.get("years_of_experience") or 0
    yoe_m = yoe * 12
    ch = c.get("career_history", []) or []
    sk = c.get("skills", []) or []
    durs = [j.get("duration_months") or 0 for j in ch]
    starts = [parse(j.get("start_date")) for j in ch if parse(j.get("start_date"))]
    ends = [parse(j.get("end_date")) if j.get("end_date") else TODAY for j in ch]
    ends = [e for e in ends if e]
    span_m = months(min(starts), max(ends)) if starts and ends else None
    return {
        "yoe_m": yoe_m, "yoe": yoe,
        "sum": sum(durs),
        "max_skill": max([s.get("duration_months") or 0 for s in sk], default=0),
        "span_m": span_m,
    }


def rules(f):
    yoe_m, s, mxs, span = f["yoe_m"], f["sum"], f["max_skill"], f["span_m"]
    out = {}
    out["A_skill_gt_yoe+12 (yoe>=4)"] = (f["yoe"] >= 4 and mxs > yoe_m + 12)
    out["A2_skill_gt_yoe+24 (yoe>=4)"] = (f["yoe"] >= 4 and mxs > yoe_m + 24)
    out["B_sum<0.6*yoe"] = (yoe_m > 0 and s < 0.6 * yoe_m)
    out["B2_sum<0.5*yoe & yoe>=10"] = (f["yoe"] >= 10 and s < 0.5 * yoe_m)
    out["C_yoe>span+12"] = (span is not None and yoe_m > span + 12)
    out["C2_yoe>span+24"] = (span is not None and yoe_m > span + 24)
    return out


# known honeypots: teacher tier0/honeypot, and which the deterministic gate misses
labels = [json.loads(l) for l in open(os.path.join(ART, "teacher_labels.jsonl"), encoding="utf-8") if l.strip()]
by_id = {x["candidate_id"]: x for x in labels}
teacher_hp = {i for i in by_id if by_id[i].get("archetype") == "honeypot" or by_id[i].get("tier") == 0}

counts = Counter()
catch_missed = Counter()
catch_all_hp = Counter()
missed_ids = set()
n = 0
with open(CAND, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        n += 1
        r = rules(feats(c))
        fired = [k for k, v in r.items() if v]
        for k in fired:
            counts[k] += 1
        cid = c["candidate_id"]
        if cid in teacher_hp:
            gate = bool(honeypot.flags(c))
            if not gate:
                missed_ids.add(cid)
                for k in fired:
                    catch_missed[k] += 1
            for k in fired:
                catch_all_hp[k] += 1

print(f"scanned {n}")
print(f"\ngate-missed teacher honeypots in labeled set: {len(missed_ids)}")
print(f"\n{'rule':<32} {'fires/100K':>10} {'/9missed':>9} {'/teacherHP':>11}")
for k in rules(feats({"profile": {}})).keys():
    print(f"{k:<32} {counts[k]:>10} {catch_missed[k]:>9} {catch_all_hp[k]:>11}")
print(f"\n(teacher honeypots total in labeled set: {len(teacher_hp)})")
print("Adopt a rule only if fires/100K stays small (precise) AND it catches real impossibilities.")
