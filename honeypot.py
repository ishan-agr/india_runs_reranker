"""
Honeypot detector (Phase 5 component) — deterministic logical-impossibility checks.

The spec says ~80 candidates have "subtly impossible profiles" (e.g. 8 yrs at a
company founded 3 yrs ago; 'expert' in 10 skills with 0 yrs used), forced to tier 0;
>10% honeypots in top-100 => disqualified. We can't see company-founding dates, so we
catch impossibilities expressible from the schema:

  R1 expert_zero_ge3      : >=3 skills at 'expert' with 0 months duration
                            ('expert' is rare overall, and 0-duration jumps 0->3)
  R2 position_gt_career   : a SINGLE job longer than the whole stated career (+12mo)
  R3 sum_gt_career        : sum of job durations >> career span (overlap; >1.5x +12mo)
  R4 skill_gt_lifetime    : a skill used longer than the candidate's adult lifetime
                            (months since earliest education start +12mo) — a true upper
                            bound, unlike professional years (juniors learn before working)
  R5 end_before_start     : a job whose end_date precedes its start_date
  R6 duration_date_mismatch: completed job where |months(start,end) - duration_months| > 12
  R7 future_date          : a start_date in the future
  R8 yoe_gt_career_span   : years_of_experience exceeds the elapsed time from the first job
                            to now by >2yr — claiming more experience than has physically
                            passed (provably impossible; FP-tested: 25/100K). Orthogonal to
                            sum_gt_career (which catches yoe-too-LOW; this catches yoe-too-HIGH).

Design goal: HIGH PRECISION. A flag must be a genuine impossibility so we never zero a
real candidate. flags() is importable by the rank-time gate; __main__ scans the pool,
reports per-rule counts/overlaps and dumps evidence for manual inspection.
"""
import json
import os
from datetime import date
from collections import Counter

CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
        r"\India_runs_data_and_ai_challenge\candidates.jsonl")
OUTDIR = r"E:\[PUB] India_runs_data_and_ai_challenge\redrob_ranker\artifacts"
TODAY = date(2026, 6, 28)


def _parse(s):
    try:
        y, m, d = str(s).split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def _months(a, b):
    return (b.year - a.year) * 12 + (b.month - a.month)


def flags(c):
    """Return dict of fired rules -> evidence. Empty dict == plausible."""
    f = {}
    prof = c.get("profile", {}) or {}
    yoe = prof.get("years_of_experience")
    yoe_m = yoe * 12 if isinstance(yoe, (int, float)) else None
    sk = c.get("skills", []) or []
    ch = c.get("career_history", []) or []

    ez = sum(1 for s in sk if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0)
    if ez >= 3:
        f["expert_zero_ge3"] = ez

    if yoe_m is not None:
        for p in ch:
            dm = p.get("duration_months") or 0
            if dm > yoe_m + 12:
                f["position_gt_career"] = max(f.get("position_gt_career", 0), dm)
        scm = sum((p.get("duration_months") or 0) for p in ch)
        if scm > yoe_m * 1.5 + 12:
            f["sum_gt_career"] = scm

    # skill used longer than the candidate's adult lifetime (since earliest edu start).
    edu_starts = [e.get("start_year") for e in (c.get("education", []) or [])
                  if isinstance(e.get("start_year"), int)]
    if edu_starts:
        lifetime_m = (TODAY.year - min(edu_starts)) * 12 + 6
        for s in sk:
            dm = s.get("duration_months") or 0
            if dm > lifetime_m + 12:
                f["skill_gt_lifetime"] = max(f.get("skill_gt_lifetime", 0), dm)

    for p in ch:
        sd = _parse(p.get("start_date"))
        ed_raw = p.get("end_date")
        if sd and sd > TODAY:
            f["future_date"] = True
        if ed_raw:  # only completed jobs (avoid 'today drift' on current roles)
            ed = _parse(ed_raw)
            if sd and ed:
                if ed < sd:
                    f["end_before_start"] = True
                dm = p.get("duration_months")
                if isinstance(dm, (int, float)) and abs(_months(sd, ed) - dm) > 12:
                    f["duration_date_mismatch"] = f.get("duration_date_mismatch", 0) + 1

    # R8: years_of_experience exceeds elapsed career time-span (provably impossible)
    starts = [s for s in (_parse(p.get("start_date")) for p in ch) if s]
    ends = [(_parse(p.get("end_date")) if p.get("end_date") else TODAY) for p in ch]
    ends = [e for e in ends if e]
    if starts and ends and yoe_m is not None:
        span = _months(min(starts), max(ends))
        if yoe_m > span + 24:
            f["yoe_gt_career_span"] = round(yoe_m - span)

    return f


def is_honeypot(c):
    return len(flags(c)) > 0


if __name__ == "__main__":
    rule_counts = Counter()
    nrules_hist = Counter()
    union = 0
    examples = {}
    multi = []  # candidates tripping >=2 rules (highest-confidence honeypots)

    with open(CAND, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            f = flags(c)
            if not f:
                continue
            union += 1
            nrules_hist[len(f)] += 1
            for r in f:
                rule_counts[r] += 1
                examples.setdefault(r, [])
                if len(examples[r]) < 4:
                    prof = c.get("profile", {}) or {}
                    examples[r].append({
                        "id": c["candidate_id"],
                        "title": prof.get("current_title"),
                        "yoe": prof.get("years_of_experience"),
                        "evidence": f[r],
                    })
            if len(f) >= 2:
                prof = c.get("profile", {}) or {}
                multi.append({"id": c["candidate_id"], "title": prof.get("current_title"),
                              "yoe": prof.get("years_of_experience"), "rules": f})

    summary = {
        "total_flagged_union": union,
        "per_rule_counts": dict(rule_counts.most_common()),
        "num_rules_fired_histogram": dict(sorted(nrules_hist.items())),
        "multi_rule_count": len(multi),
        "examples_per_rule": examples,
        "multi_rule_examples": multi[:25],
    }
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "honeypot_scan.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("UNION flagged:", union)
    print("per-rule:", dict(rule_counts.most_common()))
    print("num-rules-fired histogram:", dict(sorted(nrules_hist.items())))
    print("multi-rule (>=2) candidates:", len(multi))
    print("\n-- examples per rule --")
    print(json.dumps(examples, indent=2, ensure_ascii=False))
    print("\n-- multi-rule examples (highest confidence) --")
    print(json.dumps(multi[:15], indent=2, ensure_ascii=False))
