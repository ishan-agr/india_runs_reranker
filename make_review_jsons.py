"""
Build two review JSONs from the 400 teacher-labeled candidates:
  artifacts/honeypots_review.json  — the 12 honeypots (3 gate + 9 teacher-only) with evidence
  artifacts/top10_review.json      — top 10 by teacher fit, teacher vs our content-fit side by side
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import honeypot
import content_fit

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


def profile_view(c):
    p = c.get("profile", {}) or {}
    s = c.get("redrob_signals", {}) or {}
    return {
        "headline": p.get("headline"),
        "current": f"{p.get('current_title')} @ {p.get('current_company')} "
                   f"({p.get('current_industry')}, {p.get('current_company_size')})",
        "years_of_experience": p.get("years_of_experience"),
        "summary": (p.get("summary") or "")[:300],
        "career_history": [{
            "title": j.get("title"), "company": j.get("company"), "industry": j.get("industry"),
            "duration_months": j.get("duration_months"), "is_current": j.get("is_current"),
            "dates": f"{j.get('start_date')}..{j.get('end_date')}",
            "description": (j.get("description") or "")[:220],
        } for j in (c.get("career_history", []) or [])],
        "skills": [{"name": k.get("name"), "proficiency": k.get("proficiency"),
                    "endorsements": k.get("endorsements"), "duration_months": k.get("duration_months")}
                   for k in (c.get("skills", []) or [])],
        "education": [{"institution": e.get("institution"), "degree": e.get("degree"),
                       "field": e.get("field_of_study"), "tier": e.get("tier"),
                       "years": f"{e.get('start_year')}-{e.get('end_year')}"}
                      for e in (c.get("education", []) or [])],
        "signals": {"response_rate": s.get("recruiter_response_rate"),
                    "last_active": s.get("last_active_date"), "notice_days": s.get("notice_period_days"),
                    "open_to_work": s.get("open_to_work_flag"),
                    "interview_completion": s.get("interview_completion_rate"),
                    "github": s.get("github_activity_score"), "offer_accept": s.get("offer_acceptance_rate"),
                    "verified_email": s.get("verified_email"), "verified_phone": s.get("verified_phone"),
                    "completeness": s.get("profile_completeness_score")},
    }


def diagnostics(c):
    p = c.get("profile", {}) or {}
    yoe = p.get("years_of_experience") or 0
    ch = c.get("career_history", []) or []
    sk = c.get("skills", []) or []
    durs = [j.get("duration_months") or 0 for j in ch]
    edu_starts = [e.get("start_year") for e in (c.get("education", []) or []) if isinstance(e.get("start_year"), int)]
    return {
        "yoe_months": round(yoe * 12, 1),
        "sum_career_months": sum(durs),
        "max_position_months": max(durs) if durs else 0,
        "num_positions": len(ch),
        "max_skill_duration_months": max([k.get("duration_months") or 0 for k in sk], default=0),
        "expert_zero_count": sum(1 for k in sk if k.get("proficiency") == "expert" and (k.get("duration_months") or 0) == 0),
        "num_expert": sum(1 for k in sk if k.get("proficiency") == "expert"),
        "lifetime_months_since_edu_start": (2026 - min(edu_starts)) * 12 + 6 if edu_starts else None,
        "sum_over_yoe_ratio": round(sum(durs) / (yoe * 12), 2) if yoe else None,
    }


# ---- honeypots ----
gate = {i for i in by_id if honeypot.flags(recs[i])}
teacher = {i for i in by_id if by_id[i].get("archetype") == "honeypot" or by_id[i].get("tier") == 0}
allhp = sorted(gate | teacher, key=lambda i: (i not in gate, i))  # gate ones first
hp_out = []
for i in allhp:
    lab = by_id[i]
    src = "both" if i in gate and i in teacher else ("gate" if i in gate else "teacher_only")
    hp_out.append({
        "candidate_id": i, "detected_by": src,
        "deterministic_flags": honeypot.flags(recs[i]),
        "diagnostics": diagnostics(recs[i]),
        "teacher": {"tier": lab.get("tier"), "fit": lab.get("fit"), "archetype": lab.get("archetype"),
                    "rationale": lab.get("rationale"), "concerns": lab.get("concerns")},
        "profile": profile_view(recs[i]),
    })
json.dump(hp_out, open(os.path.join(ART, "honeypots_review.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)

# ---- top 10 by teacher fit ----
ranked = sorted(labels, key=lambda x: (x.get("fit", 0), x.get("retrieval_score", 0)), reverse=True)[:10]
top_out = []
for rank, lab in enumerate(ranked, 1):
    c = recs[lab["candidate_id"]]
    mult, arch, reasons = content_fit.content_fit(c)
    top_out.append({
        "rank": rank, "candidate_id": lab["candidate_id"],
        "teacher": {"tier": lab.get("tier"), "fit": lab.get("fit"), "archetype": lab.get("archetype"),
                    "rationale": lab.get("rationale"), "concerns": lab.get("concerns"),
                    "retrieval_cosine": round(lab.get("retrieval_score", 0), 3)},
        "our_content_fit": {"content_mult": mult, "archetype": arch,
                            "in_domain": reasons.get("in_domain"), "company": reasons.get("company"),
                            "musthave_hits": reasons.get("musthave_hits")},
        "profile": profile_view(c),
    })
json.dump(top_out, open(os.path.join(ART, "top10_review.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)

print(f"honeypots: {len(hp_out)} (gate {len(gate)}, teacher {len(teacher)}, both {len(gate & teacher)})")
print("wrote honeypots_review.json and top10_review.json")
print("\n-- teacher-only honeypots (gate missed) — quick diagnostics --")
for h in hp_out:
    if h["detected_by"] == "teacher_only":
        d = h["diagnostics"]
        print(f"  {h['candidate_id']} {h['profile']['current'][:40]:40s} "
              f"yoe_m={d['yoe_months']} sum={d['sum_career_months']} ratio={d['sum_over_yoe_ratio']} "
              f"maxpos={d['max_position_months']} ez={d['expert_zero_count']} nexp={d['num_expert']} "
              f"maxskill={d['max_skill_duration_months']}")
        print(f"      concern: {h['teacher']['concerns'][:150]}")
