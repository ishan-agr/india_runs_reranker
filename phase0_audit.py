"""
Phase 0 audit — read-only streaming pass over candidates.jsonl.

Goal: confirm structure, measure signal distributions / sentinels / out-of-range
values, and quantify honeypot "smell" prevalence BEFORE we build anything.
This runs on the dev machine (not the constrained sandbox), but still streams
the 465 MB file line-by-line so memory stays tiny.

Output:
  - artifacts/phase0_audit.json  (full machine-readable summary)
  - concise highlights printed to stdout
"""
import json
import os
import re
from collections import Counter, defaultdict

CANDIDATES = r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
OUTDIR = r"E:\[PUB] India_runs_data_and_ai_challenge\redrob_ranker\artifacts"
os.makedirs(OUTDIR, exist_ok=True)

ID_RE = re.compile(r"^CAND_\d{7}$")
CURRENT_YEAR = 2026

# Substrings that mark Indian IT-services / consulting firms (JD down-weight target).
CONSULTING = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree", "ltimindtree", "hcl", "tech mahindra", "mphasis",
    "hexaware", "syntel", "larsen", "l&t infotech", "lti", "deloitte", "ibm global",
]

TOP_KEYS = ["candidate_id", "profile", "career_history", "education", "skills",
            "certifications", "languages", "redrob_signals"]

NUMERIC_SIGNALS = [
    "profile_completeness_score", "profile_views_received_30d",
    "applications_submitted_30d", "recruiter_response_rate",
    "avg_response_time_hours", "connection_count", "endorsements_received",
    "notice_period_days", "search_appearance_30d", "saved_by_recruiters_30d",
    "interview_completion_rate",
]
BOOL_SIGNALS = ["open_to_work_flag", "willing_to_relocate", "verified_email",
                "verified_phone", "linkedin_connected"]

# Documented valid ranges -> count out-of-range values.
RANGE_CHECKS = {
    "profile_completeness_score": (0, 100),
    "recruiter_response_rate": (0.0, 1.0),
    "notice_period_days": (0, 180),
    "interview_completion_rate": (0.0, 1.0),
}


def is_consult(name):
    name = (name or "").lower()
    return any(c in name for c in CONSULTING)


def pctile(s, q):
    if not s:
        return None
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def summarize(vals):
    if not vals:
        return None
    s = sorted(vals)
    return {
        "n": len(s),
        "min": round(s[0], 3), "p10": round(pctile(s, 0.10), 3),
        "p25": round(pctile(s, 0.25), 3), "median": round(pctile(s, 0.50), 3),
        "mean": round(sum(s) / len(s), 3), "p75": round(pctile(s, 0.75), 3),
        "p90": round(pctile(s, 0.90), 3), "max": round(s[-1], 3),
    }


# accumulators
count = bad_id = dup_ids = 0
ids = set()
missing_top = Counter()
num_acc = defaultdict(list)
range_viol = Counter()
bool_true = Counter()
bool_total = Counter()
gh_sentinel = 0
gh_real = []
oar_sentinel = 0
oar_real = []
workmode = Counter()
country_c = Counter()
city_c = Counter()
industry_c = Counter()
csize_c = Counter()
edu_tier_c = Counter()
prof_c = Counter()
title_c = Counter()
assess_coverage = []  # #skills with assessment score

yoe_list, n_positions, n_skills, sum_career_months = [], [], [], []
consulting_current = consulting_only = 0

# honeypot smells (over-inclusive heuristics, NOT the final detector)
expert_zero_counts = []   # per-candidate count of expert-proficiency, 0-duration skills
hp_skill_gt_career = []
hp_sum_gt_yoe = []
hp_yoe_gt_edu = []
examples = defaultdict(list)


def add_example(tag, cid, extra=None):
    if len(examples[tag]) < 8:
        examples[tag].append(cid if extra is None else {cid: extra})


with open(CANDIDATES, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        count += 1
        try:
            r = json.loads(line)
        except Exception:
            add_example("parse_error", str(count))
            continue

        for k in TOP_KEYS:
            if k not in r:
                missing_top[k] += 1

        cid = r.get("candidate_id", "")
        if not ID_RE.match(cid):
            bad_id += 1
        if cid in ids:
            dup_ids += 1
        else:
            ids.add(cid)

        prof = r.get("profile", {}) or {}
        yoe = prof.get("years_of_experience")
        if isinstance(yoe, (int, float)):
            yoe_list.append(yoe)
        country_c[prof.get("country", "?")] += 1
        city_c[prof.get("location", "?")] += 1
        industry_c[prof.get("current_industry", "?")] += 1
        csize_c[prof.get("current_company_size", "?")] += 1
        title_c[(prof.get("current_title") or "?").strip().lower()] += 1

        ch = r.get("career_history", []) or []
        n_positions.append(len(ch))
        scm = sum((p.get("duration_months") or 0) for p in ch)
        sum_career_months.append(scm)
        companies = [(p.get("company") or "") for p in ch]
        if is_consult(prof.get("current_company")):
            consulting_current += 1
        if companies and all(is_consult(c) for c in companies):
            consulting_only += 1
            add_example("consulting_only", cid)

        edu = r.get("education", []) or []
        for e in edu:
            edu_tier_c[e.get("tier", "?")] += 1
        edu_starts = [e.get("start_year") for e in edu if isinstance(e.get("start_year"), int)]

        sk = r.get("skills", []) or []
        n_skills.append(len(sk))
        for s in sk:
            prof_c[s.get("proficiency", "?")] += 1

        sig = r.get("redrob_signals", {}) or {}
        for key in NUMERIC_SIGNALS:
            v = sig.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                num_acc[key].append(v)
                if key in RANGE_CHECKS:
                    lo, hi = RANGE_CHECKS[key]
                    if v < lo or v > hi:
                        range_viol[key] += 1
        for key in BOOL_SIGNALS:
            v = sig.get(key)
            if isinstance(v, bool):
                bool_total[key] += 1
                if v:
                    bool_true[key] += 1
        gh = sig.get("github_activity_score")
        if gh == -1:
            gh_sentinel += 1
        elif isinstance(gh, (int, float)):
            gh_real.append(gh)
        oar = sig.get("offer_acceptance_rate")
        if oar == -1:
            oar_sentinel += 1
        elif isinstance(oar, (int, float)):
            oar_real.append(oar)
        workmode[sig.get("preferred_work_mode", "?")] += 1
        aclen = len(sig.get("skill_assessment_scores", {}) or {})
        assess_coverage.append(aclen)

        # --- honeypot smells ---
        yoe_m = (yoe * 12) if isinstance(yoe, (int, float)) else None
        ez = sum(1 for s in sk if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0)
        expert_zero_counts.append(ez)
        if ez >= 3:
            add_example("expert_zero_ge3", cid, ez)
        if yoe_m is not None:
            if any((s.get("duration_months") or 0) > yoe_m + 6 for s in sk):
                hp_skill_gt_career.append(cid)
                add_example("skill_gt_career", cid)
            if scm > yoe_m * 1.5 + 12:
                hp_sum_gt_yoe.append(cid)
                add_example("sum_gt_yoe", cid)
        if edu_starts and isinstance(yoe, (int, float)):
            if yoe > (CURRENT_YEAR - min(edu_starts)) + 1:
                hp_yoe_gt_edu.append(cid)
                add_example("yoe_gt_edu", cid)

ez_dist = Counter(expert_zero_counts)
summary = {
    "record_count": count,
    "unique_ids": len(ids),
    "bad_id_format": bad_id,
    "duplicate_ids": dup_ids,
    "missing_top_keys": dict(missing_top),
    "profile": {
        "years_of_experience": summarize(yoe_list),
        "top_countries": country_c.most_common(12),
        "top_cities": city_c.most_common(15),
        "top_current_industries": industry_c.most_common(15),
        "current_company_size": csize_c.most_common(),
        "top_current_titles": title_c.most_common(20),
    },
    "career_history": {
        "num_positions": summarize(n_positions),
        "sum_duration_months": summarize(sum_career_months),
        "consulting_current_company": consulting_current,
        "consulting_only_entire_career": consulting_only,
    },
    "education": {"tier_distribution": edu_tier_c.most_common()},
    "skills": {
        "num_skills": summarize(n_skills),
        "proficiency_distribution": prof_c.most_common(),
    },
    "signals_numeric": {k: summarize(v) for k, v in num_acc.items()},
    "signals_range_violations": dict(range_viol),
    "signals_bool_true_rate": {
        k: round(bool_true[k] / bool_total[k], 3) for k in BOOL_SIGNALS if bool_total[k]
    },
    "github_activity_score": {"sentinel_minus1": gh_sentinel, "real": summarize(gh_real)},
    "offer_acceptance_rate": {"sentinel_minus1": oar_sentinel, "real": summarize(oar_real)},
    "preferred_work_mode": workmode.most_common(),
    "skill_assessment_coverage": summarize(assess_coverage),
    "honeypot_smells": {
        "expert_zero_duration_count_distribution": dict(sorted(ez_dist.items())),
        "candidates_with_expert_zero_ge1": sum(v for k, v in ez_dist.items() if k >= 1),
        "candidates_with_expert_zero_ge3": sum(v for k, v in ez_dist.items() if k >= 3),
        "candidates_with_expert_zero_ge5": sum(v for k, v in ez_dist.items() if k >= 5),
        "skill_duration_gt_career": len(hp_skill_gt_career),
        "sum_career_gt_1p5x_yoe": len(hp_sum_gt_yoe),
        "yoe_gt_education_span": len(hp_yoe_gt_edu),
    },
    "examples": {k: v for k, v in examples.items()},
}

with open(os.path.join(OUTDIR, "phase0_audit.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)


def show(title, obj):
    print("\n== " + title + " ==")
    print(json.dumps(obj, indent=2, ensure_ascii=False))


print("Records: %d | unique_ids: %d | bad_id: %d | dup: %d" % (count, len(ids), bad_id, dup_ids))
print("Missing top keys:", dict(missing_top))
show("years_of_experience", summarize(yoe_list))
show("education tier", edu_tier_c.most_common())
show("skill proficiency", prof_c.most_common())
show("numeric signals", {k: summarize(v) for k, v in num_acc.items()})
show("range violations", dict(range_viol))
show("bool true-rate", summary["signals_bool_true_rate"])
show("github sentinel/real", summary["github_activity_score"])
show("offer_acceptance sentinel/real", summary["offer_acceptance_rate"])
show("work mode", workmode.most_common())
show("consulting", summary["career_history"])
show("top countries", country_c.most_common(12))
show("top cities", city_c.most_common(15))
show("top current titles", title_c.most_common(20))
show("HONEYPOT SMELLS", summary["honeypot_smells"])
print("\nWrote", os.path.join(OUTDIR, "phase0_audit.json"))
