"""
Content-fit combiner (Phase 5) — the hybrid recruiter-intuition engine.

Authority model (your call): hybrid = HARD CAPS (vetoes) + bounded ± nudges + a rescue
floor. Order of operations:
  1. honeypot -> hard gate to 0.0 (excluded).
  2. bounded ± nudges (company, yoe-band, must-have evidence, college, depth-flavor,
     in-domain title, title-chaser) -> a multiplier in ~[0.6, 1.35].
  3. hard-cap CEILINGS (take min): keyword-stuffer, wrong-domain-primary, services-only-
     no-product. These barely fire in the clean retrieval pool (insurance), but matter on
     a swapped/broader set.
  4. rescue FLOOR: a genuine product/AI fit that lacks jargon isn't buried for low nudge.

Returns (content_mult, archetype, reasons). Combined at rank-time as
  final = base_fit(cosine) * content_mult * behavioral_mult     (honeypot/DQ gates zero it)
Weights are v1 defaults tagged CALIB -> fit against Phase-2 teacher labels, ablated in Phase 7.
behavioral availability lives in behavioral.py (kept separate); 'unavailable' is decided at
combine time, not here.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jd_spec
import company_quality
import honeypot

# ---- CALIB: tunable weights (fit vs teacher labels in Phase 2 calibration) ----
W_COMPANY_BASE, W_COMPANY_SLOPE = 0.85, 0.30   # company factor = base + slope*best_score
AI_NATIVE_BOOST = 1.05
COLLEGE_T1, COLLEGE_T2 = 1.05, 1.02
MUSTHAVE_PER_GROUP = 0.05                       # per core must-have group hit (max 3)
DEPTH_BOOST = 1.03
INDOMAIN_TITLE_BOOST = 1.05
TITLECHASER_PENALTY = 0.90
NUDGE_CLAMP = (0.60, 1.35)
# hard-cap ceilings
CAP_STUFFER, CAP_WRONGDOMAIN, CAP_SERVICES_ONLY = 0.20, 0.40, 0.60
RESCUE_FLOOR = 0.95
FINAL_CLAMP = (0.10, 1.40)

DOMAIN_TERMS = ["ml engineer", "machine learning", "ai engineer", "applied scien",
                "data scientist", "nlp", "search engineer", "ranking", "recommendation",
                "research engineer", "ml ", "ai/ml"]
SCRATCH = ["from scratch", "first principle", "implemented the", "built our own",
           "designed the algorithm", "ground up", "own implementation", "without relying"]


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _has_domain(text):
    return any(t in text for t in DOMAIN_TERMS)


FEATURE_ORDER = [
    "company_best", "company_current", "ever_ai_native", "ever_product", "services_current",
    "yoe", "yoe_ideal", "yoe_band", "musthave_hits", "in_domain_title", "depth",
    "college_tier1", "college_tier2", "num_positions", "avg_tenure_mo", "ai_density",
    "nontech_title", "skill_count", "endorse_total",
]


def content_features(c):
    """Interpretable numeric features for calibration (regressed onto teacher fit)."""
    prof = c.get("profile", {}) or {}
    blob = jd_spec.text_blob(c)
    cur_title = (prof.get("current_title") or "").lower()
    yoe = prof.get("years_of_experience") or 0
    co = company_quality.company_signals(c)
    core = ["embeddings_retrieval", "vector_db_hybrid_search", "ranking_eval"]
    musthave = sum(1 for g in core if jd_spec.any_term(blob, jd_spec.MUST_HAVES[g]))
    tiers = [(e.get("tier") or "") for e in (c.get("education", []) or [])]
    titles_text = " ".join([cur_title] + [(j.get("title") or "").lower()
                                          for j in (c.get("career_history", []) or [])])
    hist = c.get("career_history", []) or []
    durs = [j.get("duration_months") or 0 for j in hist]
    skills = c.get("skills", []) or []
    return {
        "company_best": co["best_score"], "company_current": co["current_score"],
        "ever_ai_native": float(co["ever_ai_native"]), "ever_product": float(co["ever_product"]),
        "services_current": float(co["services_current"]), "yoe": float(yoe),
        "yoe_ideal": 1.0 if 6 <= yoe <= 8 else 0.0, "yoe_band": 1.0 if 5 <= yoe <= 9 else 0.0,
        "musthave_hits": float(musthave), "in_domain_title": float(_has_domain(titles_text)),
        "depth": float(jd_spec.any_term(blob, SCRATCH)),
        "college_tier1": 1.0 if "tier_1" in tiers else 0.0,
        "college_tier2": 1.0 if "tier_2" in tiers else 0.0,
        "num_positions": float(len(hist)),
        "avg_tenure_mo": float(sum(durs) / len(durs)) if durs else 0.0,
        "ai_density": float(jd_spec.count_terms(blob, jd_spec.STUFFER_AI_KEYWORDS)),
        "nontech_title": 1.0 if any(t in cur_title for t in jd_spec.NONTECH_TITLES) else 0.0,
        "skill_count": float(len(skills)),
        "endorse_total": float(sum((s.get("endorsements") or 0) for s in skills)),
    }


def content_fit(c):
    reasons = {}
    if honeypot.flags(c):
        return 0.0, "honeypot", {"honeypot": honeypot.flags(c)}

    prof = c.get("profile", {}) or {}
    blob = jd_spec.text_blob(c)
    cur_title = (prof.get("current_title") or "").lower()
    yoe = prof.get("years_of_experience")

    # --- signals ---
    co = company_quality.company_signals(c)
    reasons["company"] = {"best": co["best_score"], "best_company": co["best_company"],
                          "ai_native": co["ever_ai_native"]}

    # must-have evidence (core 3 groups, python excluded as trivial)
    core_groups = ["embeddings_retrieval", "vector_db_hybrid_search", "ranking_eval"]
    musthave_hits = sum(1 for g in core_groups if jd_spec.any_term(blob, jd_spec.MUST_HAVES[g]))
    reasons["musthave_hits"] = musthave_hits

    # college (best tier across education)
    tiers = [(e.get("tier") or "") for e in (c.get("education", []) or [])]
    best_tier = "tier_1" if "tier_1" in tiers else "tier_2" if "tier_2" in tiers else None

    # in-domain title (current or any past role)
    titles_text = " ".join([cur_title] + [(j.get("title") or "").lower()
                                          for j in (c.get("career_history", []) or [])])
    in_domain = _has_domain(titles_text)

    # depth flavor
    depth = jd_spec.any_term(blob, SCRATCH)

    # trajectory / title-chaser
    hist = c.get("career_history", []) or []
    durs = [j.get("duration_months") or 0 for j in hist]
    avg_tenure = (sum(durs) / len(durs)) if durs else 999
    title_chaser = len(hist) >= 3 and avg_tenure < 18

    # --- bounded ± nudges ---
    mult = W_COMPANY_BASE + W_COMPANY_SLOPE * co["best_score"]
    if co["ever_ai_native"]:
        mult *= AI_NATIVE_BOOST
    # yoe-band fit (5-9 soft, 6-8 ideal)
    if isinstance(yoe, (int, float)):
        if 6 <= yoe <= 8:
            yf = 1.0
        elif 5 <= yoe < 10:
            yf = 0.98
        elif 3 <= yoe < 5:
            yf = 0.92
        elif 10 <= yoe <= 12:
            yf = 0.94
        elif yoe < 3:
            yf = 0.82
        else:  # >12
            yf = 0.82
        mult *= yf
        reasons["yoe_factor"] = round(yf, 3)
    mult *= (1.0 + MUSTHAVE_PER_GROUP * musthave_hits)
    if best_tier == "tier_1":
        mult *= COLLEGE_T1
    elif best_tier == "tier_2":
        mult *= COLLEGE_T2
    if depth:
        mult *= DEPTH_BOOST
    if in_domain:
        mult *= INDOMAIN_TITLE_BOOST
    if title_chaser:
        mult *= TITLECHASER_PENALTY
    mult = _clamp(mult, *NUDGE_CLAMP)

    # --- hard-cap ceilings (insurance) ---
    archetype = None
    ai_density = jd_spec.count_terms(blob, jd_spec.STUFFER_AI_KEYWORDS)
    is_nontech_title = any(t in cur_title for t in jd_spec.NONTECH_TITLES)
    if is_nontech_title and ai_density >= 4 and not in_domain:
        mult = min(mult, CAP_STUFFER)
        archetype = "wannabe"
    wd = jd_spec.DO_NOT_WANT["wrong_domain_primary"]
    if jd_spec.count_terms(blob, wd["anti_terms"]) >= 3 and not jd_spec.any_term(blob, wd["redeem_terms"]):
        mult = min(mult, CAP_WRONGDOMAIN)
        archetype = archetype or "wrong_domain"
    if co["services_current"] and not co["ever_product"]:
        mult = min(mult, CAP_SERVICES_ONLY)
        archetype = archetype or "services_only"

    # --- rescue floor: genuine product/AI fit lacking jargon ---
    rescued = False
    if archetype is None and (co["ever_product"] or co["ever_ai_native"] or in_domain or musthave_hits >= 2):
        if mult < RESCUE_FLOOR:
            mult = max(mult, RESCUE_FLOOR)
            rescued = True

    mult = _clamp(mult, *FINAL_CLAMP)

    # --- archetype label (content only; availability decided at combine time) ---
    if archetype is None:
        strong_company = co["best_score"] >= 0.8 or co["ever_ai_native"]
        if isinstance(yoe, (int, float)) and yoe > 12:
            archetype = "overqualified"
        elif isinstance(yoe, (int, float)) and yoe < 3:
            archetype = "junior"
        elif musthave_hits >= 2 and in_domain and strong_company:
            archetype = "standout" if mult >= 1.15 else "strong"
        elif in_domain and (musthave_hits >= 1 or co["ever_product"]):
            archetype = "solid"
        else:
            archetype = "adjacent"
    reasons.update({"in_domain": in_domain, "depth": depth, "ai_density": ai_density,
                    "avg_tenure_mo": round(avg_tenure, 1), "rescued": rescued,
                    "college": best_tier})
    return round(mult, 4), archetype, reasons


def hard_caps(c):
    """Rank-time deterministic vetoes (insurance, separate from the reranker which already
    weights these features). Returns (cap_factor in {1.0,0.6,0.4,0.2}, archetype_or_None)."""
    prof = c.get("profile", {}) or {}
    blob = jd_spec.text_blob(c)
    cur_title = (prof.get("current_title") or "").lower()
    co = company_quality.company_signals(c)
    titles_text = " ".join([cur_title] + [(j.get("title") or "").lower()
                                          for j in (c.get("career_history", []) or [])])
    in_domain = _has_domain(titles_text)
    cap, arch = 1.0, None
    if (any(t in cur_title for t in jd_spec.NONTECH_TITLES)
            and jd_spec.count_terms(blob, jd_spec.STUFFER_AI_KEYWORDS) >= 4 and not in_domain):
        cap, arch = CAP_STUFFER, "wannabe"
    wd = jd_spec.DO_NOT_WANT["wrong_domain_primary"]
    if jd_spec.count_terms(blob, wd["anti_terms"]) >= 3 and not jd_spec.any_term(blob, wd["redeem_terms"]):
        cap = min(cap, CAP_WRONGDOMAIN); arch = arch or "wrong_domain"
    if co["services_current"] and not co["ever_product"]:
        cap = min(cap, CAP_SERVICES_ONLY); arch = arch or "services_only"
    return cap, arch


if __name__ == "__main__":
    import json
    from collections import Counter
    ART = os.path.join(HERE, "artifacts")
    CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
            r"\India_runs_data_and_ai_challenge\candidates.jsonl")
    pool = json.load(open(os.path.join(ART, "retrieval_topk.json"), encoding="utf-8"))[:200]
    want = {p["id"] for p in pool}
    recs = {}
    with open(CAND, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in want:
                recs[c["candidate_id"]] = c
                if len(recs) == len(want):
                    break
    arch = Counter()
    mults = []
    for p in pool:
        c = recs.get(p["id"])
        if not c:
            continue
        m, a, r = content_fit(c)
        arch[a] += 1
        mults.append(m)
    mults.sort()
    n = len(mults)
    print("archetype histogram (pool top-200):", dict(arch.most_common()))
    print(f"content_mult: min={mults[0]:.3f} p25={mults[n//4]:.3f} median={mults[n//2]:.3f} "
          f"p75={mults[3*n//4]:.3f} max={mults[-1]:.3f}  (honeypots=0)")
