"""
Company-quality signal (Recruiter-Intuition #1).

The dataset has only 63 companies, so the prestige table is hand-curated and COMPLETE
(no coverage gap). Real recognizable names get an explicit tier; fictional placeholders
(Pied Piper, Hooli, Acme, Stark, Globex, Dunder Mifflin, Wayne) have no real prestige and
resolve through their `industry` instead. AI-native companies score highest for THIS role
(domain + demanding stack), matching the JD's 'good VC-backed startup = scale + fast stack'.

Used by the archetype/content-fit layer. We expose raw scores in [0,1] + flags; the
archetype layer chooses bounded bonus/penalty magnitudes (calibrated vs teacher labels).
`best_across_career` matters: prior product/AI experience REDEEMS a current services job
(JD: 'currently at a services firm + prior product company = OK').
"""

AI_NATIVE = {  # domain-relevant AI companies -> best signal for a Senior AI Engineer role
    "sarvam ai", "krutrim", "observe.ai", "yellow.ai", "haptik", "verloop.io",
    "mad street den", "glance", "rephrase.ai", "aganitha", "niramai", "saarthi.ai",
    "wysa", "locobuzz", "genpact ai",
}
TOP = {  # big-tech / top internet (general prestige)
    "google", "meta", "amazon", "microsoft", "apple", "netflix", "adobe",
    "salesforce", "uber", "linkedin",
}
HIGH = {  # top product unicorns
    "swiggy", "zomato", "flipkart", "razorpay", "cred", "paytm", "phonepe",
    "zoho", "freshworks", "ola", "meesho", "nykaa", "inmobi", "dream11", "policybazaar",
}
MID = {"byju", "byju's", "vedantu", "unacademy", "upgrad", "pharmeasy"}
SERVICES = {  # real IT-services / consulting -> low fit for this role
    "infosys", "wipro", "tcs", "capgemini", "hcl", "mindtree", "accenture",
    "cognizant", "tech mahindra", "mphasis",
}

INDUSTRY_TIER = {
    # AI-native industries -> top + domain flag
    "ai/ml": (1.0, True), "conversational ai": (1.0, True), "voice ai": (1.0, True),
    "ai services": (0.9, True), "healthtech ai": (0.95, True),
    # product / tech
    "internet": (0.95, False), "fintech": (0.8, False), "saas": (0.8, False),
    "food delivery": (0.78, False), "e-commerce": (0.78, False), "gaming": (0.75, False),
    "adtech": (0.75, False), "insurance tech": (0.72, False), "media": (0.7, False),
    "consumer electronics": (0.78, False), "transportation": (0.72, False),
    "software": (0.55, False), "edtech": (0.55, False), "healthtech": (0.55, False),
    # low / irrelevant
    "it services": (0.25, False), "consulting": (0.25, False),
    "manufacturing": (0.2, False), "paper products": (0.18, False),
    "conglomerate": (0.3, False),
}

NAME_SCORE = {}
for n in AI_NATIVE:
    NAME_SCORE[n] = (1.0, True)
for n in TOP:
    NAME_SCORE[n] = (0.95, False)
for n in HIGH:
    NAME_SCORE[n] = (0.8, False)
for n in MID:
    NAME_SCORE[n] = (0.55, False)
for n in SERVICES:
    NAME_SCORE[n] = (0.25, False)


def company_quality(name, industry):
    """Return (score in [0,1], is_ai_native). Name lookup first, else industry fallback."""
    nm = (name or "").strip().lower()
    if nm in NAME_SCORE:
        return NAME_SCORE[nm]
    # substring match for safety (e.g., "Google India")
    for k, v in NAME_SCORE.items():
        if k in nm:
            return v
    ind = (industry or "").strip().lower()
    if ind in INDUSTRY_TIER:
        return INDUSTRY_TIER[ind]
    return (0.4, False)  # unknown -> neutral-low


def company_signals(c):
    prof = c.get("profile", {}) or {}
    cur_score, cur_ai = company_quality(prof.get("current_company"), prof.get("current_industry"))
    best_score, best_ai, best_name = cur_score, cur_ai, prof.get("current_company")
    ever_ai = cur_ai
    for j in c.get("career_history", []) or []:
        s, ai = company_quality(j.get("company"), j.get("industry"))
        ever_ai = ever_ai or ai
        if s > best_score:
            best_score, best_ai, best_name = s, ai, j.get("company")
    return {
        "current_score": round(cur_score, 3),
        "current_is_ai_native": cur_ai,
        "best_score": round(best_score, 3),
        "best_company": best_name,
        "ever_ai_native": ever_ai,
        "ever_product": best_score >= 0.7,   # redeems services-only
        "services_current": cur_score <= 0.3,
    }


if __name__ == "__main__":
    import json
    from collections import Counter
    CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
            r"\India_runs_data_and_ai_challenge\candidates.jsonl")
    cur_buck = Counter()
    ever_ai = ever_prod = 0
    n = 0
    examples = []
    with open(CAND, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            sig = company_signals(c)
            n += 1
            cur_buck[round(sig["current_score"], 2)] += 1
            ever_ai += sig["ever_ai_native"]
            ever_prod += sig["ever_product"]
            if n <= 6:
                examples.append((c["candidate_id"], c["profile"].get("current_company"), sig))
    print("n", n)
    print("current_score buckets:", dict(sorted(cur_buck.items())))
    print("ever_ai_native:", ever_ai, "| ever_product (best>=0.7):", ever_prod)
    print("examples:")
    for e in examples:
        print(" ", e[0], e[1], "->", e[2])
