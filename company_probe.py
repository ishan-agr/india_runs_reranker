"""
Feasibility probe for the Recruiter-Intuition layer (company quality + project depth).

Answers, from the actual data:
  - How many DISTINCT companies? What % of candidates are at a RECOGNIZABLE real company
    (so a prestige lookup is usable) vs fictional/no-name (so we must proxy by size/industry)?
  - Industry x size cross — can we proxy "good VC-backed startup" deterministically?
  - Do career descriptions carry a from-scratch-vs-library-heavy signal at all?
  - Are tier_1 education institutions recognizable real names?
Pure stdlib; one streaming pass.
"""
import json
import os
import re
from collections import Counter, defaultdict

CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
        r"\India_runs_data_and_ai_challenge\candidates.jsonl")
OUTDIR = r"E:\[PUB] India_runs_data_and_ai_challenge\redrob_ranker\artifacts"

KNOWN_REAL = [
    # big tech
    "google", "meta", "facebook", "amazon", "microsoft", "apple", "netflix", "nvidia",
    "adobe", "salesforce", "oracle", "ibm", "intel", "uber", "linkedin", "twitter",
    "airbnb", "samsung", "qualcomm", "cisco", "sap", "vmware", "dell", "paypal", "expedia",
    # ai labs / data
    "openai", "anthropic", "deepmind", "hugging face", "cohere", "databricks", "scale ai",
    "snowflake", "palantir", "stripe",
    # indian product / unicorns
    "flipkart", "swiggy", "zomato", "razorpay", "cred", "paytm", "phonepe", "ola",
    "byju", "freshworks", "zoho", "postman", "meesho", "dream11", "unacademy", "groww",
    "zerodha", "nykaa", "myntra", "dunzo", "urban company", "sharechat", "pharmeasy",
    "delhivery", "policybazaar", "browserstack", "chargebee", "druva", "innovaccer",
    # services (low prestige for THIS role)
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "ltimindtree", "hcl", "tech mahindra", "mphasis", "hexaware",
]
SERVICES = ["tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
            "capgemini", "mindtree", "ltimindtree", "hcl", "tech mahindra", "mphasis", "hexaware"]

SCRATCH = ["from scratch", "first principle", "implemented the", "derived ", "custom ",
           "built our own", "without relying", "designed the algorithm", "wrote our own",
           "ground up", "from the ground", "by hand", "hand-rolled", "reimplement",
           "own implementation", "developed a novel", "designed a custom"]
LIBHEAVY = ["langchain", "llamaindex", "used the", "leveraged", "integrated the",
            "via the api", "openai api", "hugging face", "off-the-shelf", "pre-built",
            "plug-and-play", "wrapper around", "using the library", "framework to"]


def has_any(text, terms):
    return any(t in text for t in terms)


comp_counter = Counter()
cur_comp_total = 0
cur_comp_known = 0
cur_comp_services = 0
ind_size = Counter()
industry_c = Counter()
scratch_n = lib_n = both_n = neither_n = desc_total = 0
scratch_examples = []
inst_by_tier = defaultdict(Counter)

with open(CAND, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        prof = c.get("profile", {}) or {}
        cur = (prof.get("current_company") or "").lower()
        if cur:
            cur_comp_total += 1
            if any(k in cur for k in KNOWN_REAL):
                cur_comp_known += 1
            if any(k in cur for k in SERVICES):
                cur_comp_services += 1
        ind_size[(prof.get("current_industry"), prof.get("current_company_size"))] += 1
        industry_c[prof.get("current_industry", "?")] += 1

        for job in c.get("career_history", []) or []:
            comp_counter[(job.get("company") or "").strip()] += 1
            d = (job.get("description") or "").lower()
            if not d:
                continue
            desc_total += 1
            s, l = has_any(d, SCRATCH), has_any(d, LIBHEAVY)
            if s and l:
                both_n += 1
            elif s:
                scratch_n += 1
            elif l:
                lib_n += 1
            else:
                neither_n += 1
            if s and len(scratch_examples) < 6:
                scratch_examples.append({"id": c["candidate_id"], "title": job.get("title"),
                                         "desc": (job.get("description") or "")[:240]})

        for e in c.get("education", []) or []:
            inst_by_tier[e.get("tier", "?")][(e.get("institution") or "").strip()] += 1

summary = {
    "distinct_companies": len(comp_counter),
    "top_companies": comp_counter.most_common(40),
    "current_company": {
        "total_with_company": cur_comp_total,
        "matched_known_real": cur_comp_known,
        "pct_known_real": round(cur_comp_known / max(cur_comp_total, 1), 4),
        "matched_services_firm": cur_comp_services,
        "pct_services": round(cur_comp_services / max(cur_comp_total, 1), 4),
    },
    "top_industries": industry_c.most_common(20),
    "industry_size_cross_top": Counter(ind_size).most_common(25),
    "descriptions": {
        "total": desc_total,
        "scratch_only": scratch_n, "library_only": lib_n, "both": both_n, "neither": neither_n,
        "pct_scratch": round((scratch_n + both_n) / max(desc_total, 1), 4),
        "pct_library": round((lib_n + both_n) / max(desc_total, 1), 4),
    },
    "scratch_examples": scratch_examples,
    "tier1_top_institutions": inst_by_tier.get("tier_1", Counter()).most_common(15),
    "tier4_top_institutions": inst_by_tier.get("tier_4", Counter()).most_common(10),
}
os.makedirs(OUTDIR, exist_ok=True)
with open(os.path.join(OUTDIR, "company_probe.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)

print("distinct_companies:", summary["distinct_companies"])
print("current_company:", json.dumps(summary["current_company"], indent=2))
print("\ntop_companies (top 25):", json.dumps(comp_counter.most_common(25), ensure_ascii=False, indent=1))
print("\ntop_industries:", json.dumps(industry_c.most_common(20), ensure_ascii=False))
print("\nindustry x size (top 20):", json.dumps(Counter(ind_size).most_common(20), ensure_ascii=False))
print("\ndescriptions depth signal:", json.dumps(summary["descriptions"], indent=2))
print("\nscratch examples:", json.dumps(scratch_examples, indent=2, ensure_ascii=False))
print("\ntier1 institutions:", json.dumps(summary["tier1_top_institutions"], ensure_ascii=False))
print("tier4 institutions:", json.dumps(summary["tier4_top_institutions"], ensure_ascii=False))
