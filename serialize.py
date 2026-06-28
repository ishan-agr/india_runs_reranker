"""
Candidate -> E5 passage text (Phase 1).

E5 models expect "query: " / "passage: " prefixes. Candidates are passages; the JD
is the query (handled at retrieval time). E5-base has a ~512-token window, so we put
the most fit-discriminative fields FIRST (headline, current role, summary, skills,
recent roles) and let the tail truncate. We deliberately surface title + career
descriptions, because the JD says trajectory beats a keyword skills list.

Pure stdlib so it can be reused by the embedder, the teacher prompt, and analysis.
"""

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
MAX_SUMMARY = 500
MAX_DESC = 280
MAX_ROLES_WITH_DESC = 3   # most-recent roles that keep their description
MAX_SKILLS = 30


def _clean(s):
    return " ".join((s or "").split())


def to_passage_text(c):
    """Serialize a candidate record into an E5 passage string."""
    p = c.get("profile", {}) or {}
    bits = []

    headline = _clean(p.get("headline"))
    yoe = p.get("years_of_experience")
    cur_t = _clean(p.get("current_title"))
    cur_c = _clean(p.get("current_company"))
    cur_i = _clean(p.get("current_industry"))

    if headline:
        bits.append(headline + ".")
    lead = []
    if isinstance(yoe, (int, float)):
        lead.append(f"{yoe:.1f} years experience")
    if cur_t:
        role = f"currently {cur_t}"
        if cur_c:
            role += f" at {cur_c}"
        if cur_i:
            role += f" ({cur_i})"
        lead.append(role)
    if lead:
        bits.append("; ".join(lead) + ".")

    summ = _clean(p.get("summary"))
    if summ:
        bits.append("Summary: " + summ[:MAX_SUMMARY] + ".")

    # Skills with proficiency — names are key retrieval signal.
    skills = c.get("skills", []) or []
    if skills:
        names = []
        for s in skills[:MAX_SKILLS]:
            nm = _clean(s.get("name"))
            pr = _clean(s.get("proficiency"))
            names.append(f"{nm} ({pr})" if pr else nm)
        bits.append("Skills: " + ", ".join(n for n in names if n) + ".")

    # Career history — title/company/industry for all roles; descriptions for recent few.
    hist = c.get("career_history", []) or []
    if hist:
        lines = []
        for i, job in enumerate(hist):
            t = _clean(job.get("title"))
            comp = _clean(job.get("company"))
            ind = _clean(job.get("industry"))
            dur = job.get("duration_months")
            head = t or "role"
            if comp:
                head += f" at {comp}"
            meta = []
            if ind:
                meta.append(ind)
            if isinstance(dur, (int, float)):
                meta.append(f"{int(dur)}mo")
            if meta:
                head += " (" + ", ".join(meta) + ")"
            if i < MAX_ROLES_WITH_DESC:
                desc = _clean(job.get("description"))
                if desc:
                    head += ": " + desc[:MAX_DESC]
            lines.append(head)
        bits.append("Experience: " + " | ".join(lines))

    # Education (degree/field/tier) — short.
    edu = c.get("education", []) or []
    if edu:
        e = edu[0]
        deg = _clean(e.get("degree"))
        fld = _clean(e.get("field_of_study"))
        tier = _clean(e.get("tier"))
        eline = " ".join(x for x in [deg, fld] if x)
        if tier:
            eline += f" [{tier}]"
        if eline.strip():
            bits.append("Education: " + eline + ".")

    return PASSAGE_PREFIX + _clean(" ".join(bits))


if __name__ == "__main__":
    # Smoke test on the first record of candidates.jsonl.
    import json
    path = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
            r"\India_runs_data_and_ai_challenge\candidates.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())
    txt = to_passage_text(rec)
    print("LEN(chars):", len(txt))
    print(txt)
