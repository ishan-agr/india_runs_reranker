"""
phase2_fallback.py — make the LLM teacher step OPTIONAL.

The Opus teacher (phase2_teacher.py) produces artifacts/teacher_labels.jsonl and needs an
Anthropic API key. This script produces the SAME-schema file from a deterministic heuristic
(the content_fit recruiter engine + honeypot gate + retrieval cosine) so the whole training
pipeline (phase3_rum -> phase4_train_reranker) can be reproduced with NO API key and NO network.

The shipped reranker.pkl was trained on the real Opus labels (higher quality). This fallback is
for anyone reproducing without a key; it yields a functional, if slightly weaker, reranker.

    python phase2_fallback.py                 # labels the top-400 retrieval pool -> teacher_labels.jsonl
    python phase2_fallback.py --n 800         # label more of the pool
    python phase2_fallback.py --out heuristic_labels.jsonl   # write elsewhere

Refuses to overwrite an existing teacher_labels.jsonl unless --force (protects real labels).
"""
import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import content_fit
import company_quality
import jd_spec
import honeypot

ART = os.path.join(HERE, "artifacts")
CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
        r"\India_runs_data_and_ai_challenge\candidates.jsonl")


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def fit_and_tier(c, cnorm):
    """Map the deterministic content engine -> (fit 0-100, tier 0-5, archetype).

    fit is driven mainly by the content multiplier (cosine is near-useless at the top of the
    pool, per Phase-2), with a small cosine-rank nudge. Honeypots -> 0 / tier 0.
    """
    mult, archetype, _r = content_fit.content_fit(c)
    if honeypot.flags(c) or archetype == "honeypot":
        return 0, 0, "honeypot"
    base = _clamp((mult - 0.10) / 1.30, 0.0, 1.0)          # content mult in [0.10,1.40] -> [0,1]
    fit = int(round(_clamp(base * 85 + cnorm * 8, 0, 98)))  # small cosine nudge
    if fit >= 80:
        tier = 5
    elif fit >= 68:
        tier = 4
    elif fit >= 55:
        tier = 3
    elif fit >= 40:
        tier = 2
    else:
        tier = 1
    return fit, tier, archetype


def brief_rationale(c, archetype):
    co = company_quality.company_signals(c)
    blob = jd_spec.text_blob(c)
    core = ["embeddings_retrieval", "vector_db_hybrid_search", "ranking_eval"]
    mh = sum(1 for g in core if jd_spec.any_term(blob, jd_spec.MUST_HAVES[g]))
    return (f"heuristic label ({archetype}): best_company={co['best_company']} "
            f"(q={co['best_score']:.2f}, ai_native={co['ever_ai_native']}), must-have groups={mh}/3.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="how many of the retrieval pool to label")
    ap.add_argument("--out", default=os.path.join(ART, "teacher_labels.jsonl"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        sys.exit(
            f"{args.out} already exists. It may be the real Opus teacher labels.\n"
            "Refusing to overwrite. Use --force, or --out <other.jsonl> to write elsewhere."
        )

    pool = json.load(open(os.path.join(ART, "retrieval_topk.json"), encoding="utf-8"))[: args.n]
    scores = [p["score"] for p in pool]
    smin, smax = min(scores), max(scores)
    span = (smax - smin) or 1.0
    want = {p["id"]: p["score"] for p in pool}

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

    out = []
    for cid, score in want.items():
        c = recs.get(cid)
        if c is None:
            continue
        cnorm = (score - smin) / span
        fit, tier, archetype = fit_and_tier(c, cnorm)
        out.append({
            "candidate_id": cid, "tier": tier, "fit": fit, "archetype": archetype,
            "rationale": brief_rationale(c, archetype),
            "concerns": "deterministic heuristic; no LLM judgment.",
            "retrieval_score": score,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(out)} heuristic labels -> {args.out}")
    print("tier hist:", dict(sorted(Counter(r["tier"] for r in out).items())))
    print("archetype hist:", dict(Counter(r["archetype"] for r in out).most_common()))
    print("NOTE: this makes the LLM teacher OPTIONAL. Now run phase3_rum.py -> phase4_train_reranker.py.")


if __name__ == "__main__":
    main()
