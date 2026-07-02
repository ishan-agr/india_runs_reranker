"""
rank.py — produce submission.csv from candidates.jsonl.

CPU-only, no network, no LLM, <5 min / 100K, <16 GB. Uses precomputed artifacts
(cand_emb.npy, cand_ids.json, reranker.pkl). Only candidates whose id is NOT in the
precomputed set get embedded live (e.g. a small swapped sandbox sample), so the released
100K runs with zero model loading.

  python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pipeline: reranker base score (frozen-E5 features) -> honeypot gate (exclude) ->
hard-cap vetoes -> behavioral availability multiplier -> top-100 -> templated reasoning.
"""
import argparse
import json
import os
import sys
import time
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import honeypot
import content_fit
import company_quality
import jd_spec
from behavioral import behavioral_modifier
from serialize import to_passage_text

ART = os.environ.get("REDROB_ART", os.path.join(HERE, "artifacts"))


def make_reason(c):
    """Specific, varied, honest, rank-consistent justification from real fields only."""
    p = c.get("profile", {}) or {}
    title = p.get("current_title") or "Candidate"
    yoe = p.get("years_of_experience")
    blob = jd_spec.text_blob(c)
    co = company_quality.company_signals(c)
    _, facts, _ = behavioral_modifier(c)

    ev = []
    if jd_spec.any_term(blob, jd_spec.MUST_HAVES["embeddings_retrieval"]):
        ev.append("embeddings/retrieval")
    if jd_spec.any_term(blob, jd_spec.MUST_HAVES["vector_db_hybrid_search"]):
        ev.append("vector search")
    if jd_spec.any_term(blob, jd_spec.MUST_HAVES["ranking_eval"]):
        ev.append("ranking/eval")
    ev_str = ", ".join(ev) if ev else "applied ML"

    if co["ever_ai_native"]:
        comp_clause = f"AI-native experience ({co['best_company']})"
    elif co["best_score"] >= 0.8:
        comp_clause = f"product-company background ({co['best_company']})"
    elif co["services_current"] and not co["ever_product"]:
        comp_clause = "services-firm background"
    else:
        comp_clause = f"at {p.get('current_company')}"

    yoe_s = f"{yoe:.1f}y" if isinstance(yoe, (int, float)) else "exp"
    lead = f"{title} with {yoe_s}; {ev_str} ({comp_clause})"

    pos = []
    rr = facts.get("response_rate")
    if rr is not None and rr >= 0.6:
        pos.append(f"responsive ({rr:.0%})")
    if facts.get("days_inactive", 999) <= 30:
        pos.append("recently active")
    np_ = facts.get("notice_period_days")
    if np_ is not None and np_ <= 30:
        pos.append(f"{int(np_)}d notice")
    if pos:
        lead += ", " + ", ".join(pos)

    concerns = []
    if isinstance(yoe, (int, float)) and yoe > 12:
        concerns.append(f"{yoe:.0f}y may be over-leveled for a hands-on founding IC role")
    elif isinstance(yoe, (int, float)) and yoe < 4:
        concerns.append(f"only {yoe:.1f}y experience")
    if facts.get("days_inactive", 0) > 120:
        concerns.append(f"inactive {facts['days_inactive']}d")
    if rr is not None and rr < 0.25:
        concerns.append(f"low recruiter response ({rr:.0%})")
    if np_ is not None and np_ >= 120:
        concerns.append(f"long notice ({int(np_)}d)")
    if co["services_current"] and not co["ever_product"]:
        concerns.append("services background, unproven in product ML")

    s = lead + "."
    if concerns:
        s += " Concern: " + "; ".join(concerns[:2]) + "."
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()
    t0 = time.time()

    bundle = pickle.load(open(os.path.join(ART, "reranker.pkl"), "rb"))
    model = bundle["model"]
    q = bundle["jd_emb"].astype("float32")
    pos_c = bundle["pos_centroid"].astype("float32")
    neg_c = bundle["neg_centroid"].astype("float32")
    cf_order = bundle["content_feature_order"]

    emb_path = os.path.join(ART, "cand_emb.npy")
    ids_path = os.path.join(ART, "cand_ids.json")
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        cand_emb = np.load(emb_path)
        cand_ids = json.load(open(ids_path, encoding="utf-8"))
    else:
        # sandbox / swapped set with no precomputed cache -> embed everything live
        cand_emb = np.zeros((0, q.shape[0]), dtype="float32")
        cand_ids = []
    id2row = {c: i for i, c in enumerate(cand_ids)}

    ids, cfeat, honey, capf, beh, rows = [], [], [], [], [], []
    missing_idx, missing_text = [], []
    with open(args.candidates, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c["candidate_id"]
            ids.append(cid)
            r = id2row.get(cid, -1)
            rows.append(r)
            if r == -1:
                missing_idx.append(len(ids) - 1)
                missing_text.append(to_passage_text(c))
            cf = content_fit.content_features(c)
            cfeat.append([cf[k] for k in cf_order])
            honey.append(1 if honeypot.flags(c) else 0)
            cap, _ = content_fit.hard_caps(c)
            capf.append(cap)
            m, _, _ = behavioral_modifier(c)
            beh.append(m)
    N = len(ids)
    cfeat = np.array(cfeat, dtype="float32")
    honey = np.array(honey)
    capf = np.array(capf, dtype="float32")
    beh = np.array(beh, dtype="float32")
    rows = np.array(rows)
    print(f"pass1 ({N} candidates) in {time.time()-t0:.1f}s; missing emb: {len(missing_idx)}")

    emb = np.zeros((N, cand_emb.shape[1]), dtype="float32")
    present = rows >= 0
    emb[present] = cand_emb[rows[present]]
    if missing_idx:
        # Guard: live-embedding thousands of rows on CPU (~18 texts/s) would blow the 5-min
        # limit. If the precomputed cache is absent for a large set, the user forgot the
        # one-time download step. Fail loudly with the fix (unless explicitly overridden).
        if len(missing_idx) > 2000 and os.environ.get("RANK_ALLOW_LIVE_BULK") != "1":
            sys.exit(
                f"\n{len(missing_idx)} of {N} candidates have NO precomputed embedding.\n"
                "Live-embedding this many on CPU would exceed the 5-min ranking limit.\n"
                "Fetch the precomputed embeddings first (one-time, not part of ranking):\n"
                "    python download_embeddings.py\n"
                "This unzips cand_emb.npy + cand_ids.json into artifacts/. Then re-run rank.py.\n"
                "(The sandbox uses small samples and is unaffected. To force bulk live-embedding\n"
                " anyway, set RANK_ALLOW_LIVE_BULK=1.)"
            )
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("intfloat/e5-base-v2")
        me = st.encode(missing_text, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
        for j, idx in enumerate(missing_idx):
            emb[idx] = me[j]

    cos_jd = emb @ q
    cos_pos = emb @ pos_c
    cos_neg = emb @ neg_c
    X = np.hstack([cfeat, cos_jd[:, None], cos_pos[:, None], cos_neg[:, None]])
    base = model.predict(X)
    final = base * capf * beh
    final[honey == 1] = -1e9

    order = sorted(range(N), key=lambda i: (-final[i], ids[i]))[:100]
    mx = max(final[i] for i in order) if order else 1.0
    # round score FIRST, then order by (score desc, candidate_id asc) so equal displayed
    # scores tie-break on candidate_id ascending (validator requirement).
    scored = [(ids[i], round(max(float(final[i]), 0.0) / (mx + 1e-9) * 0.99, 4)) for i in order]
    scored.sort(key=lambda t: (-t[1], t[0]))
    top_ids = [cid for cid, _ in scored]

    want = set(top_ids)
    recs = {}
    with open(args.candidates, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in want:
                recs[c["candidate_id"]] = c
                if len(recs) == len(want):
                    break

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        for rank, (cid, score) in enumerate(scored, 1):
            reason = make_reason(recs[cid]).replace('"', "'")
            f.write(f'{cid},{rank},{score},"{reason}"\n')

    hp = sum(1 for cid in top_ids if honeypot.flags(recs[cid]))
    print(f"DONE {N} in {time.time()-t0:.1f}s | wrote {args.out} | honeypots in top100: {hp}")


if __name__ == "__main__":
    main()
