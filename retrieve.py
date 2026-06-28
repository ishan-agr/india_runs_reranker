"""
Phase 1->2 bridge: embed the JD as a query, retrieve top-K candidates, and VALIDATE what
raw E5 surfaces (real fits vs honeypots vs stuffers). Also saves the contender pool for the
Phase-2 teacher. HyRe deferred -> query is the JD requirements prose.
"""
import sys
import os
import json
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import honeypot
import company_quality

ART = os.path.join(HERE, "artifacts")
CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
        r"\India_runs_data_and_ai_challenge\candidates.jsonl")
POOL_K = 1500

JD_QUERY = ("query: Senior AI Engineer, founding team at a product startup, owns ranking, "
            "retrieval and matching systems. Production experience with embeddings-based "
            "retrieval and semantic search (sentence-transformers, BGE, E5, OpenAI embeddings), "
            "vector databases and hybrid search (FAISS, Pinecone, Weaviate, Qdrant, Milvus, "
            "Elasticsearch, OpenSearch). Builds and ships recommendation, search and "
            "learning-to-rank systems to real users at scale. Rigorous ranking evaluation: "
            "NDCG, MRR, MAP, A/B testing, offline-to-online correlation. LLM fine-tuning and "
            "re-ranking (LoRA, QLoRA, PEFT), RAG. Strong Python engineer, 5-9 years applied ML "
            "at product companies, not pure IT services or consulting.")

DOMAIN_TITLE = ["ml engineer", "machine learning", "ai engineer", "data scientist",
                "applied scien", "research engineer", "nlp", "search engineer", "ranking"]


def is_domain_title(t):
    t = (t or "").lower()
    return any(k in t for k in DOMAIN_TITLE)


emb = np.load(os.path.join(ART, "cand_emb.npy"))
ids = json.load(open(os.path.join(ART, "cand_ids.json"), encoding="utf-8"))
assert emb.shape[0] == len(ids) == 100000, (emb.shape, len(ids))
print("loaded emb", emb.shape)

model = SentenceTransformer("intfloat/e5-base-v2")
q = model.encode([JD_QUERY], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
scores = emb @ q

order = np.argsort(-scores)[:POOL_K]
pool = [{"id": ids[i], "score": float(scores[i])} for i in order]
json.dump(pool, open(os.path.join(ART, "retrieval_topk.json"), "w", encoding="utf-8"))

top_ids = {ids[i]: int(rank) for rank, i in enumerate(order[:200])}
recs = {}
with open(CAND, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        if c["candidate_id"] in top_ids:
            recs[c["candidate_id"]] = c
        if len(recs) == len(top_ids):
            break

# top-100 stats
top100 = [ids[i] for i in order[:100]]
hp = sv = prod = ai = dom = 0
titles = Counter()
for cid in top100:
    c = recs[cid]
    prof = c.get("profile", {}) or {}
    titles[(prof.get("current_title") or "?").lower()] += 1
    if honeypot.flags(c):
        hp += 1
    sig = company_quality.company_signals(c)
    sv += sig["services_current"]
    prod += sig["ever_product"]
    ai += sig["ever_ai_native"]
    dom += is_domain_title(prof.get("current_title"))

print("\n== TOP 30 (raw E5 retrieval) ==")
for rank, i in enumerate(order[:30], 1):
    c = recs[ids[i]]
    prof = c.get("profile", {}) or {}
    sig = company_quality.company_signals(c)
    flag = "HONEYPOT" if honeypot.flags(c) else ""
    print(f"{rank:3d} {scores[i]:.3f} {ids[i]} | {str(prof.get('current_title'))[:22]:22s} "
          f"@ {str(prof.get('current_company'))[:16]:16s} yoe={prof.get('years_of_experience')} "
          f"co={sig['best_score']} ai={int(sig['ever_ai_native'])} {flag}")

print("\n== TOP-100 aggregate ==")
print(f"honeypots: {hp}   (DQ if >10)")
print(f"services_current: {sv} | ever_product: {prod} | ever_ai_native: {ai} | domain_title: {dom}")
print("top titles:", titles.most_common(12))
print("\nsaved pool:", os.path.join(ART, "retrieval_topk.json"), f"({POOL_K} ids)")
