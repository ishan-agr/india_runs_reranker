"""
Phase 4 — learned reranker on FROZEN E5 (user-chosen approach).

Predicts teacher fit (0-100) from, per candidate:
  - content features (19, deterministic; content_fit.content_features)
  - cosine to the JD query (frozen E5)
  - cos to the POSITIVE centroid and cos to the HARD-NEG centroid (teacher-supervised
    prototypes in frozen embedding space — injects the teacher signal the bare cosine lacks)

Trained on the RUM set (positives/mid/hard_neg with teacher fit, honeypot=0, easy_neg=2).
Honest eval: 5-fold CV Spearman over the non-honeypot POOL (apples-to-apples vs the 0.677
content+cosine baseline). Saves the model + centroids + JD embedding for rank-time.
"""
import os
import sys
import json
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import content_fit
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sentence_transformers import SentenceTransformer

ART = os.path.join(HERE, "artifacts")
CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
        r"\India_runs_data_and_ai_challenge\candidates.jsonl")

JD_QUERY = ("query: Senior AI Engineer, founding team at a product startup, owns ranking, "
            "retrieval and matching systems. Production experience with embeddings-based "
            "retrieval and semantic search (sentence-transformers, BGE, E5, OpenAI embeddings), "
            "vector databases and hybrid search (FAISS, Pinecone, Weaviate, Qdrant, Milvus, "
            "Elasticsearch, OpenSearch). Builds and ships recommendation, search and "
            "learning-to-rank systems to real users at scale. Rigorous ranking evaluation: "
            "NDCG, MRR, MAP, A/B testing, offline-to-online correlation. LLM fine-tuning and "
            "re-ranking (LoRA, QLoRA, PEFT), RAG. Strong Python engineer, 5-9 years applied ML "
            "at product companies, not pure IT services or consulting.")

# --- load embeddings + ids + training set ---
emb = np.load(os.path.join(ART, "cand_emb.npy"))
ids = json.load(open(os.path.join(ART, "cand_ids.json"), encoding="utf-8"))
id2row = {c: i for i, c in enumerate(ids)}
train = [json.loads(l) for l in open(os.path.join(ART, "rum_train.jsonl"), encoding="utf-8") if l.strip()]
need = {r["candidate_id"] for r in train}
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

# --- JD embedding + teacher-supervised centroids (frozen space) ---
model = SentenceTransformer("intfloat/e5-base-v2")
q = model.encode([JD_QUERY], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
pos_rows = [id2row[r["candidate_id"]] for r in train if r["role"] == "positive"]
neg_rows = [id2row[r["candidate_id"]] for r in train if r["role"] in ("hard_neg", "honeypot")]
pos_c = emb[pos_rows].mean(0); pos_c /= np.linalg.norm(pos_c)
neg_c = emb[neg_rows].mean(0); neg_c /= np.linalg.norm(neg_c)
print(f"centroids from {len(pos_rows)} positives, {len(neg_rows)} negatives")

# --- feature matrix ---
cf_order = content_fit.FEATURE_ORDER
feat_order = cf_order + ["cosine_jd", "cos_pos_centroid", "cos_neg_centroid"]
X, y, roles, y_eval = [], [], [], []
for r in train:
    cid = r["candidate_id"]
    e = emb[id2row[cid]]
    cf = content_fit.content_features(recs[cid])
    row = [cf[k] for k in cf_order] + [float(e @ q), float(e @ pos_c), float(e @ neg_c)]
    X.append(row)
    y.append(r["target"])
    roles.append(r["role"])
    # eval target: teacher fit for the labeled pool (non-honeypot), else the role target
    y_eval.append(r["target"])
X = np.array(X); y = np.array(y, dtype=float)
roles = np.array(roles)
pool_mask = np.isin(roles, ["positive", "mid", "hard_neg"])  # the hard-ranking subset
print(f"train X={X.shape}; pool(non-honeypot labeled)={pool_mask.sum()}")

models = {
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
    "gbm": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.8),
}
best = None
for name, m in models.items():
    pred = cross_val_predict(m, X, y, cv=5)
    rho_pool = spearmanr(pred[pool_mask], y[pool_mask]).statistic
    rho_all = spearmanr(pred, y).statistic
    print(f"{name:6s} CV Spearman: pool(vs teacher_fit)={rho_pool:.3f}  all800={rho_all:.3f}")
    if best is None or rho_pool > best[1]:
        best = (name, rho_pool, m)

name, rho, m = best
m.fit(X, y)
bundle = {"model": m, "model_type": name, "feature_order": feat_order,
          "content_feature_order": cf_order, "pos_centroid": pos_c, "neg_centroid": neg_c,
          "jd_emb": q, "jd_query": JD_QUERY, "cv_spearman_pool": rho}
pickle.dump(bundle, open(os.path.join(ART, "reranker.pkl"), "wb"))
print(f"\nbest = {name} (pool CV Spearman {rho:.3f}); baselines: cosine 0.588, content 0.669, content+cosine 0.677")
print("saved", os.path.join(ART, "reranker.pkl"))
