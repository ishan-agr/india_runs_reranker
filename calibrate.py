"""
Phase 2 calibration — fit the content-fit weights to the Opus teacher labels.

Regresses interpretable content features onto the teacher's `fit` score (0-100). The
learned weights reproduce the teacher's spread (which raw cosine cannot: Spearman~0.08).
Honest evaluation via 5-fold cross-validated Spearman(pred, teacher_fit). Saves the
standardized linear model to artifacts/content_weights.json for rank-time use.
"""
import os
import sys
import json
import numpy as np
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import content_fit
import honeypot
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline

ART = os.path.join(HERE, "artifacts")
CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
        r"\India_runs_data_and_ai_challenge\candidates.jsonl")

LBL = os.path.join(ART, "teacher_labels.jsonl")
if not os.path.exists(LBL):
    sys.exit(
        "No teacher_labels.jsonl found. The teacher step is OPTIONAL — create the labels with EITHER:\n"
        "  python phase2_teacher.py --limit 400 --model claude-opus-4-8   (LLM teacher; needs ANTHROPIC key)\n"
        "  python phase2_fallback.py                                       (deterministic heuristic; no key)"
    )
labels = [json.loads(l) for l in open(LBL, encoding="utf-8") if l.strip()]
by_id = {x["candidate_id"]: x for x in labels}
print(f"labels: {len(labels)}")
print("tier hist:", dict(sorted(Counter(l.get('tier') for l in labels).items())))
print("archetype hist:", dict(Counter(l.get('archetype') for l in labels).most_common()))

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

# teacher vs cosine (baseline) over the full labeled set
cos = np.array([by_id[i].get("retrieval_score", 0) for i in by_id])
fit = np.array([by_id[i].get("fit", 0) for i in by_id], dtype=float)
print(f"\nSpearman(cosine, teacher_fit) over {len(fit)}: {spearmanr(cos, fit).statistic:.3f}  (the baseline to beat)")

# honeypot cross-check
det_hp = [i for i in by_id if honeypot.flags(recs.get(i, {}))]
teach_hp = [i for i in by_id if by_id[i].get("archetype") == "honeypot" or by_id[i].get("tier") == 0]
print(f"honeypots: deterministic-gate={len(det_hp)}, teacher-flagged={len(teach_hp)}, "
      f"union={len(set(det_hp) | set(teach_hp))}, teacher-only(missed by gate)={len(set(teach_hp)-set(det_hp))}")

# build feature matrix (exclude honeypots from weight-fitting; they're hard-gated separately)
feat_names = content_fit.FEATURE_ORDER
ids_fit = [i for i in by_id if i not in set(det_hp)]
X = np.array([[content_fit.content_features(recs[i])[f] for f in feat_names] for i in ids_fit])
y = np.array([by_id[i].get("fit", 0) for i in ids_fit], dtype=float)
print(f"\nfitting on {len(ids_fit)} non-honeypot labeled candidates, {len(feat_names)} features")

model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
pred_cv = cross_val_predict(model, X, y, cv=5)
rho_cv = spearmanr(pred_cv, y).statistic
print(f"5-fold CV Spearman(content_pred, teacher_fit) = {rho_cv:.3f}   (vs cosine {spearmanr(cos, fit).statistic:.3f})")

# combined content + cosine (what rank-time fusion approximates)
Xc = np.column_stack([X, np.array([by_id[i].get("retrieval_score", 0) for i in ids_fit])])
pred_cv2 = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=5.0)), Xc, y, cv=5)
print(f"combined (content+cosine) CV Spearman = {spearmanr(pred_cv2, y).statistic:.3f}")

model.fit(X, y)
sc = model.named_steps["standardscaler"]
rg = model.named_steps["ridge"]
order = np.argsort(-np.abs(rg.coef_))
print("\nstandardized coefficients (importance):")
for k in order:
    print(f"  {feat_names[k]:<18} {rg.coef_[k]:+.2f}")

weights = {
    "feature_order": feat_names,
    "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
    "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_),
    "cv_spearman": float(rho_cv), "n_fit": len(ids_fit),
}
json.dump(weights, open(os.path.join(ART, "content_weights.json"), "w"), indent=2)
print("\nsaved", os.path.join(ART, "content_weights.json"))
