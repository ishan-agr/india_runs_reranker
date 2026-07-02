# Redrob Hackathon — Intelligent Candidate Ranking


# the single command judged at rank-time (CPU-only, no network, ~40s on the released 100K):
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

## Run it — two ways (pick one)

Both produce `submission.csv`. Option A is the fast path for the released 100K; Option B needs no
Drive access but must embed the candidates first.

### Option A — use the precomputed embeddings from Drive (recommended, ~40 s for 100K)

```bash
# 1) install deps
pip install -r requirements.txt

# 2) one-time: download cand_emb.npy from Drive into artifacts/ (NOT part of ranking)
python download_embeddings.py
#    link is baked in; override with:  python download_embeddings.py --url "<drive link>"

# 3) the judged step — CPU only, no network
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Option B — generate the embeddings locally (no Drive; needs candidates.jsonl + torch)

```bash
# 1) install deps + CPU torch
pip install -r requirements.txt
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# 2) put candidates.jsonl in the repo root (or note its path for --candidates)

# 3) embed all candidates with E5 -> artifacts/cand_emb.npy + artifacts/cand_ids.json
#    (~25 min GPU / a few hours CPU; documented pre-computation, may exceed 5 min)
python phase1_embed.py --candidates ./candidates.jsonl

# 4) the judged step
python rank.py --candidates ./candidates.jsonl --out ./submission.csv


> Google Drive (cand_emb.npy):
> `https://drive.google.com/file/d/14cxg_ezmnPdwuOBtrr9yyvoLwFWTpR7J/view?usp=sharing`

## Approach (teacher → student, with a deterministic rule overlay)

## Setup

conda create -n redrob -c conda-forge python=3.10 -y
conda activate redrob
pip install -r requirements.txt
# CPU torch — only needed for Option B (regenerate embeddings) or the sandbox:
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

## Repo structure

| File | Role |
|---|---|
| `rank.py` | **Rank-time entry point** (the ≤5-min step). |
| `jd_spec.py` | JD parsed into must-haves / disqualifiers / do-not-want / keyword sets. |
| `serialize.py` | Candidate → E5 passage text. |
| `company_quality.py` | Curated 63-company prestige + industry signal (recruiter intuition #1). |
| `honeypot.py` | Deterministic logical-impossibility gate. |
| `behavioral.py` | redrob-signals availability multiplier [0.7, 1.15]. |
| `content_fit.py` | Content features + hard-cap vetoes (for the reranker + rank-time). |
| `download_embeddings.py` | Fetch `cand_emb.npy` from Google Drive (Option A). |
| `phase1_embed.py` | Offline: embed candidates with E5 → `artifacts/cand_emb.npy` (Option B). |
| `retrieve.py` | Offline: JD query → contender pool. |
| `phase2_teacher.py` | Offline: Claude Opus teacher labeling (needs API key). **Optional.** |
| `phase2_fallback.py` | Offline: heuristic labeler — makes the teacher step optional (no key). |
| `calibrate.py` / `phase3_rum.py` / `phase4_train_reranker.py` | Calibration, RUM, reranker training → `artifacts/reranker.pkl`. |
| `app.py` | Streamlit sandbox UI (live-embeds a small uploaded sample). |
| `artifacts/` | shipped: `reranker.pkl`, `cand_ids.json`, `teacher_labels.jsonl`, `rum_train.jsonl`, …; fetched: `cand_emb.npy`. |

## Full pipeline (regenerate everything, incl. training)

```bash
python phase1_embed.py --candidates ./candidates.jsonl   # E5 embeddings  -> artifacts/cand_emb.npy
python retrieve.py                                        # contender pool -> artifacts/retrieval_topk.json

# labels — pick ONE (the teacher is OPTIONAL):
python phase2_teacher.py --limit 400 --model claude-opus-4-8   # LLM teacher (needs ANTHROPIC key)
python phase2_fallback.py                                      # deterministic heuristic (no key)

python calibrate.py               # weight calibration   -> artifacts/content_weights.json
python phase3_rum.py              # hard-negative mining -> artifacts/rum_train.jsonl
python phase4_train_reranker.py   # learned reranker     -> artifacts/reranker.pkl
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

> The **shipped `reranker.pkl` was trained on the Opus teacher labels** (higher quality). Using it
> or reproducing the *ranking* needs no API key — the key was only used once, offline.

## AI tools

Claude Opus 4.8 as an **offline** recruiter-teacher (label generation only); Claude Code for
engineering and review. **No LLM/API at rank time.** See `submission_metadata.yaml` for the full
honest declaration.
