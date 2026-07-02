# Redrob Hackathon — Intelligent Candidate Ranking


## Approach (teacher → student, with a deterministic rule overlay)

The dataset has **no relevance labels** and rank-time is hard-constrained (**CPU-only, ≤5 min /
100K, ≤16 GB, no network/LLM**). So we manufacture the missing judgment offline and distill it
into a cheap rank-time scorer.


OFFLINE (GPU/LLM allowed)                          RANK-TIME (CPU, <40s, offline)
─────────────────────────                          ──────────────────────────────
E5-base-v2 embeds 100K                            load reranker.pkl + cand_emb.npy
retrieve ~1,500 pool                               reranker scores all 100K
Claude Opus 4.8 teacher  labels ~400             ── honeypot gate  → exclude impossible
  (tier + fit + rationale)                         ── hard-cap vetoes (stuffer / services /
RUM hard-negative mining                           ──   wrong-domain)
learned reranker (Ridge                            ── behavioral multiplier (availability)
  on frozen-E5 features) → reranker.pkl           top-100 + templated reasoning → CSV
```

Why this shape (validated, not assumed):
- Raw embedding cosine has decent *recall* but is **noise at the very top** — Spearman 0.08
  within the top-20 (where NDCG@10 lives). The teacher supplies the top ordering.
- A **frozen-E5 learned reranker** on `[19 content features ⊕ cosine ⊕ cos-to-positive-centroid
  ⊕ cos-to-hard-negative-centroid]` reaches **Spearman ≈0.71** vs the teacher (cosine 0.59 →
  content 0.67 → +embedding-prototypes 0.71). The teacher-supervised prototypes are the lift.
- Recruiter intuitions are **empirically confirmed** by the teacher (top coefficients:
  endorsements/corroboration, services-penalty, college tier-1, company quality, tenure).
- A **deterministic honeypot gate** (provable impossibilities: yoe > career date-span,
  overlapping tenures beyond the timeline, ≥3 "expert" skills at 0 months) catches ~68 of the
  ~80 honeypots with ~100% precision → **0 honeypots in the top-100**.


## Setup

conda create -n redrob -c conda-forge python=3.10 -y && conda activate redrob
pip install -r requirements.txt
# CPU torch (only needed for precompute / sandbox):
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
```

## Reproduce the ranking (evaluator quick start)

`reranker.pkl` is shipped in `artifacts/`. The one large precomputed artifact —
`artifacts/cand_emb.npy` (100K×768, ~300 MB) — is hosted on Google Drive (link in
`submission_metadata.yaml: embeddings_url`, also read by `download_embeddings.py`).

```bash
pip install -r requirements.txt

# 1) ONE-TIME setup: download + unzip the precomputed embeddings into artifacts/
python download_embeddings.py                       # or: --url "<drive link>"
#    (drops cand_emb.npy + cand_ids.json next to reranker.pkl)
[*DRIVE LINK*:- https://drive.google.com/file/d/14cxg_ezmnPdwuOBtrr9yyvoLwFWTpR7J/view?usp=sharing ]


# 2) The judged ≤5-min step (CPU-only, no network):
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
## Repo structure

| File | Role |
|---|---|
| `rank.py` | **Rank-time entry point** (the ≤5-min step). 
| `jd_spec.py` | JD parsed into must-haves / disqualifiers / do-not-want / keyword sets. 
| `serialize.py` | Candidate → E5 passage text. 
| `company_quality.py` | Curated 63-company prestige + industry signal (recruiter intuition #1). 
| `honeypot.py` | Deterministic logical-impossibility gate. 
| `behavioral.py` | redrob-signals availability multiplier [0.7, 1.15]. 
| `content_fit.py` | Content features + hard-cap vetoes (for the reranker + rank-time). 
| `phase1_embed.py` | Offline: embed 100K with E5 → `artifacts/cand_emb.npy`. 
| `retrieve.py` | Offline: JD query → contender pool. 
| `phase2_teacher.py` | Offline: Claude Opus teacher labeling (needs API key; see below). 
| `calibrate.py` / `phase3_rum.py` / `phase4_train_reranker.py` | Calibration, RUM, reranker training → `artifacts/reranker.pkl`.
| `artifacts/` | `reranker.pkl`, `teacher_labels.jsonl`, `rum_train.jsonl`, …. 


Step 1 is documented **pre-computation** (a plain download; not part of ranking). To do it by
hand instead: download the zip from the Drive link, unzip it, and place `cand_emb.npy` +
`cand_ids.json` directly inside `artifacts/`. You can also regenerate the npy from scratch
(deterministic, ~25 min GPU) with `python phase1_embed.py`.

Step 2 loads only the precomputed cache + a scikit-learn model (torch is not even imported, no
network). If the cache is missing on the full set, rank.py **stops with a clear message** telling
you to run `download_embeddings.py` first — rather than silently live-embedding 100K, which would
blow the time limit.

### Measured timings (10-candidate end-to-end, this machine, CPU)

| Path | What loads | Wall time |
|---|---|---|
| **Precomputed** (ids in cache), warm | 300 MB npy + sklearn model | **~5 s** |
| **Precomputed**, cold disk | same, npy read cold | ~22 s |
| **Live-embed** (no cache, E5 model cached, offline) | E5 model + embed the sample | **~17 s** |
| Live-embed, first ever load (HF Hub network check) | + Hub round-trip | ~50 s |

The full **100K precomputed run is ~40 s**. The live path's cost is a fixed ~13 s E5 model load
(independent of sample size), so a ~100-row sandbox upload is also ~17–20 s.

## Compute compliance

| Constraint | This system |
|---|---|
| ≤5 min / 100K | ~40 s (CPU) |
| ≤16 GB RAM | ~1–2 GB |
| CPU only at rank time | yes (torch not even imported on the released set) |
| No network / no LLM at rank time | yes (`rank.py` makes no API calls) |
| Honeypots in top-100 | **0** |
| Official validator | **passes** (`validate_submission.py`) |

## Full pipeline (regenerate everything)

```bash
python phase1_embed.py            # E5 embeddings        -> artifacts/cand_emb.npy
python retrieve.py                # contender pool       -> artifacts/retrieval_topk.json
# offline teacher (needs an Anthropic key in .secrets/anthropic_key.txt — gitignored):
python phase2_teacher.py --limit 400 --model claude-opus-4-8   -> artifacts/teacher_labels.jsonl [OPTIONAL]
python calibrate.py               # weight calibration   -> artifacts/content_weights.json
python phase3_rum.py              # hard-negative mining -> artifacts/rum_train.jsonl
python phase4_train_reranker.py   # learned reranker     -> artifacts/reranker.pkl
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
> The teacher step sends candidate profiles to the Anthropic API **offline** to generate silver
> labels (declared in `submission_metadata.yaml`). It is **not** part of ranking. The resulting
> `reranker.pkl` + `teacher_labels.jsonl` are shipped so judges need not re-run it.

## Sandbox (deployed = fully live)

The deployed sandbox does **everything live**: it ships **no** `cand_emb.npy`, so it loads the E5
model and embeds each uploaded candidate on the fly, then runs the identical reranker + honeypot
gate + hard-caps + behavioral pipeline. Same code path as `rank.py`; only the embedding source
differs (live model vs. precomputed cache).

`app.py` is a Streamlit UI over that path for a small uploaded sample (≤100). Deploy to a
**Hugging Face Space** (SDK: streamlit); the root needs `app.py`, `requirements.txt`, the `.py`
modules, and `artifacts/reranker.pkl` (**not** the 300 MB npy). Set `HF_HUB_OFFLINE=1` in the
Space env after the first run to skip the Hub round-trip (~50 s → ~17 s cold start). Put the URL
in `submission_metadata.yaml: sandbox_link`.

## AI tools

Claude Opus 4.8 as an **offline** recruiter-teacher (label generation only); Claude Code for
engineering and review. **No LLM/API at rank time.** See `submission_metadata.yaml` for the full
honest declaration.
