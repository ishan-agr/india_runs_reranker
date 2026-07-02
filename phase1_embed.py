"""
Phase 1 — embed all candidates with E5-base-v2 (offline precompute, CPU).

Streams candidates.jsonl, serializes each to an E5 passage string, encodes with
intfloat/e5-base-v2 (normalized -> cosine == dot product), saves:
  artifacts/cand_emb.npy   float32 [N, 768]
  artifacts/cand_ids.json  ordered list of candidate_id (row i <-> emb[i])

Usage:
  python phase1_embed.py --candidates ./candidates.jsonl               # full set
  python phase1_embed.py --candidates ./candidates.jsonl --limit 200   # smoke test on first 200
"""
import sys
import os
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serialize import to_passage_text  # noqa: E402

import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="Embed candidates with E5-base-v2 -> artifacts/cand_emb.npy")
ap.add_argument("--candidates", default="candidates.jsonl",
                help="path to candidates.jsonl (default: ./candidates.jsonl)")
ap.add_argument("--limit", type=int, default=None,
                help="only embed the first N candidates (smoke test)")
args = ap.parse_args()

CANDIDATES = args.candidates
OUTDIR = os.path.join(HERE, "artifacts")
MODEL = "intfloat/e5-base-v2"
BATCH = 64

os.makedirs(OUTDIR, exist_ok=True)
limit = args.limit

t_load = time.time()
ids, texts = [], []
with open(CANDIDATES, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if limit is not None and i >= limit:
            break
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        ids.append(r["candidate_id"])
        texts.append(to_passage_text(r))
print(f"loaded {len(texts)} candidates in {time.time()-t_load:.1f}s")

model = SentenceTransformer(MODEL)  # first run downloads ~440MB
print("model loaded:", MODEL, "| max_seq_len:", model.max_seq_length)

t0 = time.time()
emb = model.encode(
    texts, batch_size=BATCH, normalize_embeddings=True,
    show_progress_bar=True, convert_to_numpy=True,
)
dt = time.time() - t0
emb = emb.astype("float32")
print(f"encoded {emb.shape} in {dt:.1f}s => {len(texts)/dt:.1f} texts/s")

suffix = f"_{limit}" if limit is not None else ""
emb_path = os.path.join(OUTDIR, f"cand_emb{suffix}.npy")
ids_path = os.path.join(OUTDIR, f"cand_ids{suffix}.json")
np.save(emb_path, emb)
with open(ids_path, "w", encoding="utf-8") as f:
    json.dump(ids, f)
print("saved:", emb_path, "|", ids_path)
