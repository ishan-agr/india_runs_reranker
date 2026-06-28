"""Quick throughput bench for E5-base on this CPU at several sequence lengths."""
import os
import sys
import time
import json

import torch
torch.set_num_threads(os.cpu_count())

from sentence_transformers import SentenceTransformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serialize import to_passage_text  # noqa: E402

CANDIDATES = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
              r"\India_runs_data_and_ai_challenge\candidates.jsonl")
N = 256
texts = []
with open(CANDIDATES, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= N:
            break
        texts.append(to_passage_text(json.loads(line)))

print("cpu_count", os.cpu_count(), "| torch_threads", torch.get_num_threads())
m = SentenceTransformer("intfloat/e5-base-v2")
# warm up
m.encode(texts[:32], batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
for seq in [512, 256, 192, 128]:
    m.max_seq_length = seq
    t = time.time()
    m.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    dt = time.time() - t
    rate = len(texts) / dt
    print(f"seq={seq:4d}: {rate:6.1f} texts/s  ({dt:5.1f}s/{len(texts)})  est_100k={100000/rate/60:5.1f} min")
