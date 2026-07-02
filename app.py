"""
Sandbox UI (Streamlit / Hugging Face Spaces) — runs the EXACT rank.py pipeline on a small
uploaded sample (<=100 candidates). No precomputed cache is needed: rank.py live-embeds any
candidate whose id isn't cached, so a fresh sample is embedded on the fly (~20s for 100).

Deploy: push this repo to a HF Space (SDK: streamlit). Root needs app.py + requirements.txt +
the .py modules + artifacts/reranker.pkl. cand_emb.npy is NOT required for the sandbox.
"""
import os
import sys
import subprocess
import tempfile
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
RANK = os.path.join(HERE, "rank.py")

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob — Intelligent Candidate Ranker (sandbox)")
st.caption("Upload a small candidates JSONL (≤100). Runs the exact rank.py pipeline "
           "(frozen-E5 reranker + honeypot gate + behavioral modifier). CPU only.")

up = st.file_uploader("candidates.jsonl", type=["jsonl"])
if up is not None:
    with tempfile.TemporaryDirectory() as d:
        cin = os.path.join(d, "candidates.jsonl")
        cout = os.path.join(d, "submission.csv")
        with open(cin, "wb") as f:
            f.write(up.read())
        with st.spinner("Ranking (live-embedding the sample)…"):
            r = subprocess.run([sys.executable, RANK, "--candidates", cin, "--out", cout],
                               capture_output=True, text=True)
        st.text(r.stdout.strip() or r.stderr.strip())
        if os.path.exists(cout):
            df = pd.read_csv(cout)
            st.success(f"Ranked {len(df)} candidates.")
            st.dataframe(df, use_container_width=True, height=600)
            st.download_button("Download submission.csv", data=open(cout, "rb").read(),
                               file_name="submission.csv", mime="text/csv")
        else:
            st.error("Ranking failed — see log above.")
