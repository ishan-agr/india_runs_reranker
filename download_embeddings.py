"""
download_embeddings.py — fetch the precomputed E5 embeddings and place them in artifacts/.

ONE-TIME setup step, run BEFORE ranking the released 100K. It downloads a zip from the shared
Google Drive link and unzips `cand_emb.npy` + `cand_ids.json` into ./artifacts/. After this,
`rank.py` runs CPU-only with NO network (this download is NOT part of the ranking step).

    python download_embeddings.py                    # uses EMBEDDINGS_URL below / env var
    python download_embeddings.py --url "<drive link>"
    EMBEDDINGS_URL="<drive link>" python download_embeddings.py

The sandbox does NOT need this — it live-embeds each uploaded sample on the fly.

If you'd rather do it by hand: download the zip from the Drive link, unzip it, and place
`cand_emb.npy` and `cand_ids.json` directly inside the `artifacts/` folder next to reranker.pkl.
"""
import argparse
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

# Google Drive share link (or bare file id) of the zip containing BOTH cand_emb.npy and
# cand_ids.json. Fill this before submitting, or pass --url / set EMBEDDINGS_URL.
EMBEDDINGS_URL = os.environ.get(
    "EMBEDDINGS_URL",
    "<FILL https://drive.google.com/file/d/FILE_ID/view?usp=sharing>",
)

NEEDED = ("cand_emb.npy", "cand_ids.json")


def _flatten_into_art():
    """If the zip nested the files in a subfolder, move them up into artifacts/."""
    for name in NEEDED:
        if os.path.exists(os.path.join(ART, name)):
            continue
        for root, _dirs, files in os.walk(ART):
            if name in files:
                src = os.path.join(root, name)
                if os.path.abspath(src) != os.path.abspath(os.path.join(ART, name)):
                    shutil.move(src, os.path.join(ART, name))
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=EMBEDDINGS_URL, help="Google Drive link or file id of the zip")
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    if all(os.path.exists(os.path.join(ART, n)) for n in NEEDED):
        print(f"Already present in {ART}: {', '.join(NEEDED)} - nothing to do.")
        return

    url = args.url
    if not url or url.startswith("<FILL"):
        sys.exit(
            "No embeddings URL set. Provide the Google Drive link via --url, the EMBEDDINGS_URL "
            "env var, or by editing EMBEDDINGS_URL in download_embeddings.py."
        )

    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required for the download. Install it with:  pip install gdown")

    zip_path = os.path.join(ART, "_cand_emb_bundle.zip")
    print(f"Downloading embeddings bundle from Drive ...")
    out = gdown.download(url=url, output=zip_path, quiet=False, fuzzy=True)
    if not out or not os.path.exists(zip_path):
        sys.exit(
            "gdown could not download the file. Check the link is public ('Anyone with the "
            "link') and correct. You can also download + unzip it manually into artifacts/."
        )

    print("Unzipping into artifacts/ ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ART)
    os.remove(zip_path)
    _flatten_into_art()

    missing = [n for n in NEEDED if not os.path.exists(os.path.join(ART, n))]
    if missing:
        sys.exit(
            f"Unzip finished but these are still missing from artifacts/: {missing}. "
            "The zip must contain cand_emb.npy and cand_ids.json."
        )
    size_mb = os.path.getsize(os.path.join(ART, "cand_emb.npy")) / 1e6
    print(f"Ready: artifacts/cand_emb.npy ({size_mb:.0f} MB) + artifacts/cand_ids.json")
    print("Now run:  python rank.py --candidates ./candidates.jsonl --out ./submission.csv")


if __name__ == "__main__":
    main()
