"""
download_embeddings.py — fetch the precomputed E5 embeddings and place them in artifacts/.

ONE-TIME setup step, run BEFORE ranking the released 100K. It downloads the file at the shared
Google Drive link and puts `cand_emb.npy` (100K x 768, ~300 MB) into ./artifacts/. After this,
`rank.py` runs CPU-only with NO network (this download is NOT part of the ranking step).

`cand_ids.json` already ships inside the repo, so the Drive file only has to contain the .npy.
The link may point to a raw `cand_emb.npy` OR a `.zip` that contains it — both are handled.

    python download_embeddings.py                    # uses EMBEDDINGS_URL below / env var
    python download_embeddings.py --url "<drive link>"
    EMBEDDINGS_URL="<drive link>" python download_embeddings.py

The sandbox does NOT need this — it live-embeds each uploaded sample on the fly.

By hand instead: download the file from the Drive link, and (unzip if needed) place
`cand_emb.npy` directly inside the `artifacts/` folder next to `reranker.pkl`.
"""
import argparse
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

# Google Drive share link (or bare file id) of cand_emb.npy (or a zip containing it).
EMBEDDINGS_URL = os.environ.get(
    "EMBEDDINGS_URL",
    "https://drive.google.com/file/d/14cxg_ezmnPdwuOBtrr9yyvoLwFWTpR7J/view?usp=sharing",
)

NPY = os.path.join(ART, "cand_emb.npy")
IDS = os.path.join(ART, "cand_ids.json")


def _place_npy_from_zip(zip_path):
    """Extract cand_emb.npy (and cand_ids.json if present) from a zip into artifacts/."""
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ART)
    for name in ("cand_emb.npy", "cand_ids.json"):
        dst = os.path.join(ART, name)
        if os.path.exists(dst):
            continue
        for root, _dirs, files in os.walk(ART):
            if name in files:
                shutil.move(os.path.join(root, name), dst)
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=EMBEDDINGS_URL, help="Google Drive link or file id")
    ap.add_argument("--force", action="store_true", help="re-download even if cand_emb.npy exists")
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    if os.path.exists(NPY) and not args.force:
        print(f"Already present: {NPY} - nothing to do.")
        return

    url = args.url
    if not url or url.startswith("<FILL"):
        sys.exit("No embeddings URL set. Pass --url, set EMBEDDINGS_URL, or edit EMBEDDINGS_URL.")

    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required for the download. Install it with:  pip install gdown")

    tmp = os.path.join(ART, "_download.part")
    print("Downloading embeddings from Drive ...")
    out = gdown.download(url=url, output=tmp, quiet=False, fuzzy=True)
    if not out or not os.path.exists(tmp):
        sys.exit(
            "gdown could not download the file. Check the link is shared 'Anyone with the link'. "
            "You can also download it by hand and place cand_emb.npy in artifacts/."
        )

    if zipfile.is_zipfile(tmp):
        print("Downloaded a zip - extracting cand_emb.npy into artifacts/ ...")
        _place_npy_from_zip(tmp)
        os.remove(tmp)
    else:
        print("Placing cand_emb.npy into artifacts/ ...")
        shutil.move(tmp, NPY)

    if not os.path.exists(NPY):
        sys.exit("Download finished but artifacts/cand_emb.npy is still missing.")
    size_mb = os.path.getsize(NPY) / 1e6
    print(f"Ready: artifacts/cand_emb.npy ({size_mb:.0f} MB)")
    if not os.path.exists(IDS):
        print("WARNING: artifacts/cand_ids.json is missing — it normally ships in the repo and "
              "rank.py needs it to map embedding rows to candidate_ids.")
    print("Now run:  python rank.py --candidates ./candidates.jsonl --out ./submission.csv")


if __name__ == "__main__":
    main()
