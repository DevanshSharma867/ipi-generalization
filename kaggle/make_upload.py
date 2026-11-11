"""
make_upload.py
==============
Builds the folder to upload to Kaggle for the moved jobs (docs/deviations.md 2026-11-11):

  <dest>/ipi-assets/            -> one private Kaggle dataset, slug "ipi-assets"
      repo.bundle                 git bundle of main; the notebooks check out COMMIT from it
      COMMIT.txt
      sets_adliu/                 agent transfer sets + calibration pool (cluster: data/transfer/sets_adliu)
      v2_ckpts/deberta_s{42,123,999}/  existing V2 DeBERTa checkpoints (cluster: ~/nlp/checkpoints), no CSVs
      nltk_data/                  stopwords, wordnet, punkt (TF-IDF preprocessing)
      tokenizers_hpc/deberta_v2_ckpt/  the V3 tokenizer (pair length rules; sha256-checked by src/pairs_gen.py)
  <dest>/notebooks/{session1,tfidf,session2}.ipynb   with COMMIT filled in

The cluster files are fetched with rsync (ssh host bits-hpc). COMMIT must be pushed and equal to local HEAD.

    python kaggle/make_upload.py --dest ~/kaggle_upload
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOTEBOOKS = {
    "session1": ("V3 baselines and H1 on Kaggle (session 1)",
                 "Accelerator: GPU T4 x2. Internet: on. Input: dataset ipi-assets. Run with Save Version -> Save & Run All.",
                 "session1", "bash kaggle/session1.sh", False),
    "tfidf": ("TF-IDF V3 on Kaggle (CPU)",
              "Accelerator: None. Internet: on. Input: dataset ipi-assets. Run with Save Version -> Save & Run All.",
              "tfidf", "bash kaggle/tfidf.sh", False),
    "session2": ("Val and TRAIN pairs on Kaggle (session 2)",
                 "Accelerator: GPU T4 x2. Internet: on. Input: dataset ipi-assets (+ the previous version's output when "
                 "continuing). Secret: HF_TOKEN (Add-ons -> Secrets). Run with Save Version -> Save & Run All.",
                 "session2", "bash kaggle/session2.sh", True),
}


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def notebook(commit, title, how, mode, run, secret):
    cells = [cell("markdown", f"# {title}\n\n{how}\n\nRepo commit: `{commit}`. Outputs: zips in /kaggle/working.")]
    setup = ["import os\n", f'COMMIT = "{commit}"\n', 'os.environ["ASSETS"] = "/kaggle/input/ipi-assets"\n']
    if secret:
        setup += ["from kaggle_secrets import UserSecretsClient\n",
                  'os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")\n']
    cells.append(cell("code", setup))
    cells.append(cell("code", ["!rm -rf /tmp/nlp && git clone -q /kaggle/input/ipi-assets/repo.bundle /tmp/nlp\n",
                               "%cd /tmp/nlp\n", "!git checkout -q $COMMIT && git rev-parse HEAD\n"]))
    cells.append(cell("code", [f"!bash kaggle/setup.sh {mode}\n"]))
    cells.append(cell("code", [f"!{run}\n"]))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


def sh(*cmd, cwd=ROOT):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--skip_cluster", action="store_true", help="reuse sets_adliu/ and v2_ckpts/ already in dest")
    a = ap.parse_args()
    commit = sh("git", "rev-parse", "HEAD")
    if sh("git", "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit first: tracked files are modified")
    if not sh("git", "branch", "-r", "--contains", commit):
        raise SystemExit(f"push {commit} first")
    assets, nbs = a.dest / "ipi-assets", a.dest / "notebooks"
    assets.mkdir(parents=True, exist_ok=True)
    nbs.mkdir(parents=True, exist_ok=True)
    sh("git", "bundle", "create", str(assets / "repo.bundle"), "main")
    (assets / "COMMIT.txt").write_text(commit + "\n")
    for src, dst in ((ROOT / "nltk_data", assets / "nltk_data"),
                     (ROOT / "hf_cache/tokenizers_hpc/deberta_v2_ckpt", assets / "tokenizers_hpc/deberta_v2_ckpt")):
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.zip"))
    if not a.skip_cluster:
        sh("rsync", "-a", "bits-hpc:nlp_harness/data/transfer/sets_adliu/", str(assets / "sets_adliu") + "/")
        for s in (42, 123, 999):
            d = assets / "v2_ckpts" / f"deberta_s{s}"
            d.mkdir(parents=True, exist_ok=True)
            sh("rsync", "-a", "--exclude=*.csv", f"bits-hpc:nlp/checkpoints/deberta_s{s}/", str(d) + "/")
    for name, (title, how, mode, run, secret) in NOTEBOOKS.items():
        (nbs / f"{name}.ipynb").write_text(json.dumps(notebook(commit, title, how, mode, run, secret), indent=1))
    print(f"commit {commit}\n{assets}\n{nbs}")


if __name__ == "__main__":
    main()
