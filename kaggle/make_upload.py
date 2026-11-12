"""
make_upload.py
==============
Builds the folder to upload to Kaggle for the moved jobs (docs/deviations.md 2026-11-11):

  <dest>/ipi-assets/            -> one private Kaggle dataset, slug "ipi-assets"
      repo.bundle                 git bundle of main; the notebooks check out COMMIT from it
      COMMIT.txt
      sets_adliu/                 agent transfer sets + calibration pool (cluster: data/transfer/sets_adliu)
      v2_ckpts/deberta_s{42,123,999}/  existing V2 DeBERTa checkpoints (cluster: ~/nlp/checkpoints), no CSVs
      nltk_data.tar               the repo's nltk_data/ (stopwords, wordnet, punkt; TF-IDF preprocessing) as one
                                  deterministic tar: Kaggle extracts every .zip on upload, so the NLTK zips cannot
                                  be shipped as files; kaggle/setup.sh checks its sha256 and extracts it to /tmp
      tokenizers_hpc/deberta_v2_ckpt/  the V3 tokenizer (pair length rules; sha256-checked by src/pairs_gen.py)
  <dest>/notebooks/{session1,tfidf,session2}.ipynb   with COMMIT filled in

The cluster files are fetched with rsync (ssh host bits-hpc). COMMIT must be pushed and equal to local HEAD.

    python kaggle/make_upload.py --dest ~/kaggle_upload
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOTEBOOKS = {
    "session1": ("V3 baselines and H1 on Kaggle (session 1)",
                 "Accelerator: GPU T4 x2. Internet: on. Input: dataset ipi-assets. Run with Save Version -> Save & Run All.",
                 "session1", "kaggle/session1.sh", False),
    "tfidf": ("TF-IDF V3 on Kaggle (CPU)",
              "Accelerator: None. Internet: on. Input: dataset ipi-assets. Run with Save Version -> Save & Run All.",
              "tfidf", "kaggle/tfidf.sh", False),
    "session2": ("Val and TRAIN pairs on Kaggle (session 2)",
                 "Accelerator: GPU T4 x2. Internet: on. Input: dataset ipi-assets (+ the previous version's output when "
                 "continuing). Secret: HF_TOKEN (Add-ons -> Secrets). Run with Save Version -> Save & Run All.",
                 "session2", "kaggle/session2.sh", True),
}


NLTK_TAR_SHA256 = re.search(r'^NLTK_TAR_SHA256=([0-9a-f]{64})$', (ROOT / "kaggle/setup.sh").read_text(), re.M).group(1)


def nltk_tar(src: Path, out: Path) -> str:
    """Deterministic tar of src (sorted entries, zero mtime/owner, fixed modes) -> sha256 hex."""
    with tarfile.open(out, "w", format=tarfile.USTAR_FORMAT) as tf:
        for f in sorted([src, *src.rglob("*")]):
            if f.name == ".DS_Store":
                continue
            ti = tf.gettarinfo(str(f), arcname=str(Path("nltk_data") / f.relative_to(src)))
            ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
            ti.mode = 0o755 if f.is_dir() else 0o644
            if f.is_file():
                with open(f, "rb") as fh:
                    tf.addfile(ti, fh)
            else:
                tf.addfile(ti)
    return hashlib.sha256(out.read_bytes()).hexdigest()


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


CELL_SETUP = """import glob, os, subprocess

COMMIT = "{commit}"
KR = os.environ.get("KAGGLE_ROOT", "/kaggle")  # set only by the local layout test
REPO = os.environ.get("REPO_DIR", "/tmp/nlp")
bundles = sorted(glob.glob(f"{{KR}}/input/**/repo.bundle", recursive=True))
assert len(bundles) == 1, f"expected one repo.bundle under {{KR}}/input, found {{bundles}}"
ASSETS = os.path.dirname(bundles[0])
os.environ["ASSETS"] = ASSETS
os.environ["NLTK_DATA"] = "/tmp/nltk_data"  # extracted from nltk_data.tar by kaggle/setup.sh
{secret}
def run(cmd, cwd=None):
    \"\"\"Run a command, stream its output into the notebook, raise CalledProcessError on a non-zero exit.\"\"\"
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)

print("ASSETS =", ASSETS)
"""

CELL_CLONE = """run(["rm", "-rf", REPO])
run(["git", "clone", "-q", os.path.join(ASSETS, "repo.bundle"), REPO])
run(["git", "checkout", "-q", COMMIT], cwd=REPO)
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
assert head == COMMIT, f"checked out {head}, expected {COMMIT}"
assert os.path.exists(os.path.join(REPO, "kaggle", "setup.sh")), "kaggle/setup.sh missing in the bundle"
os.chdir(REPO)
print("repo at", head)
"""

SECRET = """from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
"""


def lines(text):
    return text.splitlines(keepends=True)


def notebook(commit, title, how, mode, run, secret):
    cells = [cell("markdown", lines(f"# {title}\n\n{how}\n\nRepo commit: `{commit}`. Outputs: zips in /kaggle/working."))]
    cells.append(cell("code", lines(CELL_SETUP.format(commit=commit, secret=SECRET if secret else ""))))
    cells.append(cell("code", lines(CELL_CLONE)))
    cells.append(cell("code", lines(f'run(["bash", "kaggle/setup.sh", "{mode}"])\n')))
    cells.append(cell("code", lines(f'run(["bash", "{run}"])\n')))
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
    sh("git", "bundle", "create", str(assets / "repo.bundle"), "HEAD", "main")
    (assets / "COMMIT.txt").write_text(commit + "\n")
    shutil.rmtree(assets / "nltk_data", ignore_errors=True)  # older builds shipped the tree (Kaggle extracts its zips)
    digest = nltk_tar(ROOT / "nltk_data", assets / "nltk_data.tar")
    if digest != NLTK_TAR_SHA256:
        raise SystemExit(f"nltk_data.tar sha256 {digest} != {NLTK_TAR_SHA256} pinned in kaggle/setup.sh")
    dst = assets / "tokenizers_hpc/deberta_v2_ckpt"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(ROOT / "hf_cache/tokenizers_hpc/deberta_v2_ckpt", dst)
    zips = sorted(str(z.relative_to(assets)) for z in assets.rglob("*.zip"))
    if zips:  # Kaggle would extract them on upload
        raise SystemExit(f"zip files in the upload folder: {zips}")
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
