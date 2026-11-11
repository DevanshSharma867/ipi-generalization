#!/bin/bash
# Kaggle environment for the moved jobs (docs/deviations.md 2026-11-11). Run from the repo root (a clone of the
# repo.bundle in the ipi-assets dataset, done by the notebook): bash kaggle/setup.sh <session1|tfidf|session2>
# Needs ASSETS (the dataset folder holding repo.bundle; set by notebook cell 1).
# - pins the libraries that affect scores to the cluster env's versions (transformers, tokenizers, scikit-learn,
#   pandas, ...); torch stays Kaggle's own build (constraint file: pip may not replace it); versions -> logs/
# - NLTK data is read from $ASSETS/nltk_data via NLTK_DATA (nothing is put in the repo); the archives must match
#   the cluster's copies (sha256 below)
# - copies the agent transfer sets to data/transfer/sets_adliu (gitignored)
# - session1: downloads microsoft/deberta-v3-base at snapshot 8ccc9b6 (the V2 runs' base) to /tmp/hf
# - fails unless the git tree is clean (src/runlog.py refuses dirty trees)
set -euo pipefail
MODE=${1:?usage: setup.sh session1|tfidf|session2}
: "${ASSETS:?set ASSETS to the dataset folder holding repo.bundle}"
export NLTK_DATA=$ASSETS/nltk_data
mkdir -p logs
python -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > /tmp/torch_constraint.txt
pip install -q -c /tmp/torch_constraint.txt transformers==5.20.1 tokenizers==0.23.3 scikit-learn==1.9.1 \
  pandas==3.0.6 huggingface_hub==2.2.0 accelerate==1.15.0 safetensors==0.8.0 sentencepiece==0.2.2 nltk==3.10.3 \
  PyYAML==6.0.3 protobuf==7.36.2
# NLTK archives used by the cluster's V2/V3 TF-IDF (~/nlp/nltk_data on the cluster; no omw-1.4 there)
python - <<'PY'
import hashlib, os, sys
want = {"corpora/stopwords.zip": "48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b",
        "corpora/wordnet.zip": "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59",
        "tokenizers/punkt_tab.zip": "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
        "tokenizers/punkt.zip": "51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec"}
root = os.environ["NLTK_DATA"]
bad = [f for f, h in want.items() if not os.path.exists(os.path.join(root, f))
       or hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest() != h]
sys.exit(f"NLTK archives missing or different from the cluster's: {bad}") if bad else print("nltk archives match the cluster")
PY
python - <<'PY'
import sys
sys.path.insert(0, "src")
import nltk
import preprocess  # sets nltk.data.path to $NLTK_DATA only
assert nltk.data.path == [__import__("os").environ["NLTK_DATA"]], nltk.data.path
out = preprocess.preprocess_for_tfidf(["The emails were running and attached files"])
assert out == ["email running attached file"], out  # the cluster's output for this sentence
print("nltk ok:", out)
PY
mkdir -p data/transfer
[ -d data/transfer/sets_adliu ] || cp -r "$ASSETS/sets_adliu" data/transfer/sets_adliu
if [ "$MODE" = session1 ]; then
  python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("microsoft/deberta-v3-base", revision="8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
                      local_dir="/tmp/hf/deberta-v3-base")
print("base model:", p)
PY
fi
python - "$MODE" <<'PY' | tee "logs/kaggle_env_$MODE.txt"
import importlib.metadata as m, platform, subprocess, sys
mode = sys.argv[1]
print("python", platform.python_version())
for p in ("torch", "transformers", "tokenizers", "scikit-learn", "pandas", "numpy", "nltk", "accelerate", "safetensors"):
    print(p, m.version(p))
import torch
print("cuda", torch.cuda.is_available(), torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("commit", subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
if mode != "tfidf" and torch.cuda.device_count() < 2 and not __import__("os").environ.get("KAGGLE_CPU_DRYRUN"):
    sys.exit("need the 'GPU T4 x2' accelerator")
PY
if [ -n "$(git status --porcelain)" ]; then git status --porcelain; echo "git tree not clean"; exit 1; fi
echo "setup $MODE ok (clean tree)"
