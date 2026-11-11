#!/bin/bash
# Kaggle environment for the moved jobs (docs/deviations.md 2026-11-11). Run from the repo root (a clone of the
# repo.bundle in the ipi-assets dataset): bash kaggle/setup.sh <session1|tfidf|session2>
# - pins the libraries that affect scores to the cluster env's versions (transformers, tokenizers, scikit-learn,
#   pandas, ...); torch stays Kaggle's own build (constraint file: pip may not replace it); versions -> logs/
# - links gitignored assets into the repo (nltk_data, the agent transfer sets); nothing tracked is touched
# - session1: downloads microsoft/deberta-v3-base at snapshot 8ccc9b6 (the V2 runs' base) to /tmp/hf
set -euo pipefail
MODE=${1:?usage: setup.sh session1|tfidf|session2}
ASSETS=${ASSETS:-/kaggle/input/ipi-assets}
mkdir -p logs
python -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > /tmp/torch_constraint.txt
pip install -q -c /tmp/torch_constraint.txt transformers==5.20.1 tokenizers==0.23.3 scikit-learn==1.9.1 \
  pandas==3.0.6 huggingface_hub==2.2.0 accelerate==1.15.0 safetensors==0.8.0 sentencepiece==0.2.2 nltk==3.10.3 \
  PyYAML==6.0.3 protobuf==7.36.2
ln -sfn "$ASSETS/nltk_data" nltk_data
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
print(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
print(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout or "clean tree")
if mode != "tfidf" and torch.cuda.device_count() < 2:
    sys.exit("need the 'GPU T4 x2' accelerator")
PY
