#!/bin/bash
# Login-node model prefetch (network I/O only; no installs, no compute): HF_HOME on scratch, the token from home.
#   bash scripts/prefetch_models.sh fetch [--only NAME ...]   # resumable; rerun after an interruption
#   bash scripts/prefetch_models.sh status                    # access + completeness, logged to results/transfer/prefetch/
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/lib/scratch.sh
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
touch_scratch >/dev/null
exec .venv/bin/python eval/transfer/prefetch.py "$@"
