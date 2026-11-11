#!/bin/bash
# Kaggle CPU notebook: TF-IDF V3 (jobs/lib/v3_tfidf_steps.sh, as on the cluster), then zip the outputs.
set -uo pipefail
source jobs/lib/v3_tfidf_steps.sh
v3_tfidf_steps > logs/kaggle_tfidf.log 2>&1 && echo "tfidf done" || echo "tfidf FAILED (logs/kaggle_tfidf.log)"
bash kaggle/package.sh tfidf
