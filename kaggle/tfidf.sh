#!/bin/bash
# Kaggle CPU notebook: TF-IDF V3 (jobs/lib/v3_tfidf_steps.sh, as on the cluster). Outputs are zipped on exit, also
# after a failure; exits non-zero on failure.
set -euo pipefail
: "${ASSETS:?set ASSETS}"
export NLTK_DATA=/tmp/nltk_data  # extracted by kaggle/setup.sh
trap 'bash kaggle/package.sh tfidf || echo "packaging failed"' EXIT
mkdir -p logs
source jobs/lib/v3_tfidf_steps.sh
if ! v3_tfidf_steps > logs/kaggle_tfidf.log 2>&1; then tail -30 logs/kaggle_tfidf.log; echo "TFIDF FAILED"; exit 1; fi
echo "TFIDF COMPLETE"
