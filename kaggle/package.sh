#!/bin/bash
# Zip a Kaggle session's outputs to $OUT (default /kaggle/working, the only directory Kaggle keeps):
#   <name>_results.zip      results/ run directories, logs/, and (session2) the generated pair files
#   <name>_checkpoints.zip  trained checkpoints (stored, not compressed)
# Paths inside the zips are repo-relative: unzip at the repo root. Missing paths are skipped.
set -euo pipefail
NAME=${1:?usage: package.sh session1|tfidf|session2}
OUT=${OUT:-${KAGGLE_ROOT:-/kaggle}/working}
case $NAME in
  session1) R="results/v3/deberta_v3_train results/v3/score_deberta_v3 results/h1"; C="checkpoints/v3/deberta_v3 checkpoints/h1" ;;
  tfidf)    R="results/v3/tfidf_v3_train results/v3/score_tfidf_v3"; C="checkpoints/v3/tfidf_lr" ;;
  session2) R="results/pairs data/pairs/val_gen data/pairs/train_gen data/pairs/ict_s0"; C="" ;;
  *) echo "unknown session $NAME"; exit 2 ;;
esac
mkdir -p "$OUT"
zipit() {  # zipit ZIP LEVEL PATHS...
  local z=$1 lvl=$2; shift 2
  local have=() p
  for p in "$@"; do [ -e "$p" ] && have+=("$p"); done
  [ ${#have[@]} -gt 0 ] || return 0
  rm -f "$z.tmp"
  zip -q "$lvl" -r "$z.tmp" "${have[@]}" && mv "$z.tmp" "$z"
}
zipit "$OUT/${NAME}_results.zip" -6 $R logs
zipit "$OUT/${NAME}_checkpoints.zip" -0 $C
ls -la "$OUT"
