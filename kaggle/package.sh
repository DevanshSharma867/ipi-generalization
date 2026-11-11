#!/bin/bash
# Zip a Kaggle session's outputs to /kaggle/working (the only directory Kaggle keeps):
#   <name>_results.zip      results/ run directories, logs/, and (session2) the generated pair files
#   <name>_checkpoints.zip  trained checkpoints (stored, not compressed)
# Paths inside the zips are repo-relative: unzip at the repo root.
set -uo pipefail
NAME=${1:?usage: package.sh session1|tfidf|session2}
OUT=${OUT:-/kaggle/working}
case $NAME in
  session1) R="results/v3/deberta_v3_train results/v3/score_deberta_v3 results/h1"; C="checkpoints/v3/deberta_v3 checkpoints/h1" ;;
  tfidf)    R="results/v3/tfidf_v3_train results/v3/score_tfidf_v3"; C="checkpoints/v3/tfidf_lr" ;;
  session2) R="results/pairs data/pairs/val_gen data/pairs/train_gen data/pairs/ict_s0"; C="" ;;
esac
EXIST=$(for p in $R logs; do [ -e "$p" ] && echo "$p"; done)
rm -f "$OUT/${NAME}_results.zip.tmp"
zip -qr "$OUT/${NAME}_results.zip.tmp" $EXIST && mv "$OUT/${NAME}_results.zip.tmp" "$OUT/${NAME}_results.zip"
CE=$(for p in $C; do [ -e "$p" ] && echo "$p"; done)
if [ -n "$CE" ]; then
  zip -0 -qr "$OUT/${NAME}_checkpoints.zip.tmp" $CE && mv "$OUT/${NAME}_checkpoints.zip.tmp" "$OUT/${NAME}_checkpoints.zip"
fi
ls -la "$OUT"/*.zip
