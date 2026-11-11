#!/bin/bash
# Kaggle session 2 (docs/deviations.md 2026-11-11; config pairs_gen_v6.yaml = v5 with gen-train-mistral in fp16):
#   1. val pairs (A32): 2 shards, as jobs/pairs_gen_val.sh       3. TRAIN pairs: 8 shards, as jobs/pairs_gen_train.sh
#   2. self-check of val (spec §8), as jobs/pairs_selfcheck.sh    4. self-check of TRAIN   5. export (jobs/pairs_export.sh)
# The model is spread over both T4s (fp16). Every step is resumable (quads and decisions already written are
# skipped). A time budget (BUDGET_S, default 10 h 40 min) stops new work in time to package; the state is zipped
# after every step. To continue in a new session, attach the previous version's output (session2_results.zip) as
# input: it is restored first. Needs the HF_TOKEN secret (Mistral is gated). Run from the repo root after setup.
# Outputs are zipped on exit, also after a failure; exits non-zero unless every step finished.
set -euo pipefail
: "${ASSETS:?set ASSETS}"
export PAIRS_V3_TOKENIZER=$ASSETS/tokenizers_hpc/deberta_v2_ckpt HF_HOME=${HF_HOME:-/tmp/hf_home}
trap 'bash kaggle/package.sh session2 || echo "packaging failed"' EXIT
CFG=configs/pairs_gen_v6.yaml
START=$(date +%s); BUDGET=${BUDGET_S:-38400}
PREV=$(find "${KAGGLE_ROOT:-/kaggle}/input" -name session2_results.zip 2>/dev/null | head -1)
[ -n "$PREV" ] && { echo "restoring $PREV"; unzip -qo "$PREV"; }
CODE_SHA=$(cat $CFG src/pairs_gen.py scripts/gen_pairs.py | sha256sum | cut -c1-12)
VAL=data/pairs/val_gen/v6_s0_$CODE_SHA; TRAIN=data/pairs/train_gen/v6_s0_$CODE_SHA
mkdir -p "$VAL" "$TRAIN" logs
step() {  # step NAME CMD...: run with the time left, log, then package
  local name=$1; shift
  local left=$(( BUDGET - $(date +%s) + START ))
  [ $left -gt 900 ] || { echo "$(date -Is) budget exhausted before $name"; return 99; }
  echo "$(date -Is) $name start (${left}s left)"
  local rc=0
  timeout $left "$@" >> "logs/kaggle_s2_$name.log" 2>&1 || rc=$?
  [ $rc = 0 ] || tail -20 "logs/kaggle_s2_$name.log"
  echo "$(date -Is) $name exit $rc"
  bash kaggle/package.sh session2 > /dev/null
  return $rc
}
ok=1
for k in 0 1; do step val_gen_$k bash kaggle/gen_shard.sh val data/pairs/docs/val_s0.jsonl "$VAL" 2 $k || ok=0; done
[ $ok = 1 ] && { step val_selfcheck python scripts/gen_pairs.py selfcheck --config $CFG --in_dir "$VAL" || ok=0; }
for k in 0 1 2 3 4 5 6 7; do step train_gen_$k bash kaggle/gen_shard.sh train data/pairs/docs/train_s0.jsonl "$TRAIN" 8 $k || ok=0; done
[ $ok = 1 ] && { step train_selfcheck python scripts/gen_pairs.py selfcheck --config $CFG --in_dir "$TRAIN" || ok=0; }
[ $ok = 1 ] && { step export bash -c "python scripts/export_pairs.py --in_root data/pairs/train_gen \
  --out data/pairs/ict_s0/train_pairs.jsonl && python scripts/export_pairs.py --in_root data/pairs/val_gen \
  --out data/pairs/ict_s0/val_pairs.jsonl" || ok=0; }
if [ $ok = 1 ]; then echo "SESSION 2 COMPLETE"; else echo "SESSION 2 INCOMPLETE: rerun with this output attached"; exit 1; fi
