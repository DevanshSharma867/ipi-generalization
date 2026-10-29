#!/bin/bash
# ==============================================================================
# Rebuild the Liu (2026) transfer sets from public sources with the upstream scripts, unmodified
# (paper_plan §3 "Building the agent sets", A16-A17). Nothing is redistributed: outputs go to the gitignored
# data/transfer/sets/, with data/transfer/sets/manifest.json (pins, environments, file hashes).
#
#   AD-Clean    agentdojo_benign_outputs.jsonl     AgentDojo v1.2 ground-truth replay, no injections
#   AD-Matched  agentdojo_pairs.jsonl              important_instructions, ignore_previous, direct
#   AD-Shift    agentdojo_pairs_shift.jsonl        tool_knowledge, injecagent, system_message, two variants
#   TB-Clean    taubench_clean.jsonl               tau-bench retail + airline ground-truth replay
#   TB-Matched  taubench_eval.jsonl                TB-Clean + InjecAgent attacks in retail product names
#   BIPIA       bipia_pairs.jsonl                  all BIPIA contexts + K=4 attacks (train and test halves)
#   general     corpus.jsonl                       READMEs, AESLC emails, FineWeb-Edu (HF streaming)
#
# Pins: upstream benign-instruction-bench, BIPIA, InjecAgent, tau-bench, PIGuard (clones under
# data/transfer/agentsec/, exposed to the upstream scripts as ~/agentsec via a HOME override); agentdojo==0.1.34.
# The general corpus streams HF datasets without a revision (upstream does not pin one); the manifest records
# the dataset revisions current at build time.
#
# CPU only; needs internet and uv. On the cluster run it through jobs/build_transfer_sets.sh (sbatch), which puts
# the venvs and caches on scratch (VENV_AD, VENV_LIU, HF_HOME override the repo-local defaults).
# Resumable: a step counts as done only once its .done marker exists (delete the marker to redo it); a venv only
# once its .build-complete marker exists. Every finished step is logged to build_steps.tsv (host, job id, time),
# which the manifest records.
#   bash scripts/build_transfer_sets.sh
# Then: python eval/transfer/gate1.py   (row counts and label sequences vs Liu's released score files)
# ==============================================================================
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T="$ROOT/data/transfer"
AGS="$T/agentsec"
OUT="$T/sets"
BIB="$ROOT/data/raw/benign-instruction-bench"
VENV_AD=${VENV_AD:-$ROOT/.venv-agentdojo}
VENV_LIU=${VENV_LIU:-$ROOT/.venv-liu}

BIB_COMMIT=5d0ae7c438a7b01be2ac0a2e8bc943e45c8f851f
AGENTDOJO=agentdojo==0.1.34
PINS="BIPIA https://github.com/microsoft/BIPIA a004b69ec0dd
InjecAgent https://github.com/uiuc-kang-lab/InjecAgent f19c9f2c79a4
tau-bench https://github.com/sierra-research/tau-bench 59a200c6d575
PIGuard https://github.com/leolee99/PIGuard 1b5751e88bf7"

clone_at() {  # clone_at <dir> <url> <commit-prefix>
  local dir=$1 url=$2 commit=$3
  [ -d "$dir/.git" ] || git clone -q "$url" "$dir"
  git -C "$dir" checkout -q "$commit"
  [[ "$(git -C "$dir" rev-parse HEAD)" == "$commit"* ]] || { echo "$dir is not at $commit"; exit 1; }
}

mkdir -p "$AGS" "$OUT" "$T/home"
clone_at "$BIB" https://github.com/lzwhehe/benign-instruction-bench "$BIB_COMMIT"
while read -r name url commit; do clone_at "$AGS/$name" "$url" "$commit"; done <<<"$PINS"
ln -sfn "$AGS" "$T/home/agentsec"

# Environments: AgentDojo replay (its own pins) and Liu's main env (his requirements.txt, Python 3.12)
log_step() { printf '%s\t%s\t%s\t%s\n' "$1" "$(hostname)" "${SLURM_JOB_ID:-none}" "$(date -Iseconds)" >> "$OUT/build_steps.tsv"; }
make_venv() {  # make_venv <dir> <uv pip install args...>; a half-built venv is completed, not skipped
  local v=$1; shift
  [ -f "$v/.build-complete" ] && return
  echo ">>> venv $v"
  uv venv -q --allow-existing -p 3.12 "$v"
  uv pip install -q -p "$v" "$@"
  touch "$v/.build-complete"; log_step "venv $(basename "$v")"
}
make_venv "$VENV_AD" "$AGENTDOJO"
make_venv "$VENV_LIU" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 -r "$BIB/requirements.txt"

# Upstream scripts run from $OUT (they write to the working directory) with ~ -> $T/home
export HOME="$T/home" PYTHONHASHSEED=0 HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
cd "$OUT"
step() {  # step <output> <python> <script> [args...]
  local out=$1 py=$2 script=$3; shift 3
  if [ -f "$out.done" ]; then echo ">>> done: $out"; return; fi
  echo ">>> $script $*"
  "$py" "$BIB/scripts/$script" "$@"
  touch "$out.done"; log_step "$script -> $out"
}
step corpus.jsonl "$VENV_LIU/bin/python" build_corpus.py --per_source 2500 --out corpus.jsonl
step agentdojo_benign_outputs.jsonl "$VENV_AD/bin/python" collect_agentdojo_outputs.py
step agentdojo_pairs.jsonl "$VENV_AD/bin/python" collect_agentdojo_pairs.py \
  important_instructions,ignore_previous,direct agentdojo_pairs.jsonl
step agentdojo_pairs_shift.jsonl "$VENV_AD/bin/python" collect_agentdojo_pairs.py \
  tool_knowledge,injecagent,system_message,important_instructions_no_names,important_instructions_wrong_model_name \
  agentdojo_pairs_shift.jsonl
# attach_requests.py (re)writes `user_request` in place; idempotent, so it always runs
"$VENV_AD/bin/python" "$BIB/scripts/attach_requests.py" agentdojo_pairs.jsonl agentdojo_pairs_shift.jsonl \
  agentdojo_benign_outputs.jsonl
log_step "attach_requests.py"
step bipia_pairs.jsonl "$VENV_LIU/bin/python" build_bipia.py
step taubench_clean.jsonl "$VENV_LIU/bin/python" collect_taubench.py
step taubench_eval.jsonl "$VENV_LIU/bin/python" build_taubench_eval.py

cd "$ROOT"
"$VENV_LIU/bin/python" eval/transfer/manifest.py --sets "$OUT" --agentsec "$AGS" --upstream "$BIB" \
  --venv "agentdojo=$VENV_AD" --venv "liu=$VENV_LIU" --agentdojo "$AGENTDOJO"
