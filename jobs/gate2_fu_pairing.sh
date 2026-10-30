#!/bin/bash
# ==============================================================================
# SLURM: Gate 2 follow-up, part "pairing" (eval/transfer/gate2_followup.py; shared setup in jobs/lib/gate2_fu.sh).
# Recover Liu's AD-Matched / AD-Shift pairing (every candidate injection task, PG2 vs his released scores,
# Liu's env), rebuild both sets with it, validate with PIGuard and Horizon; judge nf4 and bf16 on the recovered
# AD-Matched. Time: ~50 min estimated (one set build takes ~2.5 min; 30 builds run 4 at a time; 28 PG2 scorings).
# Submit from the repo root: sbatch jobs/gate2_fu_pairing.sh
# ==============================================================================
#SBATCH --job-name=g2fu_pairing
#SBATCH --output=logs/slurm_g2fu_pairing_%j.log
#SBATCH --error=logs/slurm_g2fu_pairing_%j.log
#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

source jobs/lib/gate2_fu.sh
BIB=$PWD/data/raw/benign-instruction-bench
export AD_PY FU W SETS

# set builds, 4 at a time: per set, the timestamp check (same pairing as the seeded set, rebuilt now: rows that
# differ carry wall-clock timestamps) and candidates 0-13 (14 = workspace's injection tasks, the most of any suite)
build() {
  case $1 in admatched) f=agentdojo_pairs.jsonl ;; adshift) f=agentdojo_pairs_shift.jsonl ;; esac
  if [ "$2" = tc ]; then
    [ -f "$W/timecheck/$f" ] || "$AD_PY" $FU gen --set $1 --map_from_rows "$SETS/$f" --out_dir "$W/timecheck"
  else
    "$AD_PY" $FU gen --set $1 --cands --k $2 --out_dir "$W/cands/$1"
  fi
}
export -f build
for s in admatched adshift; do echo "$s tc"; for k in $(seq 0 13); do echo "$s $k"; done; done \
  | xargs -P 4 -L 1 bash -c 'build $0 $1'
for s in admatched adshift; do
  for c in "$W/cands/$s"/c*; do
    "$LIU_PY" $DET --detectors promptguard2-86m --sets $s --sets_dir "$c" --out "$c/scores"
  done
  "$OUR_PY" $FU match --set $s --work "$W"
  [ -f "$W/recovered/$s.done" ] || { "$AD_PY" $FU gen --set $s --map "$W/map_$s.json" --out_dir "$W/recovered" \
    && touch "$W/recovered/$s.done"; }
done
# user_request for the judge, as scripts/build_transfer_sets.sh does (idempotent)
(cd "$W/recovered" && "$AD_PY" "$BIB/scripts/attach_requests.py" agentdojo_pairs.jsonl agentdojo_pairs_shift.jsonl)
"$LIU_PY" $DET --detectors piguard horizon-base promptguard2-86m --sets admatched adshift \
  --sets_dir "$W/recovered" --out "$W/recovered/scores"
"$LIU_PY" $DET --detectors judge-qwen --sets admatched --sets_dir "$W/recovered" --out "$W/judge_nf4_rec"
"$LIU_PY" $DET --detectors judge-qwen --sets admatched --sets_dir "$W/recovered" --out "$W/judge_bf16_rec" \
  --precision bf16 --gpu_mem 40GiB
"$LIU_PY" $DET --detectors judge-qwen --sets adclean --sets_dir "$SETS" --out "$W/judge_bf16_rec" \
  --precision bf16 --gpu_mem 40GiB
report pairing
