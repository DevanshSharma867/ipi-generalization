#!/bin/bash
# ==============================================================================
# SLURM: Gate 2 follow-up (docs/deviations.md 2026-10-30; paper_plan A19, A20), eval/transfer/gate2_followup.py.
#   1. Recover Liu's AD-Matched / AD-Shift pairing (every candidate injection task, PG2 vs his released scores),
#      rebuild both sets with it and validate with PIGuard and Horizon (his env).
#   2. Batch size 1 vs 16 on a 500-row BIPIA subset in his env and ours; our V2 DeBERTa checkpoints at bs 1 vs 16.
#   3. Judge: 4-bit repeat on BIPIA, one run at GPU_MEM=6GiB, bf16 vs 4-bit on AD-Matched (seeded and recovered).
# Then writes the run record to results/transfer/gate2_followup/. Resumable: finished score files are skipped.
# Needs Gate 2's work dir (jobs/gate2_liu_repro.sh) for the run-2 scores. Submit from the repo root:
#   sbatch jobs/gate2_followup.sh
# ==============================================================================
#SBATCH --job-name=gate2_fu
#SBATCH --output=logs/slurm_gate2_fu_%j.log
#SBATCH --error=logs/slurm_gate2_fu_%j.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

source jobs/lib/job_env.sh
set -e
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
AD_PY=$PWD/.venv-agentdojo/bin/python
LIU_PY=$PWD/.venv-liu/bin/python
OUR_PY=$PWD/.venv/bin/python
G2=$SCRATCH/nlp_harness/gate2
W=$SCRATCH/nlp_harness/gate2_followup
SETS=$PWD/data/transfer/sets
BIB=$PWD/data/raw/benign-instruction-bench
FU=eval/transfer/gate2_followup.py
DET=eval/transfer/detectors.py
CKPTS=$HOME/nlp_probe/checkpoints
mkdir -p "$W"
export TORCHDYNAMO_DISABLE=1 PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # as in Gate 2

# 1. pairing recovery
for s in admatched adshift; do
  "$AD_PY" $FU gen --set $s --cands --out_dir "$W/cands/$s"
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

# 2. batch size and environment
[ -f "$W/bipia500/indices.json" ] || "$OUR_PY" $FU subset --work "$W"
for b in 1 16; do
  "$LIU_PY" $DET --detectors piguard horizon-base promptguard2-86m --sets bipia --sets_dir "$W/bipia500" \
    --out "$W/bs/liu_bs$b" --bs $b
  "$OUR_PY" $DET --detectors piguard horizon-base promptguard2-86m --sets bipia --sets_dir "$W/bipia500" \
    --out "$W/bs/ours_bs$b" --bs $b
done
"$OUR_PY" $FU deberta_bs --ckpts "$CKPTS"/deberta_s42 "$CKPTS"/deberta_s123 "$CKPTS"/deberta_s999 \
  --data data/processed/test.csv --out "$W/deberta_bs"

# 3. judge (Liu's env, his Yes/No rule)
"$LIU_PY" $DET --detectors judge-qwen --sets admatched --sets_dir "$W/recovered" --out "$W/judge_nf4_rec"
"$LIU_PY" $DET --detectors judge-qwen --sets admatched adclean --sets_dir "$SETS" --out "$W/judge_bf16_seeded" \
  --precision bf16 --gpu_mem 40GiB
"$LIU_PY" $DET --detectors judge-qwen --sets admatched --sets_dir "$W/recovered" --out "$W/judge_bf16_rec" \
  --precision bf16 --gpu_mem 40GiB
"$LIU_PY" $DET --detectors judge-qwen --sets bipia --sets_dir "$SETS" --out "$W/judge_repeat" --gpu_mem 18GiB
"$LIU_PY" $DET --detectors judge-qwen --sets bipia --sets_dir "$SETS" --out "$W/judge_6gib" --gpu_mem 6GiB

"$OUR_PY" $FU report --work "$W" --gate2 "$G2" --liu_py "$LIU_PY" --our_py "$OUR_PY"
