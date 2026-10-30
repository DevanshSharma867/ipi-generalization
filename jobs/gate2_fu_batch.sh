#!/bin/bash
# ==============================================================================
# SLURM: Gate 2 follow-up, part "batch" (eval/transfer/gate2_followup.py; shared setup in jobs/lib/gate2_fu.sh).
# Batch size 1 vs 16 for PIGuard, Horizon, PG2 on a 500-row BIPIA subset in Liu's env and ours, and for our
# V2 DeBERTa checkpoints (3 seeds, V2 test) in ours. Time: ~15 min estimated (12 model loads + 3 checkpoints).
# Submit from the repo root: sbatch jobs/gate2_fu_batch.sh
# ==============================================================================
#SBATCH --job-name=g2fu_batch
#SBATCH --output=logs/slurm_g2fu_batch_%j.log
#SBATCH --error=logs/slurm_g2fu_batch_%j.log
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

source jobs/lib/gate2_fu.sh
CKPTS=$HOME/nlp_probe/checkpoints

[ -f "$W/bipia500/indices.json" ] || "$OUR_PY" $FU subset --work "$W"
for b in 1 16; do
  "$LIU_PY" $DET --detectors piguard horizon-base promptguard2-86m --sets bipia --sets_dir "$W/bipia500" \
    --out "$W/bs/liu_bs$b" --bs $b
  "$OUR_PY" $DET --detectors piguard horizon-base promptguard2-86m --sets bipia --sets_dir "$W/bipia500" \
    --out "$W/bs/ours_bs$b" --bs $b
done
"$OUR_PY" $FU deberta_bs --ckpts "$CKPTS"/deberta_s42 "$CKPTS"/deberta_s123 "$CKPTS"/deberta_s999 \
  --data data/processed/test.csv --out "$W/deberta_bs"
report batch
