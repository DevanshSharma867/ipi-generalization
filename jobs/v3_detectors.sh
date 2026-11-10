#!/bin/bash
# ==============================================================================
# SLURM: published detectors on V3 (paper_plan A30), wave 1: the 16 classifiers in Liu's env and Wolf Defender in
# ours, as for F021 (eval/transfer/detectors.py). Input: data/transfer/v3_sets/v3_all.jsonl, every distinct V3 input
# of split seeds 0-4 (test, val, val_id, raw_clean, ambiguous, long; ~9.0k rows), built here by
# scripts/v3_detector_sets.py from the committed data/v3 (deterministic; manifest records its sha256).
# Finished score files are skipped (resumable). Run records: results/v3/v3_detscores_{liuenv,wolf}/.
# Time: F021's encoder pass (~7.8k rows) took 12-22 min (jobs 425944, 434238); Wolf 2 min.
# Submit from the repo root: sbatch jobs/v3_detectors.sh   (judges: jobs/v3_judges.sh)
# ==============================================================================
#SBATCH --job-name=v3_dets
#SBATCH --output=logs/slurm_v3_dets_%j.log
#SBATCH --error=logs/slurm_v3_dets_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50
#SBATCH --requeue
#SBATCH --open-mode=append

source jobs/lib/v3_detsets.sh
D="protectai-v1 protectai-v2 protectai-small-v2 piguard jasperls testsavant-large horizon-base sheltron sheltron-taskaware \
sheltron-generic prismor-1.5b promptguard2-86m promptguard2-22m promptguard1-86m qualifire-sentinel rogue-sentinel-v2"
"$LIU_PY" $DET --detectors $D --sets v3all --sets_dir "$SETS" --out "$W/liuenv"
record v3_detscores_liuenv "$W/liuenv" "$LIU_PY" $D
"$OUR_PY" $DET --detectors wolf-defender --sets v3all --sets_dir "$SETS" --out "$W/wolf"
record v3_detscores_wolf "$W/wolf" "$OUR_PY" wolf-defender
