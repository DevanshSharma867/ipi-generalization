#!/bin/bash
# ==============================================================================
# SLURM: TRAIN-side self-consistency filter (docs/pairs_spec.md v3 §8; paper_plan A32; config pairs_gen_v5.yaml,
# prompt configs/prompts/pairs_selfcheck_v1.txt). gen-train-mistral labels each member of each valid pair blind
# (prefill only: A vs B after the chat prompt); a pair is kept only if both members get their intended label.
#   IN_ROOT    data/pairs/val_gen or data/pairs/train_gen: its single v*_s0_* dir must be a finished generate dir
#              (all quads of its docs file present). Writes <dir>/selfcheck.jsonl (labels, no text) + .meta.json.
#   DRYRUN     optional: a timing run's failures.jsonl (val docs). Runs first; pairs rebuilt from failed attempts,
#              labels and counts to results/pairs/selfcheck_dryrun/<run_id>/. Never sets a threshold (A32).
# Prefill only: ~1.5k prompts for val, ~7k for train; minutes. Resumable: decisions already written are skipped.
# Submit from the repo root (~/nlp_harness), e.g.
#   DRYRUN=results/pairs/gen_timing/20261109-094809_seed0/failures.jsonl IN_ROOT=data/pairs/val_gen \
#     sbatch --dependency=afterok:<val array job> jobs/pairs_selfcheck.sh
#   IN_ROOT=data/pairs/train_gen sbatch --dependency=afterok:443655 jobs/pairs_selfcheck.sh
# ==============================================================================
#SBATCH --job-name=pairs_selfcheck
#SBATCH --output=logs/slurm_pairs_selfcheck_%j.log
#SBATCH --error=logs/slurm_pairs_selfcheck_%j.log
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

source jobs/lib/job_env.sh
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
if [ -n "${DRYRUN:-}" ]; then
  if ls results/pairs/selfcheck_dryrun/*/finished.json >/dev/null 2>&1; then
    echo "dry run already finished: $(ls -d results/pairs/selfcheck_dryrun/*/)"
  else
    python scripts/gen_pairs.py selfcheck --failures "$DRYRUN" --docs data/pairs/docs/timing_s0.jsonl
  fi
fi
if [ -n "${IN_ROOT:-}" ]; then
  mapfile -t DIRS < <(ls -d "$IN_ROOT"/v*_s0_*/ 2>/dev/null)
  [ "${#DIRS[@]}" -eq 1 ] || { echo "expected one generate dir under $IN_ROOT, found ${#DIRS[@]}: ${DIRS[*]:-}"; exit 1; }
  python scripts/gen_pairs.py selfcheck --in_dir "${DIRS[0]%/}"
fi
