#!/bin/bash
# ==============================================================================
# SLURM (CPU work; the QoS needs --gres=mps:50 even so): A22 step 2. Picks the grid winner (`ict.py select`: highest
# best-epoch selection score over the 12 finished, clean ict_grid runs; ties to the first in grid order), writes
# configs/ict_hparams.json, commits it in this checkout and pushes it to origin main. Every later ICT job reads the
# committed file and refuses to run without it (jobs/ict_train.sh).
# Push: rebases on origin main and retries with backoff for up to ~45 min; if the push still fails the job exits 1,
# so afterok-dependent ICT jobs never start on an unpushed choice. Needs push access from the cluster.
# Idempotent: if the file is already committed it only (re)tries the push; `ict.py select` never overwrites a choice.
# Submit from the repo root (~/nlp_harness): sbatch --dependency=afterok:<grid array> jobs/ict_select.sh
# ==============================================================================
#SBATCH --job-name=ict_select
#SBATCH --output=logs/slurm_ict_select_%j.log
#SBATCH --error=logs/slurm_ict_select_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50
#SBATCH --requeue
#SBATCH --open-mode=append

source jobs/lib/job_env.sh
F=configs/ict_hparams.json
if git ls-files --error-unmatch "$F" >/dev/null 2>&1; then
  echo "$F already committed at $(git log -1 --format=%h -- "$F")"
else
  python src/models/ict.py select
  git add "$F"
  git commit -q -m "ICT hyperparameters from the A22 grid (full ICT, training seed 0, split seed 0; selection on V3 val and TRAIN-side val pairs only), written by jobs/ict_select.sh"
  echo "committed $F: $(cat "$F" | tr -d '\n' | cut -c1-200)"
fi
for i in 1 2 3 4 5 6 7 8; do
  if git pull --rebase -q origin main && git push -q origin HEAD:main; then
    git fetch -q origin && git merge-base --is-ancestor "$(git log -1 --format=%H -- "$F")" origin/main \
      && { echo "pushed: $(git log -1 --format='%h %s' -- "$F")"; exit 0; }
  fi
  echo "push attempt $i failed; retrying in $((i * 60)) s"
  sleep $((i * 60))
done
echo "could not push $F; downstream ICT jobs will not start (afterok). Push it, then resubmit them."
exit 1
