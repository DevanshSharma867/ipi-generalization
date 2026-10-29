#!/bin/bash
# ==============================================================================
# SLURM: re-score the V2 probe sets from a clean commit, and score V2 val for threshold calibration.
# ==============================================================================
# Why: the 2026-10-19 probe-scoring runs (results/v2_probes/score_*/20261019-*) were made on a dirty tree,
# so their git_commit does not identify the code. Findings built on them are "linked-dirty" until this runs.
#   1. Probe sets P1-P4 + V2 test: deberta_s42/s123/s999, tfidf_c1, protectai, piguard
#      (judge: P1-P4 only; its V2-test run 20261021-123213 is already clean).
#   2. V2 val (calibration set of the "calibrated" threshold protocol): protectai, piguard, judge
#      (DeBERTa val scores exist in results/v2_std/v2_val/; TF-IDF val was scored on CPU).
#   The judge uses the V2 Yes/No rule (--judge_rule two_token) so these runs complete the existing V2 judge runs
#   unchanged. The variant-rule re-score (paper_plan A12) is a separate job: jobs/rescore_v2_judge_variant.sh.
# Same checkpoints and options as the original runs: checkpoints/deberta_s<seed> (hpc_original; the job stops
# if one is missing rather than retrain), HF hub detectors, the frozen judge prompt, --doc_frac 1.0.
# Resumable: a step is skipped when its exp already has a clean run (git_dirty false) that scored all its sets.
# Writes via runlog to results/v2_probes/score_<name>/<run_id>/. probe_report.py and truncation_control.py
# are re-run afterwards from these runs.
#
# Prerequisites (login node): clean tree at the commit that added this file (`git status` empty outside results/;
# untracked files count, so list local-only files such as CLAUDE.md in .git/info/exclude); hf_cache holds
# protectai/deberta-v3-base-prompt-injection-v2, leolee99/PIGuard, Qwen/Qwen2.5-7B-Instruct.
# Submit from the repo root:  sbatch jobs/rescore_v2_probes.sh
# ==============================================================================

#SBATCH --job-name=ipi_rescore_v2
#SBATCH --output=logs/slurm_rescore_v2_%j.log
#SBATCH --error=logs/slurm_rescore_v2_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# BITS Hyderabad Sharanga student QoS: cpu=4, mem=80G, mps=50 (half a card)
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

MODULES=""                               # e.g. "cuda/12.8"; empty = none
VENV_PATH=".venv"
SEEDS="42 123 999"
PROBE_SETS="p1_neutral_insertion p2_benign_moved p3_attack_moved p4_whitespace_only"

set -euo pipefail
mkdir -p logs
for m in $MODULES; do module load "$m"; done
source "$VENV_PATH/bin/activate"
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date); commit $(git rev-parse HEAD)"

# Fail before any GPU work if runlog would refuse (uncommitted changes outside results/)
python -c "import sys; sys.path.insert(0, 'src'); import runlog; sys.exit(runlog.code_dirty())" \
    || { echo "Dirty tree; commit, stash or exclude these first:"; git status --porcelain | grep -v ' results/'; exit 1; }

clean_done() {  # clean_done <name> <sets...>: a clean run of score_<name> already scored every set
    python - "$@" <<'PY'
import glob, json, os, sys
name, need = sys.argv[1], set(sys.argv[2:])
for cfg in glob.glob(f"results/v2_probes/score_{name}/*/config.json"):
    c = json.load(open(cfg))
    run_dir = os.path.dirname(cfg)
    if not c["git_dirty"] and need <= set(c["args"]["sets"]) and os.path.exists(os.path.join(run_dir, "predictions.csv")):
        sys.exit(0)
sys.exit(1)
PY
}

score() {  # score <name> "<sets>" <args...>
    local name=$1 sets=$2; shift 2
    if clean_done "$name" $sets; then echo ">>> Skipping (clean run exists): $name"; return; fi
    echo ">>> Scoring $name on: $sets"
    python scripts/score_probes.py --name "$name" --sets $sets "$@"
}

# 1. Probe sets + V2 test
for seed in $SEEDS; do
    ckpt="checkpoints/deberta_s${seed}"
    [ -f "$ckpt/config.json" ] || { echo "Missing $ckpt (original V2 checkpoint); not retraining"; exit 1; }
    score "deberta_s${seed}" "$PROBE_SETS v2_test" --model deberta --checkpoint "$ckpt" \
        --checkpoint_origin hpc_original --seed "$seed"
done
score tfidf_c1 "$PROBE_SETS v2_test" --model tfidf --checkpoint_origin frozen_v2
score protectai "$PROBE_SETS v2_test" --model hf --checkpoint protectai/deberta-v3-base-prompt-injection-v2 \
    --checkpoint_origin hf_hub
score piguard "$PROBE_SETS v2_test" --model hf --checkpoint leolee99/PIGuard --checkpoint_origin hf_hub
score judge_qwen2.5_7b "$PROBE_SETS" --model judge --checkpoint Qwen/Qwen2.5-7B-Instruct --checkpoint_origin hf_hub \
    --doc_frac 1.0 --judge_rule two_token

# 2. V2 val calibration scores
score protectai_v2val v2_val --model hf --checkpoint protectai/deberta-v3-base-prompt-injection-v2 \
    --checkpoint_origin hf_hub
score piguard_v2val v2_val --model hf --checkpoint leolee99/PIGuard --checkpoint_origin hf_hub
score judge_qwen2.5_7b_v2val v2_val --model judge --checkpoint Qwen/Qwen2.5-7B-Instruct --checkpoint_origin hf_hub \
    --judge_rule two_token

echo "Done at $(date)"
