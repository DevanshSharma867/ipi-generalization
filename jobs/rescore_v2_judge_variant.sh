#!/bin/bash
# ==============================================================================
# SLURM: re-score the V2 Qwen judge under the variant-summing Yes/No rule (paper_plan A12).
# ==============================================================================
# Why: every judge now scores P(Yes)/(P(Yes)+P(No)) with all case/whitespace variants summed (src/judging.py).
# The V2 judge numbers (two-token rule) stay as they are; these are NEW runs, recorded as a new findings entry:
#   score_judge_qwen2.5_7b_vsum        P1-P4 probe sets + V2 test
#   score_judge_qwen2.5_7b_vsum_v2val  V2 val (calibration set of the "calibrated" threshold protocol)
# Same prompt (score_detectors.JUDGE_SYSTEM), model and --doc_frac 1.0 as the V2 judge runs; only the rule differs.
# Independent of jobs/rescore_v2_probes.sh (which keeps the two-token rule); either can run first.
# Resumable: a step is skipped when its exp already has a clean run (git_dirty false) that scored all its sets.
# Writes via runlog to results/v2_probes/score_<name>/<run_id>/.
#
# Prerequisites (login node): clean tree (`git status` empty outside results/); hf_cache holds
# Qwen/Qwen2.5-7B-Instruct. Submit from the repo root:  sbatch jobs/rescore_v2_judge_variant.sh
# ==============================================================================

#SBATCH --job-name=ipi_judge_vsum
#SBATCH --output=logs/slurm_judge_vsum_%j.log
#SBATCH --error=logs/slurm_judge_vsum_%j.err
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

score judge_qwen2.5_7b_vsum "$PROBE_SETS v2_test" --model judge --checkpoint Qwen/Qwen2.5-7B-Instruct \
    --checkpoint_origin hf_hub --doc_frac 1.0 --judge_rule variants
score judge_qwen2.5_7b_vsum_v2val v2_val --model judge --checkpoint Qwen/Qwen2.5-7B-Instruct \
    --checkpoint_origin hf_hub --judge_rule variants

echo "Done at $(date)"
