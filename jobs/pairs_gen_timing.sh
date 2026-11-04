#!/bin/bash
# ==============================================================================
# SLURM: pair-generation timing job (paper_plan A24; STATUS "Intent pairs", Next 4; docs/pairs_spec.md v2 §8).
# 20 V3 val docs of split seed 0 (data/pairs/docs/timing_s0.jsonl: 7 email, 7 table, 6 code), one quad each, with
#   1. gen-eval-gemma    (Gemma 3 27B NF4, EVAL prompt), batch 1 (the EVAL setting: per-quad seed)
#   2. gen-train-mistral (Mistral 7B bf16, TRAIN prompt), batch 1 and batch 8
# Half a card: torch.cuda.set_per_process_memory_fraction(0.5) in scripts/gen_pairs.py, and the MPS VRAM cap from
# jobs/lib/job_env.sh (which also runs the GPU guard). Records load time, prefill/decode tok/s, s/quad, peak VRAM,
# JSON-parse and pairs.validate rates and the GPU-hour extrapolation (~210 EVAL, ~3,141 TRAIN quads) as a runlog
# run results/pairs/gen_timing/<run_id>/. Generated text stays on scratch and is deleted after the summary; it is
# never used as EVAL or TRAIN pairs.
# Resumable: shards under $W skip finished quads; a resubmit after the summary exits at once ($W/DONE).
# Submit from the repo root (~/nlp_harness): sbatch jobs/pairs_gen_timing.sh
# ==============================================================================
#SBATCH --job-name=pairs_timing
#SBATCH --output=logs/slurm_pairs_timing_%j.log
#SBATCH --error=logs/slurm_pairs_timing_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

source jobs/lib/job_env.sh
DOCS=data/pairs/docs/timing_s0.jsonl
CFG_SHA=$(sha256sum configs/pairs_gen_v2.yaml | cut -c1-12)
W=$SCRATCH/nlp_harness/pairs_gen_timing/$CFG_SHA
mkdir -p "$W"
[ -f "$W/DONE" ] && { echo "timing already summarised: $(cat "$W/DONE")"; exit 0; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

gen() {  # gen GENERATOR BATCH_SIZE
  python scripts/gen_pairs.py generate --mode timing --generator "$1" --docs "$DOCS" --out_dir "$W/$1_bs$2" \
    --batch_size "$2"   # length rules: V3 tokenizer at $HF_HOME/tokenizers_hpc/deberta_v2_ckpt (sha256-checked)
}
gen gen-eval-gemma 1
gen gen-train-mistral 1
gen gen-train-mistral 8

python scripts/gen_pairs.py timing --run "$W/gen-eval-gemma_bs1" --run "$W/gen-train-mistral_bs1" \
  --run "$W/gen-train-mistral_bs8" | tee "$W/summary.log"
RUN=$(tail -1 "$W/summary.log" | sed 's/^wrote //')
rm -f "$W"/*/shard_*.jsonl   # generated text discarded (A24); meta files kept
echo "$RUN" > "$W/DONE"
echo "done: $RUN"
