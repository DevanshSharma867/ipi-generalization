#!/bin/bash
# ==============================================================================
# SLURM (CPU work; the QoS needs --gres=mps:50 even so): rebuild data/agent_v2/split_seed_0/{train,val}.csv on the
# cluster from the committed builder (src/agent_pipeline.py; the CSVs are gitignored) and verify them against the
# committed data/agent_v2/split_seed_0/manifest.json (files_sha256). Inputs are checked first: the BIPIA text attack
# train file against the manifest's inputs_sha256, the V3 tokenizer against pairs_gen.V3_TOKENIZER_SHA256.
# Builds into scratch; the CSVs are copied into data/agent_v2/split_seed_0/ only if both hashes match, else exit 1
# and nothing in data/ changes. The builder also re-verifies data/agent_v1/calibration_pool.csv (never rewrites it).
# Needs data/raw/BIPIA/benchmark/{text,code}_attack_{train,test}.json (gitignored; copied from the laptop clone).
# Idempotent: exits at once if both CSVs already match. About 20 s of CPU.
# Submit from the repo root (~/nlp_harness): sbatch jobs/agent_v2_rebuild.sh
# ==============================================================================
#SBATCH --job-name=agent_v2
#SBATCH --output=logs/slurm_agent_v2_%j.log
#SBATCH --error=logs/slurm_agent_v2_%j.log
#SBATCH --time=00:15:00
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
SPLIT=${SPLIT:-0}
DEST=data/agent_v2/split_seed_$SPLIT
check() {  # check DIR: 0 if DIR/{train,val}.csv match the committed manifest
  python - "$1" "$DEST/manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
d, m = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())["files_sha256"]
bad = [f for f, h in m.items() if not (d / f).exists() or hashlib.sha256((d / f).read_bytes()).hexdigest() != h]
print(f"{d}: " + ("all match" if not bad else f"MISMATCH {bad}"))
sys.exit(1 if bad else 0)
PY
}
check "$DEST" && { echo "already verified"; exit 0; }

python - "$DEST/manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
want = json.loads(Path(sys.argv[1]).read_text())["inputs_sha256"]
for f, h in want.items():
    got = hashlib.sha256(Path("data/raw/BIPIA/benchmark", f).read_bytes()).hexdigest()
    assert got == h, f"{f}: sha256 {got} != manifest {h}"
print("inputs match", list(want))
PY
TOK=hf_cache/tokenizers_hpc/deberta_v2_ckpt   # data_pipeline_v3.TOKENIZER (repo-relative)
if [ ! -e "$TOK/tokenizer.json" ]; then
  mkdir -p "$(dirname "$TOK")" && ln -sfn "$HF_HOME/tokenizers_hpc/deberta_v2_ckpt" "$TOK"
fi
python -c "import sys; sys.path.insert(0, 'src'); import pairs_gen as pg; pg.v3_tokenizer(); print('tokenizer sha256 ok')"

OUT=$SCRATCH/nlp_harness/agent_v2_rebuild/$(git rev-parse --short HEAD)
python src/agent_pipeline.py --split_seeds "$SPLIT" --output_dir "$OUT"
check "$OUT/split_seed_$SPLIT" || { echo "rebuild does not match the committed manifest; $DEST unchanged"; exit 1; }
cp "$OUT/split_seed_$SPLIT/train.csv" "$OUT/split_seed_$SPLIT/val.csv" "$DEST/"
check "$DEST"
echo "done: $DEST verified"
