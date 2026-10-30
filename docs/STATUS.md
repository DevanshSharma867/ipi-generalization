# Status: transfer harness (updated 2026-10-30)

Plan: paper_plan §3, A16–A17 (Liu 2026 sets as headline transfer; reproduction in Liu's pinned env and ours).

## Done
- **Step a, set build: Gate 1 PASS.** Run `results/transfer/set_build_gate1/20261030-124112_seed0` (commit `6c94421`),
  data version `transfer-bib5d0ae7c-agentdojo0.1.34-tb59a200c-filescf00fcba`. All six sets match Liu's released score
  files (`5d0ae7c`) row for row and label for label, and the paper's set sizes. The manifest records envs, HF dataset
  revisions, and where each step ran.
- **Build path:** `jobs/build_transfer_sets.sh` (sbatch, resumable) → `scripts/build_transfer_sets.sh`.
  Upstream scripts run unmodified; tau-bench steps run in the AgentDojo env (Liu's lockfile has no pydantic), so his
  env stays identical to his `requirements.txt`.
- **Model registry and prefetch:** `configs/detectors.yaml` lists 19 entries with pinned revisions; InjecGuard is an
  alias of PIGuard (Liu §4.1). `scripts/prefetch_models.sh` (login node) downloads them and reports access to
  `results/transfer/prefetch/`. Current state: 17/18 granted and complete.
- **Cluster storage** (`jobs/lib/scratch.sh`, sourced by every job):
  - On scratch: venvs, HF cache, uv/pip caches, the uv binary and rebuilt sets. Scratch purges files after 30
    days; each job re-touches it.
  - In home: results, checkpoints, the HF token. Home is at 17.6/40 G; jobs fail early under 2 G free.
  - Laptop: `scripts/cluster.sh run|watch` retries ssh with backoff.

- **Step b, detector wrappers:** `eval/transfer/detectors.py` (port of Liu's loaders, pinned revisions),
  `eval/transfer/gate2.py`, `jobs/gate2_liu_repro.sh`. Jobs run with `TORCHDYNAMO_DISABLE=1` (ModernBERT compile
  needs nvcc, absent on compute nodes). bitsandbytes 0.50.2 added to our env.
- **Step d, Gate 2 (A19): FAIL.** Run `results/transfer/liu_repro_gate2/20261030-160800_seed0` (job 423625).
  Port check PASS (run 2 = run 1). Run 1 (Liu's code and env) matches his files to ~1e-5 on BIPIA and TB, but
  AD-Matched injected rows differ (273/432 for PIGuard): the set text differs, not the models. Run 3 (our env,
  transformers 5) shifts scores heavily: PG2 tokenizes differently; PIGuard and Horizon tokenize identically but
  still shift on BIPIA. Judge rows differ up to 0.77 vs Liu even in run 1 (bound 0.02); metrics mostly within tolerance.

## Next
1. **Gate 2 follow-up:** identify the AD-Matched text difference (injection-task assignment via `hash()`?) and the
   transformers-5 shift; decide whether scoring uses Liu's env.
2. **c. Metrics:** TPR@1%FPR on the same benchmark's benign outputs (A16), plus `task_block_rate`.
3. **e. Results layout:** `results/transfer/<detector>/<set>/<run_id>`, which needs a nested-exp change in
   `src/runlog.py`.

## Open decisions / blockers
- **Access request:** `protectai/deberta-v3-small-prompt-injection-v2` requested 2026-10-30; re-run prefetch to confirm.
- **Cluster backup:** `~/run_backup_20261030/` holds byte-identical copies of 12 committed run files moved aside
  for a `git pull`; safe to delete once you confirm.
- **Revision pins:** Liu pins only Horizon. wolf-defender and prismor-1.5b changed days before his last commit:
  first suspects if Gate 2 mismatches (re-pin with a reason).
- **VRAM cap:** MPS does not enforce `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` (smoke test, job 422476). Half-card VRAM
  must be capped in code, e.g. `torch.cuda.set_per_process_memory_fraction(0.5)`.
- **Frontier comparator (c):** `OPENAI_API_KEY` not set, so the c′ test call is not made; score by 2026-12-31 (A11).
- **Cluster notes:** the QoS needs `--gres=mps:50` even for CPU jobs; the template file
  `jobs/templates/mock_job_csisnlp18.sh` was never found, so the header is derived from the existing jobs.
