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

## Next
1. **b. Detector wrappers:** registry-driven, using Liu's documented loaders (cls / segmented / Sheltron / Prismor /
   4-bit judges).
2. **c. Metrics:** TPR@1%FPR on the same benchmark's benign outputs (A16), plus `task_block_rate`.
3. **d. Gate 2, Liu reproduction:** in his pinned env (`.venv-liu`) and ours. Scores must match his released files;
   any env difference is reported (A17).
4. **e. Results layout:** `results/transfer/<detector>/<set>/<run_id>`, which needs a nested-exp change in
   `src/runlog.py`.

## Open decisions / blockers
- **Access request (you):** `protectai/deberta-v3-small-prompt-injection-v2` is not requested yet (auto-approved).
- **Findings entry for Gate 1:** set sizes may appear in the paper, so an F-entry is needed. It waits on another
  session's uncommitted `make_tables.py` / `make_findings.py` edits.
- **Revision pins:** Liu records no revisions except Horizon. wolf-defender (2026-10-24) and prismor-1.5b
  (2026-10-21) changed just before his last commit, so they are the first suspects if Gate 2 mismatches; re-pin
  with a reason.
- **VRAM cap:** MPS does not enforce `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` (smoke test, job 422476). Half-card VRAM
  must be capped in code, e.g. `torch.cuda.set_per_process_memory_fraction(0.5)`.
- **Frontier comparator (c):** `OPENAI_API_KEY` is not set, so the one test call (c′) has not been made; score by
  2026-12-31 (A11).
- **Cluster notes:** the QoS needs `--gres=mps:50` even for CPU jobs; the template file
  `jobs/templates/mock_job_csisnlp18.sh` was never found, so the header is derived from the existing jobs.
