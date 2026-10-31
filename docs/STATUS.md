# Status (updated 2026-10-31)

# Transfer harness

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
- **Frontier comparator (c):** withdrawn 2026-10-31 (A21, budget). No API key or scoring is needed.
- **Cluster notes:** the QoS needs `--gres=mps:50` even for CPU jobs; the template file
  `jobs/templates/mock_job_csisnlp18.sh` was never found, so the header is derived from the existing jobs.

# Intent pairs (C3)

Plan: `docs/pairs_spec.md` v1; paper_plan A18 (construction, split seed 0 only, strict agreement), A21 (comparator (c)
withdrawn), A24 (EVAL generator), A25 (annotators). Nothing has been generated, annotated or scored.

## Done
- Spec, annotation guide, `src/pairs.py` (validation, overlap class, complete quads), annotation page
  (`scripts/make_annotation_sheet.py`), agreement script (`scripts/agreement.py`).
- 20 LLM-drafted pilot pairs (10 quads) on split-seed-0 test docs: `data/pairs/pilot_v1.jsonl`. Never EVAL.
- Generators fixed and pinned in `configs/detectors.yaml` (role `generator`); access granted on the cluster token
  (2026-10-31):
  - TRAIN `gen-train-mistral`: `mistralai/Mistral-7B-Instruct-v0.3` @ `c170c70` (A18).
  - EVAL `gen-eval-gemma`: `google/gemma-3-27b-it` @ `005ad34`, 4-bit NF4 (A24).

## Next
1. **Author labels the pilot** (blocks generation): `open annotation/pilot_v1/devansh.html`, export
   `annotation/pilot_v1/devansh.csv`. Measures time per pair and tests the rules; revise E3/E4 after it if needed
   (A18 Q4).
2. **Prefetch the generators** as an sbatch job (Gemma 55 GB, Mistral 29 GB of safetensors; standing rule 10 rules
   out a long login-node download), then `prefetch status`.
3. **Generator prompts and config** (EVAL and TRAIN, no shared word 8-gram; decoding settings and seed), committed
   before any generation.
4. **20-document timing job** (`jobs/`, sbatch, ≤ 1 h, half card with `set_per_process_memory_fraction(0.5)`):
   - Input: 20 V3 **val** docs of split seed 0 (7 email, 7 table, 6 code), never test docs. Outputs discarded,
     never EVAL or TRAIN.
   - Runs both generators: Gemma 27B NF4 (EVAL prompt) and Mistral 7B bf16 (TRAIN prompt).
   - Records per generator: load time, prefill and decode tokens/s, s/quad, peak VRAM, the JSON-parse rate and the
     `pairs.validate` pass rate. Extrapolates GPU-hours for about 210 EVAL V3 quads and about 3,141 TRAIN quads.
   - Output: a runlog run under `results/pairs/gen_timing/`.
5. EVAL generation on split-seed-0 test docs after the V3 freeze (the test-doc list may move at the rebuild);
   TRAIN generation as ≤ 1 h sharded sbatch jobs.
6. Annotation (A25), then κ per set, agreed quads (≥ 200 agreed pairs per set), P3 checks.

## Annotators (A25)
- External annotators: not project members, fluent in English, independent of the design. The author adjudicates only.
- Four annotators in two teams; each team labels every EVAL pair of one set (V3-derived or agent-derived), about 300
  pairs per person, κ per set. Fallback with two people: both label both sets.
- Acknowledged by name in the paper. **Names: not yet confirmed** (to add here and in the paper once confirmed).

## Open decisions / blockers
- **Pilot labels** from the author (Next 1).
- **Annotator recruitment** (A25).
- **Agent-pair sets** wait for AD-Clean/TB-Clean contexts (transfer harness) and the agent-style pipeline.

# Paper draft
- 2026-10-31: `paper/` started (ACL template, builds with `tectonic`, 13 pp., no errors). Drafted intro, related work (31 refs re-verified against arXiv/Anthology), threat model, C1 artifacts (macros only), V3 construction (pre-freeze numbers, flagged), intent pairs, limitations; ICT, setup and results are `\todo{}` stubs.

# ICT (C4)
- 2026-10-31: `src/models/ict.py` (segment-pair cross-encoder, BCE + λ·pair hinge, request-swap augmentation; ModernBERT-large option with windowed max-pool), `tests/test_ict.py` (18 CPU tests), `jobs/ict_train.sh`, `jobs/ict_score.sh`. Nothing trained.
- 2026-10-31: **A22 recorded** (ablations a–e, val-only selection rule + 12-point grid on seed 0, swap exclusions + 50-item audit with a 10% noise switch-off, 1e-5 batch tolerance) and implemented (`ict.py select`, `ict.py swap_audit`, `GRID=1` mode in `jobs/ict_train.sh`).
- 2026-10-31: **A23** amends A22: (a) = full ICT minus pairs+margin on the same data (V3 + agent-style), (e) = (a) minus the request; one factor per ablation (table in A23, unit test). Nothing trained.
- **Next:** needs frozen V3, agent-style data, TRAIN pairs and held-out TRAIN-side val pairs; then grid → `select` → commit `configs/ict_hparams.json` → seeds 0–4 × {full, a–e}. Export the swap audit with the EVAL annotation batch.

# Course demo (CS F429 Part 3, due 2026-11-21)
- 2026-10-31: `demo.py --mode gui` (Gradio, `requirements-demo.txt`): side-by-side scores from cached TF-IDF / DeBERTa V2 (3 seeds; match frozen pairs_v0 scores to 2e-8) and V3/ICT when their checkpoints exist; pairs v0 tab; `paper/numbers.tex` tab; tiny CPU live training (TF-IDF ~3 s, DeBERTa 12 docs ~1.5 min). Next: rehearse; add V3/ICT checkpoints when trained.
- 2026-10-31: Row-level leakage claim dropped (no P4 redo; `docs/deviations.md`); C1 rests on V2 artifacts and probes. `docs/venues.md`: 2027 venue scan from official pages (TrustNLP 2027 @ COLING confirmed, 5 Jan 2027; ACL-family/ICLR 2027 workshop dates confirmed; others guide-only). 8-page budget proposed, not applied.
