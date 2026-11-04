# Status (updated 2026-11-04)

# Transfer harness

Plan: paper_plan §3, A16–A17 (Liu 2026 sets as headline transfer; reproduction in Liu's pinned env and ours).

## Done
- **Step a, set build: Gate 1 PASS.** Run `results/transfer/set_build_gate1/20261030-124112_seed0` (commit `a0b9703`, pre-rewrite `6c94421`),
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
- **Gate 2 follow-up (jobs 424070–424072, commit `bc2f183`, pre-rewrite `007c5a5`; F017, F018):**
  - Batch test: scores do not depend on batch size (≤ 7e-6, published encoders in both envs; our 3 V2 DeBERTa
    checkpoints ≤ 3.6e-6). No batch-policy change needed; dynamic padding stays.
  - The transformers-5 shift is not padding: it is present at bs 1. Liu's env reproduces his files; ours does not.
    Policy (deviations 2026-10-30): published detectors are scored in Liu's env; our models in ours.
  - AD pairing recovered: 97/97 user tasks in both sets; A19 metrics PASS with it. Pre-registered row validation
    FAIL on PIGuard only (6 + 5 rows, max 1.5e-3, no flips), all on calendar rows whose participant order depends on
    the hash seed (`results/transfer/gate2_followup_residual/`). **Adopted** for every detector (deviations
    2026-10-31); timestamp rows (51/85) stay out of matching and are reported separately.
  - Judge: deterministic (repeat ≤ 7e-8; 6 GiB identical, fully on GPU). Row gap to Liu (up to 0.77) is
    hardware/kernel, not noise. bf16 − 4-bit AD-Matched AUROC: −0.018 (seeded), −0.012 (recovered).
  - Cause of the transformers-5 shift found (job 425896, F018): PIGuard/PG2 tokenizer rebuilt from spm.model
    instead of the repo's tokenizer.json; Horizon's sliding-window RoPE theta set to the global value. Undoing
    both: ≤ 7e-6 to Liu's env, 0 rows over 1e-3. Scoring policy unchanged.
  - **Transfer scoring done (F021):** 15 classifiers (Liu's env), Wolf Defender (ours), 2 judges × {4-bit, bf16} on
    AD-Matched, AD-Shift, TB-Matched, AD-Clean, TB-Clean and the calibration pool (`data/transfer/sets_adliu`,
    `…filesfb0afded`; jobs 425944–425946, 426037). Metrics: `eval/transfer/transfer_metrics.py` →
    `results/transfer/transfer_metrics/`, table `paper/tables/transfer_detectors.tex`. Calibration-pool thresholds
    give realized FPR on AD-Clean from 0.003 to 0.23 at a 1% target; JasperLS flags the whole pool.
    Rescoring is bit-identical (96/96 files vs the first pass, `…files69e403d1`, kept as superseded runs).
  - **Gate 2: PASS** (A19 metrics, all four detectors, all sets, recovered AD pairing, Liu's env).
  - **2026-11-02: judges are bf16** for every baseline and H3 comparison (A20); 4-bit only for the Liu reproduction.
    F021 re-rendered from `results/transfer/transfer_metrics/20261102-115039_seed42` (bf16 judges only; adds
    task_block_rate on AD-Clean/TB-Clean). The crashed uncommitted run `transfer_scores_judges_nf4/20261031-184508_seed0`
    was deleted on the cluster.
  - **Secondary protocol (task-disjoint halves) and calibration transfer: F022.** The pool's thresholds miss their
    target on AgentDojo for most detectors (1% target: 12/18 with the realized-FPR CI above 1%, median 0.052); with
    in-domain half thresholds the median realized FPR at 5% is 0.053 vs 0.118 with the pool's on the same half.
  - CPU diagnostic (laptop; `results/transfer/gate2_followup_batch_cpu/`, not a reference) agrees: batch ≤ 6e-6
    for every model (our DeBERTa ≤ 1.8e-7); env shift identical in rows over 1e-3 and flips (46/9, 228/6, 389/34).

## Next
1. **protectai-small-v2:** access granted and downloaded (2026-11-02), added to `jobs/transfer_scores_encoders.sh`.
   Jobs 429434 and 429443 both failed at start with `CUDA error: uncorrectable ECC error` on gpunode8 (the partition's
   only node). Once `jobs/gpu_probe.sh` passes: `sbatch jobs/transfer_scores_encoders.sh` (finished scores are
   skipped), rerun `eval/transfer/transfer_metrics.py` with the new liuenv run; F021/F022 update via `make_tables.py`.
2. **e. Results layout:** `results/transfer/<detector>/<set>/<run_id>`, which needs a nested-exp change in
   `src/runlog.py`.

## Open decisions / blockers
- **GPU fault (still open 2026-11-04):** uncorrectable ECC errors on gpunode8 (jobs 429434, 429443; card
  `GPU-cfb39ac3-d342-8391-b482-bf2ce8a41587`). Admins drained the node (`sinfo`: `draining`, reason `ECC_Error`, root,
  2026-11-03 11:33); it is the partition's only node, so no new GPU job starts until it is resumed. Probe job 432581
  (`jobs/gpu_probe.sh`: per-GPU uncorrected ECC count + gpu_guard) is queued (`ReqNodeNotAvail`) and runs when the
  node returns; read `logs/slurm_gpu_probe_432581.log` in `~/nlp_harness`.
  Every GPU job now runs `jobs/lib/gpu_guard.sh` (via `job_env.sh`): it exits 3 before any work if the 1-tensor CUDA
  test fails or the job lands on the bad UUID, and runlog records `hardware.gpu_uuid`.
- **Revision pins:** Liu pins only Horizon. wolf-defender and prismor-1.5b changed days before his last commit:
  first suspects if Gate 2 mismatches (re-pin with a reason).
- **VRAM cap:** MPS does not enforce `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` (smoke test, job 422476). Half-card VRAM
  must be capped in code, e.g. `torch.cuda.set_per_process_memory_fraction(0.5)`.
- **Frontier comparator (c):** withdrawn 2026-10-31 (A21, budget). No API key or scoring is needed.
- **History rewrite (2026-11-04):** trailer removed from every commit message, trees identical; map in
  `docs/commit_rewrites.md`. `~/nlp_harness` reset to the new main. **Cluster `~/nlp` and `~/nlp_probe` still hold the
  old history and are not cleaned** (no local-only commits, but unique uncommitted work): `~/nlp` (main at `01d1c78`) has uncommitted edits from 2026-10-17 to `.gitignore`,
  `src/evaluate.py`, `src/models/deberta.py`, `submit_train.sh`; `~/nlp_probe` has 11 untracked
  `results/v2_probes/score_*` run dirs (2026-10-29). Author to decide (keep/diff/discard) before reset and worktree removal.
- **Cluster notes:** the QoS needs `--gres=mps:50` even for CPU jobs; the template file
  `jobs/templates/mock_job_csisnlp18.sh` was never found, so the header is derived from the existing jobs.

# Intent pairs (C3)

Plan: `docs/pairs_spec.md` v2 (A26); paper_plan A18 (construction, split seed 0 only, strict agreement), A21 (comparator (c)
withdrawn), A24 (EVAL generator), A25 (annotators), A26 (pilot-based label rules). Nothing has been generated (except the timing
job's discarded val-doc outputs), annotated or scored.

## Done
- Spec, annotation guide, `src/pairs.py` (validation, overlap class, complete quads), annotation page
  (`scripts/make_annotation_sheet.py`), agreement script (`scripts/agreement.py`).
- 20 LLM-drafted pilot pairs (10 quads) on split-seed-0 test docs: `data/pairs/pilot_v1.jsonl`. Never EVAL.
- Generators fixed and pinned in `configs/detectors.yaml` (role `generator`); access granted on the cluster token
  (2026-10-31):
  - TRAIN `gen-train-mistral`: `mistralai/Mistral-7B-Instruct-v0.3` @ `c170c70` (A18).
  - EVAL `gen-eval-gemma`: `google/gemma-3-27b-it` @ `005ad34`, 4-bit NF4 (A24).

## Next
1. **Author labels the pilot: done** (2026-11-04). `annotation/pilot_v1/devansh.csv`; report
   `results/pairs/pilot_v1/20261104-132250_seed0/report.md` (`scripts/pilot_report.py`). 39/40 members, 19/20 pairs,
   9/10 quads match the drafted labels; 1.03 min/pair → 5.15 h per 300 pairs (A25). **Revisions applied** as spec v2
   (A26): restating the request's task or constraints is aligned (case 12b), case 12 narrowed, guide rule 5 and
   example 11, notes only for unclear/low-confidence/suspect items. E3/E4 unchanged.
2. **Generator download: done.** Job 425712 (2026-10-31); prefetch status 2026-11-02: both complete.
3. **Generator prompts and config: done, v1-frozen** (2026-11-04), committed before any generation call:
   `configs/pairs_gen_v1.yaml` (decoding, seed 20261104, quads per doc per mode; pins read from detectors.yaml),
   `configs/prompts/pairs_eval_v1.txt` (Gemma) and `pairs_train_v1.txt` (Mistral), written independently, no shared
   word 8-gram (`tests/test_pairs_gen.py`). Pipeline: `src/pairs_gen.py` (mode guards: eval = split-seed-0 test only,
   train = train only, timing = val only; one quad per call; atomic resumable shards; per-quad raw output, parse and
   validate result, revision, prompt and config sha256) and `scripts/gen_pairs.py` (`docs`, `generate`, `timing`).
4. **Timing job submitted** (`jobs/pairs_gen_timing.sh`, job 432585), queued while gpunode8 is drained:
   - Input `data/pairs/docs/timing_s0.jsonl` (20 val docs of split seed 0, 7 email / 7 table / 6 code; pre-freeze
     split; manifest has its sha256 and the span caps). One quad per doc.
   - Gemma NF4 batch 1, Mistral bf16 batch 1 and batch 8; half card. Writes `results/pairs/gen_timing/<run_id>/`;
     generated text is deleted from scratch after the summary.
   - Check: `squeue -u $USER`; log `~/nlp_harness/logs/slurm_pairs_timing_<job>.log`. Resume: resubmit (finished
     quads skipped; exits at once once summarised).
5. EVAL generation on split-seed-0 test docs after the V3 freeze (the test-doc list may move at the rebuild);
   TRAIN generation as ≤ 1 h sharded sbatch jobs.
6. Annotation (A25), then κ per set, agreed quads (≥ 200 agreed pairs per set), P3 checks.

## Annotators (A25)
- External annotators: not project members, fluent in English, independent of the design. The author adjudicates only.
- Four annotators in two teams; each team labels every EVAL pair of one set (V3-derived or agent-derived), about 300
  pairs per person, κ per set. Fallback with two people: both label both sets.
- Acknowledged by name in the paper. **Names: not yet confirmed** (to add here and in the paper once confirmed).

## Open decisions / blockers
- **Timing job 432585** queued (ReqNodeNotAvail: gpunode8 drained).
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

# Agent-style data (A3)
- 2026-10-31: `src/agent_pipeline.py` (19 tool types, no LLM/GPU) built `data/agent_v1/`: calibration pool 1,140 clean outputs (sha256 `2657cbf7…`), training split seeds 0–4 from BIPIA train attacks only; overlap audit PASS, content-free gates PASS (F019). Next: V3 baselines / H1 with pool thresholds.
- 2026-10-31: Within-train CV found a span length/form shortcut in agent_v1 training data (span length 0.65, form 0.61–0.64 macro-F1). Rebuilt as `data/agent_v2` (spans matched to each split's attacks; pool unchanged): CV passes (≤ 0.599), overlap audit PASS, but the content-free gate **span_form_you_lr FAILS** (pooled up to 0.601, per tool up to 0.633; F020). **agent_v2 cleared for ICT training** (author decision 2026-10-31, `docs/deviations.md`; |AUROC − 0.5| reported per gate probe, not a gate). Predictions are now written as `.csv.gz`.
- **Venue (recorded 2026-10-31, paper_plan §8):** primary TrustNLP 2027 at COLING (deadline 2027-01-05); fallbacks ICLR 2027 workshops (~2027-02-01), NAACL 2027 workshops (2027-02-05); arXiv at submission.
