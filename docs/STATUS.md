# Status (updated 2026-11-11)

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
  - **2026-11-07: protectai-small-v2 added (F021/F022 now 19 detectors).** Encoders re-run by job 434238 (2026-11-06,
    good card `GPU-6ea41e3c…` on gpunode8, no requeue) → `transfer_scores_liuenv/20261106-144044_seed0` (16 detectors
    × 6 sets, at `7af1eba`); metrics `transfer_metrics/20261107-200228_seed42`. The other 18 detectors' values are
    unchanged. small-v2 on AD-Matched, calibrated 1%: TPR 0.266 [0.21, 0.33] at realized FPR 0.109 [0.08, 0.14].
    F022 at 1% on AgentDojo: 13/19 above target, median 0.062 (was 12/18, 0.052); half thresholds at 5%: 0.047 vs 0.118.
  - **2026-11-07: truncation made visible (F023), scoring unchanged** (author decision: Liu's protocol, detectors as
    shipped). `transfer_metrics.py` records each detector's fraction of rows over its max length per set
    (run `transfer_metrics/20261107-210130_seed42`; all F021/F022 numbers identical). Table
    `paper/tables/transfer_truncation.tex` (paper appendix); F021 cells with >25% of rows truncated carry a dagger.
    7 detectors (512-token classifiers + Prismor) truncate 0.518–0.537 of TB-Matched and 0.344–0.371 of TB-Clean;
    none exceeds 25% on AgentDojo. One sentence in the paper's setup. Paper not compiled (no LaTeX on the laptop).
  - CPU diagnostic (laptop; `results/transfer/gate2_followup_batch_cpu/`, not a reference) agrees: batch ≤ 6e-6
    for every model (our DeBERTa ≤ 1.8e-7); env shift identical in rows over 1e-3 and flips (46/9, 228/6, 389/34).

## Next
1. **e. Results layout:** `results/transfer/<detector>/<set>/<run_id>`, which needs a nested-exp change in
   `src/runlog.py`.

## Open decisions / blockers
- **GPU fault (open, 2026-11-05):** gpunode8 is back in service (`mix`) but still serves the ECC card
  `GPU-cfb39ac3-d342-8391-b482-bf2ce8a41587`: probe 432581 and pairs timing 432585 landed on it and the guard stopped
  them (exit 3). `jobs/lib/gpu_guard.sh` now requeues a job that lands on that card (`scontrol requeue`, at most 5 times
  via `SLURM_RESTART_COUNT`, then exit 3); other CUDA failures still exit 3. Every guarded job has `--requeue` and
  `--open-mode=append`, so each attempt's UUID is in its log. Resubmitted 2026-11-05: `ts_encoders` 434238,
  `pairs_timing` 434239. Admins to be asked to drain the card (email drafted by the author). 434238 got a good card
  on its first attempt (no requeue), so the requeue path is still untested on the cluster.
- **Revision pins:** Liu pins only Horizon. wolf-defender and prismor-1.5b changed days before his last commit:
  first suspects if Gate 2 mismatches (re-pin with a reason).
- **VRAM cap:** MPS does not enforce `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` (smoke test, job 422476). Half-card VRAM
  must be capped in code, e.g. `torch.cuda.set_per_process_memory_fraction(0.5)`.
- **Frontier comparator (c):** withdrawn 2026-10-31 (A21, budget). No API key or scoring is needed.
- **History rewrite (2026-11-04):** trailer removed from every commit message, trees identical; map in
  `docs/commit_rewrites.md`. Cluster clones cleaned the same day: `~/nlp_harness` and `~/nlp` reset to the new main
  (commit-msg hook in both); the `~/nlp_probe` worktree and branches `probe-v2`, `probe-v2-next` (old-history commits)
  removed. Its 11 untracked `results/v2_probes/score_*` dirs (55 files) matched the laptop by sha256 and were deleted.
  Archived in cluster `~/archive/`: `nlp_uncommitted_20261017.patch` (2026-10-17 edits; all already in main, the rest
  superseded), `nlp_untracked_20261017.tar.gz` (57 untracked files, which main already has or supersedes),
  `nlp_probe_logs_20261104.tar.gz` (Slurm logs of jobs 404384, 404404, 407923, 422174). `~/nlp_probe` no longer
  exists; the CLAUDE.md code-map line about it is out of date.
- **Cluster notes:** the QoS needs `--gres=mps:50` even for CPU jobs; the template file
  `jobs/templates/mock_job_csisnlp18.sh` was never found, so the header is derived from the existing jobs.

# V3 freeze
- **2026-11-11: `v3-frozen` tagged at `c78f3a7`** (pushed). `data/v3/split_seed_{0..4}` built from the externally
  reviewed pool (`data/benign/benign_v3.json`; 357 keep, 4 edit, 0 benign drops, 60 ambiguous confirmed). Byte-identical
  to the pre-freeze scratch build; gates PASS (F024), integrity PASS (F025), identical to the scratch runs.
  Datasheet: `docs/v3_datasheet.md`. Deviations logged 2026-11-11 (review protocol, its correction, hand-edited run
  records, benign pool assignment fixed at drafting).
- **Next:** V3 baselines; pairs EVAL generation on split-seed-0 test docs; ICT training.

# Intent pairs (C3)

Plan: `docs/pairs_spec.md` v3 (A26, A27); paper_plan A18 (construction, split seed 0 only, strict agreement), A21 (comparator (c)
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
3. **Generator prompts and config: v1-frozen, superseded by v2 (A27, E5) before any EVAL/TRAIN generation** (2026-11-04), committed before any generation call:
   `configs/pairs_gen_v1.yaml` (decoding, seed 20261104, quads per doc per mode; pins read from detectors.yaml),
   `configs/prompts/pairs_eval_v1.txt` (Gemma) and `pairs_train_v1.txt` (Mistral), written independently, no shared
   word 8-gram (`tests/test_pairs_gen.py`). Pipeline: `src/pairs_gen.py` (mode guards: eval = split-seed-0 test only,
   train = train only, timing = val only; one quad per call; atomic resumable shards; per-quad raw output, parse and
   validate result, revision, prompt and config sha256) and `scripts/gen_pairs.py` (`docs`, `generate`, `timing`).
4. **Timing done: job 434239** (2026-11-06, gpunode8, good card `GPU-6ea41e3c…`, guard attempt 0, no requeue; 14:49;
   commit 7af1eba; config v3 = v2 prompts; 20 val docs of split seed 0, one quad each). Run
   `results/pairs/gen_timing/20261106-145540_seed0/` (operational numbers, not paper numbers; no findings entry).
   - Gemma NF4 bs1: load 409 s, prefill 2,067 / decode 27.9 tok/s, 8.52 s/quad, peak 19.0 GB; parse 1.00, validate
     0.85 (2 quads request-length ratio, 1 quad 5-gram copy); E5 0/20. 210 quads 0.50 GPU-h (0.58 per valid);
     38 agent quads ~0.09 GPU-h (0.11 per valid); plus one 7-min load.
   - Mistral bf16 bs1: 3.61 s/quad, 68.8 tok/s, 15.0 GB; parse 0.90, validate 0.30; bs8: 1.49 s/quad, 179 tok/s,
     18.4 GB; parse 0.95, validate 0.25. Main failure: requests copy a 5-gram of the instruction (Q3). E5 'Reply'
     1/20 per run. 3,141 quads: bs1 3.15 GPU-h (10.5 per valid), bs8 1.30 (5.2 per valid).
   - Failure texts were deleted by design (A24); only error kinds survive. n = 20, so the rates are rough.
   - **A29 applied (2026-11-07):** a failed quad (parse, near-duplicate, validation) is regenerated with a new seed up
     to 3 times, every attempt's stage, reasons and text in the record; TRAIN prompt `pairs_train_v3.txt` + config
     `pairs_gen_v4.yaml` (default; EVAL prompt unchanged); E5 flags "reply" only as "reply to/all". Re-timing on the same
     20 val docs: job 437312 (keeps failed-attempt text in `<run>/failures.jsonl`, val only). Target: Mistral
     validate ≥ 0.6 after retries, Gemma ≥ 0.85.
5. EVAL generation on split-seed-0 test docs after the V3 freeze (the test-doc list may move at the rebuild);
   TRAIN generation as ≤ 1 h sharded sbatch jobs.
6. Annotation (A25), then κ per set, agreed quads (≥ 200 agreed pairs per set), P3 checks.
7. **Agent-derived pairs: secondary set (A28, 2026-11-05).** Field-level JSON insertion added no docs (τ-bench has
   no string value over 3 words; AgentDojo outputs are not JSON). Target: the 19 eligible AD-Clean docs (17 user tasks)
   × 2 quads = 38 quads, 76 pairs, descriptive, user-task CIs; decides no hypothesis. H3(a), H4 and the P2 floor are
   scoped to the V3-test pairs. Annotation rebalanced (~250 pairs, ~4.3 h per person). Config `pairs_gen_v3.yaml`
   (default) adds `agent: 2`. Docs: `gen_pairs.py docs --mode eval --source agent` (manifest committed, text under
   `data/transfer/pairs_docs/`).
8. **Next:** read the re-timing run (job 437312). EVAL and TRAIN generation wait for the V3 freeze.

## Annotators (A25)
- External annotators: not project members, fluent in English, independent of the design. The author adjudicates only.
- Four annotators in two teams; each team labels half of the V3 quads and half of the agent quads (A28), about 250
  pairs and 4.3 h per person, κ per set and per team. Fallback with two people: both label everything (~500 pairs, ~8.5 h).
- Acknowledged by name in the paper. **Names: not yet confirmed** (to add here and in the paper once confirmed).

## Open decisions / blockers
- **Generation fixes (2026-11-04):** length rules use only the V3 tokenizer (sha256-pinned in `pairs_gen.py`; repo
  copy or `$HF_HOME/tokenizers_hpc/deberta_v2_ckpt`, synced to the cluster; fails loudly if missing; eval refuses to
  run without it); near-duplicate quads within a doc (content-word Jaccard ≥ 0.8 on requests or instructions) are
  regenerated up to 3 times, counted in `dup_rejections`; parser takes the first complete JSON object (bug found by
  the laptop smoke test: tiny instruct model on MPS through the real HF path, NF4 skipped; not a result).
- **Rule E5 applied (2026-11-04): spec v3, A27.** Instructions may only shape the answer; actions and targets from
  the context are excluded. Guide rule 8 + example 12; `configs/pairs_gen_v2.yaml` + prompts v2-frozen (now the
  default; `pairs_gen` refuses eval/train with v1). H4 narrowed in paper_plan §3 and the paper to aligned
  answer-shaping instructions. pairs_v0 items 0–23 marked excluded in data_provenance (file kept).
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
