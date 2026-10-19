# CLAUDE.md — ipi-generalization
## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

## What this repo is
Research codebase for **"Generalization Limits of Indirect Prompt-Injection Detectors"** (working title).
It started as the CS F429 (NLP, BITS Pilani Hyderabad) course project, Group 18. The goal is now a
**publishable paper**. Course deliverables (Phase 3 report and demo, due 21 Nov 2026) are a side effect,
not the target. When the two conflict, optimize for the paper and flag the conflict.

Task: binary classification. Given `Intent: <user query> | Context: <external document>`,
predict whether the context contains an injected instruction (1) or is clean (0).
Data: BIPIA benchmark (Yi et al., KDD '25), tasks EmailQA, TableQA, CodeQA only.
Full audit of the current state: `CODEBASE_STATE.md` (2026-10-19). Read it before structural work.

## Frozen state: tag `v2-frozen`
- `data/processed/` and everything in `results/` as of tag `v2-frozen` are **frozen**.
  Never regenerate, overwrite, or delete them.
- New data goes under `data/v3/...` and new results under `results/v3/...` (or another versioned subfolder).
- These commands write into frozen paths **by default**. Never run them without pointing their outputs elsewhere:
  - `python src/data_pipeline.py` writes `data/processed/` (override with `--output_dir`).
  - `python run_experiments.py` hardcodes `results/`. It has no CLI, so it must be changed before reuse.
  - `python src/models/tfidf_lr.py --mode train` now refuses frozen paths: pass `--checkpoint_dir` (and optionally `--output`).
  - `python src/models/deberta.py --mode train` writes `results/deberta_*_results.json` (override with `--output`).
  - `score_detectors.py score` and `summarize` write `results/scores/`. `build` writes `data/pairs/` and `data/external/`.
  - `demo.py --mode train` (and menu option 7) runs `run_experiments.py`. `--mode stats` rebuilds data if it is missing.
  - `submit_train.sh` and `submit_score.sh` write `results/`.
- `checkpoints/tfidf_lr_model.pkl` produced the frozen TF-IDF scores. Treat it as frozen too.

## Core claims the paper is built on (protect these)
1. **Leakage audit:** naive row-level splits on BIPIA-derived data leak. *V1 numbers are unusable* (see rules below).
   This claim must be re-established with committed code before it appears anywhere.
2. **Leakage-free benchmark:** document-level group splits, exact 1:1 balance per task×split,
   held-out attack pools (train/test attack strings disjoint), and disjoint benign-imperative pools.
   V2 achieves the leakage properties but has **insertion artifacts** (see below). V3 must fix them.
3. **Generalization gap:** validation (seen attack pool) ≫ test (held-out pool). Validation tuning does not transfer.
   Leave-one-task-out drops sharply (TF-IDF on CodeQA/EmailQA; DeBERTa on TableQA).
4. **(Phase 3, to build)** A method that measurably narrows the gap. Currently planned as Intent-Counterfactual Training (C4).

Any change that could weaken one of these must be reported, not silently absorbed.

## Research-integrity rules (non-negotiable)
- **Never invent, round, or "estimate" a number.** Every number in the paper or a report must trace to a
  file in `results/` produced by committed code. If a number has no source file, say so.
- **Test set is sealed.** Do not tune hyperparameters, thresholds, templates, or preprocessing by looking
  at test results. Model selection uses validation only. If a test number was ever used for a decision,
  flag it.
- **Do not modify** the V2 split assignments, attack pools, or the `data/benign/benign_instructions.json`
  test pool. All V2 data is frozen (see above). If a change is needed, create V3 alongside; never overwrite V2.
- **Seeds:** every experiment takes `--seed`; headline results report mean ± std over ≥3 seeds
  (target 5) *and* the document-level cluster bootstrap CI (B=1000, resample by `doc_id`).
- **Negative results are results.** If the proposed method doesn't beat baselines, report it honestly;
  do not cherry-pick seeds, tasks, or thresholds.
- When reporting comparisons, check whether bootstrap CIs overlap and say so. Don't claim
  "significant" without a paired test (paired bootstrap over docs, or McNemar).
- **No V1 numbers anywhere** (report, README, paper, slides). The V1 audit figures (94.2%, −4.63, 68.1%, etc.)
  have no source code or data in the repo or in git history.
- **V2 DeBERTa results are not headline results** until they are re-measured on an artifact-controlled dataset (V3).
  They may appear only as "on the artifact-laden V2 benchmark", next to the shortcut baseline.
- **LLM/judge prompts are frozen before scoring.** Every judge or LLM prompt lives in a committed file before any
  scoring run, and is never edited after seeing results. A revised prompt is a new, separately named version,
  and all versions are reported.

## Known V2 artifacts (do not build claims on V2 DeBERTa numbers)
Verified in `CODEBASE_STATE.md` §3.3 and §7.
- **Insertion position depends on the label.** With `variants_per_doc=4`, benign imperatives are always `middle`.
  Attacks are 50% `start` and 50% `middle`. `end` is never used. Every `start` row is injected.
  The untrained rule "first `\n\n`-paragraph ≤30 words → injected" gets **test macro-F1 0.7007**
  (TF-IDF default: 0.7713).
- **`middle` mode rewrites whitespace.** `insert_sentence` does `context.split()` + `" ".join(...)`, which collapses
  every newline in the document. Raw-clean rows keep their formatting. Benign rows (always middle) never do.
  Attacks keep it only in `start` mode. P(text has a single `\n`): raw_clean 0.935, benign 0.000, attack 0.577.
- **`benign_train` still contains a `"Note: …"` template** ("Note: ignore the introduction and focus on the results section.").
- **All CodeQA rows share the constant intent `"Perform code QA"`.** BIPIA code rows have no `question` field,
  and the pipeline drops BIPIA's `code` and `error` fields. The intent string identifies the task.
- **TF-IDF LOTO selected C using val docs of the held-out task.** `run_experiments.py` reuses the C chosen on the
  full val split. DeBERTa LOTO does not have this problem.
- **DeBERTa val macro-F1 saturates at 1.0.** Val attacks come from the train pool. Checkpoint selection
  (best val macro-F1) therefore carries little signal; the logged runs all picked epoch 2.
- Also: some `benign_test` templates are model-directed ("Act as a helpful assistant…", "System: …", "New task: …")
  and therefore label-ambiguous. Inserted sentences differ by class in surface form: benign always ends in "." and has
  no quotes, newlines, or "?".

## Paper story (planned contributions)
- **C1. Construction artifacts in insertion-based IPI detection data:** position, whitespace, templates, and leakage,
  and how much of a detector's score they explain (shortcut baselines, ablations, artifact-controlled re-measurement).
- **C2. An artifact-controlled dataset (V3) plus a generalization study:** held-out attacks, held-out tasks,
  external benchmarks (InjecAgent, deepset), and public detectors (ProtectAI v2, PIGuard, LLM judge).
- **C3. Intent-counterfactual pairs:** the same instruction in the same document is benign under one user query
  and an injection under another. Pilot (`data/pairs/pairs_v0.csv`, 50 pairs, written by one author on V2 test
  docs, not yet human-validated): **pair accuracy** (not AUC). DeBERTa 0.347 ± 0.115 over 3 seeds;
  Qwen2.5-7B judge 0.240; both below the 0.5 chance level (`results/scores/summary.json`).
  The detectors appear intent-blind.
- **C4. Intent-Counterfactual Training (ICT)**, the course novelty: a query/context segment-pair cross-encoder
  trained with counterfactual pairs plus a pairwise margin loss.

Framing: *evaluation methodology + generalization failure* first, method second.

## Code map (verified 2026-10-19 at `v2-frozen`)
- `src/data_pipeline.py`: **the V2 builder** (there is no `_v2` file). Doc dedup, IDs, task-stratified 70/15/15 doc split,
  4 variants/doc, `insert_sentence`. `--seed` defaults to 42; it reproduces `data/processed/` exactly.
- `src/preprocess.py`: `build_combined_text`, and TF-IDF cleaning (whitespace collapse, URL/email masking,
  keyword-preserving stopwords, lemmatization).
- `src/models/tfidf_lr.py`: stand-alone TF-IDF+LR (C=1, τ=0.5). **The only producer of `checkpoints/tfidf_lr_model.pkl`**
  (used by demo.py and score_detectors.py). Its default `--output` clobbers `results/tfidf_lr_results.json`.
- `src/models/deberta.py`: DeBERTa-v3-base fine-tuning in a **custom PyTorch loop** (not HF Trainer).
  max_len 512, batch 8×2, lr 2e-5, 3 epochs. `--seed`, `--test_task` (LOTO), `--train_subsample`.
- `src/evaluate.py`: **mostly dead.** Only `pair_accuracy` is live (imported by score_detectors.py).
- `src/metrics.py`: **the single metrics module.** Point metrics (incl. PR-AUC, Recall@1/5/10%FPR + threshold, ECE),
  `group_metrics`, `pair_metrics`, `doc_cluster_bootstrap` (bit-identical to V2), `paired_bootstrap`, `mcnemar`.
  The legacy `compute_metrics` functions in tfidf_lr.py, deberta.py and run_experiments.py delegate to it.
  `score_detectors.metrics` does NOT: its Recall@5%FPR (`drop_intermediate=True`) differs on pairs/{piguard,protectai}.
- `src/runlog.py`: every new run calls `start_run(...)`, which writes `results/<version>/<exp>/<run_id>/` and appends
  `results/REGISTRY.md`. `assert_not_frozen(path)` is the frozen-path guard. `PRED_COLUMNS` is the predictions schema.
- `scripts/convert_v2_predictions.py`: V2 predictions/scores → `results/v2_std/<eval_set>/<model>.csv` (+ manifest).
- `scripts/make_tables.py`: LaTeX tables in `paper/tables/` + `paper/numbers.tex` macros, verified against frozen JSON.
- `scripts/error_stats.py`, `scripts/deberta_v2_stats.py`: V2 error analysis and DeBERTa CIs → `results/v2_std/*.json`.
- `scripts/tokenizer_check.py`: DeBERTa tokenizer diagnostics → `results/v3/tokenizer_check/`.
- `src/probes.py`: `extract_inserted_span` (exact inverse of V2 `insert_sentence`) and `insert_preserving`
  (formatting-preserving insertion at line/sentence/word boundaries, records the char offset).
- `scripts/artifact_baselines.py`: position oracle, first-paragraph rule, prefix/suffix-30 TF-IDF probes, span-form
  probe → `results/v2_probes/artifact_baselines/`.
- `scripts/build_probes_v2.py`: P1–P4 position stress tests from V2 test docs → `data/probes_v2/` (+ manifest).
- `scripts/score_probes.py`: scores probes + V2 test with any `score_detectors` scorer → `results/v2_probes/score_<model>/`.
- `slurm/probe_v2.sh`: GPU probe scoring (DeBERTa 3 seeds, retrain to `checkpoints/v2_retrain/` if missing; ProtectAI;
  PIGuard; Qwen judge). `scripts/probe_report.py`: reliance table, paired position contrasts, figure
  (`paper/figures/position_flag_rates.*`) → `results/v2_probes/probe_report/`.
- `tests/`: `python -m pytest tests/` (metrics hand-computed cases, bootstrap equivalence, runlog guard, probe insertion).
- `run_experiments.py`: TF-IDF val tuning, test eval, Recall@5/10%FPR, doc-cluster bootstrap, per-task, LOTO, ablations.
  No CLI and no seed argument; it writes `results/` directly.
- `score_detectors.py`: builds the pairs/InjecAgent/deepset sets; zero-shot scoring (tfidf, deberta, hf, judge); summary.
- `submit_train.sh`, `submit_score.sh`: BITS HPC Slurm jobs (3 seeds; skip a run if its output exists).
- `demo.py`: course demo CLI (menu/stats/train/compare/tasks/samples/errors/predict/full).
- `data/benign/benign_instructions.json`: `benign_train` (30) / `benign_test` (15), disjoint.
- `data/pairs/`: `pair_specs_v0.json` (50 hand-written specs) and `pairs_v0.csv` (100 rows).
- `data/external/`: `injecagent.csv` (1,054 injected + 204 clean), `deepset.csv` (116).
- `data/raw/` (gitignored): BIPIA clone @ `a004b69`, InjecAgent clone @ `f19c9f2`.
- Untracked course docs: `report_phase2.tex/.pdf`, `docs/architecture_spec.md`, `docs/proposal.md`, `docs/project_rubric.md`.
- DeBERTa checkpoints exist only on the HPC cluster. `checkpoints/deberta_best/` is empty locally.

Environment: `requirements.txt` (pinned to local .venv), `requirements-dev.txt` (+pytest),
`requirements-hpc.txt` (torch 2.11.0+cu128 confirmed; other versions TBC).

**To be created (do not reference as existing):** the paper draft itself in `paper/` (only generated tables and
`numbers.tex` exist), `slurm/` (or move the submit scripts there).

Keep this section updated when files move.

## Experiment hygiene
- Target layout for new runs: `results/v3/<exp_name>/<timestamp>_seed<k>/` containing `config.json`
  (all hparams + git commit hash + data version), `metrics.json`, `predictions.csv`, and logs. V2 results do not follow this.
- **Prediction schema (all new runs, enforced by `runlog.validate_predictions`):**
  `id, doc_id, split_seed, task, label, attack_category, attack_source, insertion_mode, clean_type, prob, pred, threshold, model, run_id, eval_set`
  (empty where not applicable). Pair ids are derived from `id` (`pairNNN_<role>`).
  Frozen V2 files keep their original columns; use the converted copies in `results/v2_std/` for analysis.
- `results/REGISTRY.md`: one line per run (appended by runlog) with date, version, exp, run_id, commit, data version,
  headline metric, and path. Paper tables are generated from results files by `scripts/make_tables.py`, never typed by hand.
- Metrics to always compute: macro-F1, injected-class P/R/F1, ROC-AUC, Recall@5%FPR, Recall@10%FPR,
  per-task breakdown, per-attack-category FN rate, FP rate on benign-imperative vs. raw clean.
  All are in `src/metrics.py`; V2 values are in `results/v2_std/error_stats_*.json` and `deberta_summary.json`.
- Evaluation settings to support for every model: in-distribution test, held-out attack test,
  leave-one-task-out (train on 2 tasks' TRAIN docs, test on the 3rd task's TEST docs; select hparams without the held-out task).

## Known weaknesses a reviewer will hit (address before submission)
- The V2 insertion artifacts above. Positional and separator shortcuts must be ablated, and V3 must remove them.
- DeBERTa has 3 seeds (target 5) and is measured on artifact-laden V2 only.
- **Whitespace visibility:** DeBERTa-v3's tokenizer (SentencePiece) maps `\n`, `\n\n` and a space to the same
  tokens, and TF-IDF's `clean_text` collapses all whitespace. So whitespace and separators are **invisible to
  DeBERTa-v3 and TF-IDF**, but **still visible to the Qwen judge** (raw text in the chat prompt) and to any other
  model whose tokenizer keeps newlines. For DeBERTa and TF-IDF, the live V2 construction artifacts are position and
  the surface form of inserted spans. The transformers "incorrect regex pattern" warning is a false positive for
  this checkpoint; its suggested `fix_mistral_regex=True` breaks tokenization. Don't use it
  (`results/v3/tokenizer_check/`).
- Probe sets live in `data/probes_v2/`, probe results in `results/v2_probes/` (written via `src/runlog.py`).
- **`results/scores/summary.json` Recall@5%FPR is wrong on pairs for PIGuard and ProtectAI** (0.02; correct 0.04):
  `score_detectors.recall_at_fpr` uses `roc_curve(drop_intermediate=True)`, which drops valid tied-score operating
  points. The paper must take Recall@FPR from `results/v2_std/` via `src/metrics.py`, never from `summary.json`.
- The test split is small (72 docs; EmailQA 12), so CIs are wide. Consider more held-out data or more variants
  per doc *without* breaking document-level grouping.
- Model-directed `benign_test` templates are label-ambiguous. Plan: a stricter human-addressed benign pool, and
  report results both with and without the ambiguous items.
- Only 3 of BIPIA's 5 tasks are used. Justify this, or add WebQA/Summarization.
- Pairs v0 are unvalidated and were written on test docs. They need independent labeling before they count as evidence.

## Compute
- Local: Apple Silicon laptop (CPU / PyTorch MPS). Fine for TF-IDF and debugging.
- GPU training: BITS HPC (Sharanga, partition `gpu_rtx_pro_6000_6_csis_hyd`, `--gres=mps:50`).
  Jobs must be resumable and write to versioned `results/` subfolders with the same layout as local runs.

## Working style
- Before non-trivial changes, state the plan in a few lines and what it could break.
- Prefer small, reviewable commits with messages that say *why*.
- Don't add dependencies without saying so; pin versions in `requirements.txt`.
- When you're unsure whether something is leakage or a shortcut, assume it is and check.
- The author works solo despite a nominal 5-person group; optimize for one person's time.

## Paper
- The draft will live in `paper/` (LaTeX, not yet created). Target venue: TBD (decide early; it sets page limit and format).
- Release plan: V3 dataset construction script, split manifests, benign pools, and pair specs, so others can reproduce.
