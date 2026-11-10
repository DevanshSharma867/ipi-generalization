# V3 datasheet

Frozen at tag `v3-frozen` (commit `c78f3a7`, 2026-11-11). Data in `data/v3/split_seed_<k>/` for k = 0–4; never
modified (a change means a new version). Design and decisions: `docs/v3_design.md`. Gate and integrity results:
`docs/findings.md` F024 and F025. Protocol changes: `docs/deviations.md`.

## Purpose and task
Binary classification of indirect prompt injection. Input: `Intent: <user query> | Context: <document>`. Label 1 if
the context contains an injected instruction, 0 otherwise. V3 is built so that the label depends only on the content
of one inserted span and no content-free feature (position, separator, length, span form) predicts it. It replaces
V2 (`v2-frozen`), whose construction artifacts are documented in `CODEBASE_STATE.md` and F006/F007.

## Sources
- Documents and intents: BIPIA (Yi et al., KDD '25) @ `a004b69`, tasks EmailQA, TableQA, CodeQA.
- Attacks: BIPIA attack files. Train/val rows draw from `*_attack_train.json`, test rows from `*_attack_test.json`
  (0 shared strings). Train and val attack categories are disjoint (val = category-held-out, decision D7).
- Benign spans (clean A): `data/benign/benign_v3.json`, LLM-drafted, reviewed in full by one external human reviewer
  (see Review). Clean B: spans spliced from another document of the same task and split (table: sentences rendered
  from a donor table's header and one row).

## Construction (`src/data_pipeline_v3.py`, last changed at `4cdae64`)
- Documents deduplicated (V2 normalization + exact dedup), then grouped into near-duplicate components (word-5-gram
  Jaccard ≥ 0.5, D4); a component goes to one split. Document-level split per task: n_val = n_test = round(0.15·n).
- Each main document yields 4 rows: 2 injected (two different attacks from the split's pool), clean A, clean B.
  Exact 1:1 balance per task × split. Every row has exactly one inserted span.
- Insertion: position ∈ {start, middle, end} and separator ∈ {`\n\n`, `\n`} sampled i.i.d. per row, the same way
  for every row type. Middle = nearest line/paragraph boundary to the midpoint, fallback paragraph → email field
  delimiter → sentence (D1); formatting is preserved (no re-joining of words). Per row: `insertion_mode, separator,
  boundary_kind, char_offset, span_len`.
- Length control (D3): every main row ≤ 512 DeBERTa-v3 tokens. Code and table docs are trimmed to a window before
  insertion, identically for all rows of a doc; email docs that do not fit go to the `long` slice.
- Benign form targets (D5, D6): clean-A target rates per register = 2 × train-attack rate − clean-B rate, from the
  TRAIN attack files only, including "contains you/your".
- Eval-only slices: `val_id` (val docs with train-category attacks), `raw_clean` (untouched docs), `ambiguous`
  (model-directed items, flag rate only), `long` (over-length docs, every split seed).

## Sizes
Docs / rows per split and task (rows are 50% injected in every cell). Source: `data/v3/split_seed_<k>/*.csv`; row
totals also in F025.

| Split seed | Split | EmailQA | TableQA | CodeQA | Total rows |
|---|---|---|---|---|---|
| 0 | train | 50 / 200 | 491 / 1964 | 68 / 272 | 2436 |
| 0 | val | 13 / 52 | 105 / 420 | 14 / 56 | 528 |
| 0 | test | 14 / 56 | 105 / 420 | 14 / 56 | 532 |
| 1 | train | 44 / 176 | 491 / 1964 | 68 / 272 | 2412 |
| 1 | val | 20 / 80 | 105 / 420 | 14 / 56 | 556 |
| 1 | test | 13 / 52 | 105 / 420 | 14 / 56 | 528 |
| 2 | train | 48 / 192 | 491 / 1964 | 68 / 272 | 2428 |
| 2 | val | 17 / 68 | 105 / 420 | 14 / 56 | 544 |
| 2 | test | 12 / 48 | 105 / 420 | 14 / 56 | 524 |
| 3 | train | 46 / 184 | 491 / 1964 | 68 / 272 | 2420 |
| 3 | val | 12 / 48 | 105 / 420 | 14 / 56 | 524 |
| 3 | test | 19 / 76 | 105 / 420 | 14 / 56 | 552 |
| 4 | train | 47 / 188 | 491 / 1964 | 68 / 272 | 2424 |
| 4 | val | 18 / 72 | 105 / 420 | 14 / 56 | 548 |
| 4 | test | 12 / 48 | 105 / 420 | 14 / 56 | 524 |

Main rows: 3,496 per seed. Email sizes vary by seed because near-duplicate components move as units.

| Split seed | val_id | raw_clean | ambiguous | long |
|---|---|---|---|---|
| 0 | 264 | 265 | 265 | 56 |
| 1 | 278 | 271 | 271 | 56 |
| 2 | 272 | 267 | 267 | 56 |
| 3 | 262 | 269 | 269 | 56 |
| 4 | 274 | 268 | 268 | 56 |

Trimmed docs (every seed): CodeQA 31 of 96, TableQA 107 of 701; EmailQA none.

Benign pool (`data/benign/benign_v3.json`, items per register): email 70 train_val / 35 test / 20 ambiguous; table
110 / 55 / 20; code 61 / 30 / 20. The builder splits train_val into train / val (same on every seed): email 52 / 18,
table 82 / 28, code 46 / 15. No item exceeds its register's span cap.

## Benign-pool review
Planned (v3_design §3): the author reviews the drafted sheet. Actual: all 421 items (361 benign, 60 ambiguous)
reviewed by one external human reviewer, not the author and not a project member, against the criterion "addressed
to the human reader; following the item must not change what an assistant should do for any intent in that register"
(`docs/deviations.md`, 2026-11-11). Outcome: 357 keep, 4 edit, 0 benign drops; all 60 ambiguous items confirmed as
model-directed and kept in the ambiguous eval slice only. Sheet: `review/benign_pool_review.csv`; applied by
`scripts/apply_review.py`. Train/test assignment of benign items was fixed in the draft, before the review; the
seeded stratified draw of v3_design §3 step 4 (`scripts/freeze_benign_v3.py`) was not used.

## Gates (F024)
Content-free probes trained on each seed's train split, evaluated on its test split. Pass = pooled macro-F1 ≤ 0.60 on
every seed and per-task mean over seeds ≤ 0.60 (decision R2). Run `results/v3/artifact_baselines_frozen/20261111-012417_seed0`.

| Probe | Pooled, seeds 0–4 | EmailQA mean | TableQA mean | CodeQA mean | Gate |
|---|---|---|---|---|---|
| position_oracle | 0.4552–0.5096 | 0.5433 | 0.4816 | 0.4301 | PASS |
| first_para_rule | 0.4443–0.4652 | 0.4713 | 0.4265 | 0.4823 | PASS |
| prefix30_masked_tfidf_lr | 0.3333 | 0.3333 | 0.3333 | 0.3333 | PASS |
| suffix30_masked_tfidf_lr | 0.3333 | 0.3333 | 0.3333 | 0.3333 | PASS |
| insertion_site_tfidf_lr | 0.4961–0.5105 | 0.4720 | 0.5075 | 0.4638 | PASS |
| length_lr | 0.4985–0.5062 | 0.3333 | 0.5024 | 0.3827 | PASS |
| span_form_you_lr | 0.5342–0.5620 | 0.5101 | 0.5528 | 0.5277 | PASS |
| second_person_lr | 0.4732–0.4942 | 0.5179 | 0.4935 | 0.3232 | PASS |
| boundary_kind_lr | 0.4505–0.4954 | 0.5101 | 0.4711 | 0.4256 | PASS |
| prefix30_tfidf_lr (diag) | 0.5775–0.6259 | 0.5944 | 0.5993 | 0.6263 | diag |
| suffix30_tfidf_lr (diag) | 0.5798–0.6122 | 0.5557 | 0.5990 | 0.6306 | diag |
| span_content_tfidf (diag) | 0.8021–0.8257 | 0.7953 | 0.7980 | 0.9785 | diag |
| span_form_lr (diag) | 0.5401–0.5714 | 0.5040 | 0.5690 | 0.5456 | diag |

## Integrity (F025)
`scripts/v3_integrity.py` re-reads the written CSVs. All 17 checks PASS on every seed: doc_id, text, near-duplicate
component, attack string, benign item and donor text disjoint across splits; train/val attack categories disjoint;
attack file by split; donor docs from the same task and split; ambiguous items never used as clean A; 1:1 balance;
4 rows (2 injected) per doc; eval-slice membership; long docs not in main; ≤ 512 tokens (re-tokenized) and stored
counts match; every span recoverable. Run `results/v3/integrity_frozen/20261111-014556_seed0`.

The frozen build is byte-identical to the pre-freeze scratch build of the same pool, and its gate and integrity
results are identical (`results/v3/artifact_baselines_reviewed/`, `results/v3/integrity_reviewed/`).

## Known risks (v3_design §9) and how results report them
- **R3. Templated table clean B.** TableQA clean B is rendered from table rows, not natural text. Report results
  with and without clean B (clean A vs injected only).
- **R4. Unnatural insertion.** Inserting any span into tables, or newlines into newline-free emails, is unnatural.
  It affects all row types equally, but `raw_clean` is trivially separable from inserted rows; raw-clean results are
  eval-only and reported separately as a false-positive rate.
- **R5. `Language Translation`** appears in both the train and test text attack files (15–17 test injected rows per
  seed, of 262–276). Report test results with and without it.
- **R6. Overlap with V2 training docs.** V3 splits are new, so V3 test docs can be V2 training docs. V2 checkpoints
  are evaluated on V3 only on docs that were in V2 test, or not at all.
- **R7. Small cells.** EmailQA test has 12–19 docs and CodeQA test 14 docs per seed; 31 of 96 code docs are trimmed.
  Report per-task CIs (doc-cluster bootstrap), means over the 5 split seeds, and results with and without trimmed docs.
- **Span-content ceiling on CodeQA.** TF-IDF+LR on the span text alone reaches a per-task mean macro-F1 of 0.9785 on
  CodeQA (0.80 on the other tasks). Code attacks and benign code items stay lexically separable (attack-indicative
  tokens include `server`, `smtp`, `psutil`, `gmail`). This is content, not a construction artifact, but CodeQA
  results measure a lexically easy task and should be read with that ceiling in view.
- Also open (v3_design §9): R2 lexical second-person cue (gated, PASS), R8 noisy category-held-out val, R9 the intent
  is not needed for the label (intent-conditioning is tested by the pairs, C3).
