# Findings log (numbers only)

A running log of what the committed results establish so far. Each entry `F<nnn>` opens with a provenance block
(script, commit, seeds, data version, eval set, threshold protocol, date, source runs or files). Every number is a
macro from `paper/numbers.tex` (written by `scripts/make_tables.py`), or from `paper/legacy_numbers.tex` for entries
without full provenance, shown next to its value. `[lo, hi]` is the 95% CI from `\<Name>CiLo` / `\<Name>CiHi`.

Statuses: **full** (clean committed run) and **recomputed-match** (recomputed by committed code from saved frozen
scores, equal to the frozen value) are the only ones that reach `paper/numbers.tex`. **linked-dirty** (produced from
a run on an uncommitted tree), **legacy-unverified** (cannot be recomputed), **contaminated** (a choice was made with
test or transfer data in view) and **pending** (awaiting a scheduled run) are not headline numbers. IDs are never reused.

Regenerate with `python scripts/make_tables.py && python scripts/make_findings.py`. Edit `docs/findings.src.md`,
not `docs/findings.md`. Deviations from the pre-registered protocol: `docs/deviations.md`.

## Standing caveats

- The V1 audit numbers from the Phase 2 report have no code, data, or results in the repository or its git history.
  They are removed and appear nowhere in this log or in `paper/`.
- Pairs v0 (`data/pairs/pairs_v0.csv`, 50 pairs) are LLM-drafted, not human-validated (pilot), on V2 test docs, and have not been
  independently labeled. No pair result is evidence yet, and none is listed here.
- The LLM judge prompt v1 (`score_detectors.JUDGE_SYSTEM`) was committed in the same commit as its first scores, so
  freezing before results cannot be shown; V2 judge results carry this caveat. V3 will use a new judge prompt committed
  separately before any scoring.

## F001. V2 TF-IDF+LR headline table (Phase 2 Table III)

```finding
id: F001
status: recomputed-match
script: run_experiments.py (V2 predictions) -> scripts/make_tables.py (recompute)
commit: a4a74a1
seeds: n/a (TF-IDF+LR is deterministic)
data_version: v2-frozen
eval_set: v2_test
threshold_protocol:
- default tau=0.50 (C=1)
- tau=0.71 and C=100 tuned on V2 val
- 'Recall@FPR: oracle'
date: 2026-10-19 (tag v2-frozen)
source:
- results/v2_std/v2_test/tfidf_c1.csv
- results/v2_std/v2_test/tfidf_c100_tuned.csv
checked_against: results/tfidf_lr_results.json (every point value and CI bound, tolerance 1e-4)
cluster: doc_id
macros:
- VTwoTfidfCOne*
```

TF-IDF+LR, C=1, τ=0.5: macro-F1 {{VTwoTfidfCOneFOneMacro|ci}}, ROC-AUC {{VTwoTfidfCOneRocAuc|ci}}, Recall@5%FPR (oracle) {{VTwoTfidfCOneRecFive}}.
Tuned (C=100, τ=0.71, chosen on V2 val): macro-F1 {{VTwoTfidfCOneZeroZeroTunedFOneMacro|ci}}, ROC-AUC {{VTwoTfidfCOneZeroZeroTunedRocAuc|ci}}.
Full table: `paper/tables/v2_table3_main.tex`.

## F002. V2 ablations: majority class and fused TF-IDF (Phase 2 Table IV)

```finding
id: F002
status: recomputed-match
script: run_experiments.py (V2 predictions) -> scripts/make_tables.py (recompute)
commit: a4a74a1
seeds: n/a (deterministic)
data_version: v2-frozen
eval_set: v2_test
threshold_protocol: default tau=0.50
date: 2026-10-19 (tag v2-frozen)
source: results/v2_std/v2_test/tfidf_c1.csv
checked_against: results/ablations.json (majority accuracy and macro-F1); fused = F001's checked values
macros:
- VTwoAblMajority*
- VTwoAblFused*
```

Majority class macro-F1 {{VTwoAblMajorityFOneMacro}}; fused intent + context macro-F1 {{VTwoAblFusedFOneMacro}}.

## F003. V2 ablations: length-only, intent-only, context-only

```finding
id: F003
status: legacy-unverified
script: run_experiments.py
commit: a4a74a1
seeds: n/a (deterministic)
data_version: v2-frozen
eval_set: v2_test
threshold_protocol: default tau=0.50
date: 2026-10-19 (tag v2-frozen)
source: results/ablations.json
macros:
- VTwoAblLengthOnly*
- VTwoAblIntentOnly*
- VTwoAblContextOnly*
reason: read from results/ablations.json; no per-row predictions were saved, so they cannot be recomputed or bootstrapped.
```

Length only macro-F1 {{VTwoAblLengthOnlyFOneMacro}}; intent only {{VTwoAblIntentOnlyFOneMacro}}; context only {{VTwoAblContextOnlyFOneMacro}}.

## F004. V2 TF-IDF+LR per task (Phase 2 Table V)

```finding
id: F004
status: recomputed-match
script: run_experiments.py (V2 predictions) -> scripts/make_tables.py (recompute)
commit: a4a74a1
seeds: n/a (deterministic)
data_version: v2-frozen
eval_set: v2_test (per task)
threshold_protocol: default tau=0.50
date: 2026-10-19 (tag v2-frozen)
source: results/v2_std/v2_test/tfidf_c1.csv
checked_against: results/tfidf_lr_results.json per_task_breakdown (point values and CI bounds)
cluster: doc_id
macros:
- VTwoTask*
```

Macro-F1: EmailQA {{VTwoTaskEmailFOneMacro|ci}} ({{VTwoTaskEmailNdocs}} docs), TableQA {{VTwoTaskTableFOneMacro|ci}} ({{VTwoTaskTableNdocs}} docs), CodeQA {{VTwoTaskCodeFOneMacro|ci}} ({{VTwoTaskCodeNdocs}} docs).

## F005. V2 TF-IDF+LR leave-one-task-out (Phase 2 Table VI)

```finding
id: F005
status: contaminated
script: run_experiments.py (V2 predictions) -> scripts/make_tables.py (recompute)
commit: a4a74a1
seeds: n/a (deterministic)
data_version: v2-frozen
eval_set: v2_loto_test_{email,table,code}
threshold_protocol: default tau=0.50; C=100 from the full-val tuning
date: 2026-10-19 (tag v2-frozen)
source:
- results/v2_std/v2_loto_test_email/tfidf_c100_loto.csv
- results/v2_std/v2_loto_test_table/tfidf_c100_loto.csv
- results/v2_std/v2_loto_test_code/tfidf_c100_loto.csv
checked_against: results/leave_one_task_out.json (values match)
cluster: doc_id
macros:
- VTwoLoto*
reason: C was chosen on the full V2 val split, which includes val docs of the held-out (transfer) task, so the transfer task was in view during model selection. Values do match the frozen file.
```

Held-out macro-F1: EmailQA {{VTwoLotoEmailFOneMacro|ci}}, TableQA {{VTwoLotoTableFOneMacro|ci}}, CodeQA {{VTwoLotoCodeFOneMacro|ci}}.

## F006. V2 artifact-only baselines

```finding
id: F006
status: full
script: scripts/artifact_baselines.py --dataset v2
commit: a537610
seeds: n/a (deterministic probes); bootstrap seed 42
data_version: v2
eval_set: v2_test
threshold_protocol: default tau=0.50 (learned probes); fixed rules
date: '2026-10-21'
source: results/v2_probes/artifact_baselines/20261021-155152_seed42
cluster: doc_id
macros:
- VTwoArtifact*
```

None of these probes uses the user query; the learned ones are trained on V2 train only. Reference: the TF-IDF+LR detector (C=1, τ=0.5), F001.

| Probe | What it uses | Macro-F1 | ROC-AUC |
|---|---|---|---|
| Position oracle | the `insertion_mode` field only (start → injected) | {{VTwoArtifactPositionOracleFOneMacro|ci}} | {{VTwoArtifactPositionOracleRocAuc|ci}} |
| First-paragraph rule | first `\n\n` paragraph ≤ 30 words → injected; untrained | {{VTwoArtifactFirstParaRuleFOneMacro|ci}} | {{VTwoArtifactFirstParaRuleRocAuc|ci}} |
| Prefix-30 TF-IDF+LR | first 30 words of the context | {{VTwoArtifactPrefixThreeZeroTfidfLrFOneMacro|ci}} | {{VTwoArtifactPrefixThreeZeroTfidfLrRocAuc|ci}} |
| Suffix-30 TF-IDF+LR | last 30 words of the context | {{VTwoArtifactSuffixThreeZeroTfidfLrFOneMacro|ci}} | {{VTwoArtifactSuffixThreeZeroTfidfLrRocAuc|ci}} |
| Span-form LR | 9 content-free features of the inserted span only (216 rows with an inserted span) | {{VTwoArtifactSpanFormLrFOneMacro|ci}} | {{VTwoArtifactSpanFormLrRocAuc|ci}} |

- The position field alone, and the untrained first-paragraph rule, each reach a macro-F1 whose CI lies below
  the TF-IDF detector's point value but above 0.5.
- The first 30 words carry more label signal than the last 30 words: the prefix and suffix CIs do not overlap.
- Content-free surface features of the inserted span alone give a macro-F1 whose CI lies above 0.5.

## F007. V2 position probes: reliance, truncation control, seed dependence, row groups

```finding
id: F007
status: linked-dirty
script: scripts/score_probes.py -> scripts/probe_report.py, scripts/truncation_control.py
commit: 2a370a8
seeds: DeBERTa 42, 123, 999; others zero-shot or deterministic; bootstrap seed 42
data_version: v2_probes (data/probes_v2/) + v2-frozen test
eval_set: p1-p4 probe sets, v2_test
threshold_protocol: default tau=0.50
date: 2026-10-21 (reports); 2026-10-19 (scores)
source:
- results/v2_probes/probe_report/20261021-125648_seed42
- results/v2_probes/truncation_control/20261021-125607_seed42
upstream:
- results/v2_probes/score_deberta_s42/20261019-183817_seed42
- results/v2_probes/score_deberta_s123/20261019-184210_seed123
- results/v2_probes/score_deberta_s999/20261019-184348_seed999
- results/v2_probes/score_judge_qwen2.5_7b/20261019-184624_seed42
- results/v2_probes/score_judge_qwen2.5_7b/20261021-123213_seed42
- results/v2_probes/score_piguard/20261019-184536_seed42
- results/v2_probes/score_protectai/20261019-184432_seed42
- results/v2_probes/score_tfidf_c1/20261019-180346_seed42
cluster: doc_id
macros:
- VTwoProbe*
- VTwoTrunc*
- VTwoSeedvar*
- VTwoGroups*
reason: the 2026-10-19 scoring runs were made on a dirty tree (518b572-dirty, c61e423-dirty), so their commit does not identify the scoring code; jobs/rescore_v2_probes.sh re-scores them from a clean commit.
  The TF-IDF re-score (20261028-175939) reproduces the dirty run's probabilities exactly.
```

Part of claim C1. Re-measured from clean scores before any of these numbers is used.

### DeBERTa position reliance on V2 (probes P1–P3)

Paired within-doc contrasts. "Full" uses all probe rows. "Matched" keeps only sources whose inserted span is
fully inside DeBERTa's {{VTwoTruncWindowTokens}}-token window at all three positions (`scripts/truncation_control.py`).

**P3 (attack moved), recall at end − recall at start**

| DeBERTa | Full | Matched | P3 end rows truncated |
|---|---|---|---|
| s42 | {{VTwoTruncDebertaSFourTwoPThreeendminusstartFull|ci}} | {{VTwoTruncDebertaSFourTwoPThreeendminusstartMatched|ci}} | {{VTwoTruncDebertaSFourTwoPThreeendtruncated}} |
| s123 | {{VTwoTruncDebertaSOneTwoThreePThreeendminusstartFull|ci}} | {{VTwoTruncDebertaSOneTwoThreePThreeendminusstartMatched|ci}} | {{VTwoTruncDebertaSOneTwoThreePThreeendtruncated}} |
| s999 | {{VTwoTruncDebertaSNineNineNinePThreeendminusstartFull|ci}} | {{VTwoTruncDebertaSNineNineNinePThreeendminusstartMatched|ci}} | {{VTwoTruncDebertaSNineNineNinePThreeendtruncated}} |
| mean | {{VTwoTruncDebertaMeanPThreeendminusstartFull|ci}} | {{VTwoTruncDebertaMeanPThreeendminusstartMatched|ci}} | {{VTwoTruncDebertaMeanPThreeendtruncated}} |

- On the matched subset the end − start difference is negative for every seed and the mean, and every CI excludes 0.
- The matched difference is smaller in magnitude than the full one for every seed and the mean.

Matched P3 recall by position (seed mean): start {{VTwoTruncDebertaMeanPThreeStart|ci}},
middle {{VTwoTruncDebertaMeanPThreeMiddle|ci}}, end {{VTwoTruncDebertaMeanPThreeEnd|ci}}.

**P2 (benign imperative moved), flag rate at start − flag rate at middle**

| DeBERTa | Full | Matched |
|---|---|---|
| s42 | {{VTwoProbeDebertaSFourTwoPTwoStartMinusMiddle|ci}} | {{VTwoTruncDebertaSFourTwoPTwostartminusmiddleMatched|ci}} |
| s123 | {{VTwoProbeDebertaSOneTwoThreePTwoStartMinusMiddle|ci}} | {{VTwoTruncDebertaSOneTwoThreePTwostartminusmiddleMatched|ci}} |
| s999 | {{VTwoProbeDebertaSNineNineNinePTwoStartMinusMiddle|ci}} | {{VTwoTruncDebertaSNineNineNinePTwostartminusmiddleMatched|ci}} |
| mean | {{VTwoProbeDebertaMeanPTwoStartMinusMiddle|ci}} | {{VTwoTruncDebertaMeanPTwostartminusmiddleMatched|ci}} |

- Moving the same benign imperative to the start raises the flag rate for every seed; every CI excludes 0,
  full and matched.

**P1 (neutral sentence inserted), flag rate at start − flag rate at middle**

| DeBERTa | Full | Matched |
|---|---|---|
| s42 | {{VTwoProbeDebertaSFourTwoPOneStartMinusMiddle|ci}} | {{VTwoTruncDebertaSFourTwoPOnestartminusmiddleMatched|ci}} |
| s123 | {{VTwoProbeDebertaSOneTwoThreePOneStartMinusMiddle|ci}} | {{VTwoTruncDebertaSOneTwoThreePOnestartminusmiddleMatched|ci}} |
| s999 | {{VTwoProbeDebertaSNineNineNinePOneStartMinusMiddle|ci}} | {{VTwoTruncDebertaSNineNineNinePOnestartminusmiddleMatched|ci}} |
| mean | {{VTwoProbeDebertaMeanPOneStartMinusMiddle|ci}} | {{VTwoTruncDebertaMeanPOnestartminusmiddleMatched|ci}} |

- The P1 difference is {{VTwoProbeDebertaSFourTwoPOneStartMinusMiddle}} for s42 and {{VTwoProbeDebertaSOneTwoThreePOneStartMinusMiddle}} for s123. For s999 it is positive and its CI excludes 0.

### Seed dependence of shortcut use (DeBERTa)

| Seed | V2 test macro-F1 | P1 start flag rate | P2 start − middle |
|---|---|---|---|
| s42 | {{VTwoSeedvarDebertaSFourTwoFOneMacro|ci}} | {{VTwoSeedvarDebertaSFourTwoPOnestart|ci}} | {{VTwoSeedvarDebertaSFourTwoPTwostartminusmiddle|ci}} |
| s123 | {{VTwoSeedvarDebertaSOneTwoThreeFOneMacro|ci}} | {{VTwoSeedvarDebertaSOneTwoThreePOnestart|ci}} | {{VTwoSeedvarDebertaSOneTwoThreePTwostartminusmiddle|ci}} |
| s999 | {{VTwoSeedvarDebertaSNineNineNineFOneMacro|ci}} | {{VTwoSeedvarDebertaSNineNineNinePOnestart|ci}} | {{VTwoSeedvarDebertaSNineNineNinePTwostartminusmiddle|ci}} |
| mean | {{VTwoSeedvarDebertaMeanFOneMacro|ci}} | {{VTwoSeedvarDebertaMeanPOnestart|ci}} | {{VTwoSeedvarDebertaMeanPTwostartminusmiddle|ci}} |

- The three seeds' V2 test macro-F1 CIs overlap.
- s999's P1 start flag rate CI does not overlap those of s42 and s123.
- s999's P2 start − middle CI does not overlap those of s42 and s123.

### Qwen2.5-7B judge (frozen prompt, no truncation)

The judge receives the full chat-template prompt (no truncation); no probe row exceeds its context, so the
truncated fraction is {{VTwoTruncJudgeQwenTwoFiveSevenbPThreeendtruncated}} and full = matched.

| | Value |
|---|---|
| P3 recall, start | {{VTwoProbeJudgeQwenTwoFiveSevenbPThreeStart|ci}} |
| P3 recall, middle | {{VTwoProbeJudgeQwenTwoFiveSevenbPThreeMiddle|ci}} |
| P3 recall, end | {{VTwoProbeJudgeQwenTwoFiveSevenbPThreeEnd|ci}} |
| P3 end − start | {{VTwoProbeJudgeQwenTwoFiveSevenbPThreeEndMinusStart|ci}} |
| P2 start − middle | {{VTwoProbeJudgeQwenTwoFiveSevenbPTwoStartMinusMiddle|ci}} |
| P1 start − middle | {{VTwoProbeJudgeQwenTwoFiveSevenbPOneStartMinusMiddle|ci}} |
| V2 test macro-F1 | {{VTwoProbeJudgeQwenTwoFiveSevenbVTwotestFOneMacro|ci}} |
| V2 test recall (injected rows) | {{VTwoGroupsJudgeQwenTwoFiveSevenbRecall|ci}} |
| Flag rate, V2 raw clean rows (n = {{VTwoGroupsJudgeQwenTwoFiveSevenbRawcleanflagN}}) | {{VTwoGroupsJudgeQwenTwoFiveSevenbRawcleanflag|ci}} |
| Flag rate, V2 benign-imperative rows (n = {{VTwoGroupsJudgeQwenTwoFiveSevenbBenignflagN}}) | {{VTwoGroupsJudgeQwenTwoFiveSevenbBenignflag|ci}} |
| Flag rate, P4 whitespace-only (n = {{VTwoGroupsJudgeQwenTwoFiveSevenbPfourflagN}}) | {{VTwoGroupsJudgeQwenTwoFiveSevenbPfourflag|ci}} |
| P4 − raw clean (paired docs: {{VTwoGroupsJudgeQwenTwoFiveSevenbPfourminusrawcleanNpairs}}) | {{VTwoGroupsJudgeQwenTwoFiveSevenbPfourminusrawclean|ci}} |

- The judge's P3 recall falls from start to end; the end − start CI excludes 0.
- P2 and P1 start − middle CIs include 0.
- The V2 test macro-F1 CI includes 0.5.
- P4 − raw clean CI includes 0.

### Public detectors on V2 test

| Detector | Recall (injected rows) | Flag rate: benign imperative | Flag rate: raw clean |
|---|---|---|---|
| ProtectAI v2 | {{VTwoGroupsProtectaiRecall|ci}} | {{VTwoGroupsProtectaiBenignflag|ci}} | {{VTwoGroupsProtectaiRawcleanflag|ci}} |
| PIGuard | {{VTwoGroupsPiguardRecall|ci}} | {{VTwoGroupsPiguardBenignflag|ci}} | {{VTwoGroupsPiguardRawcleanflag|ci}} |

- ProtectAI's recall point estimate is below its flag rate on both clean row groups.
- PIGuard's raw clean flag rate is {{VTwoGroupsPiguardRawcleanflag}}; its benign-imperative flag-rate CI excludes 0.

## F008. summary.json Recall@5%FPR bug on pairs v0

```finding
id: F008
status: recomputed-match
script: score_detectors.py summarize (stored value) -> scripts/make_tables.py
commit: a4a74a1
seeds: n/a (zero-shot)
data_version: v2-frozen
eval_set: pairs_v0
threshold_protocol: 'Recall@5%FPR: oracle'
date: 2026-10-19 (tag v2-frozen)
source:
- results/scores/summary.json
- results/v2_std/pairs_v0/piguard.csv
- results/v2_std/pairs_v0/protectai.csv
checked_against: the stored value is reproduced from the saved scores with roc_curve(drop_intermediate=True)
macros:
- VTwoPairsbug*
```

`results/scores/summary.json` Recall@5%FPR on pairs v0 is wrong for PIGuard and ProtectAI
  (`score_detectors.recall_at_fpr` uses `roc_curve(drop_intermediate=True)`):
  PIGuard {{VTwoPairsbugPiguardSummaryjson}} in summary.json vs {{VTwoPairsbugPiguardMetrics}} from `src/metrics.py`;
  ProtectAI {{VTwoPairsbugProtectaiSummaryjson}} vs {{VTwoPairsbugProtectaiMetrics}}.
  Recall@FPR for the paper comes from `results/v2_std/` via `src/metrics.py`.

## F009. V2 test: headline metrics, oracle threshold protocol

```finding
id: F009
status: full
script: scripts/v2_tpr_metrics.py
commit: a59aae4
seeds: DeBERTa 42, 123, 999; others deterministic or zero-shot; bootstrap B=1000 seed 42
data_version: 'v2-frozen+external (V2 test: tag v2-frozen)'
eval_set: v2_test
threshold_protocol:
- oracle (threshold chosen on the clean rows of V2 test)
- default threshold for FPR@default and macro-F1
date: '2026-10-28'
source: results/v2_metrics/tpr_protocols/20261028-175945_seed42
cluster: doc_id
macros:
- VTwoTpr*VTwoTest*
```

On the artifact-laden V2 benchmark (DeBERTa V2 numbers are not headline results). TPR@1%FPR needs at least 500 clean rows and V2 test has {{VTwoTprTfidfCOneVTwoTestNclean}}, so it is n/a.

| Model | n clean | AUROC | FPR at default threshold (clean) | TPR@1%FPR | TPR@5%FPR | Macro-F1 (secondary) |
|---|---|---|---|---|---|---|
| tfidf_c1 | {{VTwoTprTfidfCOneVTwoTestNclean}} | {{VTwoTprTfidfCOneVTwoTestAuroc|ci}} | {{VTwoTprTfidfCOneVTwoTestFprdefault|ci}} | {{VTwoTprTfidfCOneVTwoTestOracleOne|ci}} | {{VTwoTprTfidfCOneVTwoTestOracleFive|ci}} | {{VTwoTprTfidfCOneVTwoTestFOnemacro|ci}} |
| tfidf_c100_tuned | {{VTwoTprTfidfCOneZeroZeroTunedVTwoTestNclean}} | {{VTwoTprTfidfCOneZeroZeroTunedVTwoTestAuroc|ci}} | {{VTwoTprTfidfCOneZeroZeroTunedVTwoTestFprdefault|ci}} | {{VTwoTprTfidfCOneZeroZeroTunedVTwoTestOracleOne|ci}} | {{VTwoTprTfidfCOneZeroZeroTunedVTwoTestOracleFive|ci}} | {{VTwoTprTfidfCOneZeroZeroTunedVTwoTestFOnemacro|ci}} |
| deberta_s42 | {{VTwoTprDebertaSFourTwoVTwoTestNclean}} | {{VTwoTprDebertaSFourTwoVTwoTestAuroc|ci}} | {{VTwoTprDebertaSFourTwoVTwoTestFprdefault|ci}} | {{VTwoTprDebertaSFourTwoVTwoTestOracleOne|ci}} | {{VTwoTprDebertaSFourTwoVTwoTestOracleFive|ci}} | {{VTwoTprDebertaSFourTwoVTwoTestFOnemacro|ci}} |
| deberta_s123 | {{VTwoTprDebertaSOneTwoThreeVTwoTestNclean}} | {{VTwoTprDebertaSOneTwoThreeVTwoTestAuroc|ci}} | {{VTwoTprDebertaSOneTwoThreeVTwoTestFprdefault|ci}} | {{VTwoTprDebertaSOneTwoThreeVTwoTestOracleOne|ci}} | {{VTwoTprDebertaSOneTwoThreeVTwoTestOracleFive|ci}} | {{VTwoTprDebertaSOneTwoThreeVTwoTestFOnemacro|ci}} |
| deberta_s999 | {{VTwoTprDebertaSNineNineNineVTwoTestNclean}} | {{VTwoTprDebertaSNineNineNineVTwoTestAuroc|ci}} | {{VTwoTprDebertaSNineNineNineVTwoTestFprdefault|ci}} | {{VTwoTprDebertaSNineNineNineVTwoTestOracleOne|ci}} | {{VTwoTprDebertaSNineNineNineVTwoTestOracleFive|ci}} | {{VTwoTprDebertaSNineNineNineVTwoTestFOnemacro|ci}} |
| protectai | {{VTwoTprProtectaiVTwoTestNclean}} | {{VTwoTprProtectaiVTwoTestAuroc|ci}} | {{VTwoTprProtectaiVTwoTestFprdefault|ci}} | {{VTwoTprProtectaiVTwoTestOracleOne|ci}} | {{VTwoTprProtectaiVTwoTestOracleFive|ci}} | {{VTwoTprProtectaiVTwoTestFOnemacro|ci}} |
| piguard | {{VTwoTprPiguardVTwoTestNclean}} | {{VTwoTprPiguardVTwoTestAuroc|ci}} | {{VTwoTprPiguardVTwoTestFprdefault|ci}} | {{VTwoTprPiguardVTwoTestOracleOne|ci}} | {{VTwoTprPiguardVTwoTestOracleFive|ci}} | {{VTwoTprPiguardVTwoTestFOnemacro|ci}} |
| judge_qwen2.5_7b | {{VTwoTprJudgeQwenTwoFiveSevenbVTwoTestNclean}} | {{VTwoTprJudgeQwenTwoFiveSevenbVTwoTestAuroc|ci}} | {{VTwoTprJudgeQwenTwoFiveSevenbVTwoTestFprdefault|ci}} | {{VTwoTprJudgeQwenTwoFiveSevenbVTwoTestOracleOne|ci}} | {{VTwoTprJudgeQwenTwoFiveSevenbVTwoTestOracleFive|ci}} | {{VTwoTprJudgeQwenTwoFiveSevenbVTwoTestFOnemacro|ci}} |

DeBERTa, mean ± SD over seeds 42/123/999: AUROC {{VTwoTprDebertaMeanVTwoTestRocAucMean}} ± {{VTwoTprDebertaMeanVTwoTestRocAucSd}}; FPR at default threshold {{VTwoTprDebertaMeanVTwoTestFprDefaultCleanMean}} ± {{VTwoTprDebertaMeanVTwoTestFprDefaultCleanSd}}; macro-F1 {{VTwoTprDebertaMeanVTwoTestFOneMacroMean}} ± {{VTwoTprDebertaMeanVTwoTestFOneMacroSd}}; TPR@5%FPR {{VTwoTprDebertaMeanVTwoTestTprOracleFivepctFprMean}} ± {{VTwoTprDebertaMeanVTwoTestTprOracleFivepctFprSd}}.

## F010. InjecAgent: headline metrics, oracle threshold protocol

```finding
id: F010
status: full
script: scripts/v2_tpr_metrics.py
commit: a59aae4
seeds: DeBERTa 42, 123, 999; others deterministic or zero-shot; bootstrap B=1000 seed 42
data_version: 'v2-frozen+external: InjecAgent@f19c9f2 + build bab750e (score_detectors.py build)'
eval_set: injecagent
threshold_protocol:
- oracle
- default threshold for FPR@default and macro-F1
date: '2026-10-28'
source: results/v2_metrics/tpr_protocols/20261028-175945_seed42
cluster: user case (user tool + user instruction)
macros:
- VTwoTpr*Injecagent*
```

Clean rows fill the same tool-response templates with benign text, so rows of one user case are not independent; the bootstrap resamples the {{VTwoTprTfidfCOneInjecagentNclusters}} user cases.

| Model | n clean | AUROC | FPR at default threshold (clean) | TPR@1%FPR | TPR@5%FPR | Macro-F1 (secondary) |
|---|---|---|---|---|---|---|
| tfidf_c1 | {{VTwoTprTfidfCOneInjecagentNclean}} | {{VTwoTprTfidfCOneInjecagentAuroc|ci}} | {{VTwoTprTfidfCOneInjecagentFprdefault|ci}} | {{VTwoTprTfidfCOneInjecagentOracleOne|ci}} | {{VTwoTprTfidfCOneInjecagentOracleFive|ci}} | {{VTwoTprTfidfCOneInjecagentFOnemacro|ci}} |
| deberta_s42 | {{VTwoTprDebertaSFourTwoInjecagentNclean}} | {{VTwoTprDebertaSFourTwoInjecagentAuroc|ci}} | {{VTwoTprDebertaSFourTwoInjecagentFprdefault|ci}} | {{VTwoTprDebertaSFourTwoInjecagentOracleOne|ci}} | {{VTwoTprDebertaSFourTwoInjecagentOracleFive|ci}} | {{VTwoTprDebertaSFourTwoInjecagentFOnemacro|ci}} |
| deberta_s123 | {{VTwoTprDebertaSOneTwoThreeInjecagentNclean}} | {{VTwoTprDebertaSOneTwoThreeInjecagentAuroc|ci}} | {{VTwoTprDebertaSOneTwoThreeInjecagentFprdefault|ci}} | {{VTwoTprDebertaSOneTwoThreeInjecagentOracleOne|ci}} | {{VTwoTprDebertaSOneTwoThreeInjecagentOracleFive|ci}} | {{VTwoTprDebertaSOneTwoThreeInjecagentFOnemacro|ci}} |
| deberta_s999 | {{VTwoTprDebertaSNineNineNineInjecagentNclean}} | {{VTwoTprDebertaSNineNineNineInjecagentAuroc|ci}} | {{VTwoTprDebertaSNineNineNineInjecagentFprdefault|ci}} | {{VTwoTprDebertaSNineNineNineInjecagentOracleOne|ci}} | {{VTwoTprDebertaSNineNineNineInjecagentOracleFive|ci}} | {{VTwoTprDebertaSNineNineNineInjecagentFOnemacro|ci}} |
| protectai | {{VTwoTprProtectaiInjecagentNclean}} | {{VTwoTprProtectaiInjecagentAuroc|ci}} | {{VTwoTprProtectaiInjecagentFprdefault|ci}} | {{VTwoTprProtectaiInjecagentOracleOne|ci}} | {{VTwoTprProtectaiInjecagentOracleFive|ci}} | {{VTwoTprProtectaiInjecagentFOnemacro|ci}} |
| piguard | {{VTwoTprPiguardInjecagentNclean}} | {{VTwoTprPiguardInjecagentAuroc|ci}} | {{VTwoTprPiguardInjecagentFprdefault|ci}} | {{VTwoTprPiguardInjecagentOracleOne|ci}} | {{VTwoTprPiguardInjecagentOracleFive|ci}} | {{VTwoTprPiguardInjecagentFOnemacro|ci}} |
| judge_qwen2.5_7b | {{VTwoTprJudgeQwenTwoFiveSevenbInjecagentNclean}} | {{VTwoTprJudgeQwenTwoFiveSevenbInjecagentAuroc|ci}} | {{VTwoTprJudgeQwenTwoFiveSevenbInjecagentFprdefault|ci}} | {{VTwoTprJudgeQwenTwoFiveSevenbInjecagentOracleOne|ci}} | {{VTwoTprJudgeQwenTwoFiveSevenbInjecagentOracleFive|ci}} | {{VTwoTprJudgeQwenTwoFiveSevenbInjecagentFOnemacro|ci}} |

DeBERTa, mean ± SD over seeds 42/123/999: AUROC {{VTwoTprDebertaMeanInjecagentRocAucMean}} ± {{VTwoTprDebertaMeanInjecagentRocAucSd}}; FPR at default threshold {{VTwoTprDebertaMeanInjecagentFprDefaultCleanMean}} ± {{VTwoTprDebertaMeanInjecagentFprDefaultCleanSd}}; macro-F1 {{VTwoTprDebertaMeanInjecagentFOneMacroMean}} ± {{VTwoTprDebertaMeanInjecagentFOneMacroSd}}; TPR@5%FPR {{VTwoTprDebertaMeanInjecagentTprOracleFivepctFprMean}} ± {{VTwoTprDebertaMeanInjecagentTprOracleFivepctFprSd}}.

## F011. deepset prompt-injections: headline metrics

```finding
id: F011
status: full
script: scripts/v2_tpr_metrics.py
commit: a59aae4
seeds: DeBERTa 42, 123, 999; others deterministic or zero-shot; bootstrap B=1000 seed 42
data_version: 'v2-frozen+external: hf:deepset/prompt-injections@4f61ecb (test split) + build bab750e'
eval_set: deepset
threshold_protocol:
- oracle
- default threshold for FPR@default and macro-F1
date: '2026-10-28'
source: results/v2_metrics/tpr_protocols/20261028-175945_seed42
cluster: row (independent prompts)
macros:
- VTwoTpr*Deepset*
```

{{VTwoTprTfidfCOneDeepsetNclean}} clean rows: both TPR@FPR values are n/a. The build did not pin a dataset revision; 4f61ecb is the only revision in the local HF cache.

| Model | n clean | AUROC | FPR at default threshold (clean) | TPR@1%FPR | TPR@5%FPR | Macro-F1 (secondary) |
|---|---|---|---|---|---|---|
| tfidf_c1 | {{VTwoTprTfidfCOneDeepsetNclean}} | {{VTwoTprTfidfCOneDeepsetAuroc|ci}} | {{VTwoTprTfidfCOneDeepsetFprdefault|ci}} | {{VTwoTprTfidfCOneDeepsetOracleOne|ci}} | {{VTwoTprTfidfCOneDeepsetOracleFive|ci}} | {{VTwoTprTfidfCOneDeepsetFOnemacro|ci}} |
| deberta_s42 | {{VTwoTprDebertaSFourTwoDeepsetNclean}} | {{VTwoTprDebertaSFourTwoDeepsetAuroc|ci}} | {{VTwoTprDebertaSFourTwoDeepsetFprdefault|ci}} | {{VTwoTprDebertaSFourTwoDeepsetOracleOne|ci}} | {{VTwoTprDebertaSFourTwoDeepsetOracleFive|ci}} | {{VTwoTprDebertaSFourTwoDeepsetFOnemacro|ci}} |
| deberta_s123 | {{VTwoTprDebertaSOneTwoThreeDeepsetNclean}} | {{VTwoTprDebertaSOneTwoThreeDeepsetAuroc|ci}} | {{VTwoTprDebertaSOneTwoThreeDeepsetFprdefault|ci}} | {{VTwoTprDebertaSOneTwoThreeDeepsetOracleOne|ci}} | {{VTwoTprDebertaSOneTwoThreeDeepsetOracleFive|ci}} | {{VTwoTprDebertaSOneTwoThreeDeepsetFOnemacro|ci}} |
| deberta_s999 | {{VTwoTprDebertaSNineNineNineDeepsetNclean}} | {{VTwoTprDebertaSNineNineNineDeepsetAuroc|ci}} | {{VTwoTprDebertaSNineNineNineDeepsetFprdefault|ci}} | {{VTwoTprDebertaSNineNineNineDeepsetOracleOne|ci}} | {{VTwoTprDebertaSNineNineNineDeepsetOracleFive|ci}} | {{VTwoTprDebertaSNineNineNineDeepsetFOnemacro|ci}} |
| protectai | {{VTwoTprProtectaiDeepsetNclean}} | {{VTwoTprProtectaiDeepsetAuroc|ci}} | {{VTwoTprProtectaiDeepsetFprdefault|ci}} | {{VTwoTprProtectaiDeepsetOracleOne|ci}} | {{VTwoTprProtectaiDeepsetOracleFive|ci}} | {{VTwoTprProtectaiDeepsetFOnemacro|ci}} |
| piguard | {{VTwoTprPiguardDeepsetNclean}} | {{VTwoTprPiguardDeepsetAuroc|ci}} | {{VTwoTprPiguardDeepsetFprdefault|ci}} | {{VTwoTprPiguardDeepsetOracleOne|ci}} | {{VTwoTprPiguardDeepsetOracleFive|ci}} | {{VTwoTprPiguardDeepsetFOnemacro|ci}} |

DeBERTa, mean ± SD over seeds 42/123/999: AUROC {{VTwoTprDebertaMeanDeepsetRocAucMean}} ± {{VTwoTprDebertaMeanDeepsetRocAucSd}}; FPR at default threshold {{VTwoTprDebertaMeanDeepsetFprDefaultCleanMean}} ± {{VTwoTprDebertaMeanDeepsetFprDefaultCleanSd}}; macro-F1 {{VTwoTprDebertaMeanDeepsetFOneMacroMean}} ± {{VTwoTprDebertaMeanDeepsetFOneMacroSd}}.

## F012. Oracle vs calibrated threshold gap on V2 (PRELIMINARY)

```finding
id: F012
status: pending
script: scripts/v2_tpr_metrics.py
commit: a59aae4
seeds: DeBERTa 42, 123, 999; bootstrap B=1000 seed 42
data_version: 'v2-frozen+external: V2 val clean rows as calibration set'
eval_set: v2_test, injecagent
threshold_protocol:
- oracle
- calibrated (threshold fixed on the clean rows of V2 val, 5% FPR target)
date: '2026-10-28'
source: results/v2_metrics/tpr_protocols/20261028-175945_seed42
cluster: doc_id (V2 test); user case (InjecAgent)
macros:
- VTwoCalgap*
awaiting: 'jobs/rescore_v2_probes.sh: V2 val scores for ProtectAI, PIGuard and the Qwen judge'
reason: calibrated numbers exist for TF-IDF and DeBERTa only; the public detectors and the judge await their V2 val scores.
preliminary: V2 val shares the training attack pool, so it is an easier calibration set than a held-out pool would be. Re-test on V3 before any claim.
```

Calibrated thresholds are fixed on the clean V2 val rows and applied unchanged. The CI ignores calibration noise.

| Eval set | Model | n clean (calibration) | TPR@5%FPR oracle | TPR@5%FPR calibrated | Calibrated threshold | Realized FPR on eval set |
|---|---|---|---|---|---|---|
| V2 test | tfidf_c1 | {{VTwoCalgapTfidfCOneVTwoTestNcalib}} | {{VTwoCalgapTfidfCOneVTwoTestOracleFive|ci}} | {{VTwoCalgapTfidfCOneVTwoTestCalibratedFive|ci}} | {{VTwoCalgapTfidfCOneVTwoTestThreshold}} | {{VTwoCalgapTfidfCOneVTwoTestRealizedfpr|ci}} |
| V2 test | tfidf_c100_tuned | -- | {{VTwoCalgapTfidfCOneZeroZeroTunedVTwoTestOracleFive|ci}} | {{VTwoCalgapTfidfCOneZeroZeroTunedVTwoTestCalibratedFive|ci}} | -- | -- |
| V2 test | deberta_s42 | {{VTwoCalgapDebertaSFourTwoVTwoTestNcalib}} | {{VTwoCalgapDebertaSFourTwoVTwoTestOracleFive|ci}} | {{VTwoCalgapDebertaSFourTwoVTwoTestCalibratedFive|ci}} | {{VTwoCalgapDebertaSFourTwoVTwoTestThreshold}} | {{VTwoCalgapDebertaSFourTwoVTwoTestRealizedfpr|ci}} |
| V2 test | deberta_s123 | {{VTwoCalgapDebertaSOneTwoThreeVTwoTestNcalib}} | {{VTwoCalgapDebertaSOneTwoThreeVTwoTestOracleFive|ci}} | {{VTwoCalgapDebertaSOneTwoThreeVTwoTestCalibratedFive|ci}} | {{VTwoCalgapDebertaSOneTwoThreeVTwoTestThreshold}} | {{VTwoCalgapDebertaSOneTwoThreeVTwoTestRealizedfpr|ci}} |
| V2 test | deberta_s999 | {{VTwoCalgapDebertaSNineNineNineVTwoTestNcalib}} | {{VTwoCalgapDebertaSNineNineNineVTwoTestOracleFive|ci}} | {{VTwoCalgapDebertaSNineNineNineVTwoTestCalibratedFive|ci}} | {{VTwoCalgapDebertaSNineNineNineVTwoTestThreshold}} | {{VTwoCalgapDebertaSNineNineNineVTwoTestRealizedfpr|ci}} |
| V2 test | protectai | -- | {{VTwoCalgapProtectaiVTwoTestOracleFive|ci}} | {{VTwoCalgapProtectaiVTwoTestCalibratedFive|ci}} | -- | -- |
| V2 test | piguard | -- | {{VTwoCalgapPiguardVTwoTestOracleFive|ci}} | {{VTwoCalgapPiguardVTwoTestCalibratedFive|ci}} | -- | -- |
| V2 test | judge_qwen2.5_7b | -- | {{VTwoCalgapJudgeQwenTwoFiveSevenbVTwoTestOracleFive|ci}} | {{VTwoCalgapJudgeQwenTwoFiveSevenbVTwoTestCalibratedFive|ci}} | -- | -- |
| InjecAgent | tfidf_c1 | {{VTwoCalgapTfidfCOneInjecagentNcalib}} | {{VTwoCalgapTfidfCOneInjecagentOracleFive|ci}} | {{VTwoCalgapTfidfCOneInjecagentCalibratedFive|ci}} | {{VTwoCalgapTfidfCOneInjecagentThreshold}} | {{VTwoCalgapTfidfCOneInjecagentRealizedfpr|ci}} |
| InjecAgent | deberta_s42 | {{VTwoCalgapDebertaSFourTwoInjecagentNcalib}} | {{VTwoCalgapDebertaSFourTwoInjecagentOracleFive|ci}} | {{VTwoCalgapDebertaSFourTwoInjecagentCalibratedFive|ci}} | {{VTwoCalgapDebertaSFourTwoInjecagentThreshold}} | {{VTwoCalgapDebertaSFourTwoInjecagentRealizedfpr|ci}} |
| InjecAgent | deberta_s123 | {{VTwoCalgapDebertaSOneTwoThreeInjecagentNcalib}} | {{VTwoCalgapDebertaSOneTwoThreeInjecagentOracleFive|ci}} | {{VTwoCalgapDebertaSOneTwoThreeInjecagentCalibratedFive|ci}} | {{VTwoCalgapDebertaSOneTwoThreeInjecagentThreshold}} | {{VTwoCalgapDebertaSOneTwoThreeInjecagentRealizedfpr|ci}} |
| InjecAgent | deberta_s999 | {{VTwoCalgapDebertaSNineNineNineInjecagentNcalib}} | {{VTwoCalgapDebertaSNineNineNineInjecagentOracleFive|ci}} | {{VTwoCalgapDebertaSNineNineNineInjecagentCalibratedFive|ci}} | {{VTwoCalgapDebertaSNineNineNineInjecagentThreshold}} | {{VTwoCalgapDebertaSNineNineNineInjecagentRealizedfpr|ci}} |
| InjecAgent | protectai | -- | {{VTwoCalgapProtectaiInjecagentOracleFive|ci}} | {{VTwoCalgapProtectaiInjecagentCalibratedFive|ci}} | -- | -- |
| InjecAgent | piguard | -- | {{VTwoCalgapPiguardInjecagentOracleFive|ci}} | {{VTwoCalgapPiguardInjecagentCalibratedFive|ci}} | -- | -- |
| InjecAgent | judge_qwen2.5_7b | -- | {{VTwoCalgapJudgeQwenTwoFiveSevenbInjecagentOracleFive|ci}} | {{VTwoCalgapJudgeQwenTwoFiveSevenbInjecagentCalibratedFive|ci}} | -- | -- |

DeBERTa seed mean ± SD, calibrated TPR@5%FPR on V2 test: {{VTwoCalgapDebertaMeanVTwoTestTprCalibratedFivepctFprMean}} ± {{VTwoCalgapDebertaMeanVTwoTestTprCalibratedFivepctFprSd}}.

## F013. V3 pre-freeze content-free gates (split seeds 0–4, draft benign pool)

```finding
id: F013
status: full
script: scripts/artifact_baselines.py --dataset v3 --split_seeds 0 1 2 3 4
commit: 507e9ee
seeds: split_seeds 0-4 (one build each); probes deterministic; bootstrap B=1000 seed 42
data_version: v3-prefreeze-draftpool (src/data_pipeline_v3.py @ 507e9ee, data/benign/benign_v3_draft.json, unreviewed)
eval_set: v3 test split of each split seed (rows with an inserted span); probes trained on the same seed's train split
threshold_protocol: default tau=0.50 (learned probes); fixed rule; gate = macro-F1 <= 0.60
date: '2026-10-28'
source: results/v3/artifact_baselines_prefreeze/20261028-185837_seed0
cluster: doc_id
macros:
- VThreePrefreezeGate*
- VThreePrefreezeGates*
- VThreePrefreezePositionOracle*
- VThreePrefreezeFirstParaRule*
- VThreePrefreezePrefixThreeZero*
- VThreePrefreezeSuffixThreeZero*
- VThreePrefreezeInsertionSite*
- VThreePrefreezeLength*
- VThreePrefreezeSpanForm*
- VThreePrefreezeSecondPerson*
- VThreePrefreezeBoundaryKind*
- VThreePrefreezeSpanContent*
```

Part of claim C1/C2 (V3 removes the V2 construction artifacts). Gate rule (v3_design §8, decision R2): pass =
pooled macro-F1 ≤ {{VThreePrefreezeGateThreshold}} on every split seed **and** per-task mean over seeds ≤
{{VThreePrefreezeGateThreshold}}. Overall: {{VThreePrefreezeGatesStatus}}. Built from the **unreviewed draft**
benign pool; the freeze rebuild with the reviewed pool is re-gated as a new entry. Rows marked "diag" are reported
diagnostics, not gates (decision R1). Full table: `paper/tables/v3_prefreeze_gates.tex`.

| Probe | What it uses | Pooled macro-F1, range over seeds | EmailQA mean | TableQA mean | CodeQA mean | Gate |
|---|---|---|---|---|---|---|
| position_oracle | LR on position, separator, boundary_kind, relative offset | {{VThreePrefreezePositionOraclePooledMin}}–{{VThreePrefreezePositionOraclePooledMax}} | {{VThreePrefreezePositionOracleTaskMeanEmail}} | {{VThreePrefreezePositionOracleTaskMeanTable}} | {{VThreePrefreezePositionOracleTaskMeanCode}} | {{VThreePrefreezePositionOracleStatus}} |
| first_para_rule | first paragraph ≤ 30 words → injected; untrained | {{VThreePrefreezeFirstParaRulePooledMin}}–{{VThreePrefreezeFirstParaRulePooledMax}} | {{VThreePrefreezeFirstParaRuleTaskMeanEmail}} | {{VThreePrefreezeFirstParaRuleTaskMeanTable}} | {{VThreePrefreezeFirstParaRuleTaskMeanCode}} | {{VThreePrefreezeFirstParaRuleStatus}} |
| prefix30_masked_tfidf_lr | first 30 words, span removed | {{VThreePrefreezePrefixThreeZeroMaskedTfidfLrPooledMin}}–{{VThreePrefreezePrefixThreeZeroMaskedTfidfLrPooledMax}} | {{VThreePrefreezePrefixThreeZeroMaskedTfidfLrTaskMeanEmail}} | {{VThreePrefreezePrefixThreeZeroMaskedTfidfLrTaskMeanTable}} | {{VThreePrefreezePrefixThreeZeroMaskedTfidfLrTaskMeanCode}} | {{VThreePrefreezePrefixThreeZeroMaskedTfidfLrStatus}} |
| suffix30_masked_tfidf_lr | last 30 words, span removed | {{VThreePrefreezeSuffixThreeZeroMaskedTfidfLrPooledMin}}–{{VThreePrefreezeSuffixThreeZeroMaskedTfidfLrPooledMax}} | {{VThreePrefreezeSuffixThreeZeroMaskedTfidfLrTaskMeanEmail}} | {{VThreePrefreezeSuffixThreeZeroMaskedTfidfLrTaskMeanTable}} | {{VThreePrefreezeSuffixThreeZeroMaskedTfidfLrTaskMeanCode}} | {{VThreePrefreezeSuffixThreeZeroMaskedTfidfLrStatus}} |
| insertion_site_tfidf_lr | ~15 tokens each side of the insertion point, span removed | {{VThreePrefreezeInsertionSiteTfidfLrPooledMin}}–{{VThreePrefreezeInsertionSiteTfidfLrPooledMax}} | {{VThreePrefreezeInsertionSiteTfidfLrTaskMeanEmail}} | {{VThreePrefreezeInsertionSiteTfidfLrTaskMeanTable}} | {{VThreePrefreezeInsertionSiteTfidfLrTaskMeanCode}} | {{VThreePrefreezeInsertionSiteTfidfLrStatus}} |
| length_lr | chars, words, tokens of the full input | {{VThreePrefreezeLengthLrPooledMin}}–{{VThreePrefreezeLengthLrPooledMax}} | {{VThreePrefreezeLengthLrTaskMeanEmail}} | {{VThreePrefreezeLengthLrTaskMeanTable}} | {{VThreePrefreezeLengthLrTaskMeanCode}} | {{VThreePrefreezeLengthLrStatus}} |
| span_form_you_lr | span-form features + contains you/your | {{VThreePrefreezeSpanFormYouLrPooledMin}}–{{VThreePrefreezeSpanFormYouLrPooledMax}} | {{VThreePrefreezeSpanFormYouLrTaskMeanEmail}} | {{VThreePrefreezeSpanFormYouLrTaskMeanTable}} | {{VThreePrefreezeSpanFormYouLrTaskMeanCode}} | {{VThreePrefreezeSpanFormYouLrStatus}} |
| second_person_lr | contains you/your only | {{VThreePrefreezeSecondPersonLrPooledMin}}–{{VThreePrefreezeSecondPersonLrPooledMax}} | {{VThreePrefreezeSecondPersonLrTaskMeanEmail}} | {{VThreePrefreezeSecondPersonLrTaskMeanTable}} | {{VThreePrefreezeSecondPersonLrTaskMeanCode}} | {{VThreePrefreezeSecondPersonLrStatus}} |
| boundary_kind_lr | boundary kind only | {{VThreePrefreezeBoundaryKindLrPooledMin}}–{{VThreePrefreezeBoundaryKindLrPooledMax}} | {{VThreePrefreezeBoundaryKindLrTaskMeanEmail}} | {{VThreePrefreezeBoundaryKindLrTaskMeanTable}} | {{VThreePrefreezeBoundaryKindLrTaskMeanCode}} | {{VThreePrefreezeBoundaryKindLrStatus}} |
| prefix30_tfidf_lr | first 30 words, unmasked (diagnostic) | {{VThreePrefreezePrefixThreeZeroTfidfLrPooledMin}}–{{VThreePrefreezePrefixThreeZeroTfidfLrPooledMax}} | {{VThreePrefreezePrefixThreeZeroTfidfLrTaskMeanEmail}} | {{VThreePrefreezePrefixThreeZeroTfidfLrTaskMeanTable}} | {{VThreePrefreezePrefixThreeZeroTfidfLrTaskMeanCode}} | {{VThreePrefreezePrefixThreeZeroTfidfLrStatus}} |
| suffix30_tfidf_lr | last 30 words, unmasked (diagnostic) | {{VThreePrefreezeSuffixThreeZeroTfidfLrPooledMin}}–{{VThreePrefreezeSuffixThreeZeroTfidfLrPooledMax}} | {{VThreePrefreezeSuffixThreeZeroTfidfLrTaskMeanEmail}} | {{VThreePrefreezeSuffixThreeZeroTfidfLrTaskMeanTable}} | {{VThreePrefreezeSuffixThreeZeroTfidfLrTaskMeanCode}} | {{VThreePrefreezeSuffixThreeZeroTfidfLrStatus}} |
| span_content_tfidf | inserted span text (content ceiling; diagnostic) | {{VThreePrefreezeSpanContentTfidfPooledMin}}–{{VThreePrefreezeSpanContentTfidfPooledMax}} | {{VThreePrefreezeSpanContentTfidfTaskMeanEmail}} | {{VThreePrefreezeSpanContentTfidfTaskMeanTable}} | {{VThreePrefreezeSpanContentTfidfTaskMeanCode}} | {{VThreePrefreezeSpanContentTfidfStatus}} |
| span_form_lr | 9 span-form features, no you/your (diagnostic) | {{VThreePrefreezeSpanFormLrPooledMin}}–{{VThreePrefreezeSpanFormLrPooledMax}} | {{VThreePrefreezeSpanFormLrTaskMeanEmail}} | {{VThreePrefreezeSpanFormLrTaskMeanTable}} | {{VThreePrefreezeSpanFormLrTaskMeanCode}} | {{VThreePrefreezeSpanFormLrStatus}} |

- The span-form + second-person gate passes on CodeQA (per-task mean
  {{VThreePrefreezeSpanFormYouLrTaskMeanCode}}). An earlier draft pool failed it on CodeQA (dry run
  `results/v3/artifact_baselines_dryrun5/20261021-161044_seed0`, commit 65bebfd; not a findings entry). The
  code-register clean-A items were then rewritten (commit 4a2ad98).
- Unmasked prefix-30 and suffix-30 diagnostics exceed 0.60 on some seeds or tasks. They see the inserted span
  itself, so they are content probes, not content-free gates; the span-masked versions sit at 0.3333 (constant
  prediction) on every seed.

## F014. V3 pre-freeze integrity checks (split seeds 0–4, draft benign pool)

```finding
id: F014
status: full
script: scripts/v3_integrity.py --split_seeds 0 1 2 3 4
commit: 507e9ee
seeds: split_seeds 0-4
data_version: v3-prefreeze-draftpool (src/data_pipeline_v3.py @ 507e9ee, data/benign/benign_v3_draft.json, unreviewed)
eval_set: all written CSVs of each split seed (train/val/test, val_id, raw_clean, ambiguous, long)
threshold_protocol: n/a (assertions)
date: '2026-10-28'
source: results/v3/integrity_prefreeze/20261028-185841_seed0
macros:
- VThreePrefreezeIntegrity*
```

Independent re-read of the written CSVs: doc_id, exact text, near-duplicate component, attack string, benign item
and donor text disjoint across train/val/test; train/val attack categories disjoint; test attacks only from the test
files; donor docs from the same task and split; ambiguous items never used as clean A; exact 1:1 balance per
task × split; 4 rows (2 injected) per doc; eval-slice rows belong to their split's docs; ≤ 512 DeBERTa-v3 tokens
per main, val_id, raw_clean and ambiguous row (re-tokenized; the stored count matches); every span recoverable from
its offset. Overall: {{VThreePrefreezeIntegrityStatus}}.

| Split seed | Integrity | Main rows | val_id | raw_clean | ambiguous | long |
|---|---|---|---|---|---|---|
| 0 | {{VThreePrefreezeIntegritySeedZeroStatus}} | {{VThreePrefreezeIntegritySeedZeroRowsMain}} | {{VThreePrefreezeIntegritySeedZeroRowsValId}} | {{VThreePrefreezeIntegritySeedZeroRowsRawClean}} | {{VThreePrefreezeIntegritySeedZeroRowsAmbiguous}} | {{VThreePrefreezeIntegritySeedZeroRowsLong}} |
| 1 | {{VThreePrefreezeIntegritySeedOneStatus}} | {{VThreePrefreezeIntegritySeedOneRowsMain}} | {{VThreePrefreezeIntegritySeedOneRowsValId}} | {{VThreePrefreezeIntegritySeedOneRowsRawClean}} | {{VThreePrefreezeIntegritySeedOneRowsAmbiguous}} | {{VThreePrefreezeIntegritySeedOneRowsLong}} |
| 2 | {{VThreePrefreezeIntegritySeedTwoStatus}} | {{VThreePrefreezeIntegritySeedTwoRowsMain}} | {{VThreePrefreezeIntegritySeedTwoRowsValId}} | {{VThreePrefreezeIntegritySeedTwoRowsRawClean}} | {{VThreePrefreezeIntegritySeedTwoRowsAmbiguous}} | {{VThreePrefreezeIntegritySeedTwoRowsLong}} |
| 3 | {{VThreePrefreezeIntegritySeedThreeStatus}} | {{VThreePrefreezeIntegritySeedThreeRowsMain}} | {{VThreePrefreezeIntegritySeedThreeRowsValId}} | {{VThreePrefreezeIntegritySeedThreeRowsRawClean}} | {{VThreePrefreezeIntegritySeedThreeRowsAmbiguous}} | {{VThreePrefreezeIntegritySeedThreeRowsLong}} |
| 4 | {{VThreePrefreezeIntegritySeedFourStatus}} | {{VThreePrefreezeIntegritySeedFourRowsMain}} | {{VThreePrefreezeIntegritySeedFourRowsValId}} | {{VThreePrefreezeIntegritySeedFourRowsRawClean}} | {{VThreePrefreezeIntegritySeedFourRowsAmbiguous}} | {{VThreePrefreezeIntegritySeedFourRowsLong}} |

Main rows per task/split differ slightly across seeds (the total is the same); split seed 0: EmailQA {{VThreePrefreezeIntegritySeedZeroMainEmailTrain}} / {{VThreePrefreezeIntegritySeedZeroMainEmailVal}} /
{{VThreePrefreezeIntegritySeedZeroMainEmailTest}}, TableQA {{VThreePrefreezeIntegritySeedZeroMainTableTrain}} / {{VThreePrefreezeIntegritySeedZeroMainTableVal}} / {{VThreePrefreezeIntegritySeedZeroMainTableTest}}, CodeQA
{{VThreePrefreezeIntegritySeedZeroMainCodeTrain}} / {{VThreePrefreezeIntegritySeedZeroMainCodeVal}} / {{VThreePrefreezeIntegritySeedZeroMainCodeTest}} (train / val / test).
