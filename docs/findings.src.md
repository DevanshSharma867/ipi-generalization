# Findings log (numbers only)

Commit hashes recorded before 2026-11-04 are pre-rewrite hashes, kept as the run manifests record them (the provenance
gate checks them against `config.json`); `docs/commit_rewrites.md` maps them to the current history.

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
status: full
script: scripts/score_probes.py -> scripts/probe_report.py, scripts/truncation_control.py
commit: 86d9a08
seeds: DeBERTa 42, 123, 999; others zero-shot or deterministic; bootstrap seed 42
data_version: v2_probes (data/probes_v2/) + v2-frozen test
eval_set: p1-p4 probe sets, v2_test
threshold_protocol: default tau=0.50
date: '2026-10-29 (reports; truncation_control at 71750a5); 2026-10-29 (scores, job 422174 at 7f56fc9)'
source:
- results/v2_probes/probe_report/20261029-171951_seed42
- results/v2_probes/truncation_control/20261029-172734_seed42
upstream:
- results/v2_probes/score_deberta_s42/20261029-115305_seed42
- results/v2_probes/score_deberta_s123/20261029-115605_seed123
- results/v2_probes/score_deberta_s999/20261029-115638_seed999
- results/v2_probes/score_judge_qwen2.5_7b/20261029-115819_seed42
- results/v2_probes/score_judge_qwen2.5_7b/20261021-123213_seed42
- results/v2_probes/score_piguard/20261029-115747_seed42
- results/v2_probes/score_protectai/20261029-115710_seed42
- results/v2_probes/score_tfidf_c1/20261028-175939_seed42
cluster: doc_id
macros:
- VTwoProbe*
- VTwoTrunc*
- VTwoSeedvar*
- VTwoGroups*
- '!*Vsum*'
```

Part of claim C1. Re-measured from clean scores on 2026-10-29: the clean re-score reproduces the 2026-10-19
dirty-tree scores row for row, and no value here moved (the earlier dirty-linked reports were identical).

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
commit: 71750a5
seeds: DeBERTa 42, 123, 999; others deterministic or zero-shot; bootstrap B=1000 seed 42
data_version: 'v2-frozen+external (V2 test: tag v2-frozen)'
eval_set: v2_test
threshold_protocol:
- oracle (threshold chosen on the clean rows of V2 test)
- default threshold for FPR@default and macro-F1
date: '2026-10-29'
source: results/v2_metrics/tpr_protocols/20261029-172911_seed42
cluster: doc_id
macros:
- VTwoTpr*VTwoTest*
- '!*Vsum*'
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
commit: 71750a5
seeds: DeBERTa 42, 123, 999; others deterministic or zero-shot; bootstrap B=1000 seed 42
data_version: 'v2-frozen+external: InjecAgent@f19c9f2 + build bab750e (score_detectors.py build)'
eval_set: injecagent
threshold_protocol:
- oracle
- default threshold for FPR@default and macro-F1
date: '2026-10-29'
source: results/v2_metrics/tpr_protocols/20261029-172911_seed42
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
commit: 71750a5
seeds: DeBERTa 42, 123, 999; others deterministic or zero-shot; bootstrap B=1000 seed 42
data_version: 'v2-frozen+external: hf:deepset/prompt-injections@4f61ecb (test split) + build bab750e'
eval_set: deepset
threshold_protocol:
- oracle
- default threshold for FPR@default and macro-F1
date: '2026-10-29'
source: results/v2_metrics/tpr_protocols/20261029-172911_seed42
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
status: full
script: scripts/v2_tpr_metrics.py
commit: 71750a5
seeds: DeBERTa 42, 123, 999; bootstrap B=1000 seed 42
data_version: 'v2-frozen+external: V2 val clean rows as calibration set'
eval_set: v2_test, injecagent
threshold_protocol:
- oracle
- calibrated (threshold fixed on the clean rows of V2 val, 5% FPR target)
date: '2026-10-29'
source: results/v2_metrics/tpr_protocols/20261029-172911_seed42
cluster: doc_id (V2 test); user case (InjecAgent)
macros:
- VTwoCalgap*
- '!*Vsum*'
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
| V2 test | protectai | {{VTwoCalgapProtectaiVTwoTestNcalib}} | {{VTwoCalgapProtectaiVTwoTestOracleFive|ci}} | {{VTwoCalgapProtectaiVTwoTestCalibratedFive|ci}} | {{VTwoCalgapProtectaiVTwoTestThreshold}} | {{VTwoCalgapProtectaiVTwoTestRealizedfpr|ci}} |
| V2 test | piguard | {{VTwoCalgapPiguardVTwoTestNcalib}} | {{VTwoCalgapPiguardVTwoTestOracleFive|ci}} | {{VTwoCalgapPiguardVTwoTestCalibratedFive|ci}} | {{VTwoCalgapPiguardVTwoTestThreshold}} | {{VTwoCalgapPiguardVTwoTestRealizedfpr|ci}} |
| V2 test | judge_qwen2.5_7b | {{VTwoCalgapJudgeQwenTwoFiveSevenbVTwoTestNcalib}} | {{VTwoCalgapJudgeQwenTwoFiveSevenbVTwoTestOracleFive|ci}} | {{VTwoCalgapJudgeQwenTwoFiveSevenbVTwoTestCalibratedFive|ci}} | {{VTwoCalgapJudgeQwenTwoFiveSevenbVTwoTestThreshold}} | {{VTwoCalgapJudgeQwenTwoFiveSevenbVTwoTestRealizedfpr|ci}} |
| InjecAgent | tfidf_c1 | {{VTwoCalgapTfidfCOneInjecagentNcalib}} | {{VTwoCalgapTfidfCOneInjecagentOracleFive|ci}} | {{VTwoCalgapTfidfCOneInjecagentCalibratedFive|ci}} | {{VTwoCalgapTfidfCOneInjecagentThreshold}} | {{VTwoCalgapTfidfCOneInjecagentRealizedfpr|ci}} |
| InjecAgent | deberta_s42 | {{VTwoCalgapDebertaSFourTwoInjecagentNcalib}} | {{VTwoCalgapDebertaSFourTwoInjecagentOracleFive|ci}} | {{VTwoCalgapDebertaSFourTwoInjecagentCalibratedFive|ci}} | {{VTwoCalgapDebertaSFourTwoInjecagentThreshold}} | {{VTwoCalgapDebertaSFourTwoInjecagentRealizedfpr|ci}} |
| InjecAgent | deberta_s123 | {{VTwoCalgapDebertaSOneTwoThreeInjecagentNcalib}} | {{VTwoCalgapDebertaSOneTwoThreeInjecagentOracleFive|ci}} | {{VTwoCalgapDebertaSOneTwoThreeInjecagentCalibratedFive|ci}} | {{VTwoCalgapDebertaSOneTwoThreeInjecagentThreshold}} | {{VTwoCalgapDebertaSOneTwoThreeInjecagentRealizedfpr|ci}} |
| InjecAgent | deberta_s999 | {{VTwoCalgapDebertaSNineNineNineInjecagentNcalib}} | {{VTwoCalgapDebertaSNineNineNineInjecagentOracleFive|ci}} | {{VTwoCalgapDebertaSNineNineNineInjecagentCalibratedFive|ci}} | {{VTwoCalgapDebertaSNineNineNineInjecagentThreshold}} | {{VTwoCalgapDebertaSNineNineNineInjecagentRealizedfpr|ci}} |
| InjecAgent | protectai | {{VTwoCalgapProtectaiInjecagentNcalib}} | {{VTwoCalgapProtectaiInjecagentOracleFive|ci}} | {{VTwoCalgapProtectaiInjecagentCalibratedFive|ci}} | {{VTwoCalgapProtectaiInjecagentThreshold}} | {{VTwoCalgapProtectaiInjecagentRealizedfpr|ci}} |
| InjecAgent | piguard | {{VTwoCalgapPiguardInjecagentNcalib}} | {{VTwoCalgapPiguardInjecagentOracleFive|ci}} | {{VTwoCalgapPiguardInjecagentCalibratedFive|ci}} | {{VTwoCalgapPiguardInjecagentThreshold}} | {{VTwoCalgapPiguardInjecagentRealizedfpr|ci}} |
| InjecAgent | judge_qwen2.5_7b | {{VTwoCalgapJudgeQwenTwoFiveSevenbInjecagentNcalib}} | {{VTwoCalgapJudgeQwenTwoFiveSevenbInjecagentOracleFive|ci}} | {{VTwoCalgapJudgeQwenTwoFiveSevenbInjecagentCalibratedFive|ci}} | {{VTwoCalgapJudgeQwenTwoFiveSevenbInjecagentThreshold}} | {{VTwoCalgapJudgeQwenTwoFiveSevenbInjecagentRealizedfpr|ci}} |

DeBERTa seed mean ± SD, calibrated TPR@5%FPR on V2 test: {{VTwoCalgapDebertaMeanVTwoTestTprCalibratedFivepctFprMean}} ± {{VTwoCalgapDebertaMeanVTwoTestTprCalibratedFivepctFprSd}}.

- The V2 val threshold does not hold its 5% FPR target for the fine-tuned models: DeBERTa's realized FPR is above
  the target for every seed on both sets, with a CI above 0.05 for s42 and s999 on V2 test and for s42 and s123 on
  InjecAgent. TF-IDF holds it on V2 test but not on InjecAgent (CI above 0.05).
- For ProtectAI, PIGuard and the judge, every realized-FPR CI includes 0.05 on both sets.
- No oracle − calibrated difference has a paired test here; the table reports the two protocols side by side.

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

## F015. Qwen2.5-7B judge, variant-summing Yes/No rule: V2 probes and V2 test

```finding
id: F015
status: full
script: scripts/score_probes.py --judge_rule variants -> scripts/probe_report.py, scripts/truncation_control.py, scripts/v2_tpr_metrics.py
commit: 86d9a08
seeds: zero-shot (greedy next-token log-probabilities); bootstrap B=1000 seed 42
data_version: 'v2_probes (data/probes_v2/) + v2-frozen+external (V2 test; V2 val clean rows as calibration set)'
eval_set: p1-p4 probe sets, v2_test
threshold_protocol:
- default tau=0.50 (probe flag rates, row groups, FPR@default, macro-F1)
- oracle (threshold chosen on the clean rows of V2 test)
- calibrated (threshold fixed on the clean rows of V2 val, 5% FPR target)
date: '2026-10-29 (scores, job 422174 at 7f56fc9; reports at 86d9a08 and 71750a5)'
source:
- results/v2_probes/probe_report/20261029-171951_seed42
- results/v2_probes/truncation_control/20261029-172734_seed42
- results/v2_metrics/tpr_protocols/20261029-172911_seed42
upstream:
- results/v2_probes/score_judge_qwen2.5_7b_vsum/20261029-120108_seed42
- results/v2_probes/score_judge_qwen2.5_7b_vsum_v2val/20261029-120214_seed42
cluster: doc_id
macros:
- '*JudgeQwenTwoFiveSevenbVsum*'
preliminary: the calibrated value uses V2 val, which shares the training attack pool (see F012).
```

The same frozen judge prompt and model as the V2 judge (F007, F009), scored with the shared Yes/No rule of
`src/judging.py` (`yes_no_score`): P(Yes) / (P(Yes) + P(No)) with every case and whitespace variant summed
(paper_plan A12). The two-token V2 judge numbers stay as they are; this entry is the variant-rule judge. On the
artifact-laden V2 benchmark only.

These runs came from job 422174 (`jobs/rescore_v2_probes.sh` at 7f56fc9, which still held the variant-rule steps).
`jobs/rescore_v2_judge_variant.sh` (added in 1542950 for the same two runs, paper_plan A15) was never submitted and
was deleted on 2026-10-29 as redundant; the runs above are the only variant-rule V2 scores.

### Position probes and V2 test row groups (default threshold)

No probe row exceeds the judge's context, so the truncated fraction is
{{VTwoTruncJudgeQwenTwoFiveSevenbVsumPThreeendtruncated}} and full = matched.

| | Value |
|---|---|
| P3 recall, start | {{VTwoProbeJudgeQwenTwoFiveSevenbVsumPThreeStart|ci}} |
| P3 recall, middle | {{VTwoProbeJudgeQwenTwoFiveSevenbVsumPThreeMiddle|ci}} |
| P3 recall, end | {{VTwoProbeJudgeQwenTwoFiveSevenbVsumPThreeEnd|ci}} |
| P3 end − start | {{VTwoProbeJudgeQwenTwoFiveSevenbVsumPThreeEndMinusStart|ci}} |
| P2 start − middle | {{VTwoProbeJudgeQwenTwoFiveSevenbVsumPTwoStartMinusMiddle|ci}} |
| P1 start − middle | {{VTwoProbeJudgeQwenTwoFiveSevenbVsumPOneStartMinusMiddle|ci}} |
| V2 test macro-F1 | {{VTwoProbeJudgeQwenTwoFiveSevenbVsumVTwotestFOneMacro|ci}} |
| V2 test recall (injected rows, n = {{VTwoGroupsJudgeQwenTwoFiveSevenbVsumRecallN}}) | {{VTwoGroupsJudgeQwenTwoFiveSevenbVsumRecall|ci}} |
| Flag rate, V2 raw clean rows (n = {{VTwoGroupsJudgeQwenTwoFiveSevenbVsumRawcleanflagN}}) | {{VTwoGroupsJudgeQwenTwoFiveSevenbVsumRawcleanflag|ci}} |
| Flag rate, V2 benign-imperative rows (n = {{VTwoGroupsJudgeQwenTwoFiveSevenbVsumBenignflagN}}) | {{VTwoGroupsJudgeQwenTwoFiveSevenbVsumBenignflag|ci}} |
| Flag rate, P4 whitespace-only (n = {{VTwoGroupsJudgeQwenTwoFiveSevenbVsumPfourflagN}}) | {{VTwoGroupsJudgeQwenTwoFiveSevenbVsumPfourflag|ci}} |
| P4 − raw clean (paired docs: {{VTwoGroupsJudgeQwenTwoFiveSevenbVsumPfourminusrawcleanNpairs}}) | {{VTwoGroupsJudgeQwenTwoFiveSevenbVsumPfourminusrawclean|ci}} |

### V2 test headline metrics

V2 test has {{VTwoTprJudgeQwenTwoFiveSevenbVsumVTwoTestNclean}} clean rows in
{{VTwoTprJudgeQwenTwoFiveSevenbVsumVTwoTestNclusters}} doc clusters, so TPR@1%FPR is
{{VTwoTprJudgeQwenTwoFiveSevenbVsumVTwoTestOracleOne}}.

| Metric | Value |
|---|---|
| AUROC | {{VTwoTprJudgeQwenTwoFiveSevenbVsumVTwoTestAuroc|ci}} |
| FPR at the default threshold (clean rows) | {{VTwoTprJudgeQwenTwoFiveSevenbVsumVTwoTestFprdefault|ci}} |
| TPR@5%FPR, oracle | {{VTwoTprJudgeQwenTwoFiveSevenbVsumVTwoTestOracleFive|ci}} |
| TPR@5%FPR, calibrated (n clean calibration = {{VTwoCalgapJudgeQwenTwoFiveSevenbVsumVTwoTestNcalib}}) | {{VTwoCalgapJudgeQwenTwoFiveSevenbVsumVTwoTestCalibratedFive|ci}} |
| Calibrated threshold | {{VTwoCalgapJudgeQwenTwoFiveSevenbVsumVTwoTestThreshold}} |
| Realized FPR on V2 test at the calibrated threshold | {{VTwoCalgapJudgeQwenTwoFiveSevenbVsumVTwoTestRealizedfpr|ci}} |
| Macro-F1 (default threshold) | {{VTwoTprJudgeQwenTwoFiveSevenbVsumVTwoTestFOnemacro|ci}} |

- P3 recall falls from start to end; the end − start CI excludes 0. P2 and P1 start − middle CIs include 0.
- The V2 test macro-F1 CI includes 0.5, and the P4 − raw clean CI includes 0.
- The calibrated threshold's realized FPR CI on V2 test includes the 5% target.
- Against the two-token judge (F007, F009, F012): every default-threshold value and the calibrated TPR@5%FPR are
  the same. Only the ranking-based values (AUROC, oracle TPR@5%FPR) differ, and their CIs overlap those of the
  two-token judge. No paired test was run.

## F016. Transfer sets rebuilt from Liu (2026): Gate 1

```finding
id: F016
status: full
script: scripts/build_transfer_sets.sh (jobs/build_transfer_sets.sh) -> eval/transfer/gate1.py
commit: 6c94421
seeds: n/a (deterministic rebuild, PYTHONHASHSEED=0 as upstream)
data_version: transfer-bib5d0ae7c-agentdojo0.1.34-tb59a200c-filescf00fcba
eval_set: AD-Clean, AD-Matched, AD-Shift, TB-Matched, BIPIA (Liu), general corpus
threshold_protocol: n/a (set sizes and row/label alignment only; no scores)
date: 2026-10-30
source:
- results/transfer/set_build_gate1/20261030-124112_seed0
macros:
- TransferGateOne*
```

Gate 1 of the transfer harness (paper_plan A17): {{TransferGateOneStatus}}. The six sets were rebuilt from public
sources with Liu's upstream scripts at github.com/lzwhehe/benign-instruction-bench commit `5d0ae7c`, unmodified.
Each rebuilt set matches every score file Liu released for it in row count and, where the file carries labels, in
label sequence. Its size also matches the paper. The manifest in the run's `metrics.json` records environments,
HF dataset revisions and where each step ran.

| Set | Rows | Released score files matched |
|---|---|---|
| AD-Clean | {{TransferGateOneAdCleanRows}} (clean) | {{TransferGateOneAdCleanFiles}} |
| AD-Matched | {{TransferGateOneAdMatchedBenign}} benign, {{TransferGateOneAdMatchedInjected}} injected | {{TransferGateOneAdMatchedFiles}} |
| AD-Shift | {{TransferGateOneAdShiftBenign}} benign, {{TransferGateOneAdShiftInjected}} injected | {{TransferGateOneAdShiftFiles}} |
| TB-Matched | {{TransferGateOneTbMatchedBenign}} benign, {{TransferGateOneTbMatchedInjected}} injected | {{TransferGateOneTbMatchedFiles}} |
| BIPIA (Liu) | {{TransferGateOneBipiaBenign}} benign, {{TransferGateOneBipiaInjectedTest}} injected (test half) | {{TransferGateOneBipiaFiles}} |
| General corpus | {{TransferGateOneGeneralRows}} (clean) | {{TransferGateOneGeneralFiles}} |

- AD-Matched and AD-Shift benign rows repeat a small set of distinct texts per attack template (A16); they are never counted as
  clean (A16). AD-Clean is the AgentDojo negative set.
- Gate 1 checks rows and labels only. Score agreement is Gate 2.

## F017. Liu (2026) reproduction: Gate 2 with the recovered AgentDojo pairing

```finding
id: F017
status: full
script: jobs/gate2_fu_pairing.sh, jobs/gate2_fu_judge.sh -> eval/transfer/gate2_followup.py (report, residual)
commit: 007c5a5
seeds: n/a (deterministic scoring; judge greedy)
data_version: transfer-bib5d0ae7c-agentdojo0.1.34-tb59a200c-filescf00fcba
eval_set: AD-Matched (recovered pairing), AD-Clean, TB-Matched, BIPIA (Liu)
threshold_protocol: Liu's (AUROC; TPR at 1% FPR on his negatives); per-row |Δ| vs his released score files
date: 2026-10-31
source:
- results/transfer/gate2_followup_pairing/20261030-201440_seed0
- results/transfer/gate2_followup_residual/20261031-143440_seed0
- results/transfer/gate2_followup_judge/20261030-202708_seed0
macros:
- TransferGateTwo*
```

Gate 2 (paper_plan A19) in Liu's environment (torch 2.8, transformers 4.57.6), with AD-Matched rebuilt using the
injection-task pairing recovered from his released Prompt Guard 2 scores (docs/deviations.md, 2026-10-30). Other sets
are Gate 2 run 2 (`results/transfer/liu_repro_gate2/20261030-160800_seed0`). A19 metric checks: {{TransferGateTwoStatus}}.

| Detector | AD-Matched AUROC (ours / Liu) | AD-Matched TPR@1% | BIPIA AUROC | TB-Matched AUROC |
|---|---|---|---|---|
| PIGuard | {{TransferGateTwoPiguardAdmatchedAurocOurs}} / {{TransferGateTwoPiguardAdmatchedAurocLiu}} | {{TransferGateTwoPiguardAdmatchedTprOneOurs}} / {{TransferGateTwoPiguardAdmatchedTprOneLiu}} | {{TransferGateTwoPiguardBipiaAurocOurs}} / {{TransferGateTwoPiguardBipiaAurocLiu}} | {{TransferGateTwoPiguardTbevalAurocOurs}} / {{TransferGateTwoPiguardTbevalAurocLiu}} |
| Prompt Guard 2 86M | {{TransferGateTwoPromptguardTwoEightSixmAdmatchedAurocOurs}} / {{TransferGateTwoPromptguardTwoEightSixmAdmatchedAurocLiu}} | {{TransferGateTwoPromptguardTwoEightSixmAdmatchedTprOneOurs}} / {{TransferGateTwoPromptguardTwoEightSixmAdmatchedTprOneLiu}} | {{TransferGateTwoPromptguardTwoEightSixmBipiaAurocOurs}} / {{TransferGateTwoPromptguardTwoEightSixmBipiaAurocLiu}} | {{TransferGateTwoPromptguardTwoEightSixmTbevalAurocOurs}} / {{TransferGateTwoPromptguardTwoEightSixmTbevalAurocLiu}} |
| Horizon base | {{TransferGateTwoHorizonBaseAdmatchedAurocOurs}} / {{TransferGateTwoHorizonBaseAdmatchedAurocLiu}} | {{TransferGateTwoHorizonBaseAdmatchedTprOneOurs}} / {{TransferGateTwoHorizonBaseAdmatchedTprOneLiu}} | {{TransferGateTwoHorizonBaseBipiaAurocOurs}} / {{TransferGateTwoHorizonBaseBipiaAurocLiu}} | {{TransferGateTwoHorizonBaseTbevalAurocOurs}} / {{TransferGateTwoHorizonBaseTbevalAurocLiu}} |
| Qwen2.5-7B judge (4-bit) | {{TransferGateTwoJudgeQwenAdmatchedAurocOurs}} / {{TransferGateTwoJudgeQwenAdmatchedAurocLiu}} | {{TransferGateTwoJudgeQwenAdmatchedTprOneOurs}} / {{TransferGateTwoJudgeQwenAdmatchedTprOneLiu}} | {{TransferGateTwoJudgeQwenBipiaAurocOurs}} / {{TransferGateTwoJudgeQwenBipiaAurocLiu}} | {{TransferGateTwoJudgeQwenTbevalAurocOurs}} / {{TransferGateTwoJudgeQwenTbevalAurocLiu}} |

**Pairing recovery.** Every user task matched one candidate (Prompt Guard 2 per row ≤ 1e-4): AD-Matched
{{TransferGateTwoAdmatchedMatched}}/{{TransferGateTwoAdmatchedGroups}}, AD-Shift
{{TransferGateTwoAdshiftMatched}}/{{TransferGateTwoAdshiftGroups}}, no ambiguous matches. Rows carrying AgentDojo's
wall-clock `last_modified` ({{TransferGateTwoAdmatchedTimedep}} and {{TransferGateTwoAdshiftTimedep}}) are left out of
the per-row checks. Independent validation, max per-row |Δ| vs Liu on AD-Matched: Horizon
{{TransferGateTwoAdmatchedHorizonBaseMaxdiff}}, Prompt Guard 2 {{TransferGateTwoAdmatchedPromptguardTwoEightSixmMaxdiff}},
PIGuard {{TransferGateTwoAdmatchedPiguardMaxdiff}}. The pre-registered rule (PIGuard and Horizon ≤ 1e-3 per row) gives
{{TransferGateTwoRecoveryStatus}}: PIGuard exceeds it on {{TransferGateTwoAdmatchedPiguardOverbound}} AD-Matched and
{{TransferGateTwoAdshiftPiguardOverbound}} AD-Shift rows, with no label flips at 0.5. The recovered pairing was adopted for every
detector anyway (docs/deviations.md, 2026-10-31), on the residual evidence below.

**Residual.** Every row over the bound is a calendar event with two or more participants
({{TransferGateTwoResidualStatus}}). AgentDojo builds that list with `list(set(participants))`, so its order depends
on the process's hash seed, independently of the pairing. Outside those rows ({{TransferGateTwoAdmatchedMultipart}}
and {{TransferGateTwoAdshiftMultipart}} multi-participant rows), PIGuard's max |Δ| is
{{TransferGateTwoAdmatchedPiguardMaxother}} (AD-Matched) and {{TransferGateTwoAdshiftPiguardMaxother}} (AD-Shift).
Inferred, not verified: Liu's text cannot be checked, only his scores.

**Judge.** Two identical 4-bit runs on BIPIA differ by at most {{TransferGateTwoJudgeRepeatVsrun}} per row
({{TransferGateTwoJudgeRepeatFlips}} flip at 0.5, a row at the boundary); the run at GPU_MEM=6GiB differs from
run 2 by {{TransferGateTwoJudgeSixgibVsrun}} (all modules on the GPU in both). Against Liu's released file the
judge differs by up to {{TransferGateTwoJudgeVsliuMax}} per row ({{TransferGateTwoJudgeVsliuOverbound}} of
{{TransferGateTwoJudgeBipiaRows}} rows over 1e-3, {{TransferGateTwoJudgeVsliuFlips}} flips): neither run-to-run noise
nor the memory setting, so a hardware/kernel difference in 4-bit inference. Metrics still match (table).
bf16 − 4-bit AUROC on AD-Matched (A20): {{TransferGateTwoJudgeSeededGap}} on our seeded build,
{{TransferGateTwoJudgeRecoveredGap}} on the recovered one (Liu: within 0.02).

## F018. Detector scores depend on the transformers version (4.57.6 vs 5.20.1), not on batching

```finding
id: F018
status: full
script: jobs/gate2_fu_batch.sh -> eval/transfer/gate2_followup.py (subset, deberta_bs, report --part batch)
commit: 007c5a5
seeds: n/a (deterministic scoring); BIPIA subset rng seed 0
data_version: transfer-bib5d0ae7c-agentdojo0.1.34-tb59a200c-filescf00fcba
eval_set: 500-row random subset of BIPIA (Liu); V2 test (our DeBERTa checkpoints, seeds 42/123/999)
threshold_protocol: per-row |Δ| of P(injection); flips at 0.5
date: 2026-10-31
source:
- results/transfer/gate2_followup_batch/20261030-195313_seed0
- results/transfer/gate2_followup_envfix/20261031-172529_seed0
macros:
- TransferEnv*
```

Same GPU, same weights and revisions, same inputs; only the library stack differs (Liu's env: torch 2.8 /
transformers 4.57.6; ours: torch 2.11 / transformers 5.20.1). On {{TransferEnvRows}} BIPIA rows at batch size 1:

| Detector | max per-row |Δ| ours vs Liu env | rows > 1e-3 | flips at 0.5 | bs 1 vs 16 (either env) | Liu env vs released |
|---|---|---|---|---|---|
| PIGuard | {{TransferEnvPiguardMax}} | {{TransferEnvPiguardOverbound}} | {{TransferEnvPiguardFlips}} | {{TransferEnvPiguardBatch}} | {{TransferEnvPiguardLiuenv}} |
| Prompt Guard 2 86M | {{TransferEnvPromptguardTwoEightSixmMax}} | {{TransferEnvPromptguardTwoEightSixmOverbound}} | {{TransferEnvPromptguardTwoEightSixmFlips}} | {{TransferEnvPromptguardTwoEightSixmBatch}} | {{TransferEnvPromptguardTwoEightSixmLiuenv}} |
| Horizon base | {{TransferEnvHorizonBaseMax}} | {{TransferEnvHorizonBaseOverbound}} | {{TransferEnvHorizonBaseFlips}} | {{TransferEnvHorizonBaseBatch}} | {{TransferEnvHorizonBaseLiuenv}} |

- The shift is already present at batch size 1 and batching changes nothing, so it is not padding. Liu's env
  reproduces his released scores; ours does not.
- **Cause (both found; {{TransferEnvFixStatus}}).** Loaded weights, dtype and attention backend are identical in
  the two envs. (1) PIGuard and Prompt Guard 2: transformers 5 does not load the repo's `tokenizer.json` but rebuilds
  the DeBERTa-v2 tokenizer from `spm.model` with its own converter, which normalizes text differently (e.g. a
  non-breaking space becomes `[UNK]`). (2) Horizon (ModernBERT): its config has no `local_rope_theta`; 4.57 uses the
  class default (10000) for the sliding-window layers, 5.x gives them the global theta (160000). In our env with the
  repo tokenizer (PIGuard, Prompt Guard 2) or the local theta restored (Horizon), max per-row |Δ| to Liu's env is
  {{TransferEnvPiguardFixed}}, {{TransferEnvPromptguardTwoEightSixmFixed}} and {{TransferEnvHorizonBaseFixed}}
  ({{TransferEnvPiguardFixedOverbound}}, {{TransferEnvPromptguardTwoEightSixmFixedOverbound}} and
  {{TransferEnvHorizonBaseFixedOverbound}} rows over 1e-3).
- Not checked: whether other DeBERTa-v2/v3 or ModernBERT detectors (or Wolf Defender, which Liu scored under
  transformers ≥ 5.6) are affected the same way. Our own DeBERTa models are trained and scored with the same
  transformers-5 tokenizer, so their scores are internally consistent.
- Our V2 DeBERTa checkpoints are batch-independent: bs 1 vs 16 max |Δ| {{TransferEnvDebertaBatch}}
  ({{TransferEnvDebertaStatus}}, bound 1e-5).
- Consequence (docs/deviations.md, 2026-10-30): published detectors are scored in the environment their reference
  numbers came from; our models in ours.

## F019. Agent-style synthetic tool outputs: calibration pool (A3), overlap audit and content-free gates

```finding
id: F019
status: full
script: src/agent_pipeline.py -> scripts/agent_overlap_audit.py; scripts/artifact_baselines.py --dataset v3 --data_dir data/agent_v1 --eval_split val --split_seeds 0 1 2 3 4
commit: cb7d881
seeds: split_seeds 0-4 (one build each; calibration pool seed-independent); probes deterministic; bootstrap B=1000 seed 42
data_version: agent_v1 (src/agent_pipeline.py + src/agent_tools.py @ cb7d881; BIPIA text_attack_train.json only)
eval_set: calibration pool; agent_v1 val split of each split seed (gates; probes trained on the same seed's train split)
threshold_protocol: n/a for the pool and audit; gates default tau=0.50 (learned probes), fixed rule, gate = macro-F1 <= 0.60
date: 2026-10-31
source:
- results/agent_v1/agent_overlap_audit/20261031-144736_seed0
- results/agent_v1/artifact_baselines/20261031-144739_seed0
macros:
- AgentVOne*
```

**Calibration pool (A3).** `data/agent_v1/calibration_pool.csv`: {{AgentVOneCalRows}} clean, raw synthetic tool
outputs (19 tool types × 60), each with a user request, sha256 `{{AgentVOneCalSha}}`. Generated from its own
RNG stream; every training realization rejects its texts and never uses it as a donor. Recorded here before any
model is scored on an agent set. The pool is synthetic, from the same templates as the training data (§6 threat:
realized FPR on AD-Clean/TB-Clean is reported next to every calibrated TPR).

**Overlap audit**: {{AgentVOneAuditStatus}}. Cells: texts sharing ≥ 1 word 8-gram with the corpus / reference
instructions contained whole. AgentDojo = the 0.1.34 wheel's suite YAMLs and attack/task strings plus the rebuilt AD
outputs; τ-bench = rebuilt TB outputs and requests; InjecAgent = attacker/user instructions and tool-response
templates; BIPIA test = text and code test attacks. A positive control (corpus texts against their own corpus) hit
on every corpus.

| Unit | n | AgentDojo | τ-bench | InjecAgent | BIPIA test |
|---|---|---|---|---|---|
| Calibration pool outputs | {{AgentVOneAuditCalibrationPoolN}} | {{AgentVOneAuditCalibrationPoolAgentdojoEight}} / {{AgentVOneAuditCalibrationPoolAgentdojoContained}} | {{AgentVOneAuditCalibrationPoolTaubenchEight}} / {{AgentVOneAuditCalibrationPoolTaubenchContained}} | {{AgentVOneAuditCalibrationPoolInjecagentEight}} / {{AgentVOneAuditCalibrationPoolInjecagentContained}} | {{AgentVOneAuditCalibrationPoolBipiaTestEight}} / {{AgentVOneAuditCalibrationPoolBipiaTestContained}} |
| Training/val outputs (span cut out) | {{AgentVOneAuditContextsN}} | {{AgentVOneAuditContextsAgentdojoEight}} / {{AgentVOneAuditContextsAgentdojoContained}} | {{AgentVOneAuditContextsTaubenchEight}} / {{AgentVOneAuditContextsTaubenchContained}} | {{AgentVOneAuditContextsInjecagentEight}} / {{AgentVOneAuditContextsInjecagentContained}} | {{AgentVOneAuditContextsBipiaTestEight}} / {{AgentVOneAuditContextsBipiaTestContained}} |
| Clean-A spans (distinct) | {{AgentVOneAuditSpanCleanAN}} | {{AgentVOneAuditSpanCleanAAgentdojoEight}} / {{AgentVOneAuditSpanCleanAAgentdojoContained}} | {{AgentVOneAuditSpanCleanATaubenchEight}} / {{AgentVOneAuditSpanCleanATaubenchContained}} | {{AgentVOneAuditSpanCleanAInjecagentEight}} / {{AgentVOneAuditSpanCleanAInjecagentContained}} | {{AgentVOneAuditSpanCleanABipiaTestEight}} / {{AgentVOneAuditSpanCleanABipiaTestContained}} |
| Clean-B spans (distinct) | {{AgentVOneAuditSpanCleanBN}} | {{AgentVOneAuditSpanCleanBAgentdojoEight}} / {{AgentVOneAuditSpanCleanBAgentdojoContained}} | {{AgentVOneAuditSpanCleanBTaubenchEight}} / {{AgentVOneAuditSpanCleanBTaubenchContained}} | {{AgentVOneAuditSpanCleanBInjecagentEight}} / {{AgentVOneAuditSpanCleanBInjecagentContained}} | {{AgentVOneAuditSpanCleanBBipiaTestEight}} / {{AgentVOneAuditSpanCleanBBipiaTestContained}} |
| Injected spans (distinct; BIPIA train) | {{AgentVOneAuditSpanInjectedN}} | {{AgentVOneAuditSpanInjectedAgentdojoEight}} / {{AgentVOneAuditSpanInjectedAgentdojoContained}} | {{AgentVOneAuditSpanInjectedTaubenchEight}} / {{AgentVOneAuditSpanInjectedTaubenchContained}} | {{AgentVOneAuditSpanInjectedInjecagentEight}} / {{AgentVOneAuditSpanInjectedInjecagentContained}} | {{AgentVOneAuditSpanInjectedBipiaTestEight}} / {{AgentVOneAuditSpanInjectedBipiaTestContained}} |

**Content-free gates** (V3 rule R2 with tools in place of tasks: pooled macro-F1 ≤ 0.60 on every split seed and
per-tool mean over seeds ≤ 0.60): {{AgentVOneGatesStatus}}. Evaluated on the `val` split (attacks from V3's
held-out val categories of the BIPIA train file), probes trained on `train`.

| Probe | What it uses | Pooled macro-F1, range over seeds | Max per-tool mean (tool) | Gate |
|---|---|---|---|---|
| position_oracle | insertion position == start (metadata) | {{AgentVOnePositionOraclePooledMin}}–{{AgentVOnePositionOraclePooledMax}} | {{AgentVOnePositionOracleToolMax}} ({{AgentVOnePositionOracleToolMaxName}}) | {{AgentVOnePositionOracleStatus}} |
| first_para_rule | first paragraph ≤ 30 words → injected; untrained | {{AgentVOneFirstParaRulePooledMin}}–{{AgentVOneFirstParaRulePooledMax}} | {{AgentVOneFirstParaRuleToolMax}} ({{AgentVOneFirstParaRuleToolMaxName}}) | {{AgentVOneFirstParaRuleStatus}} |
| prefix30_masked_tfidf_lr | first 30 words, span removed | {{AgentVOnePrefixThreeZeroMaskedTfidfLrPooledMin}}–{{AgentVOnePrefixThreeZeroMaskedTfidfLrPooledMax}} | {{AgentVOnePrefixThreeZeroMaskedTfidfLrToolMax}} ({{AgentVOnePrefixThreeZeroMaskedTfidfLrToolMaxName}}) | {{AgentVOnePrefixThreeZeroMaskedTfidfLrStatus}} |
| suffix30_masked_tfidf_lr | last 30 words, span removed | {{AgentVOneSuffixThreeZeroMaskedTfidfLrPooledMin}}–{{AgentVOneSuffixThreeZeroMaskedTfidfLrPooledMax}} | {{AgentVOneSuffixThreeZeroMaskedTfidfLrToolMax}} ({{AgentVOneSuffixThreeZeroMaskedTfidfLrToolMaxName}}) | {{AgentVOneSuffixThreeZeroMaskedTfidfLrStatus}} |
| insertion_site_tfidf_lr | ~15 tokens each side of the insertion point, span removed | {{AgentVOneInsertionSiteTfidfLrPooledMin}}–{{AgentVOneInsertionSiteTfidfLrPooledMax}} | {{AgentVOneInsertionSiteTfidfLrToolMax}} ({{AgentVOneInsertionSiteTfidfLrToolMaxName}}) | {{AgentVOneInsertionSiteTfidfLrStatus}} |
| length_lr | chars and words of the full text | {{AgentVOneLengthLrPooledMin}}–{{AgentVOneLengthLrPooledMax}} | {{AgentVOneLengthLrToolMax}} ({{AgentVOneLengthLrToolMaxName}}) | {{AgentVOneLengthLrStatus}} |
| span_form_you_lr | span-form features + contains you/your | {{AgentVOneSpanFormYouLrPooledMin}}–{{AgentVOneSpanFormYouLrPooledMax}} | {{AgentVOneSpanFormYouLrToolMax}} ({{AgentVOneSpanFormYouLrToolMaxName}}) | {{AgentVOneSpanFormYouLrStatus}} |
| second_person_lr | contains you/your only | {{AgentVOneSecondPersonLrPooledMin}}–{{AgentVOneSecondPersonLrPooledMax}} | {{AgentVOneSecondPersonLrToolMax}} ({{AgentVOneSecondPersonLrToolMaxName}}) | {{AgentVOneSecondPersonLrStatus}} |
| boundary_kind_lr | boundary kind only | {{AgentVOneBoundaryKindLrPooledMin}}–{{AgentVOneBoundaryKindLrPooledMax}} | {{AgentVOneBoundaryKindLrToolMax}} ({{AgentVOneBoundaryKindLrToolMaxName}}) | {{AgentVOneBoundaryKindLrStatus}} |
| prefix30_tfidf_lr | first 30 words, unmasked (diagnostic) | {{AgentVOnePrefixThreeZeroTfidfLrPooledMin}}–{{AgentVOnePrefixThreeZeroTfidfLrPooledMax}} | {{AgentVOnePrefixThreeZeroTfidfLrToolMax}} ({{AgentVOnePrefixThreeZeroTfidfLrToolMaxName}}) | {{AgentVOnePrefixThreeZeroTfidfLrStatus}} |
| suffix30_tfidf_lr | last 30 words, unmasked (diagnostic) | {{AgentVOneSuffixThreeZeroTfidfLrPooledMin}}–{{AgentVOneSuffixThreeZeroTfidfLrPooledMax}} | {{AgentVOneSuffixThreeZeroTfidfLrToolMax}} ({{AgentVOneSuffixThreeZeroTfidfLrToolMaxName}}) | {{AgentVOneSuffixThreeZeroTfidfLrStatus}} |
| span_content_tfidf | inserted span text (content ceiling; diagnostic) | {{AgentVOneSpanContentTfidfPooledMin}}–{{AgentVOneSpanContentTfidfPooledMax}} | {{AgentVOneSpanContentTfidfToolMax}} ({{AgentVOneSpanContentTfidfToolMaxName}}) | {{AgentVOneSpanContentTfidfStatus}} |
| span_form_lr | 9 span-form features, no you/your (diagnostic) | {{AgentVOneSpanFormLrPooledMin}}–{{AgentVOneSpanFormLrPooledMax}} | {{AgentVOneSpanFormLrToolMax}} ({{AgentVOneSpanFormLrToolMaxName}}) | {{AgentVOneSpanFormLrStatus}} |

- Span-form probes score well below 0.5 on val. They learn "shorter span → injected" on train, and V3's val
  categories (Instruction, Programming Help, Social Interaction) are longer strings than the train categories, so the
  rule inverts. This is a property of the BIPIA attack pools, not of the insertion; V3 uses the same categories.
  Form still separates labels somewhat within a split: an uncommitted doc-grouped cross-validation of the
  span-form + you/your LR on split seed 0 train was above 0.60 (a diagnostic, inflated because the 60 attack strings
  repeat across folds; no number from it is reported).
- Clean-A templates were revised three times against these gates on dry runs (scratch builds, not recorded) before
  this run: per-tool template spreading, profile-matched selection (V3 `choose_clean_b`), denser wording.

## F020. Agent-style training data v2: within-train shortcut check and rebuild

```finding
id: F020
status: full
script: scripts/agent_form_cv.py; src/agent_pipeline.py -> scripts/agent_overlap_audit.py; scripts/artifact_baselines.py --dataset v3 --data_dir data/agent_v2 --eval_split val --split_seeds 0 1 2 3 4
commit: 08be6d6
seeds: split_seeds 0-4 (one build each); CV 5-fold GroupKFold by doc_id; probes deterministic; bootstrap B=1000 seed 42
data_version: agent_v2 (src/agent_pipeline.py @ 88400e3; pool = agent_v1, unchanged) and agent_v1 (pre-fix CV only)
eval_set: train split of each split seed (CV); val split (gates); calibration pool + all outputs and spans (audit)
threshold_protocol: CV flag = macro-F1 > 0.60 at tau 0.50; gates as F019 (R2)
date: 2026-10-31
source:
- results/agent_v2/form_cv/20261031-172308_seed0
- results/agent_v1/form_cv/20261031-171136_seed0
- results/agent_v2/agent_overlap_audit/20261031-172311_seed0
- results/agent_v2/artifact_baselines/20261031-172311_seed0
macros:
- AgentVTwo*
- AgentFormCv*
```

F019's span-form gate probes scored far below chance on val, i.e. form predicted the label with the sign flipped
between splits. A 5-fold, doc-grouped CV inside TRAIN measures the signal directly (flag: macro-F1 > 0.60).
agent_v1 was flagged ({{AgentFormCvVOneFlagged}}). agent_v2 matches every clean span jointly (length, cap
ratio, binary form) to an attack of the same split and adds short and length-variant clean-A templates; the
calibration pool is unchanged ({{AgentVTwoCalSha}}; docs/deviations.md 2026-10-31). agent_v2 flagged:
{{AgentFormCvVTwoFlagged}}. Range over split seeds 0–4:

| Data | Probe (within-train CV) | macro-F1 | AUROC |
|---|---|---|---|
| agent_v1 | span_form_you | {{AgentFormCvVOneSpanFormYouFOneMin}}–{{AgentFormCvVOneSpanFormYouFOneMax}} | {{AgentFormCvVOneSpanFormYouAurocMin}}–{{AgentFormCvVOneSpanFormYouAurocMax}} |
| agent_v1 | span_length | {{AgentFormCvVOneSpanLengthFOneMin}}–{{AgentFormCvVOneSpanLengthFOneMax}} | {{AgentFormCvVOneSpanLengthAurocMin}}–{{AgentFormCvVOneSpanLengthAurocMax}} |
| agent_v1 | text_length | {{AgentFormCvVOneTextLengthFOneMin}}–{{AgentFormCvVOneTextLengthFOneMax}} | {{AgentFormCvVOneTextLengthAurocMin}}–{{AgentFormCvVOneTextLengthAurocMax}} |
| agent_v2 | span_form_you | {{AgentFormCvVTwoSpanFormYouFOneMin}}–{{AgentFormCvVTwoSpanFormYouFOneMax}} | {{AgentFormCvVTwoSpanFormYouAurocMin}}–{{AgentFormCvVTwoSpanFormYouAurocMax}} |
| agent_v2 | span_length | {{AgentFormCvVTwoSpanLengthFOneMin}}–{{AgentFormCvVTwoSpanLengthFOneMax}} | {{AgentFormCvVTwoSpanLengthAurocMin}}–{{AgentFormCvVTwoSpanLengthAurocMax}} |
| agent_v2 | text_length | {{AgentFormCvVTwoTextLengthFOneMin}}–{{AgentFormCvVTwoTextLengthFOneMax}} | {{AgentFormCvVTwoTextLengthAurocMin}}–{{AgentFormCvVTwoTextLengthAurocMax}} |

- agent_v2's span-form probe sits just under the 0.60 flag and its AUROC is still above 0.6, so a weak form signal
  remains. The proposed symmetric gate (|AUROC − 0.5| ≤ 0.10; docs/deviations.md, not applied) would fail it.

**Content-free gates, agent_v2** (R2 as F019; agent_v1 values in F019): {{AgentVTwoGatesStatus}}. Only span_form_you_lr fails: pooled
{{AgentVTwoSpanFormYouLrPooledMin}}–{{AgentVTwoSpanFormYouLrPooledMax}} (over 0.60 on two seeds) and several per-tool
means up to {{AgentVTwoSpanFormYouLrToolMax}}. The agent_v1 inversion (below chance) is gone: the residual form signal
now points the same way in train and val. agent_v2 fails R2 on this probe
(see the acceptance below). Uncommitted scratch builds that weighted length and form differently traded the two criteria off:
none had both within-train CV ≤ 0.60 and every gate ≤ 0.60, so tuning stopped rather than fit to the gate threshold.

| Probe | agent_v2 pooled macro-F1 | max per-tool mean (tool) | gate | max \|AUROC − 0.5\| (pooled; reported, not a gate) |
|---|---|---|---|---|
| position_oracle | {{AgentVTwoPositionOraclePooledMin}}–{{AgentVTwoPositionOraclePooledMax}} | {{AgentVTwoPositionOracleToolMax}} ({{AgentVTwoPositionOracleToolMaxName}}) | {{AgentVTwoPositionOracleStatus}} | {{AgentVTwoPositionOracleAurocDevMax}} |
| first_para_rule | {{AgentVTwoFirstParaRulePooledMin}}–{{AgentVTwoFirstParaRulePooledMax}} | {{AgentVTwoFirstParaRuleToolMax}} ({{AgentVTwoFirstParaRuleToolMaxName}}) | {{AgentVTwoFirstParaRuleStatus}} | {{AgentVTwoFirstParaRuleAurocDevMax}} |
| prefix30_masked_tfidf_lr | {{AgentVTwoPrefixThreeZeroMaskedTfidfLrPooledMin}}–{{AgentVTwoPrefixThreeZeroMaskedTfidfLrPooledMax}} | {{AgentVTwoPrefixThreeZeroMaskedTfidfLrToolMax}} ({{AgentVTwoPrefixThreeZeroMaskedTfidfLrToolMaxName}}) | {{AgentVTwoPrefixThreeZeroMaskedTfidfLrStatus}} | {{AgentVTwoPrefixThreeZeroMaskedTfidfLrAurocDevMax}} |
| suffix30_masked_tfidf_lr | {{AgentVTwoSuffixThreeZeroMaskedTfidfLrPooledMin}}–{{AgentVTwoSuffixThreeZeroMaskedTfidfLrPooledMax}} | {{AgentVTwoSuffixThreeZeroMaskedTfidfLrToolMax}} ({{AgentVTwoSuffixThreeZeroMaskedTfidfLrToolMaxName}}) | {{AgentVTwoSuffixThreeZeroMaskedTfidfLrStatus}} | {{AgentVTwoSuffixThreeZeroMaskedTfidfLrAurocDevMax}} |
| insertion_site_tfidf_lr | {{AgentVTwoInsertionSiteTfidfLrPooledMin}}–{{AgentVTwoInsertionSiteTfidfLrPooledMax}} | {{AgentVTwoInsertionSiteTfidfLrToolMax}} ({{AgentVTwoInsertionSiteTfidfLrToolMaxName}}) | {{AgentVTwoInsertionSiteTfidfLrStatus}} | {{AgentVTwoInsertionSiteTfidfLrAurocDevMax}} |
| length_lr | {{AgentVTwoLengthLrPooledMin}}–{{AgentVTwoLengthLrPooledMax}} | {{AgentVTwoLengthLrToolMax}} ({{AgentVTwoLengthLrToolMaxName}}) | {{AgentVTwoLengthLrStatus}} | {{AgentVTwoLengthLrAurocDevMax}} |
| span_form_you_lr | {{AgentVTwoSpanFormYouLrPooledMin}}–{{AgentVTwoSpanFormYouLrPooledMax}} | {{AgentVTwoSpanFormYouLrToolMax}} ({{AgentVTwoSpanFormYouLrToolMaxName}}) | {{AgentVTwoSpanFormYouLrStatus}} | {{AgentVTwoSpanFormYouLrAurocDevMax}} |
| second_person_lr | {{AgentVTwoSecondPersonLrPooledMin}}–{{AgentVTwoSecondPersonLrPooledMax}} | {{AgentVTwoSecondPersonLrToolMax}} ({{AgentVTwoSecondPersonLrToolMaxName}}) | {{AgentVTwoSecondPersonLrStatus}} | {{AgentVTwoSecondPersonLrAurocDevMax}} |
| boundary_kind_lr | {{AgentVTwoBoundaryKindLrPooledMin}}–{{AgentVTwoBoundaryKindLrPooledMax}} | {{AgentVTwoBoundaryKindLrToolMax}} ({{AgentVTwoBoundaryKindLrToolMaxName}}) | {{AgentVTwoBoundaryKindLrStatus}} | {{AgentVTwoBoundaryKindLrAurocDevMax}} |
| span_form_lr | {{AgentVTwoSpanFormLrPooledMin}}–{{AgentVTwoSpanFormLrPooledMax}} | {{AgentVTwoSpanFormLrToolMax}} ({{AgentVTwoSpanFormLrToolMaxName}}) | {{AgentVTwoSpanFormLrStatus}} | {{AgentVTwoSpanFormLrAurocDevMax}} |
| span_content_tfidf | {{AgentVTwoSpanContentTfidfPooledMin}}–{{AgentVTwoSpanContentTfidfPooledMax}} | {{AgentVTwoSpanContentTfidfToolMax}} ({{AgentVTwoSpanContentTfidfToolMaxName}}) | {{AgentVTwoSpanContentTfidfStatus}} | {{AgentVTwoSpanContentTfidfAurocDevMax}} |

**Overlap audit, agent_v2**: {{AgentVTwoAuditStatus}} (cells: texts sharing an 8-gram / contained instructions).

| Unit | n | AgentDojo | τ-bench | InjecAgent | BIPIA test |
|---|---|---|---|---|---|
| calibration_pool | {{AgentVTwoAuditCalibrationPoolN}} | {{AgentVTwoAuditCalibrationPoolAgentdojoEight}} / {{AgentVTwoAuditCalibrationPoolAgentdojoContained}} | {{AgentVTwoAuditCalibrationPoolTaubenchEight}} / {{AgentVTwoAuditCalibrationPoolTaubenchContained}} | {{AgentVTwoAuditCalibrationPoolInjecagentEight}} / {{AgentVTwoAuditCalibrationPoolInjecagentContained}} | {{AgentVTwoAuditCalibrationPoolBipiaTestEight}} / {{AgentVTwoAuditCalibrationPoolBipiaTestContained}} |
| contexts | {{AgentVTwoAuditContextsN}} | {{AgentVTwoAuditContextsAgentdojoEight}} / {{AgentVTwoAuditContextsAgentdojoContained}} | {{AgentVTwoAuditContextsTaubenchEight}} / {{AgentVTwoAuditContextsTaubenchContained}} | {{AgentVTwoAuditContextsInjecagentEight}} / {{AgentVTwoAuditContextsInjecagentContained}} | {{AgentVTwoAuditContextsBipiaTestEight}} / {{AgentVTwoAuditContextsBipiaTestContained}} |
| span_clean_a | {{AgentVTwoAuditSpanCleanAN}} | {{AgentVTwoAuditSpanCleanAAgentdojoEight}} / {{AgentVTwoAuditSpanCleanAAgentdojoContained}} | {{AgentVTwoAuditSpanCleanATaubenchEight}} / {{AgentVTwoAuditSpanCleanATaubenchContained}} | {{AgentVTwoAuditSpanCleanAInjecagentEight}} / {{AgentVTwoAuditSpanCleanAInjecagentContained}} | {{AgentVTwoAuditSpanCleanABipiaTestEight}} / {{AgentVTwoAuditSpanCleanABipiaTestContained}} |
| span_clean_b | {{AgentVTwoAuditSpanCleanBN}} | {{AgentVTwoAuditSpanCleanBAgentdojoEight}} / {{AgentVTwoAuditSpanCleanBAgentdojoContained}} | {{AgentVTwoAuditSpanCleanBTaubenchEight}} / {{AgentVTwoAuditSpanCleanBTaubenchContained}} | {{AgentVTwoAuditSpanCleanBInjecagentEight}} / {{AgentVTwoAuditSpanCleanBInjecagentContained}} | {{AgentVTwoAuditSpanCleanBBipiaTestEight}} / {{AgentVTwoAuditSpanCleanBBipiaTestContained}} |
| span_injected | {{AgentVTwoAuditSpanInjectedN}} | {{AgentVTwoAuditSpanInjectedAgentdojoEight}} / {{AgentVTwoAuditSpanInjectedAgentdojoContained}} | {{AgentVTwoAuditSpanInjectedTaubenchEight}} / {{AgentVTwoAuditSpanInjectedTaubenchContained}} | {{AgentVTwoAuditSpanInjectedInjecagentEight}} / {{AgentVTwoAuditSpanInjectedInjecagentContained}} | {{AgentVTwoAuditSpanInjectedBipiaTestEight}} / {{AgentVTwoAuditSpanInjectedBipiaTestContained}} |

**Accepted for training (author decision, 2026-10-31; docs/deviations.md).** agent_v2 is used for ICT training
despite the span_form_you_lr failure, without further rebuilds, to avoid tuning the data against the gate. The agent
data is training-only and transfer is measured on independent external sets (AD/TB), so a residual training-data
shortcut can lower but not inflate transfer results; ICT ablation (d) measures the agent data's contribution.
The |AUROC − 0.5| column is a transparency metric, not a retroactive pass/fail criterion.

## F021. Published detectors on the agent transfer sets: calibrated and oracle protocols

```finding
id: F021
status: full
script: jobs/transfer_scores_{encoders,wolf,judges}.sh (eval/transfer/detectors.py, record_scores.py) -> eval/transfer/transfer_metrics.py
commit: a798dc1
seeds: n/a (deterministic scoring; judges greedy); bootstrap seed 42, B = 1000
data_version: transfer-bib5d0ae7c-agentdojo0.1.34-tb59a200c-filesfb0afded
eval_set: AD-Matched, AD-Shift (recovered pairing), TB-Matched; negatives AD-Clean / TB-Clean; calibration pool (A3)
threshold_protocol: calibrated (1%/5% FPR on the calibration pool, primary) and oracle (on the evaluated clean rows, MIN_CLEAN gates)
date: 2026-11-07 (re-rendered from run 20261107-210130_seed42, same numbers plus truncation marks, F023; adds protectai-small-v2; encoder scores re-run 2026-11-06, job 434238, run 20261106-144044_seed0. Earlier versions 2026-11-02, run 20261102-115039_seed42, encoder run 20261031-182528_seed0; 2026-10-31, run 20261031-190127_seed42, which also listed the 4-bit judges)
source:
- results/transfer/transfer_metrics/20261107-210130_seed42
upstream:
- results/transfer/transfer_scores_liuenv/20261106-144044_seed0
- results/transfer/transfer_scores_wolf/20261031-182747_seed0
- results/transfer/transfer_scores_judges_bf16/20261031-190043_seed0
macros:
- TransferDet*
- "!TransferDet*Halves*"
```

Every published detector Liu (2026) scored, including protectai-small-v2 (access granted 2026-11-02; scored
2026-11-06 by job 434238 after two jobs failed on a faulty GPU), scored in his environment (Wolf Defender in ours,
transformers 5.20.1, as he did), plus his two LLM judges with his Yes/No rule.
**Judge precision: bf16** (A20; author decision 2026-11-02). Every published-detector baseline and every H3
comparison uses the bf16 judges. The 4-bit NF4 judges (Liu's setting) are reported only for the Liu reproduction
(Gate 2, F017); their transfer scores (`transfer_scores_judges_nf4/20261031-185021_seed0`) are kept but enter no table. Sets from `eval/transfer/sets_adliu.py`: AD-Matched/AD-Shift with the recovered pairing
(deviations 2026-10-31), TB-Matched, AD-Clean, TB-Clean, and the 1,140-row calibration pool as jsonl. On AgentDojo
the negatives are AD-Clean (A16). Clusters for every CI: user task. Full table: `paper/tables/transfer_detectors.tex`.

- **The calibration pool's thresholds do not transfer evenly.** At a 1% FPR target the realized FPR on AD-Clean
  ranges from {{TransferDetHorizonBaseLiuenvAdmatchedCalOnepctFprFpr}} (Horizon) to
  {{TransferDetPiguardLiuenvAdmatchedCalOnepctFprFpr}} (PIGuard) and
  {{TransferDetTestsavantLargeLiuenvAdmatchedCalOnepctFprFpr}} (TestSavant large). JasperLS flags
  {{TransferDetJasperlsLiuenvCalpoolFlag}} of the calibration pool at 0.5, so its calibrated threshold sits near 1.
  Calibrated TPR must always be read next to its realized FPR (the risk paper_plan §8 names).
- Strongest calibrated operating point on AD-Matched at a realized FPR ≤ 1%: Horizon, TPR@1%FPR
  {{TransferDetHorizonBaseLiuenvAdmatchedCalOnepctFprTpr}} [{{TransferDetHorizonBaseLiuenvAdmatchedCalOnepctFprTprCiLo}},
  {{TransferDetHorizonBaseLiuenvAdmatchedCalOnepctFprTprCiHi}}]. The bf16 Qwen judge reaches
  {{TransferDetJudgeQwenJudgesBfOneSixAdmatchedCalOnepctFprTpr}} but at a realized FPR of
  {{TransferDetJudgeQwenJudgesBfOneSixAdmatchedCalOnepctFprFpr}}.
- **task_block_rate** (Liu's definition: share of user tasks with at least one clean output flagged; CI resamples
  tasks). At 0.5, PIGuard blocks {{TransferDetPiguardLiuenvAdcleanBlockAtZeroFive}} of AD-Clean tasks and
  {{TransferDetPiguardLiuenvTbcleanBlockAtZeroFive}} of TB-Clean tasks; Horizon
  {{TransferDetHorizonBaseLiuenvAdcleanBlockAtZeroFive}} and {{TransferDetHorizonBaseLiuenvTbcleanBlockAtZeroFive}};
  the bf16 judges {{TransferDetJudgeQwenJudgesBfOneSixAdcleanBlockAtZeroFive}} (Qwen) and
  {{TransferDetJudgeLlamaJudgesBfOneSixAdcleanBlockAtZeroFive}} (Llama) on AD-Clean. At the calibration pool's 1%
  threshold PIGuard still blocks {{TransferDetPiguardLiuenvAdcleanBlockCalibratedOnepctFpr}} of AD-Clean tasks.
  Every detector, at 0.5 and both calibrated thresholds, is in the run's `metrics.json` (`*_task_block_rate`).
- Oracle TPR@1%FPR on AgentDojo is `n/a` (AD-Clean has 339 rows < 500); oracle TPR@5%FPR is reported per cell.
- **Truncation (F023).** Detectors are scored at their native max length. A dagger in the table marks a cell where
  more than 25% of the set's injected or clean rows were truncated: TB-Matched for the 512-token classifiers and
  Prismor. Those numbers reflect truncated inputs.
- No hypothesis is decided here: these are the published-detector baselines H1–H3 compare against.

## F022. Calibration transfer: realized FPR of the calibration-pool thresholds on the agent sets

```finding
id: F022
status: full
script: eval/transfer/transfer_metrics.py (task_halves; calibration-pool thresholds, paper_plan A3) -> scripts/make_tables.py
commit: a798dc1
seeds: n/a (deterministic scores); task-half split seed 0 (sha256 of "0:<suite>/<task>"); bootstrap seed 42, B = 1000
data_version: transfer-bib5d0ae7c-agentdojo0.1.34-tb59a200c-filesfb0afded
eval_set: AD-Matched, AD-Shift, TB-Matched (negatives AD-Clean / TB-Clean); calibration pool (A3)
threshold_protocol: calibrated on the calibration pool (primary) vs task-disjoint halves (secondary, A3; never decides)
date: 2026-11-07 (adds protectai-small-v2; previous version 2026-11-02, run 20261102-115039_seed42)
source:
- results/transfer/transfer_metrics/20261107-210130_seed42
upstream:
- results/transfer/transfer_scores_liuenv/20261106-144044_seed0
- results/transfer/transfer_scores_wolf/20261031-182747_seed0
- results/transfer/transfer_scores_judges_bf16/20261031-190043_seed0
macros:
- TransferCaltransfer*
- TransferDet*Halves*
```

The calibration pool (1,140 synthetic clean tool outputs, F019) fixes every agent-set threshold (A3). This entry
measures how far those thresholds miss their FPR target on the agent sets' real clean outputs, for the
{{TransferCaltransferAdmatchedOnepctFprN}} detectors of F021 (bf16 judges).

- **AgentDojo (AD-Clean negatives; AD-Matched and AD-Shift share them, so their FPR rows are identical).** At the 1%
  target, {{TransferCaltransferAdmatchedOnepctFprAbove}} of {{TransferCaltransferAdmatchedOnepctFprN}} detectors
  have a realized-FPR CI entirely above 1% and {{TransferCaltransferAdmatchedOnepctFprBelow}} entirely below; median
  realized FPR {{TransferCaltransferAdmatchedOnepctFprMedian}}. At 5%:
  {{TransferCaltransferAdmatchedFivepctFprAbove}} above, {{TransferCaltransferAdmatchedFivepctFprBelow}} below,
  median {{TransferCaltransferAdmatchedFivepctFprMedian}}.
- **τ-bench (TB-Clean negatives).** At 1%: {{TransferCaltransferTbevalOnepctFprAbove}} above,
  {{TransferCaltransferTbevalOnepctFprBelow}} below, median {{TransferCaltransferTbevalOnepctFprMedian}}. At 5%:
  {{TransferCaltransferTbevalFivepctFprAbove}} above, {{TransferCaltransferTbevalFivepctFprBelow}} below, median
  {{TransferCaltransferTbevalFivepctFprMedian}}. The miss is smaller on τ-bench than on AgentDojo.
- **In-domain contrast (secondary protocol, task-disjoint halves).** Each set's user tasks are split 50/50 by a
  seeded hash; thresholds from the calibration half's clean rows, evaluated on the other half, next to the pool
  threshold on the same evaluation half. At 5% on AgentDojo the median realized FPR is
  {{TransferCaltransferAdmatchedFivepctFprHalvesTaskHalfMedian}} with in-domain thresholds against
  {{TransferCaltransferAdmatchedFivepctFprHalvesCalibrationPoolMedian}} with the pool's; on τ-bench at 1%,
  {{TransferCaltransferTbevalOnepctFprHalvesTaskHalfMedian}} against
  {{TransferCaltransferTbevalOnepctFprHalvesCalibrationPoolMedian}}. Example: PIGuard on AgentDojo at 5%,
  {{TransferDetPiguardLiuenvAdmatchedHalvesTaskHalfFivepctFprFpr}} (in-domain) vs
  {{TransferDetPiguardLiuenvAdmatchedHalvesCalibrationPoolFivepctFprFpr}} (pool).
- In-domain TPR@1%FPR on AgentDojo is `n/a` for every detector: the calibration half has 169 AD-Clean rows (< 500,
  MIN_CLEAN). Half sizes: AD 47/50 user tasks, TB 348/327 (in `metrics.json`). The split depends only on task ids;
  it was fixed in code before these metrics were computed, but after the scores existed.
- Reading: for most detectors the pool's clean outputs score lower than AgentDojo's (cause not tested), so the pool
  thresholds sit too low there and calibrated TPR on AD comes with a realized FPR well above target. This is
  the A3 risk (paper_plan §8): every calibrated TPR is reported next to its realized FPR, and H1 still uses the pool
  (pre-registered); the halves protocol is reported, never decides.

## F023. Published detectors: inputs over their max length on the agent transfer sets

```finding
id: F023
status: full
script: eval/transfer/transfer_metrics.py (over_max_len, from the score records) -> scripts/make_tables.py
commit: a798dc1
seeds: n/a (counts from deterministic scoring)
data_version: transfer-bib5d0ae7c-agentdojo0.1.34-tb59a200c-filesfb0afded
eval_set: AD-Matched, AD-Shift, TB-Matched, AD-Clean, TB-Clean, calibration pool (A3)
threshold_protocol: n/a (no threshold; mark at > 25% of a set's injected or clean rows truncated, chosen 2026-11-07 with these fractions in view, display only)
date: 2026-11-07
source:
- results/transfer/transfer_metrics/20261107-210130_seed42
upstream:
- results/transfer/transfer_scores_liuenv/20261106-144044_seed0
- results/transfer/transfer_scores_wolf/20261031-182747_seed0
- results/transfer/transfer_scores_judges_bf16/20261031-190043_seed0
macros:
- TransferTrunc*
```

Every published detector is scored at its native max length, as Liu (2026) does and as each detector ships
(author decision 2026-11-07: scoring unchanged, truncation made visible). For each detector and set, the fraction of
rows longer than the detector's window (tokens, its own tokenizer), recorded at scoring time
(`eval/transfer/detectors.py`). How long rows are handled: **truncated** (classifiers and Prismor: tokens past the
window are dropped), **windowed** (Prompt Guard, Sheltron: every window is scored, maximum taken; nothing is lost),
or the whole input (LLM judges). Table: `paper/tables/transfer_truncation.tex`.

- **TB-Matched is where truncation bites.** {{TransferTruncTbevalHighN}} truncating detectors drop tokens on more than
  {{TransferTruncThreshold}} of TB-Matched's rows or its clean rows: the 512-token classifiers (ProtectAI v1
  {{TransferTruncProtectaiVOneLiuenvTbeval}}, ProtectAI v2, ProtectAI small v2, PIGuard, JasperLS, TestSavant large,
  all {{TransferTruncProtectaiSmallVTwoLiuenvTbeval}} of TB-Matched and {{TransferTruncProtectaiSmallVTwoLiuenvTbclean}}
  of TB-Clean) and Prismor ({{TransferTruncPrismorOneFivebLiuenvTbeval}} / {{TransferTruncPrismorOneFivebLiuenvTbclean}}).
  Their TB-Matched numbers in F021 reflect truncated inputs and are marked there (dagger).
- **AgentDojo is mostly within the window:** {{TransferTruncAdmatchedHighN}} detectors above the mark on AD-Matched
  and {{TransferTruncAdshiftHighN}} on AD-Shift; the 512-token classifiers truncate
  {{TransferTruncProtectaiSmallVTwoLiuenvAdmatched}} of AD-Matched and {{TransferTruncProtectaiSmallVTwoLiuenvAdclean}}
  of AD-Clean. The calibration pool is within every window except Prismor's ({{TransferTruncPrismorOneFivebLiuenvCalpool}}).
- The 2,048-token detectors (Horizon, Wolf Defender, Qualifire Sentinel, Rogue Sentinel v2) truncate at most
  {{TransferTruncHorizonBaseLiuenvAdmatched}} of any set.
- Not tested: whether the dropped tokens contain the injection (position-dependent), so the effect on TPR is not
  measured here; truncation on clean rows can also move FPR either way.

## F024. V3 frozen build: content-free gates (split seeds 0–4, reviewed benign pool)

```finding
id: F024
status: full
script: scripts/artifact_baselines.py --dataset v3 --split_seeds 0 1 2 3 4
commit: 57344a6
seeds: split_seeds 0-4 (one build each); probes deterministic; bootstrap B=1000 seed 42
data_version: v3 (tag v3-frozen; data/v3 committed at 57344a6; built by src/data_pipeline_v3.py, last changed at 4cdae64, from data/benign/benign_v3.json, externally reviewed)
eval_set: v3 test split of each split seed (rows with an inserted span); probes trained on the same seed's train split
threshold_protocol: default tau=0.50 (learned probes); fixed rule; gate = macro-F1 <= 0.60
date: '2026-11-11'
source: results/v3/artifact_baselines_frozen/20261111-012417_seed0
cluster: doc_id
macros:
- VThreeFrozenGate*
- VThreeFrozenGates*
- VThreeFrozenPositionOracle*
- VThreeFrozenFirstParaRule*
- VThreeFrozenPrefixThreeZero*
- VThreeFrozenSuffixThreeZero*
- VThreeFrozenInsertionSite*
- VThreeFrozenLength*
- VThreeFrozenSpanForm*
- VThreeFrozenSecondPerson*
- VThreeFrozenBoundaryKind*
- VThreeFrozenSpanContent*
```

Part of claim C1/C2 (V3 removes the V2 construction artifacts). Gates of the frozen V3 build (`data/v3`, tag
`v3-frozen`), built from the benign pool after the external full review (docs/deviations.md, 2026-11-11). Supersedes
F013 (draft pool) for every use. Gate rule (v3_design §8, decision R2): pass = pooled macro-F1 ≤
{{VThreeFrozenGateThreshold}} on every split seed **and** per-task mean over seeds ≤ {{VThreeFrozenGateThreshold}}.
Overall: {{VThreeFrozenGatesStatus}}. Rows marked "diag" are reported diagnostics, not gates (decision R1). Full table:
`paper/tables/v3_frozen_gates.tex`. The same build in a scratch folder before the freeze
(`results/v3/artifact_baselines_reviewed/20261111-004117_seed0`, same commit of the builder, byte-identical CSVs) gave
identical metrics and predictions; only the `eval_set` label differs (`v3_reviewed_test`).

| Probe | What it uses | Pooled macro-F1, range over seeds | EmailQA mean | TableQA mean | CodeQA mean | Gate |
|---|---|---|---|---|---|---|
| position_oracle | LR on position, separator, boundary_kind, relative offset | {{VThreeFrozenPositionOraclePooledMin}}–{{VThreeFrozenPositionOraclePooledMax}} | {{VThreeFrozenPositionOracleTaskMeanEmail}} | {{VThreeFrozenPositionOracleTaskMeanTable}} | {{VThreeFrozenPositionOracleTaskMeanCode}} | {{VThreeFrozenPositionOracleStatus}} |
| first_para_rule | first paragraph ≤ 30 words → injected; untrained | {{VThreeFrozenFirstParaRulePooledMin}}–{{VThreeFrozenFirstParaRulePooledMax}} | {{VThreeFrozenFirstParaRuleTaskMeanEmail}} | {{VThreeFrozenFirstParaRuleTaskMeanTable}} | {{VThreeFrozenFirstParaRuleTaskMeanCode}} | {{VThreeFrozenFirstParaRuleStatus}} |
| prefix30_masked_tfidf_lr | first 30 words, span removed | {{VThreeFrozenPrefixThreeZeroMaskedTfidfLrPooledMin}}–{{VThreeFrozenPrefixThreeZeroMaskedTfidfLrPooledMax}} | {{VThreeFrozenPrefixThreeZeroMaskedTfidfLrTaskMeanEmail}} | {{VThreeFrozenPrefixThreeZeroMaskedTfidfLrTaskMeanTable}} | {{VThreeFrozenPrefixThreeZeroMaskedTfidfLrTaskMeanCode}} | {{VThreeFrozenPrefixThreeZeroMaskedTfidfLrStatus}} |
| suffix30_masked_tfidf_lr | last 30 words, span removed | {{VThreeFrozenSuffixThreeZeroMaskedTfidfLrPooledMin}}–{{VThreeFrozenSuffixThreeZeroMaskedTfidfLrPooledMax}} | {{VThreeFrozenSuffixThreeZeroMaskedTfidfLrTaskMeanEmail}} | {{VThreeFrozenSuffixThreeZeroMaskedTfidfLrTaskMeanTable}} | {{VThreeFrozenSuffixThreeZeroMaskedTfidfLrTaskMeanCode}} | {{VThreeFrozenSuffixThreeZeroMaskedTfidfLrStatus}} |
| insertion_site_tfidf_lr | ~15 tokens each side of the insertion point, span removed | {{VThreeFrozenInsertionSiteTfidfLrPooledMin}}–{{VThreeFrozenInsertionSiteTfidfLrPooledMax}} | {{VThreeFrozenInsertionSiteTfidfLrTaskMeanEmail}} | {{VThreeFrozenInsertionSiteTfidfLrTaskMeanTable}} | {{VThreeFrozenInsertionSiteTfidfLrTaskMeanCode}} | {{VThreeFrozenInsertionSiteTfidfLrStatus}} |
| length_lr | chars, words, tokens of the full input | {{VThreeFrozenLengthLrPooledMin}}–{{VThreeFrozenLengthLrPooledMax}} | {{VThreeFrozenLengthLrTaskMeanEmail}} | {{VThreeFrozenLengthLrTaskMeanTable}} | {{VThreeFrozenLengthLrTaskMeanCode}} | {{VThreeFrozenLengthLrStatus}} |
| span_form_you_lr | span-form features + contains you/your | {{VThreeFrozenSpanFormYouLrPooledMin}}–{{VThreeFrozenSpanFormYouLrPooledMax}} | {{VThreeFrozenSpanFormYouLrTaskMeanEmail}} | {{VThreeFrozenSpanFormYouLrTaskMeanTable}} | {{VThreeFrozenSpanFormYouLrTaskMeanCode}} | {{VThreeFrozenSpanFormYouLrStatus}} |
| second_person_lr | contains you/your only | {{VThreeFrozenSecondPersonLrPooledMin}}–{{VThreeFrozenSecondPersonLrPooledMax}} | {{VThreeFrozenSecondPersonLrTaskMeanEmail}} | {{VThreeFrozenSecondPersonLrTaskMeanTable}} | {{VThreeFrozenSecondPersonLrTaskMeanCode}} | {{VThreeFrozenSecondPersonLrStatus}} |
| boundary_kind_lr | boundary kind only | {{VThreeFrozenBoundaryKindLrPooledMin}}–{{VThreeFrozenBoundaryKindLrPooledMax}} | {{VThreeFrozenBoundaryKindLrTaskMeanEmail}} | {{VThreeFrozenBoundaryKindLrTaskMeanTable}} | {{VThreeFrozenBoundaryKindLrTaskMeanCode}} | {{VThreeFrozenBoundaryKindLrStatus}} |
| prefix30_tfidf_lr | first 30 words, unmasked (diagnostic) | {{VThreeFrozenPrefixThreeZeroTfidfLrPooledMin}}–{{VThreeFrozenPrefixThreeZeroTfidfLrPooledMax}} | {{VThreeFrozenPrefixThreeZeroTfidfLrTaskMeanEmail}} | {{VThreeFrozenPrefixThreeZeroTfidfLrTaskMeanTable}} | {{VThreeFrozenPrefixThreeZeroTfidfLrTaskMeanCode}} | {{VThreeFrozenPrefixThreeZeroTfidfLrStatus}} |
| suffix30_tfidf_lr | last 30 words, unmasked (diagnostic) | {{VThreeFrozenSuffixThreeZeroTfidfLrPooledMin}}–{{VThreeFrozenSuffixThreeZeroTfidfLrPooledMax}} | {{VThreeFrozenSuffixThreeZeroTfidfLrTaskMeanEmail}} | {{VThreeFrozenSuffixThreeZeroTfidfLrTaskMeanTable}} | {{VThreeFrozenSuffixThreeZeroTfidfLrTaskMeanCode}} | {{VThreeFrozenSuffixThreeZeroTfidfLrStatus}} |
| span_content_tfidf | inserted span text (content ceiling; diagnostic) | {{VThreeFrozenSpanContentTfidfPooledMin}}–{{VThreeFrozenSpanContentTfidfPooledMax}} | {{VThreeFrozenSpanContentTfidfTaskMeanEmail}} | {{VThreeFrozenSpanContentTfidfTaskMeanTable}} | {{VThreeFrozenSpanContentTfidfTaskMeanCode}} | {{VThreeFrozenSpanContentTfidfStatus}} |
| span_form_lr | 9 span-form features, no you/your (diagnostic) | {{VThreeFrozenSpanFormLrPooledMin}}–{{VThreeFrozenSpanFormLrPooledMax}} | {{VThreeFrozenSpanFormLrTaskMeanEmail}} | {{VThreeFrozenSpanFormLrTaskMeanTable}} | {{VThreeFrozenSpanFormLrTaskMeanCode}} | {{VThreeFrozenSpanFormLrStatus}} |

- The content ceiling `span_content_tfidf` (span text alone) is high on CodeQA (per-task mean
  {{VThreeFrozenSpanContentTfidfTaskMeanCode}}): code attacks and benign code items remain lexically separable
  (the attack-indicative tokens in the form report include `server`, `smtp`, `psutil`, `gmail`). This is content,
  not a construction artifact, but code results measure a lexically easy task.
- Unmasked prefix-30 and suffix-30 diagnostics exceed 0.60 on some seeds or tasks. They see the inserted span
  itself, so they are content probes, not content-free gates; the span-masked versions sit at 0.3333 (constant
  prediction) on every seed.

## F025. V3 frozen build: integrity checks (split seeds 0–4, reviewed benign pool)

```finding
id: F025
status: full
script: scripts/v3_integrity.py --split_seeds 0 1 2 3 4
commit: 57344a6
seeds: split_seeds 0-4
data_version: v3 (tag v3-frozen; data/v3 committed at 57344a6; built by src/data_pipeline_v3.py, last changed at 4cdae64, from data/benign/benign_v3.json, externally reviewed)
eval_set: all written CSVs of each split seed (train/val/test, val_id, raw_clean, ambiguous, long)
threshold_protocol: n/a (assertions)
date: '2026-11-11'
source: results/v3/integrity_frozen/20261111-014556_seed0
macros:
- VThreeFrozenIntegrity*
```

Frozen V3 build (`data/v3`, tag `v3-frozen`); supersedes F014. Identical to the scratch pre-freeze run `results/v3/integrity_reviewed/20261111-010256_seed0`. Independent re-read of the written CSVs: doc_id, exact text, near-duplicate component, attack string, benign item
and donor text disjoint across train/val/test; train/val attack categories disjoint; test attacks only from the test
files; donor docs from the same task and split; ambiguous items never used as clean A; exact 1:1 balance per
task × split; 4 rows (2 injected) per doc; eval-slice rows belong to their split's docs; ≤ 512 DeBERTa-v3 tokens
per main, val_id, raw_clean and ambiguous row (re-tokenized; the stored count matches); every span recoverable from
its offset. Overall: {{VThreeFrozenIntegrityStatus}}.

| Split seed | Integrity | Main rows | val_id | raw_clean | ambiguous | long |
|---|---|---|---|---|---|---|
| 0 | {{VThreeFrozenIntegritySeedZeroStatus}} | {{VThreeFrozenIntegritySeedZeroRowsMain}} | {{VThreeFrozenIntegritySeedZeroRowsValId}} | {{VThreeFrozenIntegritySeedZeroRowsRawClean}} | {{VThreeFrozenIntegritySeedZeroRowsAmbiguous}} | {{VThreeFrozenIntegritySeedZeroRowsLong}} |
| 1 | {{VThreeFrozenIntegritySeedOneStatus}} | {{VThreeFrozenIntegritySeedOneRowsMain}} | {{VThreeFrozenIntegritySeedOneRowsValId}} | {{VThreeFrozenIntegritySeedOneRowsRawClean}} | {{VThreeFrozenIntegritySeedOneRowsAmbiguous}} | {{VThreeFrozenIntegritySeedOneRowsLong}} |
| 2 | {{VThreeFrozenIntegritySeedTwoStatus}} | {{VThreeFrozenIntegritySeedTwoRowsMain}} | {{VThreeFrozenIntegritySeedTwoRowsValId}} | {{VThreeFrozenIntegritySeedTwoRowsRawClean}} | {{VThreeFrozenIntegritySeedTwoRowsAmbiguous}} | {{VThreeFrozenIntegritySeedTwoRowsLong}} |
| 3 | {{VThreeFrozenIntegritySeedThreeStatus}} | {{VThreeFrozenIntegritySeedThreeRowsMain}} | {{VThreeFrozenIntegritySeedThreeRowsValId}} | {{VThreeFrozenIntegritySeedThreeRowsRawClean}} | {{VThreeFrozenIntegritySeedThreeRowsAmbiguous}} | {{VThreeFrozenIntegritySeedThreeRowsLong}} |
| 4 | {{VThreeFrozenIntegritySeedFourStatus}} | {{VThreeFrozenIntegritySeedFourRowsMain}} | {{VThreeFrozenIntegritySeedFourRowsValId}} | {{VThreeFrozenIntegritySeedFourRowsRawClean}} | {{VThreeFrozenIntegritySeedFourRowsAmbiguous}} | {{VThreeFrozenIntegritySeedFourRowsLong}} |

Main rows per task/split differ slightly across seeds (the total is the same); split seed 0: EmailQA {{VThreeFrozenIntegritySeedZeroMainEmailTrain}} / {{VThreeFrozenIntegritySeedZeroMainEmailVal}} /
{{VThreeFrozenIntegritySeedZeroMainEmailTest}}, TableQA {{VThreeFrozenIntegritySeedZeroMainTableTrain}} / {{VThreeFrozenIntegritySeedZeroMainTableVal}} / {{VThreeFrozenIntegritySeedZeroMainTableTest}}, CodeQA
{{VThreeFrozenIntegritySeedZeroMainCodeTrain}} / {{VThreeFrozenIntegritySeedZeroMainCodeVal}} / {{VThreeFrozenIntegritySeedZeroMainCodeTest}} (train / val / test).

## F026. V3 DeBERTa sanity checks: attack-string overlap, per-attack recall, span style probe, val overlap

```finding
id: F026
status: full
script: scripts/v3_sanity.py
commit: 19cb0b8
seeds: split_seeds 0-4 (one V3 DeBERTa run each, training seed = split seed); style probe LR random_state 0; bootstrap B=1000 seed 42
data_version: v3 (tag v3-frozen)
eval_set: V3 test (and val, for overlap) of each split seed; style probe trained on each seed's train spans
threshold_protocol: default 0.5 (DeBERTa and the style probe)
date: '2026-11-12'
source: results/v3/v3_sanity/20261112-010736_seed42
upstream:
- results/v3/score_deberta_v3/20261111-174708_seed0
- results/v3/score_deberta_v3/20261111-181158_seed1
- results/v3/score_deberta_v3/20261111-183653_seed2
- results/v3/score_deberta_v3/20261111-174506_seed3
- results/v3/score_deberta_v3/20261111-180717_seed4
- results/v3/deberta_v3_train/20261111-172858_seed0
- results/v3/deberta_v3_train/20261111-175345_seed1
- results/v3/deberta_v3_train/20261111-181836_seed2
- results/v3/deberta_v3_train/20261111-172858_seed3
- results/v3/deberta_v3_train/20261111-175057_seed4
cluster: doc_id (stratified by task); attack string for the per-attack recall CI
macros:
- VThreeSanity*
```

Read-only checks of the V3 DeBERTa result (runs trained and scored on Kaggle, 2× Tesla T4, fp32, commit f5780ad;
docs/deviations.md 2026-11-11). Nothing in the data or the models was changed.

**1. Attack strings.** Unique attack strings per split seed: train {{VThreeSanityUniqueTrainMin}}–{{VThreeSanityUniqueTrainMax}},
val {{VThreeSanityUniqueValMin}}–{{VThreeSanityUniqueValMax}}, test {{VThreeSanityUniqueTestMin}}–{{VThreeSanityUniqueTestMax}}
(val_id, train-category attacks on val docs: {{VThreeSanityUniqueValIdMin}}–{{VThreeSanityUniqueValIdMax}}). Test strings
found exactly in train: {{VThreeSanityTestExactInTrain}}; with a train near-duplicate (char 5-gram Jaccard ≥ 0.5 after
lowercasing and whitespace collapse): {{VThreeSanityTestNearDups}} on every seed (largest Jaccard to any train string
{{VThreeSanityTestMaxJaccard}}). Removing near-duplicates therefore removes no row ({{VThreeSanityTestRowsKept}} of
{{VThreeSanityTestRowsAll}} kept), and test macro-F1 is unchanged: task-macro {{VThreeSanityTestFOneAllTaskMacro|ci}},
pooled {{VThreeSanityTestFOneAllPooled|ci}} (near-duplicates removed: {{VThreeSanityTestFOneNearDupsRemovedTaskMacro|ci}},
{{VThreeSanityTestFOneNearDupsRemovedPooled|ci}}).

**2. Per-attack-string recall** (V3 test rows of all split seeds, threshold 0.5): {{VThreeSanityAttackStrings}} test attack
strings; run-mean recall {{VThreeSanityAttackRecall|ci}} with the CI from resampling attack strings (not rows).
{{VThreeSanityAttackPerfect}} strings are always detected, {{VThreeSanityAttackBelowHalf}} have recall below 0.5, the
lowest is {{VThreeSanityAttackRecallMin}}. Per-category recall and the worst strings are in the run's metrics.json: the
misses are short answer-shaping requests (sentiment classification, Base16 output, "reply in German", task
automation), i.e. text attacks whose wording is close to a benign request.

**3. Span style probe (no content words).** A logistic regression on the inserted span alone, with function-word rates
(NLTK English stopword list, which includes you/your/me/my/can), punctuation and newline rates, casing, word / char /
sentence lengths and digit rate, trained per seed on train spans (benign-pool clean-A items vs attacks), scores on
test: macro-F1 {{VThreeSanityStyleCleanAOnlyFOne|ci}}, AUROC {{VThreeSanityStyleCleanAOnlyAuroc|ci}}. With clean-B spans
among the negatives (the rows of the span-form gate): macro-F1 {{VThreeSanityStyleCleanAAndBFOne|ci}}, AUROC
{{VThreeSanityStyleCleanAAndBAuroc|ci}}. By feature group (clean A only): function words
{{VThreeSanityStyleGroupFunctionWordsFOne|ci}} (AUROC {{VThreeSanityStyleGroupFunctionWordsAuroc|ci}}), punctuation
{{VThreeSanityStyleGroupPunctuationFOne|ci}} (AUROC {{VThreeSanityStyleGroupPunctuationAuroc|ci}}), casing and length
{{VThreeSanityStyleGroupCasingLengthFOne|ci}} (AUROC {{VThreeSanityStyleGroupCasingLengthAuroc|ci}}).
- This is far above the span-form gate (span_form_you_lr, 9 form features + you/your; pooled range in F024) and
  close to the span-content ceiling (span_content_tfidf, all words; F024).
  The V3 gates did not include a function-word probe; V3 passes the pre-registered gates but the inserted spans are
  separable by function-word and punctuation style.
- Function words carry part of what makes a span an instruction to the model (requests addressed to the assistant
  vs remarks addressed to a human reader), so this probe is not purely content-free; the largest coefficients
  (metrics.json) mix request-style words with format cues (underscores and double quotes lean benign, "!" and "/"
  lean attack). Whether this is an artifact or the task's signal is not settled by this check.

**4. Val (held-out categories).** Val attack strings found exactly in train: {{VThreeSanityValExactInTrain}};
near-duplicates: {{VThreeSanityValNearDups}} (largest Jaccard {{VThreeSanityValMaxJaccard}}); val categories shared
with train: {{VThreeSanityValSharedCategories}}. V3 DeBERTa val macro-F1 per split seed (the selection metric) ranges
{{VThreeSanityValFOneMin}}–{{VThreeSanityValFOneMax}}, so it is not 1.0 on every seed, and the high values do not come
from val attacks repeating train attacks. (Test shares one category with train, Language Translation, known risk R5.)

## F027. V3 DeBERTa accuracy where the span style probe is confidently wrong vs right

```finding
id: F027
status: full
script: scripts/v3_style_vs_deberta.py
commit: 6c6496f
seeds: split_seeds 0-4 (canonical V3 DeBERTa run each, configs/canonical_runs.json); style probe LR random_state 0; bootstrap B=1000 seed 42
data_version: v3 (tag v3-frozen)
eval_set: V3 test, injected and clean-A rows (the rows the F026 clean-A-only style probe scores)
threshold_protocol: default 0.5 (DeBERTa); probe confidence bins fixed before the run (probability >= 0.9 on one class)
date: '2026-11-12'
source: results/v3/v3_style_vs_deberta/20261112-123714_seed42
upstream:
- results/v3/score_deberta_v3/20261111-174708_seed0
- results/v3/score_deberta_v3/20261111-181158_seed1
- results/v3/score_deberta_v3/20261111-183653_seed2
- results/v3/score_deberta_v3/20261111-174506_seed3
- results/v3/score_deberta_v3/20261111-180717_seed4
cluster: doc_id (stratified by task); per split seed, and pooled as the run-mean over split seeds
macros:
- VThreeStyleDeb*
```

Diagnostic for F026 §3 (nothing tuned or selected on these rows). The F026 style probe (all style features, clean-A
only, trained per split seed on train spans) bins each test row: **confidently wrong** (≥ 0.9 on the wrong class;
{{VThreeStyleDebNConfidentWrong}} rows over five seeds, {{VThreeStyleDebNConfidentWrongAttack}} of them attacks), **confidently right**
({{VThreeStyleDebNConfidentRight}} rows, {{VThreeStyleDebNConfidentRightAttack}} attacks), **unconfident** ({{VThreeStyleDebNUnconfident}} rows).
V3 DeBERTa accuracy at 0.5, run-mean over split seeds:
confidently wrong {{VThreeStyleDebConfidentWrong|ci}}, confidently right {{VThreeStyleDebConfidentRight|ci}},
unconfident {{VThreeStyleDebUnconfident|ci}}; right − wrong {{VThreeStyleDebRightMinusWrong|ci}} (per seed
{{VThreeStyleDebGapSeedMin}}–{{VThreeStyleDebGapSeedMax}}, every per-seed CI above 0, lowest bound {{VThreeStyleDebGapSeedCiLoMin}}).
- Within attacks: wrong {{VThreeStyleDebConfidentWrongAttack|ci}}, right {{VThreeStyleDebConfidentRightAttack|ci}}, gap
  {{VThreeStyleDebRightMinusWrongAttack|ci}}. Within benign items: wrong {{VThreeStyleDebConfidentWrongBenign|ci}}, right
  {{VThreeStyleDebConfidentRightBenign|ci}}, gap {{VThreeStyleDebRightMinusWrongBenign|ci}} (few rows; wide CI).
- Reading: DeBERTa errs more often on spans whose style looks like the other class, mostly benign-styled attacks.
  This is consistent with DeBERTa using the same style cues, but it does not show it: benign-styled attacks are
  also the short answer-shaping requests that F026 §2 found hardest, so style and content are confounded on these
  rows. The V3.1 style gate (design pending approval) is the controlled test.

