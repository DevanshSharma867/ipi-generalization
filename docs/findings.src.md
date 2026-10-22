# Findings log (numbers only)

A running log of what the committed results establish so far. Every number is a macro from `paper/numbers.tex`
(written by `scripts/make_tables.py` from results files) shown next to its value. `[lo, hi]` is the 95% CI from
the macros `\<Name>CiLo` / `\<Name>CiHi`: a doc-cluster bootstrap (B=1000, seed 42, resampling `doc_id`), or for
differences the paired doc-cluster bootstrap. All results are on the artifact-laden V2 benchmark (288 test rows,
72 docs). V2 DeBERTa numbers are not headline results.

Regenerate with `python scripts/make_tables.py && python scripts/make_findings.py`. Edit `docs/findings.src.md`,
not `docs/findings.md`.

## F1. V2 artifact-only baselines

None of these probes uses the user query; the learned ones are trained on V2 train only. Reference: the TF-IDF+LR detector
(C=1, τ=0.5) gets V2 test macro-F1 {{VTwoTfidfCOneFOneMacro|ci}}.

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

## F2. DeBERTa position reliance on V2 (probes P1–P3)

Paired within-doc contrasts. "Full" uses all probe rows. "Matched" keeps only sources whose inserted span is
fully inside DeBERTa's 512-token window at all three positions (`scripts/truncation_control.py`).

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

- For s42 and s123 the P1 difference is 0. For s999 it is positive and its CI excludes 0.

## F3. Seed dependence of shortcut use (DeBERTa)

| Seed | V2 test macro-F1 | P1 start flag rate | P2 start − middle |
|---|---|---|---|
| s42 | {{VTwoSeedvarDebertaSFourTwoFOneMacro|ci}} | {{VTwoSeedvarDebertaSFourTwoPOnestart|ci}} | {{VTwoSeedvarDebertaSFourTwoPTwostartminusmiddle|ci}} |
| s123 | {{VTwoSeedvarDebertaSOneTwoThreeFOneMacro|ci}} | {{VTwoSeedvarDebertaSOneTwoThreePOnestart|ci}} | {{VTwoSeedvarDebertaSOneTwoThreePTwostartminusmiddle|ci}} |
| s999 | {{VTwoSeedvarDebertaSNineNineNineFOneMacro|ci}} | {{VTwoSeedvarDebertaSNineNineNinePOnestart|ci}} | {{VTwoSeedvarDebertaSNineNineNinePTwostartminusmiddle|ci}} |
| mean | {{VTwoSeedvarDebertaMeanFOneMacro|ci}} | {{VTwoSeedvarDebertaMeanPOnestart|ci}} | {{VTwoSeedvarDebertaMeanPTwostartminusmiddle|ci}} |

- The three seeds' V2 test macro-F1 CIs overlap.
- s999's P1 start flag rate CI does not overlap those of s42 and s123.
- s999's P2 start − middle CI does not overlap those of s42 and s123.

## F4. Qwen2.5-7B judge (frozen prompt, no truncation)

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
| Flag rate, V2 raw clean rows (72) | {{VTwoGroupsJudgeQwenTwoFiveSevenbRawcleanflag|ci}} |
| Flag rate, V2 benign-imperative rows (72) | {{VTwoGroupsJudgeQwenTwoFiveSevenbBenignflag|ci}} |
| Flag rate, P4 whitespace-only (72) | {{VTwoGroupsJudgeQwenTwoFiveSevenbPfourflag|ci}} |
| P4 − raw clean (same 72 docs) | {{VTwoGroupsJudgeQwenTwoFiveSevenbPfourminusrawclean|ci}} |

- The judge's P3 recall falls from start to end; the end − start CI excludes 0.
- P2 and P1 start − middle CIs include 0.
- The V2 test macro-F1 CI includes 0.5.
- P4 − raw clean CI includes 0.

## F5. Public detectors on V2 test

| Detector | Recall (injected rows) | Flag rate: benign imperative | Flag rate: raw clean |
|---|---|---|---|
| ProtectAI v2 | {{VTwoGroupsProtectaiRecall|ci}} | {{VTwoGroupsProtectaiBenignflag|ci}} | {{VTwoGroupsProtectaiRawcleanflag|ci}} |
| PIGuard | {{VTwoGroupsPiguardRecall|ci}} | {{VTwoGroupsPiguardBenignflag|ci}} | {{VTwoGroupsPiguardRawcleanflag|ci}} |

- ProtectAI's recall point estimate is below its flag rate on both clean row groups.
- PIGuard flags none of the raw clean rows; its benign-imperative flag-rate CI excludes 0.

## F6. Provenance caveats

- `results/scores/summary.json` Recall@5%FPR on pairs v0 is wrong for PIGuard and ProtectAI
  (`score_detectors.recall_at_fpr` uses `roc_curve(drop_intermediate=True)`):
  PIGuard {{VTwoPairsbugPiguardSummaryjson}} in summary.json vs {{VTwoPairsbugPiguardMetrics}} from `src/metrics.py`;
  ProtectAI {{VTwoPairsbugProtectaiSummaryjson}} vs {{VTwoPairsbugProtectaiMetrics}}.
  Recall@FPR for the paper comes from `results/v2_std/` via `src/metrics.py`.
- The V1 audit numbers from the Phase 2 report have no code, data, or results in the repository or its git history.
  They are removed and appear nowhere in this log or in `paper/`.
- Pairs v0 (`data/pairs/pairs_v0.csv`, 50 pairs) are LLM-drafted, not human-validated (pilot), on V2 test docs, and have not been
  independently labeled. No pair result is evidence yet, and none is listed here.
- The LLM judge prompt v1 (`score_detectors.JUDGE_SYSTEM`) was committed in the same commit as its first scores, so
  freezing before results cannot be shown; V2 judge results carry this caveat. V3 will use a new judge prompt committed
  separately before any scoring.
