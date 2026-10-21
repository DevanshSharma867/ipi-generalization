# Run registry

One line per run (appended by src/runlog.py). Runs before tag v2-frozen are not listed;
see CODEBASE_STATE.md §6.

| date | version | exp | run_id | commit | data version | headline metric | path |
|---|---|---|---|---|---|---|---|
| 2026-10-19 | v3 | tokenizer_check | 20261019-175159_seed0 | f3346c6-dirty | v2 | rows_differing_vs_slow=0.0000 | results/v3/tokenizer_check/20261019-175159_seed0 |
| 2026-10-19 | v2_probes | artifact_baselines | 20261019-180020_seed42 | c61e423-dirty | v2 | position_oracle_f1_macro=0.7333 | results/v2_probes/artifact_baselines/20261019-180020_seed42 |
| 2026-10-19 | v2_probes | score_tfidf_c1 | 20261019-180346_seed42 | c61e423-dirty | v2_probes |  | results/v2_probes/score_tfidf_c1/20261019-180346_seed42 |
| 2026-10-19 | v2_probes | probe_report | 20261019-180500_seed42 | c61e423-dirty | v2_probes |  | results/v2_probes/probe_report/20261019-180500_seed42 |


| 2026-10-19 | v2_probes | score_deberta_s42 | 20261019-183817_seed42 | 518b572-dirty | v2_probes |  | results/v2_probes/score_deberta_s42/20261019-183817_seed42 |
| 2026-10-19 | v2_probes | score_deberta_s123 | 20261019-184210_seed123 | 518b572-dirty | v2_probes |  | results/v2_probes/score_deberta_s123/20261019-184210_seed123 |
| 2026-10-19 | v2_probes | score_deberta_s999 | 20261019-184348_seed999 | 518b572-dirty | v2_probes |  | results/v2_probes/score_deberta_s999/20261019-184348_seed999 |
| 2026-10-19 | v2_probes | score_protectai | 20261019-184432_seed42 | 518b572-dirty | v2_probes |  | results/v2_probes/score_protectai/20261019-184432_seed42 |
| 2026-10-19 | v2_probes | score_piguard | 20261019-184536_seed42 | 518b572-dirty | v2_probes |  | results/v2_probes/score_piguard/20261019-184536_seed42 |
| 2026-10-19 | v2_probes | score_judge_qwen2.5_7b | 20261019-184624_seed42 | 518b572-dirty | v2_probes |  | results/v2_probes/score_judge_qwen2.5_7b/20261019-184624_seed42 |
| 2026-10-19 | v2_probes | probe_report | 20261019-185349_seed42 | 518b572-dirty | v2_probes |  | results/v2_probes/probe_report/20261019-185349_seed42 |
| 2026-10-19 | v2_probes | truncation_control | 20261019-190514_seed42 | 48aba4c-dirty | v2_probes | deberta_mean_p3_end_minus_start_matched=-0.0982 | results/v2_probes/truncation_control/20261019-190514_seed42 |
| 2026-10-19 | v2_probes | probe_report | 20261019-190648_seed42 | 48aba4c-dirty | v2_probes |  | results/v2_probes/probe_report/20261019-190648_seed42 |
| 2026-10-21 | v2_probes | score_judge_qwen2.5_7b | 20261021-123213_seed42 | 5a2a782 | v2_probes |  | results/v2_probes/score_judge_qwen2.5_7b/20261021-123213_seed42 |
| 2026-10-21 | v2_probes | truncation_control | 20261021-125607_seed42 | 2a370a8 | v2_probes | deberta_mean_p3_end_minus_start_matched=-0.0982 | results/v2_probes/truncation_control/20261021-125607_seed42 |
| 2026-10-21 | v2_probes | probe_report | 20261021-125648_seed42 | 2a370a8 | v2_probes |  | results/v2_probes/probe_report/20261021-125648_seed42 |
| 2026-10-21 | v3 | design_facts | 20261021-131051_seed0 | 2b1eff0 | bipia@a004b69 |  | results/v3/design_facts/20261021-131051_seed0 |
| 2026-10-21 | v2_probes | artifact_baselines | 20261021-133927_seed42 | b23108b | v2 | position_oracle_f1_macro=0.7333 | results/v2_probes/artifact_baselines/20261021-133927_seed42 |
| 2026-10-21 | v3 | artifact_baselines_dryrun | 20261021-134412_seed0 | 24198df | v3-draft-iter1 | position_oracle_f1_macro=0.4610 | results/v3/artifact_baselines_dryrun/20261021-134412_seed0 |
| 2026-10-21 | v3 | artifact_baselines_dryrun | 20261021-135210_seed0 | 1df796c | v3-draft-iter2 | position_oracle_f1_macro=0.4610 | results/v3/artifact_baselines_dryrun/20261021-135210_seed0 |
| 2026-10-21 | v3 | artifact_baselines_dryrun | 20261021-135908_seed0 | 02b7ebb | v3-draft-iter3 | position_oracle_f1_macro=0.4610 | results/v3/artifact_baselines_dryrun/20261021-135908_seed0 |
| 2026-10-21 | v3 | benign_draft_check | 20261021-140822_seed0 | fc9aa46 | v3-draft |  | results/v3/benign_draft_check/20261021-140822_seed0 |
| 2026-10-21 | v2_probes | artifact_baselines | 20261021-155152_seed42 | a537610 | v2 | position_oracle_f1_macro=0.7333 | results/v2_probes/artifact_baselines/20261021-155152_seed42 |
| 2026-10-21 | v3 | benign_draft_check | 20261021-161229_seed0 | 65bebfd | v3-draft |  | results/v3/benign_draft_check/20261021-161229_seed0 |
| 2026-10-21 | v3 | artifact_baselines_dryrun5 | 20261021-161044_seed0 | 65bebfd | v3-draft-r2-iter1 | position_oracle_f1_macro=0.4610 | results/v3/artifact_baselines_dryrun5/20261021-161044_seed0 |
| 2026-10-21 | v3 | artifact_baselines_dryrun5 | 20261021-165139_seed0 | c134cfe | v3-draft-r2-iter2 | position_oracle_f1_macro=0.4610 | results/v3/artifact_baselines_dryrun5/20261021-165139_seed0 |
