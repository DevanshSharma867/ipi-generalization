# Deviations log

Every change to a pre-registered element of the study (`docs/paper_plan.md`) made after results were seen:
adding, dropping, or changing an evaluation set, metric, baseline, or threshold protocol. One entry per change,
newest last, never edited after the fact (a correction is a new entry that points to the old one).

Each entry: date, what changed (before → after), the reason, which results had been seen when the decision was
made, and which findings entries (`docs/findings.src.md` IDs) it affects.

| date | change (before → after) | reason | results seen at decision time | affected findings |
|---|---|---|---|---|
| 2026-10-28 | Threshold protocol addition: the "calibrated" TPR@FPR protocol also requires the **calibration set** to pass the clean-count gate (≥ 500 clean rows for TPR@1%FPR, ≥ 100 for TPR@5%FPR), not only the evaluated set (`src/metrics.tpr_at_fpr_protocols`, commit c80a037). | A 1% FPR threshold fixed on fewer than 500 clean rows rests on fewer than 5 clean scores, so the calibrated number would carry no information about the threshold it claims. | Made before any V3 or ICT result existed and before the V2 recompute ran (the V2 recompute in `results/v2_metrics/tpr_protocols/` came after). No V2 number changes: every V2 calibration set has 144 clean rows, so TPR@1%FPR is n/a there anyway. `docs/paper_plan.md` (the pre-registration) does not exist yet. | F009–F012 |
| 2026-10-28 | Evaluation-set roles fixed in the first `docs/paper_plan.md`: headline transfer sets become AgentDojo (AD-Matched, AD-Shift) and τ-bench (TB-Matched), clustered by user task; InjecAgent and deepset (previously the C2 external benchmarks) become secondary and descriptive only, and no hypothesis may be decided on them. | InjecAgent has only 17 independent user cases (F010), and PIGuard was trained on 53/62 of its attack strings (Liu 2026, arXiv:2610.03448). deepset has 56 clean rows, so both TPR@FPR are gated out (F011). | Our V2 results on InjecAgent and deepset (F010, F011) and Liu 2026's published AgentDojo/τ-bench detector numbers had been seen. No model of ours had been scored on AgentDojo or τ-bench; no V3 or ICT result existed. | F010, F011 |
