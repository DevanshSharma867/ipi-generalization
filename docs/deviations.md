# Deviations log

Every change to a pre-registered element of the study (`docs/paper_plan.md`) made after results were seen:
adding, dropping, or changing an evaluation set, metric, baseline, or threshold protocol. One entry per change,
newest last, never edited after the fact (a correction is a new entry that points to the old one).

Each entry: date, what changed (before → after), the reason, which results had been seen when the decision was
made, and which findings entries (`docs/findings.src.md` IDs) it affects.

| date | change (before → after) | reason | results seen at decision time | affected findings |
|---|---|---|---|---|
| 2026-10-28 | Threshold protocol addition: the "calibrated" TPR@FPR protocol also requires the **calibration set** to pass the clean-count gate (≥ 500 clean rows for TPR@1%FPR, ≥ 100 for TPR@5%FPR), not only the evaluated set (`src/metrics.tpr_at_fpr_protocols`, commit c80a037). | A 1% FPR threshold fixed on fewer than 500 clean rows rests on fewer than 5 clean scores, so the calibrated number would carry no information about the threshold it claims. | Made before any V3 or ICT result existed and before the V2 recompute ran (the V2 recompute in `results/v2_metrics/tpr_protocols/` came after). No V2 number changes: every V2 calibration set has 144 clean rows, so TPR@1%FPR is n/a there anyway. `docs/paper_plan.md` (the pre-registration) does not exist yet. | F009–F012 |
