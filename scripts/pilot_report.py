"""
pilot_report.py
===============
One annotator's labels on a pair set against the drafted (intended) labels, and the time per pair, for the
annotation pilot (docs/pairs_spec.md §11; paper_plan A18 Q4, A25). Two-annotator agreement is
scripts/agreement.py; this script needs only one CSV.

  - agreement with the intended label per member (request), per pair (both members) and per quad (all four),
    overall, by task and by overlap class
  - every member whose label differs from the intended one, is "unclear", or has confidence < 3, with its note
  - time per pair from the page's per-label `saved_at` timestamps, and the extrapolation to --target_pairs

Timing: `saved_at` is the time of a member's last save. A pair's time is the gap between the last save of the
previous pair (in save order) and its own last save, so the first pair's reading time is not observed; the mean
is the observed span over n_pairs - 1 gaps. Gaps over --break_min minutes are reported as breaks and excluded.

    python scripts/pilot_report.py data/pairs/pilot_v1.jsonl annotation/pilot_v1/devansh.csv
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from agreement import INTENDED, read_annotations
from pairs import members
from runlog import add_run_args, start_run


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def report(pairs: list, ann: dict, saved_at: dict, target_pairs: int, break_min: float) -> dict:
    ms = members(pairs)
    missing = [f"{m['pair_id']}/{m['role']}" for m in ms
               if not ann.get((m["pair_id"], m["role"]), {}).get("label")
               or not ann[(m["pair_id"], m["role"])].get("confidence")]
    if missing:
        raise ValueError(f"{len(missing)} members without a label or confidence: {missing}")
    if len(ann) != len(ms):
        raise ValueError(f"CSV has {len(ann)} members, pairs file {len(ms)}")

    ok = {(m["pair_id"], m["role"]): ann[(m["pair_id"], m["role"])]["label"] == INTENDED[m["label"]] for m in ms}
    pair_ok = {p["pair_id"]: ok[(p["pair_id"], "aligned")] and ok[(p["pair_id"], "subverting")] for p in pairs}
    quad_ok = defaultdict(lambda: True)
    for p in pairs:
        quad_ok[p["quad_id"]] &= pair_ok[p["pair_id"]]

    def rates(key):
        out = {}
        for v in sorted({p[key] for p in pairs}):
            ps = [p for p in pairs if p[key] == v]
            out[v] = {"members": f"{sum(ok[(p['pair_id'], r)] for p in ps for r in ('aligned', 'subverting'))}/{2 * len(ps)}",
                      "pairs": f"{sum(pair_ok[p['pair_id']] for p in ps)}/{len(ps)}"}
        return out

    flagged = []
    for m in ms:
        a = ann[(m["pair_id"], m["role"])]
        if not ok[(m["pair_id"], m["role"])] or a["label"] == "unclear" or int(a["confidence"]) < 3:
            p = next(p for p in pairs if p["pair_id"] == m["pair_id"])
            flagged.append({"pair_id": m["pair_id"], "task": m["task"], "overlap_class": m["overlap_class"],
                            "role": m["role"], "intended": INTENDED[m["label"]], "label": a["label"],
                            "confidence": int(a["confidence"]), "instruction": p["instruction"],
                            "request": m["request"], "note": a["note"]})

    # timing: last save per pair, in save order
    last = sorted(max(_ts(saved_at[(p["pair_id"], r)]) for r in ("aligned", "subverting")) for p in pairs)
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(last, last[1:])]
    work = [g for g in gaps if g <= break_min]
    mean = statistics.mean(work)
    return {
        "n_pairs": len(pairs), "n_quads": len(quad_ok), "n_members": len(ms),
        "members_match_intended": sum(ok.values()), "pairs_match_intended": sum(pair_ok.values()),
        "quads_match_intended": sum(quad_ok.values()),
        "by_task": rates("task"), "by_overlap_class": rates("overlap_class"),
        "overlap_class_counts": {c: sum(p["overlap_class"] == c for p in pairs) for c in ("subtle", "explicit")},
        "label_counts": {l: sum(a["label"] == l for a in ann.values()) for l in ("aligned", "subverting", "unclear")},
        "confidence_counts": {c: sum(int(a["confidence"]) == c for a in ann.values()) for c in (1, 2, 3)},
        "members_with_note": sum(bool(a["note"].strip()) for a in ann.values()),
        "flagged_members": flagged,
        "timing": {"first_save": last[0].isoformat(), "last_save": last[-1].isoformat(),
                   "span_min": round((last[-1] - last[0]).total_seconds() / 60, 2),
                   "n_gaps": len(gaps), "breaks_excluded": [round(g, 2) for g in gaps if g > break_min],
                   "break_min": break_min,
                   "min_per_pair_mean": round(mean, 3), "min_per_pair_median": round(statistics.median(work), 3),
                   "min_per_pair_max": round(max(work), 3),
                   "target_pairs": target_pairs, "extrapolated_hours": round(target_pairs * mean / 60, 2)},
    }


def main():
    ap = add_run_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("pairs", type=Path)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--target_pairs", type=int, default=300, help="pairs per annotator to extrapolate to (A25)")
    ap.add_argument("--break_min", type=float, default=10.0, help="gaps longer than this are breaks")
    a = ap.parse_args()
    import csv
    pairs = [json.loads(line) for line in open(a.pairs, encoding="utf-8")]
    ann = read_annotations(a.csv, pairs)
    role_of = {(p["pair_id"], p[f"request_{r}"]): r for p in pairs for r in ("aligned", "subverting")}
    with open(a.csv, newline="", encoding="utf-8") as f:
        saved_at = {(row["pair_id"], role_of[(row["pair_id"], row["request"])]): row["saved_at"]
                    for row in csv.DictReader(f)}
    metrics = report(pairs, ann, saved_at, a.target_pairs, a.break_min)
    run = start_run(version="pairs", exp=a.pairs.stem, seed=0,
                    args={"pairs": str(a.pairs), "csv": str(a.csv), "target_pairs": a.target_pairs,
                          "break_min": a.break_min},
                    data_version=a.pairs.stem, data_files=[a.pairs, a.csv], allow_dirty=a.allow_dirty)
    t = metrics["timing"]
    run.log(f"members {metrics['members_match_intended']}/{metrics['n_members']}, pairs "
            f"{metrics['pairs_match_intended']}/{metrics['n_pairs']}, quads {metrics['quads_match_intended']}/"
            f"{metrics['n_quads']} match intended; {t['min_per_pair_mean']} min/pair -> {t['extrapolated_hours']} h "
            f"for {t['target_pairs']} pairs")
    lines = [f"# Pilot report: {a.csv.name} on {a.pairs.name}", "",
             f"- Match intended: members {metrics['members_match_intended']}/{metrics['n_members']}, pairs "
             f"{metrics['pairs_match_intended']}/{metrics['n_pairs']}, quads {metrics['quads_match_intended']}/"
             f"{metrics['n_quads']}",
             f"- By task: {metrics['by_task']}", f"- By overlap class: {metrics['by_overlap_class']}",
             f"- Labels {metrics['label_counts']}; confidence {metrics['confidence_counts']}; "
             f"notes on {metrics['members_with_note']}/{metrics['n_members']} members",
             f"- Time: span {t['span_min']} min over {t['n_gaps']} gaps; breaks > {t['break_min']} min excluded: "
             f"{t['breaks_excluded']}; mean {t['min_per_pair_mean']} min/pair (median {t['min_per_pair_median']}) "
             f"-> {t['extrapolated_hours']} h for {t['target_pairs']} pairs", "", "## Flagged members", ""]
    lines += [f"- {f['pair_id']} ({f['task']}, {f['overlap_class']}) {f['role']}: intended {f['intended']}, labelled "
              f"{f['label']} (conf {f['confidence']}). I: {f['instruction']!r} R: {f['request']!r}. Note: {f['note']}"
              for f in metrics["flagged_members"]]
    (run.dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run.save_metrics(metrics, headline=("members_match_intended", metrics["members_match_intended"] / metrics["n_members"]))
    run.finish()
    print(json.dumps(metrics, indent=1))
    print(f"wrote {run.dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
