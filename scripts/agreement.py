"""
agreement.py
============
Inter-annotator agreement for intent pairs (paper_plan P2; docs/pairs_spec.md §9) from two annotation CSVs
exported by the page of scripts/make_annotation_sheet.py.

  - Cohen's kappa between the two annotators over pair members, labels aligned / subverting / unclear
    (P2 requires >= 0.6), overall and per task
  - confusion matrices: annotator 1 x annotator 2, and each annotator x the intended label
  - agreed pairs: both annotators give the intended label to both members (the P2 rule). A quad counts only if
    both its pairs agree (spec §6: a quad is kept or dropped whole); counts by task and overlap class
  - adjudication list (adjudication.csv in the run directory): every pair that is not agreed, with the reason
    and both annotators' labels, confidences and notes

Every member must be labelled by both annotators; unlabelled members are listed and the script stops.
Logged with runlog to results/pairs/agreement/<run_id>/.

    python scripts/agreement.py data/pairs/pilot_v1.jsonl annotation/pilot_v1/ann1.csv annotation/pilot_v1/ann2.csv
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pairs import ANNOTATION_LABELS, complete_quads, members
from runlog import add_run_args, start_run

INTENDED = {0: "aligned", 1: "subverting"}
ADJ_FIELDS = ["pair_id", "quad_id", "task", "overlap_class", "reason", "role", "request",
              "label_1", "confidence_1", "note_1", "label_2", "confidence_2", "note_2"]


def read_annotations(path: Path, pairs: list) -> dict:
    """(pair_id, role) -> {label, confidence, note} from one exported CSV. The request text identifies the role."""
    role_of = {(p["pair_id"], p[f"request_{r}"]): r for p in pairs for r in ("aligned", "subverting")}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["pair_id"], row["request"])
            if key not in role_of:
                raise ValueError(f"{path}: {row['pair_id']} request not in the pairs file: {row['request']!r}")
            label = row["label"].strip()
            if label and label not in ANNOTATION_LABELS:
                raise ValueError(f"{path}: {row['pair_id']}: unknown label {label!r}")
            out[(row["pair_id"], role_of[key])] = {"label": label, "confidence": row["confidence"], "note": row["note"]}
    return out


def confusion(xs, ys, rows, cols) -> dict:
    c = Counter(zip(xs, ys))
    return {r: {k: c[(r, k)] for k in cols} for r in rows}


def kappa(a, b) -> float:
    """Cohen's kappa over the three annotation labels; 1.0 when both annotators use one identical label throughout."""
    if len(set(a) | set(b)) == 1:
        return 1.0
    return float(cohen_kappa_score(a, b, labels=list(ANNOTATION_LABELS)))


def agreement(pairs: list, ann1: dict, ann2: dict) -> tuple:
    """(metrics dict, adjudication rows)."""
    ms = members(pairs)
    missing = [f"{m['pair_id']}/{m['role']} ({who})" for m in ms for who, ann in (("1", ann1), ("2", ann2))
               if not ann.get((m["pair_id"], m["role"]), {}).get("label")]
    if missing:
        raise ValueError(f"{len(missing)} unlabelled members: {missing[:10]}")
    l1 = [ann1[(m["pair_id"], m["role"])]["label"] for m in ms]
    l2 = [ann2[(m["pair_id"], m["role"])]["label"] for m in ms]
    want = [INTENDED[m["label"]] for m in ms]
    tasks = sorted({m["task"] for m in ms})
    by_task = {t: kappa([x for x, m in zip(l1, ms) if m["task"] == t], [y for y, m in zip(l2, ms) if m["task"] == t])
               for t in tasks}

    agreed, adj = set(), []
    for p in pairs:
        rows = []
        for role in ("aligned", "subverting"):
            a, b = ann1[(p["pair_id"], role)], ann2[(p["pair_id"], role)]
            if a["label"] == b["label"] == role:
                continue
            reason = ("unclear" if "unclear" in (a["label"], b["label"]) else
                      "annotators disagree" if a["label"] != b["label"] else "both differ from intended")
            rows.append({"pair_id": p["pair_id"], "quad_id": p["quad_id"], "task": p["task"],
                         "overlap_class": p["overlap_class"], "reason": reason, "role": role,
                         "request": p[f"request_{role}"], "label_1": a["label"], "confidence_1": a["confidence"],
                         "note_1": a["note"], "label_2": b["label"], "confidence_2": b["confidence"],
                         "note_2": b["note"]})
        if rows:
            adj += rows
        else:
            agreed.add(p["pair_id"])
    quads = complete_quads(agreed, pairs)
    usable = {pid for ids in quads.values() for pid in ids}
    count = lambda ids, key: dict(Counter(p[key] for p in pairs if p["pair_id"] in ids))
    metrics = {
        "n_pairs": len(pairs), "n_members": len(ms),
        "kappa": kappa(l1, l2), "kappa_by_task": by_task,
        "raw_agreement": sum(x == y for x, y in zip(l1, l2)) / len(ms),
        "confusion_ann1_x_ann2": confusion(l1, l2, ANNOTATION_LABELS, ANNOTATION_LABELS),
        "confusion_intended_x_ann1": confusion(want, l1, ("aligned", "subverting"), ANNOTATION_LABELS),
        "confusion_intended_x_ann2": confusion(want, l2, ("aligned", "subverting"), ANNOTATION_LABELS),
        "n_agreed_pairs": len(agreed), "n_usable_pairs": len(usable), "n_usable_quads": len(quads),
        "usable_pairs_by_task": count(usable, "task"), "usable_pairs_by_overlap_class": count(usable, "overlap_class"),
        "usable_pair_ids": sorted(usable), "n_adjudication_pairs": len({r["pair_id"] for r in adj}),
    }
    return metrics, adj


def main():
    ap = add_run_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("pairs", type=Path)
    ap.add_argument("ann1", type=Path)
    ap.add_argument("ann2", type=Path)
    a = ap.parse_args()
    pairs = [json.loads(line) for line in open(a.pairs, encoding="utf-8")]
    metrics, adj = agreement(pairs, read_annotations(a.ann1, pairs), read_annotations(a.ann2, pairs))
    run = start_run(version="pairs", exp="agreement", seed=0,
                    args={"pairs": str(a.pairs), "ann1": str(a.ann1), "ann2": str(a.ann2)},
                    data_version=a.pairs.stem, data_files=[a.pairs, a.ann1, a.ann2], allow_dirty=a.allow_dirty)
    with open(run.dir / "adjudication.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ADJ_FIELDS)
        w.writeheader()
        w.writerows(adj)
    run.log(f"kappa {metrics['kappa']:.4f}; agreed pairs {metrics['n_agreed_pairs']}/{metrics['n_pairs']}; "
            f"usable {metrics['n_usable_pairs']} pairs in {metrics['n_usable_quads']} quads; "
            f"{metrics['n_adjudication_pairs']} pairs to adjudicate")
    run.save_metrics(metrics, headline=("kappa", metrics["kappa"]))
    run.finish()
    print(f"wrote {run.dir.relative_to(ROOT)} (kappa {metrics['kappa']:.4f}, usable pairs {metrics['n_usable_pairs']})")


if __name__ == "__main__":
    main()
