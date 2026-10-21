"""
apply_review.py
===============
Turn a reviewed sheet (exported from data/benign/review.html, or review_sheet.csv filled in by hand) into the
frozen pool data/benign/benign_v3.json, then re-check the form targets (benign_v3_check.report).

  keep -> the draft text; edit -> edited_text (must be non-empty); drop -> removed.
Every draft item must be reviewed (pass --allow_unreviewed to drop unreviewed items instead of failing).
form_tags are recomputed; the output records the sha256 of the draft and of the review CSV.

    python scripts/apply_review.py --review review_sheet_reviewed.csv [--out data/benign/benign_v3.json]
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from probes import form_tags

DRAFT = ROOT / "data" / "benign" / "benign_v3_draft.json"


def apply(review_rows, draft_items, allow_unreviewed=False):
    by_id = {it["id"]: it for it in draft_items}
    seen, out, problems = set(), [], []
    for r in review_rows:
        i, d = r["id"], (r.get("keep_edit_drop") or "").strip().lower()
        if i not in by_id:
            problems.append(f"{i}: not in the draft")
            continue
        seen.add(i)
        if d == "drop":
            continue
        if d == "keep":
            text = by_id[i]["text"]
        elif d == "edit":
            text = (r.get("edited_text") or "").strip()
            if not text:
                problems.append(f"{i}: marked edit but edited_text is empty")
                continue
        elif not d and allow_unreviewed:
            continue
        else:
            problems.append(f"{i}: decision {d!r} (expected keep/edit/drop)")
            continue
        it = by_id[i]
        out.append({"id": i, "text": text, "split_pool": it["split_pool"], "register": it["register"],
                    "form_tags": form_tags(text), "review": d})
    missing = sorted(set(by_id) - seen)
    if missing and not allow_unreviewed:
        problems.append(f"{len(missing)} draft items missing from the review: {missing[:5]}...")
    return out, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--draft", type=Path, default=DRAFT)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "benign" / "benign_v3.json")
    ap.add_argument("--allow_unreviewed", action="store_true")
    ap.add_argument("--no_report", action="store_true")
    a = ap.parse_args()
    with open(a.review, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    items, problems = apply(rows, json.load(open(a.draft))["items"], a.allow_unreviewed)
    if problems:
        sys.exit("review not applied:\n  " + "\n  ".join(problems))
    texts = Counter(it["text"] for it in items)
    dup = [t for t, n in texts.items() if n > 1]
    if dup:
        sys.exit(f"duplicate texts after review: {dup[:3]}")
    doc = {"version": "v3", "draft_sha256": hashlib.sha256(a.draft.read_bytes()).hexdigest(),
           "review_sha256": hashlib.sha256(a.review.read_bytes()).hexdigest(),
           "counts": dict(Counter(f"{it['register']}/{it['split_pool']}" for it in items)),
           "decisions": dict(Counter(it["review"] for it in items)), "items": items}
    a.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}: {doc['counts']}")
    if not a.no_report:
        from benign_v3_check import report
        report(items)


if __name__ == "__main__":
    main()
