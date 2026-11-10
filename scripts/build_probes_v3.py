"""
build_probes_v3.py
==================
V3 position probes (paper_plan A30; descriptive, decide nothing). Per split seed, from its TEST docs only: the
untouched context (the doc's raw_clean row, i.e. the trimmed context used for all of its rows) with each of the
doc's four main-row spans re-inserted at start, middle and end, with the row's own separator and the V3 insertion
code (data_pipeline_v3.make_row):

  p1_cleanb_moved   the doc's clean-B span (label 0), 3 rows/doc
  p2_cleana_moved   the doc's clean-A (benign pool) span (label 0), 3 rows/doc
  p3_attack_moved   each of the doc's 2 test attacks (label 1), 3 rows per attack
  p4_whitespace_only  the context with V2's middle-mode whitespace collapse (" ".join(ctx.split())), nothing inserted (label 0)

The row's own position must reproduce its main-set text exactly (asserted), so every probe differs from a real test
row only in where the span sits. Rows keep the V3 columns plus probe, position, source_id.

Writes data/v3_probes/split_seed_<k>.csv and data/v3_probes/manifest.json.

    python scripts/build_probes_v3.py --split_seeds 0 1 2 3 4
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_pipeline_v3 import make_row  # noqa: E402
from runlog import _git, assert_not_frozen  # noqa: E402

POSITIONS = ("start", "middle", "end")
PROBE = {"clean_b": "p1_cleanb_moved", "clean_a": "p2_cleana_moved", "injected": "p3_attack_moved"}
OUT = ROOT / "data" / "v3_probes"


def build(split_dir: Path) -> pd.DataFrame:
    test = pd.read_csv(split_dir / "test.csv", keep_default_na=False)
    raw = pd.read_csv(split_dir / "raw_clean.csv", keep_default_na=False)
    raw = raw[raw["split"] == "test"].set_index("doc_id")
    rows = []
    for _, r in test.iterrows():
        doc = raw.loc[r["doc_id"]]
        ctx = doc["text"]
        sep = r["separator"].encode().decode("unicode_escape")
        kind = r["row_type"] if r["row_type"] in PROBE else ("injected" if r["label"] == 1 else r["clean_type"])
        d = {"doc_id": r["doc_id"], "task_type": r["task_type"], "user_intent": r["user_intent"],
             "trimmed": r["trimmed"], "component": r["component"]}
        for pos in POSITIONS:
            new = make_row(d, ctx, r["span_text"], pos, sep, {})
            if pos == r["insertion_mode"] and new["text"] != r["text"]:
                raise AssertionError(f"{r['id']}: re-insertion at its own position does not reproduce the row")
            rows.append({**r.to_dict(), **new, "id": f"{r['id']}__{pos}", "source_id": r["id"],
                         "probe": PROBE[kind], "position": pos, "eval_slice": "probe"})
    for doc_id, doc in raw.iterrows():
        rows.append({**doc.to_dict(), "doc_id": doc_id, "id": f"{doc_id}__p4", "source_id": doc["id"],
                     "text": " ".join(doc["text"].split()), "probe": "p4_whitespace_only", "position": "none",
                     "eval_slice": "probe", "label": 0})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--data_dir", type=Path, default=ROOT / "data" / "v3")
    a = ap.parse_args()
    assert_not_frozen(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    man = {"commit": _git("rev-parse", "HEAD"), "data_dir": str(a.data_dir.relative_to(ROOT)), "files": {}}
    for k in a.split_seeds:
        df = build(a.data_dir / f"split_seed_{k}")
        p = OUT / f"split_seed_{k}.csv"
        df.to_csv(p, index=False, lineterminator="\n")
        man["files"][p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rows": len(df),
                                "by_probe": df["probe"].value_counts().sort_index().to_dict()}
        print(f"wrote {p}: {man['files'][p.name]['by_probe']}")
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")


if __name__ == "__main__":
    main()
