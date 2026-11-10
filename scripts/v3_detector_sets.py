"""
v3_detector_sets.py
===================
The V3 eval rows in the transfer harness's input format, for the published detectors (paper_plan A30).
One file, v3_all.jsonl: every distinct (user_request, text, function) over split seeds 0-4 and the sets
test, val, val_id, raw_clean, ambiguous and long (data/v3 at v3-frozen). Fields follow Liu's BIPIA set:
suite "bipia-<task>" -> here "v3-<task>", function = the task name (email / table / code), user_request = the
row's intent, text = the context. Each distinct input is scored once; v3_map.csv maps every V3 row
(id, split_seed, eval_set) to its line in v3_all.jsonl. manifest.json records sizes, sha256 and data_version.

    python scripts/v3_detector_sets.py --out data/transfer/v3_sets
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SETS = ("test", "val", "val_id", "raw_clean", "ambiguous", "long")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=ROOT / "data" / "v3")
    ap.add_argument("--split_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    frames = []
    for k in a.split_seeds:
        for s in SETS:
            df = pd.read_csv(a.data_dir / f"split_seed_{k}" / f"{s}.csv", keep_default_na=False)
            frames.append(df.assign(eval_set=f"v3_{s}", split_seed=k))
    rows = pd.concat(frames, ignore_index=True)
    key = list(zip(rows["user_intent"], rows["text"], rows["task_type"]))
    uniq, idx = {}, []
    labels = [None if v == "" else int(v) for v in rows["label"]]  # ambiguous rows have no label (flag rate only)
    for k_, lab in zip(key, labels):
        if k_ not in uniq:
            uniq[k_] = (len(uniq), lab)
        elif uniq[k_][1] != lab:
            raise ValueError(f"same input with two labels: {k_[1][:80]!r}")
        idx.append(uniq[k_][0])
    a.out.mkdir(parents=True, exist_ok=True)
    lines = [None] * len(uniq)
    for (req, text, task), (i, lab) in uniq.items():
        lines[i] = json.dumps({"suite": f"v3-{task}", "task": task, "user_request": req, "function": task,
                               "label": lab, "text": text}, ensure_ascii=False)
    f = a.out / "v3_all.jsonl"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    m = rows[["id", "doc_id", "split_seed", "eval_set"]].assign(line=idx)
    m.to_csv(a.out / "v3_map.csv", index=False)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    man = {"data_version": f"v3-frozen-detsets{sha[:8]}", "source": "data/v3 (tag v3-frozen)",
           "split_seeds": a.split_seeds, "sets": list(SETS), "rows": int(len(rows)), "unique_inputs": len(uniq),
           "files": {"v3_all.jsonl": sha, "v3_map.csv": hashlib.sha256((a.out / "v3_map.csv").read_bytes()).hexdigest()}}
    (a.out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"{len(rows)} rows -> {len(uniq)} unique inputs; {man['data_version']}")


if __name__ == "__main__":
    main()
