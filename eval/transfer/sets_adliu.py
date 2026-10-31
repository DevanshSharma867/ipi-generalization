"""
sets_adliu.py
=============
The transfer sets used for scoring (docs/deviations.md 2026-10-31): the Gate 1 rebuild (data/transfer/sets) with
AD-Matched and AD-Shift replaced by the builds with the pairing recovered from Liu's released scores
(results/transfer/gate2_followup_pairing/<run>/scores/recovered), plus the A3 calibration pool as jsonl
(data/agent_v1/calibration_pool.csv: clean synthetic tool outputs; user_intent becomes user_request).
Unchanged files are copied. manifest.json keeps the base manifest, records what was replaced and from where, and
recomputes data_version with manifest.py's formula over the new file hashes. Idempotent.

    python eval/transfer/sets_adliu.py --base data/transfer/sets --pairing_run <run dir> --out data/transfer/sets_adliu
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from manifest import FILES, file_stats

ROOT = Path(__file__).resolve().parents[2]
RECOVERED = ["agentdojo_pairs.jsonl", "agentdojo_pairs_shift.jsonl"]
CALPOOL = ROOT / "data" / "agent_v1" / "calibration_pool.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--pairing_run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.pairing_run = a.pairing_run.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    rec = a.pairing_run / "scores" / "recovered"
    for f in FILES:
        if (a.base / f).exists():
            shutil.copyfile(rec / f if f in RECOVERED else a.base / f, a.out / f)
    with open(CALPOOL, newline="") as fin, open(a.out / "calibration_pool.jsonl", "w") as fout:
        for r in csv.DictReader(fin):
            fout.write(json.dumps({"id": r["id"], "suite": "agent_v1-" + r["task_type"], "task": r["doc_id"],
                                   "text": r["text"], "user_request": r["user_intent"], "label": int(r["label"]),
                                   "source_type": "tool_result"}) + "\n")
    m = json.load(open(a.base / "manifest.json"))
    names = {**FILES, "calibration_pool.jsonl": "calibration_pool"}
    files = {name: file_stats(a.out / f) for f, name in names.items() if (a.out / f).exists()}
    content = hashlib.sha256(json.dumps({k: v["sha256"] for k, v in files.items()}, sort_keys=True).encode())
    base_version = m["data_version"]
    m.update(files=files, derived_from=base_version,
             replaced={FILES[f]: str(a.pairing_run.relative_to(ROOT) / "scores" / "recovered" / f) for f in RECOVERED},
             added={"calibration_pool": str(CALPOOL.relative_to(ROOT))},
             data_version=base_version.rsplit("-files", 1)[0] + "-files" + content.hexdigest()[:8])
    (a.out / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    print(m["data_version"])
    for name, s in files.items():
        print(f"  {name:16s} rows {s['rows']:5d} labels {s.get('labels', '-')}")


if __name__ == "__main__":
    main()
