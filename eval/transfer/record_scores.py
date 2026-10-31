"""
record_scores.py
================
Run record for a batch of detectors.py score files (detectors.py itself runs in Liu's pinned env, which has no
pandas for src/runlog.py). Copies <out>/<detector>__<set>.json for every requested pair into
results/transfer/<exp>/<run_id>/scores/ and records the sets' data_version, the scoring environment's package
versions, the detector revisions and the scoring options. Run in our env after the scoring call:

    python eval/transfer/record_scores.py --exp transfer_scores_liuenv --out <dir> --sets_dir <dir> \\
        --detectors piguard ... --sets admatched ... --scorer_py .venv-liu/bin/python [--note "..."]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval" / "transfer"))

from detectors import load_registry  # noqa: E402
from gate2 import env_versions  # noqa: E402
from runlog import start_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sets_dir", required=True, type=Path)
    ap.add_argument("--detectors", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--scorer_py", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--allow-dirty", dest="allow_dirty", action="store_true")
    a = ap.parse_args()
    reg, manifest = load_registry(), a.sets_dir / "manifest.json"
    files = [a.out / f"{d}__{s}.json" for d in a.detectors for s in a.sets]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        sys.exit(f"missing score files: {missing}")
    env = env_versions(a.scorer_py)
    run = start_run(version="transfer", exp=a.exp, seed=0,
                    args={"detectors": a.detectors, "sets": a.sets, "sets_dir": a.sets_dir.name,
                          "out": "<scratch>/" + a.out.name, "note": a.note,
                          "revisions": {d: reg[d]["revision"] for d in a.detectors},
                          "scorer_env": {k: env.get(k) for k in ("python", "torch", "transformers", "tokenizers",
                                                                 "bitsandbytes", "accelerate")}},
                    data_version=json.load(open(manifest))["data_version"], data_files=[manifest],
                    allow_dirty=a.allow_dirty)
    (run.dir / "scores").mkdir()
    summary = {}
    for f in files:
        shutil.copy(f, run.dir / "scores")
        r = json.load(open(f))
        opts = {k: r[k] for k in ("bs", "precision", "gpu_mem", "judge_rule", "placement") if k in r}
        summary[f.stem] = dict(n=len(r["probs"]), truncated=r["truncated"], **opts)
        run.log(f"{f.stem:40s} n={len(r['probs']):5d} truncated={r['truncated']:.3f} {opts}")
    run.save_metrics({"files": summary, "scorer_env": env}, headline=("files", len(files)))
    run.finish()


if __name__ == "__main__":
    main()
