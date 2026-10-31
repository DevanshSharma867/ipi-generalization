"""
transfer_metrics.py
===================
Headline metrics of every scored detector on the agent transfer sets (paper_plan §3 "Threshold protocols", A3, A16),
from the score run records of jobs/transfer_scores_*.sh (eval/transfer/record_scores.py):

- AD-Matched and AD-Shift: injected rows vs AD-Clean as negatives (A16; their benign rows are never counted as
  clean). TB-Matched: its own injected and clean rows (the clean rows are TB-Clean).
- Calibrated (primary): 1% / 5% FPR thresholds fixed on the calibration pool's scores (A3); TPR on the injected rows
  and the realized FPR on the evaluated clean rows, each with a 95% cluster-bootstrap CI.
- Oracle (reported next to it, never decides): threshold on the evaluated clean rows, src/metrics MIN_CLEAN gates
  (so oracle TPR@1%FPR on AgentDojo is n/a: AD-Clean has 339 rows).
- AUROC and FPR at the 0.5 default threshold on the clean rows, with cluster CIs.
Clusters are user tasks (suite/task). B = 1000, seed 42 (src/metrics defaults).

    python eval/transfer/transfer_metrics.py --sets_dir data/transfer/sets_adliu \\
        --runs results/transfer/transfer_scores_liuenv/<id> results/transfer/transfer_scores_wolf/<id> ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval" / "transfer"))

from detectors import SETS  # noqa: E402
from metrics import doc_cluster_bootstrap, tpr_at_fpr_protocols  # noqa: E402
from runlog import start_run  # noqa: E402

EVAL = {"admatched": "adclean", "adshift": "adclean", "tbeval": None}  # eval set -> clean set (None: its own)
CLEAN_ONLY = ["adclean", "tbclean", "calpool"]


def frame(sets_dir, scores, det, s, neg):
    """label/prob/cluster frame: the set's injected rows, plus `neg`'s rows (or the set's own clean rows) as 0."""
    def part(name, only=None):
        rows = [json.loads(line) for line in open(sets_dir / SETS[name])]
        p = json.load(open(scores / f"{det}__{name}.json"))["probs"]
        assert len(p) == len(rows), (det, name)
        d = pd.DataFrame({"label": [int(r.get("label", 0)) for r in rows], "prob": p,
                          "cluster": [f"{r['suite']}/{r['task']}" for r in rows]})
        return d if only is None else d[d["label"] == only]
    return pd.concat([part(s, 1), part(neg, 0).assign(label=0)] if neg else [part(s)], ignore_index=True)


def auroc_and_default_fpr(d):
    return {"auroc": float(roc_auc_score(d["label"], d["prob"])),
            "fpr_at_0.5": float((d["prob"][d["label"] == 0] >= 0.5).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets_dir", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="score run records (record_scores.py)")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--allow-dirty", dest="allow_dirty", action="store_true")
    a = ap.parse_args()
    manifest = a.sets_dir / "manifest.json"
    version = json.load(open(manifest))["data_version"]
    runs = [r.resolve() for r in a.runs]
    for r in runs:
        v = json.load(open(r / "config.json"))["data_version"]
        assert v == version, f"{r} scored data {v}, sets are {version}"
    run = start_run(version="transfer", exp="transfer_metrics", seed=42,
                    args={"runs": [str(r.relative_to(ROOT)) for r in runs], "B": a.B, "sets_dir": a.sets_dir.name},
                    data_version=version, data_files=[manifest], allow_dirty=a.allow_dirty)
    res = {}
    for r in runs:
        cfg = json.load(open(r / "config.json"))["args"]
        tag = r.parent.name.replace("transfer_scores_", "")
        for det in cfg["detectors"]:
            key = f"{det}@{tag}"
            calib = json.load(open(r / "scores" / f"{det}__calpool.json"))["probs"]
            res[key] = {"calibration_pool": {"n": len(calib), "flag_at_0.5": float(np.mean(np.array(calib) >= 0.5))}}
            for s, neg in EVAL.items():
                d = frame(a.sets_dir, r / "scores", det, s, neg)
                m = tpr_at_fpr_protocols(d, calib_clean_scores=calib, B=a.B, cluster_col="cluster",
                                         calib_gate_eval=False)  # A16
                m.update(doc_cluster_bootstrap(d, auroc_and_default_fpr, B=a.B, cluster_col="cluster"))
                m["n_injected"], m["n_clean"] = int((d["label"] == 1).sum()), int((d["label"] == 0).sum())
                res[key][s] = m
                c1, c5 = m["calibrated@1pct_fpr"], m["calibrated@5pct_fpr"]
                fmt = lambda e: (f"{e['tpr']['point']:.3f} (FPR {e['realized_fpr']['point']:.3f})"  # noqa: E731
                                 if e["status"] == "ok" else e["status"])
                o5 = m["oracle@5pct_fpr"]
                o5 = f"{o5['tpr']['point']:.3f}" if o5["status"] == "ok" else o5["status"]
                run.log(f"{key:34s} {s:9s} AUROC {m['auroc']['point']:.3f}  cal TPR@1% {fmt(c1)}  "
                        f"cal TPR@5% {fmt(c5)}  oracle TPR@5% {o5}")
    run.save_metrics(res, headline=("detectors", len(res)))
    run.finish()


if __name__ == "__main__":
    main()
