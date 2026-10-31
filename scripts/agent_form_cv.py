"""
agent_form_cv.py
================
Within-split shortcut diagnostic for the agent-style data (data/agent_v1, F019 follow-up). For each split seed,
5-fold cross-validation inside TRAIN only (GroupKFold by doc_id), standardized LR, on:

  span_form_you    the 9 V2 span-form features + has_you of the inserted span (as span_form_you_lr)
  span_length      n_words and n_chars of the inserted span
  text_length      n_chars and n_words of the full text (as length_lr)

Reports out-of-fold macro-F1 (tau 0.5) and AUROC. Flag (fixed in the request before the run): CV macro-F1 > 0.60
means the training data carries a length/form shortcut.

    python scripts/agent_form_cv.py --data_dir data/agent_v1 --split_seeds 0 1 2 3 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from probes import form_tags
from runlog import add_run_args, start_run

FLAG = 0.60
SPAN_FORM = ["n_words", "n_chars", "ends_q", "ends_dot", "ends_excl", "has_quote", "has_digit",
             "has_newline_or_code_char", "cap_ratio", "has_you"]


def features(d: pd.DataFrame) -> dict:
    tags = pd.DataFrame([form_tags(s) for s in d["span_text"]]).astype(float)
    return {"span_form_you": tags[SPAN_FORM].values, "span_length": tags[["n_words", "n_chars"]].values,
            "text_length": np.array([[len(t), len(t.split())] for t in d["text"]], float)}


def cv(X, y, groups):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return {"f1_macro": float(f1_score(y, oof >= 0.5, average="macro")), "roc_auc": float(roc_auc_score(y, oof))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=ROOT / "data" / "agent_v1")
    ap.add_argument("--split_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--exp", default="form_cv")
    a = add_run_args(ap).parse_args()
    files = [a.data_dir / f"split_seed_{k}" / "train.csv" for k in a.split_seeds]
    run = start_run(version=a.data_dir.name, exp=a.exp, seed=a.split_seeds[0],
                    args={"split_seeds": a.split_seeds, "folds": 5, "groups": "doc_id", "flag": FLAG},
                    data_version=a.data_dir.name, data_files=files, allow_dirty=a.allow_dirty)
    out = {}
    for k, f in zip(a.split_seeds, files):
        d = pd.read_csv(f, keep_default_na=False)
        y, g = d["label"].values.astype(int), d["doc_id"].values
        out[str(k)] = {name: cv(X, y, g) for name, X in features(d).items()}
        run.log(f"seed {k}: " + "  ".join(f"{n} F1 {r['f1_macro']:.4f} AUROC {r['roc_auc']:.4f}"
                                          for n, r in out[str(k)].items()))
    summary = {n: {"f1_macro_max": max(out[s][n]["f1_macro"] for s in out),
                   "auroc_max_dev": max(abs(out[s][n]["roc_auc"] - 0.5) for s in out)} for n in out[str(a.split_seeds[0])]}
    flagged = [n for n, v in summary.items() if v["f1_macro_max"] > FLAG]
    run.log(f"flagged (> {FLAG}): {flagged or 'none'}")
    run.save_metrics({"seeds": out, "summary": summary, "flag": FLAG, "flagged": flagged},
                     headline=("span_form_you_f1_max", summary["span_form_you"]["f1_macro_max"]))
    run.finish()
    print(f"wrote {run.dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
