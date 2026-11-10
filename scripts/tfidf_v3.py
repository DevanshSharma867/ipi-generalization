"""
tfidf_v3.py
===========
TF-IDF + LogReg baseline on V3 (paper_plan §7 step 4, A30). Per split seed: V2 features and solver
(run_experiments.TFIDF_PARAMS / LR_BASE_PARAMS; input build_combined_text + preprocess_for_tfidf), fitted on the
split's train.csv; C in {1, 10, 100} selected by pooled val macro-F1 at threshold 0.5 (ties -> smaller C).
Test is never read here. The selected pipeline is pickled to checkpoints/v3/tfidf_lr/split_seed_<k>.pkl and the
run (val metrics per C, the choice) is logged to results/v3/tfidf_v3_train/<run_id>/. Scoring:
scripts/score_baselines.py --model tfidf.

    python scripts/tfidf_v3.py --split_seed 0
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from preprocess import build_combined_text, preprocess_for_tfidf  # noqa: E402
from run_experiments import LR_BASE_PARAMS, TFIDF_PARAMS  # noqa: E402
from runlog import add_run_args, assert_not_frozen, start_run  # noqa: E402

C_GRID = (1.0, 10.0, 100.0)


def texts(df):
    return preprocess_for_tfidf(df.apply(build_combined_text, axis=1).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_seed", type=int, required=True)
    ap.add_argument("--data_dir", type=Path, default=ROOT / "data" / "v3")
    ap.add_argument("--ckpt_dir", type=Path, default=ROOT / "checkpoints" / "v3" / "tfidf_lr")
    a = add_run_args(ap).parse_args()
    d = a.data_dir / f"split_seed_{a.split_seed}"
    out = a.ckpt_dir / f"split_seed_{a.split_seed}.pkl"
    assert_not_frozen(out)
    run = start_run(version="v3", exp="tfidf_v3_train", seed=a.split_seed,
                    args={"split_dir": str(d), "C_grid": list(C_GRID), "tfidf": TFIDF_PARAMS, "lr": LR_BASE_PARAMS,
                          "threshold": 0.5, "selection": "pooled val macro-F1 at 0.5, ties -> smaller C",
                          "checkpoint": str(out)},
                    data_version="v3", data_files=[d / "train.csv", d / "val.csv"], allow_dirty=a.allow_dirty)
    tr = pd.read_csv(d / "train.csv", keep_default_na=False)
    va = pd.read_csv(d / "val.csv", keep_default_na=False)
    x_tr, x_va = texts(tr), texts(va)
    grid, best = [], None
    for c in C_GRID:
        pipe = Pipeline([("tfidf", TfidfVectorizer(**TFIDF_PARAMS)), ("lr", LogisticRegression(**LR_BASE_PARAMS, C=c))])
        pipe.fit(x_tr, tr["label"].values)
        f1 = float(f1_score(va["label"].values, (pipe.predict_proba(x_va)[:, 1] >= 0.5).astype(int), average="macro"))
        grid.append({"C": c, "val_f1_macro": f1})
        run.log(f"C={c:6.1f} val macro-F1 {f1:.4f}")
        if best is None or f1 > best[1]:  # strict: ties keep the smaller C
            best = (c, f1, pipe)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(best[2], f)
    run.save_metrics({"grid": grid, "selected_C": best[0], "val_f1_macro": best[1], "checkpoint": str(out)},
                     headline=("val_f1_macro", best[1]))
    run.log(f"selected C={best[0]} -> {out}")
    run.finish()


if __name__ == "__main__":
    main()
