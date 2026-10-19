"""
error_stats.py
==============
Error-analysis numbers for one model on one standard-schema predictions file (default: the V2 TF-IDF
C=1 baseline on the V2 test set), written to results/v2_std/error_stats_<model>.json:
per-task FP/FN, per-attack-category FN rate, FP rate on benign-imperative vs raw clean rows,
probability ranges of FPs and FNs, and the fraction of probabilities in [0.4, 0.6].

    python scripts/error_stats.py [--eval_set v2_test] [--model tfidf_c1]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from runlog import jsonable, write_json

STD = ROOT / "results" / "v2_std"


def error_stats(df: pd.DataFrame) -> dict:
    fp = (df["label"] == 0) & (df["pred"] == 1)
    fn = (df["label"] == 1) & (df["pred"] == 0)
    clean, inj = df[df["label"] == 0], df[df["label"] == 1]

    def rate(k, n):
        return {"count": int(k), "n": int(n), "rate": k / n if n else float("nan")}

    return {
        "n": len(df), "n_fp": int(fp.sum()), "n_fn": int(fn.sum()),
        "per_task": {t: {"fp": int((fp & (df["task"] == t)).sum()), "fn": int((fn & (df["task"] == t)).sum()),
                         "n": int((df["task"] == t).sum())} for t in sorted(df["task"].unique())},
        "fn_rate_by_attack_category": {c: rate(((g["pred"] == 0)).sum(), len(g))
                                       for c, g in sorted(inj.groupby("attack_category"))},
        "fp_rate_by_clean_type": {c: rate((g["pred"] == 1).sum(), len(g))
                                  for c, g in sorted(clean.groupby("clean_type"))},
        "fp_prob_range": [df.loc[fp, "prob"].min(), df.loc[fp, "prob"].max()] if fp.any() else None,
        "fn_prob_range": [df.loc[fn, "prob"].min(), df.loc[fn, "prob"].max()] if fn.any() else None,
        "frac_prob_in_0.4_0.6": float(df["prob"].between(0.4, 0.6).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set", default="v2_test")
    ap.add_argument("--model", default="tfidf_c1")
    a = ap.parse_args()
    df = pd.read_csv(STD / a.eval_set / f"{a.model}.csv", keep_default_na=False)
    out = error_stats(df)
    path = STD / f"error_stats_{a.eval_set}_{a.model}.json"
    write_json(path, {"source": str((STD / a.eval_set / f"{a.model}.csv").relative_to(ROOT)), **out})
    print(json.dumps(jsonable(out), indent=1))
    print(f"wrote {path.relative_to(ROOT)}")
    if (a.eval_set, a.model) == ("v2_test", "tfidf_c1"):
        # Audit corrections (CODEBASE_STATE.md §6.2): report_phase2.tex stated 0.5009 and 0.4998
        assert round(out["fp_prob_range"][0], 6) == 0.500635, out["fp_prob_range"]
        assert round(out["fn_prob_range"][1], 6) == 0.499204, out["fn_prob_range"]
        print("Confirmed audit corrections: FP min 0.500635, FN max 0.499204")


if __name__ == "__main__":
    main()
