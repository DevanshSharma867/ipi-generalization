"""
deberta_v2_stats.py
===================
What the V2 DeBERTa results were missing, computed from results/v2_std/ only:
doc-cluster bootstrap CIs (B=1000, seed 42), Recall@1/5/10% FPR, per-task, per-attack-category,
per-insertion-mode and per-clean-type breakdowns, for every seed and for the seed mean.

Seed mean: point = mean (and SD, ddof=1) of the per-seed metrics; CI = bootstrap of the per-seed
metric averaged over seeds, with the same document draw applied to every seed.

Groups: main (full V2 train, V2 test), loto_<task> (held-out task test docs), sub496 (size-matched control).
Writes results/v2_std/deberta_summary.json. V2 DeBERTa numbers are NOT headline results.

    python scripts/deberta_v2_stats.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import df_metrics, doc_cluster_bootstrap, group_metrics
from runlog import write_json

STD = ROOT / "results" / "v2_std"
SEEDS = (42, 123, 999)
CI_KEYS = ["accuracy", "f1_macro", "f1_injected", "precision", "recall", "roc_auc", "pr_auc", "ece",
           "recall_at_1pct_fpr", "recall_at_5pct_fpr", "recall_at_10pct_fpr"]
BREAKDOWN_KEYS = ["n", "f1_macro", "roc_auc", "recall", "flag_rate_clean", "flag_rate_injected", "n_fp", "n_fn"]
GROUPS = {"main": ("v2_test", "deberta_s{seed}"), "sub496": ("v2_test", "deberta_sub496_s{seed}"),
          **{f"loto_{t}": (f"v2_loto_test_{t}", "deberta_loto_s{seed}") for t in ("email", "table", "code")}}


def load(eval_set, model):
    return pd.read_csv(STD / eval_set / f"{model}.csv", keep_default_na=False)


def ci_metrics(df):
    m = df_metrics(df)
    return {k: m[k] for k in CI_KEYS}


def seed_mean_fn(frames):
    """metric_fn over a frame of row positions that averages ci_metrics across the per-seed frames."""
    def fn(rows):
        idx = rows["_row"].values
        per = [ci_metrics(f.iloc[idx]) for f in frames]
        return {k: float(np.nanmean([p[k] for p in per])) for k in CI_KEYS}
    return fn


def breakdowns(df, by_task: bool):
    keep = lambda d: {g: {k: m[k] for k in BREAKDOWN_KEYS} for g, m in d.items()}
    out = {"attack_category": keep(group_metrics(df[df["label"] == 1], "attack_category")),
           "insertion_mode": keep(group_metrics(df[df["insertion_mode"] != ""], "insertion_mode")),
           "clean_type": keep(group_metrics(df[df["label"] == 0], "clean_type"))}
    if by_task:
        out["task"] = {t: {"point": ci_metrics(sub), "bootstrap": doc_cluster_bootstrap(sub.reset_index(drop=True), ci_metrics)}
                       for t, sub in sorted(df.groupby("task"))}
    return out


def summarize_group(eval_set, pattern):
    frames = {s: load(eval_set, pattern.format(seed=s)) for s in SEEDS}
    ids = frames[SEEDS[0]]["id"]
    for s, f in frames.items():
        assert (f["id"].values == ids.values).all(), f"row order differs for seed {s}"
    by_task = frames[SEEDS[0]]["task"].nunique() > 1
    out = {"eval_set": eval_set, "models": {}}
    for s, f in frames.items():
        print(f"  {pattern.format(seed=s)}")
        out["models"][f"s{s}"] = {"source": f"results/v2_std/{eval_set}/{pattern.format(seed=s)}.csv",
                                  "point": df_metrics(f), "bootstrap": doc_cluster_bootstrap(f, ci_metrics),
                                  "breakdowns": breakdowns(f, by_task)}
    pts = [out["models"][f"s{s}"]["point"] for s in SEEDS]
    base = frames[SEEDS[0]][["doc_id"]].assign(_row=np.arange(len(ids)))
    mean = {"point_mean": {k: float(np.mean([p[k] for p in pts])) for k in CI_KEYS},
            "point_sd": {k: float(np.std([p[k] for p in pts], ddof=1)) for k in CI_KEYS},
            "bootstrap_of_seed_mean": doc_cluster_bootstrap(base, seed_mean_fn(list(frames.values())))}
    if by_task:
        mean["task"] = {}
        for t in sorted(frames[SEEDS[0]]["task"].unique()):
            m = (frames[SEEDS[0]]["task"] == t).values
            sub_frames = [f[m].reset_index(drop=True) for f in frames.values()]
            sub_base = sub_frames[0][["doc_id"]].assign(_row=np.arange(m.sum()))
            tp = [ci_metrics(f) for f in sub_frames]
            mean["task"][t] = {"point_mean": {k: float(np.mean([p[k] for p in tp])) for k in CI_KEYS},
                               "point_sd": {k: float(np.std([p[k] for p in tp], ddof=1)) for k in CI_KEYS},
                               "bootstrap_of_seed_mean": doc_cluster_bootstrap(sub_base, seed_mean_fn(sub_frames))}
    cats = {}
    for s in SEEDS:
        for c, m in out["models"][f"s{s}"]["breakdowns"]["attack_category"].items():
            cats.setdefault(c, []).append(m["recall"])
    mean["fn_rate_by_attack_category"] = {c: float(1 - np.mean(v)) for c, v in cats.items()}
    out["seed_mean"] = mean
    return out


def main():
    summary = {"note": "V2 (artifact-laden) DeBERTa results; not headline results. B=1000, seed=42, doc-cluster bootstrap.",
               "seeds": list(SEEDS), "groups": {}}
    for name, (eval_set, pattern) in GROUPS.items():
        print(name)
        summary["groups"][name] = summarize_group(eval_set, pattern)
    # Same checkpoints scored twice: training-loop eval (batched) vs score_detectors.py (one row at a time)
    summary["rescore_max_abs_prob_diff"] = {
        f"s{s}": float(np.abs(load("v2_test", f"deberta_s{s}")["prob"].values
                              - load("v2_test", f"deberta_s{s}_rescore")["prob"].values).max()) for s in SEEDS}
    write_json(STD / "deberta_summary.json", summary)
    print("wrote results/v2_std/deberta_summary.json")
    for name, g in summary["groups"].items():
        print(f"\n== {name} ({g['eval_set']})")
        for s in SEEDS:
            b = g["models"][f"s{s}"]["bootstrap"]
            print(f"  s{s:<4} " + "  ".join(f"{k}={b[k]['point']:.4f} [{b[k]['ci'][0]:.4f}, {b[k]['ci'][1]:.4f}]"
                                          for k in ("f1_macro", "roc_auc", "recall_at_5pct_fpr", "recall_at_10pct_fpr")))
        m = g["seed_mean"]
        print("  mean  " + "  ".join(f"{k}={m['point_mean'][k]:.4f}±{m['point_sd'][k]:.4f} "
                                     f"[{m['bootstrap_of_seed_mean'][k]['ci'][0]:.4f}, {m['bootstrap_of_seed_mean'][k]['ci'][1]:.4f}]"
                                     for k in ("f1_macro", "roc_auc", "recall_at_5pct_fpr", "recall_at_10pct_fpr")))
    print("\nrescore max |Δprob|:", summary["rescore_max_abs_prob_diff"])


if __name__ == "__main__":
    main()
