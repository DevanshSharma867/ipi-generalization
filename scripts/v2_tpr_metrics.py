"""
v2_tpr_metrics.py
=================
Recompute the saved V2 scores with the headline metrics: TPR@1%FPR and TPR@5%FPR under the "oracle" and
"calibrated" threshold protocols (src/metrics.tpr_at_fpr_protocols), AUROC, FPR at the default threshold on
clean rows, and macro-F1 (secondary), each with a doc-cluster bootstrap 95% CI (B=1000, seed 42).
Reads saved predictions only; nothing is re-scored. Writes results/v2_metrics/tpr_protocols/<run_id>/.

Calibration set per model (clean rows only; "n/a (no calibration scores)" where none exist):
  - tfidf_c1, deberta_s<k>, protectai, piguard, judge: the V2 val split (data/processed/val.csv),
    from results/v2_std/v2_val/ or the latest results/v2_probes/score_<name>_v2val/ run
  - deberta_loto_s<k>: the LOTO val split of the two training tasks (results/v2_std/v2_loto_val_<task>/)
  - tfidf_c100_tuned, tfidf_c100_loto: none (no C=100 checkpoint was saved)
External sets (InjecAgent, deepset) have no doc_id; their bootstrap resamples rows.

    python scripts/v2_tpr_metrics.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import df_metrics, doc_cluster_bootstrap, tpr_at_fpr_protocols
from runlog import add_run_args, start_run

STD = ROOT / "results" / "v2_std"
PROBES = ROOT / "results" / "v2_probes"
JUDGE_V2_TEST = PROBES / "score_judge_qwen2.5_7b" / "20261021-123213_seed42" / "predictions.csv"
SEEDS = (42, 123, 999)
LOTO_TASKS = ("email", "table", "code")
B, BOOT_SEED = 1000, 42


def latest_probe_run(name):
    runs = sorted((PROBES / f"score_{name}").glob("*/predictions.csv"))
    return runs[-1] if runs else None


def sources():
    """(model, eval_set, predictions file, calibration file or None)."""
    out = []
    tfidf_cal = latest_probe_run("tfidf_c1_v2val")
    for es, f in [("v2_test", "tfidf_c1"), ("injecagent", "tfidf"), ("deepset", "tfidf")]:
        out.append(("tfidf_c1", es, STD / es / f"{f}.csv", tfidf_cal))
    out.append(("tfidf_c100_tuned", "v2_test", STD / "v2_test" / "tfidf_c100_tuned.csv", None))
    for t in LOTO_TASKS:
        out.append(("tfidf_c100_loto", f"v2_loto_test_{t}", STD / f"v2_loto_test_{t}" / "tfidf_c100_loto.csv", None))
    for s in SEEDS:
        for es in ("v2_test", "injecagent", "deepset"):
            out.append((f"deberta_s{s}", es, STD / es / f"deberta_s{s}.csv", STD / "v2_val" / f"deberta_s{s}.csv"))
        for t in LOTO_TASKS:
            cal = STD / f"v2_loto_val_{t}" / f"deberta_loto_s{s}.csv"
            out.append((f"deberta_loto_s{s}", f"v2_loto_test_{t}", STD / f"v2_loto_test_{t}" / f"deberta_loto_s{s}.csv",
                        cal if cal.exists() else None))
    for name in ("protectai", "piguard"):
        cal = latest_probe_run(f"{name}_v2val")
        for es in ("v2_test", "injecagent", "deepset"):
            out.append((name, es, STD / es / f"{name}.csv", cal))
    judge_cal = latest_probe_run("judge_qwen2.5_7b_v2val")
    out.append(("judge_qwen2.5_7b", "v2_test", JUDGE_V2_TEST, judge_cal))
    out.append(("judge_qwen2.5_7b", "injecagent", STD / "injecagent" / "judge_qwen2.5_7b.csv", judge_cal))
    return out


def load(path):
    d = pd.read_csv(path, keep_default_na=False)
    d["label"] = d["label"].astype(int)
    d["prob"] = d["prob"].astype(float)
    return d.reset_index(drop=True)


def secondary(d):
    m = df_metrics(d)
    return {"roc_auc": m["roc_auc"], "fpr_default_clean": m["flag_rate_clean"], "f1_macro": m["f1_macro"]}


def evaluate(model, eval_set, path, cal_path):
    d = load(path)
    cluster = "doc_id" if (d["doc_id"].astype(str) != "").all() else "id"
    cal = None
    if cal_path is not None:
        c = load(cal_path)
        cal = c.loc[c["label"] == 0, "prob"].values
    return {
        "model": model, "eval_set": eval_set, "source": str(path.relative_to(ROOT)),
        "calibration_source": str(cal_path.relative_to(ROOT)) if cal_path else None,
        "cluster_col": cluster, "n": len(d), "n_clean": int((d["label"] == 0).sum()),
        "n_injected": int((d["label"] == 1).sum()), "n_clusters": int(d[cluster].nunique()),
        "default_threshold": float(d["threshold"].iloc[0]),
        "tpr": tpr_at_fpr_protocols(d, cal, B=B, seed=BOOT_SEED, cluster_col=cluster),
        "bootstrap": doc_cluster_bootstrap(d, secondary, B=B, seed=BOOT_SEED, cluster_col=cluster),
    }


def seed_summary(rows):
    """mean ± SD over DeBERTa seeds of each point value that is shown for every seed."""
    out = {}
    for fam in ("deberta", "deberta_loto"):
        for es in sorted({r["eval_set"] for r in rows if r["model"].startswith(fam + "_s")}):
            rs = [r for r in rows if r["eval_set"] == es and r["model"] in {f"{fam}_s{s}" for s in SEEDS}]
            if len(rs) != len(SEEDS):
                continue
            vals = {k: [r["bootstrap"][k]["point"] for r in rs] for k in ("roc_auc", "fpr_default_clean", "f1_macro")}
            for proto, e in rs[0]["tpr"].items():
                if all(r["tpr"][proto]["status"] == "ok" for r in rs):
                    vals[f"tpr_{proto}"] = [r["tpr"][proto]["tpr"]["point"] for r in rs]
            out[f"{fam}/{es}"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "n_seeds": len(v)}
                                  for k, v in vals.items()}
    return out


def main():
    a = add_run_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    src = sources()
    files = sorted({p for _, _, f, c in src for p in ([f] + ([c] if c else []))})
    run = start_run(version="v2_metrics", exp="tpr_protocols", seed=BOOT_SEED,
                    args={"bootstrap": {"B": B, "seed": BOOT_SEED}, "seeds": list(SEEDS),
                          "sources": [[m, e, str(f.relative_to(ROOT)), str(c.relative_to(ROOT)) if c else None]
                                      for m, e, f, c in src]},
                    data_version="v2-frozen", data_files=files, allow_dirty=a.allow_dirty)
    rows = []
    for model, es, path, cal in src:
        r = evaluate(model, es, path, cal)
        run.log(f"{model} / {es}: n_clean={r['n_clean']} " + " ".join(
            f"{k}={v['status'] if v['status'] != 'ok' else round(v['tpr']['point'], 4)}" for k, v in r["tpr"].items()))
        rows.append(r)
    run.save_metrics({"rows": rows, "seed_summary": seed_summary(rows)})
    run.finish()
    print(f"wrote {run.dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
