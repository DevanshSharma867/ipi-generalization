"""
v2_tpr_metrics.py
=================
Recompute the saved V2 scores with the headline metrics: TPR@1%FPR and TPR@5%FPR under the "oracle" and
"calibrated" threshold protocols (src/metrics.tpr_at_fpr_protocols), AUROC, FPR at the default threshold on
clean rows, and macro-F1 (secondary), each with a doc-cluster bootstrap 95% CI (B=1000, seed 42).
Reads saved predictions only; nothing is re-scored. Writes results/v2_metrics/tpr_protocols/<run_id>/.

Calibration set per model (clean rows only; "n/a (no calibration scores)" where none exist yet):
  - tfidf_c1, deberta_s<k>, protectai, piguard, judge: the V2 val split (data/processed/val.csv),
    from results/v2_std/v2_val/ or the latest results/v2_probes/score_<name>_v2val/ run
  - judge_qwen2.5_7b_vsum (variant-summing Yes/No rule, paper_plan A12): V2 test only, calibrated on its own V2 val run
  - LOTO models and tfidf_c100_tuned: skipped (V2 LOTO and C=100 are secondary; no C=100 checkpoint exists)
Bootstrap clusters: V2 sets resample doc_id. InjecAgent rows sharing a user case (user tool + user instruction)
come from one template, so they resample by user case (17 clusters). deepset prompts are independent: rows.

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
NOT_YET = "n/a (no calibration scores)"
SKIP_LOTO = "n/a (skipped: V2 LOTO is secondary)"
SKIP_C100 = "n/a (skipped: no C=100 checkpoint; V2 C=100 is secondary)"
EXTERNAL = {"injecagent": ROOT / "data/external/injecagent.csv", "deepset": ROOT / "data/external/deepset.csv"}
# Upstream revision + the commit of the build that wrote data/external/ (score_detectors.py build, bab750e).
# deepset's build did not pin a revision; 4f61ecb is the dataset revision in the local HF datasets cache.
DATA_VERSION = {"injecagent": "InjecAgent@f19c9f2 + build bab750e", "deepset": "hf:deepset/prompt-injections@4f61ecb + build bab750e"}


def latest_probe_run(name):
    runs = sorted((PROBES / f"score_{name}").glob("*/predictions.csv"))
    return runs[-1] if runs else None


def sources():
    """(model, eval_set, predictions file, calibration file or None, status when there is none)."""
    out = []
    tfidf_cal = latest_probe_run("tfidf_c1_v2val")
    for es, f in [("v2_test", "tfidf_c1"), ("injecagent", "tfidf"), ("deepset", "tfidf")]:
        out.append(("tfidf_c1", es, STD / es / f"{f}.csv", tfidf_cal, NOT_YET))
    out.append(("tfidf_c100_tuned", "v2_test", STD / "v2_test" / "tfidf_c100_tuned.csv", None, SKIP_C100))
    for t in LOTO_TASKS:
        out.append(("tfidf_c100_loto", f"v2_loto_test_{t}", STD / f"v2_loto_test_{t}" / "tfidf_c100_loto.csv", None,
                    SKIP_LOTO))
    for s in SEEDS:
        for es in ("v2_test", "injecagent", "deepset"):
            out.append((f"deberta_s{s}", es, STD / es / f"deberta_s{s}.csv", STD / "v2_val" / f"deberta_s{s}.csv",
                        NOT_YET))
        for t in LOTO_TASKS:
            out.append((f"deberta_loto_s{s}", f"v2_loto_test_{t}", STD / f"v2_loto_test_{t}" / f"deberta_loto_s{s}.csv",
                        None, SKIP_LOTO))
    for name in ("protectai", "piguard"):
        cal = latest_probe_run(f"{name}_v2val")
        for es in ("v2_test", "injecagent", "deepset"):
            out.append((name, es, STD / es / f"{name}.csv", cal, NOT_YET))
    judge_cal = latest_probe_run("judge_qwen2.5_7b_v2val")
    out.append(("judge_qwen2.5_7b", "v2_test", JUDGE_V2_TEST, judge_cal, NOT_YET))
    out.append(("judge_qwen2.5_7b", "injecagent", STD / "injecagent" / "judge_qwen2.5_7b.csv", judge_cal, NOT_YET))
    vsum = latest_probe_run("judge_qwen2.5_7b_vsum")  # variant-summing Yes/No rule (paper_plan A12); V2 test only
    if vsum:
        out.append(("judge_qwen2.5_7b_vsum", "v2_test", vsum, latest_probe_run("judge_qwen2.5_7b_vsum_v2val"), NOT_YET))
    return out


def clusters(d, eval_set):
    """(cluster column added to d, its description)."""
    if eval_set == "injecagent":
        ext = pd.read_csv(EXTERNAL["injecagent"]).set_index("id").loc[d["id"]]
        return d.assign(cluster=(ext["task_type"] + " | " + ext["user_intent"]).values), "user_case (user tool + user instruction)"
    if eval_set == "deepset":
        return d.assign(cluster=d["id"]), "row (independent prompts)"
    return d.assign(cluster=d["doc_id"]), "doc_id"


def load(path, eval_set):
    """The eval_set rows of a predictions file (a probe-scoring run also holds the P1-P4 rows)."""
    d = pd.read_csv(path, keep_default_na=False)
    d = d[d["eval_set"] == eval_set]
    if d.empty:
        raise ValueError(f"{path} has no {eval_set} rows")
    d["label"] = d["label"].astype(int)
    d["prob"] = d["prob"].astype(float)
    return d.reset_index(drop=True)


def secondary(d):
    m = df_metrics(d)
    return {"roc_auc": m["roc_auc"], "fpr_default_clean": m["flag_rate_clean"], "f1_macro": m["f1_macro"]}


def evaluate(model, eval_set, path, cal_path, no_cal):
    d, cluster_desc = clusters(load(path, eval_set), eval_set)
    cluster = "cluster"
    cal = None
    if cal_path is not None:
        c = load(cal_path, "v2_val")
        cal = c.loc[c["label"] == 0, "prob"].values
    return {
        "model": model, "eval_set": eval_set, "source": str(path.relative_to(ROOT)),
        "calibration_source": str(cal_path.relative_to(ROOT)) if cal_path else None,
        "cluster": cluster_desc, "data_version": DATA_VERSION.get(eval_set, "v2-frozen"), "n": len(d), "n_clean": int((d["label"] == 0).sum()),
        "n_injected": int((d["label"] == 1).sum()), "n_clusters": int(d[cluster].nunique()),
        "default_threshold": float(d["threshold"].iloc[0]),
        "tpr": tpr_at_fpr_protocols(d, cal, B=B, seed=BOOT_SEED, cluster_col=cluster, no_calib_status=no_cal),
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
    files = sorted({p for _, _, f, c, _ in src for p in ([f] + ([c] if c else []))} | set(EXTERNAL.values()))
    run = start_run(version="v2_metrics", exp="tpr_protocols", seed=BOOT_SEED,
                    args={"bootstrap": {"B": B, "seed": BOOT_SEED}, "seeds": list(SEEDS),
                          "sources": [[m, e, str(f.relative_to(ROOT)), str(c.relative_to(ROOT)) if c else None]
                                      for m, e, f, c, _ in src], "external_data_versions": DATA_VERSION},
                    data_version="v2-frozen+external", data_files=files, allow_dirty=a.allow_dirty)
    rows = []
    for model, es, path, cal, no_cal in src:
        r = evaluate(model, es, path, cal, no_cal)
        run.log(f"{model} / {es}: n_clean={r['n_clean']} " + " ".join(
            f"{k}={v['status'] if v['status'] != 'ok' else round(v['tpr']['point'], 4)}" for k, v in r["tpr"].items()))
        rows.append(r)
    run.save_metrics({"rows": rows, "seed_summary": seed_summary(rows)})
    run.finish()
    print(f"wrote {run.dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
