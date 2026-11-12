"""
v3_baselines_report.py
======================
Metrics of the V3 baselines (paper_plan §3, §7 step 4, A10, A16, A30, A31) from the scoring runs:

  Runs are the ones listed in configs/canonical_runs.json (never the latest on disk).

  ours        results/v3/score_{tfidf,deberta}_v3/<run>/predictions.csv.gz (one run per split seed)
  detectors   results/v3/v3_detscores_*/<run>/scores/<det>__v3all.json, mapped back to V3 rows with
              scripts/v3_detector_sets.py's v3_map.csv (rebuilt here from data/v3; its data_version must match
              the score run's)
  H1          results/h1/score_deberta_v2/<run> (V2 DeBERTa seeds 0, 1, 42, 123, 999) and the agent rows of
              results/v3/score_deberta_v3/<run> (V3 DeBERTa, split seeds 0-4)

Every model value is the run-mean over the five split seeds (detectors: one deterministic run per split seed),
with a 95% CI from B = 1000 doc_id-cluster resamples stratified by task (A10); a doc that occurs in several split
seeds is drawn with all of its rows. Thresholds fixed outside the evaluated rows (calibration) are not resampled.

V3 test: task-macro AUROC, FPR at 0.5 on clean rows and macro-F1, the per-task values, and pooled TPR@FPR:
calibrated on the same split seed's V3 val clean rows (primary) and oracle, under the MIN_CLEAN gates (A16: the
calibrated gate is on the calibration rows). raw_clean: FPR at 0.5. ambiguous: flag rate at 0.5. long: AUROC,
macro-F1, recall and the clean flag rate at 0.5. val_id (A31): recall at 0.5, and AUROC / macro-F1 against the
same split seed's V3 val clean rows.

H1 (§3): run-mean calibrated TPR@1%FPR (threshold per run from its calibration-pool scores) on AD-Matched,
AD-Shift and TB-Matched injected rows, V3 - V2, paired cluster bootstrap over user tasks; realized FPR on the
evaluated clean rows (AD-Clean for AgentDojo, TB-Matched's own clean rows). Decision rule as pre-registered.

Probes (A30): flag rate at 0.5 per probe and position; within-row contrasts P3 end - start, P2 start - middle,
P1 start - middle, P4 - raw clean (same doc), each the run-mean of per-seed differences over rows paired by
source row.

    python scripts/v3_baselines_report.py
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src")]
from metrics import MIN_CLEAN, _resample_indices, calibrated_threshold  # noqa: E402
from runlog import add_run_args, start_run  # noqa: E402

TASKS = ("code", "email", "table")
B, SEED = 1000, 42
CANONICAL = ROOT / "configs" / "canonical_runs.json"
OURS = {"tfidf_v3": "score_tfidf_v3", "deberta_v3": "score_deberta_v3"}
H1_SETS = {"admatched": "adclean", "adshift": "adclean", "tbeval": None}
CONTRASTS = [("p3_end_minus_start", "p3_attack_moved", "end", "start"),
             ("p2_start_minus_middle", "p2_cleana_moved", "start", "middle"),
             ("p1_start_minus_middle", "p1_cleanb_moved", "start", "middle")]


# ─── loading ──────────────────────────────────────────────────────────────────

def canonical_runs(version, exp):
    """{seed: run_dir} from configs/canonical_runs.json (the accepted run per seed; Kaggle runs for the moved jobs),
    never the latest run on disk. Each listed run must exist, be finished and clean, and carry its seed."""
    listed = json.loads(CANONICAL.read_text()).get(f"{version}/{exp}", {})
    out = {}
    for seed, rel in listed.items():
        d = ROOT / rel
        cfg = json.loads((d / "config.json").read_text())
        if not (d / "finished.json").exists() or cfg["git_dirty"] or not d.name.endswith(f"_seed{seed}") \
                or d.parent.name != exp or d.parent.parent.name != version:
            raise ValueError(f"{CANONICAL.name}: {rel} is not a finished clean {version}/{exp} run of seed {seed}")
        out[int(seed)] = d
    return out


def our_predictions(exp):
    runs = canonical_runs("v3", exp)
    frames = [pd.read_csv(d / "predictions.csv.gz", keep_default_na=False).assign(run=k) for k, d in runs.items()]
    return (pd.concat(frames, ignore_index=True) if frames else None), runs


def v3_rows():
    """Every V3 row of every split seed and set, with the metadata the metrics need."""
    frames = []
    for k in range(5):
        for s in ("test", "val", "val_id", "raw_clean", "ambiguous", "long"):
            df = pd.read_csv(ROOT / f"data/v3/split_seed_{k}/{s}.csv", keep_default_na=False)
            frames.append(pd.DataFrame({"id": df["id"], "doc_id": df["doc_id"], "task": df["task_type"],
                                        "label": df["label"], "row_type": df["row_type"], "eval_set": f"v3_{s}",
                                        "run": k}))
    return pd.concat(frames, ignore_index=True)


def detector_predictions(score_runs):
    """{detector: frame in our prediction layout} from the v3all score files; checks the set version."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(ROOT / "scripts/v3_detector_sets.py"), "--out", tmp], check=True,
                       capture_output=True)
        man = json.loads((Path(tmp) / "manifest.json").read_text())
        vmap = pd.read_csv(Path(tmp) / "v3_map.csv")
    rows = v3_rows()
    rows = rows.merge(vmap.rename(columns={"split_seed": "run"}), on=["id", "doc_id", "run", "eval_set"], how="left")
    if rows["line"].isna().any():
        raise ValueError("v3_map does not cover every V3 row")
    out, sources = {}, {}
    for d in score_runs:
        cfg = json.loads((d / "config.json").read_text())
        if cfg["data_version"] != man["data_version"]:
            raise ValueError(f"{d}: scored on {cfg['data_version']}, data/v3 builds {man['data_version']}")
        for f in sorted((d / "scores").glob("*__v3all.json")):
            r = json.loads(f.read_text())
            p = np.asarray(r["probs"], float)
            out[r["detector"]] = rows.assign(prob=p[rows["line"].astype(int).values])
            out[r["detector"]]["pred"] = (out[r["detector"]]["prob"] >= 0.5).astype(int)
            sources[r["detector"]] = {"run": str(d.relative_to(ROOT)), "truncated_all_inputs": r.get("truncated")}
    return out, sources, man["data_version"]


# ─── metrics ──────────────────────────────────────────────────────────────────

def lab(d):
    return pd.to_numeric(d["label"], errors="coerce")


def auroc(d):
    y = lab(d)
    return float(roc_auc_score(y, d["prob"])) if y.nunique() == 2 else np.nan


def macro_f1(d):
    return float(f1_score(lab(d).astype(int), d["pred"].astype(int), average="macro")) if len(d) else np.nan


def fpr05(d):
    c = d[lab(d) == 0]
    return float(c["pred"].mean()) if len(c) else np.nan


def recall05(d):
    i = d[lab(d) == 1]
    return float(i["pred"].mean()) if len(i) else np.nan


def flag(d):
    return float(d["pred"].mean()) if len(d) else np.nan


def task_macro(fn):
    return lambda d: float(np.mean([fn(d[d["task"] == t]) for t in TASKS]))


def run_mean(fn):
    return lambda d: float(np.nanmean([fn(g) for _, g in d.groupby("run")]))


def boot(d, fn, strata="task"):
    """Point value and 95% CI of fn on frame d, doc_id clusters resampled within strata."""
    point = fn(d)
    vals = np.array([fn(d.iloc[i]) for i in _resample_indices(d.reset_index(drop=True), B, SEED, "doc_id",
                                                               strata)], float)
    return {"point": point, "ci": [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))],
            "per_run": {str(k): fn(g) for k, g in d.groupby("run")}}


def tpr_protocols(test, val):
    """Pooled TPR@1%/5% FPR on V3 test: calibrated on each run's V3 val clean rows, and oracle; MIN_CLEAN gates."""
    out = {}
    for target in (0.01, 0.05):
        key = f"tpr_at_{int(target * 100)}fpr"
        n_cal = int(val.groupby("run").apply(lambda g: (lab(g) == 0).sum()).min())
        n_ev = int(test.groupby("run").apply(lambda g: (lab(g) == 0).sum()).min())
        thr = {k: calibrated_threshold(g.loc[lab(g) == 0, "prob"].values, target) for k, g in val.groupby("run")}
        if n_cal < MIN_CLEAN[target]:
            out[f"{key}_calibrated"] = {"gated": f"n/a (calibration n_clean={n_cal})"}
        else:
            fn = run_mean(lambda g: float((g.loc[lab(g) == 1, "prob"] >= thr[g["run"].iloc[0]]).mean()))
            real = run_mean(lambda g: float((g.loc[lab(g) == 0, "prob"] >= thr[g["run"].iloc[0]]).mean()))
            out[f"{key}_calibrated"] = {**boot(test, fn), "realized_fpr": boot(test, real)}
        if n_ev < MIN_CLEAN[target]:
            out[f"{key}_oracle"] = {"gated": f"n/a (evaluated n_clean={n_ev})"}
        else:
            def oracle(g):
                t = calibrated_threshold(g.loc[lab(g) == 0, "prob"].values, target)
                return float((g.loc[lab(g) == 1, "prob"] >= t).mean())
            out[f"{key}_oracle"] = boot(test, run_mean(oracle))
    return out


def model_metrics(df):
    """All V3 set metrics of one model (frame with run, eval_set, task, doc_id, label, prob, pred)."""
    g = {s: d.reset_index(drop=True) for s, d in df.groupby("eval_set")}
    test, val = g["v3_test"], g["v3_val"]
    m = {"v3_test": {
        "auroc_task_macro": boot(test, run_mean(task_macro(auroc))),
        "fpr05_task_macro": boot(test, run_mean(task_macro(fpr05))),
        "f1_macro_task_macro": boot(test, run_mean(task_macro(macro_f1))),
        "auroc_pooled": boot(test, run_mean(auroc)),
        "f1_macro_pooled": boot(test, run_mean(macro_f1)),
        "per_task": {t: {"auroc": boot(test[test["task"] == t], run_mean(auroc), None),
                         "fpr05": boot(test[test["task"] == t], run_mean(fpr05), None),
                         "f1_macro": boot(test[test["task"] == t], run_mean(macro_f1), None)} for t in TASKS},
        **tpr_protocols(test, val)}}
    m["v3_raw_clean"] = {"fpr05": boot(g["v3_raw_clean"], run_mean(fpr05))}
    m["v3_ambiguous"] = {"flag_rate": boot(g["v3_ambiguous"], run_mean(flag))}
    lg = g["v3_long"]
    m["v3_long"] = {"auroc": boot(lg, run_mean(auroc)), "f1_macro": boot(lg, run_mean(macro_f1)),
                    "recall05": boot(lg, run_mean(recall05)), "fpr05": boot(lg, run_mean(fpr05))}
    vid = pd.concat([g["v3_val_id"], val[lab(val) == 0]], ignore_index=True)  # A31
    m["v3_val_id"] = {"recall05": boot(g["v3_val_id"], run_mean(recall05)), "auroc_vs_val_clean": boot(vid, run_mean(auroc)),
                      "f1_macro_vs_val_clean": boot(vid, run_mean(macro_f1))}
    return m


def probe_metrics(df):
    """Flag rates per probe x position and the paired within-row contrasts (A30)."""
    p = df[df["eval_set"].str.startswith("v3probe_")].copy()
    p["probe"] = p["eval_set"].str.removeprefix("v3probe_")
    p["position"] = p["id"].str.rsplit("__", n=1).str[1]
    p["source"] = p["id"].str.rsplit("__", n=1).str[0]
    out = {"cells": {}, "contrasts": {}}
    for (probe, pos), d in p.groupby(["probe", "position"]):
        out["cells"][f"{probe}/{pos}"] = boot(d.reset_index(drop=True), run_mean(flag))
    for name, probe, a, b in CONTRASTS:
        d = p[p["probe"] == probe]
        w = d.pivot_table(index=["run", "source", "doc_id", "task"], columns="position", values="pred").reset_index()
        w["diff"] = w[a] - w[b]
        out["contrasts"][name] = {"n_pairs": int(len(w)),
                                  **boot(w, run_mean(lambda g: float(g["diff"].mean())))}
    raw = df[df["eval_set"] == "v3_raw_clean"].assign(source=lambda x: x["doc_id"])
    p4 = p[p["probe"] == "p4_whitespace_only"].assign(source=lambda x: x["doc_id"])
    w = p4[["run", "doc_id", "task", "pred"]].merge(raw[["run", "doc_id", "pred"]], on=["run", "doc_id"],
                                                     suffixes=("_p4", "_raw"))
    w["diff"] = w["pred_p4"] - w["pred_raw"]
    out["contrasts"]["p4_minus_raw_clean"] = {"n_pairs": int(len(w)),
                                              **boot(w, run_mean(lambda g: float(g["diff"].mean())))}
    return out


def h1(v2, v3):
    """H1 (§3): paired run-mean calibrated TPR@1%FPR, V3 - V2, clusters = user tasks. Each run's threshold comes
    from its own calibration-pool scores and is not resampled."""
    def hits(df, arm):
        thr = {k: calibrated_threshold(g.loc[g["eval_set"] == "calpool", "prob"].values, 0.01)
               for k, g in df.groupby("run")}
        d = df.assign(hit=(df["prob"] >= df["run"].map(thr)).astype(float), col=arm + "_r" + df["run"].astype(str))
        return d, {str(k): float(v) for k, v in thr.items()}
    v2, t2 = hits(v2, "v2")
    v3, t3 = hits(v3, "v3")
    both = pd.concat([v2, v3], ignore_index=True)
    res = {"thresholds": {"v2": t2, "v3": t3}}

    def wide(eval_set, label):
        d = both[(both["eval_set"] == eval_set) & (lab(both) == label)]
        return d.pivot_table(index=["id", "doc_id"], columns="col", values="hit").reset_index()

    for s, neg in H1_SETS.items():
        w = wide(s, 1)
        c3 = [c for c in w.columns if str(c).startswith("v3_r")]
        c2 = [c for c in w.columns if str(c).startswith("v2_r")]
        if len(c3) != 5 or len(c2) != 5 or w[c3 + c2].isna().any().any():
            raise ValueError(f"H1 {s}: runs {c3} {c2} do not cover the same rows")
        f3 = lambda d: float(d[c3].mean().mean())  # noqa: E731  run-mean TPR (each run's TPR, then the mean)
        f2 = lambda d: float(d[c2].mean().mean())  # noqa: E731
        idx = _resample_indices(w, B, SEED, "doc_id", None)
        samples = np.array([[f3(w.iloc[i]), f2(w.iloc[i])] for i in idx])
        q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
        cl = wide(neg or s, 0)
        g3 = lambda d: float(d[c3].mean().mean())  # noqa: E731
        g2 = lambda d: float(d[c2].mean().mean())  # noqa: E731
        idc = _resample_indices(cl, B, SEED, "doc_id", None)
        fs = np.array([[g3(cl.iloc[i]), g2(cl.iloc[i])] for i in idc])
        res[s] = {"n_injected": int(len(w)), "n_clusters": int(w["doc_id"].nunique()),
                  "clean_set": neg or s, "n_clean": int(len(cl)),
                  "tpr_v3": {"point": f3(w), "ci": q(samples[:, 0]), "per_run": {c: float(w[c].mean()) for c in c3}},
                  "tpr_v2": {"point": f2(w), "ci": q(samples[:, 1]), "per_run": {c: float(w[c].mean()) for c in c2}},
                  "diff_v3_minus_v2": {"point": f3(w) - f2(w), "ci": q(samples[:, 0] - samples[:, 1])},
                  "realized_fpr_v3": {"point": g3(cl), "ci": q(fs[:, 0])},
                  "realized_fpr_v2": {"point": g2(cl), "ci": q(fs[:, 1])}}
    up = sum(res[s]["diff_v3_minus_v2"]["ci"][0] > 0 for s in H1_SETS)
    down = sum(res[s]["diff_v3_minus_v2"]["ci"][1] < 0 for s in H1_SETS)
    res["decision"] = "supported" if up >= 2 and down == 0 else "refuted" if down >= 2 else "inconclusive"
    res["n_sets_v3_better"], res["n_sets_v2_better"] = int(up), int(down)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no_detectors", action="store_true", help="skip the published detectors (not scored yet)")
    ap.add_argument("--no_h1", action="store_true")
    a = add_run_args(ap).parse_args()
    models, sources = {}, {}
    for name, exp in OURS.items():
        df, runs = our_predictions(exp)
        if df is None or len(runs) != 5:
            sys.exit(f"{exp}: {len(runs) if runs else 0} finished clean runs, need 5")
        models[name] = df[df["eval_set"].str.startswith("v3")]
        sources[name] = {str(k): str(d.relative_to(ROOT)) for k, d in runs.items()}
    det_version = None
    if not a.no_detectors:
        det_runs = [d for e in ("v3_detscores_liuenv", "v3_detscores_wolf", "v3_detscores_judges_bf16")
                    for d in canonical_runs("v3", e).values()]
        dets, dsrc, det_version = detector_predictions(det_runs)
        models.update(dets)
        sources.update(dsrc)
    files = [Path(p) / "predictions.csv.gz" for m in OURS for p in sources[m].values()]
    run = start_run(version="v3", exp="v3_baselines_report", seed=SEED,
                    args={"B": B, "seed": SEED, "sources": sources, "detector_set_version": det_version},
                    data_version="v3", data_files=[ROOT / f for f in files], allow_dirty=a.allow_dirty)
    out = {"models": {}, "probes": {}, "sources": sources}
    for name, df in models.items():
        run.log(f"metrics: {name}")
        out["models"][name] = model_metrics(df)
    for name in OURS:
        out["probes"][name] = probe_metrics(models[name])
    if not a.no_h1:
        v2, r2 = our_predictions_version("h1", "score_deberta_v2")
        v3, _ = our_predictions_version("v3", "score_deberta_v3")
        v3 = v3[~v3["eval_set"].str.startswith("v3")]
        if len(r2) != 5:
            sys.exit(f"H1: {len(r2)} V2 DeBERTa scoring runs, need 5")
        out["h1"] = h1(v2, v3)
        out["sources"]["h1_v2"] = {str(k): str(d.relative_to(ROOT)) for k, d in r2.items()}
        run.log(f"H1: {out['h1']['decision']}")
    head = out["models"]["deberta_v3"]["v3_test"]["auroc_task_macro"]["point"]
    run.save_metrics(out, headline=("deberta_v3_test_auroc_task_macro", head))
    run.finish()


def our_predictions_version(version, exp):
    runs = canonical_runs(version, exp)
    frames = [pd.read_csv(d / "predictions.csv.gz", keep_default_na=False).assign(run=k) for k, d in runs.items()]
    return pd.concat(frames, ignore_index=True), runs


if __name__ == "__main__":
    main()
