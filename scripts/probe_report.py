"""
probe_report.py
===============
Artifact-reliance report from the latest results/v2_probes/score_<model>/<run_id>/predictions.csv of every
scored model (missing models are skipped and listed).

  - artifact reliance: flag rate on P1 and P2 by position, recall on P3 by position, flag rate on P4,
    V2 test macro-F1 -- each with a doc-cluster bootstrap 95% CI (B=1000, seed 42)
  - flag rate on the untouched raw clean V2 test rows (natural false-positive rate)
  - paired within-doc contrasts (paired cluster bootstrap of the difference; rows paired by their source
    row, i.e. same doc and same inserted text): P1 start - P1 middle, P3 end - P3 start, P2 start - P2 middle,
    and P4 - raw clean (same doc, whitespace-collapsed vs untouched)
  - figure: flag rate vs position per model, P1/P2/P3 as separate lines (paper/figures/position_flag_rates.{pdf,png})

Logged with runlog to results/v2_probes/probe_report/<run_id>/; scripts/make_tables.py turns its
metrics.json into tables and paper/numbers.tex macros.

    python scripts/probe_report.py
"""

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import df_metrics, doc_cluster_bootstrap, paired_bootstrap
from runlog import start_run

PROBES_RES = ROOT / "results" / "v2_probes"
FIG = ROOT / "paper" / "figures" / "position_flag_rates"
POSITIONS = ("start", "middle", "end")
MODEL_ORDER = ["tfidf_c1", "deberta_s42", "deberta_s123", "deberta_s999", "deberta_mean", "protectai", "piguard",
               "judge_qwen2.5_7b"]
CONTRASTS = [("p1_start_minus_middle", "p1_neutral_insertion", "start", "middle"),
             ("p3_end_minus_start", "p3_attack_moved", "end", "start"),
             ("p2_start_minus_middle", "p2_benign_moved", "start", "middle")]
LINES = [("p1_neutral_insertion", "P1 neutral (flag rate)", "#2a78d6", "o"),
         ("p2_benign_moved", "P2 benign moved (flag rate)", "#eb6834", "s"),
         ("p3_attack_moved", "P3 attack moved (recall)", "#1baf7a", "^")]


MEAN_SEEDS = ("deberta_s42", "deberta_s123", "deberta_s999")


def flag_rate(d):
    return float(d["pred"].mean())


def f1(d):
    return df_metrics(d)["f1_macro"]


def seed_cols(d):
    return [c for c in d.columns if c.startswith("pred__")]


def mean_flag_rate(d):
    """Seed mean: average of the per-seed flag rates on the same rows."""
    return float(np.mean([d[c].mean() for c in seed_cols(d)]))


def mean_f1(d):
    return float(np.mean([f1_score(d["label"], d[c], average="macro", zero_division=0) for c in seed_cols(d)]))


def seed_mean_frame(frames):
    """One frame with each seed's predictions as pred__<model>, rows aligned by (eval_set, id)."""
    base = frames[MEAN_SEEDS[0]].drop(columns=["prob", "pred", "model", "run_id"])
    for m in MEAN_SEEDS:
        f = frames[m].set_index(["eval_set", "id"]).loc[list(zip(base["eval_set"], base["id"]))]
        assert (f["label"].values == base["label"].values).all()
        base[f"pred__{m}"] = f["pred"].values
    return base


def latest_predictions():
    """model -> (predictions, sources). A model may have several scoring runs (e.g. the judge's probe run and
    its later V2-test run); for each eval_set the latest run that scored it wins."""
    found = {}
    for exp in sorted(PROBES_RES.glob("score_*")):
        per_set = {}
        for run in sorted(p for p in exp.iterdir() if (p / "predictions.csv").exists()):
            df = pd.read_csv(run / "predictions.csv", keep_default_na=False)
            for s, g in df.groupby("eval_set"):
                per_set[s] = (g, run / "predictions.csv")
        if per_set:
            frames = pd.concat([g for g, _ in per_set.values()], ignore_index=True)
            found[exp.name[len("score_"):]] = (frames, sorted({str(p.relative_to(ROOT)) for _, p in per_set.values()}))
    return found


def pair_key(df):
    """Rows built from the same source row share id minus the trailing _<position>."""
    return df.assign(id=df["id"].str.rsplit("_", n=1).str[0])


def model_report(df, rate_fn=flag_rate, f1_fn=f1):
    out = {"cells": {}, "contrasts": {}}
    for s in ("p1_neutral_insertion", "p2_benign_moved", "p3_attack_moved"):
        d = df[df["eval_set"] == s]
        for pos in POSITIONS:
            sub = d[d["insertion_mode"] == pos].reset_index(drop=True)
            if len(sub):
                out["cells"][f"{s}/{pos}"] = {"n": len(sub), **doc_cluster_bootstrap(sub, rate_fn)}
    p4 = df[df["eval_set"] == "p4_whitespace_only"].reset_index(drop=True)
    if len(p4):
        out["cells"]["p4_whitespace_only"] = {"n": len(p4), **doc_cluster_bootstrap(p4, rate_fn)}
    v2 = df[df["eval_set"] == "v2_test"].reset_index(drop=True)
    if len(v2):
        out["cells"]["v2_test_f1_macro"] = {"n": len(v2), **doc_cluster_bootstrap(v2, f1_fn)}
    raw = v2[v2["clean_type"] == "raw"].reset_index(drop=True) if len(v2) else v2
    if len(raw):
        out["cells"]["v2_test_raw_clean_flag"] = {"n": len(raw), **doc_cluster_bootstrap(raw, rate_fn)}
    for name, s, a, b in CONTRASTS:
        d = df[df["eval_set"] == s]
        da, db = (pair_key(d[d["insertion_mode"] == p]).reset_index(drop=True) for p in (a, b))
        if len(da) and len(db):
            out["contrasts"][name] = {"n_pairs": len(da), **paired_bootstrap(da, db, rate_fn)}
    if len(p4) and len(raw):
        # Same 72 docs, one row each: whitespace-collapsed vs untouched raw context
        da, db = (x.assign(id=x["doc_id"]).reset_index(drop=True) for x in (p4, raw))
        out["contrasts"]["p4_minus_raw_clean"] = {"n_pairs": len(da), **paired_bootstrap(da, db, rate_fn)}
    return out


def figure(report, run_dir):
    models = [m for m in MODEL_ORDER if m in report] + [m for m in report if m not in MODEL_ORDER]
    n = len(models)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), sharey=True, squeeze=False)
    for ax, m in zip(axes.flat, models):
        cells = report[m]["cells"]
        for s, label, color, marker in LINES:
            pts = [cells.get(f"{s}/{p}") for p in POSITIONS]
            if not all(pts):
                continue
            y = np.array([p["point"] for p in pts])
            lo = y - np.array([p["ci"][0] for p in pts])
            hi = np.array([p["ci"][1] for p in pts]) - y
            ax.errorbar(range(3), y, yerr=[lo, hi], color=color, marker=marker, markersize=6, lw=2,
                        capsize=3, label=label)
        ax.set_title(m, fontsize=10)
        ax.set_xticks(range(3), POSITIONS)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("rate flagged as injected")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}.{ext}", bbox_inches="tight", dpi=200)
        shutil.copy(f"{FIG}.{ext}", run_dir / f"position_flag_rates.{ext}")
    plt.close(fig)


def main():
    found = latest_predictions()
    missing = [m for m in MODEL_ORDER if m not in found and m != "deberta_mean"]
    sources = {m: src for m, (_, src) in found.items()}
    run = start_run(version="v2_probes", exp="probe_report", seed=42,
                    args={"sources": sources, "missing": missing, "bootstrap": {"B": 1000, "seed": 42}},
                    data_version="v2_probes", data_files=sorted({ROOT / p for src in sources.values() for p in src}))
    report, frames = {}, {}
    for m, (df, src) in found.items():
        run.log(f"{m}: {', '.join(src)}")
        frames[m] = df
        report[m] = model_report(df)
    if all(m in frames for m in MEAN_SEEDS):
        run.log(f"deberta_mean: seed mean of {MEAN_SEEDS} (same doc draw for every seed)")
        report["deberta_mean"] = model_report(seed_mean_frame(frames), mean_flag_rate, mean_f1)
    report = {m: report[m] for m in [m for m in MODEL_ORDER if m in report] + [m for m in report if m not in MODEL_ORDER]}
    figure(report, run.dir)
    run.save_metrics({"models": report, "missing_models": missing})
    run.log(f"missing models (not yet scored): {missing}")
    run.finish()
    print(f"wrote {run.dir.relative_to(ROOT)} and {FIG.relative_to(ROOT)}.pdf/.png")


if __name__ == "__main__":
    main()
