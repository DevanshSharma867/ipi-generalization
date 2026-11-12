"""
v3_style_vs_deberta.py
======================
Does V3 DeBERTa lean on the span style that the F026 style probe picks up? On V3 test rows (injected and clean-A,
the rows the F026 clean-A-only probe scores), DeBERTa accuracy at 0.5 where the style probe is confidently wrong vs
confidently right, per split seed and pooled.

Style probe: scripts/v3_sanity.py style_probe (all style features, clean-A only), trained per split seed on that
seed's train spans, as in F026. Bins, fixed before the run: confident = the probe gives >= CONF to one class;
"confident_wrong" = that class is not the row's label, "confident_right" = it is, "unconfident" = the rest.
DeBERTa: the canonical score_deberta_v3 run per split seed (configs/canonical_runs.json), threshold 0.5.

CIs: B = 1000 doc_id-cluster resamples stratified by task (as F026); per seed on that seed's rows, pooled as the
run-mean over seeds. right - wrong is computed on the same resamples. Also by label, since the bins differ in class
mix. Diagnostic only: nothing is tuned or selected on these rows.

    python scripts/v3_style_vs_deberta.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from metrics import _resample_indices  # noqa: E402
from runlog import add_run_args, start_run  # noqa: E402
from v3_baselines_report import canonical_runs  # noqa: E402
from v3_sanity import load, style_probe  # noqa: E402

CONF = 0.9
BINS = ("confident_wrong", "confident_right", "unconfident")
SEEDS = range(5)
B, SEED = 1000, 42


def bin_rows(p, y):
    pc = np.where(y == 1, p, 1 - p)  # probability the probe gives to the true class
    return np.where(pc <= 1 - CONF, "confident_wrong", np.where(pc >= CONF, "confident_right", "unconfident"))


def acc(d, b, label=None):
    s = d[(d["bin"] == b) & ((d["label"] == label) if label is not None else True)]
    return float((s["deb_pred"] == s["label"]).mean()) if len(s) else np.nan


def stats(d):
    """{name: value} for one frame (one seed, or all seeds -> run-mean)."""
    per = [d] if d["run"].nunique() == 1 else [g for _, g in d.groupby("run")]
    out = {}
    for b in BINS:
        out[f"acc_{b}"] = float(np.nanmean([acc(g, b) for g in per]))
        for lab in (0, 1):
            out[f"acc_{b}_label{lab}"] = float(np.nanmean([acc(g, b, lab) for g in per]))
    out["acc_right_minus_wrong"] = out["acc_confident_right"] - out["acc_confident_wrong"]
    for lab in (0, 1):
        out[f"acc_right_minus_wrong_label{lab}"] = out[f"acc_confident_right_label{lab}"] - \
            out[f"acc_confident_wrong_label{lab}"]
    return out


def boot(d):
    d = d.reset_index(drop=True)
    point = stats(d)
    samples = [stats(d.iloc[i]) for i in _resample_indices(d, B, SEED, "doc_id", "task")]
    return {k: {"point": v, "ci": [float(np.nanpercentile([s[k] for s in samples], 2.5)),
                                   float(np.nanpercentile([s[k] for s in samples], 97.5))]}
            for k, v in point.items()}


def counts(d):
    return {b: {"rows": int((d["bin"] == b).sum()), "label1": int(((d["bin"] == b) & (d["label"] == 1)).sum())}
            for b in BINS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = add_run_args(ap).parse_args()
    runs = canonical_runs("v3", "score_deberta_v3")
    if sorted(runs) != list(SEEDS):
        sys.exit(f"need canonical V3 DeBERTa scoring runs for split seeds 0-4: {sorted(runs)}")
    files = [d / "predictions.csv.gz" for d in runs.values()]
    run = start_run(version="v3", exp="v3_style_vs_deberta", seed=SEED,
                    args={"conf": CONF, "B": B, "seed": SEED, "probe": "v3_sanity.style_probe(all, clean_a_only)",
                          "deberta_runs": {str(k): str(d.relative_to(ROOT)) for k, d in runs.items()}},
                    data_version="v3", data_files=files, allow_dirty=a.allow_dirty)
    frames = []
    for k in SEEDS:
        main_k, _ = load(k)
        sp = style_probe(main_k, with_clean_b=False)
        deb = pd.read_csv(runs[k] / "predictions.csv.gz", keep_default_na=False)
        deb = deb[deb["eval_set"] == "v3_test"][["id", "prob", "pred"]].rename(
            columns={"prob": "deb_prob", "pred": "deb_pred"})
        m = sp.merge(deb, on="id", how="left", validate="one_to_one")
        if m["deb_pred"].isna().any():
            raise ValueError(f"split seed {k}: {int(m['deb_pred'].isna().sum())} probe rows without a DeBERTa score")
        m["bin"] = bin_rows(m["prob"].values, m["label"].values)
        frames.append(m.assign(run=k))
        run.log(f"seed {k}: {counts(m)}")
    d = pd.concat(frames, ignore_index=True)
    out = {"conf": CONF, "per_seed": {}, "counts": {}}
    for k, g in d.groupby("run"):
        out["per_seed"][str(k)] = boot(g)
        out["counts"][str(k)] = counts(g)
    out["pooled_run_mean"] = boot(d)
    out["counts"]["all"] = counts(d)
    run.save_metrics(out, headline=("acc_right_minus_wrong_pooled", out["pooled_run_mean"]["acc_right_minus_wrong"]["point"]))
    run.finish()
    print(run.dir)


if __name__ == "__main__":
    main()
