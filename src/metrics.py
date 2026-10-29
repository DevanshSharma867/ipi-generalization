"""
metrics.py
==========
Single source of truth for evaluation metrics.

Conventions
-----------
- `y` is the binary label (1 = injected), `prob` is P(injected).
- Predictions are `prob >= threshold` unless explicit `pred` values are passed
  (e.g. argmax outputs from a classifier).
- Undefined values (ROC-AUC on a single class, recall with no positives, ...) are NaN,
  except where a zero_division=0 convention is noted.
- Recall@FPR uses the full ROC curve (`drop_intermediate=False`): the largest TPR among
  operating points with FPR <= target, and the *lowest* threshold that achieves it.
  This matches run_experiments.compute_recall_at_fpr exactly.
- TPR@FPR under two labelled threshold protocols (`tpr_at_fpr_protocols`):
  "oracle": the threshold is chosen on the clean rows of the evaluated set itself (= recall_at_fpr), and
  re-chosen inside every bootstrap resample. "calibrated": the threshold is fixed in advance on the clean rows of
  a separate calibration set (default: the training-side validation split) and held fixed in every resample, so
  its CI ignores calibration noise; the realized FPR on the evaluated set is reported next to it.
  A TPR@FPR is shown only when the evaluated set (and, for "calibrated", the calibration set) has at least
  MIN_CLEAN[target] clean rows; otherwise its status is "n/a (n_clean=<n>)".
- doc_cluster_bootstrap draws doc_ids with np.random.RandomState(seed).choice in
  order-of-appearance, exactly as run_experiments.doc_cluster_bootstrap did, so V2 CIs
  are reproducible to the bit.
"""

from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

FPR_TARGETS = (0.01, 0.05, 0.10)


def _fpr_key(target: float) -> str:
    return f"{int(round(target * 100))}pct_fpr"


def recall_at_fpr(y, prob, target_fpr: float):
    """(recall, operating threshold) at the largest TPR with FPR <= target_fpr. NaN if one class."""
    y, prob = np.asarray(y), np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, prob, drop_intermediate=False)
    ok = fpr <= target_fpr
    best = tpr[ok].max()
    return float(best), float(thr[ok & (tpr == best)].min())


def expected_calibration_error(y, prob, n_bins: int = 10) -> float:
    """ECE of P(injected) against the label, `n_bins` equal-width bins on [0, 1] (1.0 in the last bin)."""
    y, prob = np.asarray(y, dtype=float), np.asarray(prob, dtype=float)
    if len(y) == 0:
        return float("nan")
    bins = np.clip(np.digitize(prob, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - prob[m].mean())
    return float(ece)


def point_metrics(y, prob, threshold: float = 0.5, pred=None) -> Dict[str, float]:
    """All point metrics for one set of predictions."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int) if pred is None else np.asarray(pred).astype(int)
    both = len(np.unique(y)) > 1
    clean, inj = y == 0, y == 1
    out = {
        "n": int(len(y)), "n_injected": int(inj.sum()), "n_clean": int(clean.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y, pred, pos_label=1, average="binary", zero_division=0)),
        "recall": float(recall_score(y, pred, pos_label=1, average="binary", zero_division=0)),
        "f1_injected": float(f1_score(y, pred, pos_label=1, average="binary", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, prob)) if both else float("nan"),
        "n_fp": int((clean & (pred == 1)).sum()),
        "n_fn": int((inj & (pred == 0)).sum()),
        "flag_rate_clean": float(pred[clean].mean()) if clean.any() else float("nan"),
        "flag_rate_injected": float(pred[inj].mean()) if inj.any() else float("nan"),
        "ece": expected_calibration_error(y, prob),
    }
    out["fn_rate"] = 1.0 - out["flag_rate_injected"]
    for t in FPR_TARGETS:
        r, th = recall_at_fpr(y, prob, t)
        out[f"recall_at_{_fpr_key(t)}"] = r
        out[f"threshold_at_{_fpr_key(t)}"] = th
    return out


def df_metrics(df: pd.DataFrame, threshold: Optional[float] = None, use_pred: bool = True) -> Dict[str, float]:
    """point_metrics on a standard-schema frame (label, prob, optional pred/threshold columns)."""
    if threshold is None:
        threshold = float(df["threshold"].iloc[0]) if "threshold" in df and len(df) else 0.5
    pred = df["pred"].values if use_pred and "pred" in df else None
    return point_metrics(df["label"].values, df["prob"].values, threshold, pred)


def group_metrics(df: pd.DataFrame, by: str, metric_fn: Callable = df_metrics) -> Dict[str, Dict]:
    """metric_fn on each group of `by` (task, attack_category, insertion_mode, clean_type, ...).
    Empty / NaN group values are reported under the key "" ."""
    keys = df[by].fillna("").astype(str)
    return {k: metric_fn(df[keys == k]) for k in sorted(keys.unique())}


def pair_keys(df: pd.DataFrame) -> pd.Series:
    """pair_id column if present, else the id with its trailing '_<role>' removed (pairs_v0 ids)."""
    if "pair_id" in df:
        return df["pair_id"].astype(str)
    return df["id"].astype(str).str.rsplit("_", n=1).str[0]


def pair_accuracy(df: pd.DataFrame, score_col: str = "prob") -> float:
    """Fraction of pairs whose label-1 row scores strictly above the label-0 row; ties count 0.5."""
    d = pd.DataFrame({"pair": pair_keys(df).values, "label": df["label"].values,
                      "score": df[score_col].astype(float).values})
    wide = d.pivot(index="pair", columns="label", values="score")
    diff = wide[1] - wide[0]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def pair_metrics(df: pd.DataFrame, score_col: str = "prob") -> Dict[str, float]:
    y = df["label"].values
    return {
        "n_pairs": int(pair_keys(df).nunique()),
        "pair_accuracy": pair_accuracy(df, score_col),
        "roc_auc": float(roc_auc_score(y, df[score_col])) if len(np.unique(y)) > 1 else float("nan"),
    }


# ─── TPR@FPR threshold protocols ──────────────────────────────────────────────

HEADLINE_FPR_TARGETS = (0.01, 0.05)
MIN_CLEAN = {0.01: 500, 0.05: 100}


def calibrated_threshold(clean_scores, target_fpr: float) -> float:
    """Smallest threshold t with mean(clean_scores >= t) <= target_fpr: just above the (k+1)-th largest clean
    score, k = floor(target_fpr * n). Ties are handled because t exceeds every tied copy of that score."""
    s = np.sort(np.asarray(clean_scores, dtype=float))[::-1]
    k = int(np.floor(target_fpr * len(s) + 1e-9))
    return float("-inf") if k >= len(s) else float(np.nextafter(s[k], np.inf))


def _gate(n_clean: int, target: float, what: str = "n_clean") -> Optional[str]:
    return None if n_clean >= MIN_CLEAN[target] else f"n/a ({what}={n_clean})"


def tpr_at_fpr_protocols(df: pd.DataFrame, calib_clean_scores=None, targets=HEADLINE_FPR_TARGETS,
                         B: int = 1000, seed: int = 42, cluster_col: str = "doc_id",
                         no_calib_status: str = "n/a (no calibration scores)") -> Dict[str, Dict]:
    """
    {"oracle@1pct_fpr": {...}, "calibrated@1pct_fpr": {...}, ...} for a frame with label/prob/cluster_col.
    Each leaf: status ("ok" or "n/a (...)"), n_clean, threshold and, when ok, tpr {"point", "ci"};
    "calibrated" also gives n_clean_calib and realized_fpr {"point", "ci"} on the evaluated set.
    Pass calib_clean_scores=None when there is no calibration set; its status is then `no_calib_status`.
    """
    df = df.reset_index(drop=True)
    y, prob = df["label"].values.astype(int), df["prob"].values.astype(float)
    n_clean = int((y == 0).sum())
    out = {}
    for t in targets:
        key = _fpr_key(t)
        gate = _gate(n_clean, t)
        o = {"status": gate or "ok", "n_clean": n_clean, "target_fpr": t}
        if gate is None:
            o["threshold"] = recall_at_fpr(y, prob, t)[1]
            o["tpr"] = doc_cluster_bootstrap(df, lambda d: recall_at_fpr(d["label"], d["prob"], t)[0],
                                             B=B, seed=seed, cluster_col=cluster_col)
        out[f"oracle@{key}"] = o

        c = {"n_clean": n_clean, "target_fpr": t}
        if calib_clean_scores is None:
            c["status"] = no_calib_status
        else:
            n_cal = int(len(calib_clean_scores))
            c["n_clean_calib"] = n_cal
            c["status"] = gate or _gate(n_cal, t, "n_clean_calib") or "ok"
            c["threshold"] = calibrated_threshold(calib_clean_scores, t)
            if c["status"] == "ok":
                thr = c["threshold"]
                both = doc_cluster_bootstrap(df, lambda d: {
                    "tpr": float((d["prob"][d["label"] == 1] >= thr).mean()),
                    "fpr": float((d["prob"][d["label"] == 0] >= thr).mean())}, B=B, seed=seed, cluster_col=cluster_col)
                c["tpr"], c["realized_fpr"] = both["tpr"], both["fpr"]
        out[f"calibrated@{key}"] = c
    return out


def format_tpr(entry: Dict, digits: int = 4) -> str:
    """'0.8125 [0.70, 0.91]' when shown, else the gate status, e.g. 'n/a (n_clean=144)'."""
    if entry["status"] != "ok":
        return entry["status"]
    lo, hi = entry["tpr"]["ci"]
    return f"{entry['tpr']['point']:.{digits}f} [{lo:.2f}, {hi:.2f}]"


# ─── Bootstrap ────────────────────────────────────────────────────────────────

def _resample_indices(df: pd.DataFrame, B: int, seed: int, cluster_col: str, strata_col: Optional[str] = None):
    """Row indices of B cluster resamples. With strata_col, clusters are drawn with replacement within each
    stratum (in order of first appearance), so every resample keeps each stratum's cluster count; a cluster must
    lie in one stratum. strata_col=None is the unstratified draw (bit-identical to the V2 bootstrap)."""
    rng = np.random.RandomState(seed)
    clusters = df[cluster_col].unique()
    values = df[cluster_col].values
    members = {c: np.where(values == c)[0] for c in clusters}
    if strata_col is None:
        groups = [clusters]
    else:
        per_cluster = df.groupby(cluster_col, sort=False)[strata_col].nunique()
        if (per_cluster > 1).any():
            raise ValueError(f"clusters span several {strata_col} values: {list(per_cluster[per_cluster > 1].index[:3])}")
        first = df.drop_duplicates(cluster_col)
        groups = [first.loc[first[strata_col] == s, cluster_col].values for s in first[strata_col].unique()]
    for _ in range(B):
        sampled = [c for g in groups for c in rng.choice(g, size=len(g), replace=True)]
        yield np.concatenate([members[c] for c in sampled])


def _summarize(samples, point, ci: float):
    lo_q, hi_q = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0
    if isinstance(point, dict):
        return {k: _summarize([s[k] for s in samples], point[k], ci) for k in point}
    arr = np.asarray(samples, dtype=float)
    if np.all(np.isnan(arr)):
        return {"point": point, "ci": [float("nan"), float("nan")]}
    return {"point": point, "ci": [float(np.nanpercentile(arr, lo_q)), float(np.nanpercentile(arr, hi_q))]}


def doc_cluster_bootstrap(df: pd.DataFrame, metric_fn: Callable, B: int = 1000, seed: int = 42,
                          ci: float = 95.0, cluster_col: str = "doc_id", strata_col: Optional[str] = None):
    """
    Cluster bootstrap over documents: resample doc_ids with replacement and keep every row of each
    sampled doc. `metric_fn(frame)` may return a float or a dict of floats; the result mirrors it
    with {"point", "ci"} leaves. Percentile CIs, NaN resamples ignored. strata_col (e.g. "task") resamples
    clusters within each stratum (paper_plan A10: V3 task-macro CIs).
    """
    df = df.reset_index(drop=True)
    samples = [metric_fn(df.iloc[idx]) for idx in _resample_indices(df, B, seed, cluster_col, strata_col)]
    return _summarize(samples, metric_fn(df), ci)


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame, metric_fn: Callable, B: int = 1000,
                     seed: int = 42, ci: float = 95.0, on: str = "id", cluster_col: str = "doc_id",
                     strata_col: Optional[str] = None):
    """
    Paired cluster bootstrap of metric(A) - metric(B) on the same rows (joined on `on`).
    Each resample applies the same doc draw to both models. Returns the point difference,
    the percentile CI and the fraction of resamples with difference <= 0.
    """
    a = df_a.reset_index(drop=True)
    b = df_b.set_index(on).loc[a[on]].reset_index()
    if not np.array_equal(a["label"].values, b["label"].values):
        raise ValueError("labels differ between the two prediction sets")
    diffs = np.array([metric_fn(a.iloc[i]) - metric_fn(b.iloc[i])
                      for i in _resample_indices(a, B, seed, cluster_col, strata_col)], dtype=float)
    lo_q, hi_q = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0
    return {
        "diff": float(metric_fn(a) - metric_fn(b)),
        "ci": [float(np.nanpercentile(diffs, lo_q)), float(np.nanpercentile(diffs, hi_q))],
        "frac_le_0": float(np.mean(diffs[~np.isnan(diffs)] <= 0)),
        "B": B, "seed": seed,
    }


def task_macro(metric_fn: Callable, task_col: str = "task") -> Callable:
    """metric_fn -> a metric that is the unweighted mean of metric_fn over the task_col groups present
    (paper_plan A10). Dict-valued metrics are averaged per key; NaN if any task's value is NaN."""
    def fn(df: pd.DataFrame):
        vals = [metric_fn(g) for _, g in df.groupby(task_col, sort=True)]
        if isinstance(vals[0], dict):
            return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        return float(np.mean(vals))
    return fn


def mcnemar(y, pred_a, pred_b) -> Dict[str, float]:
    """McNemar's test on thresholded predictions. b = only A correct, c = only B correct.
    Exact two-sided binomial p-value, plus the continuity-corrected chi-square p-value."""
    y, pa, pb = (np.asarray(v).astype(int) for v in (y, pred_a, pred_b))
    ca, cb = pa == y, pb == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    if b + c == 0:
        return {"b": b, "c": c, "p_exact": 1.0, "p_chi2_cc": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "p_exact": float(binomtest(b, b + c, 0.5).pvalue),
            "p_chi2_cc": float(chi2.sf(stat, 1))}


# ─── Legacy adapters (keep the old key names and the old single-class ROC-AUC = 0.0) ─────

LEGACY_KEYS = ("accuracy", "f1_macro", "f1_injected", "precision", "recall", "roc_auc")


def legacy_metrics(labels, preds, probs, prefix: str) -> Dict[str, float]:
    m = point_metrics(labels, probs, pred=preds)
    out = {f"{prefix}_{k}": m[k] for k in LEGACY_KEYS}
    if np.isnan(out[f"{prefix}_roc_auc"]):
        out[f"{prefix}_roc_auc"] = 0.0
    return out
