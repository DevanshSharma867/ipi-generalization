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


# ─── Bootstrap ────────────────────────────────────────────────────────────────

def _resample_indices(df: pd.DataFrame, B: int, seed: int, cluster_col: str):
    rng = np.random.RandomState(seed)
    clusters = df[cluster_col].unique()
    values = df[cluster_col].values
    members = {c: np.where(values == c)[0] for c in clusters}
    for _ in range(B):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
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
                          ci: float = 95.0, cluster_col: str = "doc_id"):
    """
    Cluster bootstrap over documents: resample doc_ids with replacement and keep every row of each
    sampled doc. `metric_fn(frame)` may return a float or a dict of floats; the result mirrors it
    with {"point", "ci"} leaves. Percentile CIs, NaN resamples ignored.
    """
    df = df.reset_index(drop=True)
    samples = [metric_fn(df.iloc[idx]) for idx in _resample_indices(df, B, seed, cluster_col)]
    return _summarize(samples, metric_fn(df), ci)


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame, metric_fn: Callable, B: int = 1000,
                     seed: int = 42, ci: float = 95.0, on: str = "id", cluster_col: str = "doc_id"):
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
                      for i in _resample_indices(a, B, seed, cluster_col)], dtype=float)
    lo_q, hi_q = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0
    return {
        "diff": float(metric_fn(a) - metric_fn(b)),
        "ci": [float(np.nanpercentile(diffs, lo_q)), float(np.nanpercentile(diffs, hi_q))],
        "frac_le_0": float(np.mean(diffs[~np.isnan(diffs)] <= 0)),
        "B": B, "seed": seed,
    }


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
