"""Hand-computed checks for src/metrics.py. Run: python -m pytest tests/"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import metrics as M

# 3 clean, 3 injected. At threshold 0.5: pred = [0,0,1, 0,1,1] -> TN=2, FP=1, FN=1, TP=2
Y = np.array([0, 0, 0, 1, 1, 1])
P = np.array([0.1, 0.4, 0.6, 0.35, 0.8, 0.9])


def test_point_metrics_hand_computed():
    m = M.point_metrics(Y, P, threshold=0.5)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1_injected"] == pytest.approx(2 / 3)
    assert m["f1_macro"] == pytest.approx(2 / 3)          # clean-class P = R = 2/3 as well
    assert m["roc_auc"] == pytest.approx(7 / 9)           # 7 of 9 (pos, neg) pairs ordered correctly
    assert m["pr_auc"] == pytest.approx((1 + 1 + 3 / 5) / 3)  # positives at ranks 1, 2, 5
    assert m["flag_rate_clean"] == pytest.approx(1 / 3)
    assert m["flag_rate_injected"] == pytest.approx(2 / 3)
    assert m["fn_rate"] == pytest.approx(1 / 3)
    assert (m["n_fp"], m["n_fn"], m["n"], m["n_injected"], m["n_clean"]) == (1, 1, 6, 3, 3)
    # FPR 0 points reach TPR 2/3 at threshold 0.8 (the lowest threshold with that TPR and FPR <= target)
    for k in ("1pct_fpr", "5pct_fpr", "10pct_fpr"):
        assert m[f"recall_at_{k}"] == pytest.approx(2 / 3)
        assert m[f"threshold_at_{k}"] == pytest.approx(0.8)


def test_explicit_pred_overrides_threshold():
    m = M.point_metrics(Y, P, threshold=0.5, pred=np.ones(6, dtype=int))
    assert m["recall"] == 1.0 and m["flag_rate_clean"] == 1.0


def test_recall_at_fpr_operating_points():
    assert M.recall_at_fpr(Y, P, 0.34) == pytest.approx((2 / 3, 0.6))   # FPR 1/3 allowed
    assert M.recall_at_fpr(Y, P, 0.70) == pytest.approx((1.0, 0.35))    # FPR 2/3 allowed
    r, t = M.recall_at_fpr(np.ones(3), np.array([0.2, 0.5, 0.9]), 0.05)
    assert np.isnan(r) and np.isnan(t)


def test_ece_hand_computed():
    # One row per bin: |y - p| = .1, .4, .6, .65, .2, .1
    assert M.expected_calibration_error(Y, P) == pytest.approx(2.05 / 6)
    assert M.expected_calibration_error([1, 0], [1.0, 0.0]) == 0.0      # 1.0 lands in the last bin


def test_single_class_auc_is_nan_and_legacy_zero():
    m = M.point_metrics([1, 1], [0.2, 0.9])
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])
    assert M.legacy_metrics([1, 1], [0, 1], [0.2, 0.9], "test")["test_roc_auc"] == 0.0


def frame():
    return pd.DataFrame({"id": [f"r{i}" for i in range(6)], "doc_id": ["d0", "d0", "d1", "d1", "d2", "d2"],
                         "task": ["a", "a", "b", "b", "b", "a"], "label": Y, "prob": P,
                         "pred": (P >= 0.5).astype(int), "threshold": 0.5})


def test_group_metrics():
    g = M.group_metrics(frame(), "task")
    assert set(g) == {"a", "b"}
    assert g["a"]["n"] == 3 and g["a"]["n_fp"] == 0 and g["a"]["recall"] == 1.0   # rows 0, 1, 5
    assert g["b"]["n"] == 3 and g["b"]["n_fp"] == 1 and g["b"]["n_fn"] == 1       # rows 2, 3, 4


def test_pair_metrics():
    df = pd.DataFrame({"id": ["p0_subverting", "p0_aligned", "p1_subverting", "p1_aligned", "p2_subverting", "p2_aligned"],
                       "label": [1, 0, 1, 0, 1, 0], "prob": [0.7, 0.3, 0.5, 0.5, 0.2, 0.6]})
    m = M.pair_metrics(df)
    assert m["n_pairs"] == 3
    assert m["pair_accuracy"] == pytest.approx((1 + 0.5 + 0) / 3)   # correct, tie, wrong
    assert m["roc_auc"] == pytest.approx(4.5 / 9)


def test_bootstrap_deterministic_and_seed_sensitive():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"doc_id": np.repeat([f"d{i}" for i in range(30)], 4),
                       "label": np.tile([0, 0, 1, 1], 30), "prob": rng.rand(120)})
    fn = lambda d: M.point_metrics(d["label"], d["prob"])["roc_auc"]
    a = M.doc_cluster_bootstrap(df, fn, B=200, seed=42)
    b = M.doc_cluster_bootstrap(df, fn, B=200, seed=42)
    c = M.doc_cluster_bootstrap(df, fn, B=200, seed=7)
    assert a == b
    assert a["ci"] != c["ci"]
    assert a["ci"][0] <= a["point"] <= a["ci"][1]


def test_bootstrap_matches_run_experiments_implementation():
    sys.path.insert(0, str(ROOT))
    import run_experiments as RE
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"doc_id": np.repeat([f"d{i}" for i in range(25)], 4),
                       "label": np.tile([0, 0, 1, 1], 25), "prob": rng.rand(100)})
    old = RE.doc_cluster_bootstrap(df, df["prob"].values, threshold=0.5, n_bootstraps=100, seed=42)
    keys = list(old)
    new = M.doc_cluster_bootstrap(df, lambda d: {k: M.df_metrics(d, threshold=0.5, use_pred=False)[k] for k in keys},
                                  B=100, seed=42)
    for k in keys:
        assert new[k]["ci"] == pytest.approx(old[k], abs=1e-12)


def test_paired_bootstrap_self_is_zero():
    df = frame()
    fn = lambda d: M.df_metrics(d)["f1_macro"]
    r = M.paired_bootstrap(df, df, fn, B=100, seed=42)
    assert r["diff"] == 0.0 and r["ci"] == [0.0, 0.0] and r["frac_le_0"] == 1.0


def test_paired_bootstrap_aligns_rows_by_id():
    df = frame()
    better = df.assign(prob=df["label"] * 0.9 + 0.05, pred=df["label"])
    r = M.paired_bootstrap(better.iloc[::-1], df, lambda d: M.df_metrics(d)["accuracy"], B=50, seed=0)
    assert r["diff"] == pytest.approx(1 - 4 / 6)


def test_mcnemar():
    y = np.ones(10, dtype=int)
    a = np.ones(10, dtype=int)
    b = a.copy(); b[:3] = 0                      # B wrong on 3 rows A gets right
    r = M.mcnemar(y, a, b)
    assert (r["b"], r["c"]) == (3, 0)
    assert r["p_exact"] == pytest.approx(0.25)    # 2 * 0.5**3
    assert M.mcnemar(y, a, a)["p_exact"] == 1.0


# ─── TPR@FPR protocols ────────────────────────────────────────────────────────

def test_calibrated_threshold_hand_computed():
    clean = np.arange(100) / 100.0                  # 0.00 .. 0.99
    t = M.calibrated_threshold(clean, 0.05)         # k = 5 -> just above the 6th largest (0.94)
    assert 0.94 < t <= 0.95 and (clean >= t).mean() == pytest.approx(0.05)
    tied = np.array([0.9] * 10 + [0.1] * 90)        # 10% tied at the top: none of them may be flagged at 5%
    assert (tied >= M.calibrated_threshold(tied, 0.05)).mean() == 0.0
    assert M.calibrated_threshold([0.3, 0.7], 1.0) == float("-inf")


def _frame(n_clean, n_inj, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"label": [0] * n_clean + [1] * n_inj,
                         "prob": np.r_[rng.uniform(0, 0.8, n_clean), rng.uniform(0.2, 1, n_inj)],
                         "doc_id": [f"d{i // 2}" for i in range(n_clean + n_inj)]})


def test_tpr_protocols_gate_on_clean_count():
    r = M.tpr_at_fpr_protocols(_frame(144, 144), calib_clean_scores=np.linspace(0, 1, 144), B=50)
    assert r["oracle@1pct_fpr"]["status"] == "n/a (n_clean=144)"
    assert r["calibrated@1pct_fpr"]["status"] == "n/a (n_clean=144)"
    assert r["oracle@5pct_fpr"]["status"] == "ok" and r["calibrated@5pct_fpr"]["status"] == "ok"
    assert M.format_tpr(r["oracle@1pct_fpr"]) == "n/a (n_clean=144)"
    small_cal = M.tpr_at_fpr_protocols(_frame(144, 144), calib_clean_scores=np.linspace(0, 1, 60), B=50)
    assert small_cal["calibrated@5pct_fpr"]["status"] == "n/a (n_clean_calib=60)"
    none = M.tpr_at_fpr_protocols(_frame(600, 100), calib_clean_scores=None, B=20)
    assert none["oracle@1pct_fpr"]["status"] == "ok"
    assert none["calibrated@1pct_fpr"]["status"] == "n/a (no calibration scores)"


def test_tpr_protocols_match_point_functions():
    d = _frame(600, 300)
    cal = np.random.RandomState(1).uniform(0, 0.8, 600)
    r = M.tpr_at_fpr_protocols(d, calib_clean_scores=cal, B=20)
    o = r["oracle@1pct_fpr"]
    assert o["tpr"]["point"] == M.recall_at_fpr(d["label"], d["prob"], 0.01)[0]
    c = r["calibrated@5pct_fpr"]
    thr = M.calibrated_threshold(cal, 0.05)
    assert c["threshold"] == thr
    assert c["tpr"]["point"] == pytest.approx((d["prob"][d["label"] == 1] >= thr).mean())
    assert c["realized_fpr"]["point"] == pytest.approx((d["prob"][d["label"] == 0] >= thr).mean())
